import abc
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
import torch.nn.utils as utils

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

class NeuralNetwork(nn.Module):

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.patience = 0
        self.best_auc = 0
        self.best_ap = 0
        self.init_clip_max_norm = None
        self.optimizer = None


    @abc.abstractmethod
    def forward(self):
        pass

    def fit(self, X_train, y_train, X_val, y_val,
                  X_train_uid=None, X_val_uid=None,
                  X_train_pid=None, X_val_pid=None):

        if torch.cuda.is_available():
            self.cuda()

        X_train = torch.LongTensor(X_train)
        if X_train_uid is not None:
            X_train_uid = torch.LongTensor(X_train_uid)
            X_train_pid = torch.LongTensor(X_train_pid)
            X_train_rid = torch.LongTensor(range(len(X_train)))
        y_train = torch.LongTensor(y_train)

        # loss_func = FocalLoss()
        loss_func = nn.CrossEntropyLoss()
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay= self.l2_reg)
        # pickle.dump(self.user_embedding.weight.data.cpu().numpy(),
        #             file=open("data/user_embedding0.pkl", 'wb'), protocol=4)

        for epoch in range(self.config['epochs']):
            print("\nEpoch ", epoch+1,"/", self.config['epochs'])
            avg_loss = 0
            avg_acc = 0
            X_train, X_train_uid, X_train_pid, X_train_rid, y_train = self.shuffle_data(X_train, X_train_uid, X_train_pid, X_train_rid, y_train)

            self.train()
            for i, bat_start in enumerate(range(0, y_train.size(0), self.bsz)):
                with torch.no_grad():
                    batch_x_text = X_train[bat_start: bat_start+self.bsz].cuda()
                    batch_x_uid = X_train_uid[bat_start: bat_start+self.bsz].cuda()
                    batch_x_pid = X_train_pid[bat_start: bat_start+self.bsz].cuda()
                    batch_x_rid = X_train_rid[bat_start: bat_start+self.bsz].cuda()
                    batch_y = y_train[bat_start: bat_start+self.bsz].cuda()

                self.optimizer.zero_grad()
                logit, rloss = self.forward(batch_x_text, batch_x_uid, batch_x_pid, batch_x_rid)
                loss = loss_func(logit, batch_y)
                if rloss is not None:
                    loss += self.alpha*rloss
                loss.backward()
                self.optimizer.step()

                corrects = (torch.max(logit, 1)[1].view(batch_y.size()).data == batch_y.data).sum()
                accuracy = 100*corrects/len(batch_y)

                print('Batch[{}] - loss: {:.6f}  acc: {:.4f}%({}/{})'.format(i, loss.item(), accuracy, corrects, batch_y.size(0)))
                if i > 0 and i % 100 == 0:
                    self.evaluate(X_val, X_val_uid, X_val_pid, y_val, epoch+1)
                    self.train()

                if epoch >= 5 and self.patience >= 4:
                    print("Reload the best model...")
                    self.load_state_dict(torch.load(self.config['save_path']))
                    self.adjust_learning_rate(self.optimizer)
                    self.patience = 0

                if self.init_clip_max_norm is not None:
                    utils.clip_grad_norm_(self.parameters(), max_norm=self.init_clip_max_norm)

                avg_loss += loss.item()
                avg_acc += accuracy
            cnt = y_train.size(0) // self.bsz + 1
            print("Average loss:{:.6f} average acc:{:.6f}%".format(avg_loss/cnt, avg_acc/cnt))

            self.evaluate(X_val, X_val_uid, X_val_pid, y_val, epoch+1)


    def shuffle_data(self, *X_need_shuffle):
        shuffle_indexes = torch.randperm(X_need_shuffle[0].size(0))
        for X in X_need_shuffle:
            if X is not None:
                X[:] = X[shuffle_indexes]
        return X_need_shuffle


    def adjust_learning_rate(self, optimizer, decay_rate=.5):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * decay_rate
            self.lr = param_group['lr']
        print("Decay learning rate to: ", self.lr)


    def evaluate(self, X_val, X_val_uid, X_val_pid, y_val, epoch):
        y_pred = self.predict(X_val, X_val_uid, X_val_pid)
        AP = average_precision_score(y_val, y_pred)
        AUC = roc_auc_score(y_val, y_pred)

        if AUC > self.best_auc:
            self.best_ap = AP
            self.best_auc = AUC
            torch.save(self.state_dict(), self.config['save_path'])
            # pickle.dump(self.user_embedding.weight.data.cpu().numpy(),
            #             file=open("data/user_embedding"+str(epoch)+".pkl", 'wb'), protocol=4)
            print("save model!!!")
        else:
            self.patience += 1
        print("Val set AP:", AP)
        print("Best val set AP:", self.best_ap)
        print("Val set AUC:", AUC)
        print("Best val set AUC:", self.best_auc)

    def predict(self, X_test, X_test_uid=None, X_test_pid=None):
        if torch.cuda.is_available():
            self.cuda()

        self.eval()
        y_pred = []
        batch_size = 100
        X_test = torch.LongTensor(X_test)
        if X_test_uid is not None:
            X_test_uid = torch.LongTensor(X_test_uid)
            X_test_pid = torch.LongTensor(X_test_pid)

        for i, bat_start in enumerate(range(0, X_test.size(0), batch_size)):
            batch_x_text = X_test[bat_start: bat_start+batch_size].cuda()
            if X_test_uid is not None:
                batch_x_uid = X_test_uid[bat_start: bat_start+batch_size].cuda()
                batch_x_pid = X_test_pid[bat_start: bat_start+batch_size].cuda()

            logits, _ = self.forward(batch_x_text, batch_x_uid, batch_x_pid)
            predicted = torch.sigmoid(logits)[:, 1]
            y_pred += predicted.data.cpu().numpy().tolist()
        return y_pred

