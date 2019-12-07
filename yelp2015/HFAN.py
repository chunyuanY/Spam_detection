import torch
import torch.nn as nn
import torch.nn.functional as F
from NeuralNetwork import NeuralNetwork
import torch.nn.init as init


class MultiAttentionUnit(nn.Module):
    def __init__(self, in_features, out_features, unit_size=1):
        super(MultiAttentionUnit, self).__init__()
        self.unit_size = unit_size
        self.multi_linearQ = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(unit_size)])
        self.multi_linearK = nn.ModuleList([nn.Linear(in_features, out_features) for _ in range(unit_size)])
        self.multi_linear = nn.ModuleList([nn.Linear(out_features, 1) for _ in range(unit_size)])
        self.init_weights()

    def init_weights(self):
        for name, parameter in self.multi_linearQ.named_parameters():
            if str(name).__contains__('weight'):
                init.xavier_normal_(parameter)
        for name, parameter in self.multi_linearK.named_parameters():
            if str(name).__contains__('weight'):
                init.xavier_normal_(parameter)
        for name, parameter in self.multi_linear.named_parameters():
            if str(name).__contains__('weight'):
                init.xavier_normal_(parameter)

    def local_attention_unit(self, Q, K, i):
        Q = self.multi_linearQ[i](Q.unsqueeze(dim=1))  # (batch_size, 1, out_features)
        K = self.multi_linearK[i](K)  # (batch_size, region_size, out_features)
        act = F.tanh(Q + K) # (batch_size, region_size, out_features)

        affine = self.multi_linear[i](act)
        score = F.softmax(affine, dim=1)
        local_att = torch.sum(score*K, dim=1)
        return local_att

    def forward(self, Q, K):
        multi_context = torch.stack([self.local_attention_unit(Q, K, i) for i in range(self.unit_size)], dim=1)
        multi_attention = torch.max(multi_context, dim=1)[0]
        return multi_attention


class FusionAttentionUnit(nn.Module):
    def __init__(self, in_features, out_features, user_features):
        super(FusionAttentionUnit, self).__init__()
        self.linear_doc = nn.Linear(in_features, out_features)
        self.linear_user = nn.Linear(user_features, out_features)
        self.W = nn.Parameter(torch.FloatTensor(out_features, out_features))
        self.init_weights()

    def init_weights(self):
        init.xavier_normal_(self.linear_doc.weight)
        init.xavier_normal_(self.linear_doc.weight)
        init.xavier_normal_(self.W)

    def forward(self, X_doc, X_user):
        '''
        :param X_doc: (bsz, max_sents, in_features)
        :param X_user: (bsz, max_sents, D)
        :return:
        '''
        X_doc = self.linear_doc(X_doc)
        X_user = self.linear_user(X_user)

        X_doc = X_doc * F.sigmoid(X_user)  # (bsz, max_sents, in_features)
        X_user = X_user * F.sigmoid(X_doc) # (bsz, max_sents, in_features)
        attentive_mat = F.tanh(torch.einsum("bsd,dd,brd->bsr", X_doc, self.W, X_user) )  # (bsz, max_sents, max_sents)

        score_d = F.softmax(attentive_mat.mean(2), dim=1).unsqueeze(-1)
        score_u = F.softmax(attentive_mat.mean(1), dim=1).unsqueeze(-1)

        attention_d = torch.sum(score_d*X_doc, dim=1)
        attention_u = torch.sum(score_u*X_user, dim=1)
        return attention_d, attention_u


class HFAN(NeuralNetwork):

    def __init__(self, config):
        super(HFAN, self).__init__()
        self.config = config
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape
        n_users, n_products, n_relations = config['num_users'], config['num_products'], config['relTotal']
        self.alpha = config['alpha']
        self.radius = config['radius']
        self.max_words = config['max_words']
        self.max_sents = config['max_sents']
        self.bsz = config.get("batch_size", 100)
        self.l2_reg = config.get("reg", 0)
        self.lr = config.get("lr", 1.0)

        ######################### model #################################
        self.word_embedding = nn.Embedding(V, D, _weight=torch.from_numpy(embedding_weights))
        self.user_embedding = nn.Embedding(n_users, 300, _weight=torch.FloatTensor(n_users, 300).uniform_(-0.1, 0.1))
        self.product_embedding = nn.Embedding(n_products, 300, _weight=torch.FloatTensor(n_products, 300).uniform_(-0.1, 0.1))
        self.transform_vector = nn.Embedding(n_relations, 300, _weight=torch.FloatTensor(n_relations, 300).uniform_(-0.1, 0.1))

        self.multi_au = MultiAttentionUnit(in_features=D, out_features=300, unit_size=1)
        self.affine = nn.Linear(600, 300)

        self.fusion_attention_up = FusionAttentionUnit(300, 300, 300)
        self.linear_f1 = nn.Linear(300, 300)
        self.linear_f2 = nn.Linear(300, 300)
        self.dropout = nn.Dropout(config['dropout'])

        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(300, 2)

        self.criterion = nn.MarginRankingLoss(1.0, False)
        self.optimizer = torch.optim.Adadelta(self.parameters(), lr=self.lr, weight_decay= self.l2_reg)
        self.init_weights()
        print(self)


    def init_weights(self):
        init.xavier_normal_(self.affine.weight)
        init.xavier_normal_(self.linear_f1.weight)
        init.xavier_normal_(self.linear_f2.weight)
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)


    def word_level_mau(self, X_word, X_context):
        X_context = X_context.unsqueeze(dim=1).repeat(self.max_sents, self.max_words, 1)
        X_context = [self.multi_au(X_context[:, i - self.radius, :],  X_word[:, i - self.radius:i + self.radius + 1, :])
                     for i in range(self.radius, self.radius+self.max_words)]

        X_context = torch.stack(X_context, dim=1)
        X_word = torch.cat([X_word[:, self.radius:self.radius+self.max_words, :], X_context], dim=-1)
        X_word = F.tanh(self.affine(X_word))
        output = X_word.max(dim=1)[0]
        return output


    def orth_decompse(self, X, direction, epsilon=1e-6):
        '''
        :param X:     # (bsz, max_sents, 300)
        :param direction:     # (bsz, 300)
        :return:
        '''
        X_para = torch.stack([((X[:,i,:]*direction).sum(dim=1) / ((direction*direction).sum(dim=1) + epsilon)  ).unsqueeze(dim=1) * direction
                              for i in range(X.size(1))], dim=1)
        X_orth = X - X_para
        return X_para, X_orth


    def TransH(self, h, r, t, r_norm):
        '''
        :param h:   (bsz, 300)
        :param r:   (bsz, 300)
        :param t:   (bsz, 300)
        :param r_norm:   (bsz, 300)
        :return:
        '''
        h_neg = h[torch.randperm(r.size(0))]
        # t_neg = t[torch.randperm(r.size(0))]

        def _transfer(e, norm):
            norm = F.normalize(norm, p = 2, dim = -1)
            return e - torch.sum(e * norm, -1, True) * norm

        h_orth = _transfer(h, r_norm)
        t_orth = _transfer(t, r_norm)
        l_pos = torch.norm(h_orth + r - t_orth, dim=-1)

        h_neg_orth = _transfer(h_neg, r_norm)
        # t_neg_orth = _transfer(t_neg, r_norm)
        l_neg = torch.norm(h_neg_orth + r - t_orth, dim=-1)

        return self.criterion(l_pos, l_neg, target=torch.Tensor([-1]).cuda()) / r.size(0)


    def forward(self, X_text_idx, X_uid, X_pid, X_rid=None):
        rloss = None
        mask = ((X_text_idx != 0).sum(dim=-1, keepdim=True) != 0).float()   # (bsz, max_sents, 1)

        X_text_pad = F.pad(X_text_idx, pad=(self.radius, self.radius,0,0,0,0))  # (batch_size, max_sents, max_words+2*radius)
        X_word = self.word_embedding(X_text_pad)  # (batch_size, max_sents, max_words, D)
        X_word = X_word.view([-1, self.max_words + 2*self.radius, X_word.size(-1)])

        X_user_embed = self.user_embedding(X_uid)      # (bsz, 300)
        X_sent_u = self.word_level_mau(X_word, X_user_embed)
        X_sent_u = X_sent_u.view([-1, self.max_sents, X_sent_u.size(-1)]) * mask  # (bsz, max_sents, 300)
        U_para, U_orth = self.orth_decompse(X_sent_u, X_user_embed)
        X_user_d = U_para.mean(dim=1)

        X_prod_embed = self.product_embedding(X_pid)   # (bsz, 300)
        X_sent_p = self.word_level_mau(X_word, X_prod_embed)
        X_sent_p = X_sent_p.view([-1, self.max_sents, X_sent_p.size(-1)]) * mask  # (bsz, max_sents, 300)
        P_para, P_orth = self.orth_decompse(X_sent_p, X_prod_embed)
        X_prod_d = P_para.mean(dim=1)

        X_doc_u, X_doc_p = self.fusion_attention_up(U_orth, P_orth)
        fusion = F.sigmoid(self.linear_f1(X_doc_u) + self.linear_f2(X_doc_p))
        X_doc = fusion * X_doc_u  + (1-fusion) * X_doc_p

        if self.training:
            r_norm = self.transform_vector(X_rid)
            rloss = self.TransH(X_user_d, X_doc, X_prod_d, r_norm)

        X_doc = self.dropout(F.relu(self.fc1(X_doc)))
        output = self.fc2(X_doc)
        return output, rloss

