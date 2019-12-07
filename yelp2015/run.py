import numpy as np
import pickle
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from HFAN import HFAN
from Mymodel import Mymodel


def loadData(tbName='yelpnyc'):
    path = "data/"
    user_dict = pickle.load(open(path+tbName+"/user_dict.pkl", 'rb'))
    product_dict = pickle.load(open(path+tbName+"/product_dict.pkl", 'rb'))
    X_train_uid, X_train_pid, X_train, _, y_train, embedding_weights = pickle.load(open(path+tbName+"/train_mat.pkl", mode='rb'))
    X_test_uid, X_test_pid, X_test, _, y_test = pickle.load(open(path+tbName+"/test_mat.pkl", mode='rb'))
    config['max_sents'] = len(X_train[0])
    config['max_words'] = len(X_train[0][0])
    config['num_users'] = len(user_dict)
    config['num_products'] = len(product_dict)
    config['relTotal'] = len(X_train)
    print(config)
    config['embedding_weights'] = embedding_weights

    print("training set:")
    y = np.array(y_train)
    print("positive:", len(y[y==1]))
    print("negtive:", len(y[y==0]))
    return X_train_uid, X_train_pid, X_train, y_train, \
           X_test_uid,  X_test_pid, X_test,   y_test


def train_and_test(experiment_type, tbName):
    X_train_uid, X_train_pid, X_train, y_train, \
    X_test_uid,  X_test_pid, X_test,   y_test  = loadData(tbName)


    model = HFAN(config)
    if experiment_type == 'o1':
        model.fit(X_train, y_train, X_test, y_test)
        model.load_state_dict(state_dict=torch.load(config['save_path']))
        y_pred = model.predict(X_test)

    elif experiment_type == 'o2':
        model.fit(X_train, y_train, X_test, y_test,
                  X_train_uid, X_test_uid, X_train_pid, X_test_pid)

        model.load_state_dict(state_dict=torch.load(config['save_path']))
        y_pred = model.predict(X_test, X_test_uid, X_test_pid)

    AP = average_precision_score(y_test, y_pred)
    AUC = roc_auc_score(y_test, y_pred)
    print("Test set AP:", AP)
    print("Test set AUC:", AUC)


config = {
    'alpha':0.1,
    'radius':3,
    'maxlen':200,
    'dropout': 0.5,
    'batch_size': 100,
    'epochs':10,
}


if __name__ == '__main__':
    import time
    start = time.time()
    task = 'yelpzip' #
    experiment_type = 'o2'

    if task == 'yelpchi': config['batch_size'] = 50
    config['save_path'] = 'checkpoint/weights.best.' + task +"."+ experiment_type + "."+ HFAN.__name__.lower().strip("text")
    train_and_test(experiment_type, task)
    end = time.time()
    print("use time: ", (end-start)/60, " min")

