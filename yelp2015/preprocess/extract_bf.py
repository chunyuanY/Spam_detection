import numpy as np
# from preprocess import readData
import pymysql
import pandas as pd
import nltk
from nltk import word_tokenize,sent_tokenize
from nltk.corpus import sentiwordnet as swn
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('sentiwordnet')

def readData(sql, dbName='yelp_spam_kdd2015'):
    config = {
        'host':'192.168.140.48',
        'port':3306,
        'user':'root',
        'password':'woaiiie',
        'db':dbName,
        'charset':'utf8',
        'cursorclass':pymysql.cursors.DictCursor,
    }
    # Connect to the database
    conn = pymysql.connect(**config)
    df = pd.read_sql(sql=sql, con=conn)
    conn.close()
    return df


def get_MNR(tbName):
    sql = "select user_id, MAX(DISTINCT cnt) as maxcnt from (SELECT user_id, date, count(*) as cnt " \
          "FROM `"+tbName+"` GROUP BY user_id, date)t GROUP BY user_id"
    print(sql)
    df = readData(sql)
    tmp = df[['user_id', 'maxcnt']].values.tolist()
    id2cnt = {idandcnt[0]:idandcnt[1] for idandcnt in tmp}
    return id2cnt


def get_PR_NR(tbName):
    sql = "SELECT user_id, count(*) as cntsum FROM `"+tbName+"` GROUP BY user_id"
    print(sql)
    df = readData(sql)
    all = df[['user_id', 'cntsum']].values.tolist()
    all = {tmp[0]:tmp[1] for tmp in all}

    sql = "SELECT user_id, sum(pcnt) as pcntsum from (SELECT user_id, rating, count(*) as pcnt FROM `"+tbName+"` " \
           "GROUP BY user_id, rating HAVING rating >=4)t GROUP BY user_id"
    print(sql)
    df = readData(sql)
    positive = df[['user_id', 'pcntsum']].values.tolist()
    positive = {tmp[0]:tmp[1] for tmp in positive}

    id2pr = {}
    for uid in all.keys():
        if not (uid in positive):
            id2pr[uid] = 0
        else:
            id2pr[uid] = positive[uid] / all[uid]

    sql = "SELECT user_id, sum(ncnt) as ncntsum from (SELECT user_id, rating, count(*) as ncnt FROM `"+tbName+"` " \
           "GROUP BY user_id, rating HAVING rating <=2)t GROUP BY user_id"
    print(sql)
    df = readData(sql)
    negtive = {tmp[0]:tmp[1] for tmp in df[['user_id', 'ncntsum']].values.tolist()}
    id2nr = {}
    for uid in all.keys():
        if not (uid in negtive):
            id2nr[uid] = 0
        else:
            id2nr[uid] = negtive[uid] / all[uid]

    return id2pr, id2nr


def get_RD(tbName):
    sql = "SELECT prod_id, avg(rating) as avg_r FROM `"+tbName+"` GROUP BY prod_id"
    print(sql)
    df = readData(sql)
    prod_avgr = {tmp[0]:float(tmp[1]) for tmp in df[['prod_id', 'avg_r']].values.tolist()}
    return prod_avgr


def get_behavior_features(X_train_uid, X_test_pid_rating, tbName):
    print("get behavior fatures ... ")
    mnr_id2cnt = get_MNR(tbName)
    pr_id2cnt, nr_id2cnt = get_PR_NR(tbName)
    prod_avgr = get_RD(tbName)

    BF = []
    for i in range(len(X_train_uid)):
        uBF = []

        pid, rating = X_test_pid_rating[i][0], float(X_test_pid_rating[i][1])
        # RD: Absolute rating deviation from product’s average rating
        uBF.append(rating - prod_avgr[pid])

        # DEV: Thresholded rating deviation of review
        uBF.append(1 if abs(rating - prod_avgr[pid])/4 > 0.5 else 0)

        # EXT: Extremity of rating
        uBF.append(1 if int(rating) in [1, 5] else 0)

        uid = X_train_uid[i]
        # maximum number of reviews
        if uid in mnr_id2cnt:
            uBF.append(mnr_id2cnt[uid])
        else:
            uBF.append(0)

        # percentage of positive reviews
        if uid in pr_id2cnt:
            uBF.append(pr_id2cnt[uid])
        else:
            uBF.append(0)

        # percentage of negtive reviews
        if uid in nr_id2cnt:
            uBF.append(nr_id2cnt[uid])
        else:
            uBF.append(0)
        BF.append(uBF)
    return BF


def get_text_features(X_train_content):
    print("get text fatures ... ")
    # review length
    TF = []
    for content in X_train_content:
        tfi = []

        capital_letters = 0
        exclamations = 0  # numbers of !
        for l in content:
            capital_letters += 1 if l.isupper() else 0
            exclamations += 1 if l == '!' else 0
        tfi.append(capital_letters)  # Percentage of capital letters
        tfi.append(exclamations)  # Ratio of exclamation


        content = word_tokenize(content)
        tfi.append(len(content))  # Review length in words

        all_capital_words = 0
        first_person = 0
        obj_words = 0
        for word in content:
            all_capital_words += 1 if word.isupper() else 0
            first_person += 1 if word.lower() in ['i', 'my', 'me'] else 0

            # word = list(swn.senti_synsets(word))
            # if len(word) > 0 and word[0].obj_score() > 0.5:
            #     obj_words += 1
        tfi.append(all_capital_words)  # Percentage of ALL-capitals words
        tfi.append(first_person)  # Ratio of 1st person pronouns
        # tfi.append(obj_words/len(content))   # Ratio of objective words
        # tfi.append(1- obj_words/len(content))  # Ratio of subjective words
        TF.append(tfi)
    return TF



def get_bf_tf(X_train_uid, X_train_pid_rating, X_train_content, tbName):
    TF = get_text_features(X_train_content)  # 6
    BF = get_behavior_features(X_train_uid, X_train_pid_rating, tbName) # 5
    X_bf_tf = np.concatenate([BF, TF], axis=1)
    return X_bf_tf