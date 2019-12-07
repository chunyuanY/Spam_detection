import itertools
import pickle
from collections import Counter
import gensim
import jieba
import numpy as np
import pandas as pd
import pymysql
import utils
import re
from sklearn.preprocessing import StandardScaler

jieba.set_dictionary('dict.txt.big')
w2v_dim = 300


def readData(sql):
    config = {
        'host':'localhost',
        'port':3306,
        'user':'root',
        'password':'xxxxxxxxxx',
        'db':'samsung',
        'charset':'utf8',
        'cursorclass':pymysql.cursors.DictCursor,
    }
    # Connect to the database
    conn = pymysql.connect(**config)
    df = pd.read_sql(sql=sql, con=conn)
    conn.close()
    return df


def transform(content):
    content = content.lower()
    content = utils.full2half(content)
    content = re.sub("=+|-+|~+|\*+|\++", "\n", content)
    return re.sub("\s+", " ", content)


def cut(doclist, is_hierarchical=False):
    print("cut words......")
    result = []
    for content in doclist:
        if content is not None:
            content = transform(content)
            if is_hierarchical:
                sent_list = []
                sent = []
                for word in jieba.cut(content):
                    if len(sent) == 20:
                        sent_list.append(sent)
                        sent = []
                    sent.append(word)
                sent_list.append(sent)
                result.append(sent_list)
            else:
                result.append(list(jieba.cut(content)))
    return result


def downsampling(X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, ratio):
    zeroidx = [i for i, tag in enumerate(y_train) if tag==0 ]
    np.random.seed(0)
    zeroidx = np.random.choice(zeroidx, size=int(len(zeroidx)*ratio), replace=False)

    oneidx = set([i for i, tag in enumerate(y_train) if tag==1])

    X_train_uid_tmp = []
    X_train_pid_tmp = []
    X_train_content_tmp = []
    X_train_bf_tmp = []
    y_train = []

    for idx in oneidx:
        # if len(X_train_uid) != 0:
        X_train_uid_tmp.append(X_train_uid[idx])
        X_train_pid_tmp.append(X_train_pid[idx])
        X_train_content_tmp.append(X_train_content[idx])
        X_train_bf_tmp.append(X_train_bf[idx])
        y_train.append(1)

    for idx in zeroidx:
        # if len(X_train_uid) != 0:
        X_train_uid_tmp.append(X_train_uid[idx])
        X_train_pid_tmp.append(X_train_pid[idx])
        X_train_content_tmp.append(X_train_content[idx])
        X_train_bf_tmp.append(X_train_bf[idx])
        y_train.append(0)

    X_train_uid = X_train_uid_tmp
    X_train_pid = X_train_pid_tmp
    X_train_content = X_train_content_tmp
    X_train_bf = X_train_bf_tmp
    return X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train


def combine(title, content):
    for i in range(len(title)):
        if title[i] is None:
            title[i] = ""
        if type(content[i]) is str:
            content[i] = title[i] + "\n" + content[i]
        elif type(title[i]) is str:
            content[i].insert(0, title[i])
        else:
            content[i] = title[i] + content[i]
    return content



def load_data_and_labels(root_path, tbName, is_hierarchical):
    import os
    filename = root_path+tbName+".pkl"
    if not os.path.isfile(filename):
        print("load data from db......")

        if tbName == 'reply':
            ratio = 0.1
            sql = 'SELECT uid, reply_train.thid, title, content, nfloor, pnum, tot_pages, clicks, reply_train.time, is_spam ' \
                  'FROM reply_train LEFT JOIN thread_info on  reply_train.thid = thread_info.thid'
        else:
            ratio = 0.4
            sql = 'SELECT uid, thid, title, content, nfloor, pnum, tot_pages, clicks, time, is_spam  FROM full_train'

        df = readData(sql)
        X_train_uid = df['uid'].tolist()
        X_train_pid = df['thid'].tolist()
        y_train = df['is_spam'].tolist()

        X_train_content = combine(df['title'].tolist(), df['content'].tolist())

        X_train_bf1 = utils.text_statistic_feature(X_train_content)
        X_train_bf2 = df[['nfloor', 'pnum', 'tot_pages', 'clicks']].values.tolist()
        X_train_time = utils.time_feature(df['time'].tolist())
        X_train_bf = np.hstack([X_train_bf1, X_train_bf2, X_train_time])
        X_train_bf[np.isnan(X_train_bf)] = 0

        X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train = downsampling(X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, ratio)
        X_train_content = cut(X_train_content, is_hierarchical)

        # =====================================================================================================
        if tbName == 'reply':
            sql = 'SELECT uid, reply_test.thid, title, content, nfloor, pnum, tot_pages, clicks, reply_test.time, is_spam FROM reply_test ' \
                  'left join thread_info on  reply_test.thid = thread_info.thid'
        else:
            sql = 'SELECT uid, thid, title, content, nfloor, pnum, tot_pages, clicks, time, is_spam FROM full_test'

        df = readData(sql)
        X_test_uid = df['uid'].tolist()
        X_test_pid = df['thid'].tolist()
        y_test = df['is_spam'].tolist()
        X_test_content = combine(df['title'].tolist(), df['content'].tolist())

        X_test_bf1 = utils.text_statistic_feature(X_test_content)
        X_test_time = utils.time_feature(df['time'].tolist())
        X_test_bf2 = df[['nfloor', 'pnum', 'tot_pages', 'clicks']].values.tolist()
        X_test_bf = np.hstack([X_test_bf1, X_test_bf2, X_test_time])
        X_test_bf[np.isnan(X_test_bf)] = 0

        X_test_content = cut(X_test_content, is_hierarchical)

        scaler = StandardScaler()
        X_train_bf = scaler.fit_transform(X_train_bf)
        X_test_bf = scaler.transform(X_test_bf)

        pickle.dump([X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train,
                     X_test_uid,  X_test_pid, X_test_content,  X_test_bf, y_test],  open(filename, 'wb') )
    else:
        X_train_uid, X_train_pid, X_train_content,  X_train_bf, y_train, \
        X_test_uid,  X_test_pid,  X_test_content,  X_test_bf, y_test = pickle.load(open(filename, mode='rb'))

    print("Training set users: ", len(set(X_train_uid)))
    print("Test set users: ", len(set(X_test_uid)))
    def avg_words(X_content):
        cnt = 0
        for content in X_content:
            cnt += len(content)
        return cnt/len(X_content)
    print("Training set avg words: ", avg_words(X_train_content))
    print("Training set avg words: ", avg_words(X_test_content))


    return X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, \
           X_test_uid,  X_test_pid, X_test_content, X_test_bf, y_test



def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            #add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)
            #print(word)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_vocab_word2vec(X_train, is_hierarchical):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    if is_hierarchical:
        sentences = []
        for doc in X_train:
            sentences += doc
    else:
        sentences = X_train

    # Build vocabulary
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 2]
    # Mapping from word to index
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}

    print("embedding_weights generation.......")
    word2vec = vocab_to_word2vec('./samsung.bin', vocabulary)
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    return vocabulary, embedding_weights


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, w2v_dim), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(w2v_dim,) )
    for idx in range(vocab_size):
        embedding_weights[idx+1] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def statistic(X_train):
    max_words = 0
    max_sents = 0
    words_count = 0
    sents_count = 0
    doc_count = 0
    for doc in X_train:
        doc_len = 0
        if len(doc) > max_sents:
            max_sents = len(doc)
        if len(doc) > 50:
            sents_count+=1

        for sent in doc:
            doc_len += len(sent)
            if len(sent) > max_words:
                max_words = len(sent)
            if len(sent) > 20:
                words_count += 1
        if doc_len >= 400:
            doc_count += 1

    print("max_sents: ", max_sents)
    print("max_words: ", max_words)
    print("doc exceeds 1000: ", doc_count)
    print("sents exceeds 50: ", sents_count)
    print("words exceeds 20: ", words_count)


def pad_sequence(X, max_words=20, max_sents=50, max_len=0):
    zeros_sent = [0]*max_words
    X_pad = []

    if max_len != 0:
        for doc in X:
            if len(doc) >= max_len:
                doc = doc[:max_len]
            else:
                doc = [0] *(max_len - len(doc)) + doc
            X_pad.append(doc)
    else:
        for doc in X:
            X_doc_pad = []
            for sent in doc:
                if len(sent) >= max_words:
                    X_doc_pad.append(sent[:max_words])
                else:
                    sent = sent + [0] *(max_words - len(sent))
                    X_doc_pad.append(sent)

            while len(X_doc_pad) < max_sents:
                X_doc_pad.append(zeros_sent)
            X_pad.append(X_doc_pad[-max_sents:])
    return X_pad


def build_input_data(X, vocabulary, is_hierarchical, max_sents=0, max_len=0):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    if is_hierarchical:
        x = [[[vocabulary[word] for word in sent if word in vocabulary] for sent in doc] for doc in X]
        statistic(x)
        print("============================")
        print("padding sequences...")
        x = pad_sequence(x, max_sents=max_sents)
    else:
        x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
        x = pad_sequence(x, max_sents=max_sents, max_len=max_len)
    return x


def feature_extract(tbName, max_len=0, is_hierarchical=False):
    if is_hierarchical:
        root_path = "../hdata/"
        if tbName == 'reply':
            max_sents = 5
        else:
            max_sents = 25
    else:
        root_path = "../data/"
        max_sents = 0

    X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, \
    X_test_uid,  X_test_pid,  X_test_content,  X_test_bf,  y_test = load_data_and_labels(root_path, tbName, is_hierarchical)
    print(str(len(X_train_content)) + " sentences read")


    print("text word2vec generation.......")
    vocabulary, word_embeddings = build_vocab_word2vec(X_train_content + X_test_content, is_hierarchical)
    pickle.dump(vocabulary, open(root_path + tbName +"/vocab.pkl", 'wb'))
    print("Vocabulary size: "+str(len(vocabulary)))

    print("uid mapping .......")
    user_dict = {uid:i for i, uid in enumerate(set(X_train_uid + X_test_uid))}
    pickle.dump(user_dict, open(root_path+tbName+"/user_dict.pkl", 'wb'))
    X_train_uid = [user_dict[uid] for uid in X_train_uid]
    X_test_uid = [user_dict[uid] for uid in X_test_uid]

    print("pid mapping .......")
    product_dict = {uid:i for i, uid in enumerate(set(X_train_pid + X_test_pid))}
    pickle.dump(product_dict, open(root_path+tbName+"/product_dict.pkl", 'wb'))
    X_train_pid = [product_dict[uid] for uid in X_train_pid]
    X_test_pid = [product_dict[uid] for uid in X_test_pid]


    print("build input data.......")
    X_train = build_input_data(X_train_content, vocabulary, is_hierarchical, max_sents, max_len)
    X_test = build_input_data(X_test_content, vocabulary, is_hierarchical, max_sents, max_len)

    pickle.dump([X_train_uid, X_train_pid, X_train, X_train_bf, y_train, word_embeddings], open(root_path+tbName+"/train_mat.pkl", 'wb') )
    pickle.dump([X_test_uid, X_test_pid, X_test, X_test_bf, y_test], open(root_path+tbName+"/test_mat.pkl", 'wb') )


if __name__ == "__main__":
    feature_extract('first_post', is_hierarchical=True)
    feature_extract('reply', is_hierarchical=True)
    print("Data created")
