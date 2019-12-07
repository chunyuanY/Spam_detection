import itertools
import pickle
import re
from collections import Counter
import gensim
import numpy as np
import pandas as pd
import pymysql
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import nltk
import extract_bf


def readData(sql, dbName='yelp_spam_kdd2015'): # yelp_spam_kdd2015
    config = {
        'host':'192.168.0.1',
        'port':3306,
        'user':'root',
        'password':'xxxxxxx',
        'db':dbName,
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
    content = re.sub("n't", " not ", content)
    content = re.sub("'d", " had ", content)
    content = re.sub("'ll", " will ", content)
    content = re.sub("'re", " are ", content)
    content = re.sub("i'm", " i am ", content)
    content = re.sub("'ve", " have ", content)
    content = re.sub("'s", " is ", content)
    for sg in ['!', '\?', ":", "？", "。", "'", '"', '/', '\-', '~', '\.', ',', '=', ';', '；', '#', '\$', '&', '\<','\>','\(', '\)', '\+', '\*' ]:
        content = re.sub(sg, " "+sg.strip("\\")+" ", content)
    content = re.sub("\s+", " ", content)
    return content


def cut(doclist, is_hierarchical=True, max_words=20):
    print("cut words......")
    result = []
    for content in doclist:
        content = transform(content)
        content = content.split()
        if is_hierarchical:
            words = [content[i:i+max_words] for i in range(0, len(content), max_words)]
        else:
            words = content
        result.append(words)
    return result


def shuffle_data(X_, y_):
    shuffle_idx = np.arange(len(X_))
    np.random.seed(42)
    np.random.shuffle(shuffle_idx)

    X_ = X_[shuffle_idx]
    y_ = y_[shuffle_idx]
    return X_, y_


def my_train_test_split(X_uid, X_pid, X_content, X_bf, y_, ratio=0.2):
    assert len(X_uid) == len(X_pid) == len(X_content) == len(X_bf) == len(y_),  "The 1st dimension must be identity"
    y_ = np.array(y_)
    zeroidx = (y_ == 0)
    oneidx = (y_ == 1)

    X_content_idx = np.arange(len(X_content))

    X = np.hstack([X_uid[:, None], X_pid[:, None], X_content_idx[:, None], X_bf])
    X_zero = X[zeroidx]
    y_zero = y_[zeroidx]
    X_one = X[oneidx]
    y_one = y_[oneidx]

    X_train_zero, X_test_zero, y_train_zero, y_test_zero = train_test_split(X_zero, y_zero, test_size=ratio, random_state=42)
    X_train_one, X_test_one, y_train_one, y_test_one = train_test_split(X_one, y_one, test_size=ratio, random_state=42)

    X_train = np.vstack([X_train_zero, X_train_one])
    y_train = np.array(y_train_zero.tolist() + y_train_one.tolist())
    X_train, y_train = shuffle_data(X_train, y_train)
    X_train_uid, X_train_pid, X_train_content_idx, X_train_bf = X_train[:, 0], X_train[:, 1], X_train[:, 2], X_train[:, 3:]
    X_train_content = [X_content[idx] for idx in X_train_content_idx]


    X_test = np.vstack([X_test_zero, X_test_one])
    y_test = np.array(y_test_zero.tolist() + y_test_one.tolist())
    X_test, y_test = shuffle_data(X_test, y_test)
    X_test_uid, X_test_pid, X_test_content_idx, X_test_bf = X_test[:, 0], X_test[:, 1], X_test[:, 2], X_test[:, 3:]
    X_test_content = [X_content[idx] for idx in X_test_content_idx]

    return X_train_uid.tolist(), X_train_pid.tolist(), X_train_content, X_train_bf.tolist(), y_train, \
           X_test_uid.tolist(),  X_test_pid.tolist(),  X_test_content, X_test_bf.tolist(), y_test


def load_data_and_labels(root_path, tbName, is_hierarchical):
    import os
    filename = root_path+tbName+".pkl"
    if not os.path.isfile(filename):
        print("load data from db......")
        sql = "SELECT * FROM " + tbName
        print(sql)
        df = readData(sql)
        X_uid = df['user_id'].values # .tolist()
        X_pid = df['prod_id'].values # .tolist()
        X_content = cut(df['content'].tolist(), is_hierarchical)
        y_ = df['is_spam'].tolist()

        X_pid_rating = df[['prod_id', 'rating']].values.tolist()
        X_bf = extract_bf.get_bf_tf(X_uid, X_pid_rating, df['content'].tolist(), tbName)

        X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, \
        X_test_uid,  X_test_pid,  X_test_content, X_test_bf, y_test = my_train_test_split(X_uid, X_pid, X_content, X_bf, y_)

        pickle.dump([X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train,
                     X_test_uid,  X_test_pid,  X_test_content, X_test_bf, y_test],  open(filename, 'wb') )
    return pickle.load(open(filename, mode='rb'))


def build_vocab(X_train, is_hierarchical=True):
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

    vocabulary_inv = []
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= 5] #
    # Mapping from word to index
    vocabulary = {x: i+1 for i, x in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


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
            word_vecs[word] = np.random.uniform(-0.25, 0.25, 300)
            # print(word)

    print(str(len(word_vecs) - count_missing)+" words found in word2vec.")
    print(str(count_missing)+" words not found, generated by random.")
    return word_vecs


def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size+1, 300), dtype='float32')
    #initialize the first row
    embedding_weights[0] = np.zeros(shape=(300,) )
    for idx in range(vocab_size):
        embedding_weights[idx+1] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size "+str(np.shape(embedding_weights)))
    return embedding_weights


def pad_sequence(X, max_words=20, max_sents=25, max_len=0):
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
        if len(doc) > 20:
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
    print("sents exceeds 20: ", sents_count)
    print("words exceeds 20: ", words_count)


def w2v_feature_extract(tbName, max_len=0, is_hierarchical=True):
    if is_hierarchical:
        root_path = "../hdata/"
        max_sents = 10
    else:
        root_path = "../data/"
        max_sents = 0

    X_train_uid, X_train_pid, X_train_content, X_train_bf, y_train, \
    X_test_uid,  X_test_pid,  X_test_content, X_test_bf, y_test = load_data_and_labels(root_path, tbName, is_hierarchical)
    # print(str(len(X_content)) + " sentences read")

    print("word2vec generation.......")
    vocabulary, vocabulary_inv = build_vocab(X_train_content, is_hierarchical)
    pickle.dump(vocabulary, open(root_path+tbName+"/vocab.pkl", 'wb') )
    print("Vocabulary size: "+str(len(vocabulary)))

    print("word embedding generation.......")
    word2vec = vocab_to_word2vec('./yelp_spam_kdd2015.bin', vocabulary)
    word_embeddings = build_word_embedding_weights(word2vec, vocabulary_inv)

    print("uid mapping .......")
    user_dict = {uid:i for i, uid in enumerate(set(X_train_uid + X_test_uid))}
    pickle.dump(user_dict, open(root_path+tbName+"/user_dict.pkl", 'wb'))
    X_train_uid = [user_dict[uid] for uid in X_train_uid]
    X_test_uid = [user_dict[uid] for uid in X_test_uid]

    print("pid mapping .......")
    product_dict = {pid:i for i, pid in enumerate(set(X_train_pid + X_test_pid))}
    pickle.dump(product_dict, open(root_path+tbName+"/product_dict.pkl", 'wb'))
    X_train_pid = [product_dict[pid] for pid in X_train_pid]
    X_test_pid = [product_dict[pid] for pid in X_test_pid]

    std = StandardScaler()
    X_train_bf = std.fit_transform(X_train_bf)
    X_test_bf = std.transform(X_test_bf)

    print("build input data.......")
    X_train = build_input_data(X_train_content, vocabulary, is_hierarchical, max_sents, max_len)
    X_test = build_input_data(X_test_content, vocabulary, is_hierarchical, max_sents, max_len)

    pickle.dump([X_train_uid, X_train_pid, X_train, X_train_bf, y_train, word_embeddings], open(root_path+tbName+"/train_mat.pkl", 'wb') )
    pickle.dump([X_test_uid, X_test_pid, X_test, X_test_bf, y_test], open(root_path+tbName+"/test_mat.pkl", 'wb') )


if __name__ == "__main__":
    w2v_feature_extract('yelpchi')
    w2v_feature_extract('yelpnyc')
    w2v_feature_extract('yelpzip')

