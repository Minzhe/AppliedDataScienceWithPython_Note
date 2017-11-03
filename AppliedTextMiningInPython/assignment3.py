import pandas as pd
import numpy as np
import operator
import re

spam_data = pd.read_csv('spam.csv')

spam_data['target'] = np.where(spam_data['target']=='spam',1,0)
spam_data.head(10)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(spam_data['text'], 
                                                    spam_data['target'], 
                                                    random_state=0)

def answer_one():
    
    return len(spam_data[spam_data.target == 1]) / len(spam_data) * 100

answer_one()

from sklearn.feature_extraction.text import CountVectorizer

def answer_two():
    vectorizer = CountVectorizer()
    vectorizer.fit(X_train)
    features = vectorizer.get_feature_names()
    length = list(map(len, features))
    return features[np.argmax(length)]

answer_two()

from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score

def answer_three():
    
    vectorizer = CountVectorizer().fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    MNB = MultinomialNB(alpha=0.1)
    MNB.fit(X_train_vectorized, y_train)

    predictions = MNB.predict(X_test_vectorized)
    return roc_auc_score(y_test, predictions)

answer_three()

from sklearn.feature_extraction.text import TfidfVectorizer

def answer_four():
    
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    features_idf = list(zip(vectorizer.get_feature_names(), vectorizer.idf_))
    small_idf = sorted(features_idf, key=operator.itemgetter(1))[:20]
    large_idf = sorted(features_idf, key=operator.itemgetter(1), reverse=True)[:20]
    small_idf = pd.Series(small_idf).apply(operator.itemgetter(0))
    large_idf = pd.Series(large_idf).apply(operator.itemgetter(0))
    return (small_idf, large_idf) #Your answer here

answer_four()

def answer_five():
    
    vectorizer = TfidfVectorizer(min_df=3).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    MNB = MultinomialNB(alpha=0.1)
    MNB.fit(X_train_vectorized, y_train)

    predictions = MNB.predict(X_test_vectorized)
    return roc_auc_score(y_test, predictions) #Your answer here

answer_five()

def answer_six():
    
    spam_length = spam_data[spam_data.target == 1].text.apply(len)
    non_spam_length = spam_data[spam_data.target == 0].text.apply(len)
    return (np.mean(non_spam_length), np.mean(spam_length)) #Your answer here

answer_six()

from sklearn.svm import SVC

def add_feature(X, feature_to_add):
    """
    Returns sparse feature matrix with added feature.
    feature_to_add can also be a list of features.
    """
    from scipy.sparse import csr_matrix, hstack
    return hstack([X, csr_matrix(feature_to_add).T], 'csr')

def answer_seven():
    
    vectorizer = TfidfVectorizer(min_df=5).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    
    X_train_vectorized = add_feature(X_train_vectorized, X_train.str.len())
    X_test_vectorized = add_feature(X_test_vectorized, X_test.str.len())
    
    clf = SVC(C=10000)
    clf.fit(X_train_vectorized, y_train)
    predictions = clf.predict(X_test_vectorized)
    return roc_auc_score(y_test, predictions) #Your answer here

answer_seven()

def answer_eight():
    
    spam_digit_length = spam_data[spam_data.target == 1].text.apply(lambda x: re.findall(r'[0-9]', x)).str.len()
    nonspam_digit_length = spam_data[spam_data.target == 0].text.apply(lambda x: re.findall(r'[0-9]', x)).str.len()
    return (np.mean(nonspam_digit_length), np.mean(spam_digit_length)) #Your answer here

answer_eight()

from sklearn.linear_model import LogisticRegression

def answer_nine():
    
    vectorizer = TfidfVectorizer(min_df=5, ngram_range=[1,3]).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X_train_vectorized = add_feature(X_train_vectorized, [X_train.apply(len),
                                                          X_train.apply(lambda x: re.findall(r'[0-9]', x)).str.len()])
    X_test_vectorized = add_feature(X_test_vectorized, [X_test.apply(len),
                                                        X_test.apply(lambda x: re.findall(r'[0-9]', x)).str.len()])

    clf = LogisticRegression(C=100)
    clf.fit(X_train_vectorized, y_train)
    predictions = clf.predict(X_test_vectorized)
    return roc_auc_score(y_test, predictions) #Your answer here

answer_nine()

def answer_ten():
    
    spam_nonw = spam_data[spam_data.target == 1].text.apply(lambda x: re.findall(r'\W', x)).str.len()
    nonspam_nonw = spam_data[spam_data.target == 0].text.apply(lambda x: re.findall(r'\W', x)).str.len()
    return (np.mean(nonspam_nonw), np.mean(spam_nonw)) #Your answer here

answer_ten()

def answer_eleven():
    
    vectorizer = CountVectorizer(min_df=5, analyzer='char_wb', ngram_range=[2,5]).fit(X_train)
    X_train_vectorized = vectorizer.transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    X_train_vectorized = add_feature(X_train_vectorized, [X_train.apply(len),
                                                          X_train.apply(lambda x: re.findall(r'[0-9]', x)).str.len(),
                                                          X_train.apply(lambda x: re.findall(r'\W', x)).str.len()])
    X_test_vectorized = add_feature(X_test_vectorized, [X_test.apply(len),
                                                        X_test.apply(lambda x: re.findall(r'[0-9]', x)).str.len(),
                                                        X_test.apply(lambda x: re.findall(r'\W', x)).str.len()])

    clf = LogisticRegression(C=100)
    clf.fit(X_train_vectorized, y_train)
    predictions = clf.predict(X_test_vectorized)
    auc_roc = roc_auc_score(y_test, predictions)
    features = np.array(vectorizer.get_feature_names() + ['length_of_doc', 'digit_count', 'non_word_char_count'])

    coeff_idx_order = clf.coef_[0].argsort()
    small_coeff = features[coeff_idx_order[:10]]
    large_coeff = features[coeff_idx_order[:-11:-1]]
    return (auc_roc, list(small_coeff), list(large_coeff)) #Your answer here

answer_eleven()