"""
Kaggle Bag of Words Tutorial

Code based on http://www.kaggle.com/c/word2vec-nlp-tutorial/forums/t/
              11261/beat-the-benchmark-with-shallow-learning-0-95-lb
"""

import os
from KaggleWord2VecUtility import KaggleWord2VecUtility
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.cross_validation import cross_val_score
import pandas as pd
import numpy as np
import logging
import cPickle as pickle
import errno


SIMPLE_VECT = True
SGDC = True
SHORT_REVIEW = True

logging.basicConfig(level=logging.DEBUG)
_logger = logging.getLogger(__name__)

base_path = os.path.dirname(__file__)
train_path = os.path.join(base_path, 'data', 'labeledTrainData.tsv')
test_path = os.path.join(base_path, 'data', 'testData.tsv')

_logger.info('loading data')
train = pd.read_csv(train_path, header=0, delimiter="\t", quoting=3)
test = pd.read_csv(test_path, header=0, delimiter="\t", quoting=3)

y = train['sentiment']
_logger.info('fraction positive class {f}'.format(f=float(sum(y))/y.shape[0]))

train_pkl = 'traindata.pkl'
test_pkl = 'testdata.pkl'
if SHORT_REVIEW:
    _logger.info('using shortened reviews')
    train_pkl = 'shortened_' + train_pkl
    test_pkl = 'shortened_' + test_pkl

try:
    traindata = pickle.load(open(os.path.join(base_path, 'data',
                                              train_pkl), 'r'))
    testdata = pickle.load(open(os.path.join(base_path, 'data',
                                             test_pkl), 'r'))
except IOError as e:
    if e.errno != errno.ENOENT:
        raise e
    else:
        _logger.info('cleaning and parsing movie reviews')

        traindata = []
        for i in xrange(0, len(train["review"])):
            review = KaggleWord2VecUtility.review_to_wordlist(train["review"][i],
                                                              False)
            if SHORT_REVIEW:
                review = review[:4]
            traindata.append(' '.join(review))
        testdata = []
        for i in xrange(0, len(test["review"])):
            review = KaggleWord2VecUtility.review_to_wordlist(test["review"][i],
                                                              False)
            if SHORT_REVIEW:
                review = review[:4]
            testdata.append(' '.join(review))

        pickle.dump(traindata, open(os.path.join(base_path, 'data',
                                                 train_pkl), 'w'))
        pickle.dump(testdata, open(os.path.join(base_path, 'data',
                                                test_pkl), 'w'))

_logger.info('number of training samples {}'.format(len(traindata)))
_logger.info('number of testing samples {}'.format(len(testdata)))

_logger.info('vectorizing')

tfv = None
if SIMPLE_VECT:
    _logger.info('using simple vectorizer')
    tfv = TfidfVectorizer()
else:
    _logger.info('using better vectorizer')
    tfv = TfidfVectorizer(min_df=3,  max_features=None,
                          strip_accents='unicode', analyzer='word',
                          token_pattern=r'\w{1,}', ngram_range=(1, 2),
                          use_idf=1, smooth_idf=1, sublinear_tf=1,
                          stop_words = 'english')

X_all = traindata + testdata
lentrain = len(traindata)

_logger.info('fitting pipeline')
tfv.fit(X_all)
X_all = tfv.transform(X_all)

X = X_all[:lentrain]
X_test = X_all[lentrain:]

if SGDC:
    _logger.info('using SGDClassifier')
    model = SGDClassifier(loss='log')
else:
    _logger.info('using LogisticRegression')
    model = LogisticRegression(penalty='l2', dual=True, tol=0.0001,
                               C=1, fit_intercept=True, intercept_scaling=1.0,
                               class_weight=None, random_state=None)

_logger.info('20 Fold CV Score: {}'.format(np.mean(cross_val_score(model, X, y, cv=20,
                                                                   scoring='roc_auc'))))

_logger.info('Retrain on all training data, predicting test labels')
model.fit(X, y)
result = model.predict_proba(X_test)[:, 1]
output = pd.DataFrame(data={"id": test["id"], "sentiment": result})

# Use pandas to write the comma-separated output file
output.to_csv(os.path.join(os.path.dirname(__file__), 'data',
                           'Bag_of_Words_model.csv'),
              index=False, quoting=3)
_logger.info('Wrote results to Bag_of_Words_model.csv')

top_ten = np.argsort(model.coef_[0])[::-1][:10]
top_ten_words = [tfv.get_feature_names()[i] for i in top_ten]
bottom_ten = np.argsort(model.coef_[0])[:10]
bottom_ten_words = [tfv.get_feature_names()[i] for i in bottom_ten]

_logger.info('Ten most positive words: {}'.format(top_ten_words))
_logger.info('Ten most negative words: {}'.format(bottom_ten_words))
