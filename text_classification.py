import os
import os.path
import sys
from tqdm import tqdm
import numpy as np
import multiprocessing
import pandas as pd
import csv
import seaborn as sns
import matplotlib.pyplot as plt
import re
import collections
import string
import time


# sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn import utils
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

# nltk
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
nltk.download('punkt')
nltk.download('stopwords')

# pytorch
import torch
from torch.nn import functional as F
from torchtext import data
from torchtext import datasets
from torchtext.vocab import Vectors, GloVe
import torch.nn as nn
import torch.optim as optim
import torchtext.vocab as vocab
from torchtext.data.utils import get_tokenizer

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import xgboost as xgb
import pickle

EMBEDDING_SIZE = 300 # embedding vector size 
TRAIN_FILE = 'trump_train.tsv'
TEST_FILE = 'trump_test.tsv'
LSTM_MODEL_FILE = 'lstm-model.pt'
RNN_MODEL_FILE = 'rnn-model.ptg'
XGB_MODEL_FILE = 'xgb-model.pt'
EMBEDDING_FILE = 'tweetemd.txt'
TFIDF_FILE = 'vectorizer_tfidf.pk'

"""# Pre-Process & Auxiliary Functions"""

def who_am_i():
    """Returns a ductionary with your name, id number and email. keys=['name', 'id','email']
        Make sure you return your own info!
    """
    return {'name': 'Chen Dahan & Miri Yitshaki', 'id': '204606651 & 025144635', 'email': 'dahac@post.bgu.ac.il'}

def load_data(path):
    """ Load csv
        Args:
            path: the csv file path
        Returns:
            df with label column created using the device column (android -> trump -> 1, else 0 )
    """
    # read csv
    df = pd.DataFrame()
    try: 
      df = pd.read_csv(path, sep='\t', names=['tweet_id', 'user_handle', 'tweet_text', 'time_stamp', 'device'],
                     quoting=csv.QUOTE_NONE)
    except IOError as e:
      return df
    # set label using the device 
    df['label'] = df['device'].apply(lambda x: 1 if x == 'android' else 0)
    return df

def load_test_file(path):
    """ Load csv test file
        Args:
            path: the csv file path
        Returns:
            df with label column created using the device column (android -> trump -> 1, else 0 )
    """
    # read csv
    df = pd.DataFrame()
    try: 
      df = pd.read_csv(path, sep='\t', names=['user_handle', 'tweet_text', 'time_stamp'],
                     quoting=csv.QUOTE_NONE)
    except IOError as e:
      return df
    return df

def preprocess_text(text):
    """ Preprocess text, remove signs 
        some punctuations removed with space instead and some without (for example ! removed without)  
        Args:
            text: the text to remove from
        Returns:
            text : the cleaned text
    """
    REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
    REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)") 
    text = REPLACE_NO_SPACE.sub("", text.lower())
    text = REPLACE_WITH_SPACE.sub(" ", text)

    return text

def normalize_text(text, stop_words = None, remove_stop_words = False):
    """ Normalize text- lower case, remove signs, stop_words, single letter words and more
        Args:
            text: the text to clean
            stop_words: boolean indicating whether to remove stop words
        Returns:
            text : the cleaned text
    """
    text = preprocess_text(text)
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    if remove_stop_words: # remove stop words
      text = [x for x in text if x not in stop_words]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    text = " ".join(text)
    return text


def add_time_features(df, featurs_names):
    """ Adding time featurs to the df 
        Args:
            df: dataframe containing the time_stamp column 
            featurs_names: list to add the added featurs names 
    """
    df['time_stamp'] = df['time_stamp'].astype('datetime64[ns]')
    df['tweet_day'] = df['time_stamp'].dt.dayofweek
    df['tweet_month'] = df['time_stamp'].dt.month
    df['tweet_hour'] = df['time_stamp'].dt.hour
    df['tweet_year'] = df['time_stamp'].dt.year
    for i in ['tweet_day','tweet_month','tweet_hour','tweet_year']:
      featurs_names.append(i) 

def add_tweet_featurs(df, featurs_names):
    """ Adding featurs related to the tweet
        Args:
            df: dataframe containing the tweet_text column
            featurs_names: list to add the added featurs names 
    """
    # Special Character Count - # hashtag or @ tag
    df['spl'] = df['tweet_text'].apply(lambda x: len([x for x in x.split() if x.startswith('@')]))
    # Word count per tweet
    df['word_count'] = df['tweet_text'].str.split().str.len()
    # Characters counter per tweet
    df['character_count'] = df['tweet_text'].str.len()
    # Count number of characters per word
    df['characters_per_word'] = df['character_count']/df['word_count']
    # Count of numbers in a tweet
    df['count_number'] = df['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
    for i in ['spl','word_count','character_count','characters_per_word','count_number']:
      featurs_names.append(i) 

def add_tf_idf_vector_from_file(df, featurs_names, column_name):
    """ Adding tf-idf vectors based on TfidfVectorizer model loded from a file 
        Args:
            df: dataframe containing the tweet_text column
            featurs_names: list to add the added featurs names 
            column_name: the text to create the tfidf vectors on 
        Returns:
            text : df containing the original df and the new columns of the TfidfVectorizer 
    """
    with open(TFIDF_FILE, 'rb') as fin:
      tfidf = pickle.load(open(TFIDF_FILE, 'rb'))
    dat_tfIdf = tfidf.transform(df[column_name])
    with open(TFIDF_FILE, 'wb') as fin:
      pickle.dump(tfidf, fin)
    count_vect_df = pd.DataFrame(dat_tfIdf.todense(), columns=tfidf.get_feature_names())
    count_vect_df = count_vect_df.add_suffix('_tf_idf')
    for i in count_vect_df.columns:
      featurs_names.append(i)
    return df.join(count_vect_df)
    

def add_tf_idf_vector(df, featurs_names, column_name):
    """ Adding tf-idf vectors based on TfidfVectorizer
        Args:
            df: dataframe containing the tweet_text column
            featurs_names: list to add the added featurs names 
            column_name: the text to create the tfidf vectors on 
        Returns:
            text : df containing the original df and the new columns of the TfidfVectorizer 
    """
    # tf-idf vector - 3528 rows, row for each tweet, EMBEDDING_SIZE entries
    tfidf = TfidfVectorizer(max_features= EMBEDDING_SIZE, lowercase=True, analyzer='word', stop_words='english', ngram_range=(1, 1))
    tfidf.fit(df[column_name])
    dat_tfIdf = tfidf.transform(df[column_name])
    with open(TFIDF_FILE, 'wb') as fin:
      pickle.dump(tfidf, fin)
    count_vect_df = pd.DataFrame(dat_tfIdf.todense(), columns=tfidf.get_feature_names())
    count_vect_df = count_vect_df.add_suffix('_tf_idf')
    for i in count_vect_df.columns:
      featurs_names.append(i)
    return df.join(count_vect_df)

def add_bow_vector(df, featurs_names, column_name):
    """ Adding bow vectors based on CountVectorizer
        Args:
            df: dataframe containing the tweet_text column
            featurs_names: list to add the added featurs names 
            column_name: the text to create the tfidf vectors on 
        Returns:
            text : df containing the original df and the new columns of the TfidfVectorizer 
    """
    # Bag-of-words Vector - 3528 rows, row for each tweet, EMBEDDING_SIZE entries 
    bag_words = CountVectorizer(max_features=EMBEDDING_SIZE, lowercase=True, ngram_range=(1,1),analyzer = "word")
    dat_BOW = bag_words.fit_transform(df[column_name])
    count_vect_df = pd.DataFrame(dat_BOW.todense(), columns=bag_words.get_feature_names())
    count_vect_df = count_vect_df.add_suffix('_bow')
    for i in count_vect_df.columns:
      featurs_names.append(i) 
    return df.join(count_vect_df)

def prepare_df(file, tf_idf_bow, normalize_featurs):
  """ Loading the df from the file and adding featurs
        Args:
            file: csv file location
            tf_idf_bow: number indicating if to add tf_idf featurs(1), bow(2), or non(3). 
            normalize_featurs: boolean indicating wether to normalize the featurs
        Returns:
            featurs_names: list of featurs to use for   
            df: data frame containing all the featurs
  """
  featurs_names = []
  # check if tsv file exists 
  df = load_data(file)
  df,featurs_names = process_df(df, tf_idf_bow, normalize_featurs, False)
  return df, featurs_names

def process_df(df, tf_idf_bow, normalize_featurs, load_tfidf):
  ''' load_tfidf - boolean indicating if to load tfidf vectorize object 
  '''
  featurs_names = []
  if not df.empty:
    stop_words = stopwords.words('english')
    # normalize text
    df['tweet_normalized'] = df['tweet_text'].apply(lambda x: normalize_text(x, stop_words, True))
    stemmer = PorterStemmer()
    df['tweet_normalized_stemmed'] = df['tweet_normalized'].apply(lambda x: " ".join(stemmer.stem(word) for word in x.split()))
    # add feature
    add_time_features(df,featurs_names)
    add_tweet_featurs(df,featurs_names)
    if tf_idf_bow ==1 :
      if load_tfidf:
        df = add_tf_idf_vector_from_file(df,featurs_names, 'tweet_normalized')
      else:
        df = add_tf_idf_vector(df,featurs_names, 'tweet_normalized')
    elif tf_idf_bow ==2 :
      df = add_bow_vector(df,featurs_names,'tweet_normalized')
    # normalization
    if normalize_featurs:
      df[featurs_names] = MinMaxScaler().fit_transform(df[featurs_names])
  return df, featurs_names

def check_performance_sklearn_requierd_models(df, featurs_names):
    """ Loading the df from the file and adding featurs
        print results and return best
        Args:
            df: data frame
            featurs_names: list of featurs names to use
        Returns:
          the best model 
    """
    clf_best = None
    X= df[featurs_names]
    y= df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)
    score_best, clf_best = grid_search_model(LogisticRegression(max_iter= 800), {'C': [0.01, 0.1, 1, 10, 100]}, 10, X_train, y_train, X_test, y_test)
    # SVC - nonlinear kernel
    score, clf = grid_search_model(SVC(probability= True),{'kernel': ['poly', 'rbf', 'sigmoid'],'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]}, 10, X_train, y_train, X_test, y_test)
    if score_best < score:
      score_best = score
      clf_best = clf
    # linear kernel
    score, clf = grid_search_model(SVC(probability= True,kernel ='linear'),{'C': [1, 10, 100, 1000]}, 10, X_train, y_train, X_test, y_test)
    if score_best < score:
      score_best = score
      clf_best = clf
    # XGboost
    score, clf = grid_search_model(xgb.XGBClassifier(objective="binary:logistic"), {"learning_rate": [0.1, 0.15, 0.20, 0.25, 0.30],
                                    "max_depth":[ 4, 6, 8, 10,15],"colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ] }
                                   , 10, X_train, y_train, X_test, y_test)
    return X, clf_best

def grid_search_model(clf, param_grid, cv, X_train, y_train, X_test, y_test):
  """ Param grid search model
          Print and return the best
        Args:
            clf: the estimator (SVC/LogisticRegression)
            param_grid: param grid
            cv: cv
        Returns:
          the best model and the best score 
  """
  # StratifiedKFold is used in GridSearchCV
  clf_gs = GridSearchCV(clf, param_grid= param_grid, cv=cv)
  clf_gs.fit(X_train, y_train)
  print("Best model:", clf_gs.best_estimator_)
  print("Best parameters:", clf_gs.best_params_)
  print("10-fold cv train:", clf_gs.best_score_)
  roc = roc_auc_score(y_train, clf_gs.predict_proba(X_train)[:,1])
  print("Train ROC: ",roc)
  print("Test accuracy: ", clf_gs.score(X_test,y_test))
  roc = roc_auc_score(y_test, clf_gs.predict_proba(X_test)[:,1])
  print("Test ROC: ",roc)
  return clf_gs.best_score_, clf_gs.best_estimator_


def print_feature_importance(X, estimator):
  """ Print_feature_importance top20
        Args:
            X: the train data
            estimator: the trained estimator to check
        Returns:
          the best model 
  """
  # feature importance
  try:
    if type(estimator) == type(xgb.XGBClassifier()):
      coefs = estimator.feature_importances_
    else:
      coefs =  estimator.coef_[0]
    feature_to_coef = {
        word: coef for word, coef in zip(
        X.columns, estimator.coef_[0] )}
    print('\npositive')
    for best_positive in sorted(
        feature_to_coef.items(),
        key = lambda x: x[1],
        reverse = True)[:20]:
        print(best_positive)
    print('\nnegative')
    for best_positive in sorted(
        feature_to_coef.items(),
        key = lambda x: x[1],
        reverse = False)[:20]:
        print(best_positive)
  except:
    print('coef_ is only available when using a linear kernel')
    return 

def train_and_save_best_model():
  ''' train and save best model - in our case xgbost with tfidf vectors
      and print ststistics using kfold cross validation 
      this function was made for our use, to get ststistics
  '''
  df, featurs_names = prepare_df(TRAIN_FILE, 1, True)
  clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                colsample_bynode=1, colsample_bytree=0.5, gamma=0,
                learning_rate=0.15, max_delta_step=0, max_depth=15,
                min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                nthread=None, objective='binary:logistic', random_state=0,
                reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                silent=None, subsample=1, verbosity=1)
  X = df[featurs_names]
  y = df['label']
  X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
  skf = StratifiedKFold(n_splits=10)
  #  kfold = KFold(n_splits=10, random_state=7)
  results = cross_val_score(clf, X_train, y_train, cv=skf)
  print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
  clf.fit(X_train,y_train)
  print('auc val: ', roc_auc_score(y_val, clf.predict_proba(X_val)[:,1]))
  print(confusion_matrix(y_val, clf.predict(X_val)))
  # clf.save_model(XGB_MODEL_FILE)
  pickle.dump(clf, open(XGB_MODEL_FILE, 'wb'))
  xgb.plot_importance(clf, max_num_features=15)
  cm = confusion_matrix(y_true=y, 
                      y_pred=clf.predict(X))
  print(pd.DataFrame(cm,index=clf.classes_, columns=clf.classes_))

"""# EDA"""

# in case you wish to print the eda we preformed 
# exploratory_data_analysis()

def exploratory_data_analysis():
  """ explore featurs in the data
  """
  df, featurs_names = prepare_df(TRAIN_FILE, 3, False)
  dist_by_hour(df)
  dist_by_hour_for_each_year(df)
  dist_by_day(df)
  dist_by_month(df)
  dist_by_year(df)
  dist_1gram(df)
  dist_spl(df)
  dist_word_count(df)
  dist_characters_count(df)
  dist_characters_per_word(df)
  dist_count_number(df)

 ## Auxiliary functions used for creating distributions as part of the final report ##

def dist_by_hour(df):
  # Distribution by hour 
  _ = plt.figure(figsize=(8,5))
  tweet_counts = (df.groupby(['label'])['tweet_hour']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values('tweet_hour'))
  _ = sns.barplot(x="tweet_hour", y="percentage", hue="label",data=tweet_counts).set_title('Distribution by hour').set_fontsize('14')

def dist_by_hour_for_each_year(df):
  # Distribution by hour - for each year separately
  _ = plt.figure(figsize=(8,8))
  tweet_counts = (df.groupby(['label','tweet_year'])['tweet_hour']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values(['tweet_year','tweet_hour']))
  _ = sns.catplot(x="tweet_hour",y='percentage', hue="label", col="tweet_year",data=tweet_counts,kind='bar')
  _.fig.subplots_adjust(top=0.8)
  _.fig.suptitle('Distribution by hour - for each year separately', fontsize=16).set_fontsize('14')

def dist_by_day(df):
  # Distribution by day 
  tweet_counts = (df.groupby(['label'])['tweet_day']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values('tweet_day'))
  _ = sns.barplot(x="tweet_day", y="percentage", hue="label", data=tweet_counts).set_title('Distribution by day').set_fontsize('14')

def dist_by_month(df):
  # Distribution by month 
  plt.figure(figsize=(8,5))
  tweet_counts = (df.groupby(['label'])['tweet_month']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values('tweet_month'))
  _ = sns.barplot(x="tweet_month", y="percentage", hue="label", data=tweet_counts).set_title('Distribution by month').set_fontsize('14')

def dist_by_year(df):
  # Distribution by year
  plt.figure(figsize=(8,5))
  tweet_counts = (df.groupby(['label'])['tweet_year']
                      .value_counts(normalize=True)
                      .rename('percentage')
                      .mul(100)
                      .reset_index()
                      .sort_values('tweet_year'))
  _ = sns.barplot(x="tweet_year", y="percentage", hue="label", data=tweet_counts).set_title('Distribution by year').set_fontsize('14')


def dist_1gram(df):
  # Create 1-gram counter for each tag
  trump_1gram = collections.Counter()
  none_trump_1gram = collections.Counter()
  for idx, row in df.iterrows(): 
    if row["label"] == 1:
      trump_1gram = trump_1gram + collections.Counter(row['tweet_normalized'].split())
    else:
      none_trump_1gram = none_trump_1gram + collections.Counter(row['tweet_normalized'].split())
  
  trump_top_grams(trump_1gram, none_trump_1gram)
  not_trump_top_grams(trump_1gram, none_trump_1gram)

def dist_spl(df):
  # Special Character Count - # hashtag or @ tag - twitter. (using the original text) 
  plt.figure(figsize=(6,6))
  # Plot distribution, normalize in the amount of sentences
  df_temp = df.groupby(["spl","label"], as_index= False)['tweet_id'].count()
  norm_trump = len(df[df['label'] == 1])
  norm_not_trump = len(df) - norm_trump
  #normalize
  df_temp['tweet_count_norm'] = df_temp.apply(lambda x: x['tweet_id']/norm_trump if x['label']==1 else x['tweet_id']/norm_not_trump, axis=1 )
  _ = sns.barplot(x="spl",y='tweet_count_norm', hue="label", data = df_temp).set_title('Distribution by year')
  
def dist_word_count(df):
  # Word count per tweet
  print(df.groupby('label')['word_count'].mean())
  _ = sns.displot(df, x="word_count", hue="label",stat="density").set_title('Word count per tweet')

def dist_characters_count(df):
  # Characters counter per tweet
  print(df.groupby('label')['character_count'].mean())
  _ = sns.displot(df, x="character_count", hue="label",stat="density").set_title('Characters counter per tweet')

def dist_characters_per_word(df):
  # Count number of characters per word
  print(df.groupby('label')['characters_per_word'].mean())
  _ = sns.displot(df, x="characters_per_word", hue="label",stat="density").set_title('Count number of characters per word')

def dist_count_number(df):
  # Count of numbers in a tweet
  df['count_number'] = df['tweet_text'].apply(lambda x: len([x for x in x.split() if x.isdigit()]))
  print(df.groupby('label')['count_number'].mean())

def trump_top_grams(trump_1gram, none_trump_1gram):
  # Trupm top 1grams distribution in percentages compared to the 1gram percentage in 'Not Trump'
  df_word_count = pd.DataFrame.from_records(trump_1gram.most_common(20), columns =['word', 'count'])
  df_word_count['count'] /= sum(trump_1gram.values())
  plt.figure(figsize=(6,6))
  ax = sns.barplot(x="count", y="word", data=df_word_count, palette="Blues_d")
  ax.set_title('Top Used Words - Trump').set_fontsize('14')
  ax.set_ylabel('words',fontsize=14)
  ax.tick_params(labelsize=12)

def not_trump_top_grams(trump_1gram, none_trump_1gram):
  # Not Trupm top 1grams distribution in percentages compared to the 1gram percentage in Trump
  df_word_count = pd.DataFrame.from_records(none_trump_1gram.most_common(20), columns =['word', 'count'])
  df_word_count['label'] = 'Not Trump'
  df_word_count['count'] /= sum(none_trump_1gram.values())
  plt.figure(figsize=(6,6))
  ax = sns.barplot(x="count", y="word", data=df_word_count, palette="Blues_d")
  ax.set_title('Top Used Words - Not Trump').set_fontsize('14')
  ax.set_ylabel('words',fontsize=14)
  ax.tick_params(labelsize=12)

"""# Gensim"""

def tokenize_text(text):
  """ Tokenize text while removing words shorter than 2
        Args:
            text: the text to tokenize
        Returns:
            tokens
  """
  tokens = []
  for sent in nltk.sent_tokenize(text):
    for word in nltk.word_tokenize(sent):
      if len(word) < 2:
        continue
      tokens.append(word)
  return tokens


def train_word2vec_loop(train_text):
  """ Training loop - gensim word2vec 
      model params- 
      vector_size = Dimensionality of the feature vectors.
      window = The maximum distance between the current and predicted word within a sentence.
      min_count = Ignores all words with total frequency lower than this.
      alpha = The initial learning rate.
        Args:
            train_text: list of gensim.models.doc2vec.TaggedDocument
        Returns:
            word2vec model
  """
  cores = multiprocessing.cpu_count()
  model_w2v = Doc2Vec(dm=1, dm_mean=1, vector_size=EMBEDDING_SIZE, window=10, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
  model_w2v.build_vocab([x for x in tqdm(train_text)])

  for epoch in range(20):
      model_w2v.train(utils.shuffle([x for x in tqdm(train_text)]), total_examples=len(train_text), epochs=1)
      model_w2v.alpha -= 0.002
      model_w2v.min_alpha = model_w2v.alpha

  model_w2v.save('dm_model_v2.model')
  return model_w2v


def train_word2vec(sentences_series):
  """ Training gensim Doc2vec
        Args:
            sentences_series: the text to tokenize
        Returns:
            df containing the vectors created by the model for each doc(tweet) 
  """
  # prepare train data 
  train_text = [TaggedDocument(words=tokenize_text(_d), tags=[i]) for i, _d in enumerate(sentences_series)]
  model = train_word2vec_loop(train_text) # train model
  doc2vec_df = [model.docvecs[doc.tags[0]] for doc in train_text]
  doc2vec_df = pd.DataFrame(doc2vec_df).add_suffix('_d2v')
  return doc2vec_df 
  

def train_and_get_w2v_vectors(column_name, df , featurs_names, normalize_vectors):
  """ Training gensim doc2vec. 
      Training using  df[column_name], adding word2vec columns names to featurs_names and 
      joining the word2vec docs vectors to the df  
        Args:
            column_name: the column of the sentences  
            df: data frame containing column_name
            featurs_names: list of featurs name
             normalize_vectors: boolean indicating wether to normalize the featurs
        Returns:
            df containing the orig df and the vectors created by the model for each doc 
  """
  doc2vec_df = train_word2vec(df[column_name])
  if normalize_vectors:
    doc2vec_df[doc2vec_df.columns] = MinMaxScaler().fit_transform(doc2vec_df[doc2vec_df.columns])
  for i in doc2vec_df.columns:
    featurs_names.append(i) 
  return df.join(doc2vec_df)

# for creating the ML models with gensim w2vec embedding
# load df
# df, featurs_names = prepare_df(TRAIN_FILE, 3, True)
# add word2vec gensim vectors featurs 
# df = train_and_get_w2v_vectors('tweet_text', df, featurs_names, True)
# df_ = df.copy()
# look for best model
# X, best_model = check_performance_sklearn_requierd_models(df_, featurs_names)
# print_feature_importance(X, best_model)

"""# This is the code we used to run the ML models """

# load df with tf-idf (change 1->2 for bow)
# df, featurs_names = prepare_df(TRAIN_FILE, 1, True)
# df_ = df.copy()
# X, best_model = check_performance_sklearn_requierd_models(df_, featurs_names)
# print_feature_importance(X, best_model)

"""# Pre-NN & Auxiliary NN Functions"""

NN_DF_FILE = 'df_filterd.csv'
BATCH_SIZE = 64
MAX_VOCAB_SIZE = 10000

def count_parameters(model):
  """ Count of parameters in a model
        Args:
            model: nn model
        Returns:
            total number of elements in all the input tensors of the model 
  """
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def binary_accuracy(preds, true_val):
  """
  Returns accuracy per batch, i.e. if you get 8/10 right, this returns 0.8, NOT 8
        Args:
          preds: model prediction
          true_val: ground truth label 
        Returns:
            accuracy
  """
  #round predictions to the closest integer
  rounded_preds = torch.round(torch.sigmoid(preds))
  correct = (rounded_preds == true_val).float() #convert into float for division 
  acc = correct.sum() / len(correct)
  return acc

def PlotLR (train_loss_list,valid_loss_list,best_val_acc,Name):
  """
  plot model learning 
        Args:
          train_loss_list:train loss 
          valid_loss_list:validation loss 
          best_val_acc: best model accuracy
          Name: network name
  """
  plt.plot(range(len(train_loss_list)),train_loss_list)
  plt.plot(range(len(train_loss_list)),valid_loss_list,color ='darkgreen')
  plt.title('Model loss:' + Name)
    
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'val'], loc='upper left')
  plt.show()
  print(f'\t Val. Acc: {best_val_acc*100:.2f}%') 

def epoch_time(start_time, end_time):
  elapsed_time = end_time - start_time
  elapsed_mins = int(elapsed_time / 60)
  elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
  return elapsed_mins, elapsed_secs

def predict_tweet(model, text):
  """ predict tweet - normalize and predict 0/1
        Args:
            model: the nn model that supports predict function(lstm/rnn)
            tweet: the text to use
        Returns:
            prediction - 0/1
  """
  stop_words = stopwords.words('english')
  normalized_text = normalize_text(text,stop_words, True)
  return model.predict(normalized_text)

# pepare the df for rnn/lstm !
# df, featurs_names = prepare_df(TRAIN_FILE, 3, True)
# # Prepare csv file for the NN
# features_list_nn = ['label','tweet_normalized' ,'tweet_text']
# df_filterd = df[features_list_nn]
# df_filterd.to_csv(NN_DF_FILE, index=False)

"""# RNN

3 Layers RNN network 

First layer – simple embedding to transform our sparse one hot vector into dense embedding vector using simple fully connected layer

Second layer – RNN with 256 hidden layers

Third – linear layer to give probability for each class 


"""

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_EPOCHS = 40


def build_rnn_dataset(file_path = NN_DF_FILE):
  """
  Build the pytorch data set object using the nltk Tweet tokenizer  
        Args:
          file_path:csv file craeted after data processin with label, tweet after processing , original tweet
        Returns:
            data_tabular - data set
            tweet_norm  pytorch filed object for out tweet    
  """
  label = data.LabelField(dtype = torch.float)
 
  tweet_norm = data.Field(tokenize = TweetTokenizer().tokenize)
  tweet_orig= data.Field()
  fields = [
    ('label', label),
    ('tweet',tweet_norm ), 
    ('tweet_orig',tweet_orig) 
  ]
  data_tabular = data.TabularDataset(
    path=file_path, format='csv',
    fields=fields,skip_header = True)

  tweet_norm.build_vocab(data_tabular)
  label.build_vocab(data_tabular)
  tweet_orig.build_vocab(data_tabular)
  print("Lets look on the common word in our nn vocabulary")
  print(tweet_norm.vocab.freqs.most_common(20))
  print(f"Unique tokens in TEXT vocabulary: {len(tweet_norm.vocab)}")
  return data_tabular, tweet_norm

# Our RNN network class 
class RNN(nn.Module):

  def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, tweet_vocab):
    super().__init__()
    self.embedding = nn.Embedding(input_dim, embedding_dim)   
    self.rnn = nn.RNN(embedding_dim, hidden_dim)
    self.fc = nn.Linear(hidden_dim, output_dim)
    self.tweet_vocab = tweet_vocab

  def forward(self, text):
    embedded = self.embedding(text)
    output, hidden = self.rnn(embedded)
    assert torch.equal(output[-1,:,:], hidden.squeeze(0))
    return self.fc(hidden.squeeze(0))

  def get_tweet_vocab(self):
    return self.tweet_vocab

  def predict(self,normalized_text):
    tokenized = normalized_text.split(' ')
    indexed = [self.tweet_vocab.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    prediction = torch.sigmoid(self(tensor))
    if prediction.item() > 0.5 :
        return 0
    else: 
        return 1 

def train_batch_rnn(model, iterator, optimizer, criterion): 
  """
  train_batch_rnn train our rnn on one batch  
        Args:
          model:our model object 
          iterator:Iterator object on the data set 
          optimizer: SGD 
          criterion:loss function
        Returns:
          loss,acc      
  """ 
  epoch_loss = 0
  epoch_acc = 0
  model.train()
  for batch in iterator:
    optimizer.zero_grad()        
    predictions = model(batch.tweet).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label) 
    loss.backward() 
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()   
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_batch_rnn(model, iterator, criterion):
  """
  evaluate_batch_rnn evlualte on our validation data  
    Args:
        model:our model object 
          iterator:Iterator object on the data set 
          optimizer: SGD 
          criterion:loss function
    Returns:
        loss,acc  
  """ 
  epoch_loss = 0
  epoch_acc = 0
  model.eval()
  with torch.no_grad():
    for batch in iterator:
      predictions = model(batch.tweet).squeeze(1)
      loss = criterion(predictions, batch.label)
      acc = binary_accuracy(predictions, batch.label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def train_loop_rnn(rnn_model, train_iterator, valid_iterator):
  """
  train_loop_rnn our main rnn loop for the training 
  First we create the loss and optimizer object and than loop and per epoc run train on batch
  After each epoc we compare of the loss is better on validation that we save the best model to file
  After going over all epoc we plot the loss per epoc for train , validation and print best accuracy for validation 
    Args:
        model:our model object 
        train_iterator:Iterator object on the data train set 
        valid_iterator: on the validation data set  
          
     
  """ 
  #optimizer = optim.SGD(rnn_model.parameters(), lr=1e-1)#
  optimizer = optim.SGD(rnn_model.parameters(), lr=0.009)
  criterion = nn.BCEWithLogitsLoss()
  best_valid_loss = float('inf')
  train_loss_list =[]
  valid_loss_list =[]
  for epoch in range(N_EPOCHS):
      start_time = time.time()
      train_loss, train_acc = train_batch_rnn(rnn_model, train_iterator, optimizer, criterion)
      valid_loss, valid_acc = evaluate_batch_rnn(rnn_model, valid_iterator, criterion)
      end_time = time.time()
      epoch_mins, epoch_secs = epoch_time(start_time, end_time)
      if valid_loss < best_valid_loss:
          best_valid_loss = valid_loss
          torch.save(rnn_model.state_dict(), RNN_MODEL_FILE)
          best_val_acc = valid_acc
      train_loss_list.append (train_loss)
      valid_loss_list.append (valid_loss)
      print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
      print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
      print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')

  PlotLR (train_loss_list,valid_loss_list,best_val_acc,"Simple RNN")


def build_and_train_rnn():
  """
  build_and_train_rnn - create our model object and train it 
    
  """ 
  data_tabular, tweet = build_rnn_dataset()
  INPUT_DIM = len(tweet.vocab)
  train_data, valid_data = data_tabular.split()
  print(vars(train_data.examples[0]))
  # Creating Itertor for on the train and Valid
  train_iterator,valid_iterator  = data.BucketIterator.splits(
      (train_data,valid_data), sort_key=lambda x: len(x.tweet),shuffle = True,
      batch_size = BATCH_SIZE)
  rnn_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, tweet)
  print(f'The Simple RNN model has {count_parameters(rnn_model):,} trainable parameters')
  train_loop_rnn(rnn_model, train_iterator, valid_iterator )
  return rnn_model


def load_rnn_model():
  # first create the model
  data_tabular, tweet_norm = build_rnn_dataset() 
  INPUT_DIM = len(tweet_norm.vocab)
  PAD_IDX = tweet_norm.vocab.stoi[tweet_norm.pad_token]
  rnn_model = RNN(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, tweet_norm)
  # load the paramters from a file containing the paramther for the best model from the training 
  rnn_model.load_state_dict(torch.load(RNN_MODEL_FILE))
  rnn_model.eval()
  return rnn_model

def load_rnn_and_predict_test_file(test_file):
  predictions = []
  rnn_model = load_rnn_model()
  df_test = load_test_file(test_file)

  for i,tweet in enumerate(df_test['tweet_text']):
    predictions.append(predict_tweet(rnn_model,tweet))
  return predictions

# to build and train rnn model from scretch 
#rnn_model = build_and_train_rnn()

"""# LSTM
Bi directional - 4 layer LSTM

First layer – here we used two options:
 Using pertained google-new vectors (We built a file only related to our corpus)
 Simple embedding 

Second, Third layer –  2 - Bi-directional LSTM layers

Forth – linear layer to give probability for each class 

We used drop out to overcome overfitting 

"""

EMBEDDING_DIM = 300
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

def load_pretrained_embedding():

  trained_embeddings = vocab.Vectors(name = EMBEDDING_FILE,
                                    cache = 'custom_embeddings',
                                    unk_init = torch.Tensor.normal_)
  EMBEDDING_DIM = 300
  return trained_embeddings

def build_lstm_dataset(file_path = NN_DF_FILE,with_pretarin_vec = False):
  """
  Build the pytorch data set object using the nltk Tweet tokenizer  
        Args:
          file_path:csv file craeted after data processin with label, tweet after processing , original tweet
          with_pretarin_vec:Flag if to use pretarin word embedding 
        Returns:
            data_tabular - data set
            tweet_norm  pytorch filed object for out tweet    
  """
  label = data.LabelField(dtype = torch.float)
  tweet_norm = data.Field(tokenize = TweetTokenizer().tokenize, include_lengths = True)
  tweet_orig= data.Field()
  fields = [
    ('label', label),
    ('tweet',tweet_norm ), 
    ('tweet_orig',tweet_orig) 
  ]
  data_tabular = data.TabularDataset(
    path=NN_DF_FILE, format='csv',
    fields=fields,skip_header = True)
  
  if (with_pretarin_vec):
    trained_embeddings = load_pretrained_embedding()
    tweet_norm.build_vocab(data_tabular, 
                 max_size = MAX_VOCAB_SIZE, 
                 vectors = trained_embeddings)
    print ("example for pretrain for president :",tweet_norm.vocab.vectors[tweet_norm.vocab.stoi['president']])

  else:
    tweet_norm.build_vocab(data_tabular, max_size = MAX_VOCAB_SIZE)

  label.build_vocab(data_tabular)
  tweet_orig.build_vocab(data_tabular)
  print("Lets look on the common word in our nn vocabulary")
  print(tweet_norm.vocab.freqs.most_common(20))
  print(f"Unique tokens in TEXT vocabulary: {len(tweet_norm.vocab)}")
  return data_tabular, tweet_norm

class LSTM(nn.Module):
  def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx, tweet_vocab):
    super().__init__()
    self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
    self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
    self.fc = nn.Linear(hidden_dim * 2, output_dim)                       
    self.dropout = nn.Dropout(dropout) 
    self.tweet_vocab = tweet_vocab  
    
  def forward(self, text, text_lengths):  
    #embedded = self.dropout(self.embedding(text))
    embedded = self.embedding(text)
    packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths)
    packed_output, (hidden, cell) = self.rnn(packed_embedded)
    output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output)
    hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))    
    return self.fc(hidden)
    
  def get_tweet_vocab(self):
    return self.tweet_vocab

  def predict(self,normalized_text):
    tokenized =  TweetTokenizer().tokenize(normalized_text)
    indexed = [self.tweet_vocab.vocab.stoi[t] for t in tokenized]
    length = [len(indexed)]
    tensor = torch.LongTensor(indexed)
    tensor = tensor.unsqueeze(1)
    length_tensor = torch.LongTensor(length)
    prediction = torch.sigmoid(self.forward(tensor, length_tensor))
    if prediction.item() < 0.5 :
        return 1
    else: 
        return 0 


def train_batch_lstm(model, iterator, optimizer, criterion):  
  """
  train_batch_lstm train our lstm on one batch  
        Args:
          model:our model object 
          iterator:Iterator object on the data set 
          optimizer: SGD 
          criterion:loss function
        Returns:
            loss,acc   
  """ 
  epoch_loss = 0
  epoch_acc = 0
  model.train() 
  for batch in iterator:
    optimizer.zero_grad()
    text, text_lengths = batch.tweet
    predictions = model(text, text_lengths).squeeze(1)
    loss = criterion(predictions, batch.label)
    acc = binary_accuracy(predictions, batch.label)
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
    epoch_acc += acc.item()   
  return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate_batch_lstm(model, iterator, criterion):
  """
  evaluate_batch_lstm evlualte the lstm on our validation data  
    Args:
        model:our model object 
          iterator:Iterator object on the data set 
          optimizer: SGD 
          criterion:loss function
    Returns:
        loss,acc  
  """ 
  epoch_loss = 0
  epoch_acc = 0  
  model.eval()  
  with torch.no_grad():
    for batch in iterator:
      text, text_lengths = batch.tweet
      predictions = model(text, text_lengths).squeeze(1)
      loss = criterion(predictions, batch.label)
      acc = binary_accuracy(predictions, batch.label)
      epoch_loss += loss.item()
      epoch_acc += acc.item()
  return epoch_loss / len(iterator), epoch_acc / len(iterator)    

def train_loop_lstm(lstm_model, train_iterator, valid_iterator):
  """
  train_loop_lstm our main rnn loop for the training 
  First we create the loss and optimizer object and than loop and per epoc run train on batch
  After each epoc we compare of the loss is better on validation that we save the best model to file
  After going over all epoc we plot the loss per epoc for train , validation and print best accuracy for validation 
    Args:
        model:our model object 
        train_iterator:Iterator object on the data train set 
        valid_iterator: on the validation data set  
          
     
  """ 
  N_EPOCHS = 20
  train_loss_list =[]
  valid_loss_list =[]
  optimizer = optim.SGD(lstm_model.parameters(), lr=1e-1)
  #optimizer = optim.Adam(lstm_model.parameters(),lr=0.006)
  criterion = nn.BCEWithLogitsLoss()
  best_valid_loss = float('inf')
  for epoch in range(N_EPOCHS):
    start_time = time.time()
    train_loss, train_acc = train_batch_lstm(lstm_model, train_iterator, optimizer, criterion)
    valid_loss, valid_acc = evaluate_batch_lstm(lstm_model, valid_iterator, criterion) 
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    if valid_loss < best_valid_loss:
      best_valid_loss = valid_loss
      torch.save(lstm_model.state_dict(), LSTM_MODEL_FILE)
      best_val_acc = valid_acc
    train_loss_list.append (train_loss)
    valid_loss_list.append (valid_loss) 
    print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
  PlotLR (train_loss_list,valid_loss_list,best_val_acc,'LSTM') 

def build_and_train_lstm(with_pretrain_vec = False):
  data_tabular, tweet_norm = build_lstm_dataset(with_pretrain_vec)
  INPUT_DIM = len(tweet_norm.vocab)
  PAD_IDX = tweet_norm.vocab.stoi[tweet_norm.pad_token]
  train_data, valid_data = data_tabular.split()
  print(vars(train_data.examples[0]))
  # Creating Itertor for on the train and Valid
  train_iterator, valid_iterator = data.BucketIterator.splits(
      (train_data, valid_data), sort_key=lambda x: len(x.tweet),
      batch_size = BATCH_SIZE,
      sort_within_batch = True)
  lstm_model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX, tweet_norm)
  
  if (with_pretrain_vec):
    print ("with pretrain ")
    embeddings = tweet_norm.vocab.vectors
    lstm_model.embedding.weight.data.copy_(embeddings) # initialize our embedding layer to use our vocabulary vectors.
    UNK_IDX = tweet_norm.vocab.stoi[tweet_norm.unk_token]
    lstm_model.embedding.weight.data[UNK_IDX] = torch.zeros(EMBEDDING_DIM)
    lstm_model.embedding.weight.data[PAD_IDX] = torch.zeros(EMBEDDING_DIM)

  print(f'The LSTM model has {count_parameters(lstm_model):,} trainable parameters')
  train_loop_lstm(lstm_model, train_iterator, valid_iterator)
  return lstm_model   

def load_lstm_model():
  # first create the model
  data_tabular, tweet_norm = build_lstm_dataset() 
  INPUT_DIM = len(tweet_norm.vocab)
  PAD_IDX = tweet_norm.vocab.stoi[tweet_norm.pad_token]
  lstm_model = LSTM(INPUT_DIM, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX, tweet_norm)
  # load the paramters from a file containing the paramther for the best model from the training 
  lstm_model.load_state_dict(torch.load(LSTM_MODEL_FILE))
  lstm_model.eval()
  return lstm_model

def load_lstm_and_predict_test_file(test_file):
  predictions = []
  lstm_model = load_lstm_model()
  df_test = load_test_file(test_file)
  for i,tweet in enumerate(df_test['tweet_text']):
    predictions.append(predict_tweet(lstm_model,tweet))
  return predictions

# to build and train lstm model from scretch 
#lstm_model = build_and_train_lstm()

"""# Driver Support"""

## IMPORTANT!!!  files needed for driver support -  (1) xgb-model.pt (2) vectorizer_tfidf.pk  (3) trump_test.tsv


def load_xgb_model():
  # load the model from disk
  loaded_model = pickle.load(open(XGB_MODEL_FILE, 'rb'))
  return loaded_model

def build_and_train_xgb(split_to_train_val = False):
  '''It wasn't clear if this model is supposed to train on all the data or 80/20 split. (fot the driver support)
   In case you want to train the model over 80% of the data you can set split_to_train_val param to True. 
  '''
  df, featurs_names = prepare_df(TRAIN_FILE, 1, True)
  clf = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                  colsample_bynode=1, colsample_bytree=0.5, gamma=0,
                  learning_rate=0.15, max_delta_step=0, max_depth=15,
                  min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
                  nthread=None, objective='binary:logistic', random_state=0,
                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                  silent=None, subsample=1, verbosity=1)
  X = df[featurs_names]
  y = df['label']
  if split_to_train_val: 
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8)
    clf.fit(X_train,y_train)
  else: 
    clf.fit(X,y)
  return clf
  

def load_best_model():
  ''' load best model - xgboost in our case 
  '''
  return load_xgb_model()

def train_best_model():
  ''' training a classifier from scratch (should be the same classifier and parameters returned by load_best_model().
    the final model could be slightly different than the one returned by  load_best_model(), due to randomization issues.
    This function call training on the data file you received.
    assume it is in the current directory. should trigger the preprocessing and the whole pipeline.
  '''
  # It wasn't clear if this model is supposd to train on all the data 
  return build_and_train_xgb()


def predict(m, fn):
  """ Adding tf-idf vectors based on TfidfVectorizer model loded from a file 
      Args:
          m : the trained model. predict expect to get the best model  !!! our model- xgboost
          fn: the full path to a file in the same format as the test set
      Returns:
          list : a list of 0s and 1s, corresponding to the lines in the specified file.
  """
  predictions = []  
  df_test = load_test_file(fn)
  if type(m) == type(xgb.XGBClassifier()):
    df_test, featurs_names = process_df(df_test, 1, True,True)
    predictions = m.predict(df_test[featurs_names])
  return predictions


# to run the driver
# xgb_load= load_best_model()
# print(predict(xgb_load,TEST_FILE))
# xgb_load= train_best_model()
# print(predict(xgb_load,TEST_FILE))


# train_and_save_best_model() - to train and save the model with statistics print.
