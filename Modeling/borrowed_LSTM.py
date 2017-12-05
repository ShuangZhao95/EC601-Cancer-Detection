import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

import gensim

import scikitplot.plotters as skplt

import nltk

from xgboost import XGBClassifier

import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM
from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.optimizers import Adam


## Load training_text and training_variants

df_train_txt = pd.read_csv('../input/training_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"], engine='python')
df_train_var = pd.read_csv('../input/training_variants', engine='python')
df_test_txt = pd.read_csv('../input/test_text', sep='\|\|', header=None, skiprows=1, names=["ID","Text"], engine='python')
df_test_var = pd.read_csv('../input/test_variants', engine='python')
df_train = pd.merge(df_train_var, df_train_txt, how='left', on='ID')
df_test = pd.merge(df_test_var, df_test_txt, how='left', on='ID')
df_train.describe(include='all')
