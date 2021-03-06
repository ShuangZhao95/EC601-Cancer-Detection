# coding: utf-8

print("begin imports")
print("numpy")
import numpy as np
print("os")
import os
print("re")
import re
print("tqdm")
import tqdm
print("string")
import string
print("pandas")
import pandas as pd
print("keras")
import keras
print("keras backend")
from keras import backend as K
print("keras np_utilz")
from keras.utils import np_utils
print("keras Sequential")
from keras.models import Sequential
print("keras layers")
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
print("keras SGD")
from keras.optimizers import SGD
print("tables")
import tables
print("sklearn shuffle")
from sklearn.utils import shuffle
print("sklearn TruncatedSVD")
from sklearn.decomposition import TruncatedSVD
print("sklearn LabelEncoder")
from sklearn.preprocessing import LabelEncoder
print("sklearn train_test_split")
from sklearn.model_selection import train_test_split
print("gensim Doc2Vec")
from gensim.models import Doc2Vec
print("gensim LabeledSentence")
from gensim.models.doc2vec import LabeledSentence
print("gensim utils")
from gensim import utils
print("nltk stopwords")
from nltk.corpus import stopwords
print("matplotlib")
import matplotlib
print("matplotlib set backend")
matplotlib.use('TkAgg')
print("matplotlib pyplot")
import matplotlib.pyplot as plt
print("seaborn")
import seaborn as sns
print("done with imports")



""" Load in setup files, throw a flag if they don't exist """

print("begin loading setup files")

setup_avail = True
filename='D2VModel.d2v'
text_model=None
if os.path.isfile(filename):
    text_model = Doc2Vec.load(filename)
else:
	setup_avail = False
	print("no setup files")

store = pd.HDFStore('store.h5')
try:
	train = store['shuffled']
except:
	setup_avail = False
else: 
	store.close()

if not setup_avail:
	train_variant = pd.read_csv("../input/training_variants")
	train_text = pd.read_csv("../input/training_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
	train = pd.merge(train_variant, train_text, how='left', on='ID')
	train = shuffle(train)
	store['shuffled'] = train
	store.close()

print("done loading setup files")

""" Load Data """

test_variant = pd.read_csv("../input/test_variants")
test_text = pd.read_csv("../input/test_text", sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
train_y = train['Class'].values
train_x = train.drop('Class', axis=1)
train_size=len(train_x)
test_x = pd.merge(test_variant, test_text, how='left', on='ID')
test_size=len(test_x)
test_index = test_x['ID'].values
all_data = np.concatenate((train_x, test_x), axis=0)
all_data = pd.DataFrame(all_data)
all_data.columns = ["ID", "Gene", "Variation", "Text"]

def constructLabeledSentences(data):
    sentences=[]
    for index, row in data.iteritems():
        sentences.append(LabeledSentence(utils.to_unicode(row).split(), ['Text' + '_%s' % str(index)]))
    return sentences

def textClean(text):
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = text.lower().split()
    stops = set(stopwords.words("english"))
    text = [w for w in text if not w in stops]    
    text = " ".join(text)
    return(text)
    
def cleanup(text):
    text = textClean(text)
    text= text.translate(str.maketrans("","", string.punctuation))
    return text

allText = all_data['Text'].apply(cleanup)
sentences = constructLabeledSentences(allText)


print("begin Doc2Vec Model")

Text_INPUT_DIM=300

if not setup_avail:
    text_model = Doc2Vec(min_count=1, window=5, size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=7, iter=10,seed=1)
    text_model.build_vocab(sentences)
    text_model.train(sentences, total_examples=text_model.corpus_count, epochs=text_model.iter)
    text_model.save(filename)

text_train_arrays = np.zeros((train_size, Text_INPUT_DIM))
text_test_arrays = np.zeros((test_size, Text_INPUT_DIM))

for i in range(train_size):
    text_train_arrays[i] = text_model.docvecs['Text_'+str(i)]

j=0
for i in range(train_size,train_size+test_size):
    text_test_arrays[j] = text_model.docvecs['Text_'+str(i)]
    j=j+1


print("end Doc2Vec Model")


label_encoder = LabelEncoder()
label_encoder.fit(train_y)
encoded_y = np_utils.to_categorical((label_encoder.transform(train_y)))

train_set = text_train_arrays
test_set = text_test_arrays

def baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=Text_INPUT_DIM, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu', kernel_initializer='normal'))
    model.add(Dropout(0.5))
    model.add(Dense(80, activation='relu', kernel_initializer='normal'))
    model.add(Dense(9, activation="softmax", kernel_initializer='normal'))
    
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)  
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model

model = baseline_model()
model.summary()

x_train, x_test, y_train, y_test = train_test_split(train_set, encoded_y, test_size=0.2, random_state=42)
estimator=model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs=20, batch_size=64)

print("Training accuracy: %.2f%% / Validation accuracy: %.2f%%" % (100*estimator.history['acc'][-1], 100*estimator.history['val_acc'][-1]))



import scikitplot.plotters as skplt

#plot confusion matrix
probas = model.predict_proba(x_test)
pred_indices = np.argmax(probas, axis=1)
classes = np.array(range(1,10))
preds = classes[pred_indices]
skplt.plot_confusion_matrix(classes[np.argmax(y_test,axis=1)],preds)
plt.show()

#plot loss over epochs
plt.plot(estimator.history['acc'])
plt.plot(estimator.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'valid'], loc='upper left')
plt.show()


# Make Submission File
y_pred = model.predict_proba(test_set)

submission = pd.DataFrame(y_pred)
submission['id'] = test_index
submission.columns = ['class1', 'class2', 'class3', 'class4', 'class5', 'class6', 'class7', 'class8', 'class9', 'id']
submission.to_csv("submission_all.csv",index=False)
submission.head()

print("reached end")
