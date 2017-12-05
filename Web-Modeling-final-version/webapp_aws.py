# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 17:06:57 2017

@author: lenovo
"""

import os
import re
import string
import pandas as pd
import numpy as np
import keras
from flask import Flask, request, url_for, send_from_directory
from werkzeug import secure_filename
from sklearn.utils import shuffle
from nltk.corpus import stopwords
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Embedding, Input, RepeatVector
from keras.optimizers import SGD
from gensim.models.doc2vec import LabeledSentence
from gensim import utils
from gensim.models import Doc2Vec


encoded_y = np.load('./encoded_y.npy')
train_set= np.load('./train_set.npy')


def baseline_model():
    model = Sequential()
    model.add(Dense(256, input_dim=300, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(180, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(90, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation="softmax"))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return model
model = baseline_model()


estimator=model.fit(train_set, encoded_y, validation_split=0.2, epochs=20, batch_size=64)


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


html = '''
    <!DOCTYPE html>
    <title>Upload File</title>
    <h1>Upload File</h1>
    <form method=post enctype=multipart/form-data>
         <input type=file name=file value=Choose file>
         <input type=submit value=Upload>
    </form>
    '''

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file_url = url_for('uploaded_file', filename=filename)

        return html + '<iframe src=' + file_url + ' name=iframe1></iframe>'+ '<br><br>'+ calculate(filename)
    return html


@app.route('/result/<filename>')


def calculate(filename):
    test_text = pd.read_csv(app.config['UPLOAD_FOLDER']+"/"+filename, sep="\|\|", engine='python', header=None, skiprows=1, names=["ID","Text"])
    test_size = 1

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

    allText9 = test_text['Text'].apply(cleanup)
    sentences9 = constructLabeledSentences(allText9)


    from gensim.models import Doc2Vec

    Text_INPUT_DIM=300


    text_model=None
    filename='docEmbeddings_9_clean.d2v'
    text_model = Doc2Vec(min_count=1, window=5, size=Text_INPUT_DIM, sample=1e-4, negative=5, workers=4, iter=5,seed=1)
    text_model.build_vocab(sentences9)
    text_model.train(sentences9, total_examples=text_model.corpus_count, epochs=text_model.iter)
    text_model.save(filename)
    text_test_arrays9 = np.zeros((test_size, Text_INPUT_DIM))
    j=0
    for i in range(0,0+test_size):
        text_test_arrays9[j] = text_model.docvecs['Text_'+str(i)]
        j=j+1
    test_set9=text_test_arrays9
    y_pred = model.predict_proba(test_set9)
    o = str(y_pred)
    return o
	
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)
