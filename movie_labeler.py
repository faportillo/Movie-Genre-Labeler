import tensorflow as tf
import pandas as pd
import numpy as np
import json
import nltk
import re
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
from utils import *
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adam

from models import *


'''
    Preprocess data from MovieSummaries dataset 
'''
# Necessary Files
dataset = "MovieSummaries/plot_summaries.txt"
metadata = "MovieSummaries/movie.metadata.tsv"
glove_embedding = "GloVe_Embeddings/glove.6B.100d.txt"

pd.set_option('display.max_colwidth', 300)

meta = pd.read_csv(metadata, sep = '\t', header = None)
meta.head()
# rename columns
meta.columns = ['movie_id', 1, 'movie_name', 3, 4, 5, 6, 7, 'genre']
plots = []
with open(dataset, 'r', encoding='utf-8') as f:
    reader = csv.reader(f, dialect='excel-tab')
    for row in tqdm(reader):
        plots.append(row)
movie_id = []
plot = []
#extract movie IDs and plot summaries
for i in tqdm(plots):
    movie_id.append(i[0])
    plot.append(i[1])
movies = pd.DataFrame({'movie_id': movie_id, 'plot': plot})
# change datatype of 'movie_id'
meta['movie_id'] = meta['movie_id'].astype(str)
# merge meta with movies
movies = pd.merge(movies, meta[['movie_id', 'movie_name', 'genre']], on = 'movie_id')

json.loads(movies['genre'][0]).values()
# an empty list
genres = []
# extract genres
for i in movies['genre']:
  genres.append(list(json.loads(i).values()))
num_labels = len(set(sum(genres, [])))
# add to 'movies' dataframe
movies['genre_new'] = genres
movies_new = movies[~(movies['genre_new'].str.len() == 0)]
all_genres = sum(genres,[])
len(set(all_genres))
all_genres = nltk.FreqDist(all_genres)
movies_new['clean_plot'] = movies_new['plot'].apply(lambda x: clean_text(x))
movies_new['clean_plot'] = movies_new['clean_plot'].apply(lambda x: remove_stopwords(x))

# Convert text to features
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(movies_new['genre_new'])
# transform target variable
y = multilabel_binarizer.transform(movies_new['genre_new'])

# split dataset into training and validation set
xtrain, xval, ytrain, yval = train_test_split(movies_new['clean_plot'], y, test_size=0.01, random_state=9)
print(xtrain[0])
# Tokenize data
max_words = 100000
max_len = 200
embed_size = 100
tokenizer = Tokenizer(num_words=max_words, lower=True)
tokenizer.fit_on_texts(xtrain)
xtrain = tokenizer.texts_to_sequences(xtrain)
xtrain = pad_sequences(xtrain, maxlen=max_len)

tokenizer.fit_on_texts(xval)
xval = tokenizer.texts_to_sequences(xval)
xval = pad_sequences(xval, maxlen=max_len)
embeddings_index = {}

with open(glove_embedding, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        embed = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embed
word_index = tokenizer.word_index
num_words = min(max_words, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size), dtype='float32')
for word, i in word_index.items():
    if i >= max_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
'''
    Load model, fit to training data and test of validation set.
'''
model = lstm_bidirectional_v2(num_labels, max_len, max_words, embed_size, embedding_matrix)

model.summary()
model.compile(loss=binary_focal_loss(), optimizer=Adam(lr=1e-3), metrics=['accuracy'])
batch_size = 128

checkpoint_path = "training_2_mV2/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

callbacks = [
    tf.keras.callbacks.EarlyStopping(patience=5, monitor='val_loss'),
    tf.keras.callbacks.TensorBoard(log_dir='./logs'),
    cp_callback
]
'''print("\n\n\n\n")
print(ytrain)
print("\n\n\n\n")'''
model.fit(xtrain, ytrain, validation_split=0.15, batch_size=batch_size,
          epochs=100, callbacks=callbacks, verbose=1)
'''
    Do predictions on data
'''
latest_chk = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest_chk)
# Find f1 score for predictions
print(xval.shape)
y_pred = model.predict(xval)
y_pred[y_pred >= 0.5] = 1
y_pred[y_pred < 0.5] = 0
score = f1_score(yval, y_pred, average="micro")
for i in range(15):
    predictions = model.predict(np.expand_dims(xval[i], 0))
    predictions[predictions >= 0.5] = 1
    predictions[predictions < 0.5] = 0
    tags = multilabel_binarizer.inverse_transform(predictions)
    actual_tags = multilabel_binarizer.inverse_transform(np.expand_dims(yval[i], 0))
    print("Predicted genre: ", tags)
    print("Actual genre: ", actual_tags, "\n")
print("\nF1 Score: ", score)


