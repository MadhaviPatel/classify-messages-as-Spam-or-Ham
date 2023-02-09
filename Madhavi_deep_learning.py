#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers.convolutional import Conv1D
from keras.models import Sequential
import tqdm
import sklearn.metrics
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.metrics import Recall, Precision
from tensorflow.keras.utils import to_categorical


# In[14]:


df= []
dataset = open('spambase.data')
reader = csv.reader(dataset)
next(reader, None)

for r in reader:
    df.append(r)
dataset.close()

X = [x[:-1] for x in df]
y = [x[-1] for x in df] 


# In[15]:


t = Tokenizer()
t.fit_on_texts(X)

X = t.texts_to_sequences(X)

X = np.array(X)
y = np.array(y)

X = pad_sequences(X, maxlen=100)


# In[16]:


SEQUENCE_LENGTH = 100 # the length of all sequences (number of words per sample)
EMBEDDING_SIZE = 100  # Using 100-Dimensional GloVe embedding vectors
TEST_SIZE = 0.25 # ratio of testing set

BATCH_SIZE = 64
EPOCHS = 10 # number of epochs

label2int = {"ham": 0, "spam": 1}
int2label = {0: "ham", 1: "spam"}


# In[17]:


X = np.array(X)
y = np.array(y)
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)


# In[8]:


X.shape


# In[18]:


y = to_categorical(y)


# In[19]:


T_SIZE = 0.25
# split and shuffle
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=T_SIZE, random_state=7)
# print our data shapes
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)


# In[22]:


def get_embedding_vectors(dim=100):
  embedding_index = {}

  with open(f"glove.6B.{dim}d.txt", encoding='utf8') as f:
        for l in tqdm.tqdm(f, "Reading GloVe"):
            values = l.split()
            word = values[0]
            vectors = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = vectors

  w_index = t.word_index

  embedding_matrix = np.zeros((len(w_index)+1, 100))
  for word, i in w_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
      embedding_matrix[i] = embedding_vector
  return embedding_matrix


# In[23]:


embedding_matrix = get_embedding_vectors()
models = Sequential()
models.add(Embedding(4512,
              EMBEDDING_SIZE,
              weights=[embedding_matrix],
              trainable=False,
              input_length=SEQUENCE_LENGTH))

models.add(LSTM(128, recurrent_dropout=0.2))
models.add(Dropout(0.3))
models.add(Dense(2, activation="softmax"))


# In[ ]:


print(models.summary())


# In[ ]:


from tensorflow.keras import optimizers


# In[ ]:


optimum=optimizers.Adam(clipvalue=0.5)
models.compile(loss='categorical_crossentropy', optimizer="rmsprop", metrics=['accuracy'])


# In[ ]:


import time
# initialize our ModelCheckpoint and TensorBoard callbacks
# model checkpoint for saving best weights
model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}.h5", save_best_only=True,
                                    verbose=1)
# for better visualization
tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")
# train the model
models.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[tensorboard, model_checkpoint],
          verbose=1)


# In[ ]:


# get the loss and metrics
r = models.evaluate(X_test, y_test)
# extract those
loss = r[0]
accuracy = r[1]

print(f"[+] Accuracy: {accuracy*100:.2f}%")


# In[ ]:




