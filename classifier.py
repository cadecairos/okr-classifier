# Tensorflow and Keras
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout

# Helper libraries
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import pandas as pd
import csv


# read the dataset in to data frames
objective_df = pd.read_csv('./data/objectives.csv', header=0)

# 80% of data for training, 20% for testing
train_size = int(len(objective_df) * .8)
train_objectives = objective_df['Objective'].values[:train_size]
train_labels = objective_df['Label'].values[:train_size]

test_objectives = objective_df['Objective'].values[train_size:]
test_labels = objective_df['Label'].values[train_size:]

# 0=good, 1=bad
classes = ['good', 'bad']
vocab_size = 500
num_labels = 2

# set a maximum number of words to tokenize
tokenizer = Tokenizer(num_words=500)
tokenizer.fit_on_texts(train_objectives)

# Not 100% sure what I'm doing beyond here yet
x_train = tokenizer.texts_to_matrix(train_objectives, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_objectives, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)

model = Sequential()
model.add(Dense(56, input_shape=(vocab_size,)))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(56))
model.add(Activation('relu'))
model.add(Dropout(0.3))
model.add(Dense(num_labels))
model.add(Activation('softmax'))

# output summary of the Model's layers
model.summary()

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=10,
                    epochs=8,
                    verbose=1,
                    validation_split=0.1)

score = model.evaluate(x_test, y_test, batch_size=10, verbose=1)

print(f'Test Accuracy: {score[1]}')

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = classes[np.argmax(prediction[0])]
    print(f'Objective text: {test_objectives[i]}')
    print(f'Actual label: {classes[test_labels[i]]}')
    print(f'Predicted label: {predicted_label}')
    print(f' ------------------ ')
