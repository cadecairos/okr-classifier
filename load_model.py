# Tensorflow and Keras
from keras.preprocessing.text import Tokenizer
from keras.models import load_model

# Helper libraries
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import numpy as np
import pickle

model = load_model('objectives.h5')

tokenizer = Tokenizer()
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


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

tokenizer.fit_on_texts(train_objectives)

# Not 100% sure what I'm doing beyond here yet
x_train = tokenizer.texts_to_matrix(train_objectives, mode='tfidf')
x_test = tokenizer.texts_to_matrix(test_objectives, mode='tfidf')

encoder = LabelBinarizer()
encoder.fit(train_labels)
y_train = encoder.transform(train_labels)
y_test = encoder.transform(test_labels)

model.summary()

score = model.evaluate(x_test, y_test, batch_size=10, verbose=1)


print(f'Test Accuracy: {score[1]}')

for i in range(10):
    prediction = model.predict(np.array([x_test[i]]))
    predicted_label = classes[np.argmax(prediction[0])]
    print(f'Objective text: {test_objectives[i]}')
    print(f'Actual label: {classes[test_labels[i]]}')
    print(f'Predicted label: {predicted_label}')
    print(f' ------------------ ')
