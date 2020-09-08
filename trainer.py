import tensorflow as tf
import numpy as np

x = np.load('x.npy')
y = np.load('y.npy')

y = tf.keras.utils.to_categorical(y, 2)
y = y.astype(int)
x = x[..., tf.newaxis]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

# for i in y_test:
#     print(i)

from tensorflow.keras.layers import Conv1D, MaxPool1D, Dropout, Dense, Flatten

model = tf.keras.models.Sequential()


model.add(Conv1D(32, 3, input_shape = (40, 1), activation = 'relu'))
model.add(Conv1D(64, 3, activation = 'relu'))
model.add(MaxPool1D(1, 1))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(2, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
model.summary()

score = model.evaluate(x_test, y_test, verbose = 0)

accuracy = 100 * score[1]

print(f'Pre-training accuracy:{accuracy}')

import time
start_time = time.time()
model.fit(x_train, y_train, epochs = 15, batch_size = 10, verbose = 1)

end_time = time.time()

second = int(end_time - start_time)
hour = int(second / 3600)
second %= 3600
minute = int(second / 60)
second %= 60

print(f'Total training time: {hour:02d}:{minute:02d}:{second}')

score = model.evaluate(x_test, y_test, verbose = 0)
print(f'Test loss: {score[0]}')
print(f'Accuracy: {(100 * score[1]):2f}%')

model_save_name = 'model_offline.h5'
path = f"{model_save_name}"
model.save(path)