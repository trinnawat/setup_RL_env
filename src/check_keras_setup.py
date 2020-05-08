from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop
from keras.datasets import mnist
from keras.utils import to_categorical


batch_size = 128
num_classes = 10  # 0 to 9
epochs = 10

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(256, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(), metrics=['accuracy'])
# Train...
history = model.fit(x_train, y_train, batch_size=batch_size,
                    epochs=epochs, validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test)
print('Loss:', score[0], '| Accuracy:', score[1])
