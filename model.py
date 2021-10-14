from tensorflow import keras
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout

mnist = keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

x_train = x_train / 255.0
x_test = x_test / 255.0

batch = 256
epochs = 235

model = keras.models.Sequential()

model.add(Conv2D(32, (5,5), activation='relu', input_shape=(28,28,1)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
#adding 2 Convs
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Conv2D(64, kernel_size=(3,3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

hist = model.fit(x_train, y_train, batch_size=batch, epochs=epochs, validation_data=(x_test, y_test))
print('Model Trained')

score = model.evaluate(x_test, y_test)
print('test lost :' + str(score[0]))
print('test accuracy :' + str(score[1]))

model.save('digits_recognizer.h5')
print('Saved the model as digits_recognizer.h5')