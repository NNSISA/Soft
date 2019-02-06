from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.datasets import mnist
from projekat import crop_num

def kreiranje_modela_za_neuronsku_mrezu(shape, n_classes):
    
    # Kreiranje LE-NET5 arhitektura neuronske mreže
    # Sekvencijalni model je linearni set slojeva
    model = Sequential()
    model.add(Conv2D(28, (3, 3), padding='same', activation='relu', input_shape=shape))
    model.add(Conv2D(28, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(56, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(56, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model

if __name__ == '__main__':
    
    # Učitavanje i obrada MNIST podataka
    (X_obuka, y_obuka), (X_test, y_test) = mnist.load_data()
    
    for i in range(len(X_test)):
        odsecena = crop_num(X_test[i])
        X_test[i] = odsecena
    for i in range(len(X_obuka)):
        odsecena = crop_num(X_obuka[i])
        X_obuka[i] = odsecena

    row, col = X_obuka.shape[1:]
    X_obuka = X_obuka.reshape(X_obuka.shape[0], row, col, 1)
    X_test = X_test.reshape(X_test.shape[0], row, col, 1)
    shape = (row, col, 1)

    X_obuka = X_obuka.astype('float32')
    X_test = X_test.astype('float32')
    # Podešavanje podataka da budu u opsegu od 0 do 1
    X_obuka /= 255
    X_test /= 255

    n_classes = 10
    
    tr_lab_kat = to_categorical(y_obuka)
    te_lab_kat = to_categorical(y_test)

    model = kreiranje_modela_za_neuronsku_mrezu(shape, n_classes)
    
    # Kompajliranje modela
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    
    # Treniranje modela
    # Za obuku Keras modela je korišćena funkcija FIT
    hist = model.fit(X_obuka, tr_lab_kat, batch_size=256, epochs=30, verbose=1,
                             validation_data=(X_test, te_lab_kat))
    loss, gain = model.evaluate(X_test, te_lab_kat, verbose=0)
    print(gain)
    model.save_weights('model.h5')