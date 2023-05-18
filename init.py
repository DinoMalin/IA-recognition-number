import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Construction du modèle


def build_model():
    model = Sequential()
    model.add(Dense(units=32, input_dim=3,
              input_shape=(28, 28, 1), activation='relu'))
    model.add(Dense(units=80, activation='relu'))
    model.add(Dense(units=1, activation='softmax'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print("Modèle construit...")
    return model


# Compilation du modèle
