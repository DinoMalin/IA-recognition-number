import numpy as np
from dataset import read_scale_dataset


def train_model(model, train_data):
    # Entraînement du modèle
    train = read_scale_dataset()

    X_train = train_data[0]
    y_train = train_data[1]

    model.fit(X_train, y_train, epochs=100, batch_size=1, use_gpu=True)

    print("Modèle entraîné...")

    # save the model
    model.save('model.h5')
