import pandas as pd


def read_scale_dataset():
    train_data = pd.read_csv(
        "./archive/mnist_train.csv", header=None)
    test_data = pd.read_csv(
        "./archive/mnist_test.csv", header=None)

    # visualizing class label frequency in the input data
    train_data[0].value_counts().plot.bar(color='cyan')

    X_train = train_data.drop(0, axis=1).values
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_train = X_train.astype('float32')
    X_train = X_train/255.0

    Y_train = train_data[0].values

    X_test = test_data.drop(0, axis=1).values
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    X_test = X_test.astype('float32')
    X_test = X_test/255.0

    Y_test = test_data[0].values
    print("Données d'entraînement chargées...")

    return X_train, Y_train, X_test, Y_test
