import pandas as pd
from sklearn.model_selection import train_test_split

def get_MNIST(test_size = .3):
    df = pd.read_csv('MNIST01.csv', index_col = 0)
    y = df.pop('label')
    df = df.as_matrix()
    df = df / 255
    y = y.as_matrix().reshape((y.shape[0], 1))
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size = test_size)
    return X_train, y_train, X_test, y_test