import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# first, we import necessary libraries:

from sklearn.gaussian_process.kernels import (
  DotProduct,
  RBF,
  Matern,
  RationalQuadratic,
)


def data_loading():
    """
    this function loads the training and test data, preprocesses it, 
    removes the NaN values and interpolates the missing
    data using imputation

    parameters
    ----------
    returns
    ----------
    x_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    x_test: matrix of floats: dim = (100, ?), test input with features
    """
    # load training data
    train_df = pd.read_csv("train.csv")
    print("training data:")
    print("shape:", train_df.shape)
    print(train_df.head(2))
    print("\n")
    # load test data
    test_df = pd.read_csv("test.csv")

    print("test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # dummy initialization of the x_train, x_test and y_train
    # modify/ignore the initialization of these variables
    train_df = train_df.dropna(subset=["price_CHF"])

    # perform data preprocessing, imputation, and feature encoding
    le = LabelEncoder()
    train_df["season"] = le.fit_transform(train_df["season"])
    test_df["season"] = le.transform(test_df["season"])

    print("training data:")
    print("shape:", train_df.shape)
    print(train_df.head(5))
    print("\n")

    print("test data:")
    print(test_df.shape)
    print(test_df.head(5))
    # extract features and labels

    x_train = train_df.drop(["price_CHF"], axis=1).to_numpy()
    y_train = train_df["price_CHF"].to_numpy()
    x_test = test_df.to_numpy()
    # standardize the data

    # replace missing values with the mean
    imputer = SimpleImputer(strategy="mean") 
    x_train = imputer.fit_transform(x_train)
    x_test = imputer.transform(x_test)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    assert (
      x_train.shape[1] == x_test.shape[1]
    ) and (
      x_train.shape[0] == y_train.shape[0]
    ) and (
      x_test.shape[0] == 100
    ), "invalid data shape"
    return x_train, y_train, x_test


def evaluate_gpr_kernel(x, y, kernel):
    """
    this function evaluates the performance of a gpr model with a specific kernel using kfold cross-validation

    parameters:
    ----------
    x: matrix of floats, training data features
    y: array of floats, training data target values
    kernel: kernel object (e.g., dotproduct(), rbf(), matern(), rationalquadratic())

    returns:
    -------
    average_mse: float, average mean squared error across all folds
    """
    # define the number of folds for cross-validation
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # adjust n_splits as needed
    mse_scores = []

    for train_index, test_index in kfold.split(x):
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # define and fit the gpr model with the specified kernel
        gpr = GaussianProcessRegressor(kernel=kernel)
        gpr.fit(x_train, y_train)

        # make predictions on the test fold
        y_pred = gpr.predict(x_test)

        # calculate mean squared error (mse) for this fold
        mse = mean_squared_error(y_test, y_pred)
        mse_scores.append(mse)

    # calculate the average mean squared error across all folds
    average_mse = np.mean(mse_scores)
    return average_mse

def modeling_and_prediction(x_train, y_train, x_test):
    """
    this function defines the model, fits training data and then does the prediction with the test data 

    parameters
    ----------
    x_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    x_test: matrix of floats: dim = (100, ?), test input with 10 features

    returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(x_test.shape[0])
    kernels = [DotProduct(), RBF(), Matern(nu=0.7), RationalQuadratic()]

    # evaluate each kernel using cross-validation
    best_kernel = None
    best_mse = np.inf

    for kernel in kernels:
        mse = evaluate_gpr_kernel(x_train, y_train, kernel)
        print(f"kernel: {type(kernel).__name__}, average mse: {mse:.4f}")
        if mse < best_mse:
            best_kernel = kernel
            best_mse = mse

    print("\nbest kernel based on average mse:", type(best_kernel).__name__)

    model = GaussianProcessRegressor(kernel=best_kernel)
    model.fit(x_train, y_train) 
    # use the trained model to make predictions on test data
    y_pred = model.predict(x_test)

    assert y_pred.shape == (100,), "invalid data shape"
    return y_pred

# main function. you don't have to change this
if __name__ == "__main__":
    # data loading
    x_train_o, y_train_o, x_test_o = data_loading()
    # the function retrieving optimal lr parameters
    y_pred_o=modeling_and_prediction(x_train_o, y_train_o, x_test_o)
    # save results in the required format
    dt = pd.DataFrame(y_pred_o)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")
