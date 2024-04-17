# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def data_loading():
    """
    This function loads the training and test data, preprocesses it, removes the NaN values and interpolates the missing 
    data using imputation

    Parameters
    ----------
    Returns
    ----------
    X_train: matrix of floats, training input with features
    y_train: array of floats, training output with labels
    X_test: matrix of floats: dim = (100, ?), test input with features
    """
    # Load training data
    train_df = pd.read_csv("train.csv")
    
    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(2))
    print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(2))

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    train_df= train_df.dropna(subset=['price_CHF'])

    
    # Perform data preprocessing, imputation, and feature encoding
    le = LabelEncoder()
    train_df['season'] = le.fit_transform(train_df['season'])
    test_df['season'] = le.transform(test_df['season'])


    

    print("Training data:")
    print("Shape:", train_df.shape)
    print(train_df.head(5))
    print('\n')

    print("Test data:")
    print(test_df.shape)
    print(test_df.head(5))
    # Extract features and labels


    X_train = train_df.drop(['price_CHF'], axis=1).to_numpy()
    y_train = train_df['price_CHF'].to_numpy()
    X_test = test_df.to_numpy()
    # Standardize the data



    imputer = SimpleImputer(strategy='mean') # Replace missing values with the mean
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

def evaluate_gpr_kernel(X, y, kernel):
  """
  This function evaluates the performance of a GPR model with a specific kernel using KFold cross-validation

  Parameters:
  ----------
  X: matrix of floats, training data features
  y: array of floats, training data target values
  kernel: kernel object (e.g., DotProduct(), RBF(), Matern(), RationalQuadratic())

  Returns:
  -------
  average_mse: float, average mean squared error across all folds
  """
  # Define the number of folds for cross-validation
  kfold = KFold(n_splits=5, shuffle=True, random_state=42)  # Adjust n_splits as needed
  mse_scores = []

  for train_index, test_index in kfold.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Define and fit the GPR model with the specified kernel
    gpr = GaussianProcessRegressor(kernel=kernel)
    gpr.fit(X_train, y_train)

    # Make predictions on the test fold
    y_pred = gpr.predict(X_test)

    # Calculate mean squared error (MSE) for this fold
    mse = mean_squared_error(y_test, y_pred)
    mse_scores.append(mse)

  # Calculate the average mean squared error across all folds
  average_mse = np.mean(mse_scores)
  return average_mse

def modeling_and_prediction(X_train, y_train, X_test):
    """
    This function defines the model, fits training data and then does the prediction with the test data 

    Parameters
    ----------
    X_train: matrix of floats, training input with 10 features
    y_train: array of floats, training output
    X_test: matrix of floats: dim = (100, ?), test input with 10 features

    Returns
    ----------
    y_test: array of floats: dim = (100,), predictions on test set
    """

    y_pred=np.zeros(X_test.shape[0])
    #TODO: Define the model and fit it using training data. Then, use test data to make predictions
    kernels = [DotProduct(), RBF(), Matern(), RationalQuadratic(), Matern(nu=0.6),Matern(nu=0.7), Matern(nu=0.8)]

    # Evaluate each kernel using cross-validation
    best_kernel = None
    best_mse = np.inf

    for kernel in kernels:
        mse = evaluate_gpr_kernel(X_train, y_train, kernel)
        print(f"Kernel: {type(kernel).__name__}, Average MSE: {mse:.4f}")
        if mse < best_mse:
            best_kernel = kernel
            best_mse = mse

    print("\nBest Kernel based on average MSE:", type(best_kernel).__name__)

    model = GaussianProcessRegressor(kernel=best_kernel)
    model.fit(X_train, y_train)
    
    # Use the trained model to make predictions on test data
    y_pred = model.predict(X_test)

    assert y_pred.shape == (100,), "Invalid data shape"
    return y_pred

# Main function. You don't have to change this
if __name__ == "__main__":
    # Data loading
    X_train, y_train, X_test = data_loading()
    # The function retrieving optimal LR parameters
    y_pred=modeling_and_prediction(X_train, y_train, X_test)
    # Save results in the required format
    dt = pd.DataFrame(y_pred)
    dt.columns = ['price_CHF']
    dt.to_csv('results.csv', index=False)
    print("\nResults file successfully generated!")

