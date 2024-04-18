# This serves as a template which will guide you through the implementation of this task.  It is advised
# to first read the whole template and get a sense of the overall structure of the code before trying to fill in any of the TODO gaps
# First, we import necessary libraries:
import numpy as np
import pandas as pd
#from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, Matern, RationalQuadratic
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
#from sklearn.metrics import mean_squared_error
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
    
    # print("Training data:")
    # print("Shape:", train_df.shape)
    # print(train_df.head(2))
    # print('\n')
    
    # Load test data
    test_df = pd.read_csv("test.csv")

    # print("Test data:")
    # print(test_df.shape)
    # print(test_df.head(2))

    #drop columns, where price_CHF is naN
    #working
    train_df = train_df.dropna(subset=['price_CHF'])
    # print('\n'+"drop nan")
    # print("Shape:", train_df.shape)
    # print(train_df.head())

    # Dummy initialization of the X_train, X_test and y_train
    # TODO: Depending on how you deal with the non-numeric data, you may want to 
    # modify/ignore the initialization of these variables   
    X_train = np.zeros_like(train_df.drop(['price_CHF'],axis=1))
    y_train = train_df['price_CHF'].to_numpy()
    X_test = np.zeros_like(test_df)

    #encode seasons
    # import labelencoder
    # instantiate labelencoder object
    le = LabelEncoder()
    le.fit(["winter", "spring", "autumn", "summer"]) #ordered by average energy consumption
    train_df["season"] = le.fit_transform(train_df["season"])
    test_df["season"] = le.transform(test_df["season"])

    print('\n'+"seasons")
    print("Shape:", train_df.shape)
    print(train_df.head())

    # TODO: Perform data preprocessing, imputation and extract X_train, y_train and X_test
    #imputation
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp.fit(train_df.drop(['price_CHF'],axis=1))
    X_train = imp.transform(train_df.drop(['price_CHF'],axis=1))
    #y_train = imp.transform(train_df['price_CHF'])
    X_test = imp.transform(test_df)
    
    print('\n'+"imputed")
    print("Shape:", X_train.shape)
    print(X_train)

    assert (X_train.shape[1] == X_test.shape[1]) and (X_train.shape[0] == y_train.shape[0]) and (X_test.shape[0] == 100), "Invalid data shape"
    return X_train, y_train, X_test

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
        mse = gpr.score(x_test, y_test)
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
    best_mse = 0

    for kernel in kernels:
        mse = evaluate_gpr_kernel(x_train, y_train, kernel)
        print(f"kernel: {type(kernel).__name__}, average mse: {mse:.4f}")
        if mse > best_mse:
            best_kernel = kernel
            best_mse = mse

    print("\nbest kernel based on average mse:", type(best_kernel).__name__)

    model = GaussianProcessRegressor(kernel=best_kernel)
    model.fit(x_train, y_train) 
    # use the trained model to make predictions on test data
    y_pred = model.predict(x_test)

    assert y_pred.shape == (100,), "invalid data shape"
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

