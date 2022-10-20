from sklearn.preprocessing import OneHotEncoder, LabelEncoder  # Transform 'string' into class number
from sklearn.model_selection import train_test_split  # Split training and testing data
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import pandas as pd  # importing the pandas lib

# displaying the full data
pd.set_option("display.max_rows", None, "display.max_columns", None)


# labeling the data to change the data from categorical to numerical
def loadData(path):
    # loading the dataset
    dataset = pd.read_csv(path)

    # Deal with missing value using the "fillna" function (Filling Nan function)
    dataset = dataset.fillna(round(dataset.mean()))

    le = LabelEncoder()  # calling the label encoder function
    one_hot_encoder = OneHotEncoder(sparse=False)  # calling the on hot encoder and setting the sparse to false

    x_onehot_encoded = one_hot_encoder.fit_transform(dataset.iloc[:, :-1])
    x = pd.DataFrame(x_onehot_encoded)
    # encoding the data to binary to be easier to classify

    y = le.fit_transform(dataset["ADM_DECS"])
    y = y.reshape(len(y), 1)
    # transforming an training on the class after converting it to binary for easier implementation

    # over sampling the data because the dataset have small number of data
    strategy = {0: 1200, 2: 1200}
    oversample = SMOTE(sampling_strategy=strategy)
    x, y = oversample.fit_resample(x, y)
    # training on the data after oversampling

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # splitting the arrays to testing an training subset
    # test size is a float set to test a portion of the dataset, it is set o use only 0.3 of the data in the dataset
    # Random Stat is an int that controls the applied to the data before splitting it

    sc = StandardScaler()  # calling the standard Scaler function to Standardize the data by removing the mean and scaling to unit variance
    x_train = sc.fit_transform(x_train)
    x_test = sc.fit_transform(x_test)
    # training on the data after standardization
    return x_train, x_test, y_train, y_test
