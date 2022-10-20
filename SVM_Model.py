from sklearn.svm import SVC
from sklearn import metrics  # Evaluate model
from PreProcessing import loadData as ld
from mlxtend.plotting import plot_learning_curves
import matplotlib.pyplot as plt


def trainSVM(data_train, label_train):
    clf = SVC(gamma='auto', decision_function_shape='ovo')  # initializing the classifier to an svc one
    # with gamma set to auto to use  1 / n_features
    # and decision function shape set to ovo
    clf.fit(data_train, label_train)  # training the classifier using the fit() function
    return clf


def testAccuracy(data_test, label_test, clf):
    return clf.score(data_test, label_test)  # Returning the accuracy score of the test classifier using the score() function


def evaluateModel(data_test, label_test, clf):
    print("The, precision ,recall ,f1-score, support of the SVM")
    print(metrics.classification_report(label_test, clf.predict(data_test)))  # getting the classification report of the labeled data and the prediction of the tested data
    print("The Confusion matrix: ")
    print(metrics.confusion_matrix(label_test, clf.predict(data_test)))  # getting the confusion matrix for the labeled data and the prediction of the tested data


def SVM():
    x_train, x_test, y_train, y_test = ld('post-operative.csv')

    # Train Model
    svm = trainSVM(x_train, y_train)

    print('Accuracy:', float(testAccuracy(x_test, y_test, svm)))  # printing the test accuracy

    # Evaluate Model
    evaluateModel(x_test, y_test, svm)
    # calling for the model evaluation that print out some measured data about the model
    # using the metrics lib executing both the classification _report function, and the confusion matrix
    # of the labeled test data and the prediction of the test data
    # display the learning of training and test

    # plotting the svm in a learning curve
    svm_clf = SVC()
    plot_learning_curves(x_train, y_train, x_test, y_test, svm_clf)
    plt.show()
