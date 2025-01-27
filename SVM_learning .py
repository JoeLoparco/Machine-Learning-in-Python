#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
#Some code also found @https://scikit-learn.org/stable/modules/svm.html
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

diabetes = pd.read_csv('diabetes.csv')
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
print(diabetes.groupby('Outcome').size())
sns.countplot(diabetes['Outcome'],label="Count")
diabetes.info()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'],
diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)

#linear kernel
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
linear_predictions = svm_linear.predict(X_test)
linear_accuracy = accuracy_score(y_test, linear_predictions)
print("Linear Kernel Accuracy:", linear_accuracy)

#polynomial kernel
svm_poly = SVC(kernel='poly', degree=3)  # You can adjust the degree parameter
svm_poly.fit(X_train, y_train)
poly_predictions = svm_poly.predict(X_test)
poly_accuracy = accuracy_score(y_test, poly_predictions)
print("Polynomial Kernel Accuracy:", poly_accuracy)
#
#rbf kernel
svm_rbf = SVC(kernel='rbf')
svm_rbf.fit(X_train, y_train)
rbf_predictions = svm_rbf.predict(X_test)
rbf_accuracy = accuracy_score(y_test, rbf_predictions)
print("RBF Kernel Accuracy:", rbf_accuracy)