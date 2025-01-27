#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
diabetes = pd.read_csv('diabetes.csv')
#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
diabetes.head()
print("dimension of diabetes data: {}".format(diabetes.shape))
print(diabetes.groupby('Outcome').size())
#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
sns.countplot(diabetes['Outcome'],label="Count")
diabetes.info()
#From webstie go here to understand more https://thecleverprogrammer.com/2020/07/13/predict-diabetes-with-machine-learning/
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(diabetes.loc[:, diabetes.columns != 'Outcome'],
diabetes['Outcome'], stratify=diabetes['Outcome'], random_state=66)
#Create Perceptron Model
diabetesModel = Perceptron()
#train model with training data
diabetesModel.fit(X_train, y_train)
#use pecptron to make pedictions
modelPredictions = diabetesModel.predict(X_test)
#print(modelPredictions)
#Evaluate accuracy between model predictions and real results
modelAccuracy = accuracy_score(y_test, modelPredictions)
print("Accuracy of Percpetron learning Algorithm: ", modelAccuracy)