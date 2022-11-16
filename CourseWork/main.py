import numpy as np                # creating arrays for linear algebra
import pandas as pd               # used to create dataframes for data processing using the csv file
import seaborn as sns             # used for visualization
import matplotlib.pyplot as plt   # used for plotting data

dataset = pd.read_csv("dataset.csv")
# Converting entire dataset into lowercase
dataset = dataset.apply(lambda x: x.astype(str).str.lower())

# change GENDER Column values into 0 = male and 1 = female
dataset['GENDER'].replace('M', 0,inplace=True)
dataset['GENDER'].replace('F', 1,inplace=True)
dataset.head()

print("Number of columns: "+str(len(dataset.columns)))
print("\nThe list of available column names:")
for index in range(len(dataset.columns)):
    print(" - " + dataset.columns[index])

dataset.isnull().sum()    # checking for Null values

from sklearn.model_selection import cross_val_score, KFold, train_test_split
X = dataset.drop(['LUNG_CANCER'], axis=1)
y = dataset['LUNG_CANCER']

# Getting the independent columns name list
for index in range(len(X.columns)):
    print(" - " + X.columns[index])

from collections import Counter
counts = Counter(y)
label_count_list = list(counts.values())
print(counts)
print("These are the values: ", list(counts.values()))

prediction_classes = ["Yes","No"]
count_of_prediction_classes = [270, 39]
totalRecords = sum(count_of_prediction_classes)

plt.title("Distribution")
plt.ylabel("Number of Records")
plt.xlabel("Cause of Cancer")
plt.bar(prediction_classes,count_of_prediction_classes)
plt.show()

# Getting the percentage of the predicted classes 
percentage_of_no_counts = (count_of_prediction_classes[0]/totalRecords) * 100
percentage_of_yes_counts = (count_of_prediction_classes[1]/totalRecords) * 100
print("Percentage of NO count: %1d \nPercentage of YES count: %2d" %(percentage_of_no_counts, percentage_of_yes_counts))

# Current shape of the Yes and No
yes = dataset[dataset['LUNG_CANCER'] == "YES"]
no = dataset[dataset['LUNG_CANCER'] == "NO"]

print("Shape of Yes", yes.shape, "\nShape of No", no.shape)

# importing the necessary library for Oversampling the data
from imblearn.over_sampling import SMOTE

# Implementing Oversampling for Handling Imbalanced data
oversample = SMOTE()
X_res, y_res = oversample.fit_resample(X, y)

print()
print("This was the shape before Oversampling: ", X.shape)
print("This was the shape after Oversampling: ", X_res.shape) # 95 new records have been added to make all the data balanced

from collections import Counter
counts = Counter(y_res)
label_count_list = list(counts.values())
print(counts)
print("These are the values: ", list(counts.values()))

prediction_classes = ["Yes","No"]
count_of_prediction_classes = list(counts.values())
totalRecords = sum(count_of_prediction_classes)

plt.title("Distribution")
plt.ylabel("Number of records")
plt.xlabel("Cancer predictions")
plt.bar(prediction_classes,count_of_prediction_classes)
plt.show()

sns.pairplot(dataset)

from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from sklearn.model_selection import train_test_split 

# dividing X_res, y_res into train and test data (Performing Train Test Split)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, random_state = 101) 

### Predicting using the Descision Tree Classifier
# training a DescisionTreeClassifier ( Accuracy score: 93%, Sensitivity: 97%, Specifictity: 96%)
# Total = 286
from sklearn.tree import DecisionTreeClassifier 

dtree_model = DecisionTreeClassifier(max_depth = 2).fit(X_train, y_train) 
dtree_predictions = dtree_model.predict(X_test) 

# creating a confusion matrix 
cf_matrix = confusion_matrix(y_test, dtree_predictions) 

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

ac = accuracy_score(y_test, dtree_predictions)
rs = recall_score(y_test, dtree_predictions, average=None)
ps = precision_score(y_test, dtree_predictions, average=None)

print("Accuracy score: " + str(ac*100)) 
print("Recall score: " + str(rs))       
print("Precision score: " + str(ps)) 

TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

sensitivity = TP/(TP+FN)  
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)

### Predicting using Support Vector Machine Classification
# training a linear SVM classifier ( Accuracy score: 97.7%, Sensitivity: 97%, Specifictity: 96%)
# Total = 290.7 ✅ Best Result
from sklearn.svm import SVC 

## training a SVM classifier 
svm_model_linear = SVC(kernel = 'linear', C = 1).fit(X_train, y_train) 

## Prediction using SVM
svm_predictions = svm_model_linear.predict(X_test) 
  
# model accuracy for X_test   
accuracy = svm_model_linear.score(X_test, y_test) 
  
# creating a confusion matrix, calculating accuracy, calculating score, calculating precision
cf_matrix = confusion_matrix(y_test, svm_predictions) 

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

ac = accuracy_score(y_test, svm_predictions)
rs = recall_score(y_test, svm_predictions, average=None)
ps = precision_score(y_test, svm_predictions, average=None)

print("Accuracy score: " + str(ac*100))
print("Recall score: " + str(rs))
print("Precision score: " + str(ps))

TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)

### Predicting using Navie Bayes Classifier
# training a Naive Bayes classifier (Accuracy score: 94.8, Sensitivity: 97%, Specifictity: 95.5%)
# Total = 287.3 (Second best result)
from sklearn.naive_bayes import GaussianNB 
gnb = GaussianNB().fit(X_train, y_train) 
gnb_predictions = gnb.predict(X_test) 
  
# accuracy on X_test 
accuracy = gnb.score(X_test, y_test) 
  
# creating a confusion matrix 
cf_matrix = confusion_matrix(y_test, gnb_predictions) 

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

ac = accuracy_score(y_test, gnb_predictions)
rs = recall_score(y_test, gnb_predictions, average=None)
ps = precision_score(y_test, gnb_predictions, average=None)

print("Accuracy score: " + str(ac*100))
print("Recall score: " + str(rs))
print("Precision score: " + str(ps))

TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)

# Predicting using Neural Network (Accuracy: 86, Sensitivity: 86.5, Specificity: 85.0)
# Traing a neural network ANN - Perceptron
# Total = 258

from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

# creating a confusion matrix
cf_matrix = confusion_matrix(y_test, predictions)

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

ac = accuracy_score(y_test, predictions)
rs = recall_score(y_test, predictions, average=None)
ps = precision_score(y_test, predictions, average=None)

print("Accuracy score: " + str(ac*100))
print("Recall score: " + str(rs))
print("Precision score: " + str(ps))

TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)

# Making use of an hybrid approach of SVM and Naive Bayes Classifier 

# Creating a hybrid model SVM + Naive Bayes
# Total = 287.3 (Second best result)
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

gnb = GaussianNB()
svm = SVC(kernel='linear', probability=True)
eclf = VotingClassifier(estimators=[('gnb', gnb), ('svm', svm)], voting='soft')

eclf.fit(X_train, y_train)
hybrid_predictions = eclf.predict(X_test)

# creating a confusion matrix
cf_matrix = confusion_matrix(y_test, hybrid_predictions)

group_names = ["True Neg","False Pos","False Neg","True Pos"]
group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(cf_matrix, annot=labels, fmt="", cmap='Blues')

ac = accuracy_score(y_test, hybrid_predictions)
rs = recall_score(y_test, hybrid_predictions, average=None)
ps = precision_score(y_test, hybrid_predictions, average=None)

print("Accuracy score: " + str(ac*100))
print("Recall score: " + str(rs))
print("Precision score: " + str(ps))

TP = cf_matrix[1, 1]
TN = cf_matrix[0, 0]
FP = cf_matrix[0, 1]
FN = cf_matrix[1, 0]

sensitivity = TP/(TP+FN)
specificity = TN/(TN+FP)

"Sensitivity: {} | Specifictity: {}".format(sensitivity, specificity)

