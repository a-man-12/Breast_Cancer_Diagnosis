# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
#from sklearn.model_selection import cross_val_score

# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

#categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

#Exploring data
plt.hist(dataset['diagnosis'].map({'M':1,'B':0}))
plt.title('Diagnosis (B=0, M=1)')
plt.show()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

# Applying LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components =None)
X_train = lda.fit_transform(X_train, y_train)
X_test = lda.transform(X_test)

#logistic regression
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
classifier.fit(X_train,y_train)
y_pred=classifier.predict(X_test)


accuracy_lr=metrics.accuracy_score(y_test,y_pred)
print("Logistic Regression:",accuracy_lr)

#Making the Confusion Matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)

#knn classifier(using eucledian distance)
from sklearn.neighbors import KNeighborsClassifier
classifier_knn=KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2)
classifier_knn.fit(X_train,y_train)
y_pred_knn=classifier_knn.predict(X_test)


accuracy_knn=metrics.accuracy_score(y_test,y_pred_knn)
print("KNN Classifier:",accuracy_knn)


cm=confusion_matrix(y_test,y_pred_knn)
print(cm)

# Fitting Decision Tree Classification to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


accuracy_dt=metrics.accuracy_score(y_test,y_pred)
print("Decision Tree:",accuracy_dt)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


accuracy_svm=metrics.accuracy_score(y_test,y_pred)
print("SVM:",accuracy_svm)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


# Fitting Kernel SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


accuracy_ksvm=metrics.accuracy_score(y_test,y_pred)
print("Kernel SVM:",accuracy_ksvm)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


accuracy_nb=metrics.accuracy_score(y_test,y_pred)
print("Naive Bayes:",accuracy_nb)

# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Fitting Random Forest Classification to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


accuracy_rf=metrics.accuracy_score(y_test,y_pred)
print("Random Forest",accuracy_rf)


# Making the Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

acc=[accuracy_lr,accuracy_knn,accuracy_dt,accuracy_nb,accuracy_svm,accuracy_ksvm,accuracy_rf]
plt.hist(acc)
plt.show()

ab=[0,1,2,3,4,5,6]
plt.scatter(ab, acc, color = 'red')
plt.title('Accuracy plot')
plt.xlabel('Classifiers: 0:lr,1:knn,2:dt,3:nb,4:svm,5:ksvm,6:rf')
plt.ylabel('Accuracy')
plt.show()