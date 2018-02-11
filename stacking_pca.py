# Importing the libraries
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from mlxtend.classifier import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


# Importing the dataset
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, 2:32].values
y = dataset.iloc[:, 1].values

#categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train= sc.fit_transform(X_train)
X_test= sc.fit_transform(X_test)

# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components =None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)



bclf=LogisticRegression()

#random forest,knn,svm
clfs=[RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      SVC(kernel = 'linear', random_state = 0)]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy0=metrics.accuracy_score(y_test,y_pred)
print("Random Forest, KNN, SVM:",accuracy0)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#knn, decision tree, svm
clfs=[DecisionTreeClassifier(criterion = 'entropy'),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      SVC(kernel = 'linear', random_state = 0)]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy1=metrics.accuracy_score(y_test,y_pred)
print("KNN, Decision Tree, SVM:",accuracy1)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#knn, svm, naive bayes
clfs=[GaussianNB(),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      SVC(kernel = 'linear', random_state = 0)]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy2=metrics.accuracy_score(y_test,y_pred)
print("KNN, SVM, Naive Bayes:",accuracy2)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#knn,decision tree, kernel svm
clfs=[DecisionTreeClassifier(criterion = 'entropy'),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      SVC(kernel = 'rbf', random_state = 0)]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy3=metrics.accuracy_score(y_test,y_pred)
print("KNN, Decision Tree, Kernel SVM:",accuracy3)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#knn,decision tree, naive bayes
clfs=[GaussianNB(),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      DecisionTreeClassifier(criterion = 'entropy')]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy4=metrics.accuracy_score(y_test,y_pred)
print("KNN, Decision Tree, Naive Bayes:",accuracy4)

cm = confusion_matrix(y_test, y_pred)
print(cm)

#knn, decision tree, random forest
clfs=[RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0),
      KNeighborsClassifier(n_neighbors=5,metric='minkowski', p=2),
      DecisionTreeClassifier(criterion = 'entropy')]
sl=StackingClassifier(classifiers=clfs,meta_classifier=bclf)
sl.fit(X_train,y_train)

y_pred=sl.predict(X_test)


accuracy5=metrics.accuracy_score(y_test,y_pred)
print("KNN, Decision Tree, Random Forest:",accuracy5)

cm = confusion_matrix(y_test, y_pred)
print(cm)

# plot
acc1=[accuracy0,accuracy1,accuracy2,accuracy3,accuracy4,accuracy5]
ab=[0,1,2,3,4,5]
plt.scatter(ab, acc1, color = 'red')
plt.title('Accuracy plot')
plt.xlabel('Classifiers')
plt.ylabel('Accuracy')
plt.show()