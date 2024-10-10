# In this file, four machine learning models are employed for Liris data classification.
#1) LogisticRegression
#2) KNN
#3) DT
#4) RF
###########################################################################################
# LogisticRegression Code (LR Code)
###########################################################################################

#step 0 

from sklearn.datasets import load_iris
iris = load_iris()

# step 1

x = iris.data
y = iris.target
print(iris.feature_names)
print(iris.target)
# print(iris.DESCR)
# print(iris.data)
# print(iris.data_module)
# print(iris.filename)
# print(iris.frame)
#print(iris.target_names)

# step 2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, shuffle = True, random_state = 42)

# step 3

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

# step 4

model.fit(x_train, y_train)

# step 5 

from sklearn.metrics import accuracy_score
y_pred=model.predict(x_train)
train_score=accuracy_score(y_train,y_pred)
print(train_score) 


# train_score_logisticregression = 0.9642857142857143  

y_pred=model.predict(x_test)
test_score=accuracy_score(y_test,y_pred)
print(test_score) 

######################################################################################################################
#  k-nearest neighbors (kNN) 
######################################################################################################################
# step 0

from sklearn.datasets import load_iris
iris = load_iris()

# step 1

x= iris.data
y=iris.target

# step 2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, shuffle = True, random_state = 42)

# step 3

from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors = 15)

# step 4

model.fit(x_train, y_train)

# step 5

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_train)
train_score = accuracy_score(y_train, y_pred)
print('train_score :',train_score)

y_pred = model.predict(x_test)
test_score = accuracy_score(y_test, y_pred)
print('test_score :' , test_score)
################################################################################################################
# Decision Tree
################################################################################################################
# step 0 
from sklearn.datasets import load_iris
iris = load_iris()

# step 1
x = iris.data
y = iris.target

# step 2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, shuffle = True, random_state = 42)

# step 3

from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier(max_depth=1,random_state=42)
model=DecisionTreeClassifier(max_depth=10,random_state=42)

# step 4

model.fit(x_train,y_train)

# step 5

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_train)
train_score = accuracy_score(y_train, y_pred)
print('train_score :',train_score)

y_pred = model.predict(x_test)
test_score = accuracy_score(y_pred, y_test)
print('train_score :',test_score)
#######################################################################################################################
# Random Forest
#######################################################################################################################
# step 0
from sklearn.datasets import load_iris
iris = load_iris()

# step 1
x = iris.data
y = iris.target

# step 2

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.25, shuffle = True, random_state = 42)

# step 3

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(random_state=42,n_estimators=3)

# step 4

model.fit(x_train,y_train)

# step 5

from sklearn.metrics import accuracy_score

y_pred = model.predict(x_train)
train_score = accuracy_score(y_train, y_pred)
print('train_score :',train_score)
##############################################################################################################################
# Conclusion
#############################################################################################################################
# Logisticregression Score
# train_score_logisticregression = 0.9642857142857143 
# test_score_logisticregression = 1
#-------------------------------------------------------------------------------------------#
# KNN Score
# Three KNN models were trained based on three different n_neighbors hyperparameters.
# n_neighbors = 2  ------->  train_score : 0.9732142857142857 ,  test_score : 1.0
# n_neighbors = 6  ------->  train_score : 0.9642857142857143 ,  test_score : 1.0
# n_neighbors = 10 ------->  train_score : 0.9642857142857143 ,  test_score : 1.0
# By increasing the n_neighbors, the train score is decreased.
#-------------------------------------------------------------------------------------------#
# DT Score
# Three DT models were trained based on three different max_depth hyperparameters.
# max_depth = 1  ------->  train_score : 0.6607142857142857 ,  test_score : 0.6842105263157895
# max_depth = 6  ------->  train_score : 0.9732142857142857 ,  test_score : 1.0
# max_depth = 10 ------->  train_score : 1 ,  test_score : 1.0
# By increasing the max_depth, the train score is increased.
#-------------------------------------------------------------------------------------------#
# FR Score
# Three FR models were trained based on three different n_estimators hyperparameters.
# n_estimators = 4  ------->  train_score : 0.9821428571428571 ,  test_score : 1
# n_estimators = 7  ------->  train_score : 1 ,  test_score : 1.0
# n_estimators = 10 ------->  train_score : 1 ,  test_score : 1.0
# By increasing the max_depth, the train score is increased.
#-------------------------------------------------------------------------------------------#  







