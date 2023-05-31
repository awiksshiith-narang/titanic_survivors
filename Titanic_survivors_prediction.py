import pandas as p
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import seaborn as sb

#Gathering data:
training_data = p.read_csv( "C:/Users/AWIKSSHIITH/OneDrive/Desktop/training_data.csv" )
testing_data = p.read_csv( "C:/Users/AWIKSSHIITH/OneDrive/Desktop/testing data.csv" )
training_data.dropna( inplace = True ) #Removes the row which don't have information
testing_data.dropna( inplace = True ) #Removes the row which don't have information
y = training_data[ 'Survived' ].values
x = training_data.drop( [ 'Survived', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' ], axis = 1 ) #We're dropping these because the 'Survived' column is our taret and other mentioned columns don't affect the surviving chance of the passenger.
x_guess = testing_data.drop( ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked' ], axis = 1 ) #We're dropping these because they don't affect the surviving chance of the passenger.

#Preparing the data:
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size = 0.2, random_state = 0 )

#Training different classification models:
##Logistic Regression:
log_reg = LogisticRegression()
log_reg.fit( x_train, y_train )
print( 'Test accuracy of Logistic Regression is {}'.format( log_reg.score( x_test, y_test ) * 100 ) )
##K-nearest neighbors:
scorelist = []
for i in range( 1, 20 ):
    knn = KNeighborsClassifier( n_neighbors = i )
    knn.fit( x_train, y_train )
    scorelist.append( knn.score( x_test, y_test ) * 100 )
plt.plot( range( 1, 20 ), scorelist )
plt.title( 'Accuracy of K-nearest neighbors' )
plt.xlabel( 'no, of neighbors' )
plt.ylabel( 'Test accuracy' )
plt.show()
print( 'Test accuracy of K-nearest neighbors is {} for no, of neighbors {}'.format( max( scorelist ), scorelist.index( max( scorelist ) ) + 1 ) )
##Support Vector Machine Algorithm:
svm = SVC( random_state = 1 )
svm.fit( x_train, y_train )
print( 'Test accuracy of Support Vector Machine algorithm is {}'.format( svm.score( x_test, y_test ) * 100 ) )
##Naive Bayes Algorithm:
nb = GaussianNB()
nb.fit( x_train, y_train )
print( 'Test accuracy of Naive Bayes algorithm is {}'.format( nb.score( x_test, y_test ) * 100 ) )
##Desicion Tree Algorithm:
dt = DecisionTreeClassifier()
dt.fit( x_train, y_train )
print( 'Test accuracy of Desicion Tree algorithm is {}'.format( dt.score( x_test, y_test ) * 100 ) )
##Random Forest Algorithm:
rf = RandomForestClassifier( n_estimators = 1000, random_state = 1 )
rf.fit( x_train, y_train )
print( 'Test accuracy of Random Forest algorithm is {}'.format( rf.score( x_test, y_test ) * 100 ) )

#Comparing the accuracy of classification models used:
methods = [ 'Logistic Regression', 'KNN', 'SVM', 'Naive Bayes', 'Desicion Tree', 'Random Forest' ]
accuracies = [ log_reg.score( x_test, y_test ) * 100, max( scorelist ), svm.score( x_test, y_test ) * 100, nb.score( x_test, y_test ) * 100, dt.score( x_test, y_test ) * 100, rf.score( x_test, y_test ) * 100 ]
colors = [ 'purple', 'green', 'orange', 'magenta', '#CFC60E', '#0FBBAE' ]
sb.set_style( 'whitegrid' )
plt.figure( figsize = ( 16, 5 ) )
sb.barplot( x = methods, y = accuracies, palette = colors )
plt.title( 'Comparing classification models' )
plt.xlabel( 'Classification models' )
plt.ylabel( 'Test accuracy' )
plt.show()

#Choosing the model having highest accuracy and predicting for testing data set:
print( 'As shown above, K-nearest neighbors and Support Vector Machine algorithm have the highest accuracies.' )
print( 'Hence we use any of the above for prediction of testing data set. I am using both.' )
knn = KNeighborsClassifier( n_neighbors = 8 )
knn.fit( x_train, y_train )
y_pred_knn = knn.predict( x_guess )
y_pred_svm = svm.predict( x_guess )
x_guess[ 'Survivors prediction from KNN' ] = y_pred_knn
x_guess[ 'Survivors prediction from SVM' ] = y_pred_svm
result = p.merge( testing_data, x_guess[ [ 'Survivors prediction from KNN', 'Survivors prediction from SVM' ] ], left_index = True, right_index = True )
print( result )