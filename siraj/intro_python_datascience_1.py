# Basic program https://github.com/llSourcell/gender_classification_challenge/blob/master/demo.py
# https://www.youtube.com/watch?v=T5pRlIbr6gg

from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB

# [height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

classifier = tree.DecisionTreeClassifier()

classifier.fit(X,Y);

NewData = [[190, 70, 43]]

prediction = classifier.predict(NewData)
print('Decision Tree Prediction => ' + prediction[0])

cl2 = svm.SVC()
cl2.fit(X,Y);

prediction = cl2.predict(NewData)
print('SVM Prediction => ' + prediction[0])

gnb = GaussianNB()

print('Bayes Prediction => '+ gnb.fit(X,Y).predict(NewData)[0])