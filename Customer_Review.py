# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 10:52:57 2018

@author: amin
"""

#Yelp Customer Review

#imports
import numpy as np
import pandas as pd

#load data

df = pd.read_csv('yelp.csv')

#EDA

df.columns.values
df.describe()
df.head()
df.tail()
df.info()

#new column text length

df['text length'] = df['text'].apply(len)


#Visualization

import matplotlib.pyplot as plt
import seaborn as sns


sns.set_style('white')


#countplot

sns.countplot(x='stars',data=df,palette='rainbow')

#histagram

g = sns.FacetGrid(df,col='stars')
g.map(plt.hist,'text length')
plt.show()

#boxplot

sns.boxplot(x='stars',y='text length',data=df,palette='rainbow')


#Group
stars = df.groupby('stars').mean()

#correclation
stars.corr()

#heatmap
sns.heatmap(stars.corr(),cmap='coolwarm',annot=True)

#NLP Classification Dataframe

yelp_class = df[(df.stars==1) | (df.stars==5)]

#Create two objects X and y. X will be the 'text' column of yelp_class and y will be the 'stars' column of yelp_class
X = yelp_class['text']
y = yelp_class['stars']


#Import CountVectorizer and create a CountVectorizer object
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()


#Use the fit_transform method on the CountVectorizer object and pass in X (the 'text' column). Save this result by overwriting X.

X = cv.fit_transform(X)

#Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


#Training a Model

from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(X_train,y_train)

#Predictions and Evaluations

predictions = nb.predict(X_test)

#Create a confusion matrix and classification report using these predictions and y_test

from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test,predictions))
print('\n')
print(classification_report(y_test,predictions))




# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Confusion Matrix
# Predicting the Test set results
y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

cm = np.array([[2,1,0], [3,4,5], [6,7,8]])
true_pos = np.diag(cm)
false_pos = np.sum(cm, axis=0) - true_pos
false_neg = np.sum(cm, axis=1) - true_pos
 
precision = np.sum(true_pos / true_pos+false_pos)
recall = np.sum(true_pos / true_pos + false_neg)









#Using Text Processing

from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.pipeline import Pipeline

#create a pipeline

pipeline = Pipeline([
    ('bow', CountVectorizer()),  # strings to token integer counts
    ('tfidf', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
    ('classifier', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
])

# Train Test Split
X = yelp_class['text']
y = yelp_class['stars']
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)


#fit the pipeline to the training data

pipeline.fit(X_train,y_train)

#Predictions and Evaluation

predictions = pipeline.predict(X_test)

print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))








