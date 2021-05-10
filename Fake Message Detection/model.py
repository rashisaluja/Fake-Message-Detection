# -*- coding: utf-8 -*-

# Importing the packages
import numpy as np
import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


from sklearn.metrics import confusion_matrix,accuracy_score
import pickle

# Loading the Dataset
data = pd.read_csv('Messages.csv')
data = data.drop(columns = ['Timestamp','Email Address','Sentiment','Unnamed: 5','Unnamed: 6','Unnamed: 7','Unnamed: 8','Unnamed: 9','Unnamed: 10'])

from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder() 
  
data['Real/Fake']= label_encoder.fit_transform(data['Real/Fake']) 

# Downloading the stopwords
nltk.download('stopwords')

corpus = []
for i in range(737):
    
    # Removing punctuations and numbers.
    review = re.sub('[^a-zA-Z]',' ',data['Message/Text'][i])
    
    # Converting to lower case
    review = review.lower()
    review = review.split()
    
    # Removing stopwords and stemming
    ps = PorterStemmer()
    review = [ ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    review = ' '.join(review)
    
    corpus.append(review)
    

corpus_df = pd.DataFrame(corpus)


corpus_df['corpus'] = corpus_df
corpus_df = corpus_df.drop([0],axis=1)


# Creating Bag of Words model
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
pickle.dump(cv,open('transform.pkl','wb'))
y = data.iloc[:,1].values


# Splitting the model
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20,random_state = 0)

m = GaussianNB()
m.fit(X_train,y_train)
#ypred = model.predict(X_test)

pickle.dump(m,open('model.pkl','wb'))



