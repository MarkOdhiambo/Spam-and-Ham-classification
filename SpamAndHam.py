# -*- coding: utf-8 -*-
"""
Fraud detection using Binary Classification

"""
import numpy as np
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords 
import string
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score

"Opening the dataset."
with open('model_data') as text_file:
    lines = text_file.read().split('\n')
    
"Dividing the data into data and labels."
lines = [line.split('\t') for line in lines if len(line.split('\t'))==2 and line.split('\t')[1]!=''] 

df =pd.DataFrame(lines,columns=['label','message'])
df.drop_duplicates(inplace=True)
df["label"].replace({"ham": "nofraud", "spam": "fraud"}, inplace=True)

def process_text(text):
    """This is for removing punctuation stopword and return a list of clean text
    for the model."""
    nopun=[char for char in text if char not in string.punctuation]
    nopun=''.join(nopun)
    clean_words=[word for word in nopun.split() if word.lower() not in stopwords.words('english')]
    return clean_words

df['message'].head().apply(process_text)

"Convert the matrix of tokens of matrix count"
count_vectorizer=CountVectorizer(analyzer=process_text)
count=count_vectorizer.fit_transform(df['message'])

"Split the data into training and testing."
x_train,x_test,y_train,y_test=train_test_split(count,df['label'],test_size=0.3,random_state=0)
classifier= MultinomialNB().fit(x_train,y_train)

"Evaluate the model on the test data"
# pred=classifier.predict(x_test)
# print(classification_report(y_test,pred))
# print()
# print('Confusion Matrix: \n',confusion_matrix(y_test,pred))
# print()
# print("Accuracy: ", accuracy_score(y_test,pred))

def user_inputer(user_input):
    user_inp=process_text(user_input)
    # label=classifier.predict(count_vectorizer.transform(user_inp))
    proba=classifier.predict_proba(count_vectorizer.transform(user_inp))
    nofraud_proba=np.mean(proba[:,0])
    fraud_proba=np.mean(proba[:,1])
    probability=[nofraud_proba,fraud_proba]
    if nofraud_proba>fraud_proba:
        label2="nofraud"
    else:
        label2="fraud"
    return probability,label2
