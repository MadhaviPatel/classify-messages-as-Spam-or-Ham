#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[3]:


d = pd.read_csv('spambase_csv.csv')


# In[4]:


d


# In[5]:


d.info()


# In[7]:


correct_dict={}
correct_dict=dict(d.corr()['class'])


# In[8]:


features = []
for key,values in correct_dict.items():
    if abs(values)<0.2:
        features.append(key)


# In[9]:


d = d.drop(features,axis=1)


# In[10]:


d


# In[11]:


y=d['class']
X=d.drop(['class'],axis=1)


# In[12]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)


# In[13]:


linear_regression = LogisticRegression(solver='liblinear', penalty='l1')
svc = SVC(kernel='sigmoid', gamma=1.0)
multi_nb = MultinomialNB(alpha=0.2)
decision_tr = DecisionTreeClassifier(min_samples_split=7, random_state=111)
k_neighbours = KNeighborsClassifier(n_neighbors=49)
random_fc = RandomForestClassifier(n_estimators=31, random_state=111)


# In[14]:


def train_classifier(object,dt,label):
  object.fit(dt,label)

def predict_classifier(pr_object,test_dt):
  return(pr_object.predict(test_dt))


# In[16]:


score_list = []
graph_sc = []
cls_dictionary = {'Linear_Regression': linear_regression,'SVC' : svc,'Multi_NB':multi_nb,'Decision_Tree': decision_tr,'KNN':k_neighbours,'Random_Forest': random_fc}
for key,value in cls_dictionary.items():
  train_classifier(value,X_train,y_train)
  pred_val = predict_classifier(value,X_test)
  graph_sc.append([accuracy_score(y_test,pred_val)])
  score_list.append((key,[accuracy_score(y_test,pred_val)]))
print(score_list)


# In[19]:


from nltk import flatten
flattened_list = flatten(graph_sc)
models = list(cls_dictionary.keys())

import matplotlib.pyplot as plt
plt.figure(figsize=(12,4))
plt.bar(models,flattened_list,width=0.3,color=['black', 'red', 'green', 'blue', 'cyan','purple'])
plt.xlabel('classifiers')
plt.ylabel('accuracy scores')


# In[ ]:




