

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

ndf=pd.read_csv("/content/drive/MyDrive/news.csv")

ndf.head()

ndf.info()

nndf=ndf.dropna()

nndf['text']=nndf['text'].astype(str)

def remove_punctuation(text):
  import string
  remover=str.maketrans('','',string.punctuation)
  return text.translate(remover)

nndf['text']=nndf['text'].apply(remove_punctuation)
nndf.head()

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words=stopwords.words("english")

def remove_stopwords(text):
  text = [word.lower() for word in text.split() if word.lower() not in stop_words]
  return " ".join(text)

nndf['text']=nndf['text'].apply(remove_stopwords)
nndf.head()

nndf['label']=np.where(nndf['label']=='REAL',1,0)

x=nndf['text']
x

y=nndf['label']
y

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

tfidvec=TfidfVectorizer()
tfidx_train=tfidvec.fit_transform(x_train)
tfidx_test=tfidvec.transform(x_test)

pac=PassiveAggressiveClassifier()
pac.fit(tfidx_train,y_train)

pac_pred=pac.predict(tfidx_test)
pac_pred

ascore=accuracy_score(y_test,pac_pred)
ascore

report=confusion_matrix(y_test,pac_pred)
report
