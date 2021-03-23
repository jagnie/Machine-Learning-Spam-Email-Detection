import string
import re
import nltk
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from numpy.random import RandomState
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from collections import Counter
import time


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Reds,
                          alpha =0.65):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range (cm.shape[0]):
        for j in range (cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Klasy Prawdziwe')
    plt.xlabel('Klasy Przewidywane')
    plt.tight_layout()
    np.set_printoptions(precision=2)
    plt.show()

def stopwords_tokenize_punctuation(emails):
    stop = set(stopwords.words("english"))
    stop.update(('enron','ect','hou','com','kaminski','subject'))
    emails = "".join([word for word in emails if word not in string.punctuation])
    white_spaces = emails.strip()
    tokens = re.split('\W+',white_spaces)
    emails = [word for word in tokens if word not in (stop)]
    return emails

def after_stemmer(emails):
    ps = PorterStemmer()
    return [ps.stem(word) for word in emails]

def after_lemmatizer(emails):
    wn = WordNetLemmatizer()
    return [wn.lemmatize(word) for word in emails]

def clean_emails(emails):
    ps = PorterStemmer()
    stop = set(stopwords.words("english")) 
    stop.update(('enron','ect','hou','com','kaminski','subject'))
    emails = "".join([word for word in emails if word not in string.punctuation])
    white_spaces = emails.strip()
    nums = re.sub('[0-9]+', '', white_spaces)
    tokens = re.split('\W+',nums)
    tokens_text=[w for w in tokens if len(w)>2]
    emails = [ps.stem(word) for word in tokens_text if word not in (stop)]
    return emails

def clean_all(df_emails):
    print("-------AFTER DELETE STOPWORDS/PUNCTUATION/STEMMER--------")
    df_emails['clean_emails'] = df_emails['email'].apply(lambda x: clean_emails(x.lower()))
    print(df_emails.head())


def save_to_new_csv(emails):
    select_list =[line for line  in emails.columns]
    write_data = emails[select_list]
    df = pd.DataFrame(write_data)
    return df.to_csv('dataset/enron2_clean_features_v2.csv',index=False)

def split_emails(df):
    rng = RandomState()
    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]
    train.to_csv('dataset/test_train/train_emails.csv', index=False)
    test.to_csv('dataset/test_train/test_emails.csv', index=False)
def dictionary(emails):
    stop = set(stopwords.words("english"))
    stop.update(('enron','ect','hou','com','kaminski','subject'))
    all_emails = emails['email'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    all_words = []       
    for mail in all_emails:
        words = mail.split()
        all_words += words
    dictionary = Counter(all_words)
    list_to_remove = list(dictionary)
    for item in list_to_remove:
        if item.isalpha() == False:
            del dictionary[item]
        elif len(item) == 1:
            del dictionary[item]
    dictionary = dictionary.most_common(2000)
    df = pd.DataFrame(dictionary, columns=['words', 'count'])
    return df.to_csv('dataset/Dictionary/dictionary.csv')

def punctuation_count(df_emails):
    punc = sum([1 for c in df_emails if c in string.punctuation])
    return punc

def avg_words(df_emails):
    words = df_emails.split()
    avg = sum(len(word) for word in words if word not in string.punctuation) / len(words)
    return avg

