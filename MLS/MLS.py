import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split,learning_curve,cross_val_score,KFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn import metrics
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from methods import *



df = pd.read_csv('dataset/enron2_clean.csv',sep=',')
print("Wczytanie pliku z wiadomościami")
print("Wyświetlenie 5 rekordów")
df.columns = ['label', 'email']
print(df.head())
clean_all(df)
pd.set_option('display.max_colwidth',50)
pd.set_option('display.max_columns',30)
print("Wyświetlenie 5 rekordów")
print(df.head())
print("Wyswietleni macierzy")
print(df.shape)
print("Mapowanie wiadomości 0 - Ham, 1 - Spam")
df['label'] = df.label.map({'ham':0, 'spam':1})
print("Usunięcie duplikatów oraz pustych rekordów")
df.drop_duplicates(inplace = True)
df.dropna()
print("Wyświetlenie 5 rekordów")
print(df.head())
print("Wyswietlenie macierzy")
print(df.shape)

X = df.clean_emails
y = df.label
print("X: ", df.shape)
print("y: ",y.shape)
print("Podzielenie X oraz y na zestawy testowe i treningowe")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print("X train: ", X_train.shape)
print("X test: ", X_test.shape)
print("y train: ", y_train.shape)
print("y test: ", y_test.shape)


vect = TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')
X_vect = vect.fit_transform(X)
print("Macierz Tfidf Vectorizer:",X_vect.shape)

X_train_dtm = vect.fit_transform(X_train)
X_test_dtm = vect.transform(X_test)
k_fold = KFold(n_splits=10, shuffle=True)

print("Wielomianowy klasyfikator Naive Bayesa")
nb = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
nb.fit(X_train_dtm, y_train)

y_pred_class_nb = nb.predict(X_test_dtm)
acc_score_nb = accuracy_score(y_test, y_pred_class_nb)


y_pred_prob_nb = nb.predict_proba(X_test_dtm)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob_nb)
auc_score_nb = roc_auc_score(y_test, y_pred_prob_nb)
plt.title('Krzywa ROC - Naive Bayes')
plt.plot(fpr,tpr,label="AUC="+str(auc_score_nb))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.ylabel('Ocena Prawdziwie Pozytywna')
plt.xlabel('Ocena Fałszywie Pozytywna')
plt.show()

clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')), ('nb', MultinomialNB())])
cv_nb = cross_val_score(clf, X, y,cv=k_fold)

print("Naive Bayes Cross-Validation Score:", cv_nb)
print("Naive Bayes Cross-Validation Accuracy Score: %0.2f (+/- %0.2f)" % (cv_nb.mean(), cv_nb.std() * 2))
print("Naive Bayes Accuracy Score:",acc_score_nb)
print("Naive Bayes ROC-AUC score:",auc_score_nb)
print('Confusion Matrix NB: \n', confusion_matrix(y_test, y_pred_class_nb))
cnf_matrix = confusion_matrix(y_test, y_pred_class_nb)
plot_confusion_matrix(cnf_matrix, classes=['Spam','Ham'],title='Macierz Pomyłek - Naive Bayes')


print("Klasyfikator - Regresja Logistyczna")
logreg = LogisticRegression(penalty='l2', C=1.0, solver='liblinear', max_iter=100000)
logreg.fit(X_train_dtm, y_train)

y_pred_class_logreg = logreg.predict(X_test_dtm)
acc_score_logreg = accuracy_score(y_test, y_pred_class_logreg)

y_pred_prob_logreg = logreg.predict_proba(X_test_dtm)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob_logreg)
auc_score_logreg = roc_auc_score(y_test, y_pred_prob_logreg)
plt.title('Krzywa ROC - Regresja Logistyczna')
plt.plot(fpr,tpr,label="AUC="+str(auc_score_logreg))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.ylabel('Ocena Prawdziwie Pozytywna')
plt.xlabel('Ocena Fałszywie Pozytywna')
plt.show()

clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')), ('logreg', LogisticRegression(max_iter=100000))])
cv_logreg = cross_val_score(clf, X, y, cv=k_fold)

print(f'LR Cross-Validation score: ', cv_logreg)
print("Naive Bayes Cross-Validation Accuracy Score: %0.2f (+/- %0.2f)" % (cv_logreg.mean(), cv_logreg.std() * 2))
print(f'LR Accuracy score: ',acc_score_logreg)
print(f'LR ROC-AUC score: ',auc_score_logreg)
print('Confusion Matrix LR: \n', confusion_matrix(y_test, y_pred_class_logreg))
cnf_matrix = confusion_matrix(y_test, y_pred_class_nb)
plot_confusion_matrix(cnf_matrix, classes=['Spam','Ham'],title='Macierz Pomyłek - Klasyfikator Regresja Logistyczna')


print("Klasyfikator - Wektor Wsparcia Liniowego SVC")
linSVC = LinearSVC(penalty='l2', C=1.0,max_iter=100000)
linSVC.fit(X_train_dtm, y_train)

y_pred_class_linSVC = linSVC.predict(X_test_dtm)
acc_score_linSVC = accuracy_score(y_test, y_pred_class_linSVC)

y_pred_prob_linSVC = linSVC._predict_proba_lr(X_test_dtm)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_prob_linSVC)
auc_score_linSVC = roc_auc_score(y_test, y_pred_prob_linSVC)
plt.title('Krzywa ROC - Wektor Wsparcia Liniowego SVC')
plt.plot(fpr,tpr,label="AUC="+str(auc_score_linSVC))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.ylabel('Ocena Prawdziwie Pozytywna')
plt.xlabel('Ocena Fałszywie Pozytywna')
plt.show()

clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')), ('linSVC', LinearSVC(max_iter=100000))])
cv_linSVC = cross_val_score(clf, X, y, cv=k_fold)

print("SVC Cross-Validation Score: ", cv_linSVC)
print("SVC Cross-Validation Accuracy Score: %0.2f (+/- %0.2f)" % (cv_linSVC.mean(), cv_linSVC.std() * 2))
print("SVC Accuracy score: ",acc_score_linSVC)
print("SVC ROC-AUC score: ",auc_score_linSVC)
print("Confusion Matrix SVC: \n", confusion_matrix(y_test, y_pred_class_linSVC))
cnf_matrix2 = confusion_matrix(y_test, y_pred_class_logreg)
plot_confusion_matrix(cnf_matrix2, classes=['Spam','Ham'],title='Macierz Pomyłek - LinearSVC')


print("Klasyfikator - Algorytm KNN")
knn = KNeighborsClassifier()
knn.fit(X_train_dtm, y_train)

y_pred_class_knn = knn.predict(X_test_dtm)
y_pred_prob_knn = knn.predict_proba(X_test_dtm)[:, 1]

acc_score_knn = accuracy_score(y_test, y_pred_class_knn)
auc_score_knn = roc_auc_score(y_test, y_pred_prob_knn)

y_pred_prob_knn = knn.predict_proba(X_test_dtm)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_prob_knn)
auc_score_knn = roc_auc_score(y_test, y_pred_prob_knn)
plt.title('Krzywa ROC - Algorytm KNN')
plt.plot(fpr,tpr,label="AUC="+str(auc_score_knn))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.ylabel('Ocena Prawdziwie Pozytywna')
plt.xlabel('Ocena Fałszywie Pozytywna')
plt.show()


clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')), ('knn', KNeighborsClassifier())])
cv_knn = cross_val_score(clf, X, y, cv=k_fold)

print(f'KNN Cross-Validation score: ', cv_knn)
print("KNN Cross-Validation Accuracy Score: %0.2f (+/- %0.2f)" % (cv_knn.mean(), cv_knn.std() * 2))
print(f'KNN Accuracy score: ',acc_score_knn)
print(f'KNN ROC-AUC score: ',auc_score_knn)
print('Confusion Matrix KNN: \n', confusion_matrix(y_test, y_pred_class_knn))
cnf_matrix2 = confusion_matrix(y_test, y_pred_class_knn)
plot_confusion_matrix(cnf_matrix2, classes=['Spam','Ham'],title='Macierz Pomyłek - Algorytm KNN')

print("Klasyfikator Sieci Neuronowe Perceptron")
per= Perceptron()
per.fit(X_train_dtm, y_train)

y_pred_class_per = per.predict(X_test_dtm)
acc_score_per = accuracy_score(y_test, y_pred_class_per)

clf_isotonic = CalibratedClassifierCV(per, cv=10, method='isotonic')
clf_isotonic.fit(X_train_dtm, y_train)
y_pred_prob_per = clf_isotonic.predict_proba(X_test_dtm)[::,1]
fpr, tpr, _ = roc_curve(y_test,  y_pred_prob_per)
auc_score_per = roc_auc_score(y_test, y_pred_prob_per)
plt.title('Krzywa ROC - Perceptron')
plt.plot(fpr,tpr,label="AUC="+str(auc_score_per))
plt.plot([0, 1], [0, 1],'r--')
plt.legend(loc=4)
plt.ylabel('Ocena Prawdziwie Pozytywna')
plt.xlabel('Ocena Fałszywie Pozytywna')
plt.show()


clf = Pipeline([('vect', TfidfVectorizer(ngram_range=(1, 1), max_df=1.0, min_df=1, norm='l2')), ('per', Perceptron())])
cv_per = cross_val_score(clf, X, y, cv=k_fold)

print(f'Perceptron Cross-Validation score: ', cv_per)
print("Perceptron Cross-Validation Accuracy Score: %0.2f (+/- %0.2f)" % (cv_per.mean(), cv_per.std() * 2))
print(f'Perceptron Accuracy score: ',acc_score_per)
print(f'Perceptron ROC-AUC score: ',auc_score_per)
print('Confusion Matrix Perceptron: \n', confusion_matrix(y_test, y_pred_class_per))
cnf_matrix2 = confusion_matrix(y_test, y_pred_class_per)
plot_confusion_matrix(cnf_matrix2, classes=['Spam','Ham'],title='Macierz Pomyłek - Perceptron')
input("Wcisnij Enter zeby zamknac")

