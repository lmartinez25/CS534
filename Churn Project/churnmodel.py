from cleanchurn import *  #imports clean data df
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
#from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

#convert categorical to integer representations using replace for outcome and
# one hot encoding for attributes
#concern with one hot is dimension will be large
df['Churn'] = df['Churn'].replace({'Yes':1,'No':0})
Xcat = pd.get_dummies(df[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']]).astype(int)
Xnum = df[['MonthlyCharges', 'TotalCharges','tenure']]
X = pd.concat([Xcat, Xnum], axis=1)

print(X.info())
#split data
y = df['Churn']
y = df['Churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state=42)


#balance training sets
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)
print(y_train_bal.value_counts())


#cross validation for models with default settings
clf1 = RandomForestClassifier(random_state=1)
clf2 = SVC()
#clf3 = XGBClassifier(random_state=1)

for clf, label in zip([clf1, clf2], ['Random Forest', 'SVM']):
    scores = cross_val_score(clf, X_train_bal, y_train_bal, scoring='accuracy', cv=5)
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))
    print(scores)

#Random forest y_test predictions and evaluation
clf1.fit(X_train_bal, y_train_bal)
predictions = clf1.predict(X_test)
print('\nAccuracy score for random forest: ' , accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))

fpr, tpr, thresholds = roc_curve(y_test, predictions)
roc_auc = roc_auc_score(y_test, predictions)
plt.plot(fpr, tpr, label='ROC Curve (area = %0.3f)' % roc_auc)
plt.title('ROC Curve (area = %0.3f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()