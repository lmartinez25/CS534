from cleanchurn import *  #imports clean data df
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.preprocessing import StandardScaler

#ADDED
from sklearn.model_selection import GridSearchCV

#ADDED
def find_parameters(clf, model_num, score):
    # uses grid search to find best parameters when running models
    # score is the scoring function used. we used f1, accuracy, precision and recall
    # model_num can be index 1 to 3
    # clf can be clf1, clf2, clf3
    names = ['Random Forest', 'SVM', 'XGboost']
    print('\n\n',names[model_num - 1])

    # RF default: n_estimators = 100, max_depth = None
    parameters1 = {'n_estimators': [100, 50, 30, 10, 200], 'max_depth': [100, 50, 30, 10,5]}

    # SVM default: WRITE IN
    parameters2 = 0 #{'kernel': ('poly', 'rbf', 'sigmoid'), 'C': [1, 10, 20] , 'gamma': ['auto']},

    # WRITE IN XGBOOST
    parameters3 = 0 # FILL IN

    param = [parameters1, parameters2, parameters3]

    #update model with desired parameters
    clfnew = GridSearchCV(clf, param[model_num - 1], cv=5, scoring=score)
    clffit = clfnew.fit(X_train_bal, y_train_bal)
    #print(clffit.cv_results_['mean_test_score'])
    #print(clffit.cv_results_['params'], '\n')
    print('Best parameters: ', clffit.best_params_)
    print(f'Best {score} score: ', clffit.best_score_)


#convert categorical to integer representations using replace for outcome and
# one hot encoding for attributes
df['Churn'] = df['Churn'].replace({'Yes':1,'No':0})
Xcat = pd.get_dummies(df[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
       'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
       'PaperlessBilling', 'PaymentMethod']]).astype(int)
Xnum = df[['MonthlyCharges', 'TotalCharges','tenure']]
X = pd.concat([Xcat, Xnum], axis=1)

print(X.info())

#split data
y = df['Churn'].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42)

# Scale features for SVC
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Balance training sets — one for each pipeline
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)           # RF, XGB (unscaled)
X_train_bal_scaled, y_train_bal_scaled = smote.fit_resample(X_train_scaled, y_train)  # SVC (scaled)
print(y_train_bal.value_counts())

clf1 = RandomForestClassifier(random_state=1, max_depth=10, n_estimators=100)
clf2 = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=1)
clf3 = XGBClassifier(random_state=1)

def print_metrics(label, y_true, y_pred):
    print(f'\n--- {label} ---')
    print(f'Accuracy:  {accuracy_score(y_true, y_pred):.4f}')
    print(f'Precision: {precision_score(y_true, y_pred):.4f}')
    print(f'Recall:    {recall_score(y_true, y_pred):.4f}')
    print(f'F1 Score:  {f1_score(y_true, y_pred):.4f}')
    print('Confusion Matrix:')
    print(confusion_matrix(y_true, y_pred))

# Random Forest — fit and evaluate on unscaled data
clf1.fit(X_train_bal, y_train_bal)
predictions_rf = clf1.predict(X_test)
print_metrics('Random Forest', y_test, predictions_rf)

fpr_rf, tpr_rf, _ = roc_curve(y_test, predictions_rf)
roc_auc_rf = roc_auc_score(y_test, predictions_rf)

# SVC — fit and evaluate on scaled data
clf2.fit(X_train_bal_scaled, y_train_bal_scaled)
predictions_svc = clf2.predict(X_test_scaled)
predictions_svc_proba = clf2.predict_proba(X_test_scaled)[:, 1]
print_metrics('SVC', y_test, predictions_svc)

fpr_svc, tpr_svc, _ = roc_curve(y_test, predictions_svc_proba)
roc_auc_svc = roc_auc_score(y_test, predictions_svc_proba)


# XGBoost — fit and evaluate on unscaled data (XGB handles raw features well)
clf3.fit(X_train_bal, y_train_bal)
predictions_xgb = clf3.predict(X_test)
predictions_xgb_proba = clf3.predict_proba(X_test)[:, 1]
print_metrics('XGBoost', y_test, predictions_xgb)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, predictions_xgb_proba)
roc_auc_xgb = roc_auc_score(y_test, predictions_xgb_proba)


# Plot all ROC curves
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf,  tpr_rf,  label='Random Forest (AUC = %0.3f)' % roc_auc_rf)
plt.plot(fpr_svc, tpr_svc, label='SVC (AUC = %0.3f)' % roc_auc_svc)
plt.plot(fpr_xgb, tpr_xgb, label='XGBoost (AUC = %0.3f)' % roc_auc_xgb)
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()



#ADDED
#used function to optimize parameters based on different scoring functions
scores = {'accuracy','precision','recall','f1'}
for score in scores:
    find_parameters(clf1, 1, score)

#ADDED
#BEST PARAMETERS FOR RANDOM FOREST
#FINAL CHOICE: max_depth = 10, n_estimators = 100
#NOTE: default is None and 100
#accuracy
#{'max_depth': 10, 'n_estimators': 100}
#precision
#{'max_depth': 30, 'n_estimators': 50}
#recall
#{'max_depth': 5, 'n_estimators': 100}
#f1
#{'max_depth': 10, 'n_estimators': 100}

