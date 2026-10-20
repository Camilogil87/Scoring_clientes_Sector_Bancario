# Importing necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost.sklearn import XGBClassifier  
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from hyperopt import tpe, hp, fmin, STATUS_OK,Trials
from hyperopt.pyll.base import scope
import mlflow
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

mlflow.set_tracking_uri('sqlite:///mlflow.db')
mlflow.set_experiment('hiper-PF')

df = pd.read_csv('/Users/miguelvillasan/Library/CloudStorage/GoogleDrive-miguelvv@alumni.mioti.es/Unidades compartidas/Proyecto Final MDB2209A/MLFlow/df_model_2.csv', sep=',')
print("##### Data Preprocessing #####\n")
print(f'Numero de datos que tenemos: {len(df)}\n')

print(df.head(5).to_markdown())

print("\n##### Dataset Balancing #####\n")
# Primero hacemos un split de nuestros datos entre variables independientes y variables objetivo
X = df.drop(columns=['id_final'])
y = df['id_final']
print(f'Numero de casos de no compra vs compra: {Counter(y)}')
print("\n##### Model Training #####\n")

# Hacemos un split de nuestros datos
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X, y, test_size=.3, stratify=y, random_state=42)

space = {
    "n_estimators": hp.choice("n_estimators", [2, 5, 10, 25, 50, 100, 200, 300]),
    "max_depth": hp.choice("max_depth", [1, 2, 3, 5, 8]),
    "criterion": hp.choice("criterion", ["gini", "entropy"]),
    "class_weight": hp.choice("class_weight", [{0:.1, 1:.9}, {0:.2, 1:.8}, {0:.3, 1:.7}, {0:.4, 1:.6}]),
}

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag('model', 'Random Forest')
        mlflow.set_tag('author', 'Miguel')
        mlflow.log_params(params)
        global clf
        clf = RandomForestClassifier(**params, n_jobs=-1, random_state=42)
        clf.fit(X_train_balanced, y_train_balanced)

        y_pred = clf.predict(X_test_balanced)
        y_score_1 = clf.predict_proba(X_test_balanced)[:,1]
        accuracy = metrics.accuracy_score(y_test_balanced, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', metrics.f1_score(y_test_balanced, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test_balanced, y_pred))
        mlflow.log_metric('precision', metrics.precision_score(y_test_balanced, y_pred))
        mlflow.log_metric('roc_auc', metrics.roc_auc_score(y_test_balanced, y_score_1))
       
    return {'loss': 1 - metrics.recall_score(y_test_balanced, y_pred), 'status': STATUS_OK}

best_result = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )
df_prob = pd.DataFrame(clf.predict_proba(X_test_balanced))
df_prob.to_csv('/Users/miguelvillasan/Library/CloudStorage/GoogleDrive-miguelvv@alumni.mioti.es/Unidades compartidas/Proyecto Final MDB2209A/MLFlow/df_prob_RandomForest.csv', sep=',')

print("\n##### Model 2 Training #####\n")


space_2 = {
    "C": hp.choice("C", [0.1, 0.3, 0.5, 0.7, 1, 1.5, 2, 3]),
    "class_weight": hp.choice("class_weight", [{0:.1, 1:.9}, {0:.2, 1:.8}, {0:.3, 1:.7}, {0:.4, 1:.6}]),
}

def objective_2(params):
    with mlflow.start_run():
        mlflow.set_tag('model', 'Logistic Regresion')
        mlflow.set_tag('author', 'Miguel')
        mlflow.log_params(params)
        global logistic
        logistic = LogisticRegression(**params, n_jobs=-1, solver='lbfgs', max_iter=3000, random_state=42)
        logistic.fit(X_train_balanced, y_train_balanced)

        y_pred = logistic.predict(X_test_balanced)
        y_score_1 = logistic.predict_proba(X_test_balanced)[:,1]
        accuracy = metrics.accuracy_score(y_test_balanced, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', metrics.f1_score(y_test_balanced, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test_balanced, y_pred))
        mlflow.log_metric('precision', metrics.precision_score(y_test_balanced, y_pred))
        mlflow.log_metric('roc_auc', metrics.roc_auc_score(y_test_balanced, y_score_1))
       
    return {'loss': 1 - metrics.recall_score(y_test_balanced, y_pred), 'status': STATUS_OK}

best_result_2 = fmin(
        fn=objective_2,
        space=space_2,
        algo=tpe.suggest,
        max_evals=100,
        trials=Trials()
    )

df_prob_2 = pd.DataFrame(logistic.predict_proba(X_test_balanced))
df_prob_2.to_csv('/Users/miguelvillasan/Library/CloudStorage/GoogleDrive-miguelvv@alumni.mioti.es/Unidades compartidas/Proyecto Final MDB2209A/MLFlow/df_prob_LogisticRegression.csv', sep=',')

print("\n##### Model 3 Training #####\n")

space_3 = {
    "n_estimators": hp.choice("n_estimators",[5, 10, 20, 40, 70, 100, 200]), 
    "max_depth": hp.choice("max_depth",[3, 5, 10, 20, 40]), 
    "learning_rate": hp.choice("learning_rate",[0.05, 0.07, 0.1, 0.2, 0.4]), 
    "colsample_bytree": hp.choice("colsample_bytree",[0.7, 0.6, 0.5, 0.4, 0.3]), 
    "subsample": hp.choice("subsample", [0.6, 0.5, 0.4]),
}

def objective_3(params):
    with mlflow.start_run():
        mlflow.set_tag('model', 'XGBClassifier')
        mlflow.set_tag('author', 'Miguel')
        mlflow.log_params(params)
        global xgboost
        xgboost = XGBClassifier(**params, n_jobs=-1, random_state=42)
        xgboost.fit(X_train_balanced, y_train_balanced)

        y_pred = xgboost.predict(X_test_balanced)
        y_score_1 = xgboost.predict_proba(X_test_balanced)[:,1]
        accuracy = metrics.accuracy_score(y_test_balanced, y_pred)
        mlflow.log_metric('accuracy', accuracy)
        mlflow.log_metric('f1', metrics.f1_score(y_test_balanced, y_pred))
        mlflow.log_metric('recall', metrics.recall_score(y_test_balanced, y_pred))
        mlflow.log_metric('precision', metrics.precision_score(y_test_balanced, y_pred))
        mlflow.log_metric('roc_auc', metrics.roc_auc_score(y_test_balanced, y_score_1))
       
    return {'loss': 1 - metrics.recall_score(y_test_balanced, y_pred), 'status': STATUS_OK}

best_result_3 = fmin(
        fn=objective_3,
        space=space_3,
        algo=tpe.suggest,
        max_evals=150,
        trials=Trials()
    )

df_prob_3 = pd.DataFrame(xgboost.predict_proba(X_test_balanced))
df_prob_3.to_csv('/Users/miguelvillasan/Library/CloudStorage/GoogleDrive-miguelvv@alumni.mioti.es/Unidades compartidas/Proyecto Final MDB2209A/MLFlow/df_prob_Xgboost.csv', sep=',')