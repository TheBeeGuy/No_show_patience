import platform, sklearn.metrics, datetime, os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_auc_score
from datetime import datetime
import multiprocess as mp

#read current working directory
##assume running notebook from No_show_patience directory
cwd = os.getcwd()

def trainxgb(params):
    full_dataset = pd.read_pickle(cwd + "/data/postFT2018.pkl")
    X = full_dataset.drop(['noshow'],axis=1)
    X = X.drop(X.filter(regex='MODE',axis=1).columns, axis=1)
    y = full_dataset[['noshow']]
    cutoff_month = 8
    X_train, X_test, y_train, y_test = (X[X['MONTH(AppointmentDate)']<=cutoff_month],
                                        X[X['MONTH(AppointmentDate)']>cutoff_month],
                                        y[X['MONTH(AppointmentDate)']<=cutoff_month],
                                        y[X['MONTH(AppointmentDate)']>cutoff_month])
    X_train, X_test = X_train.fillna(X_train.mean()), X_test.fillna(X_test.mean())
    startTime = datetime.now()
    model_xgb = (XGBClassifier(n_estimators=int(params['n_estimators']), max_depth = int(params['max_depth']),
                               eta = params['eta'], subsample = params['subsample'], alpha= params['alpha'], n_thread = mp.cpu_count(),
                               tree_method = 'hist', random_state=1).fit(X_train,y_train.values.ravel()))
    y_pred_model_xgb = pd.DataFrame(model_xgb.predict_proba(X_test)[:,1])
    finishTime = datetime.now()
    loss = 1 - sklearn.metrics.roc_auc_score(y_test, y_pred_model_xgb)
    return {"loss": loss, "time": (finishTime - startTime).total_seconds()}