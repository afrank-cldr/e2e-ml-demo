import cml.data_v1 as cmldata
import pandas as pd
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pickle

import mlflow

import os
database_password = os.environ["WORKLOAD_PASSWORD"]
databse_user = os.environ["WORKLOAD_USER"]

CONNECTION_NAME = "afrank-dev"
conn = cmldata.get_connection(CONNECTION_NAME, {"USERNAME": databse_user, "PASSWORD": database_password})

## Sample Usage to get pandas data frame
EXAMPLE_SQL_QUERY = """
SELECT *
FROM default.afrank_test
; """

dataframe = conn.get_pandas_dataframe(EXAMPLE_SQL_QUERY)

X = np.array(dataframe[['x','y','z']])
y = np.array(dataframe[['y_label']])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

kernels = ["linear", "poly", "rbf", "sigmoid"]

mlflow.set_experiment('demo_test')

for kernel in kernels:
    mlflow.start_run()
    print(f'Kernel: {kernel}')
    mlflow.log_param("Kernel", kernel)
    clf = svm.SVC(kernel=kernel)
    clf.fit(X, y.ravel())
    preds = clf.predict(X_test)
    acc_score = accuracy_score(y_test, preds)
    mlflow.log_metric("accuracy_score", acc_score)
    
    print(f'Accuracy = {acc_score}')
    
    mlflow.end_run()