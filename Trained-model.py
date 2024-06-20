# Importing necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import streamlit as st
import pickle 

# Load the dataset
def load_data():
    data = pd.read_csv('creditcard.csv')
    return data

data = load_data()

# Preprocess the data
X = data.drop(columns=['Class'])
Y = data['Class']
smot = SMOTE(random_state=42)
X_res, Y_res = smot.fit_resample(X, Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_res, Y_res, test_size=0.2, random_state=42)

xgb_model = XGBClassifier(n_estimators=50, max_depth=3, use_label_encoder=False, eval_metric='logloss')
xgb_model.fit(X_train, Y_train)

# Calculate accuracy for XGBoost
xgb_predictions = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(Y_test, xgb_predictions)
print(f"XGBoost Accuracy: {xgb_accuracy}")

# Save the model as pickle file for deployment
with open('fraud-model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

