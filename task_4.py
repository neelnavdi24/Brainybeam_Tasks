import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/Neel Navdiwala/Downloads/breast-cancer.csv")
X = df.drop("diagnosis", axis=1)  # features
y = df["diagnosis"].map({"M": 1, "B": 0})  # malignant=1, benign=0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)


#  Filter Method – Correlation-based (SelectKBest)
from sklearn.feature_selection import SelectKBest, mutual_info_classif

sel_filter = SelectKBest(score_func=mutual_info_classif, k=10)
Xf_train = sel_filter.fit_transform(X_train, y_train)
Xf_test = sel_filter.transform(X_test)

# Wrapper Method – RFE with Logistic Regression
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(max_iter=10000, solver="liblinear")
sel_wrapper = RFE(estimator=lr, n_features_to_select=10, step=1)
Xw_train = sel_wrapper.fit_transform(X_train, y_train)
Xw_test = sel_wrapper.transform(X_test)

#  Embedded Method – Random Forest Feature Importance
from sklearn.ensemble import RandomForestClassifier
import numpy as np

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
importances = rf.feature_importances_
threshold = np.sort(importances)[-10]
selected_idx = np.where(importances >= threshold)[0]
Xe_train = X_train.iloc[:, selected_idx]
Xe_test = X_test.iloc[:, selected_idx]

# Train a Model (e.g. Random Forest) on each selection + Evaluate

from sklearn.metrics import accuracy_score

def train_eval(Xtr, Xte, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    print(f"{name} accuracy:", accuracy_score(y_test, preds))

train_eval(Xf_train, Xf_test, "Filter-based")
train_eval(Xw_train, Xw_test, "Wrapper-based (RFE)")
train_eval(Xe_train, Xe_test, "Embedded (RF importance)")

# Output in visualization

import matplotlib.pyplot as plt

# Store accuracy results
results = {}

def train_eval(Xtr, Xte, name):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(Xtr, y_train)
    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name} accuracy: {acc:.4f}")

# Run models
train_eval(Xf_train, Xf_test, "Filter-based")
train_eval(Xw_train, Xw_test, "Wrapper-based (RFE)")
train_eval(Xe_train, Xe_test, "Embedded (RF importance)")

#  Plotting the accuracy comparison
plt.figure(figsize=(8, 5))
plt.bar(results.keys(), results.values(), color=['skyblue', 'salmon', 'lightgreen'])
plt.ylim(0.85, 1.0)
plt.ylabel("Accuracy")
plt.title("Model Accuracy with Different Feature Selection Techniques")
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.005, f"{v:.2f}", ha='center', fontweight='bold')
plt.tight_layout()
plt.show()
