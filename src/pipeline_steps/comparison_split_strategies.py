#!/usr/bin/env python
# coding: utf-8

# # User Story 16
# @LuiseJedlitschka
# 
# As already defined in leave-one-group-out.ipynb we use the metrics 'precision', 'recall', 'f1-score','support' to evaluate the split strategies. Here a random forest model is being trained with each split option. We evaluate the results and search for the strategy with the highest accuracy result, which is the stratified 80/20 split.

# In[7]:


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def test_model(train_path, test_path):
    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    le = LabelEncoder()
    le.fit(train_df["classification_x"])

    # detect Features
    feature_cols = [
        col for col in train_df.columns
        if col not in ["Unnamed: 0", "Geneid", "DNASequence", "classification_x", "group"]
    ]

    X_train = train_df[feature_cols]
    y_train = train_df["classification_x"]

    X_val = test_df[feature_cols]
    y_val = test_df["classification_x"]

    print(f"X_train: {X_train.shape}; y_train: {y_train.shape}")
    print(f"X_train: {X_val.shape}; y_train: {y_val.shape}")

    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    report = classification_report(y_val, y_pred, target_names=le.classes_, output_dict=True)
    print(report)

    accuracy = report["accuracy"]

    # return Accuracy and Split-Info
    return accuracy, train_path, test_path

# List of all splits
splits = [
    ("../data/combined-data-stratified-split/train_data.tsv", "../data/combined-data-stratified-split/test_data.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_0.tsv", "../data/leave-one-group-out-split/splits/test_split_0.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_1.tsv", "../data/leave-one-group-out-split/splits/test_split_1.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_2.tsv", "../data/leave-one-group-out-split/splits/test_split_2.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_3.tsv", "../data/leave-one-group-out-split/splits/test_split_3.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_4.tsv", "../data/leave-one-group-out-split/splits/test_split_4.tsv"),
    ("../data/leave-one-group-out-split/splits/train_split_5.tsv", "../data/leave-one-group-out-split/splits/test_split_5.tsv")
]

results = []
for train_path, test_path in splits:
    accuracy, train_path, test_path = test_model(train_path, test_path)
    results.append((accuracy, train_path, test_path))

# find best Split depending on accuracy and print its Info
best = max(results, key=lambda x: x[0])
print("\nBester Split:")
print(f"Train: {best[1]}\nTest: {best[2]}\nAccuracy: {best[0]:.4f}")

