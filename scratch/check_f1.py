import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

DATA_DIR = "E:/projects/scientific_project/scratch/phishing_gdrive/"

res = []
for f, fname in [('html', 'HTML_All_Features.csv'), ('pdf', 'PDF_All_features.csv'), ('word', 'Word_All_features.csv'), ('excel', 'Excel_All_Features.csv')]:
    df = pd.read_csv(os.path.join(DATA_DIR, fname))
    df.fillna(0, inplace=True)
    X = df.select_dtypes(include=[np.number])
    if 'label' in X: X = X.drop(columns=['label'])
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    res.append({'Type': f, 'Acc': accuracy_score(y_test, preds), 'F1': f1_score(y_test, preds)})

print(pd.DataFrame(res))
