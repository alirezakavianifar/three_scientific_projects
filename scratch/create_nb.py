import nbformat as nbf

nb = nbf.v4.new_notebook()

# Phase 0: Imports
nb.cells.append(nbf.v4.new_markdown_cell("""# CIC Trap4Phish Multi-File Phishing Detection (Refined)

This notebook analyzes phishing document features extracted from multiple file types (HTML, PDF, Word, Excel).

### Core Improvements in this Version:
1. **Data Integrity**: Global duplicate check across merged datasets to prevent inflated performance.
2. **Leakage Prevention**: Feature scaling (StandardScaler) is applied strictly *after* the train-test split.
3. **Analytical Depth**: Added Confusion Matrices and ROC-AUC evaluations.
4. **Comparison**: Formal performance comparison across document types."""))

nb.cells.append(nbf.v4.new_code_cell("""# Standard data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for classical ML & feature selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             classification_report, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, RFE

# TensorFlow / Keras for Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Set visual style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.dpi'] = 100"""))

# Phase 1: Data Integration & Cleaning
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 1: Data Integration & Cleaning

We merge the datasets and perform a critical check for **Duplicate Rows** which could lead to data leakage if identical samples appear in both training and testing sets."""))

nb.cells.append(nbf.v4.new_code_cell("""# Define path to the shared drive dataset
BASE_PATH = "/content/drive/MyDrive/1Id54AbOUHXQkbefV6XmabsNRUbMOFYt5/"

# Load individual file types
excel = pd.read_csv(BASE_PATH + "Excel_All_Features.csv")
word = pd.read_csv(BASE_PATH + "Word_All_features.csv")
pdf = pd.read_csv(BASE_PATH + "PDF_All_features.csv")
html = pd.read_csv(BASE_PATH + "HTML_All_Features.csv")

# Add file type label
excel['file_type'] = 'excel'
word['file_type'] = 'word'
pdf['file_type'] = 'pdf'
html['file_type'] = 'html'

# Combine all datasets
data = pd.concat([excel, word, pdf, html], ignore_index=True)

# 1. Global Duplicate Check
initial_count = len(data)
data.drop_duplicates(inplace=True)
dropped_count = initial_count - len(data)

# 2. Handle missing features
data.fillna(0, inplace=True)

print(f"Initial row count: {initial_count}")
print(f"Dropped duplicates: {dropped_count}")
print(f"Final dataset shape: {data.shape}")
display(data['file_type'].value_counts())"""))

# Phase 2: Per-File Type Baselines
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 2: Performance Comparison Across Document Types

We evaluate how well a Random Forest classifier performs on each document type in isolation."""))

nb.cells.append(nbf.v4.new_code_cell("""# Identify features (exclude label and non-numeric metadata)
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

results_per_type = []

for ftype in data['file_type'].unique():
    subset = data[data['file_type'] == ftype]
    
    X_sub = subset[numeric_cols]
    y_sub = subset['label']
    
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
    )
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_sub, y_train_sub)
    
    preds = clf.predict(X_test_sub)
    results_per_type.append({
        'File Type': ftype, 
        'Accuracy': accuracy_score(y_test_sub, preds), 
        'F1-Score': f1_score(y_test_sub, preds)
    })

# Formal Comparison Table
res_df = pd.DataFrame(results_per_type).sort_values(by='Accuracy', ascending=False)
display(res_df)

plt.figure(figsize=(10, 5))
sns.barplot(data=res_df, x='File Type', y='Accuracy', palette='magma')
plt.title('Performance comparison by Document Type')
plt.ylim(0, 1.05)
plt.show()"""))

# Phase 3: Global Model & Leakage Prevention
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 3: Global Model (Preventing Data Leakage)

**Crucial Step**: We split the data into training and testing sets **before** fitting the `StandardScaler`. This ensures the test set remains completely unseen by the preprocessing parameters."""))

nb.cells.append(nbf.v4.new_code_cell("""X = data[numeric_cols]
y = data['label']

# 1. Split first
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Scale second (Fit on train, transform both)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=X_train_raw.columns)
X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=X_test_raw.columns)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")"""))

# Phase 4: Feature Selection
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 4: Feature Selection (MI vs RFE)

We select the top 15 features using Mutual Information and Recursive Feature Elimination to find the most efficient predictors."""))

nb.cells.append(nbf.v4.new_code_cell("""# 1. Mutual Information
print("Calculating Mutual Information...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
top_15_mi_cols = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False).head(15).index.tolist()

# 2. RFE
print("Performing RFE (Recursive Feature Elimination)...")
rfe_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_model, n_features_to_select=15)
rfe.fit(X_train, y_train)
top_15_rfe_cols = X_train.columns[rfe.support_].tolist()

print(f"MI Features: {top_15_mi_cols[:3]}...")
print(f"RFE Features: {top_15_rfe_cols[:3]}...")"""))

# Phase 5: Global Evaluation
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 5: Global Model Evaluation

We evaluate the models with the 3 different feature sets (Full, MI-15, RFE-15) and add Confusion Matrices for depth."""))

nb.cells.append(nbf.v4.new_code_cell("""def evaluate_model(model, xtr, xte, title):
    model.fit(xtr, y_train)
    preds = model.predict(xte)
    probs = model.predict_proba(xte)[:, 1] if hasattr(model, 'predict_proba') else preds
    
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_title(f"CM: {title}")
    
    fpr, tpr, _ = roc_curve(y_test, probs)
    ax2.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    ax2.plot([0,1], [0,1], 'r--')
    ax2.set_title(f"ROC: {title}")
    ax2.legend()
    plt.tight_layout()
    plt.show()
    
    return acc, f1, auc

results = []
models = {'Logistic Regression': LogisticRegression(max_iter=1000), 
          'Random Forest': RandomForestClassifier(n_estimators=100)}

exp_sets = {'Full': (X_train, X_test), 
            'MI-15': (X_train[top_15_mi_cols], X_test[top_15_mi_cols]),
            'RFE-15': (X_train[top_15_rfe_cols], X_test[top_15_rfe_cols])}

for m_name, m in models.items():
    for set_name, (xtr, xte) in exp_sets.items():
        acc, f1, auc = evaluate_model(m, xtr, xte, f"{m_name} ({set_name})")
        results.append({'Model': m_name, 'Set': set_name, 'Accuracy': acc, 'F1-Score': f1, 'AUC': auc})

res_df = pd.DataFrame(results).pivot(index='Model', columns='Set', values='Accuracy')
display(res_df)"""))

# Phase 6: Deep Learning Hybrid
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 6: Deep Learning (Hybrid Approach)

This model acts as a nonlinear classifier on the pre-extracted features. It incorporates Dropout to ensure generalization."""))

nb.cells.append(nbf.v4.new_code_cell("""mlp = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = mlp.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.2, verbose=1)

mlp_loss, mlp_acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"Deep Learning Accuracy: {mlp_acc:.4f}")"""))

with open('E:/projects/scientific_project/notebooks/cic_trap4phish_classification.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Refined Notebook generated successfully at E:/projects/scientific_project/notebooks/cic_trap4phish_classification.ipynb")
