import nbformat as nbf

nb = nbf.v4.new_notebook()

# Phase 0: Imports
nb.cells.append(nbf.v4.new_markdown_cell("""# CIC Trap4Phish Multi-File Phishing Detection

This notebook analyzes phishing document features extracted from multiple file types (HTML, PDF, Word, Excel) to build robust classifiers. It compares model performance across individual file types and evaluates a global model using feature selection techniques like Mutual Information (MI) and Recursive Feature Elimination (RFE)."""))

nb.cells.append(nbf.v4.new_code_cell("""# Standard data processing
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn for classical ML & feature selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
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

# Phase 1: Data Integration
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 1: Data Integration & Preprocessing

**Project Requirement:** Features extracted from different file types.
We load the 4 specific CSV datasets provided from the Google Drive and concatenate them into a global analytical dataset."""))

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

# Combine all datasets into a single global dataframe
data = pd.concat([excel, word, pdf, html], ignore_index=True)

# Handle features that do not exist across all types by filling nulls with 0
data.fillna(0, inplace=True)

print(f"Global dataset shape: {data.shape}")
display(data['file_type'].value_counts())
display(data.head())"""))

# Phase 2: Performance Per File Type
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 2: Performance per File Type

**Project Requirement:** Compare features from different file types.
We iterate through each file type individually to evaluate a baseline Random Forest classifier."""))

nb.cells.append(nbf.v4.new_code_cell("""# Drop non-numeric identifier columns like file_path or URL for modeling
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
if 'label' in numeric_cols:
    numeric_cols.remove('label')

results_per_type = []

print("--- Performance Per File Type ---")
for ftype in data['file_type'].unique():
    subset = data[data['file_type'] == ftype]
    
    # Prepare X and y
    X_sub = subset[numeric_cols]
    y_sub = subset['label']
    
    # Train/Test Split (ensuring no leakage)
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(
        X_sub, y_sub, test_size=0.2, random_state=42, stratify=y_sub
    )
    
    # Train
    clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    clf.fit(X_train_sub, y_train_sub)
    
    # Evaluate
    preds = clf.predict(X_test_sub)
    acc = accuracy_score(y_test_sub, preds)
    f1 = f1_score(y_test_sub, preds, average='macro')
    
    results_per_type.append({'File Type': ftype, 'Accuracy': acc, 'F1-Score': f1})
    print(f"[{ftype.upper()}] Accuracy: {acc:.4f} | F1-Score: {f1:.4f}")

# Visualize Performance per File Type
res_df = pd.DataFrame(results_per_type)
plt.figure(figsize=(8, 5))
sns.barplot(data=res_df, x='File Type', y='Accuracy', palette='viridis')
plt.title('Baseline Accuracy per File Type')
plt.ylim(0, 1.05)
plt.show()"""))

# Phase 3: Global Model Setup
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 3: Global Model & Strict Train/Test Split

We define the overall target variables and perform a strict global `train_test_split` to ensure zero data leakage before any feature selection happens."""))

nb.cells.append(nbf.v4.new_code_cell("""# Extract target and numeric global features
X = data[numeric_cols]
y = data['label']

# Standardize features for logistic regression and NN
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

# Strictly split before applying any selection tools
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")"""))

# Phase 4: Feature Selection & 3-Way Comparison
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 4: Feature Selection (MI vs RFE) & Model Comparison

We will run three experiments across models:
1. **Full Features:** All extracted features across all documents.
2. **Mutual Information (MI):** Selecting the Top 15 features providing the highest information gain.
3. **Recursive Feature Elimination (RFE):** Selecting the Top 15 optimal features natively using a RandomForest estimator."""))

nb.cells.append(nbf.v4.new_code_cell("""
print("1. Calculating Mutual Information...")
mi_scores = mutual_info_classif(X_train, y_train, random_state=42)
mi_series = pd.Series(mi_scores, index=X_train.columns).sort_values(ascending=False)
top_15_mi_cols = mi_series.head(15).index.tolist()
print(f"Top 15 MI Features: {top_15_mi_cols[:5]}...")

print("2. Performing RFE (Recursive Feature Elimination)...")
rfe_model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rfe_model, n_features_to_select=15)
rfe.fit(X_train, y_train)
top_15_rfe_cols = X_train.columns[rfe.support_].tolist()
print(f"Top 15 RFE Features: {top_15_rfe_cols[:5]}...")
"""))

nb.cells.append(nbf.v4.new_code_cell("""
# Filter the datasets for our 3 test sets
X_train_mi, X_test_mi = X_train[top_15_mi_cols], X_test[top_15_mi_cols]
X_train_rfe, X_test_rfe = X_train[top_15_rfe_cols], X_test[top_15_rfe_cols]

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

datasets = {
    'Full Features': (X_train, X_test),
    'MI (Top 15)': (X_train_mi, X_test_mi),
    'RFE (Top 15)': (X_train_rfe, X_test_rfe)
}

comparison_results = []

for model_name, model in models.items():
    for dataset_name, (xtr, xte) in datasets.items():
        # Train
        model.fit(xtr, y_train)
        
        # Test
        preds = model.predict(xte)
        acc = accuracy_score(y_test, preds)
        
        comparison_results.append({
            'Model': model_name,
            'Dataset': dataset_name,
            'Accuracy': acc
        })

comp_df = pd.DataFrame(comparison_results)
comp_pivot = comp_df.pivot(index='Model', columns='Dataset', values='Accuracy')

print("--- Feature Selection Comparison Results ---")
display(comp_pivot)

plt.figure(figsize=(10, 6))
sns.barplot(data=comp_df, x='Model', y='Accuracy', hue='Dataset', palette='twilight')
plt.title("Model Performance: Full vs. MI vs. RFE Selection")
plt.ylim(0, 1.05)
plt.tight_layout()
plt.show()
"""))

# Phase 5: Deep Learning
nb.cells.append(nbf.v4.new_markdown_cell("""## Phase 5: Deep Learning on Structured Multi-File Data

*Deep learning operates on structured features rather than raw file content, thus representing a hybrid approach rather than pure feature learning.*

We use a standard Multi-layer Perceptron (MLP) architecture against the Full global feature set."""))

nb.cells.append(nbf.v4.new_code_cell("""
# Define MLP architecture
mlp = Sequential([
    Dense(64, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train MLP
history = mlp.fit(X_train, y_train, epochs=25, batch_size=64, validation_split=0.2, verbose=1)

# Evaluate
mlp_loss, mlp_acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"\\nDeep Learning (MLP) Full Features Accuracy: {mlp_acc:.4f}")
"""))

with open('E:/projects/scientific_project/notebooks/cic_trap4phish_classification.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated successfully at E:/projects/scientific_project/notebooks/cic_trap4phish_classification.ipynb")
