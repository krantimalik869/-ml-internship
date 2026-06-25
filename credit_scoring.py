# Task 1: Credit Scoring Model
# Install: pip install scikit-learn pandas matplotlib seaborn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, f1_score)

# 1. Generate Data
X, y = make_classification(
    n_samples=1000, n_features=10,
    n_informative=6, n_redundant=2,
    random_state=42
)

feature_names = [
    'income', 'debt_amount', 'credit_history_years',
    'num_credit_cards', 'payment_history', 'age',
    'employment_years', 'loan_amount', 'monthly_expenses',
    'num_late_payments'
]
df = pd.DataFrame(X, columns=feature_names)
df['creditworthy'] = y

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['creditworthy'].value_counts())

# 2. Feature Engineering
df['debt_to_income'] = df['debt_amount'] / (df['income'] + 1e-5)
df['credit_utilization'] = df['loan_amount'] / (df['income'] + 1e-5)

features = feature_names + ['debt_to_income', 'credit_utilization']
X = df[features]
y = df['creditworthy']

# 3. Train/Test Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# 4. Train Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred  = model.predict(X_test_sc)
    y_proba = model.predict_proba(X_test_sc)[:, 1]

    results[name] = {
        'model':   model,
        'y_pred':  y_pred,
        'y_proba': y_proba,
        'f1':      f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
    }
    print(f"\n── {name} ──")
    print(classification_report(y_test, y_pred))

# 5. Plot ROC Curves
plt.figure(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={r['roc_auc']:.2f})")
plt.plot([0,1],[0,1],'k--', label='Random Classifier')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend()
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150)
plt.show()

# 6. Confusion Matrix
best_name = max(results, key=lambda n: results[n]['roc_auc'])
best_cm   = confusion_matrix(y_test, results[best_name]['y_pred'])

plt.figure(figsize=(5, 4))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Purples',
            xticklabels=['Not Creditworthy','Creditworthy'],
            yticklabels=['Not Creditworthy','Creditworthy'])
plt.title(f'Confusion Matrix — {best_name}')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

# 7. Feature Importance
rf = results['Random Forest']['model']
importances = pd.Series(rf.feature_importances_, index=features)
importances.sort_values().plot(kind='barh', figsize=(7, 5), color='#7F77DD')
plt.title('Feature Importances — Random Forest')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

print("\nBest model:", best_name)
print(f"ROC-AUC: {results[best_name]['roc_auc']:.4f}")
print(f"F1-Score: {results[best_name]['f1']:.4f}")