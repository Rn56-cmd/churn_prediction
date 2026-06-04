import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

# Step 1 - Load Data

df = pd.read_csv('Telco-Customer-Churn.csv')
print(f"Shape: {df.shape}")
print(df.head())
print(f"\nChurn distribution:\n{df['Churn'].value_counts()}")

# Step 2 - Preprocessing

# Fix TotalCharges (stored as string, has spaces)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

# Drop customer ID
df = df.drop('customerID', axis=1)

# Encode Target
df['Churn'] = df['Churn'].map({'Yes':1, 'No':0})

print(f"Missing values after fix:\n{df.isnull().sum().sum()}")
print(f"\nChurn rate: {df['Churn'].mean():.1%}")


# Step 3 - Split Features and Target

X = df.drop('Churn', axis=1)
y = df['Churn']

numeric_cols = ['SeniorCitizen','tenure','MonthlyCharges', 'TotalCharges']
cat_cols = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# Step 4 - Build Pipeline

num_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_pipe, numeric_cols),
    ('cat', cat_pipe, cat_cols)
])

#Step 5 - Train3 Models & Compare

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
}

results = {}
best_score = 0
best_name = ''
best_pipeline = None

for name, clf in models.items():
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    results[name] = {
        'Accuracy': acc, 'F1': f1, 'AUC': auc, 'CM': cm,
        'pipeline': pipeline
    }

    print(f"\n{name}")
    print(f" Accuracy : {acc:.4f}")
    print(f" F1 Score : {f1:.4f}")
    print(f" AUC-ROC : {auc:.4f}")
    print(f" Confusion Matrix:\n{cm}")

    if auc > best_score:
        best_score = auc
        best_name = name
        best_pipeline = pipeline

print(f"\n Best Model: {best_name} (AUC: {best_score:.4f})")

# Step 6 - Feature Importance (random forest)

rf_pipeline = results['Random Forest']['pipeline']
rf_model = rf_pipeline.named_steps['classifier']
ohe_cols = rf_pipeline.named_steps['preprocessor']\
            .named_transformers_['cat']\
            .named_steps['onehot']\
            .get_feature_names_out(cat_cols).tolist()

feature_names = numeric_cols + ohe_cols

importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False).head(15)

print("Top 15 most important features:")
print(importance.to_string())


# Step 7 - Save Model

joblib.dump({
    'model': best_pipeline,
    'model_name': best_name,
    'results': {k: {m: v for m, v in r.items() if m != 'pipeline'}
                for k, r in results.items()},
    'feature_importance': importance,
    'numeric_cols': numeric_cols,
    'cat_cols': cat_cols,
}, 'churn_model.pkl')

print(f" Best_model: {best_name}")
print(f" AUC-ROC: {best_score:.4f}")