import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Load dataset
df = pd.read_csv('../data/diabetes.csv')
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define pipelines with scaling
pipe_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(max_iter=1000))
])

pipe_rf = Pipeline([
    ('scaler', StandardScaler()),  # RF often does not need scaling but keep for consistent pipeline
    ('rf', RandomForestClassifier())
])

# Hyperparameter grids
param_grid_lr = {
    'logreg__C': [0.01, 0.1, 1, 10, 100],
    'logreg__solver': ['lbfgs']
}

param_grid_rf = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 5, 10],
    'rf__min_samples_split': [2, 5]
}

# GridSearch for Logistic Regression
grid_lr = GridSearchCV(pipe_lr, param_grid_lr, cv=5, scoring='accuracy', n_jobs=-1)
grid_lr.fit(X_train, y_train)

# GridSearch for Random Forest
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1)
grid_rf.fit(X_train, y_train)

# Evaluate best Logistic Regression
y_pred_lr = grid_lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
print(f"LogisticRegression Best Params: {grid_lr.best_params_}")
print(f"LogisticRegression - Accuracy: {acc_lr:.4f}, Precision: {precision_score(y_test, y_pred_lr):.4f}, Recall: {recall_score(y_test, y_pred_lr):.4f}, F1: {f1_score(y_test, y_pred_lr):.4f}")

# Evaluate best Random Forest
y_pred_rf = grid_rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"RandomForest Best Params: {grid_rf.best_params_}")
print(f"RandomForest - Accuracy: {acc_rf:.4f}, Precision: {precision_score(y_test, y_pred_rf):.4f}, Recall: {recall_score(y_test, y_pred_rf):.4f}, F1: {f1_score(y_test, y_pred_rf):.4f}")

# Choose best model
if acc_lr > acc_rf:
    best_model = grid_lr.best_estimator_
    best_score = acc_lr
    print("Selected Logistic Regression as best model")
else:
    best_model = grid_rf.best_estimator_
    best_score = acc_rf
    print("Selected Random Forest as best model")

# Save best model
joblib.dump(best_model, '../backend/diabetes_model.pkl')
print(f"Best model saved with accuracy {best_score:.4f}")
