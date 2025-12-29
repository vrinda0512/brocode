import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

rf = joblib.load("models/random_forest.pkl")
xgb = joblib.load("models/xgboost.pkl")
lr = joblib.load("models/log_reg.pkl")
X_test = joblib.load("models/X_test.joblib")
y_test = joblib.load("models/y_test.joblib")

for name, m in [("RandomForest", rf), ("XGBoost", xgb), ("LogReg", lr)]:
    preds = m.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"{name} Accuracy: {acc:.4f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, digits=3))
    print("-"*50)