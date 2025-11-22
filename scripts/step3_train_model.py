import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

# ============================================
# SET UP FOLDER PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

print("=" * 70)
print("CHAINGUARD - STEP 3: TRAINING THE AI FRAUD DETECTOR")
print("=" * 70)
print(f"\n📁 Working directory: {BASE_DIR}")

# ============================================
# PART 1: LOAD THE PREPARED DATA
# ============================================
print("\n📂 Loading preprocessed data from Step 2...")

X = np.load(os.path.join(PROCESSED_DIR, 'X_data.npy'))
y = np.load(os.path.join(PROCESSED_DIR, 'y_data.npy'))

print(f"✅ Loaded data successfully!")
print(f"   Features (X): {X.shape}")
print(f"   Labels (y): {y.shape}")

# ============================================
# PART 2: SPLIT INTO TRAINING & TESTING SETS
# ============================================
print("\n" + "=" * 70)
print("📊 SPLITTING DATA INTO TRAIN & TEST SETS")
print("=" * 70)

# 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3,      # 30% for testing
    random_state=42,    # Makes results reproducible
    stratify=y          # Keeps fraud ratio same in train/test
)

print(f"\n✅ Data split complete!")
print(f"   Training set: {X_train.shape[0]} transactions ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"   Testing set: {X_test.shape[0]} transactions ({X_test.shape[0]/len(X)*100:.1f}%)")

# Check class distribution
train_fraud = (y_train == 1).sum()
test_fraud = (y_test == 1).sum()
print(f"\n   Fraud in training: {train_fraud} ({train_fraud/len(y_train)*100:.1f}%)")
print(f"   Fraud in testing: {test_fraud} ({test_fraud/len(y_test)*100:.1f}%)")

# ============================================
# PART 3: SCALE THE FEATURES
# ============================================
print("\n" + "=" * 70)
print("⚖️  SCALING FEATURES (Normalizing data)")
print("=" * 70)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Features scaled! (This helps the AI learn better)")

# ============================================
# PART 4: TRAIN MODEL #1 - RANDOM FOREST
# ============================================
print("\n" + "=" * 70)
print("🌲 TRAINING MODEL #1: RANDOM FOREST")
print("=" * 70)

print("\n⏳ Training Random Forest... (this takes 30-60 seconds)")

rf_model = RandomForestClassifier(
    n_estimators=100,        # 100 decision trees
    max_depth=20,            # How deep each tree can go
    random_state=42,
    n_jobs=-1,               # Use all CPU cores (faster!)
    class_weight='balanced'  # Handle imbalanced data
)

rf_model.fit(X_train_scaled, y_train)

print("✅ Random Forest trained!")

# Predict on test set
rf_pred = rf_model.predict(X_test_scaled)
rf_proba = rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of fraud

# Evaluate
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_precision = precision_score(y_test, rf_pred)
rf_recall = recall_score(y_test, rf_pred)
rf_f1 = f1_score(y_test, rf_pred)

print(f"\n📊 Random Forest Performance:")
print(f"   Accuracy:  {rf_accuracy*100:.2f}%")
print(f"   Precision: {rf_precision*100:.2f}% (How many flagged transactions are actually fraud)")
print(f"   Recall:    {rf_recall*100:.2f}% (How many frauds we caught)")
print(f"   F1-Score:  {rf_f1*100:.2f}% (Overall balance)")

# ============================================
# PART 5: TRAIN MODEL #2 - XGBOOST (BEST!)
# ============================================
print("\n" + "=" * 70)
print("🚀 TRAINING MODEL #2: XGBOOST (The Champion!)")
print("=" * 70)

print("\n⏳ Training XGBoost... (this takes 30-60 seconds)")

# Calculate scale_pos_weight to handle imbalance
scale_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_weight,  # Handle imbalanced data
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)

print("✅ XGBoost trained!")

# Predict on test set
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]

# Evaluate
xgb_accuracy = accuracy_score(y_test, xgb_pred)
xgb_precision = precision_score(y_test, xgb_pred)
xgb_recall = recall_score(y_test, xgb_pred)
xgb_f1 = f1_score(y_test, xgb_pred)

print(f"\n📊 XGBoost Performance:")
print(f"   Accuracy:  {xgb_accuracy*100:.2f}%")
print(f"   Precision: {xgb_precision*100:.2f}% (How many flagged transactions are actually fraud)")
print(f"   Recall:    {xgb_recall*100:.2f}% (How many frauds we caught)")
print(f"   F1-Score:  {xgb_f1*100:.2f}% (Overall balance)")

# ============================================
# PART 6: COMPARE MODELS
# ============================================
print("\n" + "=" * 70)
print("🏆 MODEL COMPARISON")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost'],
    'Accuracy': [rf_accuracy*100, xgb_accuracy*100],
    'Precision': [rf_precision*100, xgb_precision*100],
    'Recall': [rf_recall*100, xgb_recall*100],
    'F1-Score': [rf_f1*100, xgb_f1*100]
})

print("\n", comparison.to_string(index=False))

# Pick the best model (higher F1-score)
if xgb_f1 >= rf_f1:
    best_model = xgb_model
    best_name = "XGBoost"
    best_proba = xgb_proba
    best_pred = xgb_pred
    best_f1 = xgb_f1
    best_precision = xgb_precision
else:
    best_model = rf_model
    best_name = "Random Forest"
    best_proba = rf_proba
    best_pred = rf_pred
    best_f1 = rf_f1
    best_precision = rf_precision

print(f"\n🏆 WINNER: {best_name} (F1-Score: {best_f1*100:.2f}%)")

# ============================================
# PART 7: CREATE VISUALIZATIONS
# ============================================
print("\n" + "=" * 70)
print("📊 CREATING PERFORMANCE VISUALIZATIONS")
print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'ChainGuard: {best_name} Model Performance', fontsize=16, fontweight='bold')

# Plot 1: Confusion Matrix
ax1 = axes[0, 0]
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
ax1.set_title('Confusion Matrix')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')
ax1.set_xticklabels(['Normal', 'Fraud'])
ax1.set_yticklabels(['Normal', 'Fraud'])

# Plot 2: Model Comparison
ax2 = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
rf_scores = [rf_accuracy*100, rf_precision*100, rf_recall*100, rf_f1*100]
xgb_scores = [xgb_accuracy*100, xgb_precision*100, xgb_recall*100, xgb_f1*100]

x = np.arange(len(metrics))
width = 0.35
ax2.bar(x - width/2, rf_scores, width, label='Random Forest', color='skyblue')
ax2.bar(x + width/2, xgb_scores, width, label='XGBoost', color='orange')
ax2.set_ylabel('Score (%)')
ax2.set_title('Model Comparison')
ax2.set_xticks(x)
ax2.set_xticklabels(metrics, rotation=45)
ax2.legend()
ax2.set_ylim(0, 105)

# Plot 3: Risk Score Distribution
ax3 = axes[1, 0]
risk_scores = best_proba * 100  # Convert to 0-100 scale
ax3.hist(risk_scores[y_test == 0], bins=50, alpha=0.5, label='Normal', color='green')
ax3.hist(risk_scores[y_test == 1], bins=50, alpha=0.5, label='Fraud', color='red')
ax3.set_xlabel('Risk Score (0-100)')
ax3.set_ylabel('Frequency')
ax3.set_title('Risk Score Distribution')
ax3.legend()

# Plot 4: Precision-Recall by Threshold
ax4 = axes[1, 1]
thresholds = np.linspace(0, 1, 100)
precisions = []
recalls = []

for thresh in thresholds:
    pred_at_thresh = (best_proba >= thresh).astype(int)
    if pred_at_thresh.sum() > 0:
        prec = precision_score(y_test, pred_at_thresh, zero_division=0)
        rec = recall_score(y_test, pred_at_thresh, zero_division=0)
        precisions.append(prec * 100)
        recalls.append(rec * 100)
    else:
        precisions.append(0)
        recalls.append(0)

ax4.plot(thresholds * 100, precisions, label='Precision', color='blue')
ax4.plot(thresholds * 100, recalls, label='Recall', color='orange')
ax4.set_xlabel('Risk Score Threshold')
ax4.set_ylabel('Score (%)')
ax4.set_title('Precision vs Recall Trade-off')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save to visualizations/ folder
viz_path = os.path.join(VIZ_DIR, 'model_performance.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: {viz_path}")

# ============================================
# PART 8: SAVE THE TRAINED MODEL
# ============================================
print("\n" + "=" * 70)
print("💾 SAVING TRAINED MODEL")
print("=" * 70)

# Save the best model to models/ folder
model_path = os.path.join(MODELS_DIR, 'chainguard_model.pkl')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)

# Save test data to processed_data/ folder
X_test_path = os.path.join(PROCESSED_DIR, 'X_test.npy')
y_test_path = os.path.join(PROCESSED_DIR, 'y_test.npy')
risk_path = os.path.join(PROCESSED_DIR, 'risk_scores.npy')

np.save(X_test_path, X_test_scaled)
np.save(y_test_path, y_test)
np.save(risk_path, best_proba)

print("✅ Saved files:")
print(f"   → {model_path}")
print(f"   → {scaler_path}")
print(f"   → {X_test_path}")
print(f"   → {y_test_path}")
print(f"   → {risk_path}")

# ============================================
# PART 9: DETAILED CLASSIFICATION REPORT
# ============================================
print("\n" + "=" * 70)
print("📋 DETAILED CLASSIFICATION REPORT")
print("=" * 70)

print("\n", classification_report(y_test, best_pred, target_names=['Normal', 'Fraud']))

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "=" * 70)
print("🎉 STEP 3 COMPLETE!")
print("=" * 70)

print(f"""
✅ Model Training: Complete
✅ Model Evaluation: Complete
✅ Visualization: Complete
✅ Model Saved: Complete

🏆 Best Model: {best_name}
📊 Performance Metrics:
   • Accuracy:  {best_f1*100:.2f}%
   • Precision: {best_precision*100:.2f}% (Minimize false alarms!)
   • F1-Score:  {best_f1*100:.2f}%

📁 Generated Files:
   • models/chainguard_model.pkl - Your trained AI!
   • models/scaler.pkl - Data preprocessor
   • visualizations/model_performance.png - Performance charts
   • processed_data/ - Test data files for Step 4

✅ Your AI can now detect fraud! Ready for Step 4 (Risk Scoring System)
""")

print("=" * 70)