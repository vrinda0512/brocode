"""
Advanced Ensemble Model Training
Combines Random Forest + XGBoost + Neural Network
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("🤖 CHAINGUARD: ENSEMBLE MODEL TRAINING")
print("=" * 80)

# ============================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================
print("\n📂 Step 1: Loading training data...")

# Load your existing dataset
df = pd.read_csv('data/bitcoin_transactions.csv')
print(f"✅ Loaded {len(df)} transactions")

# Prepare features and labels
X = df[['Amount', 'Num_Inputs', 'Num_Outputs', 'Fee']].values
y = df['Is_Fraud'].values

print(f"\n📊 Data shape:")
print(f"   Features (X): {X.shape}")
print(f"   Labels (y): {y.shape}")
print(f"   Fraud rate: {(y.sum() / len(y) * 100):.2f}%")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✅ Data split:")
print(f"   Training: {len(X_train)} samples")
print(f"   Testing: {len(X_test)} samples")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================
# STEP 2: TRAIN MODEL #1 - RANDOM FOREST
# ============================================
print("\n" + "=" * 80)
print("🌲 Step 2: Training Random Forest Classifier...")
print("=" * 80)

rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1,
    class_weight='balanced'
)

rf_model.fit(X_train_scaled, y_train)
rf_pred = rf_model.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"✅ Random Forest trained!")
print(f"   Accuracy: {rf_accuracy * 100:.2f}%")

# ============================================
# STEP 3: TRAIN MODEL #2 - XGBOOST
# ============================================
print("\n" + "=" * 80)
print("🚀 Step 3: Training XGBoost Classifier...")
print("=" * 80)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum()  # Handle imbalance
)

xgb_model.fit(X_train_scaled, y_train)
xgb_pred = xgb_model.predict(X_test_scaled)
xgb_accuracy = accuracy_score(y_test, xgb_pred)

print(f"✅ XGBoost trained!")
print(f"   Accuracy: {xgb_accuracy * 100:.2f}%")

# ============================================
# STEP 4: TRAIN MODEL #3 - NEURAL NETWORK
# ============================================
print("\n" + "=" * 80)
print("🧠 Step 4: Training Neural Network...")
print("=" * 80)

# Build neural network
nn_model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(4,)),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n🔄 Training neural network (this may take a minute)...")

# Train with validation split
history = nn_model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=0
)

nn_pred_proba = nn_model.predict(X_test_scaled, verbose=0)
nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
nn_accuracy = accuracy_score(y_test, nn_pred)

print(f"✅ Neural Network trained!")
print(f"   Accuracy: {nn_accuracy * 100:.2f}%")

# ============================================
# STEP 5: CREATE ENSEMBLE (VOTING CLASSIFIER)
# ============================================
print("\n" + "=" * 80)
print("🎯 Step 5: Creating Ensemble Model...")
print("=" * 80)

# For ensemble, we'll use a simple approach: average the predictions
def ensemble_predict(X):
    """Combine predictions from all 3 models"""
    rf_pred = rf_model.predict(X)
    xgb_pred = xgb_model.predict(X)
    nn_pred_proba = nn_model.predict(X, verbose=0)
    nn_pred = (nn_pred_proba > 0.5).astype(int).flatten()
    
    # Majority voting
    votes = rf_pred + xgb_pred + nn_pred
    ensemble_pred = (votes >= 2).astype(int)  # At least 2 out of 3 vote fraud
    
    return ensemble_pred

def ensemble_predict_proba(X):
    """Get probability predictions from ensemble"""
    rf_proba = rf_model.predict_proba(X)[:, 1]
    xgb_proba = xgb_model.predict_proba(X)[:, 1]
    nn_proba = nn_model.predict(X, verbose=0).flatten()
    
    # Average probabilities
    avg_proba = (rf_proba + xgb_proba + nn_proba) / 3
    
    return avg_proba

print("✅ Ensemble model created with 3 base models")
print("   • Random Forest")
print("   • XGBoost")
print("   • Neural Network")
print("   • Voting Strategy: Majority (2 out of 3)")

# Test ensemble
ensemble_pred = ensemble_predict(X_test_scaled)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)

print(f"\n✅ Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")

# ============================================
# STEP 6: COMPARE ALL MODELS
# ============================================
print("\n" + "=" * 80)
print("📊 PERFORMANCE COMPARISON")
print("=" * 80)

results = pd.DataFrame({
    'Model': ['Random Forest', 'XGBoost', 'Neural Network', '🏆 ENSEMBLE'],
    'Accuracy': [rf_accuracy, xgb_accuracy, nn_accuracy, ensemble_accuracy]
})

print("\n" + results.to_string(index=False))
print("\n" + "=" * 80)

best_model = results.loc[results['Accuracy'].idxmax(), 'Model']
best_accuracy = results['Accuracy'].max()
print(f"🏆 Best Model: {best_model} with {best_accuracy * 100:.2f}% accuracy")

# ============================================
# STEP 7: DETAILED CLASSIFICATION REPORT
# ============================================
print("\n" + "=" * 80)
print("📋 DETAILED ENSEMBLE PERFORMANCE")
print("=" * 80)

print("\nClassification Report:")
print(classification_report(y_test, ensemble_pred, target_names=['Normal', 'Fraudulent']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, ensemble_pred)
print(cm)
print(f"\nTrue Negatives:  {cm[0][0]}")
print(f"False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}")
print(f"True Positives:  {cm[1][1]}")

# ============================================
# STEP 8: SAVE MODELS
# ============================================
print("\n" + "=" * 80)
print("💾 Step 8: Saving Models...")
print("=" * 80)

import os
os.makedirs('models', exist_ok=True)

# Save individual models
joblib.dump(rf_model, 'models/rf_model.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
nn_model.save('models/nn_model.h5')
joblib.dump(scaler, 'models/ensemble_scaler.pkl')

print("✅ Saved:")
print("   • Random Forest: models/rf_model.pkl")
print("   • XGBoost: models/xgb_model.pkl")
print("   • Neural Network: models/nn_model.h5")
print("   • Scaler: models/ensemble_scaler.pkl")

# Save ensemble metadata
ensemble_info = {
    'rf_accuracy': rf_accuracy,
    'xgb_accuracy': xgb_accuracy,
    'nn_accuracy': nn_accuracy,
    'ensemble_accuracy': ensemble_accuracy,
    'test_size': len(X_test),
    'fraud_rate': y.sum() / len(y)
}

joblib.dump(ensemble_info, 'models/ensemble_info.pkl')
print("   • Ensemble Info: models/ensemble_info.pkl")

print("\n💡 To use ensemble in production, load all 3 models and use majority voting")

# ============================================
# STEP 9: VISUALIZATION
# ============================================
print("\n" + "=" * 80)
print("📊 Step 9: Creating Visualizations...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Model Comparison Bar Chart
axes[0, 0].bar(results['Model'], results['Accuracy'] * 100, 
               color=['#3498db', '#e74c3c', '#9b59b6', '#2ecc71'])
axes[0, 0].set_ylabel('Accuracy (%)')
axes[0, 0].set_title('Model Performance Comparison')
axes[0, 0].set_ylim([80, 100])
for i, v in enumerate(results['Accuracy'] * 100):
    axes[0, 0].text(i, v + 0.5, f'{v:.2f}%', ha='center')

# 2. Confusion Matrix Heatmap
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'], ax=axes[0, 1])
axes[0, 1].set_title('Ensemble Confusion Matrix')
axes[0, 1].set_ylabel('True Label')
axes[0, 1].set_xlabel('Predicted Label')

# 3. Feature Importance (from Random Forest)
feature_names = ['Amount', 'Num_Inputs', 'Num_Outputs', 'Fee']
importances = rf_model.feature_importances_
axes[1, 0].barh(feature_names, importances, color='#3498db')
axes[1, 0].set_xlabel('Importance')
axes[1, 0].set_title('Feature Importance (Random Forest)')

# 4. Neural Network Training History
axes[1, 1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_title('Neural Network Training History')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/ensemble_performance.png', dpi=300, bbox_inches='tight')
print("✅ Visualization saved: outputs/ensemble_performance.png")

plt.show()

print("\n" + "=" * 80)
print("✅ ENSEMBLE MODEL TRAINING COMPLETE!")
print("=" * 80)
print(f"\n🎯 Final Ensemble Accuracy: {ensemble_accuracy * 100:.2f}%")
print(f"📈 Improvement over single model: {(ensemble_accuracy - max(rf_accuracy, xgb_accuracy, nn_accuracy)) * 100:.2f}%")
print("\n💡 Next step: Integrate ensemble into your Flask app!")