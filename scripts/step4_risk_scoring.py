import numpy as np
import pandas as pd
import joblib
import hashlib
import os
from datetime import datetime

# ============================================
# SET UP FOLDER PATHS
# ============================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

print("=" * 70)
print("CHAINGUARD - STEP 4: RISK SCORING & ALERT SYSTEM")
print("=" * 70)
print(f"\n📁 Working directory: {BASE_DIR}")

# ============================================
# PART 1: LOAD TRAINED MODEL & TEST DATA
# ============================================
print("\n📂 Loading trained model and test data...")

# Load the trained model and scaler
model = joblib.load(os.path.join(MODELS_DIR, 'chainguard_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# Load test data
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
risk_scores = np.load(os.path.join(PROCESSED_DIR, 'risk_scores.npy'))

print(f"✅ Loaded successfully!")
print(f"   Test transactions: {len(X_test)}")
print(f"   Model type: {type(model).__name__}")

# ============================================
# PART 2: RISK SCORING FUNCTIONS
# ============================================
print("\n" + "=" * 70)
print("🎯 CREATING RISK SCORING SYSTEM")
print("=" * 70)

def calculate_risk_score(probability):
    """
    Convert model probability (0-1) to risk score (0-100)
    
    Args:
        probability: Float between 0 and 1
    
    Returns:
        Integer risk score between 0 and 100
    """
    return int(probability * 100)

def generate_alert(risk_score):
    """
    Generate security alert based on risk score
    
    Args:
        risk_score: Integer between 0-100
    
    Returns:
        Dictionary with alert level, message, and action
    """
    if risk_score >= 90:
        return {
            'level': 'CRITICAL',
            'color': '🔴',
            'message': 'IMMEDIATE ACTION REQUIRED',
            'action': 'Freeze wallet immediately and initiate investigation',
            'priority': 1
        }
    elif risk_score >= 75:
        return {
            'level': 'HIGH',
            'color': '🟠',
            'message': 'HIGH RISK DETECTED',
            'action': 'Flag wallet for investigation and add to watchlist',
            'priority': 2
        }
    elif risk_score >= 50:
        return {
            'level': 'MEDIUM',
            'color': '🟡',
            'message': 'SUSPICIOUS ACTIVITY',
            'action': 'Add sender to monitoring watchlist',
            'priority': 3
        }
    elif risk_score >= 25:
        return {
            'level': 'LOW',
            'color': '🟢',
            'message': 'MINOR ANOMALY DETECTED',
            'action': 'Log transaction for routine review',
            'priority': 4
        }
    else:
        return {
            'level': 'NORMAL',
            'color': '✅',
            'message': 'TRANSACTION APPEARS NORMAL',
            'action': 'No action required - continue monitoring',
            'priority': 5
        }

def obfuscate_transaction_id(tx_id):
    """
    BONUS: Privacy-preserving obfuscation using SHA-256 hashing
    
    Args:
        tx_id: Transaction identifier
    
    Returns:
        Obfuscated hash string (first 16 characters)
    """
    hash_object = hashlib.sha256(str(tx_id).encode())
    return hash_object.hexdigest()[:16]

print("✅ Risk scoring functions created!")
print("   • calculate_risk_score() - Converts probability to 0-100 scale")
print("   • generate_alert() - Maps risk to actionable alerts")
print("   • obfuscate_transaction_id() - Privacy-preserving hashing")

# ============================================
# PART 3: ANALYZE ALL TEST TRANSACTIONS
# ============================================
print("\n" + "=" * 70)
print("🔍 ANALYZING ALL TEST TRANSACTIONS")
print("=" * 70)

print(f"\n⏳ Processing {len(X_test)} transactions...")

# Create results dataframe
results = []

for i in range(len(X_test)):
    # Get prediction and probability
    prediction = model.predict([X_test[i]])[0]
    probability = risk_scores[i]
    
    # Calculate risk score
    risk_score = calculate_risk_score(probability)
    
    # Generate alert
    alert = generate_alert(risk_score)
    
    # Obfuscate transaction ID
    obfuscated_id = obfuscate_transaction_id(i)
    
    # Store results
    results.append({
        'Transaction_ID': i,
        'Obfuscated_ID': obfuscated_id,
        'True_Label': 'Fraud' if y_test[i] == 1 else 'Normal',
        'Predicted_Label': 'Fraud' if prediction == 1 else 'Normal',
        'Risk_Score': risk_score,
        'Alert_Level': alert['level'],
        'Alert_Message': alert['message'],
        'Recommended_Action': alert['action'],
        'Priority': alert['priority'],
        'Correct_Prediction': '✅' if prediction == y_test[i] else '❌'
    })

# Convert to DataFrame
df_results = pd.DataFrame(results)

print(f"✅ Analysis complete!")
print(f"   Total transactions analyzed: {len(df_results)}")
print(f"   Correct predictions: {(df_results['Correct_Prediction'] == '✅').sum()}")
print(f"   Accuracy: {(df_results['Correct_Prediction'] == '✅').sum() / len(df_results) * 100:.2f}%")

# ============================================
# PART 4: IDENTIFY TOP 10 RISKIEST TRANSACTIONS
# ============================================
print("\n" + "=" * 70)
print("🚨 TOP 10 RISKIEST TRANSACTIONS")
print("=" * 70)

# Get top 10 by risk score
top_10_risky = df_results.nlargest(10, 'Risk_Score')

print("\n" + "=" * 70)
for idx, row in top_10_risky.iterrows():
    alert = generate_alert(row['Risk_Score'])
    print(f"\n{alert['color']} TRANSACTION #{row['Transaction_ID']}")
    print(f"   Obfuscated ID: {row['Obfuscated_ID']}")
    print(f"   Risk Score: {row['Risk_Score']}/100")
    print(f"   Alert Level: {row['Alert_Level']}")
    print(f"   True Status: {row['True_Label']}")
    print(f"   Predicted: {row['Predicted_Label']} {row['Correct_Prediction']}")
    print(f"   Action: {row['Recommended_Action']}")
    print("-" * 70)

# ============================================
# PART 5: RISK DISTRIBUTION ANALYSIS
# ============================================
print("\n" + "=" * 70)
print("📊 RISK DISTRIBUTION ANALYSIS")
print("=" * 70)

# Count alerts by level
alert_distribution = df_results['Alert_Level'].value_counts().sort_index()

print("\n🎯 Alert Level Distribution:")
for level, count in alert_distribution.items():
    pct = (count / len(df_results)) * 100
    print(f"   {level:12s}: {count:5d} ({pct:5.2f}%)")

# Analyze by true label
print("\n📈 Risk Score Statistics by True Label:")
for label in ['Normal', 'Fraud']:
    subset = df_results[df_results['True_Label'] == label]
    print(f"\n   {label} Transactions:")
    print(f"      Count: {len(subset)}")
    print(f"      Avg Risk Score: {subset['Risk_Score'].mean():.2f}")
    print(f"      Max Risk Score: {subset['Risk_Score'].max()}")
    print(f"      Min Risk Score: {subset['Risk_Score'].min()}")

# ============================================
# PART 6: SAVE REPORTS
# ============================================
print("\n" + "=" * 70)
print("💾 SAVING REPORTS")
print("=" * 70)

# Save top 10 riskiest transactions
top_10_path = os.path.join(OUTPUTS_DIR, 'top_10_risky_transactions.csv')
top_10_risky.to_csv(top_10_path, index=False)
print(f"✅ Saved: {top_10_path}")

# Save full results
full_results_path = os.path.join(OUTPUTS_DIR, 'all_transaction_results.csv')
df_results.to_csv(full_results_path, index=False)
print(f"✅ Saved: {full_results_path}")

# Save alert summary
alert_summary = df_results.groupby('Alert_Level').agg({
    'Transaction_ID': 'count',
    'Risk_Score': ['mean', 'min', 'max']
}).round(2)
alert_summary_path = os.path.join(OUTPUTS_DIR, 'alert_summary.csv')
alert_summary.to_csv(alert_summary_path)
print(f"✅ Saved: {alert_summary_path}")

# ============================================
# PART 7: CREATE ACTIONABLE SECURITY REPORT
# ============================================
print("\n" + "=" * 70)
print("📋 GENERATING SECURITY REPORT")
print("=" * 70)

report_path = os.path.join(OUTPUTS_DIR, 'security_report.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 70 + "\n")
    f.write("CHAINGUARD SECURITY REPORT\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("EXECUTIVE SUMMARY\n")
    f.write("-" * 70 + "\n")
    f.write(f"Total Transactions Analyzed: {len(df_results):,}\n")
    f.write(f"Model Accuracy: {(df_results['Correct_Prediction'] == '✅').sum() / len(df_results) * 100:.2f}%\n")
    f.write(f"High-Risk Transactions Detected: {(df_results['Alert_Level'].isin(['CRITICAL', 'HIGH'])).sum()}\n\n")
    
    f.write("ALERT DISTRIBUTION\n")
    f.write("-" * 70 + "\n")
    for level, count in alert_distribution.items():
        pct = (count / len(df_results)) * 100
        f.write(f"{level:12s}: {count:5d} ({pct:5.2f}%)\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("TOP 10 HIGHEST RISK TRANSACTIONS\n")
    f.write("=" * 70 + "\n\n")
    
    for idx, row in top_10_risky.iterrows():
        f.write(f"Transaction #{row['Transaction_ID']}\n")
        f.write(f"  Obfuscated ID: {row['Obfuscated_ID']}\n")
        f.write(f"  Risk Score: {row['Risk_Score']}/100\n")
        f.write(f"  Alert Level: {row['Alert_Level']}\n")
        f.write(f"  Status: {row['True_Label']} | Predicted: {row['Predicted_Label']}\n")
        f.write(f"  Recommended Action: {row['Recommended_Action']}\n")
        f.write("-" * 70 + "\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("=" * 70 + "\n")
    f.write("1. Immediately investigate all CRITICAL and HIGH risk transactions\n")
    f.write("2. Add flagged wallets to monitoring watchlist\n")
    f.write("3. Review transaction patterns for MEDIUM risk alerts\n")
    f.write("4. Implement automated freezing for CRITICAL risk scores (>90)\n")
    f.write("5. Conduct weekly reviews of watchlist transactions\n")

print(f"✅ Saved: {report_path}")

# ============================================
# PART 8: DEMO - SINGLE TRANSACTION PREDICTION
# ============================================
print("\n" + "=" * 70)
print("🎬 DEMO: REAL-TIME TRANSACTION ANALYSIS")
print("=" * 70)

def predict_transaction(transaction_features):
    """
    Complete pipeline for predicting a single transaction
    
    Args:
        transaction_features: Raw transaction features (166 features)
    
    Returns:
        Dictionary with prediction results
    """
    # Make prediction
    prediction = model.predict([transaction_features])[0]
    probability = model.predict_proba([transaction_features])[0][1]
    
    # Calculate risk score
    risk_score = calculate_risk_score(probability)
    
    # Generate alert
    alert = generate_alert(risk_score)
    
    return {
        'prediction': 'Fraud' if prediction == 1 else 'Normal',
        'risk_score': risk_score,
        'alert_level': alert['level'],
        'alert_message': alert['message'],
        'recommended_action': alert['action'],
        'alert_color': alert['color']
    }

# Demo with 3 sample transactions
print("\n📝 Sample Transaction Analysis:\n")

for i in range(3):
    result = predict_transaction(X_test[i])
    
    print(f"{result['alert_color']} Transaction #{i+1}")
    print(f"   Status: {result['prediction']}")
    print(f"   Risk Score: {result['risk_score']}/100")
    print(f"   Alert: {result['alert_level']} - {result['alert_message']}")
    print(f"   Action: {result['recommended_action']}")
    print()

# ============================================
# FINAL SUMMARY
# ============================================
print("=" * 70)
print("🎉 STEP 4 COMPLETE!")
print("=" * 70)

print(f"""
✅ Risk Scoring System: Complete
✅ Alert Generation: Complete
✅ Privacy Obfuscation: Complete (SHA-256 hashing)
✅ Reports Generated: Complete

📊 Results Summary:
   • Transactions Analyzed: {len(df_results):,}
   • Critical Alerts: {(df_results['Alert_Level'] == 'CRITICAL').sum()}
   • High Risk Alerts: {(df_results['Alert_Level'] == 'HIGH').sum()}
   • Model Accuracy: {(df_results['Correct_Prediction'] == '✅').sum() / len(df_results) * 100:.2f}%

📁 Generated Files:
   • outputs/top_10_risky_transactions.csv
   • outputs/all_transaction_results.csv
   • outputs/alert_summary.csv
   • outputs/security_report.txt

✅ Your fraud detection system is FULLY OPERATIONAL!
   Ready for Step 5 (Dashboard & Final Demo)
""")

print("=" * 70)