"""
Generate synthetic Bitcoin transaction data for training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

print("=" * 80)
print("📊 GENERATING TRAINING DATA")
print("=" * 80)

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# ============================================
# CONFIGURATION
# ============================================
NUM_NORMAL = 8000      # Normal transactions
NUM_FRAUD = 2000       # Fraudulent transactions
TOTAL = NUM_NORMAL + NUM_FRAUD

print(f"\n🎯 Generating {TOTAL} transactions:")
print(f"   • Normal: {NUM_NORMAL}")
print(f"   • Fraudulent: {NUM_FRAUD}")

# ============================================
# GENERATE NORMAL TRANSACTIONS
# ============================================
print("\n🟢 Generating normal transactions...")

normal_data = {
    'Amount': np.random.lognormal(mean=1.0, sigma=1.5, size=NUM_NORMAL),  # 0.1 - 50 BTC
    'Num_Inputs': np.random.randint(1, 6, size=NUM_NORMAL),                # 1-5 inputs
    'Num_Outputs': np.random.randint(1, 6, size=NUM_NORMAL),               # 1-5 outputs
    'Fee': np.random.uniform(0.0001, 0.001, size=NUM_NORMAL),             # Normal fees
    'Is_Fraud': np.zeros(NUM_NORMAL, dtype=int)
}

# Adjust amounts to reasonable range
normal_data['Amount'] = np.clip(normal_data['Amount'], 0.01, 50)

# Make fees proportional to amount (normal behavior)
for i in range(NUM_NORMAL):
    normal_data['Fee'][i] = normal_data['Amount'][i] * random.uniform(0.00005, 0.0002)

print(f"✅ Generated {NUM_NORMAL} normal transactions")

# ============================================
# GENERATE FRAUDULENT TRANSACTIONS
# ============================================
print("\n🔴 Generating fraudulent transactions...")

# Fraudulent patterns:
# 1. Large amounts
# 2. Many inputs/outputs (mixing)
# 3. Abnormally low fees
# 4. Suspicious patterns

fraud_data = {
    'Amount': [],
    'Num_Inputs': [],
    'Num_Outputs': [],
    'Fee': [],
    'Is_Fraud': np.ones(NUM_FRAUD, dtype=int)
}

for i in range(NUM_FRAUD):
    fraud_type = random.choice(['large_amount', 'mixing', 'low_fee', 'complex'])
    
    if fraud_type == 'large_amount':
        # Very large transactions
        amount = random.uniform(50, 500)
        inputs = random.randint(3, 15)
        outputs = random.randint(3, 15)
        fee = amount * random.uniform(0.00001, 0.0001)  # Lower than normal
        
    elif fraud_type == 'mixing':
        # Mixing/tumbling pattern (many inputs and outputs)
        amount = random.uniform(10, 100)
        inputs = random.randint(10, 50)
        outputs = random.randint(10, 50)
        fee = amount * random.uniform(0.00005, 0.0002)
        
    elif fraud_type == 'low_fee':
        # Suspiciously low fees
        amount = random.uniform(20, 150)
        inputs = random.randint(5, 20)
        outputs = random.randint(5, 20)
        fee = amount * random.uniform(0.000001, 0.00003)  # Very low
        
    else:  # complex
        # Complex pattern - all suspicious factors
        amount = random.uniform(80, 300)
        inputs = random.randint(20, 80)
        outputs = random.randint(20, 80)
        fee = amount * random.uniform(0.000001, 0.00005)
    
    fraud_data['Amount'].append(amount)
    fraud_data['Num_Inputs'].append(inputs)
    fraud_data['Num_Outputs'].append(outputs)
    fraud_data['Fee'].append(fee)

print(f"✅ Generated {NUM_FRAUD} fraudulent transactions")

# ============================================
# COMBINE AND SHUFFLE
# ============================================
print("\n🔄 Combining and shuffling data...")

# Convert to DataFrames
normal_df = pd.DataFrame(normal_data)
fraud_df = pd.DataFrame(fraud_data)

# Combine
combined_df = pd.concat([normal_df, fraud_df], ignore_index=True)

# Shuffle
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Add additional features
combined_df['Transaction_ID'] = [f"TX{str(i).zfill(6)}" for i in range(len(combined_df))]
combined_df['Timestamp'] = [
    (datetime.now() - timedelta(days=random.randint(0, 365))).isoformat()
    for _ in range(len(combined_df))
]

# Reorder columns
combined_df = combined_df[[
    'Transaction_ID', 'Amount', 'Num_Inputs', 'Num_Outputs', 
    'Fee', 'Is_Fraud', 'Timestamp'
]]

print(f"✅ Created dataset with {len(combined_df)} transactions")

# ============================================
# DISPLAY STATISTICS
# ============================================
print("\n" + "=" * 80)
print("📊 DATASET STATISTICS")
print("=" * 80)

print(f"\nTotal Transactions: {len(combined_df)}")
print(f"Normal: {(combined_df['Is_Fraud'] == 0).sum()} ({(combined_df['Is_Fraud'] == 0).sum() / len(combined_df) * 100:.1f}%)")
print(f"Fraudulent: {(combined_df['Is_Fraud'] == 1).sum()} ({(combined_df['Is_Fraud'] == 1).sum() / len(combined_df) * 100:.1f}%)")

print("\n📈 Feature Statistics:")
print(combined_df[['Amount', 'Num_Inputs', 'Num_Outputs', 'Fee']].describe())

print("\n🔍 Sample Normal Transactions:")
print(combined_df[combined_df['Is_Fraud'] == 0].head(3))

print("\n⚠️ Sample Fraudulent Transactions:")
print(combined_df[combined_df['Is_Fraud'] == 1].head(3))

# ============================================
# SAVE TO CSV
# ============================================
print("\n" + "=" * 80)
print("💾 SAVING DATA")
print("=" * 80)

# Create data directory if it doesn't exist
import os
os.makedirs('data', exist_ok=True)

# Save to CSV
output_file = 'data/bitcoin_transactions.csv'
combined_df.to_csv(output_file, index=False)

print(f"✅ Data saved to: {output_file}")
print(f"📁 File size: {os.path.getsize(output_file) / 1024:.2f} KB")

print("\n" + "=" * 80)
print("✅ TRAINING DATA GENERATION COMPLETE!")
print("=" * 80)
print("\n🚀 Next step: Run 'python train_ensemble_model.py'")