import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ============================================
# SET UP FOLDER PATHS
# ============================================
# Get the base directory (brocode folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define folder paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
VIZ_DIR = os.path.join(BASE_DIR, 'visualizations')

print("=" * 60)
print("CHAINGUARD - STEP 2: DATA EXPLORATION")
print("=" * 60)
print(f"\n📁 Working directory: {BASE_DIR}")

# ============================================
# PART 1: LOAD THE DATA
# ============================================
print("\n📂 Loading dataset files from data/ folder...")

# Load all 3 files from data/ folder
features = pd.read_csv(os.path.join(DATA_DIR, 'elliptic_txs_features.csv'), header=None)
classes = pd.read_csv(os.path.join(DATA_DIR, 'elliptic_txs_classes.csv'))
edges = pd.read_csv(os.path.join(DATA_DIR, 'elliptic_txs_edgelist.csv'))

print("✅ Files loaded successfully!\n")

# ============================================
# PART 2: UNDERSTAND FEATURES FILE
# ============================================
print("=" * 60)
print("📊 FEATURES FILE ANALYSIS")
print("=" * 60)

print(f"\nShape: {features.shape}")
print(f"  → {features.shape[0]} transactions")
print(f"  → {features.shape[1]} columns (1 ID + 166 features)")

print("\n🔍 First 5 rows:")
print(features.head())

print("\n📌 Column 0 is Transaction ID")
print("📌 Columns 1-166 are anonymous features (amounts, timestamps, etc.)")

# ============================================
# PART 3: UNDERSTAND CLASSES FILE (LABELS)
# ============================================
print("\n" + "=" * 60)
print("🏷️  CLASSES FILE ANALYSIS (These are our labels!)")
print("=" * 60)

print(f"\nShape: {classes.shape}")
print(classes.head(10))

print("\n📊 Class Distribution:")
print(classes['class'].value_counts())

# Calculate percentages
class_counts = classes['class'].value_counts()
total = len(classes)
print("\n📈 Percentages:")
for label, count in class_counts.items():
    pct = (count / total) * 100
    print(f"  {label}: {count} ({pct:.2f}%)")

print("\n📌 Legend:")
print("  '1' = Licit (Normal/Legal transaction) ✅")
print("  '2' = Illicit (Fraudulent transaction) ⚠️")
print("  'unknown' = Unlabeled data (we'll ignore these)")

# ============================================
# PART 4: MERGE FEATURES WITH LABELS
# ============================================
print("\n" + "=" * 60)
print("🔗 MERGING FEATURES WITH LABELS")
print("=" * 60)

# Merge on transaction ID
data = features.merge(classes, left_on=0, right_on='txId', how='left')

print(f"\nMerged data shape: {data.shape}")
print("\nFirst few rows with labels:")
print(data[[0, 'class']].head(10))

# ============================================
# PART 5: CLEAN THE DATA
# ============================================
print("\n" + "=" * 60)
print("🧹 DATA CLEANING")
print("=" * 60)

print(f"\nBefore cleaning: {len(data)} rows")

# Remove 'unknown' class
data_clean = data[data['class'] != 'unknown'].copy()
print(f"After removing 'unknown': {len(data_clean)} rows")

# Convert class labels to 0 and 1
data_clean['class'] = data_clean['class'].map({'1': 0, '2': 1})
print("\n✅ Converted labels:")
print("  '1' → 0 (Normal)")
print("  '2' → 1 (Fraud)")

print("\n📊 Final Class Distribution:")
print(data_clean['class'].value_counts())

fraud_count = (data_clean['class'] == 1).sum()
normal_count = (data_clean['class'] == 0).sum()
fraud_pct = (fraud_count / len(data_clean)) * 100

print(f"\n🎯 Key Insight:")
print(f"  Fraud rate: {fraud_pct:.2f}%")
print(f"  This is an IMBALANCED dataset! ⚖️")

# ============================================
# PART 6: VISUALIZE THE DATA
# ============================================
print("\n" + "=" * 60)
print("📊 CREATING VISUALIZATIONS")
print("=" * 60)

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ChainGuard: Data Exploration Dashboard', fontsize=16, fontweight='bold')

# Plot 1: Class distribution
ax1 = axes[0, 0]
class_counts = data_clean['class'].value_counts()
colors = ['#2ecc71', '#e74c3c']  # Green for normal, Red for fraud
ax1.bar(['Normal', 'Fraud'], class_counts.values, color=colors, alpha=0.7)
ax1.set_ylabel('Count')
ax1.set_title('Transaction Class Distribution')
ax1.set_ylim(0, max(class_counts.values) * 1.1)
for i, v in enumerate(class_counts.values):
    ax1.text(i, v + 100, str(v), ha='center', fontweight='bold')

# Plot 2: Pie chart
ax2 = axes[0, 1]
ax2.pie(class_counts.values, labels=['Normal', 'Fraud'], autopct='%1.1f%%',
        colors=colors, startangle=90)
ax2.set_title('Class Proportion')

# Plot 3: Sample feature distribution (column 1)
ax3 = axes[1, 0]
ax3.hist(data_clean[1], bins=50, alpha=0.7, color='steelblue')
ax3.set_xlabel('Feature 1 Value')
ax3.set_ylabel('Frequency')
ax3.set_title('Sample Feature Distribution (Feature 1)')

# Plot 4: Feature comparison between classes
ax4 = axes[1, 1]
normal_feature = data_clean[data_clean['class'] == 0][1]
fraud_feature = data_clean[data_clean['class'] == 1][1]
ax4.hist(normal_feature, bins=30, alpha=0.5, label='Normal', color='green')
ax4.hist(fraud_feature, bins=30, alpha=0.5, label='Fraud', color='red')
ax4.set_xlabel('Feature 1 Value')
ax4.set_ylabel('Frequency')
ax4.set_title('Feature 1: Normal vs Fraud')
ax4.legend()

plt.tight_layout()

# Save to visualizations/ folder
viz_path = os.path.join(VIZ_DIR, 'data_exploration.png')
plt.savefig(viz_path, dpi=300, bbox_inches='tight')
print(f"✅ Saved visualization: {viz_path}")

# ============================================
# PART 7: PREPARE DATA FOR MODEL
# ============================================
print("\n" + "=" * 60)
print("💾 PREPARING DATA FOR MODEL TRAINING")
print("=" * 60)

# Separate features (X) and labels (y)
X = data_clean.iloc[:, 1:167].values  # All feature columns
y = data_clean['class'].values         # Label column

print(f"\n✅ Features (X) shape: {X.shape}")
print(f"✅ Labels (y) shape: {y.shape}")

# Save processed data to processed_data/ folder
X_path = os.path.join(PROCESSED_DIR, 'X_data.npy')
y_path = os.path.join(PROCESSED_DIR, 'y_data.npy')

np.save(X_path, X)
np.save(y_path, y)

print("\n💾 Saved files:")
print(f"  → {X_path}")
print(f"  → {y_path}")

# ============================================
# PART 8: SUMMARY STATISTICS
# ============================================
print("\n" + "=" * 60)
print("📋 FINAL SUMMARY")
print("=" * 60)

print(f"""
✅ Data Loading: Complete
✅ Data Cleaning: Complete
✅ Visualization: Complete
✅ Data Preparation: Complete

📊 Dataset Statistics:
   • Total transactions: {len(data_clean):,}
   • Normal transactions: {normal_count:,} ({100-fraud_pct:.2f}%)
   • Fraudulent transactions: {fraud_count:,} ({fraud_pct:.2f}%)
   • Number of features: 166
   
🎯 Key Takeaway:
   This is a HIGHLY IMBALANCED dataset!
   We need to handle this in our model training (Step 3).

📁 Generated Files:
   • processed_data/X_data.npy - Feature matrix
   • processed_data/y_data.npy - Labels
   • visualizations/data_exploration.png - Visualizations
   
✅ STEP 2 COMPLETE! Ready for Step 3 (Model Training)
""")

print("=" * 60)