from flask import Flask, render_template, jsonify, request, send_file, make_response, Response
import pandas as pd
import numpy as np
import joblib
import os
from fpdf import FPDF
from datetime import datetime
import io
import csv
from werkzeug.utils import secure_filename
import json
from collections import Counter
from dotenv import load_dotenv
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import threading
import hashlib
import secrets
from ecdsa import SigningKey, SECP256k1, VerifyingKey, BadSignatureError
from database import db
from blockchain_api import BlockchainAPI
from ensemble_predictor import EnsemblePredictor

# ✅ ADD THIS IMPORT (around line 10)


# Load environment variables
load_dotenv()

app = Flask(__name__)
try:
    db.init_db()
    print("✅ Database initialized successfully")
except Exception as e:
    print(f"❌ Database initialization failed: {e}")

# Email configuration
EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
SENDER_EMAIL = os.getenv('SENDER_EMAIL', '')
SENDER_PASSWORD = os.getenv('SENDER_PASSWORD', '')
ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
SMTP_SERVER = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))

# Alert thresholds
ALERT_THRESHOLD_CRITICAL = int(os.getenv('ALERT_THRESHOLD_CRITICAL', 90))
ALERT_THRESHOLD_HIGH = int(os.getenv('ALERT_THRESHOLD_HIGH', 75))
ALERT_THRESHOLD_MEDIUM = int(os.getenv('ALERT_THRESHOLD_MEDIUM', 50))

# Base transaction fee used to derive fee from inputs/outputs when user does not provide fee
# Formula used by the app and dashboard JS:
# Transaction Fee (BTC) = BASE_TRANSACTION_FEE + (inputs * 0.00002) + (outputs * 0.000015)
BASE_TRANSACTION_FEE = float(os.getenv('BASE_TRANSACTION_FEE', 0.00005))

# Load model and data
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'
PROCESSED_DIR = 'processed_data'

# Load the ensemble predictor (uses 3 models)
try:
    ensemble = EnsemblePredictor()
    print("✅ Ensemble models loaded successfully!")
    USE_ENSEMBLE = True
except Exception as e:
    print(f"⚠️ Could not load ensemble, using single model: {e}")
    model = joblib.load(os.path.join(MODELS_DIR, 'chainguard_model.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    USE_ENSEMBLE = False
    print("✅ Single model loaded as fallback")

# Load results
results_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'all_transaction_results.csv'))
top_10_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'top_10_risky_transactions.csv'))

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# History file path
HISTORY_FILE = 'history.json'

# Security salt for hashing sensitive identifiers (wallets, user IDs, etc.)
SECURITY_SALT = os.getenv('SECURITY_SALT', 'dev_default_salt')

def obfuscate_identifier(identifier: str) -> str:
    """One-way, salted hash for sensitive identifiers.

    This is used to obfuscate wallet addresses, sender IDs, etc. so the
    ML / analytics pipeline only ever sees pseudonymous IDs.
    """
    if not identifier:
        return ""
    value = f"{SECURITY_SALT}:{identifier}".encode("utf-8")
    digest = hashlib.sha256(value).hexdigest()
    return digest[:16]


# ================= WALLET CRYPTO HELPERS =================

def public_key_to_address(public_key_hex: str) -> str:
    """Derive a simple wallet address from a public key.

    This is NOT a real blockchain address format. We take a SHA-256 hash
    of the raw public key bytes and keep the last 40 hex chars, prefixed
    with 0x, to simulate an address.
    """
    try:
        pk_bytes = bytes.fromhex(public_key_hex)
    except Exception:
        return ""

    digest = hashlib.sha256(pk_bytes).hexdigest()
    return "0x" + digest[-40:]


def generate_wallet_keys():
    """Generate a new ECDSA keypair and derived wallet address."""
    sk = SigningKey.generate(curve=SECP256k1)
    vk = sk.get_verifying_key()

    private_key_hex = sk.to_string().hex()
    public_key_hex = vk.to_string().hex()
    address = public_key_to_address(public_key_hex)

    return private_key_hex, public_key_hex, address


def sign_wallet_message(private_key_hex: str, message: str) -> str:
    """Sign an arbitrary message with a hex-encoded private key."""
    sk = SigningKey.from_string(bytes.fromhex(private_key_hex), curve=SECP256k1)
    signature = sk.sign(message.encode("utf-8"))
    return signature.hex()


def verify_wallet_message(public_key_hex: str, message: str, signature_hex: str) -> bool:
    """Verify that signature matches message for the given public key."""
    try:
        vk = VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=SECP256k1)
        vk.verify(bytes.fromhex(signature_hex), message.encode("utf-8"))
        return True
    except (BadSignatureError, Exception):
        return False

# Add this function after the imports and model loading (around line 50)

def calculate_risk_score(transaction):
    """
    Calculate risk score for a transaction
    
    Args:
        transaction (dict): Transaction data with keys:
            - amount: Transaction amount in BTC
            - num_inputs: Number of input addresses
            - num_outputs: Number of output addresses
            - fee: Transaction fee in BTC
    
    Returns:
        dict: Risk assessment with score, prediction, and alert level
    """
    
    amount = transaction.get('amount', 0)
    num_inputs = transaction.get('num_inputs', 0)
    num_outputs = transaction.get('num_outputs', 0)
    fee = transaction.get('fee', 0.0001)
    
    # Base risk score
    risk_score = 15
    risk_factors = []
    
    # 1. Amount-based risk
    if amount > 100:
        risk_score += 30
        risk_factors.append(f"Very large amount: {amount} BTC")
    elif amount > 50:
        risk_score += 20
        risk_factors.append(f"Large amount: {amount} BTC")
    elif amount > 20:
        risk_score += 10
        risk_factors.append(f"Medium amount: {amount} BTC")
    
    # 2. Input/Output complexity
    total_io = num_inputs + num_outputs
    if total_io > 50:
        risk_score += 25
        risk_factors.append(f"High I/O complexity: {total_io} addresses")
    elif total_io > 30:
        risk_score += 15
        risk_factors.append(f"Moderate I/O complexity: {total_io} addresses")
    
    # 3. Fee analysis (detect abnormally low fees)
    if amount > 0:
        expected_fee = amount * 0.0001
        if fee < expected_fee * 0.5:
            risk_score += 20
            risk_factors.append(f"Suspiciously low fee: {fee} BTC")
    
    # 4. Mixing pattern detection
    if num_inputs > 5 and num_outputs > 5:
        risk_score += 30
        risk_factors.append("Possible mixing/tumbling pattern")
    
    # 5. Fund consolidation pattern
    if num_inputs > 20:
        risk_score += 12
        risk_factors.append(f"Many inputs: {num_inputs}")
    
    # 6. Fund distribution pattern
    if num_outputs > 20:
        risk_score += 12
        risk_factors.append(f"Many outputs: {num_outputs}")
    
    # Cap at 100
    risk_score = min(100, risk_score)
    
    # Determine prediction and alert level
    if risk_score >= 90:
        prediction = "Fraudulent"
        alert_level = "CRITICAL"
    elif risk_score >= 75:
        prediction = "Fraudulent"
        alert_level = "HIGH"
    elif risk_score >= 50:
        prediction = "Suspicious"
        alert_level = "MEDIUM"
    else:
        prediction = "Normal"
        alert_level = "LOW"
    
    return {
        'risk_score': risk_score,
        'prediction': prediction,
        'alert_level': alert_level,
        'risk_factors': risk_factors
    }

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_history():
    """Load transaction history from JSON file"""
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r') as f:
                return json.load(f)
        return []
    except:
        return []

def save_to_history(transaction_data):
    """Save transaction to history"""
    try:
        history = load_history()
        
        # Add timestamp
        transaction_data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        transaction_data['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # Add to beginning of list (newest first)
        history.insert(0, transaction_data)
        
        # Keep only last 1000 transactions
        history = history[:1000]
        
        # Save to file
        with open(HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2)
        
        return True
    except Exception as e:
        print(f"Error saving to history: {e}")
        return False

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('dashboard.html')

def add_to_history(transaction_data):
    """Add transaction to history (alias for save_to_history)"""
    return save_to_history(transaction_data)

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    try:
        watchlist_count = len(db.get_all_watchlist())
    except:
        watchlist_count = 0
    
    stats = {
        'total_scanned': len(results_df),
        'critical_alerts': len(results_df[results_df['Alert_Level'] == 'CRITICAL']),
        'high_alerts': len(results_df[results_df['Alert_Level'] == 'HIGH']),
        'medium_alerts': len(results_df[results_df['Alert_Level'] == 'MEDIUM']),
        'accuracy': f"{(results_df['Correct_Prediction'] == '✅').sum() / len(results_df) * 100:.2f}",
        'avg_risk_score': f"{results_df['Risk_Score'].mean():.2f}",
        'watchlist_count': watchlist_count
    }
    return jsonify(stats)

@app.route('/api/top10')
def get_top10():
    """Get top 10 riskiest transactions"""
    top_10_data = top_10_df.to_dict('records')
    return jsonify(top_10_data)

@app.route('/api/all_transactions')
def get_all_transactions():
    """Get all transactions with pagination"""
    page = int(request.args.get('page', 1))
    per_page = 50
    
    start = (page - 1) * per_page
    end = start + per_page
    
    transactions = results_df.iloc[start:end].to_dict('records')
    total_pages = (len(results_df) + per_page - 1) // per_page
    
    return jsonify({
        'transactions': transactions,
        'page': page,
        'total_pages': total_pages,
        'total_count': len(results_df)
    })

@app.route('/api/alert_distribution')
def get_alert_distribution():
    """Get alert level distribution for chart"""
    distribution = results_df['Alert_Level'].value_counts().to_dict()
    
    # Order by severity
    ordered = {
        'CRITICAL': distribution.get('CRITICAL', 0),
        'HIGH': distribution.get('HIGH', 0),
        'MEDIUM': distribution.get('MEDIUM', 0),
        'LOW': distribution.get('LOW', 0),
        'NORMAL': distribution.get('NORMAL', 0)
    }
    
    return jsonify(ordered)

@app.route('/api/risk_distribution')
def get_risk_distribution():
    """Get risk score distribution"""
    # Create bins for risk scores
    bins = [0, 25, 50, 75, 90, 100]
    labels = ['0-25', '25-50', '50-75', '75-90', '90-100']
    
    results_df['Risk_Bin'] = pd.cut(results_df['Risk_Score'], bins=bins, labels=labels, include_lowest=True)
    distribution = results_df['Risk_Bin'].value_counts().sort_index().to_dict()
    
    return jsonify(distribution)

@app.route('/api/search/<int:tx_id>')
def search_transaction(tx_id):
    """Search for specific transaction"""
    tx = results_df[results_df['Transaction_ID'] == tx_id]
    
    if len(tx) == 0:
        return jsonify({'error': 'Transaction not found'}), 404
    
    return jsonify(tx.iloc[0].to_dict())



@app.route('/api/predict', methods=['POST'])
def predict_new_transaction():
    """Predict risk for new transaction (demo with random test data)"""
    try:
        # Generate random transaction data
        amount = np.random.uniform(0.1, 200)
        num_inputs = np.random.randint(1, 80)
        num_outputs = np.random.randint(1, 80)
        fee = np.random.uniform(0.00001, 0.01)
        
        # Create transaction object
        transaction = {
            'amount': amount,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fee': fee
        }
        
        # Calculate risk score using proper function
        risk_result = calculate_risk_score(transaction)
        risk_score = risk_result['risk_score']
        risk_factors = risk_result.get('risk_factors', [])
        
        # Enhance with ensemble if available
        if USE_ENSEMBLE:
            try:
                result = ensemble.predict(amount, num_inputs, num_outputs, fee)
                
                # Adjust risk score based on ensemble confidence
                if result['is_fraud'] and result['confidence'] > 75:
                    risk_score = max(risk_score, 85)  # Boost if ensemble very confident
                    risk_factors.append(f"🤖 AI Ensemble: {result['votes_for_fraud']}/3 models predict FRAUD ({result['confidence']:.1f}% confidence)")
                elif not result['is_fraud'] and result['confidence'] < 25:
                    risk_score = min(risk_score, 40)  # Lower if ensemble confident it's normal
                    risk_factors.append(f"✅ AI Ensemble: All models agree - NORMAL transaction")
            except Exception as e:
                print(f"Ensemble prediction error: {e}")
        
        # Determine prediction
        prediction = 1 if risk_score >= 50 else 0
        
        # Generate alert
        if risk_score >= 90:
            alert_level = 'CRITICAL'
            alert_message = '🚨 IMMEDIATE ACTION REQUIRED'
            action = 'Freeze wallet immediately'
        elif risk_score >= 75:
            alert_level = 'HIGH'
            alert_message = '⚠️ HIGH RISK DETECTED'
            action = 'Flag wallet for investigation'
        elif risk_score >= 50:
            alert_level = 'MEDIUM'
            alert_message = '⚡ SUSPICIOUS ACTIVITY'
            action = 'Add to watchlist'
        elif risk_score >= 25:
            alert_level = 'LOW'
            alert_message = '👀 Minor anomaly'
            action = 'Log for review'
        else:
            alert_level = 'NORMAL'
            alert_message = '✅ Transaction appears normal'
            action = 'Continue monitoring'
        
        result_data = {
            'transaction_id': f"NEW_{np.random.randint(10000, 99999)}",
            'prediction': 'Fraud' if prediction == 1 else 'Normal',
            'risk_score': risk_score,
            'alert_level': alert_level,
            'alert_message': alert_message,
            'recommended_action': action,
            'risk_factors': risk_factors,
            'details': {
                'amount': round(amount, 4),
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'fee': round(fee, 6)
            }
        }
        
        return jsonify(result_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_custom', methods=['POST'])
def predict_custom_transaction():
    """Predict risk for custom transaction with user input"""
    try:
        data = request.json
        
        # Get user inputs
        transaction_amount = float(data.get('amount', 1.5))
        num_inputs = int(data.get('num_inputs', 2))
        num_outputs = int(data.get('num_outputs', 2))
        # Compute transaction fee from inputs/outputs using agreed formula
        # Transaction Fee (BTC) = BASE_TRANSACTION_FEE + (inputs * 0.00002) + (outputs * 0.000015)
        transaction_fee = (
            BASE_TRANSACTION_FEE +
            (num_inputs * 0.00002) +
            (num_outputs * 0.000015)
        )
        
        # Create transaction object
        transaction = {
            'amount': transaction_amount,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fee': transaction_fee
        }
        
        # Calculate risk score using proper function
        risk_result = calculate_risk_score(transaction)
        risk_score = risk_result['risk_score']
        risk_factors = risk_result.get('risk_factors', [])
        
        # Enhance with ensemble if available
        ensemble_details = None
        if USE_ENSEMBLE:
            try:
                result = ensemble.predict(transaction_amount, num_inputs, num_outputs, transaction_fee)
                
                # Store ensemble details for display
                ensemble_details = {
                    'confidence': result['confidence'],
                    'votes': result['ensemble_votes'],
                    'votes_for_fraud': result['votes_for_fraud'],
                    'is_fraud': result['is_fraud']
                }
                
                # Adjust risk score based on ensemble
                if result['is_fraud'] and result['confidence'] > 75:
                    risk_score = max(risk_score, 85)
                    risk_factors.append(f"🤖 AI Ensemble: {result['votes_for_fraud']}/3 models predict FRAUD ({result['confidence']:.1f}% confidence)")
                elif not result['is_fraud'] and result['confidence'] < 25:
                    risk_score = min(risk_score, 40)
                    risk_factors.append(f"✅ AI Ensemble: {result['votes_for_normal']}/3 models predict NORMAL")
                else:
                    risk_factors.append(f"🤖 AI Ensemble: Mixed signals - {result['votes_for_fraud']}/3 vote fraud")
                    
            except Exception as e:
                print(f"Ensemble prediction error: {e}")
        
        # Determine prediction
        prediction = 1 if risk_score >= 50 else 0
        
        # Generate alert based on FINAL risk score
        if risk_score >= 90:
            alert_level = 'CRITICAL'
            alert_message = '🚨 IMMEDIATE ACTION REQUIRED'
            action = 'Freeze wallet immediately and investigate'
            color = 'danger'
        elif risk_score >= 75:
            alert_level = 'HIGH'
            alert_message = '⚠️ HIGH RISK DETECTED'
            action = 'Flag for manual review within 1 hour'
            color = 'warning'
        elif risk_score >= 50:
            alert_level = 'MEDIUM'
            alert_message = '⚡ SUSPICIOUS ACTIVITY'
            action = 'Add to monitoring watchlist'
            color = 'info'
        elif risk_score >= 25:
            alert_level = 'LOW'
            alert_message = '👀 Minor anomaly detected'
            action = 'Log for periodic review'
            color = 'secondary'
        else:
            alert_level = 'NORMAL'
            alert_message = '✅ Transaction appears normal'
            action = 'Continue monitoring'
            color = 'success'
        
        # Generate transaction ID
        random_idx = np.random.randint(10000, 99999)
        
        result_data = {
            'transaction_id': f"CUSTOM_{random_idx}",
            'prediction': 'Fraud' if prediction == 1 else 'Normal',
            'risk_score': risk_score,
            'alert_level': alert_level,
            'alert_message': alert_message,
            'recommended_action': action,
            'color': color,
            'risk_factors': risk_factors,
            'ensemble_details': ensemble_details,
            'using_ensemble': USE_ENSEMBLE,
                'details': {
                    'amount': transaction_amount,
                    'num_inputs': num_inputs,
                    'num_outputs': num_outputs,
                    'fee': transaction_fee
                }
        }
        
        # Save to history
        history_entry = {
            'transaction_id': result_data['transaction_id'],
            'amount': transaction_amount,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fee': transaction_fee,
            'risk_score': risk_score,
            'alert_level': alert_level,
            'prediction': result_data['prediction'],
            'alert_message': alert_message,
            'recommended_action': action,
            'timestamp': datetime.now().isoformat()
        }
        save_to_history(history_entry)
        
        # Trigger alert if needed
        if risk_score >= ALERT_THRESHOLD_HIGH:
            alert_type = 'CRITICAL' if risk_score >= ALERT_THRESHOLD_CRITICAL else 'HIGH'
            trigger_alert_async(history_entry, alert_type)
        
        return jsonify(result_data)
        
    except Exception as e:
        print(f"Error in predict_custom: {e}")
        return jsonify({'error': str(e)}), 400
    

# Add this route (around line 300, after other routes)
@app.route('/api/fetch-live-transactions', methods=['POST'])
def fetch_live_transactions():
    """Fetch real transactions from blockchain"""
    try:
        data = request.json
        count = int(data.get('count', 10))
        
        # Initialize API
        api = BlockchainAPI()
        
        # Fetch transactions
        transactions = api.fetch_latest_transactions(count=count)
        
        if not transactions:
            return jsonify({'error': 'Failed to fetch transactions'}), 500
        
        # Analyze each transaction with our model
        analyzed_results = []
        for tx in transactions:
            # Calculate risk score
            risk_result = calculate_risk_score(tx)
            
            analyzed_results.append({
                'transaction_id': tx['transaction_id'],
                'amount': tx['amount'],
                'num_inputs': tx['num_inputs'],
                'num_outputs': tx['num_outputs'],
                'fee': tx['fee'],
                'risk_score': risk_result['risk_score'],
                'prediction': risk_result['prediction'],
                'alert_level': risk_result['alert_level'],
                'timestamp': datetime.fromtimestamp(tx['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            })
        
        # Save to database
        for result in analyzed_results:
            db.add_transaction_result(result)
        
        return jsonify({
            'status': 'success',
            'count': len(analyzed_results),
            'transactions': analyzed_results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/live-monitoring')
def live_monitoring():
    """Live blockchain monitoring page"""
    return render_template('live_monitoring.html')

# ============================================
# FEATURE 1: EXPORT REPORTS
# ============================================

@app.route('/api/export/csv/<transaction_type>')
def export_csv(transaction_type):
    """Export transactions as CSV"""
    try:
        if transaction_type == 'top10':
            df = top_10_df
            filename = f"top_10_risky_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif transaction_type == 'all':
            df = results_df
            filename = f"all_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif transaction_type == 'critical':
            df = results_df[results_df['Alert_Level'] == 'CRITICAL']
            filename = f"critical_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        elif transaction_type == 'high':
            df = results_df[results_df['Alert_Level'].isin(['CRITICAL', 'HIGH'])]
            filename = f"high_risk_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        else:
            return jsonify({'error': 'Invalid transaction type'}), 400
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename={filename}'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/pdf/summary')
def export_pdf_summary():
    """Export summary report as PDF"""
    try:
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 10, 'ChainGuard Security Report', ln=True, align='C')
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Overall Statistics', ln=True)
        pdf.ln(5)
        
        pdf.set_font('Arial', '', 11)
        stats_data = [
            ('Total Transactions Scanned', str(len(results_df))),
            ('Critical Alerts', str(len(results_df[results_df['Alert_Level'] == 'CRITICAL']))),
            ('High Risk Alerts', str(len(results_df[results_df['Alert_Level'] == 'HIGH']))),
            ('Medium Risk Alerts', str(len(results_df[results_df['Alert_Level'] == 'MEDIUM']))),
            ('Model Accuracy', f"{(results_df['Correct_Prediction'] == '✅').sum() / len(results_df) * 100:.2f}%"),
            ('Average Risk Score', f"{results_df['Risk_Score'].mean():.2f}")
        ]
        
        for label, value in stats_data:
            pdf.cell(100, 8, label + ':', border=0)
            pdf.cell(0, 8, value, border=0, ln=True)
        
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Top 10 Highest Risk Transactions', ln=True)
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 9)
        pdf.cell(15, 8, '#', border=1)
        pdf.cell(35, 8, 'Transaction ID', border=1)
        pdf.cell(25, 8, 'Risk Score', border=1)
        pdf.cell(30, 8, 'Alert Level', border=1)
        pdf.cell(85, 8, 'Action', border=1, ln=True)
        
        pdf.set_font('Arial', '', 8)
        for idx, row in top_10_df.head(10).iterrows():
            pdf.cell(15, 8, str(idx + 1), border=1)
            pdf.cell(35, 8, str(row['Transaction_ID'])[:10], border=1)
            pdf.cell(25, 8, f"{row['Risk_Score']}/100", border=1)
            pdf.cell(30, 8, str(row['Alert_Level']), border=1)
            pdf.cell(85, 8, str(row['Recommended_Action'])[:30] + '...', border=1, ln=True)
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=chainguard_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/pdf/transaction/<int:tx_id>')
def export_pdf_transaction(tx_id):
    """Export individual transaction report as PDF"""
    try:
        tx = results_df[results_df['Transaction_ID'] == tx_id]
        
        if len(tx) == 0:
            return jsonify({'error': 'Transaction not found'}), 404
        
        tx = tx.iloc[0]
        
        pdf = FPDF()
        pdf.add_page()
        
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 10, 'Transaction Analysis Report', ln=True, align='C')
        pdf.ln(10)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Transaction Details', ln=True)
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        details = [
            ('Transaction ID', str(tx['Transaction_ID'])),
            ('Obfuscated ID', str(tx['Obfuscated_ID'])),
            ('Risk Score', f"{tx['Risk_Score']}/100"),
            ('Alert Level', str(tx['Alert_Level'])),
            ('Prediction', str(tx['Prediction'])),
            ('Actual Class', str(tx['Actual_Class'])),
            ('Correct Prediction', str(tx['Correct_Prediction']))
        ]
        
        for label, value in details:
            pdf.cell(70, 7, label + ':', border=0)
            pdf.cell(0, 7, value, border=0, ln=True)
        
        pdf.ln(5)
        
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recommended Action', ln=True)
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 7, str(tx['Recommended_Action']))
        
        pdf.ln(5)
        
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f'Generated by ChainGuard | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')
        
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=transaction_{tx_id}_report.pdf'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# FEATURE 2: BATCH ANALYSIS
# ============================================

@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """Upload and analyze CSV file with multiple transactions"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        df = pd.read_csv(filepath)
        
        required_columns = ['amount', 'num_inputs', 'num_outputs', 'fee']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.remove(filepath)
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'required': required_columns
            }), 400
        
        results = []
        
        # ✅ ANALYZE EACH TRANSACTION
        for idx, row in df.iterrows():
            try:
                amount = float(row['amount'])
                num_inputs = int(row['num_inputs'])
                num_outputs = int(row['num_outputs'])
                fee = float(row['fee'])
                
                # ✅ USE ENSEMBLE IF AVAILABLE
                if USE_ENSEMBLE:
                    result = ensemble.predict(amount, num_inputs, num_outputs, fee)
                    risk_score = int(result['confidence'])
                else:
                    # Fallback: use calculate_risk_score
                    transaction = {
                        'amount': amount,
                        'num_inputs': num_inputs,
                        'num_outputs': num_outputs,
                        'fee': fee
                    }
                    risk_result = calculate_risk_score(transaction)
                    risk_score = risk_result['risk_score']
                
                # Determine alert level
                if risk_score >= 90:
                    alert_level = 'CRITICAL'
                elif risk_score >= 75:
                    alert_level = 'HIGH'
                elif risk_score >= 50:
                    alert_level = 'MEDIUM'
                elif risk_score >= 25:
                    alert_level = 'LOW'
                else:
                    alert_level = 'NORMAL'
                
                results.append({
                    'row': idx + 1,
                    'amount': amount,
                    'num_inputs': num_inputs,
                    'num_outputs': num_outputs,
                    'fee': fee,
                    'risk_score': risk_score,
                    'alert_level': alert_level,
                    'prediction': 'Fraud' if risk_score >= 50 else 'Normal'
                })
                
            except Exception as e:
                results.append({
                    'row': idx + 1,
                    'error': str(e)
                })
        
        # ✅ CALCULATE SUMMARY STATISTICS
        valid_results = [r for r in results if 'error' not in r]
        
        summary = {
            'total_transactions': len(df),
            'successful_analyses': len(valid_results),
            'failed_analyses': len(results) - len(valid_results),
            'critical_count': len([r for r in valid_results if r['alert_level'] == 'CRITICAL']),
            'high_count': len([r for r in valid_results if r['alert_level'] == 'HIGH']),
            'medium_count': len([r for r in valid_results if r['alert_level'] == 'MEDIUM']),
            'low_count': len([r for r in valid_results if r['alert_level'] == 'LOW']),
            'normal_count': len([r for r in valid_results if r['alert_level'] == 'NORMAL']),
            'avg_risk_score': round(np.mean([r['risk_score'] for r in valid_results]), 2) if valid_results else 0,
            'max_risk_score': max([r['risk_score'] for r in valid_results]) if valid_results else 0,
            'min_risk_score': min([r['risk_score'] for r in valid_results]) if valid_results else 0
        }
        
        # ✅ SAVE RESULTS TO CSV
        results_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_filepath = os.path.join(OUTPUTS_DIR, results_filename)
        
        results_df_batch = pd.DataFrame(valid_results)
        results_df_batch.to_csv(results_filepath, index=False)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'success': True,
            'summary': summary,
            'results': results,
            'download_url': f'/api/download/{results_filename}'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download batch analysis results"""
    try:
        filepath = os.path.join(OUTPUTS_DIR, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/api/batch/template')
def download_template():
    """Download CSV template for batch analysis"""
    try:
        template_data = [
            ['amount', 'num_inputs', 'num_outputs', 'fee'],
            [1.5, 2, 2, 0.0001],
            [28, 7, 10, 0.0001],
            [60, 15, 20, 0.00001],
            [150, 50, 60, 0.000001]
        ]
        
        template_filename = 'batch_analysis_template.csv'
        template_filepath = os.path.join(UPLOAD_FOLDER, template_filename)
        
        with open(template_filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(template_data)
        
        return send_file(template_filepath, as_attachment=True)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# FEATURE 3: HISTORICAL TRACKING
# ============================================

@app.route('/api/history')
def get_history():
    """Get transaction analysis history"""
    try:
        history = load_history()
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        
        start = (page - 1) * per_page
        end = start + per_page
        
        paginated_history = history[start:end]
        
        return jsonify({
            'history': paginated_history,
            'total': len(history),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(history) + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/stats')
def get_history_stats():
    """Get statistics from history"""
    try:
        history = load_history()
        
        if not history:
            return jsonify({
                'total_analyzed': 0,
                'avg_risk': 0,
                'alert_distribution': {},
                'trend': []
            })
        
        risk_scores = [h['risk_score'] for h in history]
        alert_levels = [h['alert_level'] for h in history]
        
        alert_dist = Counter(alert_levels)
        
        today = datetime.now()
        trend_data = []
        
        for i in range(6, -1, -1):
            date = (today - pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            day_transactions = [h for h in history if h.get('date') == date]
            
            trend_data.append({
                'date': date,
                'count': len(day_transactions),
                'avg_risk': np.mean([h['risk_score'] for h in day_transactions]) if day_transactions else 0,
                'critical': len([h for h in day_transactions if h['alert_level'] == 'CRITICAL']),
                'high': len([h for h in day_transactions if h['alert_level'] == 'HIGH'])
            })
        
        return jsonify({
            'total_analyzed': len(history),
            'avg_risk': np.mean(risk_scores),
            'max_risk': max(risk_scores),
            'min_risk': min(risk_scores),
            'alert_distribution': dict(alert_dist),
            'trend': trend_data,
            'recent_high_risk': len([h for h in history[:50] if h['risk_score'] >= 75])
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/clear', methods=['POST'])
def clear_history():
    """Clear transaction history"""
    try:
        with open(HISTORY_FILE, 'w') as f:
            json.dump([], f)
        
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history/export')
def export_history():
    """Export history as CSV"""
    try:
        history = load_history()
        
        if not history:
            return jsonify({'error': 'No history to export'}), 400
        
        df = pd.DataFrame(history)
        
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=transaction_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ============================================
# FEATURE 4: ALERT SYSTEM
# ============================================

def send_email_alert(transaction_data, alert_type='HIGH'):
    """Send email alert for high-risk transaction"""
    if not EMAIL_ENABLED:
        print(f"[ALERT] Email disabled. Would send {alert_type} alert for transaction {transaction_data.get('transaction_id')}")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SENDER_EMAIL
        msg['To'] = ALERT_EMAIL
        msg['Subject'] = f"🚨 ChainGuard Alert: {alert_type} Risk Transaction Detected"
        
        html_body = f"""
        <html>
            <body style="font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px;">
                <div style="max-width: 600px; margin: 0 auto; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    
                    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center;">
                        <h1 style="color: white; margin: 0;">🛡️ ChainGuard Alert</h1>
                        <p style="color: rgba(255,255,255,0.9); margin: 10px 0 0 0;">AI-Powered Fraud Detection System</p>
                    </div>
                    
                    <div style="padding: 20px; text-align: center; background: {'#ff4757' if alert_type == 'CRITICAL' else '#ff6348' if alert_type == 'HIGH' else '#ffa502'};">
                        <h2 style="color: white; margin: 0;">{'🚨 CRITICAL' if alert_type == 'CRITICAL' else '⚠️ HIGH RISK' if alert_type == 'HIGH' else '⚡ MEDIUM RISK'} TRANSACTION DETECTED</h2>
                    </div>
                    
                    <div style="padding: 30px;">
                        <h3 style="color: #333; margin-bottom: 20px;">Transaction Details</h3>
                        
                        <table style="width: 100%; border-collapse: collapse;">
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Transaction ID:</td>
                                <td style="padding: 12px 0; color: #333;"><code>{transaction_data.get('transaction_id', 'N/A')}</code></td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Risk Score:</td>
                                <td style="padding: 12px 0;">
                                    <span style="background: #ff4757; color: white; padding: 5px 15px; border-radius: 20px; font-weight: bold; font-size: 18px;">
                                        {transaction_data.get('risk_score', 0)}/100
                                    </span>
                                </td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Alert Level:</td>
                                <td style="padding: 12px 0; color: #333;">{transaction_data.get('alert_level', 'N/A')}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Amount:</td>
                                <td style="padding: 12px 0; color: #333;">{transaction_data.get('amount', 0)} BTC</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Inputs/Outputs:</td>
                                <td style="padding: 12px 0; color: #333;">{transaction_data.get('num_inputs', 0)} / {transaction_data.get('num_outputs', 0)}</td>
                            </tr>
                            <tr style="border-bottom: 1px solid #eee;">
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Fee:</td>
                                <td style="padding: 12px 0; color: #333;">{transaction_data.get('fee', 0)} BTC</td>
                            </tr>
                            <tr>
                                <td style="padding: 12px 0; color: #666; font-weight: bold;">Timestamp:</td>
                                <td style="padding: 12px 0; color: #333;">{transaction_data.get('timestamp', 'N/A')}</td>
                            </tr>
                        </table>
                        
                        <div style="margin-top: 30px; padding: 20px; background: #fff3cd; border-left: 4px solid #ffa502; border-radius: 5px;">
                            <h4 style="color: #856404; margin-top: 0;">📋 Recommended Action:</h4>
                            <p style="color: #856404; margin: 0; line-height: 1.6;">{transaction_data.get('recommended_action', 'Please review this transaction immediately.')}</p>
                        </div>
                    </div>
                    
                    <div style="background: #f8f9fa; padding: 20px; text-align: center; border-top: 1px solid #eee;">
                        <p style="color: #666; margin: 0; font-size: 14px;">
                            This is an automated alert from ChainGuard AI Fraud Detection System
                        </p>
                        <p style="color: #999; margin: 10px 0 0 0; font-size: 12px;">
                            Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
                        </p>
                    </div>
                    
                </div>
            </body>
        </html>
        """
        
        msg.attach(MIMEText(html_body, 'html'))
        
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        
        print(f"[ALERT] Email sent successfully for transaction {transaction_data.get('transaction_id')}")
        return True
        
    except Exception as e:
        print(f"[ALERT] Error sending email: {e}")
        return False

def trigger_alert_async(transaction_data, alert_type):
    """Trigger alert in background thread"""
    thread = threading.Thread(target=send_email_alert, args=(transaction_data, alert_type))
    thread.daemon = True
    thread.start()

@app.route('/api/alerts/config', methods=['GET', 'POST'])
def alert_config():
    """Get or update alert configuration"""
    global EMAIL_ENABLED, ALERT_EMAIL, ALERT_THRESHOLD_CRITICAL, ALERT_THRESHOLD_HIGH, ALERT_THRESHOLD_MEDIUM
    
    if request.method == 'GET':
        return jsonify({
            'email_enabled': EMAIL_ENABLED,
            'sender_email': SENDER_EMAIL,
            'alert_email': ALERT_EMAIL,
            'thresholds': {
                'critical': ALERT_THRESHOLD_CRITICAL,
                'high': ALERT_THRESHOLD_HIGH,
                'medium': ALERT_THRESHOLD_MEDIUM
            }
        })
    
    elif request.method == 'POST':
        try:
            data = request.json
            
            env_lines = []
            if os.path.exists('.env'):
                with open('.env', 'r') as f:
                    env_lines = f.readlines()
            
            updates = {
                'EMAIL_ENABLED': str(data.get('email_enabled', EMAIL_ENABLED)),
                'ALERT_EMAIL': data.get('alert_email', ALERT_EMAIL),
                'ALERT_THRESHOLD_CRITICAL': str(data.get('threshold_critical', ALERT_THRESHOLD_CRITICAL)),
                'ALERT_THRESHOLD_HIGH': str(data.get('threshold_high', ALERT_THRESHOLD_HIGH)),
                'ALERT_THRESHOLD_MEDIUM': str(data.get('threshold_medium', ALERT_THRESHOLD_MEDIUM))
            }
            
            with open('.env', 'w') as f:
                updated_keys = set()
                for line in env_lines:
                    if '=' in line:
                        key = line.split('=')[0].strip()
                        if key in updates:
                            f.write(f"{key}={updates[key]}\n")
                            updated_keys.add(key)
                        else:
                            f.write(line)
                
                for key, value in updates.items():
                    if key not in updated_keys:
                        f.write(f"{key}={value}\n")
            
            load_dotenv(override=True)
            
            EMAIL_ENABLED = os.getenv('EMAIL_ENABLED', 'False').lower() == 'true'
            ALERT_EMAIL = os.getenv('ALERT_EMAIL', '')
            ALERT_THRESHOLD_CRITICAL = int(os.getenv('ALERT_THRESHOLD_CRITICAL', 90))
            ALERT_THRESHOLD_HIGH = int(os.getenv('ALERT_THRESHOLD_HIGH', 75))
            ALERT_THRESHOLD_MEDIUM = int(os.getenv('ALERT_THRESHOLD_MEDIUM', 50))
            
            return jsonify({'success': True, 'message': 'Alert configuration updated'})
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/test', methods=['POST'])
def test_alert():
    """Send test alert email"""
    try:
        test_data = {
            'transaction_id': 'TEST_12345',
            'risk_score': 95,
            'alert_level': 'CRITICAL',
            'amount': 150,
            'num_inputs': 50,
            'num_outputs': 60,
            'fee': 0.000001,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'recommended_action': 'This is a test alert. Freeze wallet immediately and investigate.'
        }
        
        success = send_email_alert(test_data, 'CRITICAL')
        
        if success:
            return jsonify({'success': True, 'message': 'Test alert sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Email is disabled or failed to send'})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/alerts/history')
def get_alert_history():
    """Get history of triggered alerts"""
    try:
        history = load_history()
        
        alerts = [
            h for h in history 
            if h.get('risk_score', 0) >= ALERT_THRESHOLD_MEDIUM
        ]
        
        alert_stats = {
            'critical': len([a for a in alerts if a.get('risk_score', 0) >= ALERT_THRESHOLD_CRITICAL]),
            'high': len([a for a in alerts if ALERT_THRESHOLD_HIGH <= a.get('risk_score', 0) < ALERT_THRESHOLD_CRITICAL]),
            'medium': len([a for a in alerts if ALERT_THRESHOLD_MEDIUM <= a.get('risk_score', 0) < ALERT_THRESHOLD_HIGH])
        }
        
        return jsonify({
            'alerts': alerts[:50],
            'stats': alert_stats,
            'total': len(alerts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    

# ============================================
# WATCHLIST API ENDPOINTS
# ============================================

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def manage_watchlist_legacy():
    """Legacy watchlist endpoint for compatibility"""
    if request.method == 'GET':
        try:
            watchlist_items = db.get_all_watchlist()
            return jsonify({
                'success': True,
                'data': watchlist_items,
                'watchlist': [item['value'] for item in watchlist_items]
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    
    elif request.method == 'POST':
        try:
            data = request.json
            transaction_id = data.get('transaction_id')
            
            if transaction_id:
                result = db.add_to_watchlist(
                    type='transaction',
                    value=str(transaction_id),
                    risk_level='HIGH',
                    reason='Added from dashboard',
                    tags=[],
                    notes=f'Transaction ID: {transaction_id}'
                )
                return jsonify(result)
            else:
                return jsonify({'success': False, 'error': 'Missing transaction_id'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/items', methods=['GET'])
def get_watchlist_items():
    """Get all watchlist items"""
    try:
        watchlist_items = db.get_all_watchlist()
        return jsonify({
            'success': True,
            'data': watchlist_items,
            'count': len(watchlist_items)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist_endpoint():
    """Add new item to watchlist"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['type', 'value', 'risk_level', 'reason']
        for field in required_fields:
            if field not in data or not data[field]:
                return jsonify({
                    'success': False,
                    'error': f'Missing or empty required field: {field}'
                }), 400
        
        # Check if value already exists
        if db.check_watchlist(data['value']):
            return jsonify({
                'success': False,
                'error': f'{data["type"].title()} "{data["value"]}" is already in the watchlist'
            }), 409
        
        result = db.add_to_watchlist(
            type=data['type'],
            value=data['value'],
            risk_level=data['risk_level'],
            reason=data['reason'],
            tags=data.get('tags', []) if isinstance(data.get('tags'), list) else [],
            notes=data.get('notes', '')
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/<int:watchlist_id>', methods=['DELETE'])
def remove_from_watchlist_endpoint(watchlist_id):
    """Remove item from watchlist"""
    try:
        result = db.remove_from_watchlist(watchlist_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/check/<value>', methods=['GET'])
def check_watchlist_endpoint(value):
    """Check if value is on watchlist"""
    try:
        is_watched = db.check_watchlist(value)
        return jsonify({
            'success': True,
            'is_watchlisted': is_watched,
            'value': value
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/stats', methods=['GET'])
def get_watchlist_stats_endpoint():
    """Get watchlist statistics"""
    try:
        stats = db.get_watchlist_stats()
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/watchlist/export', methods=['GET'])
def export_watchlist_endpoint():
    """Export watchlist as CSV"""
    try:
        watchlist_items = db.get_all_watchlist()
        
        # Create CSV
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['ID', 'Type', 'Value', 'Risk Level', 'Reason', 'Tags', 
                        'Activity Count', 'Added Date', 'Last Seen', 'Total Volume', 'Notes'])
        
        # Write data
        for item in watchlist_items:
            writer.writerow([
                item['id'],
                item['type'],
                item['value'],
                item['risk_level'],
                item['reason'],
                ', '.join(item['tags']) if item['tags'] else '',
                item['activity_count'],
                item['added_date'],
                item['last_seen'] or 'N/A',
                item['total_volume'],
                item['notes'] or ''
            ])
        
        output.seek(0)
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment; filename=watchlist_export.csv'}
        )
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# Update the predict_custom endpoint to check watchlist
@app.route('/api/predict_custom', methods=['POST'])
def predict_custom():
    """Predict fraud for custom transaction with watchlist checking"""
    try:
        data = request.json
        
        amount = float(data.get('amount', 0))
        num_inputs = int(data.get('num_inputs', 1))
        num_outputs = int(data.get('num_outputs', 1))
        # Compute fee from inputs/outputs using same formula as dashboard
        fee = (
            BASE_TRANSACTION_FEE +
            (num_inputs * 0.00002) +
            (num_outputs * 0.000015)
        )
        sender_address = data.get('sender_address', '')  # New field
        receiver_address = data.get('receiver_address', '')  # New field
        
        # Check if addresses are on watchlist
        sender_watchlisted = db.check_watchlist(sender_address) if sender_address else False
        receiver_watchlisted = db.check_watchlist(receiver_address) if receiver_address else False
        
        # Calculate base risk
        base_risk = 15.0
        risk_score = base_risk
        risk_factors = []
        
        # Watchlist check - HIGHEST PRIORITY
        if sender_watchlisted or receiver_watchlisted:
            risk_score += 40  # Major risk increase
            if sender_watchlisted:
                risk_factors.append("⚠️ Sender address is WATCHLISTED")
                db.update_activity(sender_address, amount)
            if receiver_watchlisted:
                risk_factors.append("⚠️ Receiver address is WATCHLISTED")
                db.update_activity(receiver_address, amount)
        
        # Amount-based risk
        if amount > 100:
            risk_increase = min(30, (amount / 100) * 5)
            risk_score += risk_increase
            risk_factors.append(f"Large transaction amount: {amount} BTC (+{risk_increase:.1f}%)")
        elif amount > 50:
            risk_score += 15
            risk_factors.append(f"Medium-high amount: {amount} BTC (+15%)")
        
        # Input/Output complexity
        if num_inputs > 50 or num_outputs > 50:
            risk_score += 25
            risk_factors.append(f"High I/O complexity: {num_inputs} inputs, {num_outputs} outputs (+25%)")
        elif num_inputs > 10 or num_outputs > 10:
            risk_score += 12
            risk_factors.append(f"Moderate I/O complexity (+12%)")
        
        # Fee analysis
        expected_fee = amount * 0.0001
        if fee < expected_fee * 0.5 or fee > expected_fee * 3:
            risk_score += 20
            risk_factors.append(f"Unusual fee: {fee} BTC (expected ~{expected_fee:.6f}) (+20%)")
        
        # I/O ratio
        if num_inputs > 0 and num_outputs > 0:
            io_ratio = num_inputs / num_outputs
            if io_ratio > 10 or io_ratio < 0.1:
                risk_score += 12
                risk_factors.append(f"Suspicious I/O ratio: {io_ratio:.2f} (+12%)")
        
        # Mixing pattern detection
        if num_inputs > 5 and num_outputs > 5:
            risk_score += 30
            risk_factors.append("Possible mixing/tumbling pattern detected (+30%)")
        
        # Cap risk score
        risk_score = min(100, risk_score)
        
        # Determine alert level
        if risk_score >= 90:
            alert_level = "CRITICAL"
            alert_color = "red"
            recommendation = "🚨 IMMEDIATE ACTION: Freeze transaction and initiate investigation"
        elif risk_score >= 75:
            alert_level = "HIGH"
            alert_color = "orange"
            recommendation = "⚠️ URGENT: Manual review required within 1 hour"
        elif risk_score >= 50:
            alert_level = "MEDIUM"
            alert_color = "yellow"
            recommendation = "⚡ CAUTION: Add to watchlist and monitor closely"
        elif risk_score >= 25:
            alert_level = "LOW"
            alert_color = "lightblue"
            recommendation = "📝 LOG: Review when convenient"
        else:
            alert_level = "NORMAL"
            alert_color = "green"
            recommendation = "✅ SAFE: Continue monitoring standard procedures"
        
        # Generate transaction ID
        tx_id = hashlib.sha256(f"{amount}{num_inputs}{num_outputs}{fee}{time.time()}".encode()).hexdigest()[:16]
        
        # Save to history
        transaction_data = {
            'transaction_id': tx_id,
            'timestamp': datetime.now().isoformat(),
            'amount': amount,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fee': fee,
            'risk_score': round(risk_score, 2),
            'alert_level': alert_level,
            'prediction': 1 if risk_score >= 50 else 0,
            'actual': None,
            'risk_factors': risk_factors,
            'recommendation': recommendation,
            'sender_watchlisted': sender_watchlisted,
            'receiver_watchlisted': receiver_watchlisted
        }
        
        add_to_history(transaction_data)
        
        return jsonify({
            'success': True,
            'transaction_id': tx_id,
            'risk_score': round(risk_score, 2),
            'alert_level': alert_level,
            'alert_color': alert_color,
            'prediction': 'Fraudulent' if risk_score >= 50 else 'Legitimate',
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'watchlist_alert': sender_watchlisted or receiver_watchlisted,
            'sender_watchlisted': sender_watchlisted,
            'receiver_watchlisted': receiver_watchlisted
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================
# WALLET API ENDPOINTS
# ============================================

@app.route('/api/wallet/generate', methods=['POST'])
def api_wallet_generate():
    """Generate a new simulated wallet (no blockchain)."""
    private_key, public_key, address = generate_wallet_keys()
    return jsonify({
        'success': True,
        'private_key': private_key,
        'public_key': public_key,
        'address': address
    })


@app.route('/api/wallet/sign', methods=['POST'])
def api_wallet_sign():
    """Sign an arbitrary message using a hex-encoded private key."""
    data = request.json or {}
    private_key = data.get('private_key')
    message = data.get('message', '')

    if not private_key or not message:
        return jsonify({'success': False, 'error': 'private_key and message are required'}), 400

    try:
        signature = sign_wallet_message(private_key, message)
        # Derive public key and address from the provided private key
        sk = SigningKey.from_string(bytes.fromhex(private_key), curve=SECP256k1)
        vk = sk.get_verifying_key()
        public_key = vk.to_string().hex()
        address = public_key_to_address(public_key)

        return jsonify({
            'success': True,
            'signature': signature,
            'public_key': public_key,
            'address': address
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/wallet/verify', methods=['POST'])
def api_wallet_verify():
    """Verify that a signature matches a message and public key."""
    data = request.json or {}
    public_key = data.get('public_key', '')
    message = data.get('message', '')
    signature = data.get('signature', '')

    if not public_key or not message or not signature:
        return jsonify({'success': False, 'error': 'public_key, message and signature are required'}), 400

    try:
        is_valid = verify_wallet_message(public_key, message, signature)
        address = public_key_to_address(public_key) if is_valid else None
        return jsonify({
            'success': True,
            'valid': is_valid,
            'address': address
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/wallet/scan', methods=['POST'])
def api_wallet_scan():
    """Verify a signed message and run it through the risk engine."""
    try:
        data = request.json or {}

        public_key = data.get('public_key', '')
        message = data.get('message', '')
        signature = data.get('signature', '')
        amount = float(data.get('amount', 0) or 0)
        receiver_address = data.get('receiver_address', '') or ''

        if not public_key or not message or not signature:
            return jsonify({'success': False, 'error': 'public_key, message and signature are required'}), 400

        # Signature validity
        signature_valid = verify_wallet_message(public_key, message, signature)
        wallet_address = public_key_to_address(public_key)

        # Obfuscate identifiers before any storage/analysis
        obfuscated_sender = obfuscate_identifier(wallet_address) if wallet_address else None
        obfuscated_receiver = obfuscate_identifier(receiver_address) if receiver_address else None

        # Watchlist checks use real addresses but are never exposed to ML directly
        sender_watchlisted = db.check_watchlist(wallet_address) if wallet_address else False
        receiver_watchlisted = db.check_watchlist(receiver_address) if receiver_address else False

        # --- Base risk using existing risk engine ---
        base_tx = {
            'amount': amount,
            'num_inputs': 1,
            'num_outputs': 1,
            'fee': 0.0,
        }
        base_result = calculate_risk_score(base_tx)
        risk_score = float(base_result.get('risk_score', 10.0))
        risk_factors = list(base_result.get('risk_factors', []))

        # Invalid signature is an immediate red flag
        if not signature_valid:
            risk_score = max(risk_score, 95.0)
            risk_factors.append("Signature verification failed (message or signature tampered)")

        # Watchlist boost
        if sender_watchlisted or receiver_watchlisted:
            risk_score += 40
            if sender_watchlisted:
                risk_factors.append("⚠️ Sender wallet is WATCHLISTED")
                db.update_activity(wallet_address, amount)
            if receiver_watchlisted and receiver_address:
                risk_factors.append("⚠️ Receiver wallet is WATCHLISTED")
                db.update_activity(receiver_address, amount)

        # Message-content heuristics for common scam language
        lowered = message.lower()
        keyword_boosts = {
            "urgent": 10,
            "immediately": 10,
            "now": 5,
            "gift": 8,
            "airdrop": 12,
            "giveaway": 12,
            "double": 10,
            "seed phrase": 25,
            "private key": 25,
            "withdraw all": 15,
        }

        for kw, inc in keyword_boosts.items():
            if kw in lowered:
                risk_score += inc
                risk_factors.append(f"Suspicious phrase detected in message: '{kw}' (+{inc}%)")

        # Clamp risk score
        risk_score = max(0.0, min(100.0, risk_score))

        # Determine alert level and recommendation (Bootstrap-friendly colors)
        if risk_score >= 90:
            alert_level = "CRITICAL"
            alert_color = "danger"
            recommendation = "🚨 IMMEDIATE ACTION: Reject transaction and investigate wallet owner"
        elif risk_score >= 75:
            alert_level = "HIGH"
            alert_color = "warning"
            recommendation = "⚠️ URGENT: Manual review required within 1 hour"
        elif risk_score >= 50:
            alert_level = "MEDIUM"
            alert_color = "info"
            recommendation = "⚡ CAUTION: Add wallet to watchlist and monitor closely"
        elif risk_score >= 25:
            alert_level = "LOW"
            alert_color = "secondary"
            recommendation = "📝 LOG: Record and review during routine checks"
        else:
            alert_level = "NORMAL"
            alert_color = "success"
            recommendation = "✅ SAFE: No immediate action required"

        prediction_label = "Fraudulent" if risk_score >= 50 else "Legitimate"

        # Create anonymized history entry (no raw message stored)
        message_hash = hashlib.sha256(message.encode('utf-8')).hexdigest()[:16]
        tx_id = hashlib.sha256(
            f"{wallet_address}{receiver_address}{amount}{message_hash}{time.time()}wallet".encode('utf-8')
        ).hexdigest()[:16]

        history_entry = {
            'transaction_id': tx_id,
            'timestamp': datetime.now().isoformat(),
            'amount': amount,
            'num_inputs': 1,
            'num_outputs': 1,
            'fee': 0.0,
            'risk_score': round(risk_score, 2),
            'alert_level': alert_level,
            'prediction': 1 if risk_score >= 50 else 0,
            'actual': None,
            'risk_factors': risk_factors,
            'recommended_action': recommendation,
            'sender_watchlisted': sender_watchlisted,
            'receiver_watchlisted': receiver_watchlisted,
            'sender_id_obfuscated': obfuscated_sender,
            'receiver_id_obfuscated': obfuscated_receiver,
            'source': 'wallet',
            'message_hash': message_hash,
        }

        add_to_history(history_entry)

        return jsonify({
            'success': True,
            'transaction_id': tx_id,
            'signature_valid': signature_valid,
            'recovered_address': wallet_address,
            'risk_score': round(risk_score, 2),
            'alert_level': alert_level,
            'alert_color': alert_color,
            'prediction': prediction_label,
            'recommendation': recommendation,
            'risk_factors': risk_factors,
            'watchlist_alert': sender_watchlisted or receiver_watchlisted,
            'sender_watchlisted': sender_watchlisted,
            'receiver_watchlisted': receiver_watchlisted,
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import time



    
# ============================================
# ANALYTICS API ENDPOINTS
# ============================================

@app.route('/api/analytics/risk-trend', methods=['GET'])
def get_risk_trend():
    """Get risk score trend over last 7 days"""
    try:
        history = load_history()
        
        if len(history) == 0:
            # Return sample data if no history
            return jsonify({
                'success': True,
                'data': [
                    {'date': f'Day {i+1}', 'avg_risk': 0, 'count': 0}
                    for i in range(7)
                ]
            })
        
        # Get date 7 days ago
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        # Group by date
        daily_data = defaultdict(lambda: {'sum': 0, 'count': 0})
        
        for tx in history:
            try:
                tx_date = datetime.fromisoformat(tx['timestamp']).date()
                if datetime.combine(tx_date, datetime.min.time()) >= seven_days_ago:
                    date_str = tx_date.strftime('%Y-%m-%d')
                    daily_data[date_str]['count'] += 1
                    daily_data[date_str]['sum'] += tx['risk_score']
            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue
        
        # Calculate averages for last 7 days
        trend_data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=6-i)).date()
            date_str = date.strftime('%Y-%m-%d')
            
            if date_str in daily_data and daily_data[date_str]['count'] > 0:
                avg_risk = daily_data[date_str]['sum'] / daily_data[date_str]['count']
                count = daily_data[date_str]['count']
            else:
                avg_risk = 0
                count = 0
            
            trend_data.append({
                'date': date.strftime('%b %d'),
                'avg_risk': round(avg_risk, 2),
                'count': count
            })
        
        return jsonify({
            'success': True,
            'data': trend_data
        })
    except Exception as e:
        print(f"❌ Error in risk-trend: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/alert-timeline', methods=['GET'])
def get_alert_timeline():
    """Get alert distribution over time"""
    try:
        history = load_history()
        
        if len(history) == 0:
            return jsonify({
                'success': True,
                'data': [
                    {
                        'date': f'Day {i+1}',
                        'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'normal': 0
                    }
                    for i in range(7)
                ]
            })
        
        # Get last 7 days
        seven_days_ago = datetime.now() - timedelta(days=7)
        
        # Group by date and alert level
        daily_alerts = defaultdict(lambda: {
            'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NORMAL': 0
        })
        
        for tx in history:
            try:
                tx_date = datetime.fromisoformat(tx['timestamp']).date()
                if datetime.combine(tx_date, datetime.min.time()) >= seven_days_ago:
                    date_str = tx_date.strftime('%Y-%m-%d')
                    alert_level = tx.get('alert_level', 'NORMAL').upper()
                    if alert_level in daily_alerts[date_str]:
                        daily_alerts[date_str][alert_level] += 1
            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue
        
        # Format data
        timeline_data = []
        for i in range(7):
            date = (datetime.now() - timedelta(days=6-i)).date()
            date_str = date.strftime('%Y-%m-%d')
            
            timeline_data.append({
                'date': date.strftime('%b %d'),
                'critical': daily_alerts[date_str]['CRITICAL'],
                'high': daily_alerts[date_str]['HIGH'],
                'medium': daily_alerts[date_str]['MEDIUM'],
                'low': daily_alerts[date_str]['LOW'],
                'normal': daily_alerts[date_str]['NORMAL']
            })
        
        return jsonify({
            'success': True,
            'data': timeline_data
        })
    except Exception as e:
        print(f"❌ Error in alert-timeline: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/hourly-heatmap', methods=['GET'])
def get_hourly_heatmap():
    """Get risk distribution by hour of day"""
    try:
        history = load_history()
        
        if len(history) == 0:
            return jsonify({
                'success': True,
                'data': [
                    {'hour': f"{h:02d}:00", 'avg_risk': 0, 'count': 0}
                    for h in range(24)
                ]
            })
        
        # Group by hour
        hourly_data = defaultdict(lambda: {'sum': 0, 'count': 0})
        
        for tx in history:
            try:
                hour = datetime.fromisoformat(tx['timestamp']).hour
                hourly_data[hour]['sum'] += tx['risk_score']
                hourly_data[hour]['count'] += 1
            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue
        
        # Calculate averages
        heatmap_data = []
        for hour in range(24):
            if hourly_data[hour]['count'] > 0:
                avg_risk = hourly_data[hour]['sum'] / hourly_data[hour]['count']
            else:
                avg_risk = 0
            
            heatmap_data.append({
                'hour': f"{hour:02d}:00",
                'avg_risk': round(avg_risk, 2),
                'count': hourly_data[hour]['count']
            })
        
        return jsonify({
            'success': True,
            'data': heatmap_data
        })
    except Exception as e:
        print(f"❌ Error in hourly-heatmap: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/volume-analysis', methods=['GET'])
def get_volume_analysis():
    """Get transaction volume analysis"""
    try:
        history = load_history()
        
        # Volume ranges
        ranges = [
            {'label': '0-1 BTC', 'min': 0, 'max': 1, 'count': 0, 'sum_risk': 0, 'avg_risk': 0},
            {'label': '1-10 BTC', 'min': 1, 'max': 10, 'count': 0, 'sum_risk': 0, 'avg_risk': 0},
            {'label': '10-50 BTC', 'min': 10, 'max': 50, 'count': 0, 'sum_risk': 0, 'avg_risk': 0},
            {'label': '50-100 BTC', 'min': 50, 'max': 100, 'count': 0, 'sum_risk': 0, 'avg_risk': 0},
            {'label': '100+ BTC', 'min': 100, 'max': float('inf'), 'count': 0, 'sum_risk': 0, 'avg_risk': 0}
        ]
        
        if len(history) == 0:
            return jsonify({
                'success': True,
                'data': [
                    {'label': r['label'], 'count': 0, 'avg_risk': 0}
                    for r in ranges
                ]
            })
        
        for tx in history:
            try:
                amount = float(tx.get('amount', 0))
                risk_score = float(tx.get('risk_score', 0))
                
                for range_data in ranges:
                    if range_data['min'] <= amount < range_data['max']:
                        range_data['count'] += 1
                        range_data['sum_risk'] += risk_score
                        break
            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue
        
        # Calculate averages
        result_data = []
        for range_data in ranges:
            if range_data['count'] > 0:
                avg_risk = range_data['sum_risk'] / range_data['count']
            else:
                avg_risk = 0
            
            result_data.append({
                'label': range_data['label'],
                'count': range_data['count'],
                'avg_risk': round(avg_risk, 2)
            })
        
        return jsonify({
            'success': True,
            'data': result_data
        })
    except Exception as e:
        print(f"❌ Error in volume-analysis: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/accuracy-metrics', methods=['GET'])
def get_accuracy_metrics():
    """Get prediction accuracy metrics"""
    try:
        history = load_history()
        
        # Filter transactions with actual labels
        labeled = [tx for tx in history if tx.get('actual') is not None]
        
        if len(labeled) == 0:
            return jsonify({
                'success': True,
                'data': {
                    'total': 0,
                    'correct': 0,
                    'incorrect': 0,
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'true_positives': 0,
                    'false_positives': 0,
                    'true_negatives': 0,
                    'false_negatives': 0
                }
            })
        
        # Calculate metrics
        correct = sum(1 for tx in labeled if tx.get('prediction') == tx.get('actual'))
        
        # True Positives, False Positives, etc.
        tp = sum(1 for tx in labeled if tx.get('prediction') == 1 and tx.get('actual') == 1)
        fp = sum(1 for tx in labeled if tx.get('prediction') == 1 and tx.get('actual') == 0)
        tn = sum(1 for tx in labeled if tx.get('prediction') == 0 and tx.get('actual') == 0)
        fn = sum(1 for tx in labeled if tx.get('prediction') == 0 and tx.get('actual') == 1)
        
        accuracy = (correct / len(labeled)) * 100 if len(labeled) > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
        
        return jsonify({
            'success': True,
            'data': {
                'total': len(labeled),
                'correct': correct,
                'incorrect': len(labeled) - correct,
                'accuracy': round(accuracy, 2),
                'precision': round(precision, 2),
                'recall': round(recall, 2),
                'true_positives': tp,
                'false_positives': fp,
                'true_negatives': tn,
                'false_negatives': fn
            }
        })
    except Exception as e:
        print(f"❌ Error in accuracy-metrics: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/top-patterns', methods=['GET'])
def get_top_patterns():
    """Get top fraud patterns detected"""
    try:
        history = load_history()
        
        if len(history) == 0:
            return jsonify({
                'success': True,
                'data': [
                    {'pattern': 'No data yet', 'count': 0}
                ]
            })
        
        # Analyze patterns
        patterns = {
            'high_amount': {'count': 0, 'label': 'Large Transactions (>50 BTC)'},
            'high_io': {'count': 0, 'label': 'Complex I/O (>10 inputs/outputs)'},
            'mixing': {'count': 0, 'label': 'Mixing Patterns (High I/O both)'},
            'unusual_fee': {'count': 0, 'label': 'Unusual Transaction Fees'},
            'watchlist': {'count': 0, 'label': 'Watchlisted Addresses'}
        }
        
        for tx in history:
            try:
                amount = float(tx.get('amount', 0))
                num_inputs = int(tx.get('num_inputs', 0))
                num_outputs = int(tx.get('num_outputs', 0))
                fee = float(tx.get('fee', 0))
                
                # High amount
                if amount > 50:
                    patterns['high_amount']['count'] += 1
                
                # High I/O
                if num_inputs > 10 or num_outputs > 10:
                    patterns['high_io']['count'] += 1
                
                # Mixing pattern
                if num_inputs > 5 and num_outputs > 5:
                    patterns['mixing']['count'] += 1
                
                # Unusual fee
                if amount > 0:
                    expected_fee = amount * 0.0001
                    if fee > 0 and (fee < expected_fee * 0.5 or fee > expected_fee * 3):
                        patterns['unusual_fee']['count'] += 1
                
                # Watchlist
                if tx.get('sender_watchlisted') or tx.get('receiver_watchlisted'):
                    patterns['watchlist']['count'] += 1
                    
            except Exception as e:
                print(f"Error processing transaction: {e}")
                continue
        
        # Sort by count
        sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['count'], reverse=True)
        
        pattern_data = [
            {
                'pattern': item[1]['label'],
                'count': item[1]['count']
            }
            for item in sorted_patterns[:5]  # Top 5
        ]
        
        # If all counts are 0, return friendly message
        if all(p['count'] == 0 for p in pattern_data):
            pattern_data = [{'pattern': 'No patterns detected yet', 'count': 0}]
        
        return jsonify({
            'success': True,
            'data': pattern_data
        })
    except Exception as e:
        print(f"❌ Error in top-patterns: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500
    

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CHAINGUARD DASHBOARD STARTING...")
    print("=" * 70)
    print("\n✅ Dashboard running at: http://127.0.0.1:5000")
    print("\n📊 Open your browser and go to: http://127.0.0.1:5000")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)