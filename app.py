from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
import os
from fpdf import FPDF
from datetime import datetime
import io
from flask import send_file, make_response
import csv
from werkzeug.utils import secure_filename
import os
import json


app = Flask(__name__)

# Load model and data
MODELS_DIR = 'models'
OUTPUTS_DIR = 'outputs'
PROCESSED_DIR = 'processed_data'

model = joblib.load(os.path.join(MODELS_DIR, 'chainguard_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))

# Load results
results_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'all_transaction_results.csv'))
top_10_df = pd.read_csv(os.path.join(OUTPUTS_DIR, 'top_10_risky_transactions.csv'))

# Watchlist (in-memory for demo)
watchlist = []

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    stats = {
        'total_scanned': len(results_df),
        'critical_alerts': len(results_df[results_df['Alert_Level'] == 'CRITICAL']),
        'high_alerts': len(results_df[results_df['Alert_Level'] == 'HIGH']),
        'medium_alerts': len(results_df[results_df['Alert_Level'] == 'MEDIUM']),
        'accuracy': f"{(results_df['Correct_Prediction'] == '✅').sum() / len(results_df) * 100:.2f}",
        'avg_risk_score': f"{results_df['Risk_Score'].mean():.2f}",
        'watchlist_count': len(watchlist)
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

@app.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def manage_watchlist():
    """Manage watchlist"""
    global watchlist
    
    if request.method == 'GET':
        return jsonify(watchlist)
    
    elif request.method == 'POST':
        data = request.json
        tx_id = data.get('transaction_id')
        
        if tx_id not in watchlist:
            watchlist.append(tx_id)
        
        return jsonify({'status': 'added', 'watchlist': watchlist})
    
    elif request.method == 'DELETE':
        data = request.json
        tx_id = data.get('transaction_id')
        
        if tx_id in watchlist:
            watchlist.remove(tx_id)
        
        return jsonify({'status': 'removed', 'watchlist': watchlist})

@app.route('/api/predict', methods=['POST'])
def predict_new_transaction():
    """Predict risk for new transaction (demo with random test data)"""
    # In real scenario, this would receive transaction features
    # For demo, we pick a random transaction from test set
    
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
    random_idx = np.random.randint(0, len(X_test))
    
    features = X_test[random_idx]
    
    # Predict
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1]
    risk_score = int(probability * 100)
    
    # Generate alert
    if risk_score >= 90:
        alert_level = 'CRITICAL'
        alert_message = 'IMMEDIATE ACTION REQUIRED'
        action = 'Freeze wallet immediately'
    elif risk_score >= 75:
        alert_level = 'HIGH'
        alert_message = 'HIGH RISK DETECTED'
        action = 'Flag wallet for investigation'
    elif risk_score >= 50:
        alert_level = 'MEDIUM'
        alert_message = 'SUSPICIOUS ACTIVITY'
        action = 'Add to watchlist'
    else:
        alert_level = 'LOW'
        alert_message = 'Minor anomaly'
        action = 'Log for review'
    
    result = {
        'transaction_id': f"NEW_{random_idx}",
        'prediction': 'Fraud' if prediction == 1 else 'Normal',
        'risk_score': risk_score,
        'alert_level': alert_level,
        'alert_message': alert_message,
        'recommended_action': action
    }
    
    return jsonify(result)

# ============================================
# IMPROVED CUSTOM TRANSACTION PREDICTION
# ============================================
@app.route('/api/predict_custom', methods=['POST'])
def predict_custom_transaction():
    """Predict risk for custom transaction with user input"""
    try:
        data = request.json
        
        # Get user inputs
        transaction_amount = float(data.get('amount', 1.5))
        num_inputs = int(data.get('num_inputs', 2))
        num_outputs = int(data.get('num_outputs', 2))
        transaction_fee = float(data.get('fee', 0.0001))
        
        # ============================================
        # FIXED: Use MEDIAN base risk instead of random
        # ============================================
        
        # Load test data
        X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
        y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
        
        # Calculate median risk from NORMAL transactions only
        normal_indices = np.where(y_test == 0)[0]  # Get only normal transactions
        
        if len(normal_indices) > 100:
            # Sample 100 random normal transactions
            sample_indices = np.random.choice(normal_indices, 100, replace=False)
            sample_features = X_test[sample_indices]
            
            # Get their predictions
            sample_predictions = model.predict_proba(sample_features)[:, 1]
            
            # Use median as base risk
            base_probability = np.median(sample_predictions)
        else:
            # Fallback: use 10% as typical normal transaction base
            base_probability = 0.10
        
        # ============================================
        # IMPROVED RISK CALCULATION
        # ============================================
        
        # Calculate risk adjustments
        risk_adjustment = 0.0
        
        # 1. Amount-based risk
        if transaction_amount > 100:
            risk_adjustment += 0.30
        elif transaction_amount > 50:
            risk_adjustment += 0.20
        elif transaction_amount > 20:
            risk_adjustment += 0.10
        elif transaction_amount > 10:
            risk_adjustment += 0.05
        
        # 2. Input/Output complexity
        total_io = num_inputs + num_outputs
        if total_io > 50:
            risk_adjustment += 0.25
        elif total_io > 30:
            risk_adjustment += 0.15
        elif total_io > 15:
            risk_adjustment += 0.08
        elif total_io > 10:
            risk_adjustment += 0.03
        
        # 3. Fee anomalies
        # Too low fee for large transaction
        if transaction_amount > 10 and transaction_fee < 0.00001:
            risk_adjustment += 0.20
        elif transaction_amount > 5 and transaction_fee < 0.00005:
            risk_adjustment += 0.10
        
        # Extremely high fee (suspicious)
        if transaction_fee > 0.01:
            risk_adjustment += 0.15
        elif transaction_fee > 0.005:
            risk_adjustment += 0.08
        
        # 4. Unusual input/output ratio
        if num_inputs > 0:
            io_ratio = num_outputs / num_inputs
            if io_ratio > 10 or io_ratio < 0.1:
                risk_adjustment += 0.12
            elif io_ratio > 5 or io_ratio < 0.2:
                risk_adjustment += 0.06
        
        # 5. Many inputs (mixing/tumbling pattern)
        if num_inputs > 20:
            risk_adjustment += 0.15
        elif num_inputs > 10:
            risk_adjustment += 0.08
        
        # 6. Many outputs (distribution pattern)
        if num_outputs > 20:
            risk_adjustment += 0.15
        elif num_outputs > 10:
            risk_adjustment += 0.08
        
        # Calculate final probability
        adjusted_probability = base_probability + risk_adjustment
        
        # Cap between 0 and 1
        adjusted_probability = max(0.0, min(1.0, adjusted_probability))
        
        # Convert to risk score
        risk_score = int(adjusted_probability * 100)
        
        # Determine prediction
        prediction = 1 if risk_score >= 50 else 0
        
        # ============================================
        # GENERATE ALERT
        # ============================================
        
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
        
        result = {
            'transaction_id': f"CUSTOM_{random_idx}",
            'prediction': 'Fraud' if prediction == 1 else 'Normal',
            'risk_score': risk_score,
            'alert_level': alert_level,
            'alert_message': alert_message,
            'recommended_action': action,
            'color': color,
            'details': {
                'amount': transaction_amount,
                'num_inputs': num_inputs,
                'num_outputs': num_outputs,
                'fee': transaction_fee,
                'base_risk': int(base_probability * 100),
                'adjustment': f"+{int(risk_adjustment * 100)}"
            }
        }
                # Save to history
        history_entry = {
            'transaction_id': result['transaction_id'],
            'amount': transaction_amount,
            'num_inputs': num_inputs,
            'num_outputs': num_outputs,
            'fee': transaction_fee,
            'risk_score': risk_score,
            'alert_level': alert_level,
            'prediction': result['prediction'],
            'alert_message': alert_message,
            'recommended_action': action
        }
        save_to_history(history_entry)
        
        return jsonify(result)
        
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400



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
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Send as download
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
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 20)
        pdf.cell(0, 10, 'ChainGuard Security Report', ln=True, align='C')
        
        pdf.set_font('Arial', '', 12)
        pdf.cell(0, 10, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
        pdf.ln(10)
        
        # Statistics
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
        
        # Top 10 Riskiest
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
        
        # Save PDF to memory
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        # Send as download
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
        # Find transaction
        tx = results_df[results_df['Transaction_ID'] == tx_id]
        
        if len(tx) == 0:
            return jsonify({'error': 'Transaction not found'}), 404
        
        tx = tx.iloc[0]
        
        # Create PDF
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font('Arial', 'B', 18)
        pdf.cell(0, 10, 'Transaction Analysis Report', ln=True, align='C')
        pdf.ln(10)
        
        # Transaction Details
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
        
        # Recommended Action
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 8, 'Recommended Action', ln=True)
        pdf.ln(3)
        
        pdf.set_font('Arial', '', 10)
        pdf.multi_cell(0, 7, str(tx['Recommended_Action']))
        
        pdf.ln(5)
        
        # Footer
        pdf.set_y(-30)
        pdf.set_font('Arial', 'I', 8)
        pdf.cell(0, 10, f'Generated by ChainGuard | {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', align='C')
        
        # Save PDF to memory
        pdf_output = pdf.output(dest='S').encode('latin-1')
        
        # Send as download
        response = make_response(pdf_output)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename=transaction_{tx_id}_report.pdf'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500




# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ============================================
# FEATURE 2: BATCH ANALYSIS
# ============================================

@app.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    """Upload and analyze CSV file with multiple transactions"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        
        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Validate columns
        required_columns = ['amount', 'num_inputs', 'num_outputs', 'fee']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.remove(filepath)  # Clean up
            return jsonify({
                'error': f'Missing required columns: {", ".join(missing_columns)}',
                'required': required_columns
            }), 400
        
        # Analyze each transaction
        results = []
        
        for idx, row in df.iterrows():
            try:
                # Get values
                amount = float(row['amount'])
                num_inputs = int(row['num_inputs'])
                num_outputs = int(row['num_outputs'])
                fee = float(row['fee'])
                
                # Calculate risk (same logic as predict_custom)
                X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
                y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'))
                
                normal_indices = np.where(y_test == 0)[0]
                sample_indices = np.random.choice(normal_indices, min(100, len(normal_indices)), replace=False)
                sample_predictions = model.predict_proba(X_test[sample_indices])[:, 1]
                base_probability = np.median(sample_predictions)
                
                risk_adjustment = 0.0
                
                # Amount risk
                if amount > 100:
                    risk_adjustment += 0.30
                elif amount > 50:
                    risk_adjustment += 0.20
                elif amount > 20:
                    risk_adjustment += 0.10
                elif amount > 10:
                    risk_adjustment += 0.05
                
                # I/O complexity
                total_io = num_inputs + num_outputs
                if total_io > 50:
                    risk_adjustment += 0.25
                elif total_io > 30:
                    risk_adjustment += 0.15
                elif total_io > 15:
                    risk_adjustment += 0.08
                elif total_io > 10:
                    risk_adjustment += 0.03
                
                # Fee anomalies
                if amount > 10 and fee < 0.00001:
                    risk_adjustment += 0.20
                elif amount > 5 and fee < 0.00005:
                    risk_adjustment += 0.10
                
                if fee > 0.01:
                    risk_adjustment += 0.15
                elif fee > 0.005:
                    risk_adjustment += 0.08
                
                # I/O ratio
                if num_inputs > 0:
                    io_ratio = num_outputs / num_inputs
                    if io_ratio > 10 or io_ratio < 0.1:
                        risk_adjustment += 0.12
                    elif io_ratio > 5 or io_ratio < 0.2:
                        risk_adjustment += 0.06
                
                # Many inputs
                if num_inputs > 20:
                    risk_adjustment += 0.15
                elif num_inputs > 10:
                    risk_adjustment += 0.08
                
                # Many outputs
                if num_outputs > 20:
                    risk_adjustment += 0.15
                elif num_outputs > 10:
                    risk_adjustment += 0.08
                
                adjusted_probability = max(0.0, min(1.0, base_probability + risk_adjustment))
                risk_score = int(adjusted_probability * 100)
                
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
        
        # Calculate summary statistics
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
            'avg_risk_score': np.mean([r['risk_score'] for r in valid_results]) if valid_results else 0,
            'max_risk_score': max([r['risk_score'] for r in valid_results]) if valid_results else 0,
            'min_risk_score': min([r['risk_score'] for r in valid_results]) if valid_results else 0
        }
        
        # Save results to CSV
        results_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results_filepath = os.path.join(OUTPUTS_DIR, results_filename)
        
        results_df = pd.DataFrame(valid_results)
        results_df.to_csv(results_filepath, index=False)
        
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
        # Create template CSV
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

import json
from collections import Counter

# History file path
HISTORY_FILE = 'history.json'

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

# ============================================
# FEATURE 3: HISTORICAL TRACKING
# ============================================

@app.route('/api/history')
def get_history():
    """Get transaction analysis history"""
    try:
        history = load_history()
        
        # Pagination
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
        
        # Calculate statistics
        risk_scores = [h['risk_score'] for h in history]
        alert_levels = [h['alert_level'] for h in history]
        
        # Alert distribution
        alert_dist = Counter(alert_levels)
        
        # Daily trend (last 7 days)
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
        
        # Convert to DataFrame
        df = pd.DataFrame(history)
        
        # Create CSV
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Send as download
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f'attachment; filename=transaction_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        return response
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CHAINGUARD DASHBOARD STARTING...")
    print("=" * 70)
    print("\n✅ Dashboard running at: http://127.0.0.1:5000")
    print("\n📊 Open your browser and go to: http://127.0.0.1:5000")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)