from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
import joblib
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

if __name__ == '__main__':
    print("=" * 70)
    print("🚀 CHAINGUARD DASHBOARD STARTING...")
    print("=" * 70)
    print("\n✅ Dashboard running at: http://127.0.0.1:5000")
    print("\n📊 Open your browser and go to: http://127.0.0.1:5000")
    print("\n⚠️  Press CTRL+C to stop the server")
    print("=" * 70)
    
    app.run(debug=True, host='0.0.0.0', port=5000)