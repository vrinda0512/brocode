from flask import Flask, render_template, jsonify, request, send_file, make_response, current_app
from collections import defaultdict
import os
import json
import threading
import time
try:
    import joblib
except Exception:
    joblib = None
import pandas as pd
import numpy as np
from datetime import datetime

# Local imports
from database import db

# Blueprint will be registered below
def _make_app():
    app = Flask(__name__, template_folder='templates')

    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
    PROCESSED_DIR = os.path.join(BASE_DIR, 'processed_data')
    HISTORY_FILE = os.path.join(BASE_DIR, 'history.json')

    # Dummy fallback model/scaler
    class DummyModel:
        def predict(self, X):
            return [0 for _ in range(len(X))]
        def predict_proba(self, X):
            return np.vstack([[1.0 - 0.05, 0.05] for _ in range(len(X))])

    class DummyScaler:
        def transform(self, X):
            return X

    # Load model (fallback to DummyModel)
    model_path = os.path.join(MODELS_DIR, 'chainguard_model.pkl')
    model = None
    if joblib and os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            model = DummyModel()
    else:
        model = DummyModel()

    # Try to load outputs CSVs
    def _safe_read_csv(path, default_cols=None):
        if os.path.exists(path):
            try:
                return pd.read_csv(path)
            except Exception:
                return pd.DataFrame(columns=default_cols or [])
        return pd.DataFrame(columns=default_cols or [])

    results_df = _safe_read_csv(os.path.join(OUTPUTS_DIR, 'all_transaction_results.csv'), default_cols=['Transaction_ID'])
    top_10_df = _safe_read_csv(os.path.join(OUTPUTS_DIR, 'top_10_risky_transactions.csv'), default_cols=['Transaction_ID'])

    # Helper utilities
    def allowed_file(filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

    def load_history():
        if not os.path.exists(HISTORY_FILE):
            return []
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return []

    def save_to_history(entry):
        history = load_history()
        history.insert(0, entry)
        try:
            with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(history, f, indent=2)
        except Exception:
            pass

    def send_email_alert(transaction_data, alert_type='HIGH'):
        # Stub: email not configured; return False to indicate not sent
        print(f"[alert] {alert_type} - {transaction_data.get('transaction_id')}")
        return False

    def trigger_alert_async(transaction_data, alert_type='HIGH'):
        def _worker():
            try:
                send_email_alert(transaction_data, alert_type)
            except Exception:
                pass
        threading.Thread(target=_worker, daemon=True).start()

    # Expose config used by routes
    app.config['model'] = model
    app.config['scaler'] = DummyScaler()
    app.config['results_df'] = results_df
    app.config['top_10_df'] = top_10_df
    app.config['PROCESSED_DIR'] = PROCESSED_DIR
    app.config['OUTPUTS_DIR'] = OUTPUTS_DIR
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['HISTORY_FILE'] = HISTORY_FILE
    app.config['ALERT_THRESHOLD_HIGH'] = 75
    app.config['ALERT_THRESHOLD_CRITICAL'] = 90
    app.config['trigger_alert_async'] = trigger_alert_async
    app.config['save_to_history'] = save_to_history
    app.config['load_history'] = load_history
    app.config['send_email_alert'] = send_email_alert
    app.config['db'] = db
    app.config['allowed_file'] = allowed_file

    # Register blueprint routes
    try:
        from routes import bp as routes_bp
        app.register_blueprint(routes_bp)
    except Exception as e:
        print(f"Failed to register routes blueprint: {e}")

    return app


app = _make_app()

@app.route('/api/analytics/hourly-heatmap', methods=['GET'])
def get_hourly_heatmap():
    """Get risk distribution by hour of day"""
    try:
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []

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
                hourly_data[hour]['sum'] += tx.get('risk_score', 0)
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


@app.route('/api/analytics/risk-trend', methods=['GET'])
def get_risk_trend():
    """Get risk score trend over last 7 days"""
    try:
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []

        if len(history) == 0:
            return jsonify({
                'success': True,
                'data': [
                    {'date': f'Day {i+1}', 'avg_risk': 0, 'count': 0}
                    for i in range(7)
                ]
            })

        seven_days_ago = datetime.now() - pd.Timedelta(days=7)
        daily_data = defaultdict(lambda: {'sum': 0, 'count': 0})

        for tx in history:
            try:
                tx_date = datetime.fromisoformat(tx['timestamp']).date()
                if datetime.combine(tx_date, datetime.min.time()) >= seven_days_ago:
                    date_str = tx_date.strftime('%Y-%m-%d')
                    daily_data[date_str]['count'] += 1
                    daily_data[date_str]['sum'] += tx.get('risk_score', 0)
            except Exception:
                continue

        trend_data = []
        for i in range(7):
            date = (datetime.now() - pd.Timedelta(days=6-i)).date()
            date_str = date.strftime('%Y-%m-%d')
            if date_str in daily_data and daily_data[date_str]['count'] > 0:
                avg_risk = daily_data[date_str]['sum'] / daily_data[date_str]['count']
                count = daily_data[date_str]['count']
            else:
                avg_risk = 0
                count = 0
            trend_data.append({'date': date.strftime('%b %d'), 'avg_risk': round(avg_risk, 2), 'count': count})

        return jsonify({'success': True, 'data': trend_data})
    except Exception as e:
        print(f"❌ Error in risk-trend: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/alert-timeline', methods=['GET'])
def get_alert_timeline():
    """Get alert distribution over time (last 7 days)"""
    try:
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []

        if len(history) == 0:
            return jsonify({'success': True, 'data': [{'date': f'Day {i+1}', 'critical': 0, 'high': 0, 'medium': 0, 'low': 0, 'normal': 0} for i in range(7)]})

        seven_days_ago = datetime.now() - pd.Timedelta(days=7)
        daily_alerts = defaultdict(lambda: {'CRITICAL': 0, 'HIGH': 0, 'MEDIUM': 0, 'LOW': 0, 'NORMAL': 0})

        for tx in history:
            try:
                tx_date = datetime.fromisoformat(tx['timestamp']).date()
                if datetime.combine(tx_date, datetime.min.time()) >= seven_days_ago:
                    date_str = tx_date.strftime('%Y-%m-%d')
                    alert_level = tx.get('alert_level', 'NORMAL').upper()
                    if alert_level in daily_alerts[date_str]:
                        daily_alerts[date_str][alert_level] += 1
            except Exception:
                continue

        timeline_data = []
        for i in range(7):
            date = (datetime.now() - pd.Timedelta(days=6-i)).date()
            date_str = date.strftime('%Y-%m-%d')
            timeline_data.append({
                'date': date.strftime('%b %d'),
                'critical': daily_alerts[date_str]['CRITICAL'],
                'high': daily_alerts[date_str]['HIGH'],
                'medium': daily_alerts[date_str]['MEDIUM'],
                'low': daily_alerts[date_str]['LOW'],
                'normal': daily_alerts[date_str]['NORMAL']
            })

        return jsonify({'success': True, 'data': timeline_data})
    except Exception as e:
        print(f"❌ Error in alert-timeline: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/analytics/volume-analysis', methods=['GET'])
def get_volume_analysis():
    """Get transaction volume analysis"""
    try:
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []
        
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
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []
        
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
        load_history_fn = current_app.config.get('load_history')
        history = load_history_fn() if load_history_fn else []
        
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