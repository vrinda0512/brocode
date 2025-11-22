from flask import Blueprint, render_template, jsonify, request, send_file, make_response, current_app, Response
import pandas as pd
import numpy as np
import io
import csv
import os
from werkzeug.utils import secure_filename
from fpdf import FPDF
from datetime import datetime

bp = Blueprint('routes', __name__)


def _get(k):
    return current_app.config.get(k)


@bp.route('/')
def home():
    return render_template('dashboard.html')


@bp.route('/api/stats')
def get_stats():
    results_df = _get('results_df')
    watchlist = _get('watchlist') or []
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


@bp.route('/api/top10')
def get_top10():
    top_10_df = _get('top_10_df')
    top_10_data = top_10_df.to_dict('records')
    return jsonify(top_10_data)


@bp.route('/api/all_transactions')
def get_all_transactions():
    results_df = _get('results_df')
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


@bp.route('/api/alert_distribution')
def get_alert_distribution():
    results_df = _get('results_df')
    distribution = results_df['Alert_Level'].value_counts().to_dict()
    ordered = {
        'CRITICAL': distribution.get('CRITICAL', 0),
        'HIGH': distribution.get('HIGH', 0),
        'MEDIUM': distribution.get('MEDIUM', 0),
        'LOW': distribution.get('LOW', 0),
        'NORMAL': distribution.get('NORMAL', 0)
    }
    return jsonify(ordered)


@bp.route('/api/risk_distribution')
def get_risk_distribution():
    results_df = _get('results_df')
    bins = [0, 25, 50, 75, 90, 100]
    labels = ['0-25', '25-50', '50-75', '75-90', '90-100']
    results_df['Risk_Bin'] = pd.cut(results_df['Risk_Score'], bins=bins, labels=labels, include_lowest=True)
    distribution = results_df['Risk_Bin'].value_counts().sort_index().to_dict()
    return jsonify(distribution)


@bp.route('/api/search/<int:tx_id>')
def search_transaction(tx_id):
    results_df = _get('results_df')
    tx = results_df[results_df['Transaction_ID'] == tx_id]
    if len(tx) == 0:
        return jsonify({'error': 'Transaction not found'}), 404
    return jsonify(tx.iloc[0].to_dict())


@bp.route('/api/watchlist', methods=['GET', 'POST', 'DELETE'])
def manage_watchlist():
    # simple in-memory watchlist for demo
    watchlist = current_app.config.setdefault('watchlist', [])
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


@bp.route('/api/predict', methods=['POST'])
def predict_new_transaction():
    model = _get('model')
    PROCESSED_DIR = _get('PROCESSED_DIR')
    X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy')) if os.path.exists(os.path.join(PROCESSED_DIR, 'X_test.npy')) else np.zeros((1,1))
    random_idx = np.random.randint(0, len(X_test)) if len(X_test) > 0 else 0
    features = X_test[random_idx]
    prediction = model.predict([features])[0]
    probability = model.predict_proba([features])[0][1] if hasattr(model, 'predict_proba') else 0.0
    risk_score = int(probability * 100)
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


@bp.route('/api/predict_custom', methods=['POST'])
def predict_custom_transaction():
    try:
        data = request.json
        transaction_amount = float(data.get('amount', 1.5))
        num_inputs = int(data.get('num_inputs', 2))
        num_outputs = int(data.get('num_outputs', 2))
        transaction_fee = float(data.get('fee', 0.0001))
        PROCESSED_DIR = _get('PROCESSED_DIR')
        X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy')) if os.path.exists(os.path.join(PROCESSED_DIR, 'X_test.npy')) else np.zeros((1,1))
        y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy')) if os.path.exists(os.path.join(PROCESSED_DIR, 'y_test.npy')) else np.array([0])
        normal_indices = np.where(y_test == 0)[0]
        model = _get('model')
        if len(normal_indices) > 100:
            sample_indices = np.random.choice(normal_indices, 100, replace=False)
            sample_features = X_test[sample_indices]
            sample_predictions = model.predict_proba(sample_features)[:, 1] if hasattr(model, 'predict_proba') else np.zeros(len(sample_features))
            base_probability = np.median(sample_predictions)
        else:
            base_probability = 0.10
        risk_adjustment = 0.0
        if transaction_amount > 100:
            risk_adjustment += 0.30
        elif transaction_amount > 50:
            risk_adjustment += 0.20
        elif transaction_amount > 20:
            risk_adjustment += 0.10
        elif transaction_amount > 10:
            risk_adjustment += 0.05
        total_io = num_inputs + num_outputs
        if total_io > 50:
            risk_adjustment += 0.25
        elif total_io > 30:
            risk_adjustment += 0.15
        elif total_io > 15:
            risk_adjustment += 0.08
        elif total_io > 10:
            risk_adjustment += 0.03
        if transaction_amount > 10 and transaction_fee < 0.00001:
            risk_adjustment += 0.20
        elif transaction_amount > 5 and transaction_fee < 0.00005:
            risk_adjustment += 0.10
        if transaction_fee > 0.01:
            risk_adjustment += 0.15
        elif transaction_fee > 0.005:
            risk_adjustment += 0.08
        if num_inputs > 0:
            io_ratio = num_outputs / num_inputs
            if io_ratio > 10 or io_ratio < 0.1:
                risk_adjustment += 0.12
            elif io_ratio > 5 or io_ratio < 0.2:
                risk_adjustment += 0.06
        if num_inputs > 20:
            risk_adjustment += 0.15
        elif num_inputs > 10:
            risk_adjustment += 0.08
        if num_outputs > 20:
            risk_adjustment += 0.15
        elif num_outputs > 10:
            risk_adjustment += 0.08
        adjusted_probability = base_probability + risk_adjustment
        adjusted_probability = max(0.0, min(1.0, adjusted_probability))
        risk_score = int(adjusted_probability * 100)
        prediction = 1 if risk_score >= 50 else 0
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
        if risk_score >= _get('ALERT_THRESHOLD_HIGH'):
            alert_type = 'CRITICAL' if risk_score >= _get('ALERT_THRESHOLD_CRITICAL') else 'HIGH'
            trigger_alert = _get('trigger_alert_async')
            if trigger_alert:
                trigger_alert({
                    'transaction_id': result['transaction_id'],
                    'amount': transaction_amount,
                    'num_inputs': num_inputs,
                    'num_outputs': num_outputs,
                    'fee': transaction_fee,
                    'risk_score': risk_score,
                    'alert_level': alert_level,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'recommended_action': action
                }, alert_type)
        save_fn = _get('save_to_history')
        if save_fn:
            save_fn({
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
            })
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 400


# Export endpoints
@bp.route('/api/export/csv/<transaction_type>')
def export_csv(transaction_type):
    try:
        OUTPUTS_DIR = _get('OUTPUTS_DIR')
        results_df = _get('results_df')
        top_10_df = _get('top_10_df')
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


@bp.route('/api/export/pdf/summary')
def export_pdf_summary():
    try:
        results_df = _get('results_df')
        top_10_df = _get('top_10_df')
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


@bp.route('/api/export/pdf/transaction/<int:tx_id>')
def export_pdf_transaction(tx_id):
    try:
        results_df = _get('results_df')
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


@bp.route('/api/batch/upload', methods=['POST'])
def batch_upload():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        allowed_file = _get('allowed_file')
        if not allowed_file(file.filename):
            return jsonify({'error': 'Only CSV files are allowed'}), 400
        filename = secure_filename(file.filename)
        UPLOAD_FOLDER = _get('UPLOAD_FOLDER')
        filepath = os.path.join(UPLOAD_FOLDER, filename)
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
        PROCESSED_DIR = _get('PROCESSED_DIR')
        model = _get('model')
        for idx, row in df.iterrows():
            try:
                amount = float(row['amount'])
                num_inputs = int(row['num_inputs'])
                num_outputs = int(row['num_outputs'])
                fee = float(row['fee'])
                X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy')) if os.path.exists(os.path.join(PROCESSED_DIR, 'X_test.npy')) else np.zeros((1,1))
                y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy')) if os.path.exists(os.path.join(PROCESSED_DIR, 'y_test.npy')) else np.array([0])
                normal_indices = np.where(y_test == 0)[0]
                sample_indices = np.random.choice(normal_indices, min(100, len(normal_indices)), replace=False) if len(normal_indices)>0 else []
                sample_predictions = model.predict_proba(X_test[sample_indices])[:, 1] if hasattr(model, 'predict_proba') and len(sample_indices)>0 else np.array([0])
                base_probability = np.median(sample_predictions) if len(sample_predictions)>0 else 0
                risk_adjustment = 0.0
                if amount > 100:
                    risk_adjustment += 0.30
                elif amount > 50:
                    risk_adjustment += 0.20
                elif amount > 20:
                    risk_adjustment += 0.10
                elif amount > 10:
                    risk_adjustment += 0.05
                total_io = num_inputs + num_outputs
                if total_io > 50:
                    risk_adjustment += 0.25
                elif total_io > 30:
                    risk_adjustment += 0.15
                elif total_io > 15:
                    risk_adjustment += 0.08
                elif total_io > 10:
                    risk_adjustment += 0.03
                if amount > 10 and fee < 0.00001:
                    risk_adjustment += 0.20
                elif amount > 5 and fee < 0.00005:
                    risk_adjustment += 0.10
                if fee > 0.01:
                    risk_adjustment += 0.15
                elif fee > 0.005:
                    risk_adjustment += 0.08
                if num_inputs > 0:
                    io_ratio = num_outputs / num_inputs
                    if io_ratio > 10 or io_ratio < 0.1:
                        risk_adjustment += 0.12
                    elif io_ratio > 5 or io_ratio < 0.2:
                        risk_adjustment += 0.06
                if num_inputs > 20:
                    risk_adjustment += 0.15
                elif num_inputs > 10:
                    risk_adjustment += 0.08
                if num_outputs > 20:
                    risk_adjustment += 0.15
                elif num_outputs > 10:
                    risk_adjustment += 0.08
                adjusted_probability = max(0.0, min(1.0, base_probability + risk_adjustment))
                risk_score = int(adjusted_probability * 100)
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
        results_filename = f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        OUTPUTS_DIR = _get('OUTPUTS_DIR')
        results_filepath = os.path.join(OUTPUTS_DIR, results_filename)
        results_df_batch = pd.DataFrame(valid_results)
        results_df_batch.to_csv(results_filepath, index=False)
        os.remove(filepath)
        return jsonify({
            'success': True,
            'summary': summary,
            'results': results,
            'download_url': f'/api/download/{results_filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/download/<filename>')
def download_file(filename):
    try:
        OUTPUTS_DIR = _get('OUTPUTS_DIR')
        filepath = os.path.join(OUTPUTS_DIR, filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@bp.route('/api/batch/template')
def download_template():
    try:
        UPLOAD_FOLDER = _get('UPLOAD_FOLDER')
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


@bp.route('/api/history')
def get_history():
    try:
        history = _get('load_history')()
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


@bp.route('/api/history/stats')
def get_history_stats():
    try:
        history = _get('load_history')()
        if not history:
            return jsonify({
                'total_analyzed': 0,
                'avg_risk': 0,
                'alert_distribution': {},
                'trend': []
            })
        risk_scores = [h['risk_score'] for h in history]
        alert_levels = [h['alert_level'] for h in history]
        from collections import Counter
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


@bp.route('/api/history/clear', methods=['POST'])
def clear_history():
    try:
        HISTORY_FILE = _get('HISTORY_FILE')
        with open(HISTORY_FILE, 'w') as f:
            import json
            json.dump([], f)
        return jsonify({'success': True, 'message': 'History cleared'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/history/export')
def export_history():
    try:
        history = _get('load_history')()
        if not history:
            return jsonify({'error': 'No history to export'}), 400
        df = pd.DataFrame(history)
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        response = make_response(output.getvalue())
        response.headers['Content-Type'] = 'text/csv'
        response.headers['Content-Disposition'] = f"attachment; filename=transaction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        return response
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@bp.route('/api/alerts/test', methods=['POST'])
def test_alert():
    try:
        send_email = _get('send_email_alert')
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
        success = send_email(test_data, 'CRITICAL') if send_email else False
        if success:
            return jsonify({'success': True, 'message': 'Test alert sent successfully'})
        else:
            return jsonify({'success': False, 'message': 'Email is disabled or failed to send'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Watchlist DB-backed endpoints
@bp.route('/api/watchlist', methods=['GET'])
def get_watchlist():
    try:
        db = _get('db')
        watchlist = db.get_all_watchlist()
        return jsonify({'success': True, 'data': watchlist, 'count': len(watchlist)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/watchlist/add', methods=['POST'])
def add_to_watchlist():
    try:
        data = request.json
        required_fields = ['type', 'value', 'risk_level', 'reason']
        for field in required_fields:
            if field not in data:
                return jsonify({'success': False, 'error': f'Missing required field: {field}'}), 400
        db = _get('db')
        result = db.add_to_watchlist(
            type=data['type'],
            value=data['value'],
            risk_level=data['risk_level'],
            reason=data['reason'],
            tags=data.get('tags', []),
            notes=data.get('notes', '')
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/watchlist/<int:watchlist_id>', methods=['DELETE'])
def remove_from_watchlist(watchlist_id):
    try:
        db = _get('db')
        result = db.remove_from_watchlist(watchlist_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/watchlist/check/<value>', methods=['GET'])
def check_watchlist(value):
    try:
        db = _get('db')
        is_watched = db.check_watchlist(value)
        return jsonify({'success': True, 'is_watchlisted': is_watched, 'value': value})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/watchlist/stats', methods=['GET'])
def get_watchlist_stats():
    try:
        db = _get('db')
        stats = db.get_watchlist_stats()
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/watchlist/export', methods=['GET'])
def export_watchlist():
    try:
        db = _get('db')
        watchlist = db.get_all_watchlist()
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(['ID', 'Type', 'Value', 'Risk Level', 'Reason', 'Tags', 'Activity Count', 'Added Date', 'Last Seen', 'Total Volume', 'Notes'])
        for item in watchlist:
            writer.writerow([
                item['id'], item['type'], item['value'], item['risk_level'], item['reason'], ', '.join(item['tags']), item['activity_count'], item['added_date'], item['last_seen'] or 'N/A', item['total_volume'], item['notes'] or ''
            ])
        output.seek(0)
        return Response(output.getvalue(), mimetype='text/csv', headers={'Content-Disposition': 'attachment; filename=watchlist_export.csv'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@bp.route('/api/predict_custom', methods=['POST'])
def predict_custom():
    try:
        data = request.json
        amount = float(data.get('amount', 0))
        num_inputs = int(data.get('num_inputs', 1))
        num_outputs = int(data.get('num_outputs', 1))
        fee = float(data.get('fee', 0))
        sender_address = data.get('sender_address', '')
        receiver_address = data.get('receiver_address', '')
        db = _get('db')
        sender_watchlisted = db.check_watchlist(sender_address) if sender_address else False
        receiver_watchlisted = db.check_watchlist(receiver_address) if receiver_address else False
        base_risk = 15.0
        risk_score = base_risk
        risk_factors = []
        if sender_watchlisted or receiver_watchlisted:
            risk_score += 40
            if sender_watchlisted:
                risk_factors.append("⚠️ Sender address is WATCHLISTED")
                db.update_activity(sender_address, amount)
            if receiver_watchlisted:
                risk_factors.append("⚠️ Receiver address is WATCHLISTED")
                db.update_activity(receiver_address, amount)
        if amount > 100:
            risk_increase = min(30, (amount / 100) * 5)
            risk_score += risk_increase
            risk_factors.append(f"Large transaction amount: {amount} BTC (+{risk_increase:.1f}%)")
        elif amount > 50:
            risk_score += 15
            risk_factors.append(f"Medium-high amount: {amount} BTC (+15%)")
        if num_inputs > 50 or num_outputs > 50:
            risk_score += 25
            risk_factors.append(f"High I/O complexity: {num_inputs} inputs, {num_outputs} outputs (+25%)")
        elif num_inputs > 10 or num_outputs > 10:
            risk_score += 12
            risk_factors.append(f"Moderate I/O complexity (+12%)")
        expected_fee = amount * 0.0001
        if fee < expected_fee * 0.5 or fee > expected_fee * 3:
            risk_score += 20
            risk_factors.append(f"Unusual fee: {fee} BTC (expected ~{expected_fee:.6f}) (+20%)")
        if num_inputs > 0 and num_outputs > 0:
            io_ratio = num_inputs / num_outputs
            if io_ratio > 10 or io_ratio < 0.1:
                risk_score += 12
                risk_factors.append(f"Suspicious I/O ratio: {io_ratio:.2f} (+12%)")
        if num_inputs > 5 and num_outputs > 5:
            risk_score += 30
            risk_factors.append("Possible mixing/tumbling pattern detected (+30%)")
        risk_score = min(100, risk_score)
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
        import hashlib, time
        tx_id = hashlib.sha256(f"{amount}{num_inputs}{num_outputs}{fee}{time.time()}".encode()).hexdigest()[:16]
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
        save_fn = _get('save_to_history')
        if save_fn:
            save_fn(transaction_data)
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
