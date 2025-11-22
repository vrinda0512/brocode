"""
Ensemble Model Predictor
Loads and uses all 3 models for prediction
"""

import joblib
import numpy as np
from tensorflow import keras

class EnsemblePredictor:
    """Ensemble model for fraud detection"""
    
    def __init__(self):
        print("🔄 Loading ensemble models...")
        
        # Load models
        self.rf_model = joblib.load('models/rf_model.pkl')
        self.xgb_model = joblib.load('models/xgb_model.pkl')
        self.nn_model = keras.models.load_model('models/nn_model.h5')
        self.scaler = joblib.load('models/ensemble_scaler.pkl')
        self.ensemble_info = joblib.load('models/ensemble_info.pkl')
        
        print("✅ Ensemble models loaded successfully!")
        print(f"   • Random Forest Accuracy: {self.ensemble_info['rf_accuracy']*100:.2f}%")
        print(f"   • XGBoost Accuracy: {self.ensemble_info['xgb_accuracy']*100:.2f}%")
        print(f"   • Neural Network Accuracy: {self.ensemble_info['nn_accuracy']*100:.2f}%")
        print(f"   • Ensemble Accuracy: {self.ensemble_info['ensemble_accuracy']*100:.2f}%")
    
    def predict(self, amount, num_inputs, num_outputs, fee):
        """
        Predict if transaction is fraudulent using ensemble
        
        Args:
            amount: Transaction amount in BTC
            num_inputs: Number of input addresses
            num_outputs: Number of output addresses
            fee: Transaction fee in BTC
        
        Returns:
            dict with prediction, confidence, and model votes
        """
        # Prepare features
        features = np.array([[amount, num_inputs, num_outputs, fee]])
        features_scaled = self.scaler.transform(features)
        
        # Get predictions from each model
        rf_pred = self.rf_model.predict(features_scaled)[0]
        rf_proba = self.rf_model.predict_proba(features_scaled)[0][1]
        
        xgb_pred = self.xgb_model.predict(features_scaled)[0]
        xgb_proba = self.xgb_model.predict_proba(features_scaled)[0][1]
        
        nn_proba = self.nn_model.predict(features_scaled, verbose=0)[0][0]
        nn_pred = 1 if nn_proba > 0.5 else 0
        
        # Ensemble voting (majority)
        total_votes = rf_pred + xgb_pred + nn_pred
        ensemble_pred = 1 if total_votes >= 2 else 0
        
        # Average confidence
        avg_confidence = (rf_proba + xgb_proba + nn_proba) / 3
        
        # Determine prediction label
        if ensemble_pred == 1:
            prediction = "Fraudulent"
        else:
            prediction = "Normal"
        
        return {
            'prediction': prediction,
            'is_fraud': bool(ensemble_pred),
            'confidence': float(avg_confidence * 100),
            'ensemble_votes': {
                'random_forest': {'vote': 'Fraud' if rf_pred == 1 else 'Normal', 'confidence': float(rf_proba * 100)},
                'xgboost': {'vote': 'Fraud' if xgb_pred == 1 else 'Normal', 'confidence': float(xgb_proba * 100)},
                'neural_network': {'vote': 'Fraud' if nn_pred == 1 else 'Normal', 'confidence': float(nn_proba * 100)}
            },
            'votes_for_fraud': int(total_votes),
            'votes_for_normal': int(3 - total_votes)
        }
    
    def get_model_info(self):
        """Get information about the ensemble"""
        return self.ensemble_info


# Test the ensemble
if __name__ == '__main__':
    print("=" * 80)
    print("🧪 TESTING ENSEMBLE PREDICTOR")
    print("=" * 80)
    
    predictor = EnsemblePredictor()
    
    # Test cases
    test_cases = [
        ("Normal small transaction", 0.5, 1, 2, 0.0001),
        ("Normal medium transaction", 10.0, 3, 4, 0.001),
        ("Suspicious large amount", 150.0, 5, 8, 0.005),
        ("Fraudulent mixing pattern", 80.0, 30, 35, 0.0001),
        ("Critical fraud", 500.0, 50, 60, 0.00001)
    ]
    
    print("\n" + "=" * 80)
    for name, amount, inputs, outputs, fee in test_cases:
        print(f"\n📊 Test: {name}")
        print(f"   Amount: {amount} BTC, I/O: {inputs}/{outputs}, Fee: {fee} BTC")
        
        result = predictor.predict(amount, inputs, outputs, fee)
        
        print(f"\n   🎯 Ensemble Prediction: {result['prediction']}")
        print(f"   📈 Confidence: {result['confidence']:.2f}%")
        print(f"   🗳️  Votes: {result['votes_for_fraud']}/3 for Fraud, {result['votes_for_normal']}/3 for Normal")
        print(f"\n   Individual Model Votes:")
        for model, vote_info in result['ensemble_votes'].items():
            print(f"      • {model.replace('_', ' ').title()}: {vote_info['vote']} ({vote_info['confidence']:.2f}%)")
        print("   " + "-" * 76)
    
    print("\n" + "=" * 80)
    print("✅ Ensemble predictor test complete!")
    print("=" * 80)