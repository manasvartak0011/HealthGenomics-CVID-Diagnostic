# app.py - Fixed Flask Backend with ML Models
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables for models and scaler
binary_model = None
multiclass_model = None
scaler = None
feature_names = None

# Load and preprocess data
def load_data():
    global feature_names
    # Use the exact filename from your system
    try:
        df = pd.read_csv('genes dataset.csv')
    except FileNotFoundError:
        # Try alternative names
        try:
            df = pd.read_csv('genes_dataset.csv')
        except FileNotFoundError:
            raise FileNotFoundError("Could not find the dataset file. Please make sure 'genes dataset.csv' is in the same directory.")
    
    feature_names = df.columns[1:].tolist()  # All gene names
    
    # For demo purposes, we'll create synthetic labels
    # In a real scenario, you would have actual labels
    np.random.seed(42)
    n_samples = len(df)
    
    # Create binary labels (Healthy vs CVID) - 70% CVID, 30% Healthy
    binary_labels = np.random.choice(['Healthy', 'CVID'], size=n_samples, p=[0.3, 0.7])
    
    # Create multiclass labels for CVID samples only
    multiclass_labels = ['Unknown'] * n_samples
    for i in range(n_samples):
        if binary_labels[i] == 'CVID':
            multiclass_labels[i] = np.random.choice(['Cluster 0', 'Cluster 1', 'Cluster 2'], p=[0.4, 0.35, 0.25])
        else:
            multiclass_labels[i] = 'Healthy'
    
    df['Binary_Label'] = binary_labels
    df['Multiclass_Label'] = multiclass_labels
    
    print(f"Loaded dataset with {n_samples} samples and {len(feature_names)} genes")
    return df

# Train models
def train_models():
    global binary_model, multiclass_model, scaler, feature_names
    
    print("Training models...")
    df = load_data()
    
    # Prepare features and labels
    X = df[feature_names].values
    y_binary = df['Binary_Label'].values
    
    # Get CVID-only samples for multiclass
    cvid_mask = df['Multiclass_Label'] != 'Healthy'
    y_multiclass = df[cvid_mask]['Multiclass_Label'].values
    X_multiclass = df[cvid_mask][feature_names].values
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_multiclass_scaled = scaler.transform(X_multiclass)
    
    # Split data
    X_train, X_test, y_binary_train, y_binary_test = train_test_split(
        X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )
    
    if len(X_multiclass_scaled) > 0:
        X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
            X_multiclass_scaled, y_multiclass, test_size=0.2, random_state=42, stratify=y_multiclass
        )
    
    # Train binary model (Random Forest)
    binary_model = RandomForestClassifier(n_estimators=100, random_state=42)
    binary_model.fit(X_train, y_binary_train)
    
    # Train multiclass model (SVM) only if we have CVID samples
    if len(X_multiclass_scaled) > 0:
        multiclass_model = SVC(kernel='rbf', probability=True, random_state=42)
        multiclass_model.fit(X_multi_train, y_multi_train)
    
    # Evaluate models
    binary_acc = accuracy_score(y_binary_test, binary_model.predict(X_test))
    print(f"Binary Model Accuracy: {binary_acc:.4f}")
    
    if len(X_multiclass_scaled) > 0:
        multi_acc = accuracy_score(y_multi_test, multiclass_model.predict(X_multi_test))
        print(f"Multiclass Model Accuracy: {multi_acc:.4f}")
    
    # Save models and scaler
    joblib.dump(binary_model, 'binary_model.pkl')
    if multiclass_model:
        joblib.dump(multiclass_model, 'multiclass_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("Models trained and saved successfully!")

# Load pre-trained models
def load_models():
    global binary_model, multiclass_model, scaler, feature_names
    try:
        # Check if model files exist
        if (os.path.exists('binary_model.pkl') and 
            os.path.exists('multiclass_model.pkl') and 
            os.path.exists('scaler.pkl') and 
            os.path.exists('feature_names.pkl')):
            
            binary_model = joblib.load('binary_model.pkl')
            multiclass_model = joblib.load('multiclass_model.pkl')
            scaler = joblib.load('scaler.pkl')
            feature_names = joblib.load('feature_names.pkl')
            print("Models loaded successfully")
        else:
            print("Model files not found. Training new models...")
            train_models()
    except Exception as e:
        print(f"Error loading models: {e}. Training new models...")
        train_models()

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Validate input
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Check if all required features are present
        missing_features = [feature for feature in feature_names if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {missing_features}'}), 400
        
        gene_values = [data[gene] for gene in feature_names]
        gene_array = np.array(gene_values).reshape(1, -1)
        
        # Scale input
        scaled_input = scaler.transform(gene_array)
        
        # Binary prediction
        binary_pred = binary_model.predict(scaled_input)[0]
        binary_prob = binary_model.predict_proba(scaled_input)[0]
        
        result = {
            'binary_prediction': binary_pred,
            'binary_confidence': float(max(binary_prob)),
            'multiclass_prediction': 'N/A',
            'multiclass_confidence': 0.0
        }
        
        # If CVID, predict subtype
        if binary_pred == 'CVID' and multiclass_model:
            multi_pred = multiclass_model.predict(scaled_input)[0]
            multi_prob = multiclass_model.predict_proba(scaled_input)[0]
            result['multiclass_prediction'] = multi_pred
            result['multiclass_confidence'] = float(max(multi_prob))
            
            # Add recommendations based on subtype
            recommendations = get_recommendations(multi_pred)
            result['recommendations'] = recommendations
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_recommendations(subtype):
    recommendations = {
        'Cluster 0': [
            'Regular immunoglobulin replacement therapy',
            'Monitor for respiratory infections',
            'Consider antibiotic prophylaxis'
        ],
        'Cluster 1': [
            'Aggressive infection management',
            'Monitor autoimmune manifestations',
            'Consider immunomodulatory therapy'
        ],
        'Cluster 2': [
            'Comprehensive immune workup',
            'Monitor for gastrointestinal complications',
            'Consider targeted biologic therapy'
        ]
    }
    return recommendations.get(subtype, ['Consult with immunology specialist'])

# Get feature names
@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features': feature_names})

# Health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy', 
        'models_loaded': binary_model is not None,
        'features_count': len(feature_names) if feature_names else 0
    })

# Root endpoint
@app.route('/')
def home():
    return jsonify({
        'message': 'CVID Diagnostic API is running',
        'endpoints': {
            '/health': 'Check API status',
            '/features': 'Get list of genes',
            '/predict': 'POST gene expression data for diagnosis'
        }
    })

if __name__ == '__main__':
    print("Starting CVID Diagnostic API...")
    load_models()
    print(f"Available genes: {len(feature_names)}")
    print("API is ready on http://localhost:5000")
    app.run(debug=True, port=5000)