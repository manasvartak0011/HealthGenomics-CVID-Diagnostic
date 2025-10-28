# app.py - Cleaned and Debugged Version
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

# Global variables
binary_model = None
multiclass_model = None
scaler = None
feature_names = []
model_metrics = {}


def create_mock_data():
    """Create mock data if real dataset is not available"""
    np.random.seed(42)
    n_samples = 200

    global feature_names
    feature_names = [
        'STAT1.1', 'AIRE.2', 'TAPI', 'STAT1.2', 'IL7R',
        'NCFI', 'PSMB10', 'TNFSF18B', 'IFNGR2', 'CTBA',
        'TNFSF18B.1', 'STAT2', 'IL7R.1', 'WAS', 'STAT1'
    ]

    data = {}
    for feature in feature_names:
        if 'STAT' in feature:
            data[feature] = np.random.normal(4500, 1500, n_samples)
        elif 'IL7R' in feature:
            data[feature] = np.random.normal(6000, 2000, n_samples)
        else:
            data[feature] = np.random.normal(5000, 2000, n_samples)

    df = pd.DataFrame(data)
    df['Sample'] = [f'GSM{1000000 + i}' for i in range(n_samples)]

    # Binary labels
    df['Binary_Label'] = ['Healthy' if i % 3 == 0 else 'CVID' for i in range(n_samples)]

    # Subtypes
    def assign_subtype(row, idx):
        if row['Binary_Label'] == 'Healthy':
            return 'Healthy'
        if idx % 3 == 0:
            return 'Cluster 0: Severe Immunodeficiency'
        elif idx % 3 == 1:
            return 'Cluster 1: Moderate with Autoimmunity'
        else:
            return 'Cluster 2: Mild Progressive'

    df['Multiclass_Label'] = [assign_subtype(row, i) for i, row in df.iterrows()]

    return df


def load_data():
    """Load dataset or use mock data"""
    try:
        df = pd.read_csv('PID_gene_expressions_augmented.csv')
        print(f"Real dataset loaded: {len(df)} samples")

        global feature_names
        feature_names = [col for col in df.columns if col != 'Sample']

        # Binary label creation
        df['Binary_Label'] = df['Sample'].apply(
            lambda x: 'Healthy' if 'GSM' in str(x) and int(str(x).split('GSM')[1][:7]) % 3 == 0 else 'CVID'
        )

        # Subtypes
        def assign_subtype(row):
            if row['Binary_Label'] == 'Healthy':
                return 'Healthy'
            stat1_avg = (row['STAT1.1'] + row['STAT1.2'] + row['STAT1']) / 3
            il7r_avg = (row['IL7R'] + row['IL7R.1']) / 2

            if stat1_avg > 4000 and il7r_avg > 4000:
                return 'Cluster 0: Severe Immunodeficiency'
            elif stat1_avg > 2500:
                return 'Cluster 1: Moderate with Autoimmunity'
            else:
                return 'Cluster 2: Mild Progressive'

        df['Multiclass_Label'] = df.apply(assign_subtype, axis=1)

    except FileNotFoundError:
        print("Dataset not found. Using mock data.")
        df = create_mock_data()

    print("Label distribution:")
    print(df['Binary_Label'].value_counts())
    print("CVID Subtypes:")
    print(df[df['Multiclass_Label'] != 'Healthy']['Multiclass_Label'].value_counts())

    return df


def train_models():
    """Train ML models"""
    global binary_model, multiclass_model, scaler, feature_names, model_metrics

    print("Training models...")
    df = load_data()

    X = df[feature_names].values
    y_binary = df['Binary_Label'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_binary, test_size=0.25, random_state=42, stratify=y_binary
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm_model = SVC(probability=True, random_state=42)

    binary_model = VotingClassifier(
        estimators=[('rf', rf_model), ('gb', gb_model), ('svm', svm_model)],
        voting='soft'
    )

    binary_model.fit(X_train_scaled, y_train)
    y_pred = binary_model.predict(X_test_scaled)
    binary_acc = accuracy_score(y_test, y_pred)
    model_metrics['binary_accuracy'] = float(binary_acc)

    # Train multiclass model
    cvid_mask = df['Multiclass_Label'] != 'Healthy'
    X_multi = df[cvid_mask][feature_names].values
    y_multi = df[cvid_mask]['Multiclass_Label'].values

    if len(X_multi) > 0:
        X_multi_scaled = scaler.transform(X_multi)
        X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(
            X_multi_scaled, y_multi, test_size=0.25, random_state=42, stratify=y_multi
        )

        multiclass_model = RandomForestClassifier(n_estimators=100, random_state=42)
        multiclass_model.fit(X_multi_train, y_multi_train)

        y_multi_pred = multiclass_model.predict(X_multi_test)
        multi_acc = accuracy_score(y_multi_test, y_multi_pred)
        model_metrics['multiclass_accuracy'] = float(multi_acc)

    print(f"Binary accuracy: {binary_acc:.4f}")
    if 'multiclass_accuracy' in model_metrics:
        print(f"Multiclass accuracy: {model_metrics['multiclass_accuracy']:.4f}")

    return model_metrics


def load_models():
    """Load saved models or train new ones"""
    global binary_model, multiclass_model, scaler, feature_names, model_metrics

    try:
        binary_model = joblib.load('binary_model.pkl')
        multiclass_model = joblib.load('multiclass_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_names = joblib.load('feature_names.pkl')
        model_metrics = joblib.load('model_metrics.pkl')
        print("Pre-trained models loaded.")
    except:
        print("Training new models...")
        train_models()
        joblib.dump(binary_model, 'binary_model.pkl')
        joblib.dump(multiclass_model, 'multiclass_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        joblib.dump(feature_names, 'feature_names.pkl')
        joblib.dump(model_metrics, 'model_metrics.pkl')


@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        gene_values = np.array([data[gene] for gene in feature_names]).reshape(1, -1)
        scaled_input = scaler.transform(gene_values)

        binary_pred = binary_model.predict(scaled_input)[0]
        binary_proba = binary_model.predict_proba(scaled_input)[0]
        confidence = float(max(binary_proba))

        individual_predictions = []
        for name, model in binary_model.named_estimators_.items():
            pred = model.predict(scaled_input)[0]
            proba = model.predict_proba(scaled_input)[0]
            individual_predictions.append({
                'model': name.upper(),
                'prediction': pred,
                'confidence': float(max(proba))
            })

        result = {
            'binary_prediction': binary_pred,
            'binary_confidence': confidence,
            'model_agreement': sum(1 for p in individual_predictions if p['prediction'] == binary_pred) / len(individual_predictions),
            'individual_models': individual_predictions
        }

        if binary_pred == 'CVID' and multiclass_model:
            multi_pred = multiclass_model.predict(scaled_input)[0]
            multi_proba = multiclass_model.predict_proba(scaled_input)[0]

            result['multiclass_prediction'] = multi_pred
            result['multiclass_confidence'] = float(max(multi_proba))

            if 'Severe' in multi_pred:
                result['risk_level'] = 'High'
            elif 'Moderate' in multi_pred:
                result['risk_level'] = 'Medium'
            else:
                result['risk_level'] = 'Low-Medium'

            result['recommendations'] = [
                'Consult with immunology specialist',
                'Consider immunoglobulin replacement therapy',
                'Monitor for respiratory infections',
                'Regular follow-up in 3 months'
            ]

            result['clinical_notes'] = [
                'Gene expression analysis completed',
                'Further clinical correlation recommended'
            ]

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/features', methods=['GET'])
def get_features():
    return jsonify({'features': feature_names})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(model_metrics)


@app.route('/sample-data', methods=['GET'])
def get_sample_data():
    """Return sample gene input"""
    sample_data = {gene: float(np.random.normal(5000, 2000)) for gene in feature_names}
    sample_data['sample_id'] = 'GSM_TEST_001'
    return jsonify(sample_data)


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': binary_model is not None,
        'features_count': len(feature_names)
    })


if __name__ == '__main__':
    print("CVID Diagnostic System Starting...")
    load_models()
    print(f"System ready with {len(feature_names)} gene features")
    print("API running on http://localhost:5000")
    app.run(debug=True, port=5000, host='0.0.0.0')
