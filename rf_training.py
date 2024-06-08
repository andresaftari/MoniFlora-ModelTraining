import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Init Flask
app = Flask('moniflora')

def initialize_firebase():
    cred = credentials.Certificate('/Users/andresaftari/Development/Kuliah/MoniFlora-Training/moniflora-7d3a3-firebase-adminsdk-xtibx-b38ec6e08d.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://moniflora-7d3a3-default-rtdb.firebaseio.com/'
    })

def get_dataset():
    ref = db.reference('sensor')
    return ref.get()

def determine_label(value):
    temp = value['temperature']
    light = value['light']
    ec = value['conductivity']
    moisture = value['moisture']
    
def determine_label(value):
    temp = value['temperature']
    light = value['light']
    ec = value['conductivity']
    moisture = value['moisture']
    
    if (22 <= temp <= 27 and 3500 <= light <= 5000 and 1500 <= ec <= 2000 and 35 <= moisture <= 50):
        return 0  # Optimal
    elif ((20 <= temp < 22 or 27 <= temp <= 30) or 
        (1500 <= light < 3500 or 5000 < light <= 6500) or 
        (950 <= ec < 1500 or 2000 < ec <= 3000) or 
        (30 <= moisture < 35 or 50 < moisture <= 60)):
        return 1  # Caution
    else:
        return 2  # Extreme

def prepare_data(dataset):
    data = {
        'temperature': [],
        'light': [],
        'conductivity': [],
        'moisture': [],
        'label': []
    }

    for _, value in dataset.items():
        data['temperature'].append(value['temperature'])
        data['light'].append(value['light'])
        data['conductivity'].append(value['conductivity'])
        data['moisture'].append(value['moisture'])
        data['label'].append(determine_label(value))
    
    return pd.DataFrame(data)

def train_random_forest(X_train, y_train):
    forest = RandomForestClassifier(n_estimators=100, random_state=42)
    forest.fit(X_train, y_train)
    
    return forest

# Load data
initialize_firebase()
dataset = get_dataset()

# Prepare data
data = prepare_data(dataset)
    
X = np.array([data['temperature'], data['light'], data['conductivity'], data['moisture']]).T
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Initialize and fit scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = train_random_forest(X_train_scaled, y_train_balanced)
    
# Use 3-fold cross-validation with stratified splits
stratified_kfold = StratifiedKFold(n_splits=3)
validator = cross_val_score(rf_model, X_test_scaled, y_test, cv=stratified_kfold)
score = rf_model.score(X_test_scaled, y_test)
    
print(f'Random Forest Score: {score}')
print(f'Random Forest Cross Validation Score: {validator}')
    
y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

print('\n=========================== DEBUG ===========================')

example_value = {
    'temperature': 27.2,
    'light': 1912,
    'conductivity': 1201,
    'moisture': 43
}

# Scale example_value and make a prediction
example_features = np.array([[example_value['temperature'], example_value['light'], example_value['conductivity'], example_value['moisture']]])
example_scaled = scaler.transform(example_features)
example_prediction = rf_model.predict(example_scaled)
example_prediction_proba = rf_model.predict_proba(example_scaled)

print(f'Scaled example features: {example_scaled}')
print(f'Example prediction: {example_prediction}')
print(f'Example prediction probabilities: {example_prediction_proba}')

print('=========================== DEBUG ===========================\n')

# Save the model and scaler
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))    
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Load ML Model
model = pickle.load(open('rf_model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/predict', methods=['POST'])
def serve_model():
    data = request.json
    features = np.array([
        [
            data['temperature'], 
            data['light'], 
            data['conductivity'], 
            data['moisture']
        ]
    ])
    
    print(f"Received features: {features}")
    
    # Apply the same scaling as during training
    features_scaled = scaler.transform(features)
    print(f'Scaled features: {features_scaled}')
    
    label_names = {
        0: 'Optimal',
        1: 'Caution',
        2: 'Extreme'
    }
    
    pred = model.predict(features_scaled)
    pred_proba = model.predict_proba(features_scaled)
    
    pred_label_index = int(pred[0])
    pred_label_name = label_names[pred_label_index]
    
    print('\n=========================== DEBUG ===========================')
    print(f'Prediction: {pred_label_name} - Prediction index: {pred}')
    print(f'Prediction probability: {pred_proba}')
    print('=========================== DEBUG ===========================\n')
    
    return jsonify({
        'prediction': pred_label_name,
        'prediction_index': pred_label_index,
        'probability': pred_proba[0].tolist()
    })


if __name__ == "__main__":
    app.run(debug=True)