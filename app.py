import firebase_admin
from firebase_admin import credentials, db
from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter

app = Flask('moniflora')

def initialize_firebase():
    cred = credentials.Certificate('moniflora-backup-firebase-adminsdk-uh66r-b9af29113e.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://moniflora-backup-default-rtdb.firebaseio.com/'
    })

def get_dataset():
    ref = db.reference('sensor')
    query = ref.order_by_child('light').start_at(600).limit_to_last(2000)
    results = query.get()

    if results:
        print(f'Total dataset: {len(ref.get())}')
        print(f'Selected dataset with light >= 600: {len(results)}')
        print(f'Dataset limited to the last 2000 entries: {len(results)}')
    else:
        print('No records found with light >= 600')

    return results

def determine_label(value):
    temp = value['temperature']
    light = value['light']
    ec = value['conductivity']
    moisture = value['moisture']

    if temp < 10 or ec < 10 or moisture < 30 or light < 100:
        return 2  # Extreme

    if 22 <= temp <= 27 and 3500 <= light <= 5600 and 1500 <= ec <= 2500 and 35 <= moisture <= 50:
        return 0  # Optimal

    if ((20 <= temp < 22 or 27 <= temp <= 30) or
        (1500 <= light < 3500 or 5600 < light <= 6000) or
        (950 <= ec < 1500 or 2500 < ec <= 3000) or
        (30 <= moisture < 35 or 50 < moisture <= 60)):
        return 1  # Caution

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

# Load data
initialize_firebase()
dataset = get_dataset()
data = prepare_data(dataset)

X = np.array([data['temperature'], data['light'], data['conductivity'], data['moisture']]).T
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=13, stratify=y)

# Balance the dataset
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

def train_random_forest(X_train, y_train):
    param_grid = {
        'min_samples_split': [2, 3, 4],
        'min_samples_leaf': [1, 2, 4],
    }

    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    forest = RandomForestClassifier(n_estimators=50, max_depth=5, max_features='log2', **grid_search.best_params_, warm_start=True, oob_score=True)
    forest.fit(X_train, y_train)
    print(f'Best parameters: {best_params}')

    return forest

rf_model = train_random_forest(X_train_scaled, y_train_balanced)

stratified_kfold = StratifiedKFold(n_splits=5)
validator = cross_val_score(rf_model, X_test_scaled, y_test, cv=stratified_kfold)
print(f'Random Forest Cross Validation Score: {validator}')

pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))


@app.route('/predict', methods=['POST'])
def serve_model():
    # Load ML Model
    model = pickle.load(open('rf_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    data = request.json
    features = np.array([
        [
            data['temperature'],
            data['light'],
            data['conductivity'],
            data['moisture']
        ]
    ])

    # Apply the same scaling as during training
    features_scaled = scaler.transform(features)

    label_names = {
        0: 'Optimal',
        1: 'Caution',
        2: 'Extreme'
    }

    pred = model.predict(features_scaled)
    pred_proba = model.predict_proba(features_scaled)

    pred_label_index = int(pred[0])
    pred_label_name = label_names[pred_label_index]

    return jsonify({
        'prediction': pred_label_name,
        'prediction_index': pred_label_index,
        'probability': pred_proba[0].tolist()
    })
