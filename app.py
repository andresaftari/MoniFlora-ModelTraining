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
from collections import Counter

app = Flask('moniflora')

def initialize_firebase():
    cred = credentials.Certificate('/Users/andresaftari/Development/Kuliah/skripsyit-training/moniflora-7d3a3-firebase-adminsdk-xtibx-b38ec6e08d.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://moniflora-7d3a3-default-rtdb.firebaseio.com/'
    })


def get_dataset():
    ref = db.reference('sensor')
    
    query = ref.order_by_child('light').start_at(600)
    results = query.get()

    if results:
        print(f'Total dataset: {len(ref.get())}')
        print(f'Selected dataset with light >= 600: {len(results)}')
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

    if (22 <= temp <= 27 and 3500 <= light <= 5600 and 1500 <= ec <= 2500 and 35 <= moisture <= 50):
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


def train_random_forest(X_train, y_train):
    # Define parameter grid for GridSearchCV
    # param_grid = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [None, 10, 20, 30],
    #     'min_samples_split': [2, 5, 10],
    #     'min_samples_leaf': [1, 2, 4],
    #     'max_features': ['auto', 'sqrt', 'log2']
    # }
    
    # Initialized
    # forest = RandomForestClassifier()
    
    # grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, n_jobs=-1)
    # grid_search.fit(X_train, y_train)

    # best_params = grid_search.best_params_
    forest = RandomForestClassifier(n_estimators=50, random_state=13, max_features='log2')
    forest.fit(X_train, y_train)
    # print(f'Best parameters: {best_params}')

    return forest


# Load data
initialize_firebase()
dataset = get_dataset()

# Prepare data
data = prepare_data(dataset)

# Count the occurrences of each label
label_counts = Counter(data['label'])
label_names = {0: 'Optimal', 1: 'Caution', 2: 'Extreme'}

# Print the counts for each label
for label, count in label_counts.items():
    print(f'{label_names[label]}: {count} datasets')

X = np.array([data['temperature'], data['light'], data['conductivity'], data['moisture']]).T
y = np.array(data['label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=13, stratify=y)


### Chart
# before SMOTE
# counter_before = Counter(y_train)
# classes = list(counter_before.keys())
# counts_before = list(counter_before.values())
### Chart


# Balance the dataset
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)


### Chart
# after SMOTE
# counter_after = Counter(y_train_balanced)
# counts_after = list(counter_after.values())
### Chart


### Chart
# bar chart
# fig, ax = plt.subplots()
# bar_width = 0.35
# index = np.arange(len(classes))

# bar1 = ax.bar(index, counts_before, bar_width, label='Before SMOTE')
# bar2 = ax.bar(index + bar_width, counts_after, bar_width, label='After SMOTE')

# ax.set_xlabel('Class')
# ax.set_ylabel('Count')
# ax.set_title('Class Distribution Before and After SMOTE')
# ax.set_xticks(index + bar_width / 2)
# ax.set_xticklabels(classes)
# ax.legend()

# plt.show()
### Chart


# Initialize and fit scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest model
rf_model = train_random_forest(X_train_scaled, y_train_balanced)


### Learning Curves Chart
# train_sizes, train_scores, test_scores = learning_curve(rf_model, X_train_scaled, y_train_balanced, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

# train_scores_mean = np.mean(train_scores, axis=1)
# test_scores_mean = np.mean(test_scores, axis=1)

# plt.figure()
# plt.plot(train_sizes, train_scores_mean, label="Training score", color="r")
# plt.plot(train_sizes, test_scores_mean, label="Cross-validation score", color="g")
# plt.title("Learning Curves")
# plt.xlabel("Training examples")
# plt.ylabel("Score")
# plt.legend(loc="best")
# plt.grid()
# plt.show()
### Learning Curves Chart


# Use 3-fold cross-validation with stratified splits
stratified_kfold = StratifiedKFold(n_splits=3)
validator = cross_val_score(rf_model, X_test_scaled, y_test, cv=stratified_kfold)
score = rf_model.score(X_test_scaled, y_test)

print(f'Random Forest Score: {score}')
print(f'Random Forest Cross Validation Score: {validator}')

y_pred = rf_model.predict(X_test_scaled)
print(classification_report(y_test, y_pred, digits=4))

print('\n=========================== DEBUG ===========================')

example_value = {
    "temperature": 26.1,
    "light": 3712,
    "conductivity": 1921,
    "moisture": 32
}

# Scale example_value and make a prediction
example_features = np.array([[example_value['temperature'], example_value['light'], example_value['conductivity'], example_value['moisture']]])
example_scaled = scaler.transform(example_features)
example_prediction = rf_model.predict(example_scaled)
example_prediction_proba = rf_model.predict_proba(example_scaled)

print(f'Example features: {example_value}')
print(f'Scaled example features: {example_scaled}')
print(f'Example prediction: {example_prediction}')
print(f'Example prediction probabilities: {example_prediction_proba}')

print('=========================== DEBUG ===========================\n')

# Save the model and scaler
pickle.dump(rf_model, open('rf_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

@app.route('/')
def hello_world():
    return 'Hello from Flask!'


@app.route('/predict', methods=['GET'])
def coba():
    return 'Coba'


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

    # print(f'Received features: {features}')

    # Apply the same scaling as during training
    features_scaled = scaler.transform(features)
    # print(f'Scaled features: {features_scaled}')

    label_names = {
        0: 'Optimal',
        1: 'Caution',
        2: 'Extreme'
    }

    pred = model.predict(features_scaled)
    pred_proba = model.predict_proba(features_scaled)

    pred_label_index = int(pred[0])
    pred_label_name = label_names[pred_label_index]

    # print('\n=========================== DEBUG ===========================')
    # print(f'Prediction: {pred_label_name} - Prediction index: {pred}')
    # print(f'Prediction probability: {pred_proba}')
    # print('=========================== DEBUG ===========================\n')

    return jsonify({
        'prediction': pred_label_name,
        'prediction_index': pred_label_index,
        'probability': pred_proba[0].tolist()
    })