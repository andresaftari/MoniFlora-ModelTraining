import firebase_admin
from firebase_admin import credentials, db
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, learning_curve, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from collections import Counter


cred = credentials.Certificate('/Users/andresaftari/Development/Kuliah/skripsyit-training/moniflora-backup-firebase-adminsdk-uh66r-b9af29113e.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://moniflora-backup-default-rtdb.firebaseio.com/'
})

def get_dataset():
    ref = db.reference('sensor')
    query = ref.order_by_child('light').start_at(600).limit_to_last(2000)
    results = query.get()

    if results:
        # results_list = [(key, value) for key, value in results.items()]
        # results_list.sort(key=lambda x: x[1]['timestamp'], reverse=True)

        # limited_results = results_list[:1500]

        print(f'Total dataset: {len(ref.get())}')
        print(f'Selected dataset with light >= 600: {len(results)}')
        print(f'Dataset limited to the last 2000 entries: {len(results)}')

        # limited_results_dict = {key: value for key, value in results}
    else:
        print('No records found with light >= 600')
        # limited_results_dict = {}

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

    if ((20 <= temp < 22 or 27 < temp <= 30) or
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
    param_grid = {
        
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2, 4],
    }
        
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    forest = RandomForestClassifier(n_estimators=50, max_depth=5, max_features='log2', **grid_search.best_params_, warm_start=True, oob_score=True)
    forest.fit(X_train, y_train)
    print('Best parameters:', best_params)
    
    for _ in range(50):
        forest.n_estimators += 1
        forest.fit(X_train, y_train)
        oob_score = forest.oob_score
        
        # print(f"Iteration {i+1}, OOB score: {oob_score:.4f}")
        if oob_score < 0.9:  # stop when OOB score drops below 0.9
            break
    
    
    # forest = RandomForestClassifier(max_depth=5, max_features='log2', criterion='entropy', warm_start=True, min_samples_leaf=2, n_estimators=50, oob_score=True)
    # forest.fit(X_train, y_train)
    
    return forest


# Load data
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

# print(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=13, stratify=y)

# Balance the dataset
smote = SMOTE()
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# for label, count in Counter(y_test).items():
#     print(f'{label_names[label]}: {count} datasets')

# Initialize and fit scaler on the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_balanced)
X_test_scaled = scaler.transform(X_test)


# Train Random Forest model
rf_model = train_random_forest(X_train_scaled, y_train_balanced)

# Use 3-fold cross-validation with stratified splits
stratified_kfold = StratifiedKFold(n_splits=5)
validator = cross_val_score(rf_model, X_test_scaled, y_test, cv=stratified_kfold)
score = rf_model.score(X_test_scaled, y_test)

# Predict and apply fuzzification
y_pred = rf_model.predict(X_test_scaled)
# fuzzified_pred = fuzzify_predictions(y_pred)

print(f'Random Forest Score: {score}')
print(f'Random Forest Cross Validation Score: {validator}')
print(classification_report(y_test, y_pred, digits=2))

# Example value testing
example_input = np.array([[25, 4000, 2000, 40]])  # example values for temperature, light, conductivity, and moisture
example_pred = rf_model.predict(example_input)
print("Example value prediction:", example_pred)

### Learning Curves Chart
train_sizes, train_scores, test_scores = learning_curve(rf_model, X_train_scaled, y_train_balanced, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))

train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label='Training score', color='g')
plt.plot(train_sizes, test_scores_mean, label='Cross-validation score', color='r')
plt.title("Learning Curves")
plt.xlabel("Training examples")
plt.ylabel("Score")
plt.legend(loc="best")
plt.grid()
plt.show()
### Learning Curves Chart