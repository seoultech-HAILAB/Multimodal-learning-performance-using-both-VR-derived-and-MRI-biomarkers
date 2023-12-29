import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

# Load the dataset
data = pd.read_csv('data.csv')

# Define features and target variable
features = {
    'VR': [
        'Movement_speed_in_all_steps_in_all_stepsms',
        'Scanpath_length_in_all_stepsm',
        'The_number_of_errors_in_all_steps',
        'The_time_to_completion_in_all_stepss',
    ],
    'MRI': [
        'PERC_OF_ICV_subctx_lh_amygdala',
        'PERC_OF_ICV_subctx_rh_amygdala',
        'PERC_OF_ICV_subctx_lh_hippocampus',
        'PERC_OF_ICV_subctx_rh_hippocampus',
        'PERC_OF_ICV_ctx_rh_entorhinal',
        'PERC_OF_ICV_ctx_lh_entorhinal'
    ]
}
target = data['Group']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(data[features['VR'] + features['MRI']], target, test_size=0.3, random_state=42)

# Define the classifier (SVC)
svc_classifier = SVC()

# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': [0.01, 0.1, 1],
    'kernel': ['linear', 'rbf'],
}

# Perform grid search with 3-fold cross-validation
grid_search = GridSearchCV(svc_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters
best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Use the best parameters to train the final model
best_svc_classifier = SVC(**best_params)
best_svc_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = best_svc_classifier.predict(X_test)

# Evaluate the performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", accuracy)

# Use the best parameters to train the final SVC model
final_svc_classifier = SVC(**best_params)
final_svc_classifier.fit(X_train, y_train)

# Function to calculate and print evaluation metrics
def evaluate_model(y_true, y_pred, selected_features, params_used):
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred, pos_label=1, average='binary')  # Adjust pos_label to 1 for 'MCI'
    specificity = recall_score(y_true, y_pred, pos_label=0, average='binary')  # Adjust pos_label to 0 for 'HC'
    precision = precision_score(y_true, y_pred, pos_label=1, average='binary')  # Adjust pos_label to 1 for 'MCI'
    f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')  # Adjust pos_label to 1 for 'MCI'
    auc = roc_auc_score(y_true, y_pred)

    print("Selected Features:", selected_features)
    print(f"Accuracy: {round(accuracy * 100, 1)}%")
    print(f"Sensitivity: {round(sensitivity * 100, 1)}%")
    print(f"Specificity: {round(specificity * 100, 1)}%")
    print(f"Precision: {round(precision * 100, 1)}%")
    print(f"F1 Score: {round(f1 * 100, 1)}%")
    print(f"AUC: {round(auc, 2)}")

# Iterate through different feature subsets
best_evaluation_metrics = {
    'accuracy': 0,
    'sensitivity': 0,
    'specificity': 0,
    'precision': 0,
    'f1_score': 0,
    'auc': 0
}

for i in range(1, len(features['VR'] + features['MRI']) + 1):
    for subset_features in combinations(features['VR'] + features['MRI'], i):
        subset_features = list(subset_features)
        X_train_subset = X_train[subset_features]
        X_test_subset = X_test[subset_features]

        # Train the model on the subset of features
        final_svc_classifier.fit(X_train_subset, y_train)

        # Make predictions on the test set
        y_pred_subset = final_svc_classifier.predict(X_test_subset)

        # Evaluate the performance on the subset of features
        print(f"\nEvaluation metrics for {len(subset_features)} features:")
        evaluate_model(y_test, y_pred_subset, subset_features, final_svc_classifier.get_params())

        # Update the best evaluation metrics if necessary
        if best_evaluation_metrics['accuracy'] < accuracy_score(y_test, y_pred_subset):
            best_evaluation_metrics['accuracy'] = accuracy_score(y_test, y_pred_subset)
            best_evaluation_metrics['sensitivity'] = recall_score(y_test, y_pred_subset, pos_label=1, average='binary')
            best_evaluation_metrics['specificity'] = recall_score(y_test, y_pred_subset, pos_label=0, average='binary')
            best_evaluation_metrics['precision'] = precision_score(y_test, y_pred_subset, pos_label=1, average='binary')
            best_evaluation_metrics['f1_score'] = f1_score(y_test, y_pred_subset, pos_label=1, average='binary')
            best_evaluation_metrics['auc'] = roc_auc_score(y_test, y_pred_subset)

# Print the best evaluation metrics
print("\nBest Evaluation Metrics:")
for metric, value in best_evaluation_metrics.items():
    if metric == 'auc':
        print(f"{metric.capitalize()}: {value:.2f}")
    else:
        print(f"{metric.capitalize()}: {value:.1%}")