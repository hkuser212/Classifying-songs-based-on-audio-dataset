import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('features_3_sec.csv')
df = df.iloc[0:, 1:]

# Separate target variable and features
y = df['label']
X = df.drop('label', axis=1)

# MinMax scaling
min_max_scaler = preprocessing.MinMaxScaler()
np_scaled = min_max_scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns=X.columns)

# Map labels to integers
y = y.map({'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
           'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def model_assess(model, title="Default"):
    # Fit the model
    model.fit(X_train, y_train)

    # Make predictions
    preds = model.predict(X_test)

    # Print evaluation metrics
    print(f'Accuracy {title}: {accuracy_score(y_test, preds):.5f}')
    print('Confusion Matrix:\n', confusion_matrix(y_test, preds))

    # If multiclass, calculate ROC-AUC
    try:
        print('ROC-AUC:', round(roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'), 5))
    except ValueError:
        pass  # Ignore if ROC-AUC cannot be calculated for this model

    # Plot confusion matrix
    sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt="d", cmap="Blues")
    plt.title(f'Confusion Matrix: {title}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

# Check the number of features used for training
print(f'Number of features the model was trained on: {X_train.shape[1]}')

# Initialize the model
xgb_model = XGBClassifier(n_estimators=1000, learning_rate=0.05)

# Evaluate the model
model_assess(xgb_model, "Cross Gradient Booster")

# Save the model
xgb_model.save_model('xgb.model')
# Get feature importance

importances = xgb_model.get_booster().get_score(importance_type='weight')

# Print feature importance
for feature, importance in importances.items():
    print(f'{feature}: {importance}')
importances = xgb_model.feature_importances_

# Visualizing feature importances
plt.barh(X.columns, importances)
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.savefig('feature_importance.png')
