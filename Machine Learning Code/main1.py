import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
import joblib

# Read the CSV file
df = pd.read_csv("/content/drive/MyDrive/Diabetes Project ML/diabetes_prediction_dataset.csv")

# Replace 'Male' with 0 and 'Female' with 1 in the 'gender' column
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Mapping string values in 'smoking_history' column to numerical values
smoking_history_mapping = {'never': 0, 'No Info': 1, 'current': 2, 'former': 3, 'ever': 4, 'not current': 5}
df['smoking_history'] = df['smoking_history'].map(smoking_history_mapping)

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features (x) and target (y)
x = df.drop(columns=['diabetes'])
y = df['diabetes']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)

# Impute missing values with the mean of each column
imputer = SimpleImputer(strategy='mean')
x_train_imputed = imputer.fit_transform(x_train)
x_test_imputed = imputer.transform(x_test)

# Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train_imputed)
x_test_scaled = scaler.transform(x_test_imputed)

# Initialize base estimator (Decision Tree)
base_estimator = DecisionTreeClassifier(max_depth=1, random_state=0)

# Initialize AdaBoost classifier
ada_model = AdaBoostClassifier(base_estimator=base_estimator, n_estimators=50, random_state=0)

# Fit the training dataset into the AdaBoost model
ada_model.fit(x_train_scaled, y_train)

# Save the trained model
joblib.dump(ada_model, '/content/drive/MyDrive/Diabetes Project ML/ada_model.sav')

# Make predictions
predictions_ada = ada_model.predict(x_test_scaled)

# Calculate accuracy
accuracy_ada = accuracy_score(y_test, predictions_ada)
print("AdaBoost Accuracy:", accuracy_ada)
