# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
data = pd.read_csv(r'C:\Users\DELL\Desktop\Intern Intelligence\heart\dataset_heart.csv')  # Replace with your dataset file name

# Rename columns for consistency
data.rename(columns={
    'sex ': 'sex',
    'chest pain type': 'chest_pain_type',
    'resting blood pressure': 'resting_blood_pressure',
    'serum cholestoral': 'serum_cholesterol',
    'fasting blood sugar': 'fasting_blood_sugar',
    'resting electrocardiographic results': 'resting_ecg',
    'max heart rate': 'max_heart_rate',
    'exercise induced angina': 'exercise_induced_angina',
    'ST segment': 'st_segment',
    'major vessels': 'major_vessels',
    'heart disease': 'heart_disease'
}, inplace=True)

# Define features and target
X = data.drop('heart_disease', axis=1)
y = data['heart_disease']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Save the trained model
joblib.dump(model, 'heart_disease_model.pkl')
print("Model saved as 'heart_disease_model.pkl'")
