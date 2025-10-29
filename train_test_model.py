import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib

# Load data
with open('./data.pickle', 'rb') as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print(f"Dataset shape: {data.shape}")
print(f"Classes: {np.unique(labels)}")

# Split data
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Standardize features (important for Random Forest with normalized data)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Random Forest with better parameters
model = RandomForestClassifier(
    n_estimators=200,           
    max_depth=20,                
    min_samples_split=5,         
    min_samples_leaf=2,          
    max_features='sqrt',         
    random_state=42,
    n_jobs=-1,                   
    class_weight='balanced'      
)

print("Training model...")
model.fit(x_train_scaled, y_train)

# testing the model
y_pred = model.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n Test Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save both model and scaler
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\n Model and scaler saved!")