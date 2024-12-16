import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump

# Load dataset
data_dict = pickle.load(open('./alphabet_data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, shuffle=True, stratify=labels, random_state=42
)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(x_train, y_train)

# Evaluate model
y_predict = model.predict(x_test)
print(f"Accuracy: {accuracy_score(y_test, y_predict) * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_predict))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_predict))

# Cross-validation
cv_scores = cross_val_score(model, data, labels, cv=5, scoring='accuracy')
print(f"\nCross-validation accuracy: {cv_scores.mean() * 100:.2f}%")

# Save trained model
dump(model, 'alphabet_model.joblib')
print("Model saved as 'alphabet_model.joblib'")
