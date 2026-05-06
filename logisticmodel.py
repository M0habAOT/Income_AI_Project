import pickle
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

train_data = pd.read_csv('Preprocessing/processed_train_data.csv')
test_data = pd.read_csv('Preprocessing/processed_test_data.csv')

print("--- Train Data Sample ---")
print(train_data.head())

X_train = train_data.drop('Income', axis=1)
y_train = train_data['Income']

model = LogisticRegression(C=1.0, max_iter=1000)
model.fit(X_train, y_train)

print("\nModel trained successfully!")

X_test = test_data.drop('Income', axis=1)
y_test = test_data['Income']

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

filename = 'Logistic_Model.pkl'
pickle.dump(model, open(filename, 'wb'))

print(f"Model saved successfully as {filename}")