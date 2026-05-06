import pandas as pd
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

train = pd.read_csv("Preprocessing/processed_train_data.csv")
test  = pd.read_csv("Preprocessing/processed_test_data.csv")

X_train = train.drop(columns=["Income"])
y_train = train["Income"]
X_test  = test.drop(columns=["Income"])
y_test  = test["Income"]

# SVM
svm = SVC(kernel='rbf', C=1)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)

print("\n=== SVM ===")
print("Accuracy :", accuracy_score(y_test, svm_pred))
print("Precision:", precision_score(y_test, svm_pred))
print("Recall   :", recall_score(y_test, svm_pred))
print("F1-Score :", f1_score(y_test, svm_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

filename = 'SVMmodel.pkl'
pickle.dump(svm, open(filename, 'wb'))

print(f"Model saved successfully as {filename}")