import pandas as pd
import joblib

# Load feature names (from any dataset)
train = pd.read_csv("Preprocessing/processed_train_data.csv")
features = train.drop(columns=["Income"]).columns


print("\n=== ML Predictor ===")
print("Choose Model:")
print("1 - SVM")
print("2 - Decision Tree")
print("3 - Logistic Regression")

choice = input("\nEnter choice: ")

# map choice to model file
if choice == "1":
    model_path = "models/SVMmodel.pkl"
    model_name = "SVM"

elif choice == "2":
    model_path = "best_decision_tree_model.pkl"
    model_name = "Decision Tree"

elif choice == "3":
    model_path = "models/Logistic_Model.pkl"
    model_name = "Logistic Regression"

else:
    print("Invalid choice!")
    exit()


# load model
model = joblib.load(model_path)

print(f"\nYou selected: {model_name}")

# user input
user_data = []

for col in features:
    val = float(input(f"{col}: "))
    user_data.append(val)

# prediction
prediction = model.predict([user_data])

print("\n===================")
print("Prediction:", prediction[0])
print("===================")