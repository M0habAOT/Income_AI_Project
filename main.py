import pandas as pd
import joblib

# Load all fitted objects from pkl files
encoders = joblib.load("encoders.pkl")
onehotencoder = joblib.load("onehotencoder.pkl")
scale = joblib.load("scale.pkl")

# Load feature names from processed train data
train = pd.read_csv("Preprocessing/processed_train_data.csv")
features = train.drop(columns=["Income"]).columns

print("\n=== ML Predictor ===")
print("Choose Model:")
print("1 - SVM")
print("2 - Decision Tree")
print("3 - Logistic Regression")

choice = input("\nEnter choice: ")

if choice == "1":
    model_path = "SVMmodel.pkl"
    model_name = "SVM"
elif choice == "2":
    model_path = "best_decision_tree_model.pkl"
    model_name = "Decision Tree"
elif choice == "3":
    model_path = "Logistic_Model.pkl"
    model_name = "Logistic Regression"
else:
    print("Invalid choice!")
    exit()

model = joblib.load(model_path)
print(f"\nYou selected: {model_name}")

# User input
user = {
    "age": int(input("Age: ")),
    "workclass": input("Workclass: "),
    "fnlwgt": int(input("fnlwgt: ")),
    "education": input("Education: "),
    "education-num": int(input("Education Number: ")),
    "marital-status": input("Marital Status: "),
    "occupation": input("Occupation: "),
    "relationship": input("Relationship: "),
    "race": input("Race: "),
    "sex": input("Sex: "),
    "capital-gain": int(input("Capital Gain: ")),
    "capital-loss": int(input("Capital Loss: ")),
    "hours-per-week": int(input("Hours per week: ")),
    "native-country": input("Native Country: ")
}

user_df = pd.DataFrame([user])

# Strip whitespace from string columns (same as clean() in preprocessing)
for col in user_df.select_dtypes(include="object").columns:
    user_df[col] = user_df[col].str.strip().str.rstrip(".")

# Label encode 'sex' only (Income is the target, not an input)
user_df["sex"] = encoders["sex"].transform(user_df["sex"])

# One-hot encode categorical columns
cat_cols = ["workclass", "education", "marital-status",
            "occupation", "relationship", "race", "native-country"]

encoded_data = onehotencoder.transform(user_df[cat_cols])
encoded_df = pd.DataFrame(
    encoded_data,
    columns=onehotencoder.get_feature_names_out(cat_cols),
    index=user_df.index
)
user_df = pd.concat([user_df.drop(columns=cat_cols), encoded_df], axis=1)

# Scale numerical columns
num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
user_df[num_cols] = scale.transform(user_df[num_cols])

# Align columns with training data (fill any missing one-hot columns with 0)
user_df = user_df.reindex(columns=features, fill_value=0)

# Predict
prediction = model.predict(user_df)

# Decode prediction back to readable label
predicted_label = encoders["Income"].inverse_transform(prediction)

print("\n===================")
print(f"Predicted Income: {predicted_label[0]}")
print("===================")