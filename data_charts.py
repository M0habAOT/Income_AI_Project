# ///////
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (8,5)

processed_data=pd.read_csv("Preprocessing/processed_train_data.csv")

def draw():
    num_cols = ["age","fnlwgt","education-num","capital-gain","capital-loss","hours-per-week"]

    for col in num_cols:
        plt.figure()
        sns.histplot(processed_data[col], kde=True)
        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()

    plt.figure()
    sns.countplot(x="Income", data=processed_data)
    plt.title("Target Distribution (Income)")
    plt.xlabel("Income")
    plt.ylabel("Count")
    plt.show()

    print("Target :")
    print(processed_data["Income"].value_counts(normalize=True))

    plt.figure(figsize=(10,8))
    corr = processed_data[num_cols].corr()

    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap (Numerical Features)")
    plt.show()


draw()