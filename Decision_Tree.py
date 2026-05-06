import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns

train=pd.read_csv("https://raw.githubusercontent.com/Nana0191/ai_project/refs/heads/main/processed_train_data.csv")
test=pd.read_csv("https://raw.githubusercontent.com/Nana0191/ai_project/refs/heads/main/processed_test_data.csv")

x_train=train.drop("Income",axis=1)
y_train=train["Income"]

x_test=test.drop("Income",axis=1)
y_test=test["Income"]

model=DecisionTreeClassifier()

param_grid={
    'max_depth':[6,8,10,12],
    'min_samples_split':[5,10,15],
    'min_samples_leaf':[5,10,15],
    'criterion':['gini','entropy']
}

grid=GridSearchCV(estimator=model,param_grid=param_grid,cv=5,scoring="accuracy",n_jobs=-1)
grid.fit(x_train,y_train)

print("Best Parameters:",grid.best_params_)
print("Best Score:",grid.best_score_)

best_model=grid.best_estimator_
y_pred=best_model.predict(x_test)

train_pred=best_model.predict(x_train)

print("TRAIN ACCURACY:", accuracy_score(y_train,train_pred))
print("TEST ACCURACY:", accuracy_score(y_test,y_pred))

importances = best_model.feature_importances_
feat_imp = pd.Series(importances, index=x_train.columns).sort_values(ascending=False)

print(feat_imp.head(10))
print("clas.report:",classification_report(y_test,y_pred))
print("conf.matrix:",confusion_matrix(y_test,y_pred))

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.show()

filename = 'best_decision_tree_model.pkl'
pickle.dump(best_model, open(filename, 'wb'))

print(f"Model saved successfully as {filename}")