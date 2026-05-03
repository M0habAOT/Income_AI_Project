# Income_AI_Project

## 3. Modeling and Hyperparameter Tuning

### Model 1: Logistic Regression
I chose Logistic Regression as the primary classifier. To optimize the model, I experimented with the **'C' parameter** (the inverse of regularization strength) to observe its impact on performance.

#### **Experimentation Results:**
| Hyperparameter (C) | Overall Accuracy | Precision (Class 1) | Recall (Class 1) |
| :--- | :--- | :--- | :--- |
| **C = 0.01 (Strong Reg)** | 83.84% | **0.72** | 0.52 |
| **C = 1.0 (Default)** | 83.84% | 0.70 | 0.56 |
| **C = 100 (Weak Reg)** | 83.82% | 0.69 | 0.56 |

#### **Analysis of Hyperparameters:**
* **Strong Regularization ($C=0.01$):** Improved the **Precision to 0.72**. This means the model became more reliable when predicting high-income individuals, although it became more conservative (lower recall).
* **Weak Regularization ($C=100$):** The model became more aggressive in its predictions, which slightly decreased precision and increased the number of **False Positives (957)**.
* **Impact:** The experimentation showed that a smaller 'C' value helps in achieving more reliable predictions for the high-income class by reducing potential overfitting.

---

## 4. Data Visualization & Analysis
Based on the feature coefficients from the Logistic Regression model, the following key factors were identified:

* **Education-num:** Had a strong positive correlation with income, meaning higher education levels significantly increase the probability of earning >50K.
* **Age:** Also showed a positive trend, as older individuals generally have more work experience and higher salaries.
* **Capital-gain:** Identified as a major factor in predicting high-income individuals.

---

## 5. Conclusion
The Logistic Regression model performed consistently with an accuracy of approximately **84%**. Through hyperparameter tuning, I found that adjusting the regularization strength (**C**) allows for a strategic trade-off between precision and recall. For this specific dataset, a balanced 'C' provides a robust model capable of identifying high-income individuals while maintaining high overall accuracy.


### Model 2: Support Vector Machine (SVM)

I selected the **Support Vector Machine (SVM)** as the classification model due to its effectiveness in handling high-dimensional data and its ability to create non-linear decision boundaries using the RBF kernel.

To optimize the model, I experimented with the **'C' hyperparameter**, which controls the trade-off between maximizing the margin and minimizing classification error.

---

### **Experimentation Results:**

| Hyperparameter (C)                   | Overall Accuracy | Precision (Class 1) | Recall (Class 1) | F1-Score |
| :----------------------------------- | :--------------- | :------------------ | :--------------- | :------- |
| **C = 0.01 (Strong Regularization)** | 82.56%           | **0.74**            | 0.41             | 0.52     |
| **C = 1 (Default)**                  | **83.74%**       | 0.70                | 0.54             | **0.61** |
| **C = 100 (Weak Regularization)**    | 82.52%           | 0.65                | **0.57**         | 0.61     |

---

### **Confusion Matrix Analysis**

* **C = 0.01:**

  * Very high Precision (0.74)
  * Low Recall (0.41)
  * The model is conservative → predicts fewer positive cases
  * High False Negatives (2281)

* **C = 1:**

  * Balanced performance across all metrics
  * Best overall Accuracy and F1-score
  * Moderate False Positives and False Negatives

* **C = 100:**

  * Higher Recall (0.57)
  * Lower Precision (0.65)
  * The model becomes more aggressive → predicts more positives
  * Increased False Positives (1208)

---

### **Analysis of Hyperparameters**

* **Strong Regularization (C = 0.01):**
  The model focuses on simplicity and avoids overfitting. This leads to **higher precision** but significantly reduces recall, meaning many high-income individuals are missed.

* **Moderate Regularization (C = 1):**
  Provides the **best balance** between precision and recall. This results in the highest overall performance, making it the most suitable choice.

* **Weak Regularization (C = 100):**
  The model tries to fit the training data more closely, increasing recall but also increasing false positives, which reduces precision.

---

## 4. Data Interpretation & Insights

From the behavior of the SVM model and the dataset characteristics:

* The model shows sensitivity to class imbalance, where predicting the majority class (low income) is easier.
* Increasing the value of **C** allows the model to capture more complex patterns, improving recall but reducing precision.
* A balanced configuration (C = 1) provides the best trade-off between detecting high-income individuals and avoiding incorrect predictions.

---

## 5. Conclusion

The SVM model achieved a maximum accuracy of approximately **83.7%**.

Through hyperparameter tuning, it was observed that the value of **C** plays a crucial role in controlling the model behavior:

* Lower values → safer, more precise predictions
* Higher values → more aggressive predictions with better recall

For this dataset, **C = 1** provided the best balance between **Precision, Recall, and F1-score**, making it the optimal choice for the final model.

---
