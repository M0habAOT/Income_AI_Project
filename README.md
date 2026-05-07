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

---

### Model 2: Support Vector Machine (SVM)

### Model: Support Vector Machine (SVM)
For this classification task, I selected the **Support Vector Machine (SVM)**. This choice was driven by SVM's proven effectiveness in high-dimensional spaces and its ability to define complex, non-linear decision boundaries using kernel functions. Specifically, the **Radial Basis Function (RBF)** kernel was used to capture intricate patterns within the census data.

To optimize the model, I focused on the **'C' hyperparameter**, which acts as a regularization parameter. It controls the trade-off between achieving a low error on the training data and maximizing the margin of the decision boundary.

---

### **Evaluation Results (C = 1)**
Using the default regularization strength ($C=1$), the model achieved a robust balance between precision and recall. The results on the test set are as follows:

| Metric | Value |
| :--- | :--- |
| **Accuracy** | 83.74% |
| **Precision** | 70.15% |
| **Recall** | 54.26% |
| **F1-Score** | 61.19% |

---

### **Confusion Matrix Analysis**
The confusion matrix below illustrates the model's performance in terms of True Positives, True Negatives, False Positives, and False Negatives.

| | Predicted: <=50K | Predicted: >50K |
| :--- | :--- | :--- |
| **Actual: <=50K** | **11,542** (TN) | **888** (FP) |
| **Actual: >50K** | **1,759** (FN) | **2,087** (TP) |

* **True Negatives (11,542):** The model accurately identifies the majority of individuals in the lower-income bracket.
* **False Positives (888):** A relatively small number of individuals were incorrectly predicted to earn >50K.
* **False Negatives (1,759):** This indicates that approximately 45% of high-income individuals were misclassified, reflecting the model's conservative nature in the face of class imbalance.
* **True Positives (2,087):** The model correctly identified over half of the high-income population.

---

### **Hyperparameter Analysis**
Based on previous experiments with varying values of **C**:
* **Low C (0.01):** Produced a "stiffer" boundary. While this increased precision (fewer false alarms), it severely hampered recall, as the model became too cautious about predicting the positive class (>50K).
* **High C (100):** Made the model more aggressive. While this improved recall (finding more high-income individuals), it led to a higher rate of false positives, effectively reducing overall precision.
* **Optimal C (1):** As shown in the results above, $C=1$ provided the most balanced performance across all evaluation metrics, particularly for the F1-score.

---

## 4. Data Interpretation & Insights
The SVM model's performance confirms several key aspects of the dataset:
* **Non-Linearity:** The effectiveness of the RBF kernel suggests that income levels are not separated by simple linear thresholds but by complex interactions between features like age, education level, and capital gains.
* **Class Imbalance:** The higher number of False Negatives compared to False Positives shows that the model is naturally biased towards the majority class (<=50K), requiring careful tuning to ensure high-income individuals are caught.

---

## 5. Conclusion
The Support Vector Machine model reached an accuracy of **83.74%**. Through the analysis of the confusion matrix and hyperparameter tuning, it was determined that $C=1$ with an RBF kernel serves as the optimal configuration for this specific problem. 

The final model has been serialized and saved as `SVMmodel.pkl` using the `pickle` library. This ensures that the exact state of the trained model, including the learned support vectors and decision boundaries, is preserved for future inference and deployment.
