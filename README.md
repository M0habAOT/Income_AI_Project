# Income_Classification_Project
---
### Model 2: Support Vector Machine (SVM)
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

To understand the effect of the regularization parameter **C**, experiments were conducted with three values: **C = 0.01**, **C = 1**, and **C = 100**.

#### **Evaluation Metrics Across C Values**

| Hyperparameter (C) | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **C = 0.01 (Strong Reg)** | 82.56% | 73.72% | 40.69% | 52.44% |
| **C = 1 (Default)** | 83.74% | 70.15% | 54.26% | 61.19% |
| **C = 100 (Weak Reg)** | 82.52% | 64.65% | 57.44% | 60.83% |

#### **Confusion Matrices Across C Values**

**C = 0.01**

| | Predicted: <=50K | Predicted: >50K |
| :--- | :--- | :--- |
| **Actual: <=50K** | **11,872** (TN) | **558** (FP) |
| **Actual: >50K** | **2,281** (FN) | **1,565** (TP) |

**C = 1**

| | Predicted: <=50K | Predicted: >50K |
| :--- | :--- | :--- |
| **Actual: <=50K** | **11,542** (TN) | **888** (FP) |
| **Actual: >50K** | **1,759** (FN) | **2,087** (TP) |

**C = 100**

| | Predicted: <=50K | Predicted: >50K |
| :--- | :--- | :--- |
| **Actual: <=50K** | **11,222** (TN) | **1,208** (FP) |
| **Actual: >50K** | **1,637** (FN) | **2,209** (TP) |

#### **Observations**
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