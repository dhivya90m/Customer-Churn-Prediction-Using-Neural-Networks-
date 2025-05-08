# Customer-Churn-Prediction-Using-Neural-Networks

# Bank Churn Prediction 🚀
**Author**: Dhivya Marimuthu  
**Program**: Post Graduate Program in Artificial Intelligence and Machine Learning (UT Austin)

📌 **Project Overview**  
The banking industry faces the challenge of customer churn, where customers leave the bank for competitors. This project aims to build a neural network-based model to predict whether a customer will leave the bank in the next 6 months. By identifying at-risk customers early, the bank can take action to retain them, ultimately reducing churn and improving customer loyalty.

🎯 **Objective**  
- Predict whether a customer will churn (leave the bank) within the next 6 months.
- Identify the key factors that influence customer churn.
- Provide actionable insights for the bank to reduce churn and improve customer retention.

📊 **Dataset Description**  
The dataset consists of customer records, with a mix of numerical and categorical features.

| **Feature**        | **Description**                                                |
|--------------------|----------------------------------------------------------------|
| **CustomerId**     | Unique ID assigned to each customer                            |
| **Surname**        | Last name of the customer                                      |
| **CreditScore**    | Credit score of the customer                                   |
| **Geography**      | Customer’s location                                            |
| **Gender**         | Gender of the customer                                         |
| **Age**            | Age of the customer                                            |
| **Tenure**         | Number of years the customer has been with the bank            |
| **NumOfProducts**  | Number of products the customer has with the bank               |
| **Balance**        | Account balance of the customer                                |
| **HasCrCard**      | Whether the customer has a credit card (1 = Yes, 0 = No)        |
| **EstimatedSalary**| Estimated salary of the customer                               |
| **isActiveMember** | Whether the customer is an active member (1 = Yes, 0 = No)      |
| **Exited**         | Target variable: whether the customer left the bank (1 = Yes, 0 = No) |

🔍 **Exploratory Data Analysis (EDA)**  
- Univariate & bivariate analysis were conducted using histograms, boxplots, and scatter plots.
  
**Key Insights**:
- Older customers and those with higher balances are less likely to churn.
- Customers with fewer products and lower credit scores have a higher likelihood of leaving the bank.
- Customers with low tenure are at higher risk of churning.

🏗️ **Data Preprocessing**  
- **Missing Values**: No missing values found in the dataset.
- **Outlier Treatment**: Outliers were analyzed using boxplots, especially in numerical features like `CreditScore` and `Balance`.
- **Feature Engineering**: 
  - Removed non-predictive features (e.g., `Surname`, `CustomerId`).
  - Encoded categorical variables (e.g., `Geography`, `Gender`, `HasCrCard`, `isActiveMember`).

💻 **Technologies Used**  
- Python (pandas, NumPy, scikit-learn, TensorFlow, matplotlib, seaborn)
- Neural Networks (MLP) for churn prediction
- Data visualization using matplotlib and seaborn

🔥 **Model Building - Neural Network**  
A multi-layer perceptron (MLP) neural network was implemented for the churn prediction task. The model was trained using the Adam optimizer and tuned for optimal performance.  

**Model Evaluation Metrics**:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix

📈 **Model Performance & Evaluation**  
| **Model Version**       | **Train Accuracy** | **Test Accuracy**  | **Notes**                           |
|-------------------------|--------------------|--------------------|-------------------------------------|
| **Initial Neural Network** | 84%                | 50%                | High recall, but low precision     |
| **Optimized Model (Post)** | 82%                | 58%                | Reduced overfitting, more balanced performance |

**Feature Importance**:
- `Age`, `Balance`, `NumOfProducts`, and `CreditScore` are the most important predictors for churn.

**Confusion Matrix Analysis**:
- The model focuses on correctly identifying churners (high recall) at the cost of precision. This is aligned with business goals where preventing churn is a priority.

💡 **Actionable Insights & Business Recommendations**  
🔹 **Target High-Risk Customers**: Focus retention efforts on customers with low tenure, fewer products, and lower credit scores.  
🔹 **Leverage Account Balances**: Customers with lower balances are at higher risk of churn. Personalized offers can be made to these customers to retain them.  
🔹 **Customer Engagement**: Increase engagement with customers who have been with the bank for fewer years and have low activity levels (`isActiveMember`).  
🔹 **Personalized Offers**: Customers who are predicted to churn could be offered loyalty programs or targeted promotions based on their behavior.  
🔹 **Monitor High-Risk Segments**: Implement early-warning systems for customers at high risk of churning, focusing on features like `Age`, `Tenure`, and `CreditScore`.

## Installation

### Requirements  
- Python 3.8 or higher  
- Libraries:  
  - `pandas`  
  - `numpy`  
  - `scikit-learn`  
  - `tensorflow`  
  - `imbalanced-learn`  
  - `matplotlib`  
  - `seaborn`

### Installation Instructions

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/bank-churn-prediction.git
   cd bank-churn-prediction
