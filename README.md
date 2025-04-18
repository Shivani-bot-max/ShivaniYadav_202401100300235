# Customer Churn Classifier

## Problem Statement

Customer churn is a critical issue for many businesses, especially in subscription-based services like telecommunications, streaming platforms, and online services. **Churn** refers to customers who stop using a service over a period of time. High churn rates can significantly impact a business's profitability and growth. 

The goal of this project is to **predict customer churn** using machine learning techniques. By identifying customers who are likely to churn, businesses can take preventive measures such as targeted offers, customer retention strategies, or personalized services to retain those customers.

The dataset provided contains various customer-related features and whether the customer has churned or not. The task is to build a predictive model that can accurately forecast customer churn based on these features.

## Libraries Used

- **pandas**: For data manipulation and cleaning.
- **numpy**: For numerical operations and array manipulation.
- **seaborn**: For statistical data visualization and plotting.
- **matplotlib**: For creating various types of visualizations.
- **scikit-learn (sklearn)**: For machine learning tasks like data splitting, preprocessing, and model training.

## Dataset

The dataset contains customer information and whether they churned (left the service or not). Key features include:

- **customerID**: Unique identifier for each customer.
- **Gender**: Customer's gender.
- **SeniorCitizen**: Whether the customer is a senior citizen.
- **Partner**: Whether the customer has a partner.
- **Dependents**: Whether the customer has dependents.
- **Tenure**: Duration the customer has been with the service.
- **PhoneService**: Whether the customer uses phone service.
- **InternetService**: Whether the customer uses internet service.
- **MonthlyCharges**: Monthly charges for the customer.
- **TotalCharges**: Total charges for the customer up until now.
- **Churn**: Whether the customer has churned (target variable).

## Steps in the Project

1. **Data Preprocessing**:
   - Dropped unnecessary columns.
   - Converted 'TotalCharges' to numeric.
   - Filled missing values with median values.
   - Encoded categorical variables using Label Encoding.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized churn distribution using count plots.
   - Created a correlation heatmap to understand relationships between features.

3. **Data Splitting**:
   - Split the dataset into training and testing sets using `train_test_split`.
   - Scaled numerical features using `StandardScaler`.

4. **Model Training**:
   - Trained a **Random Forest Classifier** model to predict churn.

5. **Evaluation**:
   - Evaluated the model using accuracy, precision, and recall.
   - Displayed a confusion matrix to visualize model performance.
   - Plotted feature importance to understand which features are most important in predicting churn.

## Results

After training the **Random Forest Classifier** and evaluating its performance, the following results were obtained:

- **Accuracy**: `XX.XX%` (The percentage of correctly predicted instances out of all predictions).
- **Precision**: `YY.YY%` (The percentage of positive predictions that were actually positive).
- **Recall**: `ZZ.ZZ%` (The percentage of actual positive cases correctly identified by the model).

### Confusion Matrix

The confusion matrix for the model is as follows:

|               | Predicted: No Churn | Predicted: Churn |
|---------------|---------------------|------------------|
| **Actual: No Churn** |  **TN**           | **FP**          |
| **Actual: Churn**    |  **FN**           | **TP**          |

Where:
- **TP** = True Positive
- **TN** = True Negative
- **FP** = False Positive
- **FN** = False Negative

### Feature Importance

The feature importance plot reveals which features contribute the most to predicting churn. Based on the trained model, the top features influencing churn prediction are:

1. **MonthlyCharges**
2. **Tenure**
3. **TotalCharges**
4. **InternetService**

## Visualizations

### 1. Churn Distribution

![Churn Distribution](images/churn_distribution.png)

### 2. Correlation Heatmap

![Correlation Heatmap](images/correlation_heatmap.png)

### 3. Confusion Matrix

![Confusion Matrix](images/confusion_matrix.png)

### 4. Feature Importance

![Feature Importance](images/feature_importance.png)

## Usage

1. Install the required libraries:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
