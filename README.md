
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

The confusion matrix is a key metric for understanding how well the model is classifying both churned and non-churned customers.

### Feature Importance

The feature importance plot reveals which features contribute the most to predicting churn. Based on the trained model, the top features influencing churn prediction are:

1. **MonthlyCharges**
2. **Tenure**
3. **TotalCharges**
4. **InternetService**

## Usage

1. Install the required libraries:
   ```bash
   ![Screenshot 2025-04-18 151054](https://github.com/user-attachments/assets/853b6c65-ea71-4099-bd40-d351dcd5e6de)
![Screenshot 2025-04-18 151054](https://github.com/user-attachments/assets/853b6c65-ea71-4099-bd40-d351dcd5e6de)
![Screenshot 2025-04-18 151127](https://github.com/user-attachments/assets/ab7e426c-37b0-4153-9e2a-a2f5a951fa25)
![Screenshot 2025-04-18 151201](https://github.com/user-attachments/assets/0254fdb3-34c9-4e28-9eec-530be3d7476d)
![Screenshot 2025-04-18 151230](https://github.com/user-attachments/assets/403c317f-8299-4dcc-a2c3-7b6cc989fabb)
![Screenshot 2025-04-18 151249](https://github.com/user-attachments/assets/d2aea82a-e1cb-4bff-8f4c-5f1d4bcd8eb3)
   pip install pandas numpy seaborn matplotlib scikit-learn
