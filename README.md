# Customer Churn Classifier

This project predicts customer churn using a **Random Forest Classifier** based on customer data. The model predicts whether a customer will leave a service (churn) or stay, based on several features.

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

## Usage

1. Install the required libraries:
   ```bash
   pip install pandas numpy seaborn matplotlib scikit-learn
