# Customer Churn Prediction Project

This repository contains a project focused on predicting customer churn using a publicly available dataset.
The goal is to identify customers who are likely to discontinue their service, allowing businesses to proactively implement retention strategies.

# Project Goal

The primary objective is to build a classification model that can accurately predict customer churn based on various features such as customer demographics, services subscribed, and billing information.

# Dataset

The dataset used for this project is the **Telco Customer Churn dataset** that is available on Kaggle. It contains information about a telecommunications company's customers, including whether they churned or not.

* **Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# Project Structure

`churn_prediction.ipynb`: Jupyter Notebook containing the data loading, exploratory data analysis (EDA), data preprocessing, model building, and evaluation steps.
`WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset file (downloaded from Kaggle and placed here).
`requirements.txt`: Lists all the Python libraries and their versions required to run the project.

# How to Run the Project

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/lorettayong/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. **Download the Dataset:**
   * Go to the [Kaggle Telco Customer Churn dataset page](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).
   * Click on 'Download' to get the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file.
   * Place this CSV file directly into the `customer-churn-prediction` directory.

3. **Create and Activate a Virtual Environment**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

4. **Install Dependencies:**
    (After running the initial Python code in the notebook, generate this using `pip freeze > requirements.txt`)
    ```bash
    pip install -r requirements.txt
    ```

5. **Launch JupyterLab:**
    ```bash
    jupyter lab
    ```
6. **Open and Run the Notebook:**
   * In the JupyterLab interface, open `churn-prediction.ipynb`.
   * Run the cells sequentially to see the data loading, exploration, and subsequent steps.

## Initial Data Overview (from `churn_prediction.ipynb`)

* **Shape:** `(7043, 21)` - 7043 customers, 21 features
* **Target Variable:** 'Churn' (Yes/No)
* **Initial Churn Rate:** Approximately 26.5% of customers churned.
* **Data Types:** Mix of numerical and categorical features. Noted a specific issue with `TotalCharges` being an object type despite containing numeric values, requiring conversion and handling of missing entries.

## Project Phases

### Phase 1: Initial Data Exploration and Cleaning
* **Objective:** Load the dataset, understand its structure, identify data types, check for missing values, perform initial cleaning.
* **Key Activities:**
  * Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
  * Inspected data using `df.head()`, `df.info()`, `df.describe()` and `df.isnull().sum()`.
  * Handled the `TotalCharges` column by converting it to numeric and filling missing values (which were spaces for new customers) with 0.
  * Dropped the `customerID` column as it is not relevant for prediction.
  * Mapped the `Churn` target variable from 'Yes/No' to 1/0.
  * Identified categorical and numerical features.
  * Performed basic EDA with visualisations (churn distribution, numerical feature distributions, churn vs. categorical features)

### Phase 2: Data Preprocessing and Feature Engineering
* **Objective:** Transform raw data into a suitable format for machine learning models, including handling categorical variables and scaling numerical ones, and splitting the data for training and testing.
* **Key Activities:**
* Separated features (X) and target (y) variables.
* Identified categorical and numerical features within X.
* Created a preprocessing pipeline using `ColumnTransformer`:
  * `StandardScaler` applies to numerical features for normalisation.
  * `OneHotEncoder` applies to categorical features to convert them into numerical (binary) format.
* Split the data into training (80%) and testing (20%) sets using `train_test_split`, ensuring stratification on the `Churn` variable to maintain class balance.
* Applied the `fit_transform` method on the training data and `transform` on the testing data to prevent data leakage.
* Converted processed NumPy arrays back to Pandas DataFrames for easier inspection.

### Phase 3: Model Building and Evaluation
* **Objective:** Train and evaluate multiple classification models to predict customer churn, and compare their initial performance.
* **Key Activities:**
* Model Selection: Implemented and evaluated three distinct classification algorithms:
  * Logistic Regression: A linear model serving as a strong baseline.
  * Decision Tree Classifier: A non-linear model capable of capturing complex relationships.
  * Random Forest Classifier: An ensemble method that combines multiple decision trees for improved accuracy and robustness.
* Model Training: Each model was trained on the preprocessed training data (`X_train_processed_df`, `y_train`).
* Prediction: Predictions were generated on the unseen test data (`X_train_processed_df`) for each trained model.
* Performance Evaluation: Each model's performance was rigorously assessed using a suite of classification metrics:
  * Accuracy: The overall proportion of correct predictions.
  * Precision: The model's ability to correctly identify positive predictions (minimise false positives).
  * Recall: The model's ability to find all actual positive instances (minimising false negatives).
  * F1-Score: The harmonic mean of precision and recall, providing a balanced measure, especially useful in cases of class imbalance.
* Confusion Matrix Visualisation: A Confusion Matrix was generated and visualised for each model, providing a detailed breakdown of True Positives, True Negatives, False Positives, and False Negatives.
* Model Comparison: A summary table and bar charts were created to compare the Accuracy, Precision, Recall, and F1-Score of all three models side-by-side, offering insights into their relative strengths and weaknesses for this churn prediction task.
* **Model Performance Comparison**
Our primary business objective is to reduce customer attrition, which fundamentally means minimising the rate at which customers discontinue their service. In this context, Recall would be the metric of high importance, since it directly measures our ability to "catch" as many actual churners as possible. Missing a customer who is about to churn (a False Negative) typically incurs a higher financial cost, in the form of lost customer lifetime value, than the cost of a potentially wasted retention effort on a customer who would not have churned (a False Positive). Therefore, while aiming for high recall, we also need to maintain a reasonable Precision to ensure our retention strategies are efficient and not overly wasteful. The F1-Score serves as an excellent composite metric, providing a balanced assessment of both Recall and Precision.

Based on the comprehensive evaluation metrics, the Logistic Regression model demonstrates the most suitable performance for predicting customer churn in this dataset, offering a practical balance for business intervention.

While the Random Forest model showed a competitive Accuracy of 78.64%, the Logistic Regression model achieved a superior F1-Score of 60.40%, which is notably higher than Random Forest's 54.74%. This higher F1-score is crucial for churn prediction as it effectively balances the need to identify actual churners with the goal of minimising unnecessary retention efforts, especially given the inherent class imbalance in churn datasets.

Diving into the specific performance of Logistic Regression:
- It achieved a Recall of 55.88%, indicating that it successfully identified more than half of the customers who actually churned. This is vital for proactively engaging a significant portion.
- Its Precision of 65.72% suggests that when the model predicts a customer will churn, it is correct almost two-thirds of the time, thereby minimising wasted resources on customers who were not genuinely at-risk.

The Confusion Matrix for Logistic Regression further illustrates this balance:
- True Positives (TP): 209 - correctly identified churners.
- False Positives (FP): 109 - non-churners incorrectly flagged as churners (wasted effort).
- False Negatives (FN): 165 - actual churners incorrectly missed by the model (lost opportunity).
- True Negatives (TN): 926 - correctly identified non-churners.

This performance profile makes Logistic Regression a practical and actionable choice for the business, enabling targeted retention strategies that effectively balance the costs of intervention with the imperative of reducing customer attrition.

## Next Steps (Future Work)

* ~~**Further Data Preprocessing:** Handle categorical features (e.g. One-Hot Encoding), scaling numerical features.~~
* ~~**Feature Selection/Engineering:** Explore creating new features or selecting the most important ones.~~
* ~~**Model Building:** Experiment with various classification algorithms (e.g. Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).~~
* ~~**Model Evaluation:** Use appropriate metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) and techniques (Confusion Matrix).~~
* **Hyperparameter Tuning and Cross Validation:** Optimise model parameters using techniques such as GridSearchCV or RandomizedSearchCV, and employ cross validation for more robust performance estimates.
* **ROC Curve and AUC Analysis:** Conduct a detailed analysis of Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) to assess model discrimination across various thresholds, which is particularly valuable for imbalanced datasets.
* **Addressing Class Imbalance:** If needed, explore advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data and potentially improve the model's ability to identify the minority churn class.
* **Feature Importance Analysis:** Investigate which features are most influential in the models' predictions to gain deeper business insights.
* **Deployment:** Develop a simple web application (e.g. using Streamlit or Flask) to allow interactive churn predictions, demonstrating the end-to-end project lifecycle.