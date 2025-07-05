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

## Next Steps (Future Work)

* ~~**Further Data Preprocessing:** Handle categorical features (e.g. One-Hot Encoding), scaling numerical features.~~
* ~~**Feature Selection/Engineering:** Explore creating new features or selecting the most important ones.~~
* **Model Building:** Experiment with various classification algorithms (e.g. Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).
* **Model Evaluation:** Use appropriate metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) and techniques (Confusion Matrix).
* **Hyperparameter Tuning:** Optimise model performance.
* **Deployment:** Consider deploying a simple predictive interface (e.g. using Streamlit).