# Customer Churn Prediction Project

This repository contains a project focused on predicting customer churn using a publicly available dataset.
The goal is to identify customers who are likely to discontinue their service, allowing businesses to proactively implement retention strategies.

# Project Goal

The primary objective is to build a classification model that can accurately predict customer churn based on various features such as customer demographics, services subscribed, and billing information.

# Dataset

The dataset used for this project is the **Telco Customer Churn dataset** that is available on Kaggle. It contains information about a telecommunications company's customers, including whether they churned or not.

* **Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

# Project Structure

- `churn_prediction.ipynb`: Jupyter Notebook containing the data loading, exploratory data analysis (EDA), data preprocessing, model building, and evaluation steps.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: The dataset file (downloaded from Kaggle and placed here).
- `requirements.txt`: Lists all the Python libraries and their versions required to run the project.
- `app.py`: The Streamlit web application for interactive predictions.
- `model_pipeline.joblib`: The saved complete scikit-learn pipeline, encapsulating the preprocessor (`ColumnTransformer` with `StandardScaler` and `OneHotEncoder`) and the trained Tuned Decision Tree Classifier (after SMOTE).

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

7. **Run the Streamlit Application (Optional):**
   * Open the Command Line Interface (CLI) with the virtual environment activated.
   * Navigate to the project's root directory.
   * Run `streamlit run app.py` to open the interactive churn prediction web application in the browser.

# Initial Data Overview (from `churn_prediction.ipynb`)

* **Shape:** `(7043, 21)` - 7043 customers, 21 features
* **Target Variable:** 'Churn' (Yes/No)
* **Initial Churn Rate:** Approximately 26.5% of customers churned.
* **Data Types:** Mix of numerical and categorical features. Noted a specific issue with `TotalCharges` being an object type despite containing numeric values, requiring conversion and handling of missing entries.

# Project Phases

## Phase 1: Initial Data Exploration and Cleaning
* **Objective:** Load the dataset, understand its structure, identify data types, check for missing values, perform initial cleaning.
* **Key Activities:**
* Loaded `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
* Inspected data using `df.head()`, `df.info()`, `df.describe()` and `df.isnull().sum()`.
* Handled the `TotalCharges` column by converting it to numeric and filling missing values (which were spaces for new customers) with 0.
* Dropped the `customerID` column as it is not relevant for prediction.
* Mapped the `Churn` target variable from 'Yes/No' to 1/0.
* Identified categorical and numerical features.
* Performed basic EDA with visualisations (churn distribution, numerical feature distributions, churn vs. categorical features)

## Phase 2: Data Preprocessing and Feature Engineering
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

## Phase 3: Model Building and Evaluation
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

### Model Performance Comparison

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

## Phase 4: Hyperparameter Tuning and Cross-Validation
* **Objective:** Improve the performance of the selected models by systematically searching for optimal hyperparameters and obtaining more robust performance estimates through cross-validation.
* **Key Activities:**
* Defined a Stratified K-Fold Cross-Validation strategy (`n_split=5`) to ensure balanced class distribution across all folds, which is critical for imbalanced datasets such as churn.
* Performed a Grid Search Cross-Validation (`GridSearchCV`) for each model (i.e. Logistic Regression, Decision Tree, and Random Forest) to find the best combination of hyperparameters. The optimisation metric used was F1-Score, which aligns with our business objective of balancing precision and recall.
* Logistic Regression Tuning:
  * Explored `C` (inverse of regularisation strength) and `solver` parameters.
  * Best parameters found: {'C': 100, 'solver': 'liblinear'}
  * Best cross-validation F1-Score: 59.74%
  * Test set performance:
    * Accuracy: 80.13%
    * Precision: 64.78%
    * Recall: 55.08%
    * F1-Score: 59.54%
* Decision Tree Tuning:
  * Explored `max_depth`, `min_samples_split`, `min_samples_leaf`, and `criterion` parameters.
  * Best parameters found: {'criterion': 'entropy', 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
  * Best cross-validation F1-Score: 57.03%
  * Test set performance:
    * Accuracy: 79.70%
    * Precision: 63.10%
    * Recall: 56.68%
    * F1-Score: 59.72%
* Random Forest Tuning:
  * Explored `n_estimators`, 'max_depth`, `min_samples_split`, and `min_samples_leaf` parameters.
  * Best parameters found: {'max_depth': 10, 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 100}
  * Best cross-validation F1-Score: 58.57%
  * Test set performance:
    * Accuracy: 80.48%
    * Precision: 67.37%
    * Recall: 51.34%
    * F1-Score: 58.27%
* Impact of tuning: While hyperparameter tuning generally aims for significant performance gains, the improvements across all three models were relatively modest in this case. This suggests that the initial default parameters were already quite robust, or that the inherent predictability within the dataset, given the current features and preprocessing, might be approaching its limit for these specific algorithms. The most notable change was that the Tuned Decision Tree slightly surpassed the Tuned Logistic Regression in F1-Score, indicating a more optimal configuration for balancing precision and recall for this model.

### Model Performance Comparison (After Tuning)

After comprehensive hyperparameter tuning and cross-validation, we re-evaluated the models to identify the most effective one for our churn prediction objective, maintaining our focus on the F1-Score as the primary metric for balancing Recall and Precision.

Comparing the tuned models, the Tuned Decision Tree emerged as the top performer, achieving an F1-Score of 59.72% on the test set. This represents a slight improvement over its untuned performance (48.36%) and narrowly outperforms the Tuned Logistic Regression (59.54%) and Tuned Random Forest (58.27%).

Deliving into the specific performance of the Tuned Decision Tree:
- It achieved a Recall of 56.68%, indicating a strong ability to identify a significant portion of actual churners.
- Its Precision of 63.10% suggests that when the model predicts churn, it is correct a solid majority of the time, helping to manage retention campaign costs.

While the improvements from tuning were not dramatic, the Tuned Decision Tree now offers the best balance of identifying at-risk customers (Recall) while maintaining acceptable efficiency in targeting (Precision), making it the most reliable choice for proactive customer retention strategies based on this analysis.

## Phase 5: ROC Curve and AUC Analysis
* **Objective:** Gain a deeper understanding of the models' discriminatory power across all possible classification thresholds, especially given the potential class imbalance in churn data.
* **Key Activities:**
* Determined the predicted probabilities of the positive class (churn=1) for each tuned model (i.e. Logistic Regression, Decision Tree, and Random Forest).
* Generated the ROC Curve for each model to plot the True Positive Rate (TPR) against the False Positive Rate (FPR) for various thresholds and calculated AUC Score for each model to compute the Area Under the Curve (AUC).
* AUC scores of each model:
  * Logistic Regression: 84.04%
  * Decision Tree: 83.12%
  * Random Forest: 83.78%
* The Tuned Logistic Regression model performed best in terms of AUC, which implies that it has a very strong ability to distinguish between churners and non-churners across different thresholds.
* Alignment with F1-Scores and Implications: While the Tuned Decision Tree showed the highest F1-Score of 59.72%, the Tuned Logistic Regression achieved the highest AUC of 84.04%. This difference highlights the distinct insights provided by these metrics:
  - F1-Score is a threshold-dependent metric that indicates performance at a specific operating point, which is typically the default 0.5 probability threshold. The Decision Tree's slightly higher F1-Score suggests it offers the best balance of precision and recall at its default operating point.
  - On the other hand, AUC is a threshold-independent measure of the model's overall discriminatory power or its ability to rank positive instances higher than negative instances across all possible thresholds. The higher AUC for Logistic Regression indicates that it is generally better at separating the churn and non-churn classes. This means that even if its F1-Score at the default threshold was slightly lower, there might be other thresholds where Logistic Regression could achieve an even better balance of TPR and FPR, or where its overall ranking capability is superior. This further emphasises that no single metric tells the complete story, and understanding both F1-Score (for actionable performance at a specific point) and AUC (for overall discriminatory power) is crucial for robust model evaluation.

## Phase 6: Addressing Class Imbalance
* **Objective:** Improve the model's ability to correctly identify the minority class (churners) by handling class imbalance in the training data.
* **Key Activities:**
* Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to create synthetic samples of the minority class (churners), thereby balancing the class distribution.
* Retrained the best-performing model from Phase 4 (the Tuned Decision Tree Classifier) on this SMOTE-resampled training data.
* Evaluated the retrained model's performance on the original, untouched test set to ensure an unbiased assessment.
* Test Set Performance (Tuned Decision Tree after SMOTE):
  * Accuracy: 72.96%
  * Precision: 49.38%
  * Recall: 74.87%
  * F1-Score: 59.51%
  * AUC Score: 81.03%
* Impact of Applying SMOTE: Applying SMOTE resulted in a significant increase in Recall for the churn class, drastically improving from 56.68% (for the Tuned Decision Tree before SMOTE) to 74.87%. This indicates the model is now much better at identifying actual churners, which aligns strongly with our business objective of minimising lost customers. However, this improvement came with a notable trade-off in Precision, which decreased from 63.10% to 49.38%. This means that the model now produces more false positives of predicting churn for customers who don't actually churn. The F1-Score, a balanced metric, saw a slight decrease from 59.72% to 59.51%. Despite the minor dip in F1-Score and the reduction in Precision, the substantial gain in Recall is considered acceptable given that the primary goal is to proactively identify and intervene with as many potential churners as possible, even if it means a higher rate of "false alarms" for retention efforts since the cost of losing a customer is typically higher than the cost of a potentially unnecessary retention offer.

### Model Performance Conclusion (Final Assessment)

After comprehensive hyperparameter tuning, cross-validation, and an attempt to address class imbalance, we conducted a final assessment to identify the most effective model for our churn prediction objective. Our primary focus remains on balancing Recall (to catch as many churners as possible) and Precision (to minimise wasted retention efforts), with the F1-Score serving as our key composite metric.

Let's compare the performance of the best-tuned models and the SMOTE-enhanced model:

| Model                         | Accuracy | Precision | Recall | F1-Score | AUC    |
|:------------------------------|:---------|:----------|:-------|:---------|:-------|
| Logistic Regression (Tuned)   | 80.13%   | 64.78%    | 55.08% | 59.54%   | 84.04% |
| Decision Tree (Tuned)         | 79.70%   | 63.10%    | 56.68% | 59.72%   | 83.12% |
| Random Forest (Tuned)         | 80.48%   | 67.37%    | 51.34% | 58.27%   | 83.78% |
| Decision Tree (Tuned + SMOTE) | 72.96%   | 49.38%    | 74.87% | 59.51%   | 81.03% |

The application of SMOTE to the Tuned Decision Tree model yielded a significant increase in Recall (from 56.68% to 74.87%), making it exceptionally better at identifying actual churners. While this came at the cost of a reduced Precision (from 63.10% to 49.38%) and a slight dip in F1-Score (from 59.72% to 59.51%), this trade-off is acceptable and even desirable given our business objective. Prioritising the identification of churners (high Recall) is crucial to minimise the high cost of lost customers, even if it means a higher volume of retention efforts (more false positives).

Therefore, for this churn prediction task, the Tuned Decision Tree Classifier with SMOTE is considered the most effective model. Its strong Recall ensures that a large proportion of at-risk customers are identified, allowing for proactive intervention, which aligns directly with the goal of reducing customer attrition. Further optimisation could involve exploring different thresholds for this model to find an even more precise balance between Precision and Recall, depending on the specific budget and impact of retention campaigns.

## Phase 7: Feature Importance Analysis
* **Objective:** Understand which features (customer attributes) are most influential in the best model's predictions so as to provide valuable business insights and help in identifyin the key drivers of churn.
* **Key Activities:**
* Extracted feature importances from the retrained Tuned Decision Tree (after SMOTE) using the `feature_importances_` attribute.
* Mapped the processed feature names back to the original, understandable labels.
* Visualised the top 10 most important features using a bar chart.
* **Top 10 Most Important Features Identified:**
  1. `Contract_Month-to-month` - Importance: 0.566889
  2. `OnlineSecurity_No` - Importance: 0.106041
  3. `PaymentMethod_Electronic check` - Importance: 0.098924
  4. `StreamingMovies_Yes` - Importance: 0.052225
  5. `Contract_One year` - Importance: 0.039045
  6. `tenure` - Importance: 0.038502
  7. `OnlineSecurity_Yes` - Importance: 0.020704
  8. `InternetService_DSL` - Importance: 0.018743
  9. `TechSupport_No` - Importance: 0.013819
  10. `MonthlyCharges` - Importance: 0.012164

### Implications for Churn and Actionable Insights

The feature importance analysis has provided critical insights into the underlying reasons for customer churn, which would directly inform business strategies:

* Dominant Driver: Contract Type (`Contract_Month-to-month`): This feature is overwhelmingly the most significant predictor of churn. Customers on month-to-month contracts exhibit minimal commitment and are highly susceptible to churn due to perceived lack of value, minor dissatisfaction, or competitive offers.
   * Actionable insight: The top priority for retention should be to convert these customers to longer-term contracts (e.g. one- or two-years) through targeted promotions, bundled services, or loyalty programmes that highlight long-term value and benefits.
    
* Security and Payment Vulnerabilities (`OnlineSecurity_No`, `PaymentMethod_Electronic check`): Customers lacking online security services and those using electronic checks are significantly more prone to churn. This suggests either a feeling of vulnerability / lack of stickiness or potential friction points in their billing experience.
   * Actionable insight: Actively promote the value and benefits of online security features by perhaps offering free trials or bundling them together. For electronic check users, investigate underlying causes of frustrations and encourage migration to more stable and convenient payment methods, such as auto-payment via credit card, with appropriate incentives.

* Service Engagement and Support Gaps (`StreamingMovies_Yes`, `InternetService_DSL`, `TechSupport_No`): While seemingly engaged, customers who stream movies are also at higher risk, possibly due to high expectations of service quality or competitive offerings. Conversely, those without tech support are more likely to leave, which indicates a need for reliable assistance. DSL internet service, potentially perceived as slower, also contributes to churn risk.
   * Actionable insight: For streaming users, ensure robust internet performance and consider offering competitive streaming bundles. For customers without tech support, emphasise the availability and effectiveness of support channels for prompt addressing of issues. Evaluate DSL service quality and consider upgrade incentives.

* Customer Lifecycle and Value Perception (`Contract_One year`, `tenure`, `OnlineSecurity_Yes`, `MonthlyCharges`): Features such as one-year contracts and tenure indicate that customers at different stages of their lifecycle (e.g. nearing contract end, very new, or very old) have varying churn risks. `OnlineSecurity_Yes` acts as a sticky feature, while `MonthlyCharges` still plays a role, likely when perceived value does not match cost.
   * Actionable insight: Implement lifecycle-based retention strategies, such as proactive retention offers for one-year contracts and personalised engagement for long-term customers. Continue to highlight the value of "sticky" services such as online security. Regularly review pricing structures to ensure competitiveness and perceived value.

These insights provide a clear roadmap for the business to develop targeted strategies and improve customer retention.

## Phase 8: Model Deployment
* **Objective:** Develop a simple web application (using Streamlit) to allow interactive churn predictions, demonstrating the end-to-end project lifecycle.
* **Key Activities:**
* Saved the completed sci-kit learn pipeline (`model_pipeline.job.lib`), which includes the fitted preprocessor (`ColumnTransformer`) and the trained Tuned Decision Tree Classifier (after SMOTE).
* Developed `app.py`, a Streamlit application that loads this single `model_pipeline.joblib` file, that provides a user-friendly interface for inputting customer features and receiving real-time churn predictions.

# Next Steps (Future Work)

* ~~**Further Data Preprocessing:** Handle categorical features (e.g. One-Hot Encoding), scaling numerical features.~~
* ~~**Feature Selection/Engineering:** Explore creating new features or selecting the most important ones.~~
* ~~**Model Building:** Experiment with various classification algorithms (e.g. Logistic Regression, Decision Trees, Random Forests, Gradient Boosting).~~
* ~~**Model Evaluation:** Use appropriate metrics (Accuracy, Precision, Recall, F1-score, ROC-AUC) and techniques (Confusion Matrix).~~
* ~~**Hyperparameter Tuning and Cross-Validation:** Optimise model parameters using techniques such as GridSearchCV or RandomizedSearchCV, and employ cross-validation for more robust performance estimates.~~
* ~~**ROC Curve and AUC Analysis:** Conduct a detailed analysis of Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) to assess model discrimination across various thresholds, which is particularly valuable for imbalanced datasets.~~
* ~~**Addressing Class Imbalance:** If needed, explore advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) to balance the training data and potentially improve the model's ability to identify the minority churn class.~~
* ~~**Feature Importance Analysis:** Investigate which features are most influential in the models' predictions to gain deeper business insights.~~
* ~~**Deployment:** Develop a simple web application (e.g. using Streamlit or Flask) to allow interactive churn predictions, demonstrating the end-to-end project lifecycle.~~