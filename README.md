Credit Card Fraud Detection
Overview

This project uses machine learning to detect fraudulent credit card transactions. It includes data exploration, preprocessing, visualization, model training and evaluation.

 Dataset
The project uses the Credit Card Fraud Detection Dataset, which contains anonymized credit card transactions.

The dataset includes:
* Time: Time elapsed between transactions.
* V1–V28: Anonymized numerical features.
* Amount: Transaction amount.
* Class: Target variable:

  * 0 = Legitimate transaction
  * 1 = Fraudulent transaction

Dataset source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

 Project Steps
1. Data exploration and cleaning
2. Checking missing values and duplicates
3. Data visualization
4. Standardizing Time and Amount
5. Splitting data into training and testing sets
6. Training a Logistic Regression model
7. Evaluating the model

The dataset is divided into:
* 80% training data
* 20% testing data
Stratified splitting is used to preserve the proportion of fraudulent and legitimate transactions.

 Evaluation Metrics
* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* Precision–Recall curve

 Technologies Used
* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

Conclusion
This project demonstrates a basic machine-learning workflow for credit card fraud detection, including data exploration, preprocessing, model training, evaluation, and prediction. The Precision Recall curve and classification metrics provide a more useful assessment than accuracy alone because the dataset is highly imbalanced.
