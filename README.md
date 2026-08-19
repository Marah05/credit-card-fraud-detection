Credit Card Fraud Detection
Project Overview

This project uses machine learning to detect fraudulent credit card transactions. It explores and preprocesses transaction data, trains a Logistic Regression model, and evaluates its ability to distinguish between legitimate and fraudulent transactions.

 Dataset

The project uses the **Credit Card Fraud Detection Dataset**, which contains anonymized credit card transactions.

The dataset includes:

* Time: Time elapsed between transactions.
* V1–V28: Anonymized numerical features.
* Amount: Transaction amount.
* Class: Target variable:

  * 0 = Legitimate transaction
  * 1 = Fraudulent transaction

Dataset source: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

 Project Steps

1. Importing the required Python libraries.
2. Loading and exploring the dataset.
3. Checking for missing values and duplicate records.
4. Removing duplicate records.
5. Visualizing the class distribution.
6. Examining transaction amounts and correlations.
7. Checking for outliers using a box plot.
8. Standardizing the `Time` and `Amount` features.
9. Splitting the data into training and testing sets.
10. Training a Logistic Regression model.
11. Evaluating the model.
12. Making a sample transaction prediction.

 Data Visualizations

The exploratory data analysis includes:

* Class distribution count plot
* Transaction amount histogram
* Correlation heatmap
* Transaction amount box plot
* Confusion matrix
* Precision–Recall curve

 Machine Learning Model

The project uses **Logistic Regression**, a supervised machine-learning classification algorithm, to classify transactions as legitimate or fraudulent.

The dataset is divided into:

* 80% training data
* 20% testing data

Stratified splitting is used to preserve the proportion of fraudulent and legitimate transactions.

 Evaluation Metrics

Because fraudulent transactions are much less common than legitimate transactions, accuracy alone is not sufficient. The model is evaluated using:

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

 How to Run the Project

1. Download the project files.
2. Download `creditcard.csv` from the linked Kaggle dataset.
3. Place the dataset in the same folder as the Python file.
4. Install the required libraries:
pip install pandas numpy matplotlib seaborn scikit-learn
5. Run the Python file:
python fraud_detection.py

Conclusion

This project demonstrates a basic machine-learning workflow for credit card fraud detection, including data exploration, preprocessing, model training, evaluation, and prediction. The Precision–Recall curve and classification metrics provide a more useful assessment than accuracy alone because the dataset is highly imbalanced.

 Future Improvements

Future improvements may include:

* Comparing Logistic Regression with other classification models.
* Applying techniques for handling class imbalance.
* Performing hyperparameter tuning.
* Evaluating the model using additional metrics.
* Deploying the model through a simple web application.
