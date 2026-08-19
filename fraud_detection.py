import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("creditcard.csv")
#Exploring the Data Set
print(df.head())
columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']

df = pd.read_csv("creditcard.csv", header=None, names=columns)

print(df.columns)
print(df.shape)
#Cleaning the Data Set
#1.Checking Missing Values
#2.Checking Duplicate Rows
print(df.isnull().sum())
print(df.duplicated().sum())
df.drop_duplicates(inplace=True)
print(df.duplicated().sum())
import matplotlib.pyplot as plt
import seaborn as sns
#Class Distribution using a countplot
sns.countplot(data=df, x='Class', color='purple')
plt.title('Class Distribution')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()
#Distribution of Transaction Amount using a histplot
plt.figure(figsize=(8,5))

sns.histplot(df["Amount"], bins=50 , color ='deeppink')

plt.title("Distribution of Transaction Amount")

plt.xlabel("Amount")

plt.show()
#The relationships among variables using a Correlation Heatmap
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), cmap='Blues')
plt.title("Correlation Heatmap")
plt.show()
# Checking for outliers using a Boxplot
sns.boxplot(data=df, x="Class", y="Amount")
# checking for data types
print(df.select_dtypes(include=np.number))
print(df.head())
# Standardize Amount and Time
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["Time", "Amount"]] = scaler.fit_transform(df[["Time", "Amount"]])
#Checking the Standardization
print(df.head())
# Splitting features and target
X = df.drop("Class", axis=1)
y = df["Class"]
# Splitting data into 20% for testing and 80% for training
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
# Creating the model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
# Training the model
model.fit(X_train, y_train)
# predictions : make the ppredictions
y_pred = model.predict(X_test)
print (y_pred)
# Accuracy : The percentage of correct predictions
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print(accuracy)
# Precision: Of all transactions predicted as fraud, how many were actually fraud?
from sklearn.metrics import precision_score

precision = precision_score(y_test, y_pred)

print(precision)
     # recall: Of all actual fraud transactions, how many did the model successfully detect?
from sklearn.metrics import recall_score

recall = recall_score(y_test, y_pred)

print(recall)

# F1 Score:It balances precision and recall into a single metric.
from sklearn.metrics import f1_score

f1_score = f1_score(y_test, y_pred)

print(f1_score)
# Classification Report : It provides Precision, Recall, F1-score & Support (number of samples in each class)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()
# Precision-Recall Curve: It hows the trade-off between detecting more fraud cases and avoiding false alarms
from sklearn.metrics import PrecisionRecallDisplay

PrecisionRecallDisplay.from_estimator(
    model,
    X_test,
    y_test
)
plt.title("Precision-Recall Curve")
plt.show()
     # New Prediction
sample = X.iloc[[10]]

prediction = model.predict(sample)

print(prediction)
print("Prediction:", prediction[0])
print("Actual:", y.iloc[10])

     
