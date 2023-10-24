# credit-risk-classification
Module 20 Challenge for the Tec de Monterrey Data Analysis Bootcamp, Introduction to Supervised Learning


Jupyter Notebook file that takes data from a CSV and performs supervised machine learning processes on it, specifically classification, in order to determine the health of loans, as well as the risk in relation to other financial variables.

credit_risk_classification.ipynb is the main file, it uses data related to loans in order to build Machine Learning models that can classify the loan status and help with the decision of future loans. The basic processes for the building of these models are the creation of labels and features from the data, preprocessing and splitting into training and testings sets, creating the Logistic Regression Models and adjusting for the imbalance of the data in training.

This file exists in the main directory, which also contains the README, as well as the Resources Directory, which has the lending_data.csv file that is used as an starting point. A detailed report for our findings can also be found in the README file.

Contributions:
- Data Analysis Bootcamp Classes
- https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
- https://www.datacamp.com/tutorial/understanding-logistic-regression-python




## Credit Risk Analysis Report:

This analysis serves the purpose of using supervised machine learning models to predict the risk of loaning according to a variety of factors, such as:
- Loan Size
- Interest Rate
- Borrower Income
- Debt to Income Ratio
- Number of Accounts
- Derogatory Marks
- Total Debt
- Loan Status


The objective was to create models for predicting the health of a loan status, or the risk that it had, and using these models for the prediction of future loan applicants' risk

This analysis was made up of the following general steps:
- Creating labels and features from our original data
- Splitting the data into training and testing datasets by using train_test_split.
- Creating a Logistic Regression Model with the Original Data
- Evaluating the model's performance
- Oversampling the data to correct the imbalance and re-evaluate

It's important to consider the balancing step, as the original dataset had around 75,000 samples with a healthy score, while only about 2,500 with a high-risk score, thus, a balancing through oversampling is done in order to correctly train the model.

## Results

0: Healthy Loans
1: High-Risk Loans

* Machine Learning Model 1:
  * Description of Model 1 Accuracy, Precision, and Recall scores:

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.85      0.91      0.88       619

    accuracy                           0.99     19384
   macro avg       0.92      0.95      0.94     19384
weighted avg       0.99      0.99      0.99     19384



* Machine Learning Model 2:
  * Description of Model 2 Accuracy, Precision, and Recall scores.

              precision    recall  f1-score   support

           0       1.00      0.99      1.00     18765
           1       0.84      0.99      0.91       619

    accuracy                           0.99     19384
   macro avg       0.92      0.99      0.95     19384
weighted avg       0.99      0.99      0.99     19384


### Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

Analyzing the results for both models we can conclude that the 2 of them have very good accuracy for predicting our target labels, but considering the slight increase in recall and f-1 score with the over-sampling method, as well as the ease of implementation, we can decide that using this method is best, specially considering how it addresses a flaw a very real and common flaw in these types of real-world problem, where data is not balanced.

And even considering the slight decrease in precision for the High-Risk Loan target with the second model, we can also conclude that since our primary target should be the Health of the loans, this small tradeoff of precision in our secondary target with an increase in the other measures is acceptable.