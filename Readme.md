## EDA insights 
## 	•	Dataset shows class imbalance (~22% default)
##	•	Credit limit and repayment behavior strongly influence default
##	•	Payment delay features are the most predictive
##	•	Demographic features have secondary impact
##	•	Findings guided feature engineering and model selection
## MODEL TRAINING
## LOGISTIC REGRESSION: Finds a linear boundary default vs Non-default and outputs a probability
## DECISION TREES: Splits data using if-else rules, captures non-linear patterns
## RANDOM FORESTS(BAGGING): Builds many decision trees on random samples and average their predictions, reduces overfitting, handles non-linear interactions
## GRADIENT BOOSTING(BOOSTING): Builds Trees sequentially where each new tree focuses on previous mistakes, learn complex patterns
## NAIVE BAYES(PROBABILITIC): Uses bayes theorem and assumes feature independence 
## K-NEAREST NEIGHBOURS(DISTANCE-BASSED): A point's class depends on nearest neighbours, sensitive to scaling, slow for large datasets, usefull in small datasets
## SUPPORT VECTOR MACHINE(SVM)(MARGIN-BASSED): Finds a maximum margin boundary between classes, effective in high dimensional, slower on larger dataset, good on clear seperations
## Setup.py has a function which reads file requirements.txt and install all packages inside it 
## requirements.txt contains all packages needed for project
