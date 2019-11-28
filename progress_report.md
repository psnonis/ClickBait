# W261 - Final Project - Clickthrough Rate Prediction

Team 14
Brian Musisi, Pri Nonis, Vinicio Del Sola, Laura Chutny
Fall 2019, Section 3

## Progress Report Nov 27, 2019

## Laura
- To date: Project planning, started final write up, numerical feature analysis and standardization, literature review, group coordination
- To come: toy model for chosen algorithm(s), with documentation and write up, work on categorical variable feature engineering. Testing parallel algorithm and continued project communication and coordination. Final report preparation.

## Vinicio
- To date: Started the Toy homegrown model - SVM with L2, with all the math explanation. Started the gradient descent code for the toy model using an RDD implementation. Review literature in cathegorical variables (bin, hashed features, hot-encode). Participated in the discussions of EDA preparatations.
- To come: Finish the SVM model and add L1 regularization and, if time permits, the kernel trick for capturing non-linearities. Help to implement Baseline Production model (Logistic Regression) and Vector model as comparison.

## Pri
- To date: Infrastructure setup for collaborative EDA on IBM Cloud. Categorical feature analysis and encoding. Exploration of Spark ML feature extractors and transformers for One-hot categorical encoding and some preliminary EDA. Review Field-aware Factorization Machines. Exploration of LogisticalRegression with Spark ML pipelines
- To come: Complete feature engineering and dimentionality reduction for the Categorical variables. Rate and select the most significant features using statistical signifiance tets such as ChiSquare. Save finalized features as parquet file and help implement baseline logistic regression on 10% of training data.

## Brian
- To date: EDA to gain an understanding o the data, including analyze features for null values and correlation with the outcome and examining the cardinality of the categorical features. Exploring ways to reduce the cardinality of the categorical features. Exploring the effects of feature engineering techniques on the data including normalization, hashing, imputing etc
- To come: Choose a method to use to handle the high cardinality of some of the categorical features. Narrow down the feature engineering techniques to use and apply them to the data. Implement a Random Forest model
