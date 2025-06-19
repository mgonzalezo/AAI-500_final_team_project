# AAI-500_final_team_project
University of San Diego AAI-500 Final team project - Stroke Prediction Dataset

# AAI-500_final_team_project
University of San Diego AAI-500 Final team project - Stroke Prediction Dataset

## 📑 Table of Contents

- **🧠 [Stroke Prediction Dataset - Data Cleaning Pipeline](#-stroke-prediction-dataset---data-cleaning-pipeline)**
  - [1.1 Import Dependencies](#11-import-dependencies)
  - [1.2 Dataset Attributes](#12-dataset-attributes)
  - [1.3 Standardize Categorical Values](#13-standardize-categorical-values)
  - [1.4 Inspecting Missing Values](#14-inspecting-missing-values)
  - [1.5 Replacing the Missing Values](#15-replacing-the-missing-values)
  - [1.6 Categorizing the data in different types](#16-categorizing-the-data-in-different-types)
  - [1.7 Duplicate Check in Dataset](#17-duplicate-check-in-dataset)
  - [1.8 Outlier Detection](#18-outlier-detection)
  - [1.9 Treatment of Outliers](#19-treatment-of-outliers)

- **💾 [Explore Data](#-explore-data)**
  - [1. Sampling Distributions](#1-sampling-distributions)
  - [2. The Central Limit Theorem (CLT)](#2-the-central-limit-theorem-clt)
  - [3. Methods of Estimation](#3-methods-of-estimation)
  - [4. Confidence Intervals (CIs)](#4-confidence-intervals-cis)
  - [5. The Bootstrap](#5-the-bootstrap)
  - [6. Bayesian Approach (Conceptual Example)](#6-bayesian-approach-conceptual-example)

- **🔍📊⚙️ [Systematic Model Selection Process](#-systematic-model-selection-process)**
  - [Step 1: Understand and Preprocess the Data](#step-1-understand-and-preprocess-the-data)
  - [Step 2: Split the Dataset](#step-2-split-the-dataset)
  - [Step 3: Address Class Imbalance](#step-3-address-class-imbalance)
  - [Step 4: Train and Evaluate Multiple Models](#step-4-train-and-evaluate-multiple-models)
  - [Step 5: Compare Model Performance](#step-5-compare-model-performance)
  - [Step 6: Final Model Selection](#step-6-final-model-selection)
  - [Model Choice: Logistic Regression](#model-choice-logistic-regression)

- **📎 [Reference](#-reference)**
  - [Links](#links)


# 🧠 Stroke Prediction Dataset - Data Cleaning Pipeline

This README walks through the data cleaning steps performed on the [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset) from Kaggle. The goal is to prepare the dataset for machine learning tasks by addressing missing values, correcting datatypes, and handling outliers or inconsistencies.

## 📁 Dataset Overview

The dataset contains demographic and health-related information of patients, with the target column `stroke` indicating whether a patient had a stroke.

### Features
- `id`: Unique identifier
- `gender`: "Male", "Female", or "Other"
- `age`: Age of the patient
- `hypertension`: 0 if no hypertension, 1 if yes
- `heart_disease`: 0 if no heart disease, 1 if yes
- `ever_married`: "Yes" or "No"
- `work_type`: Type of work ("Private", "Self-employed", etc.)
- `Residence_type`: "Urban" or "Rural"
- `avg_glucose_level`: Average glucose level in blood
- `bmi`: Body mass index
- `smoking_status`: "formerly smoked", "never smoked", "smokes", or "Unknown"
- `stroke`: 0 or 1 (target variable)

---

# 🧼 Data Cleaning Steps

### 1.1 Import Dependencies

* **Summary:** Importing all the necessary programming libaries and modules for efficient data manipulation
* **Outcome:** Getting the dataset available for initial inspection and all subsequent cleaning, transformation and analysis

### 1.2 Dataset Attributes

* **Summary:** Checking out the different dataset variables
      * Inspecting all the columns and understanding the data types assigned to each feature (e.g., integer, float, object) and getting clarity of dataset
      * Identifying potential data issues
      * Preparing strategy for the subsequest data cleaning and preprocessing steps
  
### 1.3 Standardize Categorical Values

* **Summary:** Focuses on ensuring uniformity within the categorical features, which often has inconsistent data entry. For example, variations like "Male", "male", "MALE" or "M" or "NAN".  
      * Final goal is to have consistent value so that statistical analyses doesn't misinterpret the slight variation.
      * This helps in accurate counting, filtering, and encoding of categorical variables, significantly improving the integrity and reliability.
  
### 1.4 Inspecting Missing Values

* **Summary:** Involves identification and quantifying all the missing data points across entire dataset.
      * Finding what needs to be done with the missing, replace, deletion of the row entirely.
      * The detailed inspection is critical for making informed decision.
  
### 1.5 Replacing the Missing Values

* **Summary:** Once the missing value have been accurately identified we can apply different methods to fill those gaps.
      * Common methods used to replace missing values is generally to delete the rows.
      * As this will cause the final result to change if the rows are deleted.
      * Using mean, median, mode to replace the missing values and here we are using mean to replace the values are the mean and median are very similar.
  
### 1.6 Categorizing the data in different types

* **Summary:** Effective categorization is essential for data management and processing. The data is categorized into 3 types here
      * Categorical, Binary Numerical, Continous Numerical to identify data.
      * Once the data is separated into these categories we can move certain columns into different types from there original type.
      * Like gender, ever_married, from categorical to Binary Numerical.
  
### 1.7 Duplicate Check in Dataset

* **Summary:** This crucial step is dedicated to proactively identifying and effectively managing any redundant or identical rows that may exist within the dataset. Methods like df.duplicated().sum() are used to count them.

### 1.8 Outlier Detection

* **Summary:** Outliers can arise from various sources, and they can have potential impact
      * Finding correlation between high, low and weak pairs.
      * KDE and Histogram inteference
  
### 1.9 Treatment of Outliers

* **Summary:** Categorical labels are often more intuitive and user-friendly for human understanding than precise numerical ranges.
      * The specific strategy adopted here for handling outliers is binning, which was judiciously applied to the continuous numerical features: bmi, avg_glucose_level, and age. This transformation converts continuous data into discrete, ordered categories.
      * bmi was grouped into categories like 'Underweight', 'Ideal', 'Overweight', and 'Obesity'.
      * age was categorized into 'Children', 'Teens', 'Adults', 'Mid Adults', and 'Elderly'.
      * avg_glucose_level was binned into 'Low', 'Normal', 'High', and 'Very High'.


# 💾 Explore Data

# Exploratory Data Analysis (EDA) Steps for Stroke Prediction Dataset

This document summarizes the key statistical exploration steps performed on the stroke prediction dataset. Each section corresponds to a part of the analysis script.

## 1. Sampling Distributions

* **Summary:** This section explores the concept of a sampling distribution. Specifically, it focuses on the 'age' column from the dataset.
    * The 'age' column is treated as the population.
    * Multiple random samples are drawn from this 'population' of ages.
    * The mean of each sample is calculated.
    * A histogram of these sample means is plotted, representing the empirical sampling distribution of the mean age.
    * The mean of this sampling distribution is compared to the original population mean age.
    * The theoretical standard error (population std dev / sqrt(sample size)) is compared to the empirical standard error (std dev of the sample means).

## 2. The Central Limit Theorem (CLT)

* **Summary:** This section demonstrates the Central Limit Theorem using the sampling distribution of the mean age created in the previous step.
    * It highlights that even if the original distribution of 'age' in the dataset is not perfectly normal, the sampling distribution of its mean tends to be approximately normal, especially with a sufficient sample size.
    * A normal distribution curve is overlaid on the histogram of sample mean ages to visually assess this approximation.
    * The original distribution of 'age' is also plotted for comparison.

## 3. Methods of Estimation

* **Summary:** This section covers point estimation for key parameters using a random sample drawn from the dataset.
    * **Point Estimate for Mean Age:** The mean 'age' of the sample is calculated as a point estimate for the true population mean age.
    * **Point Estimate for Stroke Proportion:** The proportion of 'stroke' occurrences in the sample is calculated as a point estimate for the true population stroke proportion.
    * **Maximum Likelihood Estimator (MLE) Concept:** It's conceptually explained that for the stroke proportion (a Bernoulli trial), the sample proportion is the MLE.
    * Properties of good estimators (unbiased, consistent, efficient) are mentioned conceptually.

## 4. Confidence Intervals (CIs)

* **Summary:** This section focuses on constructing interval estimates (confidence intervals) for population parameters based on the sample data.
    * **CI for Mean Age:** A confidence interval for the population mean 'age' is calculated using the t-distribution (since the population standard deviation is unknown and estimated from the sample).
    * **CI for Stroke Proportion:** A confidence interval for the population proportion of strokes is calculated using the normal approximation to the binomial distribution. Conditions for this approximation (e.g., `n*p > 5`) are checked.
    * The calculated CIs provide a range within which the true population parameters are likely to fall, with a specified level of confidence (e.g., 95%).

## 5. The Bootstrap

* **Summary:** This section demonstrates the bootstrap resampling technique to estimate a confidence interval for a statistic where a simple formula might not be readily available or assumptions are hard to meet.
    * The median 'bmi' (Body Mass Index) from the dataset is chosen as the statistic of interest.
    * A large number of bootstrap samples are created by resampling with replacement from the original 'bmi' data.
    * The median is calculated for each bootstrap sample.
    * The distribution of these bootstrap medians is plotted.
    * A percentile confidence interval for the population median 'bmi' is derived from this distribution.

## 6. Bayesian Approach (Conceptual Example)

* **Summary:** This section provides a conceptual illustration of Bayesian inference by estimating the probability of stroke.
    * A Beta-Binomial model is used, where the Beta distribution serves as the conjugate prior for the Binomial likelihood of observing strokes.
    * A (weakly informative) prior distribution for the stroke probability is defined (e.g., Beta(1,1) - a uniform distribution).
    * This prior is updated using the observed stroke data (number of strokes and total observations from a sample) to form a posterior distribution.
    * The mean of this posterior distribution is taken as a point estimate for the stroke probability.
    * A credible interval (the Bayesian equivalent of a confidence interval) is calculated from the posterior distribution.
    * The prior and posterior distributions are plotted to visualize how beliefs about the stroke probability are updated by the data.

## Stroke Prediction: Model Selection and Evaluation

This section outlines the process of selecting, building, and evaluating a machine learning model for the stroke prediction dataset. We begin with a justification for our initial model choice, logistic regression, and then detail a systematic approach to compare multiple models and select the best performer.

---


# 🔍📊⚙️Systematic Model Selection Process

To ensure we select the most effective model, we will follow a structured process to train, evaluate, and compare several classification algorithms.

### Step 1: Understand and Preprocess the Data

- **Analyze Variables**: Identify the data type (categorical, numerical, binary) for each feature.
- **Handle Missing Values**: Implement a strategy for missing data, such as imputation for the `bmi` column.
- **Encode Categorical Variables**: Convert categorical features (e.g., `gender`, `smoking_status`) into a numerical format using one-hot encoding.
- **Scale Numerical Features**: Normalize or standardize numerical variables (`age`, `avg_glucose_level`, `bmi`) to ensure they are on a comparable scale, which is important for models like SVM and k-NN.

### Step 2: Split the Dataset

Divide the data into a **training set (70%)** and a **testing set (30%)**. This allows the model to be evaluated on unseen data, providing a more realistic measure of its performance.

### Step 3: Address Class Imbalance

Apply a resampling technique to the training data only to prevent data leakage. **SMOTE** is a recommended approach to generate synthetic samples for the minority class (stroke cases).

### Step 4: Train and Evaluate Multiple Models

- Train a variety of classification models on the preprocessed and resampled training data.
- Use **5-fold or 10-fold cross-validation** during training to get a more robust estimate of performance.
- Evaluate each model on the held-out test set using a standard set of classification metrics.

### Step 5: Compare Model Performance

The performance of each model will be compiled into a comparison table. The primary metrics for evaluation are:

- **Accuracy**: Overall percentage of correct predictions.
- **Precision**: Of the predicted strokes, how many were actual strokes? (Measures correctness of positive predictions).
- **Recall (Sensitivity)**: Of all the actual strokes, how many did the model detect? (Measures completeness).
- **F1-Score**: The harmonic mean of Precision and Recall, providing a single score that balances both.
- **ROC-AUC Score**: The area under the Receiver Operating Characteristic curve, which measures the model's ability to distinguish between the two classes.

#### Model Performance Comparison Table

| Model                    | Accuracy (%) | Precision | Recall / Sensitivity | F1-Score | ROC-AUC Score |
|--------------------------|--------------|-----------|-----------------------|----------|----------------|
| Logistic Regression      | 95.00        | 0.95      | 1.00                  | 0.9732   | 0.8361         |
| Naive Bayes              | 56.75        | 0.99      | 0.55                  | 0.7123   | 0.8021         |
| Decision Tree            | 92.56        | 0.96      | 0.970                 | 0.9601   | 0.5530         |
| Support Vector Machine   | 81.21        | 0.1306    | 0.5057                | 0.2075   | 0.7818         |
| Random Forest            | 92.95        | 0.1100    | 0.0632                | 0.0803   | 0.7697         |
| k-Nearest Neighbors      | 80.43        | 0.1015    | 0.3851                | 0.1607   | 0.6583         |
| XGBoost                  | TBD          | TBD       | TBD                   | TBD      | TBD            |


### Step 6: Final Model Selection

The best model will be chosen based on a balance of performance metrics, with a particular focus on **Recall** and **ROC-AUC**, as failing to detect a stroke (a false negative) is more critical than a false positive in this context.

Interpretability and computational cost will also be considered in the final decision.

## Model Choice: Logistic Regression

For the initial analysis of the stroke prediction dataset, **Logistic Regression** was chosen as the primary model. This decision is based on several key characteristics of the data and the modeling goals.

### Why Logistic Regression?

- **Binary Outcome**:  
  The target variable, `stroke`, is binary (`0` for no stroke, `1` for stroke). Logistic regression is a statistical method specifically designed to model the probability of such binary outcomes.

- **Handles Mixed Data Types**:  
  The dataset contains both continuous (e.g., `age`, `avg_glucose_level`) and categorical (e.g., `gender`, `work_type`) predictors. Logistic regression can effectively handle both types of variables with appropriate preprocessing, such as one-hot encoding for categorical features.

- **High Interpretability**:  
  A major advantage of logistic regression is its interpretability. The model's coefficients can be translated into odds ratios, providing clear insights into how each variable influences the likelihood of a stroke. This is especially valuable in a medical context where understanding risk factors is crucial.

- **Strong Statistical Foundation**:  
  As a type of Generalized Linear Model (GLM), logistic regression is a well-established and robust statistical method. It does not assume normality of predictors, making it more flexible than some other models like Linear Discriminant Analysis.

- **Implementation Efficiency**:  
  Logistic regression is computationally efficient and widely available in standard data science libraries like Scikit-learn (Python) and R, making it a practical and accessible baseline model.

### Potential Challenges

- **Class Imbalance**:  
  The dataset is highly imbalanced, with far fewer instances of strokes than non-strokes. This can bias the model towards the majority class. Techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) or adjusting class weights are necessary to mitigate this issue.

- **Linearity Assumption**:  
  Logistic regression assumes a linear relationship between the predictor variables and the log-odds of the outcome. More complex models like Random Forests or Gradient Boosting might capture non-linear relationships and interactions more effectively, potentially leading to higher accuracy.


# 📎 Reference

## Links

Orduz, J. C. (2020, May 5). Getting started with spectral clustering. KDnuggets. https://www.kdnuggets.com/2020/05/getting-started-spectral-clustering.html  

Deshpande, T. (n.d.). Stroke prediction: Effect of data leakage | SMOTE. Kaggle. Retrieved from https://www.kaggle.com/code/tanmay111999/stroke-prediction-effect-of-data-leakage-smote  

Joshua's Words. (n.d.). Predicting a stroke [SHAP, LIME Explainer & ELI5]. Kaggle. Retrieved from https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5  

Chennoju, B. (n.d.). Data storytelling AUC focus on strokes. Kaggle. Retrieved from https://www.kaggle.com/code/bhuvanchennoju/data-storytelling-auc-focus-on-strokes