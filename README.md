# AAI-500_final_team_project
University of San Diego AAI-500 Final team project - Stroke Prediction Dataset


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

## 🧼 Data Cleaning Steps

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


## 💾 Explore Data

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

## ✅ Ready for ML

## 📎 Notes
