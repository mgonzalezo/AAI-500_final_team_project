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
### 1.2 Dataset Attributes
### 1.3 Standardize Categorical Values
### 1.4 Inspecting Missing Values
### 1.5 Replacing the Missing Values
### 1.6 Categorizing the data in different types
### 1.7 Duplicate Check in Dataset
### 1.8 Outlier Detection
### 1.9 Treatment of Outliers

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
