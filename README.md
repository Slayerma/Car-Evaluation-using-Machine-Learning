# Car Evaluation Using Machine Learning

## Overview

This project aims to develop a predictive model for car acceptability using various machine learning algorithms. The workflow follows data preprocessing, exploratory data analysis (EDA), feature engineering, model training, evaluation, and ensemble learning techniques to optimize performance.

### Objectives
- Clean and preprocess the car evaluation dataset.
- Analyze the dataset through EDA.
- Train and evaluate multiple models: Decision Trees, Random Forests, XGBoost, and ensemble methods.
- Optimize models using hyperparameter tuning.
- Compare performance and select the best model based on accuracy and other metrics.

## Workflow Steps
1. **Data Collection**: Loaded the car evaluation dataset with attributes like price, maintenance cost, safety, and others.
2. **Data Preprocessing**: Handled missing values, encoded categorical features, and normalized numerical data.
3. **EDA**: Analyzed the data distribution, identified patterns, and visualized key relationships.
4. **Model Training**: Trained models using Decision Trees, Random Forests, and XGBoost, along with ensemble methods.
5. **Evaluation**: Compared models using accuracy, precision, recall, and F1-score.
6. **Hyperparameter Tuning**: Used Grid Search and Random Search to optimize model parameters.
7. **Ensemble Learning**: Applied stacking to improve model performance by combining multiple models.
8. **Model Saving**: Saved the best-performing model for future use.

## Models and Algorithms
- **Custom Decision Tree Classifier**
- **Random Forest**
- **XGBoost** with and without **SMOTE**
- **Stacking Ensemble**

## Dataset
The dataset includes features such as:
- **Buying Price**: vhigh, high, med, low
- **Maintenance**: vhigh, high, med, low
- **Doors**: 2, 3, 4, 5+
- **Persons**: 2, 4, more
- **Luggage Boot**: small, med, big
- **Safety**: low, med, high
- **Acceptability**: unacc, acc, good, vgood

## Performance Metrics
The models were evaluated using the following metrics:
- **Accuracy**
- **Precision**
- **Recall**
- **F1-score**

The best-performing models were:
- **XGBoost with SMOTE**: Achieved the highest accuracy by handling class imbalance effectively.
- **Stacking Ensemble**: Combined multiple models to leverage their strengths and achieved excellent overall performance.

## Insights and Future Improvements
- **Class Imbalance**: SMOTE helped address the imbalance in the 'acceptability' variable, improving recall for minority classes.
- **Feature Engineering**: Interaction features enhanced model performance, and further feature selection or dimensionality reduction may be explored.

## Conclusion
The data science workflow successfully developed a robust predictive model for car acceptability. Ensemble methods, particularly SMOTE-enhanced XGBoost and Stacking, demonstrated superior performance.
