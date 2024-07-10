Certainly! Here's the updated README file with the specified headings and descriptions for both models:

---

# Spam Email Classifier

## Overview

This repository hosts a machine learning project designed to classify emails as spam or non-spam (ham) using Naive Bayes and Support Vector Machine (SVM) algorithms. The project leverages Python, scikit-learn, and NLTK libraries to preprocess data, train models, and evaluate their performance.

## Steps

1. **Data Visualization:** Explore dataset characteristics using Matplotlib and Seaborn.
   
2. **Data Cleaning:** Preprocess data by encoding categorical variables, handling missing values, removing duplicates, and filtering by predefined categories (0 for ham, 1 for spam).
   
3. **Text Preprocessing:** Prepare text data in the "Message" column by removing stopwords, punctuation, and applying stemming using NLTK.
   
4. **Feature Extraction:** Utilize CountVectorizer to transform text data into numerical features suitable for machine learning models.
   
5. **Model Training:** Train Naive Bayes (MultinomialNB) and SVM (Support Vector Classifier) models on preprocessed data.
   
6. **Model Evaluation:** Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score to assess their ability to classify spam and ham emails.
   
7. **Testing:** Validate models on unseen data to gauge their generalization capabilities.

## Accuracy

- **Naive Bayes Metrics:**
  - Accuracy: 0.9812
  - Precision: 0.9306
  - Recall: 0.9241
  - F1 Score: 0.9273

- **SVM Metrics:**
  - Accuracy: 0.9848
  - Precision: 0.9923
  - Recall: 0.8897
  - F1 Score: 0.9382

## Brief Description of Models

- **Naive Bayes:** Naive Bayes is a probabilistic classifier based on applying Bayes' theorem with strong independence assumptions between features. It is particularly effective for text classification tasks like spam detection due to its simplicity and efficiency in handling large datasets.
  
- **Support Vector Machine (SVM):** SVM is a powerful supervised learning algorithm capable of performing classification, regression, and outlier detection tasks. It works by finding the hyperplane that best separates classes in a high-dimensional feature space, making it effective for complex classification problems with well-defined boundaries.

## Dependencies

- Python 3
- Libraries: scikit-learn, NLTK, Matplotlib, Seaborn, Panadas, Numpy

## Acknowledgments

This project draws inspiration from educational materials and examples provided by scikit-learn documentation, NLTK documentation, and various online tutorials.
