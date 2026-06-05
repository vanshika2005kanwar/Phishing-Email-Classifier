# Phishing-Email-Classifier
A machine learning model that detects phishing emails using NLP and  Logistic Regression. Achieved 99.39% accuracy on 39,154 real emails.
# 📧 Phishing Email Classifier

A machine learning project that classifies emails as phishing or 
legitimate using Natural Language Processing (NLP) and Logistic 
Regression.

## 📥 Dataset
Download CEAS_08.csv from Kaggle and place it in the root folder:
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset


## 🎯 Project Overview
Phishing emails are one of the most common cybersecurity threats today.
This project builds a classifier trained on 39,154 real emails from the 
CEAS 2008 dataset that detects phishing emails with 99.39% accuracy.

## 🏆 Results
| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 99.39% | 99% | 100% | 99% |
| Naive Bayes | 97.47% | 100% | 96% | 98% |

Logistic Regression was selected as the final model due to its 
superior recall — catching 100% of phishing emails.
