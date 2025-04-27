# Predicting-the-Propagation-Patterns-of-Health-Misinformation-Across-Social-Media-Platforms

**Overview:**   
Health misinformation, especially during crises like COVID-19, significantly impacts public well-being. This project aims to answer the following research questions
1. How does misinformation cluster within communities on social media, and what does this reveal about echo chambers and information silos?
2. How do thematic clusters of hashtags correlate with the misinformation ratio, and what does this imply for targeted fact-checking and public health messaging?
3. How do retrieval-augmented transformer models compare with traditional approaches, fine-tuned transformer models and prompt-based large language models in terms of accuracy, robustness, and explainability for classifying health misinformation on social media?


**What will you find in this repository?:**   
The data used is a combination of three datasets. COVIDLIES, HealthLies, and CONSTRAINT.
We have implemented the following models:
1. Classical Machine Learning Models:
Logistic Regression: Linear classifier with L2 regularization
Naive Bayes: Probabilistic classifier based on Bayes' theorem
Random Forest: Ensemble of decision trees with bootstrap sampling
Support Vector Machine (SVM): Implementation with RBF kernel
2. Transformer-Based Models:
BERT Baseline: Pre-trained BERT-base model fine-tuned on our dataset
RoBERTa: Optimized BERT architecture with improved training methodology
RoBERTa + RAG: RoBERTa model enhanced with Retrieval Augmented Generation 
Qwen 2.5 (7B): zero-shot instruction prompting for misinformation detection
Qwen 2.5 + RAG: Qwen 2.5 model augmented with domain-specific knowledge retrieval

In addition to the modelling, we have implemented Network Analysis, which includes the following:
1. Community Detection via Content Similarity
2. Hashtag Network Analysis



**Group Members:**   
Nazeefa Muzammil  
Md Mehedi Hasan Jibon   
Alejandro Delgado Rios   
Fahim Rahman
