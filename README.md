Titanic Survival Prediction: A Multi-Model ML & ANN Approach

This repository presents a comprehensive portfolio project focused on predicting survival on the Titanic using various machine learning and deep learning methods. The goal is not only to maximize model performance but also to showcase applied understanding of feature engineering, model tuning, evaluation, and explainability.

📁 Files Included
	•	Steve's_titanic_RF_XG_ANN_CNN.ipynb: Complete notebook with preprocessing, modeling, evaluations.
	•	README.md: This file.

⸻

 Project Overview

This notebook explores multiple approaches to the Titanic dataset from Kaggle using:
	•	Random Forest Classifier (RF)
	•	XGBoost Classifier
	•	Artificial Neural Networks (ANN)
	•	Convolutional Neural Networks (CNN)

Each model was developed incrementally, allowing for a progressive learning structure that illustrates concept mastery and experimentation.

Techniques Used

Data Cleaning & Feature Engineering
	•	Deck Imputation via Random Forest: Imputed missing deck values by training a classifier on Fare and Pclass after hypothesizing deck affected survival likelihood.
	•	Family Size + Derived Features:
 TRD['Family Size'] = TRD['SibSp'] + TRD['Parch'] + 1
TRD['Is Alone'] = (TRD['Family Size'] == 1).astype(int)
TRD['BigFamily'] = (TRD['Family Size'] > 4).astype(int)

These features were crafted to capture social travel behavior and group survival tendencies.

 Model Building
	•	Random Forest Classifier: Tuned using GridSearchCV. Achieved up to 82.68% accuracy.
	•	XGBoost Classifier: Tuned using RandomizedSearchCV. Achieved ~82.12% accuracy.
	•	ANN:
	•	3–4 dense layers with dropout and batch normalization.
	•	Binary focal loss for handling class imbalance.
	•	Class weights computed dynamically.
	•	Achieved ~82.68% accuracy.
	•	CNN:
	•	Exploratory CNN adaptation on tabular data reshaped into 2D.
	•	Accuracy ~72.07% (shown for experimentation, not optimization).

 Evaluation Metrics
	•	Accuracy
	•	Confusion Matrix
	•	Validation Plots (loss/accuracy over epochs)


  Key Takeaways
	•	Importance of scaling and label encoding before training.
	•	How ANN depth, dropout, and regularization affect performance.
	•	Real-world utility of class_weight and binary_focal_crossentropy in imbalanced datasets.
	•	Ensemble stacking boosts stability but not always accuracy.
	•	CNNs underperform on small, structured datasets not suited for spatial feature learning.

 How to Run
	1.	Clone the repository:
 git clone https://github.com/yourusername/titanic-survival-prediction.git
 	2.	Open the notebook:
  jupyter notebook "Steve's_titanic_RF_XG_ANN_CNN.ipynb"
	3.	Install requirements:
 pip install -r requirements.txt
