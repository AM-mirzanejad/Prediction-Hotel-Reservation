# Prediction-Hotel-Reservation

<p align="center">
  <img src="https://img.shields.io/badge/Made%20with-Python-0078D4?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white">
  <img src="https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white">
  <img src="https://img.shields.io/badge/Scikit_Learn-0078D4?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <br>
  <img src="https://img.shields.io/badge/Jupyter_Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white">
  <img src="https://img.shields.io/badge/SVC-F37626?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/GradientBoostingClassifier-2C2D72?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/RandomForestClassifier-0078D4?style=for-the-badge&logo=scikit-learn&logoColor=white">
</p>

---

## Project Overview

This project utilizes machine learning algorithms implemented in Python for predicting hotel reservation cancellations. The goal is to determine whether a hotel booking is likely to be canceled based on various features associated with the reservation.

### Key Technologies Used:

- **Python**: Programming language used for implementation.
- **Numpy**: Library for numerical computations in Python.
- **Pandas**: Library for data manipulation and analysis.
- **Scikit-Learn**: Toolkit for building and deploying machine learning models.
- **Jupyter Notebook**: Interactive environment used for development and analysis.

### Machine Learning Models:

The following machine learning classifiers are employed in this project:

- **Support Vector Classifier (SVC)**: A supervised learning model used for classification tasks.
- **Gradient Boosting Classifier**: An ensemble learning method that builds models sequentially to correct errors of previous models.
- **Random Forest Classifier**: Another ensemble learning method based on decision trees.

---
## Model Performance

The machine learning model developed for predicting hotel reservation cancellations achieved an initial prediction accuracy of 95% as of April 20th. This accuracy represents the performance of the model on the current dataset and experimental conditions.

### Future Improvements

While the initial version of the model has shown promising accuracy, further enhancements and optimizations are possible to improve performance. Potential strategies for improving model performance include:

- **Feature Engineering**: Exploring additional relevant features or transforming existing features to better capture patterns in the data.
  
- **Hyperparameter Tuning**: Adjusting model parameters and settings to optimize performance and generalize better to unseen data.
  
- **Model Selection**: Experimenting with different machine learning algorithms or ensemble methods to identify the most suitable model for this prediction task.

### Next Steps

Moving forward, the project will focus on refining the machine learning pipeline, exploring new techniques, and evaluating the model's performance on diverse datasets. Contributions and suggestions for improving the model's accuracy and robustness are welcome.

Stay tuned for updates on model enhancements and performance improvements!

---

## Jupyter Notebook

For a detailed demonstration and usage of the Heart Failure Predictor, refer to the Jupyter Notebook:

[![Jupyter Notebook](https://img.shields.io/badge/Open%20in-Jupyter%20Notebook-orange?style=for-the-badge&logo=jupyter)](https://github.com/AM-mirzanejad/Heart-Failure-Prediction/blob/main/Heart-Prediction.ipynb)

---
### Machine Learning Models Used:

- **Support Vector Classifier (SVC):**
  - Description: SVC is a supervised learning model used for classification tasks. It works by finding the hyperplane that best separates different classes in the feature space.
  - Implementation: Utilized the `sklearn.svm.SVC` class from scikit-learn.

- **Random Forest Classifier:**
  - Description: Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the class that is the mode of the classes (classification) or mean prediction (regression) of the individual trees.
  - Implementation: Used the `sklearn.ensemble.RandomForestClassifier` class from scikit-learn.

- **Gradient Boosting Classifier:**
  - Description: Gradient Boosting is an ensemble learning technique that builds a strong model by combining multiple weak learners sequentially. It builds trees one at a time, where each new tree helps to correct errors made by previously trained trees.
  - Implementation: Utilized the `sklearn.ensemble.GradientBoostingClassifier` class from scikit-learn.

- **K-Nearest Neighbors (KNN) Imputer:**
  - Description: KNN Imputer is used for imputing missing values by using the K-Nearest Neighbors approach, where missing values are imputed based on the values of neighboring data points.
  - Implementation: Utilized the `sklearn.impute.KNNImputer` class from scikit-learn.

- **MinMaxScaler:**
  - Description: MinMaxScaler is used for scaling feature values to a specified range, usually between 0 and 1.
  - Implementation: Used the `sklearn.preprocessing.MinMaxScaler` class from scikit-learn.

- **GridSearchCV:**
  - Description: GridSearchCV is used for hyperparameter tuning and model selection by exhaustively searching through a specified parameter grid and cross-validating the results.
  - Implementation: Utilized the `sklearn.model_selection.GridSearchCV` class from scikit-learn.

---


### Libraries and Packages Utilized:

- **pandas** (imported as `pd`):
  - Description: pandas is a powerful library for data manipulation and analysis, providing data structures and operations for manipulating numerical tables and time series.
  - Icon: <img src="https://img.icons8.com/color/48/000000/pandas.png" width="50" height="50"/>

- **numpy** (imported as `np`):
  - Description: numpy is a fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays.
  - Icon: <img src="https://img.icons8.com/color/48/000000/numpy.png" width="50" height="50"/>

- **scikit-learn** (imported as `sklearn`):
  - Description: scikit-learn is a popular machine learning library in Python that provides simple and efficient tools for data mining and data analysis, including a wide variety of machine learning algorithms and utilities for model selection, evaluation, and preprocessing.
  - Icon: <img src="https://icon.icepanel.io/Technology/svg/scikit-learn.svg" width="50" height="50"/>



## Heart Disease Dataset

| Column Name                            | Description                                                             |
|----------------------------------------|-------------------------------------------------------------------------|
| Booking_ID                             | Unique identifier of each booking                                        |
| no_of_adults                           | Number of adults                                                        |
| no_of_children                         | Number of children                                                     |
| no_of_weekend_nights                   | Number of weekend nights (Saturday or Sunday) stayed or booked at hotel |
| no_of_week_nights                      | Number of week nights (Monday to Friday) stayed or booked at hotel      |
| type_of_meal_plan                      | Type of meal plan booked by the customer                                 |
| required_car_parking_space             | Does the customer require a car parking space? (0 - No, 1 - Yes)         |
| room_type_reserved                     | Type of room reserved by the customer (encoded)                          |
| lead_time                              | Number of days between booking and arrival date                          |
| arrival_year                           | Year of arrival date                                                     |
| arrival_month                          | Month of arrival date                                                    |
| arrival_date                           | Date of the month                                                        |
| market_segment_type                    | Market segment designation                                               |
| repeated_guest                         | Is the customer a repeated guest? (0 - No, 1 - Yes)                     |
| no_of_previous_cancellations           | Number of previous bookings canceled by the customer                     |
| no_of_previous_bookings_not_canceled   | Number of previous bookings not canceled by the customer                 |
| avg_price_per_room                     | Average price per day of the reservation (in euros)                     |
| no_of_special_requests                 | Total number of special requests made by the customer                    |
| booking_status                         | Flag indicating if the booking was canceled or not                      |


## Installation

Clone the repository using `git`:

```bash
git clone <https://github.com/AM-mirzanejad/Prediction-Hotel-Reservation.git>
cd <dPrediction-Hotel-Reservation>
pip install -r requirements.txt
