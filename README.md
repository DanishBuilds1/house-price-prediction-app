\# 🏠 House Price Prediction



An end-to-end machine learning project that explores house-price prediction using Python, Scikit-learn, XGBoost, and Streamlit.



\## 🚀 Live Demo



\*\*\[Try the House Price Prediction App](https://danish-house-price-predictor.streamlit.app/)\*\*



Enter property details into the web application to see the model's predicted price.



\## 📌 Project Overview



This project explores the complete machine learning workflow, from data analysis and feature engineering to model training, evaluation, interpretation, and deployment.



The application uses property characteristics to generate predictions using a trained XGBoost regression model.



\## ✨ Features



\* Exploratory data analysis and data cleaning

\* Feature engineering and preprocessing

\* Handling categorical and numerical features

\* Comparison of multiple regression algorithms

\* Hyperparameter tuning using cross-validation

\* Model interpretation using SHAP

\* Interactive web interface built with Streamlit

\* Saved model pipeline for predictions on new inputs



\## 🛠️ Technologies



\* Python

\* Pandas and NumPy

\* Matplotlib

\* Scikit-learn

\* XGBoost

\* SHAP

\* Streamlit

\* Git and GitHub



## 🤖 Final Model

The selected model is a tuned XGBoost regressor trained using a log-transformed target.

| Parameter | Value |
|---|---:|
| Number of estimators | 300 |
| Maximum depth | 5 |
| Learning rate | 0.1 |
| Subsample | 1.0 |
| Column subsampling | 0.8 |

## 📊 Model Performance

Results on the held-out test set:

| Metric | Result |
|---|---:|
| R² | 0.9111 |
| MAE | 61,215.65 |
| RMSE | 111,903.04 |

These metrics are expressed in the units of the dataset's `Price` column. They describe performance on the test split and do not establish accuracy for real-world property valuations.



\## 📂 Project Structure



\* `app.py` — Streamlit application.

\* `house\_price\_model.pkl` — Saved preprocessing and prediction pipeline.

\* `requirements.txt` — Python dependencies.

\* `1st project House pricing.ipynb` — Original machine learning notebook.



\## 💻 Run Locally



Clone the repository:



```bash

git clone https://github.com/DanishBuilds1/house-price-prediction-app.git

```



Enter the project directory:



```bash

cd house-price-prediction-app

```



Install the dependencies:



```bash

pip install -r requirements.txt

```



Run the application:



```bash

streamlit run app.py

```



\## ⚠️ Limitations



The dataset is named `House Price India.csv`, but its geographic consistency and price units have not yet been independently verified. Consequently, the model should currently be treated as an educational demonstration rather than a validated Indian house-price estimator.



Predictions are estimates generated from patterns learned in the training data and should not be used as the sole basis for financial or property-buying decisions.



\## 👨‍💻 Author



\*\*DanishBuilds1\*\*



\[GitHub Profile](https://github.com/DanishBuilds1)



