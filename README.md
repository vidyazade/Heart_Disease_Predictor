# Heart_Disease_Predictor
The Heart Disease Predictor is a machine learning-powered web application built using Python and Streamlit. It helps users assess the risk of heart disease based on various clinical parameters. This tool is designed for educational and diagnostic support purposes and can be used by healthcare students, professionals, and individuals seeking early risk insights.

# Features
* Single Prediction: Enter details of one individual and get an instant prediction.

* Bulk Prediction: Upload a CSV file with multiple records to get batch predictions.

* Model Information: Visual explanations, feature importance charts, and performance comparison of different ML models (e.g., Logistic Regression, Random Forest).

# Input Parameters & Descriptions

| **Parameter** | **Description**                                                                                           |
| ------------- | --------------------------------------------------------------------------------------------------------- |
| `age`         | Age of the individual (in years).                                                                         |
| `sex`         | Sex of the individual (1 = male, 0 = female).                                                             |
| `cp`          | Chest pain type:<br>0 = Typical angina<br>1 = Atypical angina<br>2 = Non-anginal pain<br>3 = Asymptomatic |
| `trestbps`    | Resting blood pressure (in mm Hg).                                                                        |
| `chol`        | Serum cholesterol (in mg/dl).                                                                             |
| `fbs`         | Fasting blood sugar > 120 mg/dl (1 = true, 0 = false).                                                    |
| `restecg`     | Resting electrocardiographic results:<br>0 = Normal<br>1 = ST-T abnormality<br>2 = LV hypertrophy         |
| `thalach`     | Maximum heart rate achieved.                                                                              |
| `exang`       | Exercise-induced angina (1 = yes, 0 = no).                                                                |
| `oldpeak`     | ST depression induced by exercise relative to rest.                                                       |
| `slope`       | Slope of the peak ST segment:<br>0 = Upsloping<br>1 = Flat<br>2 = Downsloping                             |
| `ca`          | Number of major vessels (0–3) colored by fluoroscopy.                                                     |
| `thal`        | Thalassemia:<br>1 = Normal<br>2 = Fixed defect<br>3 = Reversible defect                                   |

# Tech Stack
* Python

* Streamlit (for the frontend)

* Pandas, NumPy, Scikit-learn (for model training and data handling)

* Plotly (for visualizations)

# How to run locally
git clone https://github.com/your-username/heart-disease-predictor.git
cd heart-disease-predictor
pip install -r requirements.txt
streamlit run app1.py
