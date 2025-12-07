HR Employee Attrition Data App

Author: TRAN Thi Ngoc Anh
Program: MSc AIBA – Toulouse Business School
Course: Python for Data Science – Final Project

📊 HR Employee Attrition – Interactive Data Application

This project transforms the HR Employee Attrition dataset into a complete interactive Streamlit web application.
It demonstrates advanced Python skills across:

Data cleaning & feature engineering

Exploratory Data Analysis (EDA)

Interactive visual analytics

Machine learning prediction

REST API integration

PEP8-compliant architecture

The application is designed to help HR managers explore workforce characteristics, identify risk factors for attrition, and evaluate employee-level attrition probability through a predictive ML model.

🚀 Live Application (Streamlit Cloud)

👉 (Include your deployed link here once published)

https://your-app-name.streamlit.app

📁 Project Structure
hr-attrition-app/
│
├── webapp.py                   # Main Streamlit application
├── HR Employee Attrition.csv   # Dataset used for analysis
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation (this file)
└── images/                     # Optional screenshots for demonstration

🧠 Key Features
1️⃣ Data Import

Load the default dataset (HR Employee Attrition.csv)

OR upload a custom HR dataset using a file uploader

Automatic preprocessing and feature engineering

2️⃣ Feature Engineering

Generated columns include:

IncomeLevel

Low (< €5,000)

Medium (€5,000–€10,000)

High (> €10,000)

SeniorityLevel

Junior (< 3 years)

Mid-level (3–10 years)

Senior (> 10 years)

These help segment employees for targeted insights.

3️⃣ Dashboard & Data Visualizations

Interactive data exploration tools:

Histogram / Boxplot / Violin plot

Dynamic filtering widgets

Numeric column selection

Age distribution, income distribution, attrition levels, and more

Each plot is rendered with Matplotlib or Seaborn.

4️⃣ Insights Page

Provides deeper workforce insights:

Education level vs. monthly income

Trends in satisfaction, seniority, and compensation

Data-driven commentary & interpretation

5️⃣ Attrition Analysis

Explore attrition trends:

Attrition counts (Yes/No)

Visual comparison of employee groups

Identification of key risk factors

6️⃣ Machine Learning Model

The app includes an ML-driven Attrition Predictor using:

Logistic Regression

Scikit-Learn Pipeline (Scaler + Model)

User-adjustable numerical inputs

Probability-based prediction output

Output example:

Predicted Probability of Attrition: 73%
⚠️ Employee LIKELY to leave

7️⃣ External API Integration

The app calls the publicly available AdviceSlip API:

https://api.adviceslip.com/advice


Used in the “External API Demo” page to fetch HR-related advice dynamically.

🛠️ Installation & Running Locally
1. Clone this repository:
git clone https://github.com/YOUR_USERNAME/hr-attrition-app.git
cd hr-attrition-app

2. Create a virtual environment (optional but recommended):
python -m venv env
source env/bin/activate        # Mac/Linux
env\Scripts\activate           # Windows

3. Install dependencies:
pip install -r requirements.txt

4. Run the application:
streamlit run webapp.py

📦 Dependencies

requirements.txt should include:

streamlit
pandas
numpy
seaborn
matplotlib
requests
scikit-learn

🧪 Pylint & Code Quality

This project follows PEP8 standards, uses docstrings, proper naming conventions, and modular functions.

To evaluate Pylint score:

pylint webapp.py


Target score: 9.0+ / 10 (projects receive points based on pylint score).

📊 Dataset Overview

The dataset includes:

Demographic information (Age, Gender, Education, etc.)

Job-related features (Department, JobRole, OverTime, etc.)

Compensation (MonthlyIncome, HourlyRate)

Satisfaction scores (Environment, Job, Work-Life Balance)

Attrition label (Yes/No)

This dataset enables end-to-end HR analytics and ML modeling.

🧩 What This Project Demonstrates

✔ Advanced Python programming
✔ Data import & preprocessing
✔ Data visualization with Seaborn/Matplotlib
✔ UI interactivity with Streamlit
✔ ML pipelines with Scikit-Learn
✔ REST API consumption
✔ PEP8 coding best practices
✔ Applied data science for HR decision-making

🙌 Acknowledgements

This project was created as part of the Advanced Python for Data Science course at Toulouse Business School.

Instructor(s):

Nicolas Vannson, PhD

Dataset:

IBM HR Analytics Employee Attrition Dataset (publicly available)

📬 Contact

TRAN Thi Ngoc Anh
tna.tran@tbs-education.org
MSc Artificial Intelligence & Business Analytics
Toulouse Business School
