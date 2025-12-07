"""
HR Employee Attrition Data App
Author: TRAN Thi Ngoc Anh
TBS MSc AIBA – Python for Data Science Final Project

This Streamlit data application:
- Loads HR Employee Attrition dataset or a user-uploaded CSV
- Performs EDA visualizations with user-controlled settings
- Includes an ML-based attrition prediction tool
- Calls an external API (Advice Slip)
- Follows PEP8 and modern Streamlit structure
"""

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# STREAMLIT CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="HR Employee Attrition Analytics",
    page_icon="📊",
    layout="wide",
)

sns.set_style("whitegrid")


# -------------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------------
@st.cache_data
def load_default_data() -> pd.DataFrame:
    """Load the default HR dataset from CSV."""
    return pd.read_csv("HR Employee Attrition.csv")


@st.cache_data
def load_uploaded_data(file) -> pd.DataFrame:
    """Load a user-uploaded CSV file."""
    return pd.read_csv(file)


# -------------------------------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------------------------------
def categorize_income(income: float) -> str:
    """Categorize employees by income bracket."""
    if income < 5000:
        return "Low"
    if income < 10000:
        return "Medium"
    return "High"


def add_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Add IncomeLevel and SeniorityLevel columns."""
    data = dataframe.copy()
    data["IncomeLevel"] = data["MonthlyIncome"].apply(categorize_income)

    seniority_levels: List[str] = []
    for years in data["YearsAtCompany"]:
        if years < 3:
            seniority_levels.append("Junior")
        elif years < 10:
            seniority_levels.append("Mid-level")
        else:
            seniority_levels.append("Senior")

    data["SeniorityLevel"] = seniority_levels
    return data


# -------------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------------
@st.cache_resource
def train_attrition_model(
    dataframe: pd.DataFrame,
) -> Tuple[Pipeline, List[str], Dict[str, Dict[str, float]]]:
    """
    Train a logistic regression model to predict attrition.

    Args:
        dataframe: Preprocessed HR dataframe.

    Returns:
        model: Trained sklearn pipeline.
        feature_names: List of numeric feature column names.
        feature_stats: Dict of {feature: {min, max, mean}} for UI defaults.
    """
    data = dataframe.copy()

    # Create binary target
    data["AttritionBinary"] = (data["Attrition"] == "Yes").astype(int)

    # Numeric features, excluding target
    x_features = data.select_dtypes(include="number").drop(
        columns=["AttritionBinary"],
        errors="ignore",
    )
    y_target = data["AttritionBinary"]

    # Stats used to build interactive UI inputs
    feature_stats: Dict[str, Dict[str, float]] = {
        col: {
            "min": float(x_features[col].min()),
            "max": float(x_features[col].max()),
            "mean": float(x_features[col].mean()),
        }
        for col in x_features.columns
    }

    # ML pipeline
    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("logreg", LogisticRegression(max_iter=500)),
        ],
    )

    model.fit(x_features, y_target)

    return model, list(x_features.columns), feature_stats


# -------------------------------------------------------------------
# EXTERNAL API CALL (BONUS)
# -------------------------------------------------------------------
def get_random_advice() -> str:
    """Call Advice Slip API and return a short motivation/advice text."""
    url = "https://api.adviceslip.com/advice"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        return data["slip"]["advice"]
    except (requests.RequestException, ValueError, KeyError):
        return "Could not retrieve advice at this time."


# -------------------------------------------------------------------
# MAIN APP
# -------------------------------------------------------------------
def main() -> None:
    """Main function to run the Streamlit app."""
    st.sidebar.title("Navigation")

    page = st.sidebar.radio(
        "Go to:",
        [
            "Home",
            "About Project",
            "Dashboard",
            "Insights",
            "Attrition Analysis",
            "Attrition Predictor (AI Model)",
            "External API Demo",
            "Visualizations",
        ],
    )

    st.sidebar.markdown("---")

    # FILE UPLOAD FEATURE
    uploaded_file = st.sidebar.file_uploader(
        "Upload your HR CSV file",
        type=["csv"],
    )

    if uploaded_file:
        hr_df = load_uploaded_data(uploaded_file)
        st.sidebar.success("Using uploaded dataset.")
    else:
        hr_df = load_default_data()
        st.sidebar.info("Using default dataset: HR Employee Attrition.")

    # Add engineered features
    hr_df = add_features(hr_df)

    # Quick stats
    avg_age = hr_df["Age"].mean()
    avg_income = hr_df["MonthlyIncome"].mean()
    corr_age_income = np.corrcoef(
        hr_df["Age"],
        hr_df["MonthlyIncome"],
    )[0, 1]

    # ----------------------------------------------------------------
    # PAGE: HOME
    # ----------------------------------------------------------------
    if page == "Home":
        st.title("HR Employee Attrition Analytics")
        st.subheader("TBS MSc AIBA | Python for Data Science Final Project")
        st.caption("Author: TRAN Thi Ngoc Anh")

        st.markdown(
            """
Welcome to the **HR Employee Attrition Analytics Platform**.

This data app allows you to:
- Explore the HR Attrition dataset
- Visualize trends and insights
- Filter employees with advanced controls
- Predict attrition with a Machine Learning model
- Interact with an External API
            """,
        )

        col1, col2, col3 = st.columns(3)
        col1.metric("Average Age", f"{avg_age:.1f}")
        col2.metric("Average Income", f"€{avg_income:,.0f}")
        col3.metric("Age–Income Correlation", f"{corr_age_income:.2f}")

    # ----------------------------------------------------------------
    # PAGE: ABOUT PROJECT
    # ----------------------------------------------------------------
    elif page == "About Project":
        st.title("ℹ️ About This Project")

        st.markdown(
            """
### 🎯 Project Objective
Use Python & Streamlit to build an interactive **data app** with:
- File Import  
- Data Visualization  
- User Inputs  
- Machine Learning  
- External API Interaction (*bonus*)  

### 🧩 Tools
- Python, Pandas, NumPy  
- Matplotlib & Seaborn  
- Streamlit  
- Scikit-Learn (Logistic Regression)  
- AdviceSlip API  

### 📈 Interpretation of Results  
- Workforce is mostly young to mid-career  
- Higher education → higher income  
- Medium-income bracket most common  
- Low environment satisfaction strongly predicts attrition  
            """,
        )

    # ----------------------------------------------------------------
    # PAGE: DASHBOARD
    # ----------------------------------------------------------------
    elif page == "Dashboard":
        st.title("📊 Workforce Dashboard")
        st.markdown("Explore summary statistics and distributions.")

        # USER INPUT FOR CHART TYPE
        chart_type = st.selectbox(
            "Select chart type:",
            ["Histogram", "Boxplot", "Violin"],
        )

        col1, col2 = st.columns(2)

        numeric_cols = hr_df.select_dtypes(include="number").columns
        selected_col = col1.selectbox(
            "Select a numeric column",
            numeric_cols,
        )
        bins = col2.slider("Bins (for histogram)", 5, 60, 20)

        fig, ax = plt.subplots(figsize=(10, 6))

        if chart_type == "Histogram":
            ax.hist(
                hr_df[selected_col],
                bins=bins,
                color="skyblue",
                edgecolor="black",
            )
        elif chart_type == "Boxplot":
            sns.boxplot(x=hr_df[selected_col], ax=ax)
        else:
            sns.violinplot(x=hr_df[selected_col], ax=ax)

        ax.set_title(f"{chart_type} of {selected_col}")
        st.pyplot(fig)

    # ----------------------------------------------------------------
    # PAGE: INSIGHTS
    # ----------------------------------------------------------------
    elif page == "Insights":
        st.title("🔍 Key Insights")

        st.subheader("Education vs Income")
        ed_income = hr_df.groupby("Education")["MonthlyIncome"].mean()

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(
            x=ed_income.index,
            y=ed_income.values,
            marker="o",
            ax=ax,
        )
        ax.set_xlabel("Education Level")
        ax.set_ylabel("Average Monthly Income")
        ax.set_title("Average Income by Education Level")
        st.pyplot(fig)

    # ----------------------------------------------------------------
    # PAGE: ATTRITION ANALYSIS
    # ----------------------------------------------------------------
    elif page == "Attrition Analysis":
        st.title("📉 Attrition Analysis")

        attr_counts = hr_df["Attrition"].value_counts()

        fig, ax = plt.subplots(figsize=(10, 6))
        attr_counts.plot(
            kind="bar",
            ax=ax,
            color=["salmon", "lightgreen"],
        )
        ax.set_xlabel("Attrition")
        ax.set_ylabel("Number of Employees")
        ax.set_title("Attrition Counts")
        st.pyplot(fig)

    # ----------------------------------------------------------------
    # PAGE: ATTRITION PREDICTOR (AI MODEL)
    # ----------------------------------------------------------------
    elif page == "Attrition Predictor (AI Model)":
        st.title("🤖 Attrition Prediction Model")

        model, feature_names, feature_stats = train_attrition_model(hr_df)

        st.markdown(
            "Enter employee details to estimate attrition probability:",
        )

        user_inputs: Dict[str, float] = {}

        for feat in feature_names:
            stats = feature_stats[feat]
            user_inputs[feat] = st.number_input(
                feat,
                min_value=stats["min"],
                max_value=stats["max"],
                value=stats["mean"],
            )

        input_df = pd.DataFrame([user_inputs])

        pred_prob = model.predict_proba(input_df)[0][1]

        st.metric("Predicted Probability of Attrition", f"{pred_prob:.2%}")

        if pred_prob > 0.5:
            st.error("⚠️ Employee LIKELY to leave")
        else:
            st.success("✅ Employee UNLIKELY to leave")

    # ----------------------------------------------------------------
    # PAGE: EXTERNAL API
    # ----------------------------------------------------------------
    elif page == "External API Demo":
        st.title("🌐 External API Interaction")

        st.write("Fetching motivational HR advice for HR managers...")
        advice = get_random_advice()
        st.success(advice)

    # ----------------------------------------------------------------
    # PAGE: VISUALIZATIONS
    # ----------------------------------------------------------------
    elif page == "Visualizations":
        st.title("📈 Gallery of Visualizations")

        fig, ax = plt.subplots(figsize=(12, 7))
        sns.histplot(
            hr_df["Age"],
            bins=20,
            kde=True,
            ax=ax,
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Number of Employees")
        ax.set_title("Age Distribution with KDE")
        st.pyplot(fig)


# -------------------------------------------------------------------
# MAIN GUARD
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
