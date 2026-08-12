import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Financial Stress Early Warning System",
    page_icon="📊",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "xgboost_financial_stress.joblib"
)

IMPUTER_PATH = os.path.join(
    BASE_DIR,
    "models",
    "financial_imputer.joblib"
)

THRESHOLD_PATH = os.path.join(
    BASE_DIR,
    "models",
    "financial_stress_threshold.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "models",
    "feature_columns.joblib"
)

SHAP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "shap_feature_importance.csv"
)

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        MODEL_PATH
    )

    imputer = joblib.load(
        IMPUTER_PATH
    )

    threshold = joblib.load(
        THRESHOLD_PATH
    )

    features = joblib.load(
        FEATURE_PATH
    )

    return (
        model,
        imputer,
        threshold,
        features
    )


try:

    model, imputer, threshold, features = (
        load_model()
    )

except Exception as e:

    st.error(
        "Model files could not be loaded."
    )

    st.code(str(e))

    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title(
    "📊 Financial Stress Early Warning System"
)

st.markdown(
    """
    ### XGBoost-Based Financial Stress Prediction

    This system predicts the probability that a company
    will experience **financial stress in the following year**
    using historical financial indicators.
    """
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header(
    "Company Information"
)

company_name = st.sidebar.text_input(
    "Company Name",
    "Sample Company"
)

financial_year = st.sidebar.number_input(
    "Financial Year",
    min_value=2015,
    max_value=2035,
    value=2025
)

st.sidebar.info(
    f"Model threshold: {threshold:.2f}"
)

# ============================================================
# INPUTS
# ============================================================

st.header(
    "Enter Financial Indicators"
)

st.write(
    "Enter the company's financial indicators below."
)

# ------------------------------------------------------------
# Main financial values
# ------------------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:

    revenue = st.number_input(
        "Revenue",
        min_value=0.0,
        value=100000.0
    )

    operating_profit = st.number_input(
        "Operating Profit",
        value=10000.0
    )

    other_income = st.number_input(
        "Other Income",
        value=1000.0
    )

    interest = st.number_input(
        "Interest",
        min_value=0.0,
        value=1000.0
    )

    depreciation = st.number_input(
        "Depreciation",
        min_value=0.0,
        value=2000.0
    )

with col2:

    profit_before_tax = st.number_input(
        "Profit Before Tax",
        value=8000.0
    )

    pat = st.number_input(
        "PAT",
        value=6000.0
    )

    equity_capital = st.number_input(
        "Equity Capital",
        min_value=0.0,
        value=5000.0
    )

    reserves = st.number_input(
        "Reserves",
        value=30000.0
    )

    borrowings = st.number_input(
        "Borrowings",
        min_value=0.0,
        value=10000.0
    )

with col3:

    cfo = st.number_input(
        "CFO",
        value=7000.0
    )

    cfi = st.number_input(
        "CFI",
        value=-3000.0
    )

    cff = st.number_input(
        "CFF",
        value=-2000.0
    )

    fcf = st.number_input(
        "FCF",
        value=5000.0
    )

    other_assets = st.number_input(
        "Other Assets",
        min_value=0.0,
        value=20000.0
    )

# ============================================================
# RATIOS
# ============================================================

st.subheader(
    "Financial Ratios"
)

r1, r2, r3 = st.columns(3)

with r1:

    profit_margin = st.number_input(
        "Profit Margin (%)",
        value=10.0
    )

    interest_coverage = st.number_input(
        "Interest Coverage",
        value=5.0
    )

with r2:

    cash_conversion_cycle = st.number_input(
        "Cash Conversion Cycle",
        value=60.0
    )

    debtor_days = st.number_input(
        "Debtor Days",
        value=60.0
    )

with r3:

    revenue_growth = st.number_input(
        "Revenue Growth (%)",
        value=10.0
    )

    fcf_revenue = st.number_input(
        "FCF / Revenue",
        value=5.0
    )

cfo_op = st.number_input(
    "CFO / Operating Profit",
    value=0.7
)

# ============================================================
# BUILD INPUT DATA
# ============================================================

if st.button(
    "🔍 Predict Financial Stress",
    type="primary",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create empty feature dataframe
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        np.nan,
        index=[0],
        columns=features
    )

    # --------------------------------------------------------
    # Fill known features
    # --------------------------------------------------------

    values = {

        "Revenue":
            revenue,

        "Operating_Profit":
            operating_profit,

        "Other_Income":
            other_income,

        "Interest":
            interest,

        "Depreciation":
            depreciation,

        "Profit_Before_Tax":
            profit_before_tax,

        "PAT":
            pat,

        "Equity_Capital":
            equity_capital,

        "Reserves":
            reserves,

        "Borrowings":
            borrowings,

        "CFO":
            cfo,

        "CFI":
            cfi,

        "CFF":
            cff,

        "FCF":
            fcf,

        "Other_Assets":
            other_assets,

        "Profit_Margin":
            profit_margin,

        "Interest_Coverage":
            interest_coverage,

        "Cash_Conversion_Cycle":
            cash_conversion_cycle,

        "Debtor_Days":
            debtor_days,

        "Revenue_Growth":
            revenue_growth,

        "FCF_Revenue":
            fcf_revenue,

        "CFO_OP":
            cfo_op
    }

    # --------------------------------------------------------
    # Insert available values
    # --------------------------------------------------------

    for feature, value in values.items():

        if feature in input_data.columns:

            input_data.loc[
                0,
                feature
            ] = value

    # --------------------------------------------------------
    # Convert to numeric
    # --------------------------------------------------------

    input_data = input_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # --------------------------------------------------------
    # Impute exactly like training
    # --------------------------------------------------------

    input_processed = imputer.transform(
        input_data
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    probability = model.predict_proba(
        input_processed
    )[0, 1]

    prediction = (
        probability >= threshold
    )

    # ========================================================
    # RESULT
    # ========================================================

    st.divider()

    st.header(
        "Prediction Result"
    )

    result_col1, result_col2 = st.columns(2)

    with result_col1:

        st.metric(
            "Financial Stress Probability",
            f"{probability * 100:.2f}%"
        )

    with result_col2:

        if prediction:

            st.error(
                "⚠️ FINANCIALLY STRESSED"
            )

        else:

            st.success(
                "✅ LOW FINANCIAL STRESS"
            )

    # --------------------------------------------------------
    # Probability bar
    # --------------------------------------------------------

    st.progress(
        float(probability)
    )

    st.caption(
        f"Decision threshold: "
        f"{threshold:.2f}"
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    st.subheader(
        "Interpretation"
    )

    if prediction:

        st.warning(
            """
            The model estimates that the company's
            probability of financial stress is above
            the selected decision threshold.

            Further financial analysis and monitoring
            are recommended.
            """
        )

    else:

        st.info(
            """
            The model estimates that the company's
            probability of financial stress is below
            the selected decision threshold.
            """
        )

# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

st.header(
    "Model Information"
)

info1, info2, info3 = st.columns(3)

with info1:

    st.metric(
        "Model",
        "XGBoost"
    )

with info2:

    st.metric(
        "Training Period",
        "2017–2023"
    )

with info3:

    st.metric(
        "Final Test Period",
        "2024–2025"
    )

st.caption(
    """
    The model was evaluated using a chronological
    train/validation/test methodology. The final
    2024–2025 period was kept as a holdout test set.
    """
)

# ============================================================
# SHAP IMPORTANCE
# ============================================================

if os.path.exists(
    SHAP_PATH
):

    st.divider()

    st.header(
        "📈 Model Feature Importance"
    )

    shap_df = pd.read_csv(
        SHAP_PATH
    )

    st.write(
        "Top features influencing the model's predictions:"
    )

    st.bar_chart(
        shap_df
        .head(10)
        .set_index("Feature")[
            "Mean_Absolute_SHAP"
        ]
    )

    st.caption(
        """
        SHAP importance indicates how strongly features
        contribute to model predictions. It does not imply
        causation.
        """
    )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Financial Stress Early Warning System | "
    "XGBoost + SHAP"
)