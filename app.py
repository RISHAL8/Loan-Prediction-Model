import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="Credit Decision Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load models
# -----------------------------
xgb_model = joblib.load("models/xgb_model.pkl")
bagging_model = joblib.load("models/bagging_model.pkl")

# -----------------------------
# Premium styling
# -----------------------------
st.markdown("""
<style>
    .main {
        background: #0b0f19;
        color: #f4f7fb;
    }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #101827 0%, #0b0f19 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .hero {
        padding: 1.2rem 1.4rem;
        border-radius: 22px;
        background: linear-gradient(135deg, rgba(26,35,58,0.95), rgba(12,18,30,0.95));
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
        letter-spacing: -0.03em;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: rgba(255,255,255,0.72);
        font-size: 0.98rem;
    }
    .card {
        padding: 1rem 1.1rem;
        border-radius: 18px;
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 10px 24px rgba(0,0,0,0.18);
        margin-bottom: 1rem;
    }
    .kpi-label {
        font-size: 0.82rem;
        color: rgba(255,255,255,0.65);
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
    }
    .subtle {
        color: rgba(255,255,255,0.72);
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helpers
# -----------------------------
def build_input(gender, married, dependents, education, self_employed, property_area,
                total_income, loan_amount, loan_term, credit_history):
    return pd.DataFrame({
        "Gender": [str(gender)],
        "Married": [str(married)],
        "Dependents": [str(dependents)],
        "Education": [str(education)],
        "Self_Employed": [str(self_employed)],
        "Property_Area": [str(property_area)],
        "TotalIncome": [float(total_income)],
        "LoanAmount": [float(loan_amount)],
        "Loan_Amount_Term": [float(loan_term)],
        "Credit_History": [float(credit_history)]
    })

def get_selected_model(choice):
    return xgb_model if choice == "XGBoost" else bagging_model

def predict_model(model, df):
    pred = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1])
    return pred, proba

def risk_bucket(prob):
    if prob >= 0.75:
        return "Low Risk", "🟢"
    elif prob >= 0.45:
        return "Medium Risk", "🟡"
    return "High Risk", "🔴"

def make_gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 46}},
        title={"text": "Approval Probability", "font": {"size": 20}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#7dd3fc"},
            "bgcolor": "rgba(0,0,0,0)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 40], "color": "rgba(239,68,68,0.2)"},
                {"range": [40, 70], "color": "rgba(234,179,8,0.2)"},
                {"range": [70, 100], "color": "rgba(34,197,94,0.2)"},
            ],
        }
    ))
    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=55, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def make_donut(prob):
    fig = go.Figure(data=[go.Pie(
        labels=["Approve", "Reject"],
        values=[prob, 1 - prob],
        hole=0.68,
        marker=dict(colors=["#60a5fa", "#1f2937"]),
        textinfo="percent"
    )])
    fig.update_layout(
        height=290,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white")
    )
    return fig

def make_stress_chart(base_prob, stressed_prob):
    df = pd.DataFrame({
        "Scenario": ["Baseline", "Stress"],
        "Probability": [base_prob * 100, stressed_prob * 100]
    })
    fig = px.bar(
        df,
        x="Scenario",
        y="Probability",
        text="Probability",
        title="Baseline vs Stress Scenario",
    )
    fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig.update_layout(
        yaxis_title="Approval Probability (%)",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.02)",
        font=dict(color="white"),
        title_font=dict(size=20)
    )
    return fig

def decision_comment(prob, loan_to_income, credit_history, stress_drop):
    notes = []
    if credit_history == 0:
        notes.append("No prior credit history is a major negative factor.")
    else:
        notes.append("Strong credit history supports approval.")
    if loan_to_income > 0.35:
        notes.append("Loan amount is relatively high versus income.")
    elif loan_to_income < 0.08:
        notes.append("Loan burden looks comfortable versus income.")
    if stress_drop > 0:
        notes.append(f"Stress test assumes income falls by {stress_drop}%, reducing affordability.")
    if prob >= 0.75:
        notes.append("Overall profile is strong.")
    elif prob >= 0.45:
        notes.append("Profile is balanced, but risk monitoring is advisable.")
    else:
        notes.append("Profile indicates elevated credit risk.")
    return notes

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox("Select Model", ["XGBoost", "Bagging"])
stress_pct = st.sidebar.slider("Stress Test: Income drop (%)", 0, 40, 10, 1)

st.sidebar.markdown("---")
st.sidebar.caption("Current setup")
st.sidebar.write(f"Model: **{model_choice}**")
st.sidebar.write(f"Stress: **-{stress_pct}% income**")

model = get_selected_model(model_choice)

# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="hero">
    <h1>Credit Decision Platform</h1>
    <p>Loan approval engine with scenario analysis, risk classification, and executive dashboards.</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Applicant form
# -----------------------------
st.markdown("### Applicant Profile")

with st.form("loan_form"):
    c1, c2 = st.columns(2)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with c2:
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        total_income = st.number_input("Total Income", min_value=0.0, value=5000.0, step=100.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0, value=100.0, step=10.0)
        loan_term = st.number_input("Loan Amount Term", min_value=0.0, value=360.0, step=12.0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0])

    run = st.form_submit_button("Run Credit Decision")

# -----------------------------
# Main output
# -----------------------------
if run:
    input_df = build_input(
        gender, married, dependents, education, self_employed, property_area,
        total_income, loan_amount, loan_term, credit_history
    )

    stressed_df = input_df.copy()
    stressed_df["TotalIncome"] = stressed_df["TotalIncome"] * (1 - stress_pct / 100.0)

    pred, prob = predict_model(model, input_df)
    _, stressed_prob = predict_model(model, stressed_df)

    loan_to_income = loan_amount / max(total_income, 1.0)
    risk_label, risk_emoji = risk_bucket(prob)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Approval Probability", f"{prob * 100:.2f}%")
    c2.metric("Stressed Probability", f"{stressed_prob * 100:.2f}%")
    c3.metric("Stress Impact", f"{(stressed_prob - prob) * 100:.2f}%")
    c4.metric("Loan / Income", f"{loan_to_income:.3f}")

    st.markdown("### Executive Decision")
    if pred == 1:
        st.success(f"✅ Approved by {model_choice}")
    else:
        st.error(f"❌ Rejected by {model_choice}")

    st.info(f"Risk Category: **{risk_emoji} {risk_label}**")

    st.markdown("### Snapshot")
    s1, s2 = st.columns([1, 1])

    with s1:
        st.plotly_chart(make_gauge(prob), use_container_width=True)
    with s2:
        st.plotly_chart(make_donut(prob), use_container_width=True)

    st.markdown("### Applicant Data")
    st.dataframe(input_df, use_container_width=True)

    st.markdown("### Stress Test")
    st.plotly_chart(make_stress_chart(prob, stressed_prob), use_container_width=True)

    st.markdown("### Credit Committee Notes")
    notes = decision_comment(prob, loan_to_income, credit_history, stress_pct)
    for n in notes:
        st.write(f"• {n}")

    st.markdown("### Scenario Table")
    scenario_df = pd.DataFrame({
        "Scenario": ["Base", f"Stress (-{stress_pct}% income)"],
        "TotalIncome": [float(total_income), float(stressed_df["TotalIncome"].iloc[0])],
        "ApprovalProbability": [prob * 100, stressed_prob * 100],
        "Risk": [risk_label, risk_bucket(stressed_prob)[0]]
    })
    st.dataframe(scenario_df, use_container_width=True)

    st.markdown("### Model View")
    st.caption("Use the sidebar to switch between XGBoost and Bagging, then rerun the decision.")