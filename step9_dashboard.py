"""
STEP 9: Streamlit Dashboard - Telecom Customer Churn Analysis
Run: streamlit run step9_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.main { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }

.stApp { background: linear-gradient(135deg, #0f0c29 0%, #1a1a3e 50%, #24243e 100%); }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a3e 0%, #0f0c29 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

.kpi-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 16px;
    padding: 24px 20px;
    text-align: center;
    backdrop-filter: blur(10px);
    transition: transform 0.3s, box-shadow 0.3s;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 8px 32px rgba(0,0,0,0.3); }
.kpi-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
.kpi-label { font-size: 0.85rem; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px; }
.kpi-delta { font-size: 0.8rem; margin-top: 4px; }

.header-title {
    font-size: 2.4rem; font-weight: 700;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.header-sub { color: rgba(255,255,255,0.5); font-size: 1rem; margin-top: 0; }

.section-header {
    font-size: 1.3rem; font-weight: 600; color: #fff;
    border-left: 4px solid #667eea; padding-left: 12px;
    margin: 30px 0 16px 0;
}

.prediction-box {
    background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
    border: 1px solid rgba(102,126,234,0.3);
    border-radius: 16px; padding: 24px; margin: 16px 0;
}
.pred-churn { color: #ff6b6b; font-size: 2rem; font-weight: 700; }
.pred-safe  { color: #51cf66; font-size: 2rem; font-weight: 700; }

div[data-testid="stMetricValue"] { font-size: 1.8rem; }

.stSelectbox > div > div { background: rgba(255,255,255,0.05); border-color: rgba(255,255,255,0.15); }
</style>
""", unsafe_allow_html=True)

# ── Load & Cache Data ────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("TelecomCustomerChurn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

@st.cache_resource
def train_model(df):
    """Prepare features and train Random Forest model."""
    data = df.drop(columns=["customerID"]).copy()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})

    # Derived features
    data["AvgMonthlyCharge"] = np.where(data["Tenure"] > 0, data["TotalCharges"] / data["Tenure"], data["MonthlyCharges"])
    service_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    data["NumServices"] = data[service_cols].apply(
        lambda r: sum(1 for v in r if v in ["Yes","Fiber optic","DSL"]), axis=1)
    data["HasInternet"] = (data["InternetService"] != "No").astype(int)
    data["ChargePerService"] = np.where(data["NumServices"] > 0, data["MonthlyCharges"] / data["NumServices"], 0)

    # Encode
    binary_map = {"Yes":1,"No":0,"Male":1,"Female":0}
    for col in ["Gender","Partner","Dependents","PhoneService","PaperlessBilling"]:
        data[col] = data[col].map(binary_map)
    for col in ["MultipleLines","OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]:
        data[col] = data[col].map({"Yes":1,"No":0})
    data = pd.get_dummies(data, columns=["InternetService","Contract","PaymentMethod"], drop_first=True, dtype=int)

    X = data.drop(columns=["Churn"])
    y = data["Churn"]

    scaler = StandardScaler()
    num_cols = ["Tenure","MonthlyCharges","TotalCharges","AvgMonthlyCharge","NumServices","ChargePerService"]
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight="balanced", n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {"accuracy": accuracy_score(y_test, y_pred), "f1": f1_score(y_test, y_pred), "auc": roc_auc_score(y_test, y_prob)}

    feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    return model, scaler, X.columns.tolist(), num_cols, metrics, feat_imp

df_raw = load_data()
model, scaler, feature_cols, num_cols, model_metrics, feat_imp = train_model(df_raw)

# ── Sidebar Filters ──────────────────────────────────────────
st.sidebar.markdown("## Filters")
contracts = st.sidebar.multiselect("Contract Type", df_raw["Contract"].unique(), default=df_raw["Contract"].unique())
internet = st.sidebar.multiselect("Internet Service", df_raw["InternetService"].unique(), default=df_raw["InternetService"].unique())
tenure_range = st.sidebar.slider("Tenure (months)", 0, int(df_raw["Tenure"].max()), (0, int(df_raw["Tenure"].max())))
charge_range = st.sidebar.slider("Monthly Charges ($)", float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max()),
                                  (float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max())))

df = df_raw[
    (df_raw["Contract"].isin(contracts)) &
    (df_raw["InternetService"].isin(internet)) &
    (df_raw["Tenure"].between(tenure_range[0], tenure_range[1])) &
    (df_raw["MonthlyCharges"].between(charge_range[0], charge_range[1]))
]

# ── Header ───────────────────────────────────────────────────
st.markdown('<p class="header-title">Telecom Customer Churn Analytics</p>', unsafe_allow_html=True)
st.markdown(f'<p class="header-sub">Analyzing {len(df):,} customers | Filtered from {len(df_raw):,} total</p>', unsafe_allow_html=True)

# ── KPI Cards ────────────────────────────────────────────────
churn_rate = (df["Churn"] == "Yes").mean() * 100
avg_charges = df["MonthlyCharges"].mean()
avg_tenure = df["Tenure"].mean()
total_rev = df["TotalCharges"].sum()

c1, c2, c3, c4 = st.columns(4)
with c1:
    color = "#ff6b6b" if churn_rate > 30 else "#ffd93d" if churn_rate > 20 else "#51cf66"
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Churn Rate</div>
        <div class="kpi-value" style="color:{color}">{churn_rate:.1f}%</div>
        <div class="kpi-delta">{(df["Churn"]=="Yes").sum():,} of {len(df):,}</div>
    </div>""", unsafe_allow_html=True)
with c2:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Avg Monthly Charges</div>
        <div class="kpi-value" style="color:#667eea">${avg_charges:.2f}</div>
        <div class="kpi-delta">per customer</div>
    </div>""", unsafe_allow_html=True)
with c3:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Avg Tenure</div>
        <div class="kpi-value" style="color:#f093fb">{avg_tenure:.1f} mo</div>
        <div class="kpi-delta">customer lifetime</div>
    </div>""", unsafe_allow_html=True)
with c4:
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">Total Revenue</div>
        <div class="kpi-value" style="color:#51cf66">${total_rev:,.0f}</div>
        <div class="kpi-delta">lifetime value</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Plot Theme ───────────────────────────────────────────────
plot_layout = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="white", family="Inter"), margin=dict(t=50, b=40, l=40, r=20),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
)

# ── Row 1: Churn Distribution + Contract Analysis ───────────
st.markdown('<div class="section-header">Churn Overview</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    churn_counts = df["Churn"].value_counts()
    fig = go.Figure(go.Pie(
        labels=churn_counts.index, values=churn_counts.values,
        hole=0.55, marker=dict(colors=["#51cf66","#ff6b6b"]),
        textinfo="label+percent", textfont=dict(size=14, color="white"),
        hovertemplate="<b>%{label}</b><br>Count: %{value:,}<br>Share: %{percent}<extra></extra>"
    ))
    fig.update_layout(**plot_layout, title="Churn Distribution", showlegend=False, height=380)
    fig.add_annotation(text=f"{churn_rate:.1f}%", x=0.5, y=0.5, font=dict(size=28, color="#ff6b6b"), showarrow=False)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ct = df.groupby("Contract")["Churn"].value_counts().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Retained", x=ct.index, y=ct_pct.get("No", [0]*len(ct)), marker_color="#51cf66", text=ct_pct.get("No", [0]*len(ct)).round(1).astype(str) + "%", textposition="inside"))
    fig.add_trace(go.Bar(name="Churned", x=ct.index, y=ct_pct.get("Yes", [0]*len(ct)), marker_color="#ff6b6b", text=ct_pct.get("Yes", [0]*len(ct)).round(1).astype(str) + "%", textposition="inside"))
    fig.update_layout(**plot_layout, barmode="stack", title="Churn Rate by Contract Type", height=380, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

# ── Row 2: Tenure + Monthly Charges ─────────────────────────
st.markdown('<div class="section-header">Customer Segmentation</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(df, x="Tenure", color="Churn", nbins=36,
                       color_discrete_map={"Yes":"#ff6b6b","No":"#51cf66"},
                       title="Tenure Distribution by Churn", barmode="overlay", opacity=0.7)
    fig.update_layout(**plot_layout, height=380, legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.box(df, x="Churn", y="MonthlyCharges", color="Churn",
                 color_discrete_map={"Yes":"#ff6b6b","No":"#51cf66"},
                 title="Monthly Charges: Churned vs Retained")
    fig.update_layout(**plot_layout, height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 3: Internet + Payment ───────────────────────────────
st.markdown('<div class="section-header">Service & Payment Analysis</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    inet = df.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()*100).sort_values(ascending=True)
    colors_inet = ["#ff6b6b" if v > 30 else "#ffd93d" if v > 20 else "#51cf66" for v in inet.values]
    fig = go.Figure(go.Bar(x=inet.values, y=inet.index, orientation="h",
                           marker_color=colors_inet, text=[f"{v:.1f}%" for v in inet.values], textposition="outside"))
    fig.update_layout(**plot_layout, title="Churn Rate by Internet Service", height=340)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    pay = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x=="Yes").mean()*100).sort_values(ascending=True)
    colors_pay = ["#ff6b6b" if v > 30 else "#ffd93d" if v > 20 else "#51cf66" for v in pay.values]
    fig = go.Figure(go.Bar(x=pay.values, y=pay.index, orientation="h",
                           marker_color=colors_pay, text=[f"{v:.1f}%" for v in pay.values], textposition="outside"))
    fig.update_layout(**plot_layout, title="Churn Rate by Payment Method", height=340)
    st.plotly_chart(fig, use_container_width=True)

# ── Row 4: Feature Importance + Model Metrics ───────────────
st.markdown('<div class="section-header">Model Insights (Random Forest)</div>', unsafe_allow_html=True)
col7, col8 = st.columns([3, 2])

with col7:
    top10 = feat_imp.head(10).sort_values()
    fig = go.Figure(go.Bar(
        x=top10.values, y=top10.index, orientation="h",
        marker=dict(color=top10.values, colorscale="Viridis"),
        text=[f"{v:.4f}" for v in top10.values], textposition="outside"
    ))
    fig.update_layout(**plot_layout, title="Top 10 Feature Importances", height=420)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div class="kpi-card" style="margin-bottom:12px">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value" style="color:#51cf66">{model_metrics['accuracy']:.1%}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="kpi-card" style="margin-bottom:12px">
        <div class="kpi-label">F1 Score</div>
        <div class="kpi-value" style="color:#667eea">{model_metrics['f1']:.4f}</div>
    </div>""", unsafe_allow_html=True)
    st.markdown(f"""<div class="kpi-card">
        <div class="kpi-label">AUC-ROC</div>
        <div class="kpi-value" style="color:#f093fb">{model_metrics['auc']:.4f}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CHURN PREDICTION SECTION
# ══════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-header">Predict Churn for a New Customer</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        p_gender = st.selectbox("Gender", ["Male","Female"])
        p_senior = st.selectbox("Senior Citizen", ["No","Yes"])
        p_partner = st.selectbox("Partner", ["No","Yes"])
        p_dependents = st.selectbox("Dependents", ["No","Yes"])
    with pc2:
        p_tenure = st.slider("Tenure (months)", 0, 72, 12)
        p_monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0, 5.0)
        p_total = st.number_input("Total Charges ($)", 0.0, 9000.0, p_tenure * p_monthly, 50.0)
        p_phone = st.selectbox("Phone Service", ["Yes","No"])
    with pc3:
        p_internet = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        p_security = st.selectbox("Online Security", ["Yes","No"])
        p_backup = st.selectbox("Online Backup", ["Yes","No"])
        p_protection = st.selectbox("Device Protection", ["Yes","No"])
    with pc4:
        p_contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        p_payment = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        p_paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        p_techsup = st.selectbox("Tech Support", ["Yes","No"])

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if submitted:
    # Build feature vector matching training columns
    inp = {}
    inp["Gender"] = 1 if p_gender == "Male" else 0
    inp["SeniorCitizen"] = 1 if p_senior == "Yes" else 0
    inp["Partner"] = 1 if p_partner == "Yes" else 0
    inp["Dependents"] = 1 if p_dependents == "Yes" else 0
    inp["Tenure"] = p_tenure
    inp["PhoneService"] = 1 if p_phone == "Yes" else 0
    inp["MultipleLines"] = 0
    inp["OnlineSecurity"] = 1 if p_security == "Yes" else 0
    inp["OnlineBackup"] = 1 if p_backup == "Yes" else 0
    inp["DeviceProtection"] = 1 if p_protection == "Yes" else 0
    inp["TechSupport"] = 1 if p_techsup == "Yes" else 0
    inp["StreamingTV"] = 0
    inp["StreamingMovies"] = 0
    inp["PaperlessBilling"] = 1 if p_paperless == "Yes" else 0
    inp["MonthlyCharges"] = p_monthly
    inp["TotalCharges"] = p_total
    inp["AvgMonthlyCharge"] = p_total / p_tenure if p_tenure > 0 else p_monthly
    svc_count = sum([inp["PhoneService"], 1 if p_internet != "No" else 0,
                     inp["OnlineSecurity"], inp["OnlineBackup"],
                     inp["DeviceProtection"], inp["TechSupport"]])
    inp["NumServices"] = svc_count
    inp["HasInternet"] = 1 if p_internet != "No" else 0
    inp["ChargePerService"] = p_monthly / svc_count if svc_count > 0 else 0
    # One-hot encoded columns
    inp["InternetService_Fiber optic"] = 1 if p_internet == "Fiber optic" else 0
    inp["InternetService_No"] = 1 if p_internet == "No" else 0
    inp["Contract_One year"] = 1 if p_contract == "One year" else 0
    inp["Contract_Two year"] = 1 if p_contract == "Two year" else 0
    inp["PaymentMethod_Credit card (automatic)"] = 1 if p_payment == "Credit card (automatic)" else 0
    inp["PaymentMethod_Electronic check"] = 1 if p_payment == "Electronic check" else 0
    inp["PaymentMethod_Mailed check"] = 1 if p_payment == "Mailed check" else 0

    input_df = pd.DataFrame([inp])
    # Ensure all model columns exist
    for c in feature_cols:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[feature_cols]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prob = model.predict_proba(input_df)[0][1]
    pred = "CHURN" if prob >= 0.5 else "RETAIN"

    r1, r2 = st.columns([1, 2])
    with r1:
        cls = "pred-churn" if pred == "CHURN" else "pred-safe"
        icon = "Warning" if pred == "CHURN" else "Safe"
        st.markdown(f"""<div class="prediction-box" style="text-align:center">
            <div class="{cls}">{icon}: {pred}</div>
            <div style="color:rgba(255,255,255,0.7);font-size:1.1rem;margin-top:8px">
                Churn Probability: <b>{prob:.1%}</b>
            </div>
        </div>""", unsafe_allow_html=True)
    with r2:
        # Risk gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number=dict(suffix="%"),
            gauge=dict(axis=dict(range=[0, 100]), bar=dict(color="#667eea"),
                       steps=[dict(range=[0,30], color="#51cf66"),
                              dict(range=[30,60], color="#ffd93d"),
                              dict(range=[60,100], color="#ff6b6b")],
                       threshold=dict(line=dict(color="white", width=3), value=50)),
            title=dict(text="Churn Risk Score", font=dict(size=16))
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"), height=250, margin=dict(t=60,b=20,l=40,r=40))
        st.plotly_chart(fig, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────
st.markdown("---")
st.markdown("""<div style="text-align:center;color:rgba(255,255,255,0.3);font-size:0.8rem;padding:10px">
    Telecom Customer Churn Analytics Dashboard | Built with Streamlit + Plotly | Data Science Project
</div>""", unsafe_allow_html=True)
