"""
STEP 9: Streamlit Dashboard — Telecom Customer Churn Analysis
Elite redesign: Dark Luxury Data Terminal aesthetic
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

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CHURN//INTEL",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Design System CSS ─────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ─── Reset & Root ─── */
:root {
    --black:      #050507;
    --void:       #0a0a0f;
    --surface-1:  #0f0f18;
    --surface-2:  #141420;
    --surface-3:  #1c1c2e;
    --border:     rgba(255,255,255,0.06);
    --border-hi:  rgba(255,255,255,0.14);

    --acid:       #c8ff00;
    --acid-dim:   rgba(200,255,0,0.12);
    --acid-glow:  rgba(200,255,0,0.06);

    --red:        #ff3a3a;
    --red-dim:    rgba(255,58,58,0.12);
    --green:      #00e676;
    --green-dim:  rgba(0,230,118,0.12);
    --amber:      #ffab00;
    --amber-dim:  rgba(255,171,0,0.12);
    --cyan:       #00cfff;
    --violet:     #a259ff;

    --text-1: rgba(255,255,255,0.92);
    --text-2: rgba(255,255,255,0.55);
    --text-3: rgba(255,255,255,0.28);

    --mono: 'Space Mono', monospace;
    --display: 'Syne', sans-serif;
    --body: 'DM Sans', sans-serif;

    --radius: 4px;
    --radius-lg: 8px;
}

html, body, [class*="css"] {
    font-family: var(--body);
    color: var(--text-1);
}

/* ─── App Background ─── */
.stApp {
    background: var(--void);
    background-image:
        radial-gradient(ellipse 80% 40% at 50% -10%, rgba(200,255,0,0.04) 0%, transparent 60%),
        radial-gradient(ellipse 60% 50% at 90% 80%, rgba(162,89,255,0.04) 0%, transparent 50%);
}
.main .block-container { padding: 0 2rem 4rem 2rem; max-width: 100%; }

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--surface-1);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] .block-container { padding: 2rem 1.2rem; }
[data-testid="stSidebar"] label {
    font-family: var(--mono);
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3) !important;
}
[data-testid="stSidebar"] .stSlider > div > div { color: var(--acid) !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background: var(--acid-dim);
    border: 1px solid var(--acid);
    border-radius: 2px;
}
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] span {
    font-family: var(--mono);
    font-size: 0.65rem;
    color: var(--acid);
}

/* ─── Section Divider ─── */
.rule {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 44px 0 24px 0;
}
.rule-label {
    font-family: var(--mono);
    font-size: 0.62rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--acid);
    white-space: nowrap;
}
.rule-line {
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border-hi), transparent);
}
.rule-num {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-3);
}

/* ─── KPI Cards ─── */
.kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 24px 0; }
.kpi {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 18px 16px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.kpi.acid::before { background: var(--acid); }
.kpi.red::before  { background: var(--red); }
.kpi.cyan::before { background: var(--cyan); }
.kpi.violet::before { background: var(--violet); }
.kpi:hover { border-color: var(--border-hi); }

.kpi-tag {
    font-family: var(--mono);
    font-size: 0.58rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--text-3);
    margin-bottom: 10px;
}
.kpi-val {
    font-family: var(--display);
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1;
    letter-spacing: -0.02em;
    margin-bottom: 6px;
}
.kpi-sub {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-3);
}
.kpi-corner {
    position: absolute;
    top: 14px; right: 14px;
    font-family: var(--mono);
    font-size: 0.55rem;
    color: var(--text-3);
}

/* ─── Metric Cards (model) ─── */
.metric-row { display: flex; flex-direction: column; gap: 10px; }
.metric-card {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.metric-card-label {
    font-family: var(--mono);
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--text-3);
}
.metric-card-val {
    font-family: var(--mono);
    font-size: 1.3rem;
    font-weight: 700;
}

/* ─── Prediction Section ─── */
.pred-panel {
    background: var(--surface-2);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 28px 24px;
    margin-bottom: 16px;
}
.pred-verdict {
    font-family: var(--display);
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    line-height: 1;
}
.pred-verdict.churn { color: var(--red); }
.pred-verdict.retain { color: var(--green); }
.pred-prob {
    font-family: var(--mono);
    font-size: 0.72rem;
    color: var(--text-2);
    margin-top: 10px;
    letter-spacing: 0.08em;
}
.pred-prob span {
    color: var(--text-1);
    font-weight: 700;
}

/* ─── Form Styling ─── */
.stSelectbox label, .stSlider label, .stNumberInput label {
    font-family: var(--mono) !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text-3) !important;
}
.stSelectbox > div > div {
    background: var(--surface-2) !important;
    border: 1px solid var(--border-hi) !important;
    border-radius: var(--radius) !important;
    font-family: var(--body) !important;
    font-size: 0.9rem !important;
}
.stSelectbox > div > div:focus-within {
    border-color: var(--acid) !important;
    box-shadow: 0 0 0 1px var(--acid) !important;
}

/* ─── Submit Button ─── */
.stFormSubmitButton button {
    background: var(--acid) !important;
    color: var(--black) !important;
    font-family: var(--mono) !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: var(--radius) !important;
    padding: 14px 32px !important;
    transition: opacity 0.15s !important;
    width: 100% !important;
}
.stFormSubmitButton button:hover { opacity: 0.88 !important; }

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 2px; }

/* ─── Plotly chart border wrapping ─── */
.plot-wrapper {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--surface-2);
}

/* ─── Sidebar brand tag ─── */
.sidebar-brand {
    font-family: var(--mono);
    font-size: 0.62rem;
    color: var(--text-3);
    letter-spacing: 0.08em;
    padding-bottom: 24px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.sidebar-brand span { color: var(--acid); }

/* ─── HR ─── */
hr { border-color: var(--border) !important; margin: 40px 0 !important; }

/* ─── Status dot ─── */
.dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
</style>
""", unsafe_allow_html=True)

# ── Data & Model ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("TelecomCustomerChurn.csv")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    return df

@st.cache_resource
def train_model(df):
    data = df.drop(columns=["customerID"]).copy()
    data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
    data["AvgMonthlyCharge"] = np.where(data["Tenure"] > 0, data["TotalCharges"] / data["Tenure"], data["MonthlyCharges"])
    service_cols = ["PhoneService","MultipleLines","InternetService","OnlineSecurity",
                    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies"]
    data["NumServices"] = data[service_cols].apply(
        lambda r: sum(1 for v in r if v in ["Yes","Fiber optic","DSL"]), axis=1)
    data["HasInternet"] = (data["InternetService"] != "No").astype(int)
    data["ChargePerService"] = np.where(data["NumServices"] > 0, data["MonthlyCharges"] / data["NumServices"], 0)
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

# ── Shared Plot Config ────────────────────────────────────────────────────────
BG   = "rgba(0,0,0,0)"
FONT = dict(color="rgba(255,255,255,0.75)", family="'Space Mono', monospace", size=11)
GRID = "rgba(255,255,255,0.04)"
BASE_LAYOUT = dict(
    paper_bgcolor=BG, plot_bgcolor=BG,
    font=FONT,
    margin=dict(t=52, b=36, l=40, r=20),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,0.08)"),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="rgba(255,255,255,0.08)"),
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <span>◈ CHURN//INTEL</span> v2.0<br>
        <span style="color:rgba(255,255,255,0.2)">Telecom Analytics Platform</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div style="font-family:var(--mono,monospace);font-size:0.6rem;letter-spacing:0.14em;text-transform:uppercase;color:rgba(255,255,255,0.28);margin-bottom:16px">Data Filters</div>', unsafe_allow_html=True)

    contracts    = st.multiselect("Contract Type", df_raw["Contract"].unique(), default=df_raw["Contract"].unique())
    internet     = st.multiselect("Internet Service", df_raw["InternetService"].unique(), default=df_raw["InternetService"].unique())
    tenure_range = st.slider("Tenure (months)", 0, int(df_raw["Tenure"].max()), (0, int(df_raw["Tenure"].max())))
    charge_range = st.slider("Monthly Charges ($)",
                             float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max()),
                             (float(df_raw["MonthlyCharges"].min()), float(df_raw["MonthlyCharges"].max())))

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:rgba(200,255,0,0.05);border:1px solid rgba(200,255,0,0.15);
                border-radius:4px;padding:14px;font-family:'Space Mono',monospace">
        <div style="font-size:0.55rem;letter-spacing:0.14em;text-transform:uppercase;
                    color:rgba(255,255,255,0.3);margin-bottom:6px">Model Status</div>
        <div style="font-size:0.75rem;color:#c8ff00">
            <span class="dot" style="background:#c8ff00;display:inline-block;width:6px;height:6px;
                  border-radius:50%;margin-right:6px"></span>RF Online
        </div>
        <div style="font-size:0.6rem;color:rgba(255,255,255,0.35);margin-top:4px">
            AUC {model_metrics['auc']:.3f} · ACC {model_metrics['accuracy']:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Filter Data ───────────────────────────────────────────────────────────────
df = df_raw[
    (df_raw["Contract"].isin(contracts)) &
    (df_raw["InternetService"].isin(internet)) &
    (df_raw["Tenure"].between(tenure_range[0], tenure_range[1])) &
    (df_raw["MonthlyCharges"].between(charge_range[0], charge_range[1]))
]

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:32px'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style="display:flex;align-items:flex-end;justify-content:space-between;margin-bottom:8px">
    <div>
        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;
                    text-transform:uppercase;color:#c8ff00;margin-bottom:8px">
            ◈ Telecom Intelligence System
        </div>
        <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;
                    letter-spacing:-0.04em;line-height:0.95;color:rgba(255,255,255,0.95)">
            Customer<br>Churn<span style="color:#c8ff00">.</span>
        </div>
    </div>
    <div style="text-align:right;font-family:'Space Mono',monospace">
        <div style="font-size:0.58rem;color:rgba(255,255,255,0.25);letter-spacing:0.08em">DATASET</div>
        <div style="font-size:1.6rem;font-weight:700;color:rgba(255,255,255,0.9)">{len(df):,}</div>
        <div style="font-size:0.58rem;color:rgba(255,255,255,0.25)">of {len(df_raw):,} records</div>
    </div>
</div>
<div style="height:1px;background:linear-gradient(90deg,#c8ff00 0%,rgba(255,255,255,0.06) 40%,transparent 100%);
            margin-bottom:28px"></div>
""", unsafe_allow_html=True)

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
churn_rate = (df["Churn"] == "Yes").mean() * 100
avg_charges = df["MonthlyCharges"].mean()
avg_tenure  = df["Tenure"].mean()
total_rev   = df["TotalCharges"].sum()
n_churned   = (df["Churn"] == "Yes").sum()

churn_color  = "red" if churn_rate > 30 else "amber" if churn_rate > 20 else "acid"
churn_hex    = "#ff3a3a" if churn_rate > 30 else "#ffab00" if churn_rate > 20 else "#c8ff00"

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi {churn_color}">
        <div class="kpi-corner">01</div>
        <div class="kpi-tag">Churn Rate</div>
        <div class="kpi-val" style="color:{churn_hex}">{churn_rate:.1f}<span style="font-size:1.2rem">%</span></div>
        <div class="kpi-sub">{n_churned:,} / {len(df):,} customers</div>
    </div>
    <div class="kpi cyan">
        <div class="kpi-corner">02</div>
        <div class="kpi-tag">Avg Monthly Charges</div>
        <div class="kpi-val" style="color:#00cfff">${avg_charges:.0f}<span style="font-size:1.2rem">.{int(round((avg_charges%1)*100)):02d}</span></div>
        <div class="kpi-sub">per customer / month</div>
    </div>
    <div class="kpi violet">
        <div class="kpi-corner">03</div>
        <div class="kpi-tag">Avg Tenure</div>
        <div class="kpi-val" style="color:#a259ff">{avg_tenure:.1f}<span style="font-size:1.2rem">mo</span></div>
        <div class="kpi-sub">customer lifetime</div>
    </div>
    <div class="kpi acid">
        <div class="kpi-corner">04</div>
        <div class="kpi-tag">Lifetime Revenue</div>
        <div class="kpi-val" style="color:#c8ff00">${total_rev/1_000_000:.2f}<span style="font-size:1.2rem">M</span></div>
        <div class="kpi-sub">total charges billed</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── SECTION 01: CHURN OVERVIEW ────────────────────────────────────────────────
st.markdown("""
<div class="rule">
    <span class="rule-label">Churn Overview</span>
    <span class="rule-line"></span>
    <span class="rule-num">01 / 04</span>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])

with col1:
    churn_counts = df["Churn"].value_counts()
    retained = churn_counts.get("No", 0)
    churned  = churn_counts.get("Yes", 0)
    fig = go.Figure(go.Pie(
        labels=["Retained", "Churned"],
        values=[retained, churned],
        hole=0.72,
        marker=dict(
            colors=["rgba(0,230,118,0.85)", "rgba(255,58,58,0.85)"],
            line=dict(color=["#050507"], width=3)
        ),
        textinfo="none",
        hovertemplate="<b>%{label}</b><br>%{value:,} customers<br>%{percent}<extra></extra>"
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Churn Split", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        showlegend=True,
        legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center",
                    font=dict(size=10, family="'Space Mono',monospace")),
        height=340,
        annotations=[
            dict(text=f"<b>{churn_rate:.1f}%</b>", x=0.5, y=0.55,
                 font=dict(size=30, color="#ff3a3a", family="'Syne',sans-serif"), showarrow=False),
            dict(text="CHURN", x=0.5, y=0.36,
                 font=dict(size=9, color="rgba(255,255,255,0.3)", family="'Space Mono',monospace"), showarrow=False),
        ]
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    ct = df.groupby("Contract")["Churn"].value_counts().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    retained_pct = ct_pct.get("No", pd.Series([0]*len(ct), index=ct.index))
    churned_pct  = ct_pct.get("Yes", pd.Series([0]*len(ct), index=ct.index))
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Retained", x=ct.index, y=retained_pct,
        marker=dict(color="rgba(0,230,118,0.7)", line=dict(width=0)),
        text=[f"{v:.0f}%" for v in retained_pct], textposition="inside",
        textfont=dict(size=10, family="'Space Mono',monospace", color="rgba(0,0,0,0.7)")
    ))
    fig.add_trace(go.Bar(
        name="Churned", x=ct.index, y=churned_pct,
        marker=dict(color="rgba(255,58,58,0.75)", line=dict(width=0)),
        text=[f"{v:.0f}%" for v in churned_pct], textposition="inside",
        textfont=dict(size=10, family="'Space Mono',monospace", color="rgba(255,255,255,0.9)")
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        barmode="stack",
        title=dict(text="Churn Rate by Contract Type", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=340,
        legend=dict(orientation="h", y=1.1, font=dict(size=10, family="'Space Mono',monospace")),
        bargap=0.35,
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 02: SEGMENTATION ──────────────────────────────────────────────────
st.markdown("""
<div class="rule">
    <span class="rule-label">Customer Segmentation</span>
    <span class="rule-line"></span>
    <span class="rule-num">02 / 04</span>
</div>
""", unsafe_allow_html=True)

col3, col4 = st.columns(2)

with col3:
    fig = px.histogram(
        df, x="Tenure", color="Churn", nbins=36,
        color_discrete_map={"Yes": "rgba(255,58,58,0.7)", "No": "rgba(0,230,118,0.5)"},
        barmode="overlay", opacity=0.85
    )
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Tenure Distribution by Churn Status", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=360,
        legend=dict(orientation="h", y=1.12, font=dict(size=10, family="'Space Mono',monospace")),
        bargap=0.05,
    )
    st.plotly_chart(fig, use_container_width=True)

with col4:
    fig = px.box(
        df, x="Churn", y="MonthlyCharges", color="Churn",
        color_discrete_map={"Yes": "rgba(255,58,58,0.8)", "No": "rgba(0,230,118,0.8)"},
        points="outliers"
    )
    fig.update_traces(marker=dict(size=3, opacity=0.5))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Monthly Charges Distribution", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=360,
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 03: SERVICE & PAYMENT ────────────────────────────────────────────
st.markdown("""
<div class="rule">
    <span class="rule-label">Service & Payment Analysis</span>
    <span class="rule-line"></span>
    <span class="rule-num">03 / 04</span>
</div>
""", unsafe_allow_html=True)

col5, col6 = st.columns(2)

with col5:
    inet = df.groupby("InternetService")["Churn"].apply(lambda x: (x=="Yes").mean()*100).sort_values()
    clrs = ["#ff3a3a" if v > 30 else "#ffab00" if v > 20 else "#c8ff00" for v in inet.values]
    fig = go.Figure(go.Bar(
        x=inet.values, y=inet.index, orientation="h",
        marker=dict(color=clrs, line=dict(width=0)),
        text=[f"  {v:.1f}%" for v in inet.values], textposition="outside",
        textfont=dict(size=10, family="'Space Mono',monospace", color="rgba(255,255,255,0.6)")
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Churn Rate by Internet Service", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=300, bargap=0.45
    )
    fig.update_xaxes(range=[0, inet.max()*1.3], gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

with col6:
    pay = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x=="Yes").mean()*100).sort_values()
    clrs2 = ["#ff3a3a" if v > 30 else "#ffab00" if v > 20 else "#c8ff00" for v in pay.values]
    fig = go.Figure(go.Bar(
        x=pay.values, y=pay.index, orientation="h",
        marker=dict(color=clrs2, line=dict(width=0)),
        text=[f"  {v:.1f}%" for v in pay.values], textposition="outside",
        textfont=dict(size=10, family="'Space Mono',monospace", color="rgba(255,255,255,0.6)")
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Churn Rate by Payment Method", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=300, bargap=0.45
    )
    fig.update_xaxes(range=[0, pay.max()*1.35], gridcolor=GRID, zeroline=False)
    st.plotly_chart(fig, use_container_width=True)

# ── SECTION 04: MODEL INSIGHTS ────────────────────────────────────────────────
st.markdown("""
<div class="rule">
    <span class="rule-label">Model Insights — Random Forest</span>
    <span class="rule-line"></span>
    <span class="rule-num">04 / 04</span>
</div>
""", unsafe_allow_html=True)

col7, col8 = st.columns([3, 1])

with col7:
    top10 = feat_imp.head(10).sort_values()
    # Color bar by rank (brightest = most important)
    bar_colors = [f"rgba(200,255,0,{0.35 + 0.65*(i/9):.2f})" for i in range(len(top10))]
    fig = go.Figure(go.Bar(
        x=top10.values, y=top10.index, orientation="h",
        marker=dict(color=bar_colors, line=dict(width=0)),
        text=[f"  {v:.4f}" for v in top10.values], textposition="outside",
        textfont=dict(size=9, family="'Space Mono',monospace", color="rgba(255,255,255,0.5)")
    ))
    fig.update_layout(
        **BASE_LAYOUT,
        title=dict(text="Top 10 Feature Importances", font=dict(size=12, color="rgba(255,255,255,0.45)", family="'Space Mono',monospace")),
        height=400, bargap=0.3
    )
    fig.update_xaxes(range=[0, top10.max()*1.3], gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)

with col8:
    acc_pct = model_metrics['accuracy'] * 100
    f1_val  = model_metrics['f1']
    auc_val = model_metrics['auc']
    st.markdown(f"""
    <div class="metric-row" style="margin-top:52px">
        <div class="metric-card">
            <div>
                <div class="metric-card-label">Accuracy</div>
                <div class="metric-card-val" style="color:#c8ff00">{acc_pct:.1f}%</div>
            </div>
            <div style="font-size:1.4rem">◎</div>
        </div>
        <div class="metric-card">
            <div>
                <div class="metric-card-label">F1 Score</div>
                <div class="metric-card-val" style="color:#00cfff">{f1_val:.4f}</div>
            </div>
            <div style="font-size:1.4rem">◎</div>
        </div>
        <div class="metric-card">
            <div>
                <div class="metric-card-label">AUC-ROC</div>
                <div class="metric-card-val" style="color:#a259ff">{auc_val:.4f}</div>
            </div>
            <div style="font-size:1.4rem">◎</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── PREDICTION ENGINE ─────────────────────────────────────────────────────────
st.markdown("""
<div style="height:20px"></div>
<div style="height:1px;background:linear-gradient(90deg,rgba(255,255,255,0.06),transparent)"></div>
<div style="height:40px"></div>
""", unsafe_allow_html=True)

st.markdown("""
<div style="font-family:'Space Mono',monospace;font-size:0.6rem;letter-spacing:0.2em;
            text-transform:uppercase;color:#c8ff00;margin-bottom:8px">
    ◈ Inference Engine
</div>
<div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;
            letter-spacing:-0.02em;margin-bottom:4px">
    Predict Churn Risk
</div>
<div style="font-family:'DM Sans',sans-serif;font-size:0.88rem;color:rgba(255,255,255,0.4);
            margin-bottom:28px">
    Enter customer profile parameters to generate a real-time churn probability score.
</div>
""", unsafe_allow_html=True)

with st.form("prediction_form"):
    pc1, pc2, pc3, pc4 = st.columns(4)
    with pc1:
        st.markdown('<div style="font-family:monospace;font-size:0.6rem;letter-spacing:0.14em;color:rgba(255,255,255,0.2);text-transform:uppercase;margin-bottom:12px">— Demographics</div>', unsafe_allow_html=True)
        p_gender     = st.selectbox("Gender", ["Male","Female"])
        p_senior     = st.selectbox("Senior Citizen", ["No","Yes"])
        p_partner    = st.selectbox("Partner", ["No","Yes"])
        p_dependents = st.selectbox("Dependents", ["No","Yes"])
    with pc2:
        st.markdown('<div style="font-family:monospace;font-size:0.6rem;letter-spacing:0.14em;color:rgba(255,255,255,0.2);text-transform:uppercase;margin-bottom:12px">— Account</div>', unsafe_allow_html=True)
        p_tenure   = st.slider("Tenure (months)", 0, 72, 12)
        p_monthly  = st.number_input("Monthly Charges ($)", 18.0, 120.0, 50.0, 5.0)
        p_total    = st.number_input("Total Charges ($)", 0.0, 9000.0, float(p_tenure * p_monthly), 50.0)
        p_phone    = st.selectbox("Phone Service", ["Yes","No"])
    with pc3:
        st.markdown('<div style="font-family:monospace;font-size:0.6rem;letter-spacing:0.14em;color:rgba(255,255,255,0.2);text-transform:uppercase;margin-bottom:12px">— Services</div>', unsafe_allow_html=True)
        p_internet   = st.selectbox("Internet Service", ["DSL","Fiber optic","No"])
        p_security   = st.selectbox("Online Security", ["Yes","No"])
        p_backup     = st.selectbox("Online Backup", ["Yes","No"])
        p_protection = st.selectbox("Device Protection", ["Yes","No"])
    with pc4:
        st.markdown('<div style="font-family:monospace;font-size:0.6rem;letter-spacing:0.14em;color:rgba(255,255,255,0.2);text-transform:uppercase;margin-bottom:12px">— Billing</div>', unsafe_allow_html=True)
        p_contract  = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        p_payment   = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
        p_paperless = st.selectbox("Paperless Billing", ["Yes","No"])
        p_techsup   = st.selectbox("Tech Support", ["Yes","No"])

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("▶  Run Prediction", use_container_width=True)

if submitted:
    inp = {
        "Gender": 1 if p_gender == "Male" else 0,
        "SeniorCitizen": 1 if p_senior == "Yes" else 0,
        "Partner": 1 if p_partner == "Yes" else 0,
        "Dependents": 1 if p_dependents == "Yes" else 0,
        "Tenure": p_tenure,
        "PhoneService": 1 if p_phone == "Yes" else 0,
        "MultipleLines": 0,
        "OnlineSecurity": 1 if p_security == "Yes" else 0,
        "OnlineBackup": 1 if p_backup == "Yes" else 0,
        "DeviceProtection": 1 if p_protection == "Yes" else 0,
        "TechSupport": 1 if p_techsup == "Yes" else 0,
        "StreamingTV": 0, "StreamingMovies": 0,
        "PaperlessBilling": 1 if p_paperless == "Yes" else 0,
        "MonthlyCharges": p_monthly, "TotalCharges": p_total,
        "AvgMonthlyCharge": p_total / p_tenure if p_tenure > 0 else p_monthly,
    }
    svc_count = sum([inp["PhoneService"], 1 if p_internet != "No" else 0,
                     inp["OnlineSecurity"], inp["OnlineBackup"], inp["DeviceProtection"], inp["TechSupport"]])
    inp.update({
        "NumServices": svc_count,
        "HasInternet": 1 if p_internet != "No" else 0,
        "ChargePerService": p_monthly / svc_count if svc_count > 0 else 0,
        "InternetService_Fiber optic": 1 if p_internet == "Fiber optic" else 0,
        "InternetService_No": 1 if p_internet == "No" else 0,
        "Contract_One year": 1 if p_contract == "One year" else 0,
        "Contract_Two year": 1 if p_contract == "Two year" else 0,
        "PaymentMethod_Credit card (automatic)": 1 if p_payment == "Credit card (automatic)" else 0,
        "PaymentMethod_Electronic check": 1 if p_payment == "Electronic check" else 0,
        "PaymentMethod_Mailed check": 1 if p_payment == "Mailed check" else 0,
    })
    input_df = pd.DataFrame([inp])
    for c in feature_cols:
        if c not in input_df.columns:
            input_df[c] = 0
    input_df = input_df[feature_cols]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    prob = model.predict_proba(input_df)[0][1]
    pred = "CHURN" if prob >= 0.5 else "RETAIN"

    risk_label  = "HIGH" if prob >= 0.6 else "MEDIUM" if prob >= 0.35 else "LOW"
    risk_color  = "#ff3a3a" if prob >= 0.6 else "#ffab00" if prob >= 0.35 else "#c8ff00"
    verdict_cls = "churn" if pred == "CHURN" else "retain"

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    r1, r2 = st.columns([1, 2])
    with r1:
        st.markdown(f"""
        <div class="pred-panel">
            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;letter-spacing:0.16em;
                        text-transform:uppercase;color:rgba(255,255,255,0.25);margin-bottom:14px">
                Prediction Output
            </div>
            <div class="pred-verdict {verdict_cls}">{pred}</div>
            <div class="pred-prob" style="margin-top:12px">
                Churn probability: <span>{prob:.1%}</span>
            </div>
            <div style="margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06)">
                <span style="font-family:'Space Mono',monospace;font-size:0.6rem;
                             letter-spacing:0.1em;text-transform:uppercase;
                             color:rgba(255,255,255,0.3)">Risk Level&nbsp;&nbsp;</span>
                <span style="font-family:'Space Mono',monospace;font-size:0.7rem;
                             font-weight:700;color:{risk_color}">{risk_label}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with r2:
        # Needle gauge — clean and sharp
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(prob * 100, 1),
            number=dict(suffix="%", font=dict(size=36, color="rgba(255,255,255,0.85)", family="'Syne',sans-serif")),
            gauge=dict(
                axis=dict(range=[0, 100], tickfont=dict(size=9, color="rgba(255,255,255,0.3)", family="'Space Mono',monospace"), nticks=6),
                bar=dict(color=risk_color, thickness=0.18),
                bgcolor="rgba(255,255,255,0.03)",
                bordercolor="transparent",
                steps=[
                    dict(range=[0, 35], color="rgba(200,255,0,0.08)"),
                    dict(range=[35, 60], color="rgba(255,171,0,0.08)"),
                    dict(range=[60, 100], color="rgba(255,58,58,0.1)"),
                ],
                threshold=dict(line=dict(color="rgba(255,255,255,0.4)", width=2), value=50)
            ),
            title=dict(text="Churn Risk Score", font=dict(size=11, color="rgba(255,255,255,0.3)", family="'Space Mono',monospace"))
        ))
        fig.update_layout(
            paper_bgcolor="rgba(20,20,32,0.95)",
            font=dict(color="white"),
            height=260,
            margin=dict(t=60, b=20, l=40, r=40),
        )
        st.plotly_chart(fig, use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="height:60px"></div>
<div style="height:1px;background:linear-gradient(90deg,rgba(255,255,255,0.06),transparent)"></div>
<div style="display:flex;justify-content:space-between;align-items:center;
            padding:20px 0;font-family:'Space Mono',monospace;font-size:0.58rem;
            color:rgba(255,255,255,0.18);letter-spacing:0.08em">
    <span>◈ CHURN//INTEL — Telecom Analytics Platform</span>
    <span>Random Forest · Streamlit · Plotly</span>
</div>
""", unsafe_allow_html=True)