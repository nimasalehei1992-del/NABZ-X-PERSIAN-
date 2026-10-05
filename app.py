import streamlit as st
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="ظ†ط¨ط¶ ط§غŒع©ط³",
    page_icon="âڑ،",
    layout="wide",
)

# ---------- ط¸ط§ظ‡ط± ظˆ ظپظˆظ†طھ ظپط§ط±ط³غŒ ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"], .stApp, .stMarkdown,
    .stButton, input, textarea, select {
        font-family: 'Vazirmatn', Tahoma, sans-serif !important;
    }

    .stApp {
        direction: rtl;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        direction: rtl;
    }

    [data-testid="stSidebar"] {
        direction: rtl;
    }

    .hero {
        padding: 26px 30px;
        border-radius: 20px;
        margin-bottom: 22px;
        background: linear-gradient(135deg, #071a33, #0d3b66 55%, #1769aa);
        color: white;
    }

    .hero h1 {
        margin: 0 0 8px 0;
        font-size: 34px;
        font-weight: 800;
    }

    .hero p {
        margin: 0;
        font-size: 17px;
    }

    .demo {
        padding: 12px 16px;
        border-radius: 12px;
        background: #fff4e5;
        border-right: 5px solid #f59e0b;
        margin-bottom: 20px;
    }

    .card {
        border: 1px solid #dfe7ef;
        border-radius: 16px;
        padding: 18px;
        background: white;
        box-shadow: 0 5px 18px rgba(20, 50, 80, .06);
    }

    .warning {
        padding: 18px;
        border-radius: 16px;
        background: #fff7ed;
        border-right: 5px solid #f97316;
    }

    .action {
        padding: 18px;
        border-radius: 16px;
        background: #eff6ff;
        border-right: 5px solid #2563eb;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- ط¹ظ†ظˆط§ظ† ----------
st.markdown(
    """
    <div class="hero">
        <h1>ًں«€ ظ†ط¨ط¶ ط§غŒع©ط³</h1>
        <p>ظ…ظˆطھظˆط± ظ‡ظˆط´ظ…ظ†ط¯ ظ¾غŒط´â€Œط¨غŒظ†غŒ ظˆ ظ…ط¯ط§ط®ظ„ظ‡ ظ¾غŒط´ع¯غŒط±ط§ظ†ظ‡ ط±غŒط³ع© ط§ط¹طھط¨ط§ط±غŒ</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="demo">ظ†ط³ط®ظ‡ Prototype â€” ط¯ط§ط¯ظ‡â€Œظ‡ط§غŒ ط§غŒظ† ظ†ط³ط®ظ‡ ظ…طµظ†ظˆط¹غŒ ظ‡ط³طھظ†ط¯ ظˆ ظپظ‚ط· ط¨ط±ط§غŒ ظ†ظ…ط§غŒط´ ظ…ظ†ط·ظ‚ ظ…ط­طµظˆظ„ ط§ط³طھظپط§ط¯ظ‡ ظ…غŒâ€Œط´ظˆظ†ط¯.</div>',
    unsafe_allow_html=True,
)

# ---------- ط¯ط§ط¯ظ‡ ----------
@st.cache_data
def load_data():
    return pd.read_csv("nabz_x_synthetic_credit_risk_dataset.csv")


# ---------- ظ…ط¯ظ„ ----------
@st.cache_resource
def train_model(df):
    features = [
        "income_m_toman",
        "account_turnover_m_toman",
        "loan_amount_m_toman",
        "number_of_loans",
        "late_payment_count",
        "max_delay_days",
        "monthly_cashflow_m_toman",
        "existing_obligations_m_toman",
        "credit_history_years",
        "recent_balance_change_pct",
    ]

    X = df[features]
    y = df["default_12m"]

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    model.fit(X, y)
    return model, features


df = load_data()
model, features = train_model(df)

# ---------- ظ¾ظ†ظ„ ط§ظ†طھط®ط§ط¨ ظ…ط´طھط±غŒ ----------
st.sidebar.header("طھط­ظ„غŒظ„ ظ…ط´طھط±غŒ")

customer_ids = df["customer_id"].astype(str).tolist()
cid = st.sidebar.selectbox("ط´ظ†ط§ط³ظ‡ ظ…ط´طھط±غŒ", customer_ids)

row = df[df["customer_id"].astype(str) == cid].iloc[0].copy()

st.sidebar.subheader("ط´ط¨غŒظ‡â€Œط³ط§ط²غŒ طھط؛غŒغŒط± ط±ظپطھط§ط±")

late = st.sidebar.slider("طھط£ط®غŒط± ط§ط¶ط§ظپظ‡ ط¯ط± ظ¾ط±ط¯ط§ط®طھ", 0, 5, 0)
cash = st.sidebar.slider("طھط؛غŒغŒط± ط¬ط±غŒط§ظ† ظ†ظ‚ط¯غŒ (%)", -50, 30, 0)
oblig = st.sidebar.slider("ط§ظپط²ط§غŒط´ طھط¹ظ‡ط¯ط§طھ (ظ…غŒظ„غŒظˆظ† طھظˆظ…ط§ظ†)", 0, 500, 0)

row["late_payment_count"] += late
row["monthly_cashflow_m_toman"] *= (1 + cash / 100)
row["existing_obligations_m_toman"] += oblig

# ---------- ظ¾غŒط´â€Œط¨غŒظ†غŒ ----------
X = pd.DataFrame([[row[f] for f in features]], columns=features)
prob = float(model.predict_proba(X)[0, 1])
score = round(prob * 100, 1)

if score >= 70:
    level, icon = "ظ¾ط±ط±غŒط³ع©", "ًں”´"
    trend = "ط§ظپط²ط§غŒط´غŒ / ظ†غŒط§ط²ظ…ظ†ط¯ ط¨ط±ط±ط³غŒ"
elif score >= 40:
    level, icon = "ظ…طھظˆط³ط·", "ًںں "
    trend = "ظ†غŒط§ط²ظ…ظ†ط¯ ظ¾ط§غŒط´"
else:
    level, icon = "ع©ظ…â€Œط±غŒط³ع©", "ًںں¢"
    trend = "ظ¾ط§غŒط¯ط§ط±"

# ---------- ع©ط§ط±طھâ€Œظ‡ط§غŒ ط§طµظ„غŒ ----------
st.subheader("ظ†ظ…ط§غŒ ع©ظ„غŒ ط±غŒط³ع©")

a, b, c, d = st.columns(4)

a.metric("ط´ظ†ط§ط³ظ‡ ظ…ط´طھط±غŒ", cid)
b.metric("ط§ظ…طھغŒط§ط² ط±غŒط³ع©", f"{score:.1f}")
c.metric("ط§ط­طھظ…ط§ظ„ ظ†ع©ظˆظ„ غ±غ² ظ…ط§ظ‡ظ‡", f"{prob:.1%}")
d.metric("ط³ط·ط­ ط±غŒط³ع©", f"{icon} {level}")

st.info(f"ط±ظˆظ†ط¯ ط±غŒط³ع©: **{trend}**")

# ---------- ط¯ظ„ط§غŒظ„ ----------
st.subheader("ع†ط±ط§ ط±غŒط³ع© طھط؛غŒغŒط± ع©ط±ط¯ظ‡طں")

coef = model.named_steps["model"].coef_[0]
scaler = model.named_steps["scale"]

labels = {
    "income_m_toman": "ط¯ط±ط¢ظ…ط¯ ظ…ط§ظ‡ط§ظ†ظ‡",
    "account_turnover_m_toman": "ع¯ط±ط¯ط´ ط­ط³ط§ط¨",
    "loan_amount_m_toman": "ظ…ط¨ظ„ط؛ طھط³ظ‡غŒظ„ط§طھ",
    "number_of_loans": "طھط¹ط¯ط§ط¯ طھط³ظ‡غŒظ„ط§طھ",
    "late_payment_count": "طھط¹ط¯ط§ط¯ طھط£ط®غŒط± ظ¾ط±ط¯ط§ط®طھ",
    "max_delay_days": "ط¨غŒط´طھط±غŒظ† ط±ظˆط² طھط£ط®غŒط±",
    "monthly_cashflow_m_toman": "ط¬ط±غŒط§ظ† ظ†ظ‚ط¯غŒ ظ…ط§ظ‡ط§ظ†ظ‡",
    "existing_obligations_m_toman": "طھط¹ظ‡ط¯ط§طھ ظ…ظˆط¬ظˆط¯",
    "credit_history_years": "ط³ط§ط¨ظ‚ظ‡ ط§ط¹طھط¨ط§ط±غŒ",
    "recent_balance_change_pct": "طھط؛غŒغŒط± ط§ط®غŒط± ظ…ط§ظ†ط¯ظ‡ ط­ط³ط§ط¨",
}

items = []

for i, feature in enumerate(features):
    z = (X.iloc[0, i] - scaler.mean_[i]) / scaler.scale_[i]
    contribution = z * coef[i]
    items.append((feature, contribution))

for feature, contribution in sorted(
    items,
    key=lambda item: abs(item[1]),
    reverse=True,
)[:4]:
    effect = "ط§ظپط²ط§غŒط´ ط±غŒط³ع©" if contribution > 0 else "ع©ط§ظ‡ط´ ط±غŒط³ع©"
    st.write(f"â€¢ **{labels[feature]}** â†’ {effect}")

# ---------- ظ‡ط´ط¯ط§ط± ظˆ ط§ظ‚ط¯ط§ظ… ----------
st.subheader("ظ‡ط´ط¯ط§ط± ط²ظˆط¯ظ‡ظ†ع¯ط§ظ… ظˆ ط§ظ‚ط¯ط§ظ… ظ¾غŒط´ظ†ظ‡ط§ط¯غŒ")

if score >= 70:
    st.error("ظ‡ط´ط¯ط§ط±: ط§ظپط²ط§غŒط´ ط±غŒط³ع© ظ‚ط§ط¨ظ„ طھظˆط¬ظ‡ ط§ط³طھ ظˆ ظ†غŒط§ط² ط¨ظ‡ ط¨ط±ط±ط³غŒ ط§ط¹طھط¨ط§ط±غŒ ط¯ط§ط±ط¯.")
    st.info("ط§ظ‚ط¯ط§ظ… ظ¾غŒط´ظ†ظ‡ط§ط¯غŒ: ط¨ط±ط±ط³غŒ ظ¾ط±ظˆظ†ط¯ظ‡طŒ ظˆط¶ط¹غŒطھ طھط¹ظ‡ط¯ط§طھ ظˆ ط¯ط± طµظˆط±طھ ظ†غŒط§ط² طھظ…ط§ط³ ظ¾غŒط´ع¯غŒط±ط§ظ†ظ‡ ط¨ط§ ظ…ط´طھط±غŒ.")
elif score >= 40:
    st.warning("ظ‡ط´ط¯ط§ط±: ظ†ط´ط§ظ†ظ‡â€Œظ‡ط§غŒغŒ ط§ط² ط§ظپط²ط§غŒط´ ط±غŒط³ع© ظ…ط´ط§ظ‡ط¯ظ‡ ط´ط¯ظ‡ ط§ط³طھ.")
    st.info("ط§ظ‚ط¯ط§ظ… ظ¾غŒط´ظ†ظ‡ط§ط¯غŒ: ط§ظپط²ط§غŒط´ ط¯ظپط¹ط§طھ ظ¾ط§غŒط´ ظˆ ط¨ط±ط±ط³غŒ ط¬ط±غŒط§ظ† ظ†ظ‚ط¯غŒ ظˆ طھط¹ظ‡ط¯ط§طھ.")
else:
    st.success("ط±غŒط³ع© ط¯ط± ظ…ط­ط¯ظˆط¯ظ‡ ظ¾ط§غŒغŒظ† ظ‚ط±ط§ط± ط¯ط§ط±ط¯.")
    st.info("ط§ظ‚ط¯ط§ظ… ظ¾غŒط´ظ†ظ‡ط§ط¯غŒ: ط§ط¯ط§ظ…ظ‡ ظ¾ط§غŒط´ ط¯ظˆط±ظ‡â€Œط§غŒ.")

# ---------- ط¬ط²ط¦غŒط§طھ ----------
with st.expander("ط¬ط²ط¦غŒط§طھ ط¯ط§ط¯ظ‡ ظ…ط´طھط±غŒ"):
    st.dataframe(row.to_frame("ظ…ظ‚ط¯ط§ط±"), use_container_width=True)

with st.expander("ظ…ظ†ط·ظ‚ ظپظ†غŒ Prototype"):
    st.write(
        "ط¯ط± ط§غŒظ† ظ†ط³ط®ظ‡ ط§ط² Logistic Regression ط¨ظ‡â€Œط¹ظ†ظˆط§ظ† ظ…ط¯ظ„ ظ¾ط§غŒظ‡ ط§ط³طھظپط§ط¯ظ‡ ط´ط¯ظ‡ ط§ط³طھ. "
        "ط¯ط± ظ…ط­طµظˆظ„ ظˆط§ظ‚ط¹غŒطŒ ظ…ط¯ظ„â€Œظ‡ط§ ظ¾ط³ ط§ط² ط¨ط±ط±ط³غŒ ط¯ط§ط¯ظ‡ ظˆط§ظ‚ط¹غŒ ط¨ط§ظ†ع©طŒ ع©غŒظپغŒطھ ط¯ط§ط¯ظ‡طŒ "
        "ط¯ظ‚طھطŒ ع©ط§ظ„غŒط¨ط±ط§ط³غŒظˆظ†طŒ طھظˆط¶غŒط­â€Œظ¾ط°غŒط±غŒ ظˆ ط§ظ„ط²ط§ظ…ط§طھ ط¹ظ…ظ„غŒط§طھغŒ ط§ظ†طھط®ط§ط¨ ظˆ ط§ط¹طھط¨ط§ط±ط³ظ†ط¬غŒ ظ…غŒâ€Œط´ظˆظ†ط¯."
    )

st.caption(
    "ظ†ط¨ط¶ ط§غŒع©ط³ طھطµظ…غŒظ… ظ†ظ‡ط§غŒغŒ ط§ط¹طھط¨ط§ط±غŒ ط±ط§ ط¬ط§غŒع¯ط²غŒظ† ظ†ظ…غŒâ€Œع©ظ†ط¯ط› "
    "ط§غŒظ† ط³غŒط³طھظ… غŒع© Decision Support ط¨ط±ط§غŒ ط´ظ†ط§ط³ط§غŒغŒ ط²ظˆط¯ظ‡ظ†ع¯ط§ظ… ط±غŒط³ع© ط§ط³طھ."
)
