import streamlit as st
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

st.set_page_config(
    page_title="نبض ایکس",
    page_icon="❤️",
    layout="wide"
)

# =========================
# ظاهر و فونت فارسی
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;500;600;700;800&display=swap');

html, body, .stApp, .stApp * {
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
    padding: 28px 32px;
    border-radius: 22px;
    margin-bottom: 22px;
    background: linear-gradient(135deg, #071a33, #0d3b66 55%, #1769aa);
    color: white;
    text-align: right;
}

.hero h1 {
    margin: 0 0 10px 0;
    font-size: 38px;
    font-weight: 800;
}

.hero p {
    margin: 0;
    font-size: 18px;
    line-height: 1.9;
}

.demo {
    padding: 14px 18px;
    border-radius: 14px;
    background: #fff4e5;
    border-right: 5px solid #f59e0b;
    margin-bottom: 22px;
    direction: rtl;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)


# =========================
# سربرگ
# =========================
st.markdown("""
<div class="hero">
    <h1>❤️ نبض ایکس</h1>
    <p>
        موتور هوشمند پیش‌بینی و مداخله پیشگیرانه ریسک اعتباری
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="demo">
    نسخه Prototype — داده‌های این نسخه مصنوعی هستند و فقط برای نمایش منطق محصول استفاده می‌شوند.
</div>
""", unsafe_allow_html=True)


# =========================
# بارگذاری داده
# =========================
@st.cache_data
def load_data():
    return pd.read_csv("nabz_x_synthetic_credit_risk_dataset.csv")


# =========================
# آموزش مدل
# =========================
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
        "recent_balance_change_pct"
    ]

    model = Pipeline([
        ("scale", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                class_weight="balanced"
            )
        )
    ])

    model.fit(
        df[features],
        df["default_12m"]
    )

    return model, features


# =========================
# آماده‌سازی
# =========================
df = load_data()

model, features = train_model(df)


# =========================
# پنل سمت راست
# =========================
st.sidebar.header("تحلیل مشتری")

customer_ids = df["customer_id"].astype(str).tolist()

cid = st.sidebar.selectbox(
    "شناسه مشتری",
    customer_ids
)

row = df[
    df["customer_id"].astype(str) == cid
].iloc[0].copy()


# =========================
# شبیه‌سازی تغییر رفتار
# =========================
st.sidebar.subheader("شبیه‌سازی تغییر رفتار")

late = st.sidebar.slider(
    "تأخیر اضافه در پرداخت",
    0,
    5,
    0
)

cash = st.sidebar.slider(
    "تغییر جریان نقدی (%)",
    -50,
    30,
    0
)

oblig = st.sidebar.slider(
    "افزایش تعهدات (میلیون تومان)",
    0,
    500,
    0
)

row["late_payment_count"] += late

row["monthly_cashflow_m_toman"] *= (
    1 + cash / 100
)

row["existing_obligations_m_toman"] += oblig


# =========================
# پیش‌بینی ریسک
# =========================
X = pd.DataFrame(
    [[row[f] for f in features]],
    columns=features
)

prob = float(
    model.predict_proba(X)[0, 1]
)

score = round(
    prob * 100,
    1
)


# =========================
# تعیین سطح ریسک
# =========================
if score >= 70:

    level = "پرریسک"
    icon = "🔴"
    trend = "افزایشی / نیازمند بررسی"

elif score >= 40:

    level = "متوسط"
    icon = "🟠"
    trend = "نیازمند پایش"

else:

    level = "کم‌ریسک"
    icon = "🟢"
    trend = "پایدار"


# =========================
# نمای کلی
# =========================
st.subheader("نمای کلی ریسک")

a, b, c, d = st.columns(4)

a.metric(
    "شناسه مشتری",
    cid
)

b.metric(
    "امتیاز ریسک",
    f"{score:.1f}"
)

c.metric(
    "احتمال نکول ۱۲ ماهه",
    f"{prob:.1%}"
)

d.metric(
    "سطح ریسک",
    f"{icon} {level}"
)

st.info(
    f"روند ریسک: **{trend}**"
)


# =========================
# دلایل تغییر ریسک
# =========================
st.subheader("چرا ریسک تغییر کرده؟")

coef = model.named_steps["model"].coef_[0]

scaler = model.named_steps["scale"]


labels = {

    "income_m_toman":
        "درآمد ماهانه",

    "account_turnover_m_toman":
        "گردش حساب",

    "loan_amount_m_toman":
        "مبلغ تسهیلات",

    "number_of_loans":
        "تعداد تسهیلات",

    "late_payment_count":
        "تعداد تأخیر پرداخت",

    "max_delay_days":
        "بیشترین روز تأخیر",

    "monthly_cashflow_m_toman":
        "جریان نقدی ماهانه",

    "existing_obligations_m_toman":
        "تعهدات موجود",

    "credit_history_years":
        "سابقه اعتباری",

    "recent_balance_change_pct":
        "تغییر اخیر مانده حساب"
}


items = []

for i, feature in enumerate(features):

    z = (
        X.iloc[0, i]
        - scaler.mean_[i]
    ) / scaler.scale_[i]

    contribution = z * coef[i]

    items.append(
        (feature, contribution)
    )


for feature, contribution in sorted(
    items,
    key=lambda x: abs(x[1]),
    reverse=True
)[:4]:

    if contribution > 0:
        effect = "افزایش ریسک"
    else:
        effect = "کاهش ریسک"

    st.write(
        f"• **{labels[feature]}** → {effect}"
    )


# =========================
# هشدار و اقدام
# =========================
st.subheader(
    "هشدار زودهنگام و اقدام پیشنهادی"
)

if score >= 70:

    st.error(
        "هشدار: افزایش ریسک قابل توجه است و نیاز به بررسی اعتباری دارد."
    )

    st.info(
        "اقدام پیشنهادی: بررسی پرونده، وضعیت تعهدات و در صورت نیاز تماس پیشگیرانه با مشتری."
    )

elif score >= 40:

    st.warning(
        "هشدار: نشانه‌هایی از افزایش ریسک مشاهده شده است."
    )

    st.info(
        "اقدام پیشنهادی: افزایش دفعات پایش و بررسی جریان نقدی و تعهدات."
    )

else:

    st.success(
        "ریسک در محدوده پایین قرار دارد."
    )

    st.info(
        "اقدام پیشنهادی: ادامه پایش دوره‌ای."
    )


# =========================
# جزئیات مشتری
# =========================
with st.expander(
    "جزئیات داده مشتری"
):

    st.dataframe(
        row.to_frame("مقدار"),
        use_container_width=True
    )


# =========================
# توضیح فنی
# =========================
with st.expander(
    "منطق فنی Prototype"
):

    st.write(
        "در این نسخه از Logistic Regression "
        "به‌عنوان مدل پایه استفاده شده است. "
        "در محصول واقعی، مدل پس از بررسی داده واقعی "
        "بانک، کیفیت داده، دقت، توضیح‌پذیری و الزامات "
        "عملیاتی انتخاب و اعتبارسنجی می‌شود."
    )


# =========================
# توضیح نهایی
# =========================
st.caption(
    "نبض ایکس تصمیم نهایی اعتباری را جایگزین نمی‌کند؛ "
    "این سیستم یک Decision Support برای شناسایی زودهنگام ریسک است."
)
