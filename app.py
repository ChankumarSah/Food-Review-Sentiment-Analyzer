import streamlit as st
import joblib
import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Food Sentiment AI",
    page_icon="🍽️",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --cream:    #fdf6ec;
    --charcoal: #1a1a1a;
    --warm:     #c8522a;
    --gold:     #e8a838;
    --green:    #2d6a4f;
    --red:      #b5292a;
    --card-bg:  rgba(255,255,255,0.72);
    --blur:     blur(14px);
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--charcoal) !important;
}

/* ── Background ── */
.stApp {
    background:
        linear-gradient(rgba(253,246,236,0.82), rgba(253,246,236,0.88)),
        radial-gradient(ellipse at 15% 20%, rgba(200,82,42,0.10) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 75%, rgba(232,168,56,0.08) 0%, transparent 55%),
        url("https://images.unsplash.com/photo-1414235077428-338989a2e8c0?q=80&w=2070&auto=format&fit=crop");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

.block-container {
    padding-top: 2.5rem;
    max-width: 1100px;
}

/* ── Hero ── */
.hero-wrap {
    text-align: center;
    padding: 3rem 1rem 1.5rem;
}

.hero-label {
    display: inline-block;
    background: var(--warm);
    color: white !important;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 999px;
    margin-bottom: 18px;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.4rem, 5vw, 3.8rem);
    font-weight: 800;
    color: var(--charcoal) !important;
    line-height: 1.12;
    margin-bottom: 14px;
}

.hero-title span {
    color: var(--warm);
}

.hero-sub {
    font-size: 1.05rem;
    color: #6b5c4e !important;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.7;
    font-weight: 400;
}

/* ── Divider ── */
.fancy-divider {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 0 auto 2.2rem;
    max-width: 340px;
}
.fancy-divider hr {
    flex: 1;
    border: none;
    border-top: 1.5px solid rgba(200,82,42,0.25);
    margin: 0;
}
.fancy-divider span {
    color: var(--warm);
    font-size: 1.1rem;
}

/* ── Glass card ── */
.glass-card {
    background: var(--card-bg);
    border-radius: 28px;
    padding: 40px 44px;
    border: 1px solid rgba(200,82,42,0.12);
    box-shadow:
        0 4px 6px rgba(0,0,0,0.04),
        0 20px 60px rgba(200,82,42,0.08);
    backdrop-filter: var(--blur);
    -webkit-backdrop-filter: var(--blur);
    margin-bottom: 28px;
}

/* ── Labels ── */
.stTextArea label,
.stSelectbox label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.04em;
    text-transform: uppercase;
    color: #6b5c4e !important;
    margin-bottom: 6px !important;
}

/* ── Textarea ── */
.stTextArea textarea {
    background: rgba(253,246,236,0.9) !important;
    color: var(--charcoal) !important;
    border-radius: 18px !important;
    border: 1.5px solid rgba(200,82,42,0.2) !important;
    padding: 18px 20px !important;
    font-size: 16px !important;
    font-family: 'DM Sans', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--warm) !important;
    box-shadow: 0 0 0 3px rgba(200,82,42,0.12) !important;
}

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: rgba(253,246,236,0.9) !important;
    border-radius: 14px !important;
    border: 1.5px solid rgba(200,82,42,0.2) !important;
}
.stSelectbox div[data-baseweb="select"] div {
    color: var(--charcoal) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 15px !important;
}
div[role="listbox"] {
    background-color: #fdf6ec !important;
    border-radius: 14px !important;
    border: 1px solid rgba(200,82,42,0.15) !important;
}
div[role="option"] {
    background-color: #fdf6ec !important;
    color: var(--charcoal) !important;
}
div[role="option"]:hover {
    background-color: rgba(200,82,42,0.08) !important;
    color: var(--warm) !important;
}
.stSelectbox svg { fill: var(--warm) !important; }

/* ── Analyse button ── */
.stButton > button {
    width: 100%;
    height: 56px;
    border: none;
    border-radius: 16px;
    background: linear-gradient(90deg, #c8522a, #e8a838);
    color: white !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 17px !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 20px rgba(200,82,42,0.35);
    margin-top: 8px;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(200,82,42,0.5);
}
.stButton > button:active {
    transform: translateY(0);
}

/* ── Result banner ── */
.result-banner {
    border-radius: 22px;
    padding: 28px 32px;
    text-align: center;
    animation: slideUp 0.45s cubic-bezier(0.22,1,0.36,1) both;
    margin-bottom: 24px;
}
.result-banner.positive {
    background: linear-gradient(135deg, #1b4332, #2d6a4f);
    border: 1px solid rgba(255,255,255,0.12);
}
.result-banner.negative {
    background: linear-gradient(135deg, #7f1d1d, #b5292a);
    border: 1px solid rgba(255,255,255,0.12);
}
.result-banner .verdict {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 700;
    color: white !important;
    margin: 0;
}
.result-banner .verdict-sub {
    font-size: 0.9rem;
    color: rgba(255,255,255,0.72) !important;
    margin-top: 6px;
    font-weight: 400;
}

/* ── Probability pills ── */
.prob-row {
    display: flex;
    gap: 16px;
    margin-bottom: 20px;
}
.prob-card {
    flex: 1;
    background: var(--card-bg);
    border-radius: 18px;
    padding: 22px 20px;
    text-align: center;
    border: 1px solid rgba(200,82,42,0.1);
    box-shadow: 0 4px 16px rgba(0,0,0,0.04);
    animation: slideUp 0.5s cubic-bezier(0.22,1,0.36,1) both;
}
.prob-card.pos { animation-delay: 0.06s; }
.prob-card.neg { animation-delay: 0.12s; }
.prob-card .prob-icon { font-size: 1.8rem; margin-bottom: 6px; }
.prob-card .prob-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9b7c60 !important;
    margin-bottom: 4px;
}
.prob-card .prob-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    font-weight: 700;
    color: var(--charcoal) !important;
    line-height: 1;
}
.prob-card.pos .prob-value { color: var(--green) !important; }
.prob-card.neg .prob-value { color: var(--red) !important; }

/* ── Confidence bar ── */
.conf-wrap {
    background: var(--card-bg);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(200,82,42,0.1);
    animation: slideUp 0.55s cubic-bezier(0.22,1,0.36,1) both;
    animation-delay: 0.18s;
}
.conf-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9b7c60 !important;
    margin-bottom: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.conf-label span {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--warm) !important;
    letter-spacing: 0;
    text-transform: none;
}
.conf-bar-bg {
    height: 12px;
    border-radius: 999px;
    background: rgba(200,82,42,0.12);
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 999px;
    background: linear-gradient(90deg, #c8522a, #e8a838);
    transition: width 0.8s cubic-bezier(0.22,1,0.36,1);
}

/* ── Sentiment meter ── */
.meter-wrap {
    background: var(--card-bg);
    border-radius: 18px;
    padding: 20px 24px;
    margin-bottom: 20px;
    border: 1px solid rgba(200,82,42,0.1);
    animation: slideUp 0.6s cubic-bezier(0.22,1,0.36,1) both;
    animation-delay: 0.22s;
}
.meter-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #9b7c60 !important;
    margin-bottom: 14px;
}
.emoji-scale {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 4px;
}
.emoji-scale .e {
    flex: 1;
    text-align: center;
    font-size: 1.6rem;
    opacity: 0.25;
    transition: opacity 0.3s, transform 0.3s;
    cursor: default;
}
.emoji-scale .e.active {
    opacity: 1;
    transform: scale(1.35);
}
.emoji-ticks {
    display: flex;
    justify-content: space-between;
    padding: 0 8px;
    margin-top: 6px;
}
.emoji-ticks span {
    font-size: 0.68rem;
    color: #9b7c60 !important;
    font-weight: 500;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(200,82,42,0.06) !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    color: var(--charcoal) !important;
}

/* ── History table ── */
[data-testid="stDataFrame"] {
    background: rgba(255,255,255,0.7) !important;
    border-radius: 18px !important;
    overflow: hidden !important;
    border: 1px solid rgba(200,82,42,0.1) !important;
}
thead tr th {
    background-color: rgba(200,82,42,0.08) !important;
    color: var(--charcoal) !important;
    font-weight: 600 !important;
    font-size: 13px !important;
}
tbody tr td {
    background-color: transparent !important;
    color: var(--charcoal) !important;
}

/* ── Char counter ── */
.char-counter {
    font-size: 0.78rem;
    color: #9b7c60;
    text-align: right;
    margin-top: -8px;
    margin-bottom: 12px;
    font-weight: 500;
}
.char-counter.over { color: var(--red); }

/* ── Section title ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.35rem;
    font-weight: 700;
    color: var(--charcoal) !important;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a1a, #2d1810) !important;
}
section[data-testid="stSidebar"] * {
    color: #f5ede4 !important;
}
section[data-testid="stSidebar"] .sidebar-stat {
    background: rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 14px 16px;
    margin: 8px 0;
    border: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] .sidebar-stat .st-val {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8a838 !important;
}
section[data-testid="stSidebar"] .sidebar-stat .st-desc {
    font-size: 0.78rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: rgba(245,237,228,0.55) !important;
}
section[data-testid="stSidebar"] a {
    color: #e8a838 !important;
}

/* ── Warning ── */
.stAlert {
    border-radius: 14px !important;
}

/* ── Footer ── */
.footer-wrap {
    margin-top: 48px;
    padding: 36px 32px;
    border-radius: 24px;
    background: rgba(26,26,26,0.92);
    text-align: center;
    border: 1px solid rgba(255,255,255,0.07);
}
.footer-wrap h3 {
    font-family: 'Playfair Display', serif;
    color: white !important;
    font-size: 1.4rem;
    margin-bottom: 6px;
}
.footer-wrap p { color: rgba(255,255,255,0.55) !important; font-size: 0.9rem; }
.footer-wrap a { color: #e8a838 !important; font-weight: 600; }

/* ── Animations ── */
@keyframes slideUp {
    from { opacity: 0; transform: translateY(20px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(200,82,42,0.3); }
    50%       { box-shadow: 0 0 0 8px rgba(200,82,42,0); }
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--warm) !important; }

/* ── Progress override (hide default) ── */
.stProgress { display: none !important; }

</style>
""", unsafe_allow_html=True)


# =========================================================
# STOPWORDS
# =========================================================

sw = list(ENGLISH_STOP_WORDS)
sw.remove('not')
sw.remove('no')


# =========================================================
# TEXT CLEANING
# =========================================================

def text_cleaning(doc):
    doc = doc.lower()
    tokens = doc.split()
    new_doc = ""
    for t in tokens:
        if t not in sw:
            new_doc = new_doc + " " + t
    new_doc = new_doc.strip()
    return re.sub("[^a-z ]", "", new_doc)


# =========================================================
# EMOJI METER HELPER
# =========================================================

def get_active_emoji_index(positive_prob):
    """Returns 0–4 index for the emoji scale based on positive probability."""
    if positive_prob >= 80:
        return 4
    elif positive_prob >= 60:
        return 3
    elif positive_prob >= 40:
        return 2
    elif positive_prob >= 20:
        return 1
    else:
        return 0


# =========================================================
# LOAD MODEL
# =========================================================

model = joblib.load("food_review_sentiment.pkl")


# =========================================================
# HERO HEADER
# =========================================================

st.markdown("""
<div class="hero-wrap">
    <div class="hero-label">✦ Powered by Machine Learning</div>
    <h1 class="hero-title">Food Review<br><span>Sentiment Analyzer</span></h1>
    <p class="hero-sub">
        Paste any restaurant review and our NLP model will instantly tell you
        whether it's positive or negative — with confidence scores.
    </p>
    <div class="fancy-divider">
        <hr/><span>🍽️</span><hr/>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.markdown("""
<div style="padding: 8px 0 20px;">
    <div style="font-family:'Playfair Display',serif; font-size:1.4rem; font-weight:700; color:white;">
        🍽️ About
    </div>
    <div style="font-size:0.8rem; color:rgba(245,237,228,0.5); letter-spacing:0.06em; text-transform:uppercase; margin-top:2px;">
        AI Food Sentiment
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div class="sidebar-stat">
    <div class="st-val">LR</div>
    <div class="st-desc">Logistic Regression Model</div>
</div>
<div class="sidebar-stat">
    <div class="st-val">NLP</div>
    <div class="st-desc">CountVectorizer + Custom Cleaner</div>
</div>
<div class="sidebar-stat">
    <div class="st-val">CV=5</div>
    <div class="st-desc">5-Fold Cross Validation</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

st.sidebar.markdown("""
**🚀 Stack**
- Python · Scikit-Learn
- NLTK / sklearn stopwords
- Streamlit · Joblib

---

**🔗 Connect**
- [LinkedIn](https://www.linkedin.com/in/chandan-kumar-sah-752803387)
- [GitHub](https://github.com/ChankumarSah)
""")


# =========================================================
# SAMPLE REVIEWS
# =========================================================

sample_reviews = [
    "The burger was absolutely amazing and delicious!",
    "Worst restaurant ever. Very bad service and cold food.",
    "Pizza was tasty but delivery was painfully slow.",
    "Horrible experience, rude staff and overpriced dishes.",
    "Excellent food, warm ambiance, and super friendly staff!",
    "Food was great but the wait was almost two hours — unbearable.",
    "Perfectly cooked steak, lovely presentation. Will return!",
]

selected_review = st.selectbox(
    "📌 Try a sample review",
    ["— Choose a sample —"] + sample_reviews
)


# =========================================================
# MAIN GLASS CARD
# =========================================================

st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown('<div class="section-title">✍️ Your Review</div>', unsafe_allow_html=True)

MAX_CHARS = 500

review = st.text_area(
    "Enter Your Food Review",
    value="" if selected_review == "— Choose a sample —" else selected_review,
    placeholder="e.g. The pasta was divine and the service was absolutely top-notch...",
    height=130,
    label_visibility="collapsed"
)

# Character counter
char_count = len(review)
counter_class = "over" if char_count > MAX_CHARS else ""
st.markdown(
    f'<div class="char-counter {counter_class}">{char_count} / {MAX_CHARS}</div>',
    unsafe_allow_html=True
)

predict_btn = st.button("🔍 Analyse Sentiment")

st.markdown("</div>", unsafe_allow_html=True)


# =========================================================
# PREDICTION
# =========================================================

if predict_btn:

    if review.strip() == "":
        st.warning("⚠️ Please enter a review before analysing.")

    elif char_count > MAX_CHARS:
        st.warning(f"⚠️ Review exceeds {MAX_CHARS} characters. Please shorten it.")

    else:
        with st.spinner("Analysing your review…"):

            # Pipeline has preprocessor=text_cleaning built-in, so pass raw review
            # Cleaning manually here would double-clean and break predictions
            prediction = model.predict([review])[0]
            probability = model.predict_proba([review])
            cleaned = text_cleaning(review)  # only used for display
            positive_prob = probability[0][1] * 100
            negative_prob = probability[0][0] * 100
            confidence = max(positive_prob, negative_prob)

        # ── Result banner ──
        if prediction == 1:
            st.markdown(f"""
            <div class="result-banner positive">
                <p class="verdict">✅ Positive Review</p>
                <p class="verdict-sub">The model detected a positive sentiment in your review.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-banner negative">
                <p class="verdict">❌ Negative Review</p>
                <p class="verdict-sub">The model detected a negative sentiment in your review.</p>
            </div>
            """, unsafe_allow_html=True)

        # ── Probability cards ──
        st.markdown(f"""
        <div class="prob-row">
            <div class="prob-card pos">
                <div class="prob-icon">😊</div>
                <div class="prob-label">Positive</div>
                <div class="prob-value">{positive_prob:.1f}%</div>
            </div>
            <div class="prob-card neg">
                <div class="prob-icon">😡</div>
                <div class="prob-label">Negative</div>
                <div class="prob-value">{negative_prob:.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Confidence bar ──
        st.markdown(f"""
        <div class="conf-wrap">
            <div class="conf-label">
                Model Confidence
                <span>{confidence:.1f}%</span>
            </div>
            <div class="conf-bar-bg">
                <div class="conf-bar-fill" style="width:{confidence}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Emoji sentiment meter ──
        emojis = ["😡", "😟", "😐", "😊", "🤩"]
        labels = ["Very Neg", "Negative", "Neutral", "Positive", "Very Pos"]
        active_idx = get_active_emoji_index(positive_prob)
        emoji_html = "".join(
            f'<div class="e {"active" if i == active_idx else ""}">{e}</div>'
            for i, e in enumerate(emojis)
        )
        ticks_html = "".join(f"<span>{l}</span>" for l in labels)
        st.markdown(f"""
        <div class="meter-wrap">
            <div class="meter-label">📊 Sentiment Meter</div>
            <div class="emoji-scale">{emoji_html}</div>
            <div class="emoji-ticks">{ticks_html}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Cleaned text expander ──
        with st.expander("🧹 View cleaned text (as fed to model)"):
            st.code(cleaned, language=None)

        # ── Save to history ──
        if "history" not in st.session_state:
            st.session_state.history = []

        result_label = "✅ Positive" if prediction == 1 else "❌ Negative"
        st.session_state.history.append({
            "Review": review[:80] + ("…" if len(review) > 80 else ""),
            "Verdict": result_label,
            "Confidence": f"{confidence:.1f}%",
            "+ve Prob": f"{positive_prob:.1f}%",
            "-ve Prob": f"{negative_prob:.1f}%",
        })


# =========================================================
# HISTORY
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if len(st.session_state.history) > 0:

    col_h, col_c = st.columns([5, 1])

    with col_h:
        st.markdown('<div class="section-title">🕘 Prediction History</div>', unsafe_allow_html=True)

    with col_c:
        if st.button("🗑️ Clear"):
            st.session_state.history = []
            st.rerun()

    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)


# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<div class="footer-wrap">
    <h3>👨‍💻 Developed by Chandan Sah</h3>
    <p>🚀 AI-Powered NLP Web Application &nbsp;·&nbsp; Built with Streamlit & Scikit-Learn</p>
    <p style="margin-top:12px;">
        🔗 <a href="https://www.linkedin.com/in/chandan-kumar-sah-752803387" target="_blank">LinkedIn</a>
        &nbsp;&nbsp;
        💻 <a href="https://github.com/ChankumarSah" target="_blank">GitHub</a>
    </p>
</div>
""", unsafe_allow_html=True)