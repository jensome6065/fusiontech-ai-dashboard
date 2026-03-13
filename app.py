import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

st.set_page_config(page_title="FusionTech AI Review Intelligence", page_icon="⚡", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Space+Grotesk:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; background-color: #0d0d0f; color: #e8e4dc; }
.stApp { background-color: #0d0d0f; }
section[data-testid="stSidebar"] { background: #111115; border-right: 1px solid #2a2a35; }
[data-testid="metric-container"] { background: #16161e; border: 1px solid #2a2a35; border-radius: 8px; padding: 16px; }
[data-testid="metric-container"] label { color: #888 !important; font-family: 'IBM Plex Mono', monospace !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 0.08em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { font-family: 'IBM Plex Mono', monospace !important; font-size: 26px !important; font-weight: 600 !important; color: #f5f0e8 !important; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.02em; }
.stSelectbox label { color: #aaa !important; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; }
.stTable table { background: #16161e; border: 1px solid #2a2a35; border-radius: 6px; }
.stTable th { background: #1e1e28 !important; color: #888 !important; font-family: 'IBM Plex Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 0.08em; border-bottom: 1px solid #2a2a35 !important; }
.stTable td { color: #e8e4dc !important; font-size: 13px; border-color: #1e1e28 !important; }
hr { border-color: #2a2a35; }
.stTextArea textarea { background: #16161e !important; border: 1px solid #2a2a35 !important; color: #e8e4dc !important; font-family: 'IBM Plex Mono', monospace; font-size: 13px; }
.section-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #555; letter-spacing: 0.15em; text-transform: uppercase; margin-bottom: 4px; }
.badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-family: 'IBM Plex Mono', monospace; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; }
.badge-pos { background: #1a3a2a; color: #4ade80; border: 1px solid #2d6b44; }
.badge-neg { background: #3a1a1a; color: #f87171; border: 1px solid #6b2d2d; }
.badge-neu { background: #2a2a1a; color: #facc15; border: 1px solid #5a4d10; }
.alert-card { background: #1e0e0e; border: 1px solid #6b2d2d; border-left: 4px solid #f87171; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }
.alert-card .alert-title { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; color: #f87171; text-transform: uppercase; letter-spacing: 0.08em; }
.alert-card .alert-body { font-size: 13px; color: #c8b8b8; margin-top: 4px; line-height: 1.5; }
.alert-card .alert-action { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #fb923c; margin-top: 8px; padding-top: 8px; border-top: 1px solid #3a1a1a; }
.action-card { background: #0e1a0e; border: 1px solid #2d6b44; border-left: 4px solid #4ade80; border-radius: 8px; padding: 14px 18px; margin-bottom: 10px; }
.action-card .action-title { font-family: 'IBM Plex Mono', monospace; font-size: 12px; font-weight: 600; color: #4ade80; text-transform: uppercase; letter-spacing: 0.08em; }
.action-card .action-body { font-size: 13px; color: #b8c8b8; margin-top: 4px; line-height: 1.5; }
.pipeline-step { background: #16161e; border: 1px solid #2a2a35; border-radius: 8px; padding: 12px 16px; margin-bottom: 8px; display: flex; align-items: flex-start; gap: 12px; }
.pipeline-icon { font-size: 20px; min-width: 28px; text-align: center; padding-top: 2px; }
.pipeline-text .pipeline-label { font-family: 'IBM Plex Mono', monospace; font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 0.12em; }
.pipeline-text .pipeline-value { font-size: 13px; color: #e8e4dc; margin-top: 2px; }
.hitl-banner { background: #16121e; border: 1px solid #4a2d6b; border-left: 4px solid #a78bfa; border-radius: 8px; padding: 12px 18px; margin-top: 12px; }
.hitl-banner .hitl-title { font-family: 'IBM Plex Mono', monospace; font-size: 11px; color: #a78bfa; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600; }
.hitl-banner .hitl-body { font-size: 12px; color: #c8b8d8; margin-top: 4px; line-height: 1.5; }
.chip { display: inline-block; padding: 3px 10px; border-radius: 4px; font-size: 11px; font-family: 'IBM Plex Mono', monospace; margin: 2px 3px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }
.chip-hardware { background: #1a1e3a; color: #93c5fd; border: 1px solid #2d3d6b; }
.chip-software { background: #1a2a1a; color: #86efac; border: 1px solid #2d5a2d; }
.chip-support  { background: #2a1a1a; color: #fca5a5; border: 1px solid #6b2d2d; }
.chip-battery  { background: #2a1e0a; color: #fdba74; border: 1px solid #6b4a1a; }
.chip-build    { background: #1e1a2a; color: #c4b5fd; border: 1px solid #4a3a6b; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────
COLORS  = {"Positive": "#4ade80", "Neutral": "#facc15", "Negative": "#f87171"}
BG      = "#0d0d0f"
SURFACE = "#16161e"
BORDER  = "#2a2a35"
TEXT    = "#e8e4dc"
MUTED   = "#888888"

SHORT_NAMES = {
    "FusionTech 15.6-Inch Gaming Laptop (6th Gen Intel Quad-Core i5-6300HQ Processor up to 3.2GHz, 8GB DDR3, 256GB SSD, Nvidia GeForce GTX 960M, Windows 10)":
        "G3 Gaming — i5 / GTX 960M",
    'FusionTech Worktop i7559-5012GRY 15.6" UHD (3840x2160) 4k Touchscreen Laptop (Intel Quad Core i7-6700HQ, 8 GB RAM, 1 TB HDD) NVIDIA GeForce GTX 960M, Microsoft Signature Edition':
        "Worktop 4K Touch — i7 / GTX 960M",
    "Advance AW17R3-4175SLV 17.3-Inch FHD Laptop (6th Generation Intel Core i7, 16 GB RAM, 1 TB HDD + 256 GB SATA SSD,NVIDIA GeForce GTX 970M, Windows 10 Home), Silver)":
        "Advance AW17 — i7 / GTX 970M",
    'FusionTech Worktop 15 5000 5577 Gaming Laptop - (15.6" Full HD (1920x1080), Intel Quad-Core i5-7300HQ Processor, 1TB HDD, 8GB DDR4 DRAM, NVIDIA GeForce GTX 1050 4GB VRAM, Windows 10':
        "Worktop 15 — i5 / GTX 1050",
    "Advance AW15R2-8469SLV 15.6-Inch UHD Laptop (6th Generation Intel Core i7, 16 GB RAM, 1 TB HDD + 256 GB SATA SSD) NVIDIA GeForce GTX 970M, Microsoft Signature Edition, Windows 10 Home), Silver":
        "Advance AW15 — i7 / GTX 970M",
}

TOPIC_KEYWORDS = {
    "Hardware Failure": ["broken","dead","failed","failure","stopped working","doesn't work","hardware","motherboard","hinge","screen","display","keyboard","port","fan","overheating","overheat","heat"],
    "Battery Life":     ["battery","charge","charging","power","unplugged","dies","drain","runtime","hours"],
    "Performance":      ["slow","lag","laggy","freeze","freezes","frozen","crash","crashes","boot","startup","speed","performance","ram","memory","ssd","hdd","drive"],
    "Customer Support": ["support","service","customer service","warranty","response","rep","agent","return","refund","replacement","helped","unhelpful"],
    "Build Quality":    ["build","cheap","plastic","flimsy","quality","feels","material","finish","durability","durable"],
}
TOPIC_CHIP  = {"Hardware Failure":"chip-hardware","Battery Life":"chip-battery","Performance":"chip-hardware","Customer Support":"chip-support","Build Quality":"chip-build","Other":"chip-software"}
TOPIC_EMOJI = {"Hardware Failure":"🔧","Battery Life":"🔋","Performance":"⚡","Customer Support":"📞","Build Quality":"🏗️","Other":"📌"}
ACTION_MAP  = {
    "Hardware Failure": ("🔧 Hardware Failure Alert",
        "Multiple reviews flag hardware defects — broken hinges, dead ports, display failures.",
        "→ Pause promotions. Flag for QC review. Surface warranty messaging."),
    "Battery Life": ("🔋 Battery Complaint Spike",
        "Customers consistently cite poor battery endurance as a top pain point.",
        "→ Add battery disclaimer to listing. Consider accessory bundle discount."),
    "Performance": ("⚡ Performance Issues Detected",
        "Reviews cite lag, crashes, and slow boot — likely to drive returns.",
        "→ Deprioritize in weekly email. Escalate to product team for driver update."),
    "Customer Support": ("📞 Support Experience Driving Churn",
        "Customers frustrated with support response times, not just the product.",
        "→ Route to CX team. Add FAQ to product page. Flag for agent training."),
    "Build Quality": ("🏗️ Build Quality Concerns",
        "Customers report flimsy materials and poor finish quality.",
        "→ Adjust product tier positioning. Review supplier QC standards."),
}

WC_STOPWORDS = {
    "a","an","the","this","that","these","those","my","your","his","her","our","their","its",
    "i","me","we","us","you","he","she","it","they","them","him","who","which","what",
    "is","are","was","were","be","been","being","am","have","has","had","do","does","did",
    "get","got","getting","go","went","going","come","came","coming","make","made","say",
    "said","know","think","use","used","using","would","could","should","will","can","may",
    "might","need","want","keep","put","see","look","take","let","try","set",
    "and","or","but","so","for","nor","yet","as","at","by","in","of","on","to","up","if",
    "out","about","with","from","into","than","then","when","where","after","before","over",
    "back","just","even","also","only","too","very","well","still","now","more","most",
    "some","any","all","no","not","same","own","other","again","away",
    "really","actually","basically","literally","definitely","probably","maybe","quite",
    "though","because","since","while","although","however","already","always","never",
    "much","many","how","re","ll","ve","don","doesn","didn","won","can","couldn","wasn",
    "computer","laptop","fusiontech","device","product","pc","machine","system","buy",
    "bought","purchase","purchased","ordered","order","item","one","two","three",
    "month","months","week","weeks","day","days","year","years","time","times",
    "review","reviewer","star","stars","rating","amazon","price","money","paid",
    "received","delivery","shipped","shipping","package","box","sent","send",
}
BIGRAM_BLOCK = {
    "buy fusiontech","fusiontech laptop","fusiontech gaming","dell laptop","new laptop",
    "new computer","got laptop","got computer","windows 10","great laptop","great computer",
    "good laptop","ve had","i ve","i m","it s","don t","didn t","i got","i bought","i purchased","i have","i had",
}

def classify_topic(text):
    tl = str(text).lower()
    scores = {t: sum(1 for kw in kws if kw in tl) for t, kws in TOPIC_KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Other"

# ── Data ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fusiontech_cleaned.csv")
    df = df[["text","rating","brand","title_y"]].dropna(subset=["text","title_y"])
    top5 = df["title_y"].value_counts().head(5).index.tolist()
    df = df[df["title_y"].isin(top5)].copy()
    df["short_name"] = df["title_y"].map(SHORT_NAMES).fillna(df["title_y"].str[:40])
    df["sentiment"] = df["rating"].apply(lambda r: "Positive" if r >= 4 else ("Neutral" if r == 3 else "Negative"))
    df["topic"]     = df["text"].apply(classify_topic)
    return df

df = load_data()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ FusionTech")
    st.markdown('<p class="section-label">AI Review Intelligence · Challenge 1</p>', unsafe_allow_html=True)
    st.markdown("---")

    short_names    = df["short_name"].unique().tolist()
    selected_short = st.selectbox("Select product", short_names)
    product_df     = df[df["short_name"] == selected_short]

    total   = len(product_df)
    avg_r   = product_df["rating"].mean()
    pct_pos = (product_df["sentiment"] == "Positive").mean() * 100
    pct_neg = (product_df["sentiment"] == "Negative").mean() * 100

    if pct_neg > 25 or avg_r < 3.5:
        st.markdown("""
        <div style="background:#1e0e0e;border:1px solid #f87171;border-radius:6px;padding:10px 14px;margin:8px 0;">
            <span style="font-family:'IBM Plex Mono',monospace;font-size:11px;color:#f87171;font-weight:600;">⚠ ACTION REQUIRED</span><br>
            <span style="font-size:12px;color:#c8b8b8;">High complaint rate detected</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.metric("Total Reviews", f"{total:,}")
    st.metric("Avg Rating",    f"{avg_r:.2f} / 5.0")
    st.metric("Positive",      f"{pct_pos:.0f}%")
    st.metric("Negative",      f"{pct_neg:.0f}%")
    st.markdown("---")
    st.markdown('<p class="section-label">Powered by Dell PowerEdge</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:11px;color:#555;font-family:\'IBM Plex Mono\',monospace;">GPU-accelerated NLP<br>Real-time inference · &lt;1s latency</p>', unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:4px;">
  <div>
    <h1 style="font-size:1.9rem;margin:0;font-weight:700;color:#f5f0e8;">Customer Feedback Intelligence</h1>
    <p style="color:{MUTED};font-family:'IBM Plex Mono',monospace;font-size:11px;margin-top:4px;">
        Challenge 1: Eliminate Feedback Blind Spots · Top 5 products by review volume
    </p>
  </div>
  <div style="text-align:right;">
    <span style="font-family:'IBM Plex Mono',monospace;font-size:10px;color:#555;">AI PIPELINE</span><br>
    <span style="font-size:12px;color:#a78bfa;">Sentiment Analysis + Topic Classification</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ── ROW 1: Overview bar + Donut ────────────────────────────
col_left, col_right = st.columns([3, 2], gap="large")

with col_left:
    st.markdown('<p class="section-label">Sentiment overview — all 5 products</p>', unsafe_allow_html=True)
    overview = []
    for short in short_names:
        sub = df[df["short_name"] == short]
        overview.append({"Product": short,
                         "Positive": (sub["sentiment"] == "Positive").sum(),
                         "Neutral":  (sub["sentiment"] == "Neutral").sum(),
                         "Negative": (sub["sentiment"] == "Negative").sum(),
                         "pct_neg":  (sub["sentiment"] == "Negative").mean() * 100})
    ov_df = pd.DataFrame(overview).set_index("Product")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    x, w = range(len(ov_df)), 0.28
    ax.bar([i - w for i in x], ov_df["Positive"], width=w, color="#4ade80", label="Positive", zorder=3)
    ax.bar([i     for i in x], ov_df["Neutral"],  width=w, color="#facc15", label="Neutral",  zorder=3)
    ax.bar([i + w for i in x], ov_df["Negative"], width=w, color="#f87171", label="Negative", zorder=3)
    for idx, (short, row) in enumerate(ov_df.iterrows()):
        if row["pct_neg"] > 30:
            ax.annotate("⚠", xy=(idx + w, row["Negative"]), xytext=(0, 4),
                        textcoords="offset points", ha="center", fontsize=10, color="#f87171")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.split(" — ")[0] for s in ov_df.index], color=TEXT, fontsize=9, fontfamily="monospace")
    ax.tick_params(axis="y", colors=MUTED, labelsize=9)
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, labelcolor=MUTED, prop={"family": "monospace", "size": 9}, ncol=3, loc="upper right")
    plt.tight_layout(pad=0.5); st.pyplot(fig); plt.close()

with col_right:
    st.markdown('<p class="section-label">Sentiment split — selected product</p>', unsafe_allow_html=True)
    sent_counts = product_df["sentiment"].value_counts()
    labels = sent_counts.index.tolist()
    fig2, ax2 = plt.subplots(figsize=(3.5, 3.4))
    fig2.patch.set_facecolor(SURFACE); ax2.set_facecolor(SURFACE)
    wedges, _, autotexts = ax2.pie(sent_counts.values, colors=[COLORS[l] for l in labels],
        autopct="%1.0f%%", startangle=90, pctdistance=0.72,
        wedgeprops=dict(width=0.55, edgecolor=BG, linewidth=2))
    for t in autotexts:
        t.set_color(BG); t.set_fontsize(11); t.set_fontweight("bold"); t.set_fontfamily("monospace")
    ax2.legend(handles=[mpatches.Patch(color=COLORS[l], label=l) for l in labels],
        frameon=False, labelcolor=MUTED, loc="lower center",
        prop={"family": "monospace", "size": 9}, ncol=3, bbox_to_anchor=(0.5, -0.08))
    ax2.set_title(selected_short.split(" — ")[0], color=TEXT, fontsize=11, fontfamily="monospace", pad=8)
    plt.tight_layout(pad=0.3); st.pyplot(fig2); plt.close()

st.markdown("---")

# ── ROW 2: Topic breakdown + Bigrams ──────────────────────
col_topic, col_bigram = st.columns([1, 1], gap="large")
negative_reviews = product_df[product_df["sentiment"] == "Negative"]["text"]
neg_df_full      = product_df[product_df["sentiment"] == "Negative"]

with col_topic:
    st.markdown('<p class="section-label">Complaint categories — AI topic classification</p>', unsafe_allow_html=True)
    if not neg_df_full.empty:
        topic_counts = neg_df_full["topic"].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        topic_counts["Share"] = (topic_counts["Count"] / topic_counts["Count"].sum() * 100).round(1)

        fig_t, ax_t = plt.subplots(figsize=(5, 3.2))
        fig_t.patch.set_facecolor(SURFACE); ax_t.set_facecolor(SURFACE)
        palette = ["#f87171","#fb923c","#facc15","#a78bfa","#93c5fd","#86efac"]
        bars = ax_t.barh(topic_counts["Topic"], topic_counts["Count"],
                         color=palette[:len(topic_counts)], height=0.55, zorder=3)
        for bar, share in zip(bars, topic_counts["Share"]):
            ax_t.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                      f"{share}%", va="center", ha="left", color=MUTED, fontsize=9, fontfamily="monospace")
        ax_t.spines[:].set_visible(False)
        ax_t.tick_params(axis="y", colors=TEXT, labelsize=10)
        ax_t.tick_params(axis="x", colors=MUTED, labelsize=8)
        ax_t.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0)
        ax_t.set_axisbelow(True); ax_t.invert_yaxis()
        plt.tight_layout(pad=0.4); st.pyplot(fig_t); plt.close()

        chips_html = "".join(
            f'<span class="chip {TOPIC_CHIP.get(r.Topic,"chip-software")}">'
            f'{TOPIC_EMOJI.get(r.Topic,"📌")} {r.Topic}</span>'
            for r in topic_counts.itertuples()
        )
        st.markdown(chips_html, unsafe_allow_html=True)
    else:
        st.info("No negative reviews for this product.")

with col_bigram:
    st.markdown('<p class="section-label">Top complaint phrases (bigrams)</p>', unsafe_allow_html=True)
    if len(negative_reviews) >= 5:
        vec = CountVectorizer(stop_words="english", ngram_range=(2, 2), max_features=80, min_df=2,
                              token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b")
        X = vec.fit_transform(negative_reviews.astype(str))
        all_phrases = pd.DataFrame({"Phrase": vec.get_feature_names_out(), "Mentions": X.toarray().sum(axis=0)})
        def is_meaningful(p):
            return not any(t in WC_STOPWORDS for t in p.split()) and p not in BIGRAM_BLOCK
        complaints = (all_phrases[all_phrases["Phrase"].apply(is_meaningful)]
                      .sort_values("Mentions", ascending=False).head(8).reset_index(drop=True))
        complaints.index += 1
        st.table(complaints)
    else:
        st.info("Not enough negative reviews for phrase analysis.")

st.markdown("---")

# ── ROW 3: Word cloud + Rating dist ───────────────────────
col_wc, col_rating = st.columns([3, 2], gap="large")

with col_wc:
    st.markdown('<p class="section-label">Issue keywords — negative reviews word cloud</p>', unsafe_allow_html=True)
    neg_text = " ".join(negative_reviews.astype(str))
    if neg_text.strip():
        wc = WordCloud(width=700, height=300, background_color=SURFACE, colormap="RdYlGn_r",
                       stopwords=WC_STOPWORDS, max_words=60, prefer_horizontal=0.85,
                       collocations=False, min_word_length=4, relative_scaling=0.6).generate(neg_text)
        fig3, ax3 = plt.subplots(figsize=(6.5, 2.8))
        fig3.patch.set_facecolor(SURFACE); ax3.imshow(wc); ax3.axis("off")
        plt.tight_layout(pad=0); st.pyplot(fig3); plt.close()
    else:
        st.info("No negative review text available.")

with col_rating:
    st.markdown('<p class="section-label">Rating distribution</p>', unsafe_allow_html=True)
    rating_counts = product_df["rating"].value_counts().sort_index()
    star_colors = {1:"#f87171",2:"#fb923c",3:"#facc15",4:"#a3e635",5:"#4ade80"}
    fig4, ax4 = plt.subplots(figsize=(4.5, 2.8))
    fig4.patch.set_facecolor(SURFACE); ax4.set_facecolor(SURFACE)
    for star, count in rating_counts.items():
        ax4.barh(f"{'★'*int(star)}", count, color=star_colors.get(int(star), MUTED), height=0.55, zorder=3)
    ax4.spines[:].set_visible(False)
    ax4.tick_params(axis="y", colors=TEXT, labelsize=11)
    ax4.tick_params(axis="x", colors=MUTED, labelsize=9)
    ax4.xaxis.grid(True, color=BORDER, linewidth=0.5, zorder=0); ax4.set_axisbelow(True)
    plt.tight_layout(pad=0.4); st.pyplot(fig4); plt.close()

st.markdown("---")

# ── ROW 4: Business Action Panel ──────────────────────────
st.markdown('<p class="section-label">AI-generated business actions — for merchandising team</p>', unsafe_allow_html=True)

top_topics = neg_df_full["topic"].value_counts().head(2).index.tolist() if not neg_df_full.empty else []
col_act1, col_act2 = st.columns(2, gap="large")

for i, topic in enumerate(top_topics):
    if topic in ACTION_MAP:
        title, body, action = ACTION_MAP[topic]
        card = f"""<div class="alert-card">
            <div class="alert-title">{title}</div>
            <div class="alert-body">{body}</div>
            <div class="alert-action">{action}</div>
        </div>"""
        (col_act1 if i == 0 else col_act2).markdown(card, unsafe_allow_html=True)

if pct_pos >= 70:
    target_col = col_act2 if len(top_topics) >= 1 else col_act1
    target_col.markdown(f"""<div class="action-card">
        <div class="action-title">✅ Promote This Product</div>
        <div class="action-body">{pct_pos:.0f}% positive sentiment · {avg_r:.2f}/5.0 avg rating.<br>
        Strong candidate for next weekly email campaign to 2M customers.</div>
    </div>""", unsafe_allow_html=True)

st.markdown("---")

# ── ROW 5: Live tester + Pipeline explainer ───────────────
col_demo, col_pipeline = st.columns([3, 2], gap="large")

with col_demo:
    st.markdown('<p class="section-label">Live AI review analyser — sentiment + topic + recommended action</p>', unsafe_allow_html=True)
    user_review = st.text_area("Paste a customer review", height=110, label_visibility="collapsed",
        placeholder="e.g. 'Battery drains in 2 hours and the hinge broke after a month. Support never responded.'")

    if user_review.strip():
        tl = user_review.lower()
        pos_words = ["great","love","excellent","amazing","perfect","fast","best","good","happy","easy","recommend","fantastic","awesome","solid","impressed","outstanding","superb"]
        neg_words = ["bad","poor","worst","terrible","broken","slow","hate","disappointing","awful","defective","useless","junk","waste","horrible","fails","crashed","problem","issue","return","refund","stopped working","overheating","freezes"]
        score = sum(1 for w in pos_words if w in tl) - sum(1 for w in neg_words if w in tl)
        sentiment, badge_cls = ("Positive","badge-pos") if score > 0 else (("Negative","badge-neg") if score < 0 else ("Neutral","badge-neu"))
        topic = classify_topic(user_review)
        topic_chip = f'<span class="chip {TOPIC_CHIP.get(topic,"chip-software")}">{TOPIC_EMOJI.get(topic,"📌")} {topic}</span>'
        _, _, action_text = ACTION_MAP.get(topic, ("","",""))

        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-icon">🤖</div>
            <div class="pipeline-text">
                <div class="pipeline-label">Step 1 — Sentiment Analysis</div>
                <div class="pipeline-value"><span class="badge {badge_cls}">{sentiment}</span></div>
            </div>
        </div>
        <div class="pipeline-step">
            <div class="pipeline-icon">🏷️</div>
            <div class="pipeline-text">
                <div class="pipeline-label">Step 2 — Topic Classification</div>
                <div class="pipeline-value">{topic_chip}</div>
            </div>
        </div>
        {'<div class="pipeline-step"><div class="pipeline-icon">📋</div><div class="pipeline-text"><div class="pipeline-label">Step 3 — Recommended Action</div><div class="pipeline-value" style="color:#fb923c;">' + action_text + '</div></div></div>' if action_text and sentiment == "Negative" else ""}
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="hitl-banner">
            <div class="hitl-title">🧑 Human-in-the-Loop · Responsible AI</div>
            <div class="hitl-body">
                Low-confidence predictions (&lt;60% score) are flagged for manual review before any
                product action is taken. Monthly bias audits ensure no customer segment is
                systematically misclassified. Customer data is anonymized at ingestion.
            </div>
        </div>""", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<p class="section-label">How this works — end-to-end AI pipeline</p>', unsafe_allow_html=True)
    steps = [
        ("📥","Ingestion",         "New reviews stream from FusionTech.com in real time via API"),
        ("🧹","Preprocessing",     "Deduplication, language detection, stopword removal, tokenization"),
        ("🤖","Sentiment Model",   "NLP classifier → Positive / Neutral / Negative with confidence score"),
        ("🏷️","Topic Classifier", "Keyword + ML model maps each review to a complaint category"),
        ("📊","Dashboard + Alerts","Merchandising team sees insights in <1 hour, not 5 days"),
        ("🖥️","Dell PowerEdge",    "GPU-accelerated NLP inference; eliminates outages that stall AI"),
    ]
    for icon, label, value in steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="pipeline-icon">{icon}</div>
            <div class="pipeline-text">
                <div class="pipeline-label">{label}</div>
                <div class="pipeline-value">{value}</div>
            </div>
        </div>""", unsafe_allow_html=True)