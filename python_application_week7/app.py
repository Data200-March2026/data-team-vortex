import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
import warnings
warnings.filterwarnings("ignore")

matplotlib.rcParams.update({
    "font.family":      "serif",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.color":       "#e8eaf0",
    "grid.linewidth":   0.7,
    "axes.facecolor":   "#fafbff",
    "figure.facecolor": "#ffffff",
    "axes.labelcolor":  "#2d3748",
    "xtick.color":      "#4a5568",
    "ytick.color":      "#4a5568",
    "axes.titleweight": "bold",
    "axes.titlesize":   11,
    "axes.labelsize":   9,
    "xtick.labelsize":  8,
    "ytick.labelsize":  8,
})

st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'DM Serif Display', serif !important; }
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton > button {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #f0c040; border: none; border-radius: 8px;
        font-weight: 600; letter-spacing: 0.5px;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #16213e 0%, #0f3460 100%);
        color: #f0c040; box-shadow: 0 4px 16px rgba(0,0,0,0.25);
    }
    .prediction-box {
        background: linear-gradient(135deg, #1a1a2e 0%, #0f3460 100%);
        border-radius: 16px; padding: 2rem; text-align: center;
        margin: 1rem 0; box-shadow: 0 8px 32px rgba(0,0,0,0.2);
    }
    .prediction-box .price {
        font-family: 'DM Serif Display', serif;
        font-size: 2.8rem; color: #f0c040; margin: 0;
    }
    .prediction-box .label {
        color: #a0aec0; font-size: 0.95rem;
        letter-spacing: 1px; text-transform: uppercase; margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebarContent"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
    div[data-testid="stSidebarContent"] * { color: #e2e8f0 !important; }
    div[data-testid="stSidebarContent"] h1,
    div[data-testid="stSidebarContent"] h2,
    div[data-testid="stSidebarContent"] h3 { color: #f0c040 !important; }
    .section-header {
        font-family: 'DM Serif Display', serif; font-size: 1.3rem; color: #1a1a2e;
        border-bottom: 2px solid #f0c040; padding-bottom: 0.4rem;
        margin: 1.5rem 0 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE     = ["#1a1a2e", "#f0c040", "#0f3460", "#e94560", "#533483", "#2b9348"]
PROX_ORDER  = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
PROX_COLORS = dict(zip(PROX_ORDER, PALETTE))

# ── Model & data ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    BASE_DIR = Path(__file__).parent
    MODEL_PATH = BASE_DIR / "model" / "housing_model.pkl"
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_housing.csv")
    def label(row):
        if row["ocean_proximity_INLAND"]:     return "INLAND"
        if row["ocean_proximity_ISLAND"]:     return "ISLAND"
        if row["ocean_proximity_NEAR BAY"]:   return "NEAR BAY"
        if row["ocean_proximity_NEAR OCEAN"]: return "NEAR OCEAN"
        return "<1H OCEAN"
    df["ocean_proximity"] = df.apply(label, axis=1)
    return df

model = load_model()
df    = load_data()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 House Details")
    st.markdown("---")
    st.markdown("### 📍 Location")
    longitude = st.slider("Longitude", -124.0, -114.0, -118.25, 0.01,
                          help="Western ≈ -122, Southern ≈ -118")
    latitude  = st.slider("Latitude",  32.0,   42.0,   34.05,  0.01,
                          help="Northern ≈ 40, Southern ≈ 33")
    ocean_proximity = st.selectbox("Ocean Proximity", PROX_ORDER)

    st.markdown("---")
    st.markdown("### 🏘️ Neighborhood")
    housing_median_age = st.slider("Median House Age (yrs)", 1, 52, 20)
    median_income      = st.slider("Median Income (×$10k)", 0.5, 15.0, 4.0, 0.1)
    population         = st.number_input("Block Population", 10, 35000, 1500, 50)
    households         = st.number_input("Households", 5, 6000, 500, 10)

    st.markdown("---")
    st.markdown("### 🛏️ Housing Details")
    total_rooms    = st.number_input("Total Rooms in Block",    10, 40000, 2500, 50)
    total_bedrooms = st.number_input("Total Bedrooms in Block",  5,  7000,  500,  10)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Price", use_container_width=True, type="primary")

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0'>🏠 House Price Predictor</h1>",
            unsafe_allow_html=True)
st.caption("Adjust inputs in the sidebar → explore the dataset → predict your home's value.")

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_btn:
    inland     = 1 if ocean_proximity == "INLAND"     else 0
    island     = 1 if ocean_proximity == "ISLAND"     else 0
    near_bay   = 1 if ocean_proximity == "NEAR BAY"   else 0
    near_ocean = 1 if ocean_proximity == "NEAR OCEAN" else 0

    features   = np.array([[longitude, latitude, housing_median_age,
                             total_rooms, total_bedrooms, population, households,
                             median_income, inland, island, near_bay, near_ocean]])
    prediction = max(model.predict(features)[0], 0)

    st.markdown(f"""
    <div class="prediction-box">
        <p class="label">Estimated House Value</p>
        <p class="price">${prediction:,.0f}</p>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rooms / Household",    f"{total_rooms/max(households,1):.1f}")
    c2.metric("Bedrooms / Household", f"{total_bedrooms/max(households,1):.1f}")
    c3.metric("People / Household",   f"{population/max(households,1):.1f}")
    st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
# CHARTS  (all matplotlib)
# ════════════════════════════════════════════════════════════════════════════

fmt_k = plt.FuncFormatter(lambda x, _: f"${x:.0f}k")

# ── Row 1 : Histogram | Box plot ─────────────────────────────────────────────
st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.hist(df["median_house_value"] / 1_000, bins=60,
            color=PALETTE[0], edgecolor=PALETTE[1], linewidth=0.4, alpha=0.9)
    ax.set_xlabel("House Value ($k)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of House Values")
    ax.xaxis.set_major_formatter(fmt_k)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col2:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    vdata = [df.loc[df["ocean_proximity"] == p, "median_house_value"].values / 1_000
             for p in PROX_ORDER]
    bp = ax.boxplot(vdata, patch_artist=True,
                    medianprops=dict(color="#f0c040", linewidth=2),
                    whiskerprops=dict(color="#4a5568"),
                    capprops=dict(color="#4a5568"),
                    flierprops=dict(marker=".", markersize=2, alpha=0.3, color="#4a5568"))
    for patch, color in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticks(range(1, len(PROX_ORDER) + 1))
    ax.set_xticklabels(PROX_ORDER, rotation=15, ha="right")
    ax.set_ylabel("House Value ($k)")
    ax.set_title("House Value by Ocean Proximity")
    ax.yaxis.set_major_formatter(fmt_k)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Row 2 : Income scatter | Age grouped bar ─────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    sample = df.sample(min(2000, len(df)), random_state=42)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for prox in PROX_ORDER:
        sub = sample[sample["ocean_proximity"] == prox]
        ax.scatter(sub["median_income"], sub["median_house_value"] / 1_000,
                   color=PROX_COLORS[prox], s=8, alpha=0.5, label=prox)
    ax.set_xlabel("Median Income (×$10k)")
    ax.set_ylabel("House Value ($k)")
    ax.set_title("Income vs. House Value")
    ax.yaxis.set_major_formatter(fmt_k)
    ax.legend(fontsize=7, markerscale=1.8, framealpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col4:
    df["age_bin"] = pd.cut(df["housing_median_age"], bins=8)
    age_agg = (df.groupby(["age_bin", "ocean_proximity"], observed=True)["median_house_value"]
               .mean().unstack("ocean_proximity").fillna(0) / 1_000)
    x = np.arange(len(age_agg))
    n = len(age_agg.columns)
    width = 0.8 / n

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for i, prox in enumerate(age_agg.columns):
        offset = (i - n / 2 + 0.5) * width
        ax.bar(x + offset, age_agg[prox], width=width * 0.9,
               color=PROX_COLORS.get(prox, "#888"), alpha=0.85, label=prox)
    ax.set_xticks(x)
    ax.set_xticklabels([str(b) for b in age_agg.index], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Avg Value ($k)")
    ax.set_title("Avg House Value by Age & Proximity")
    ax.yaxis.set_major_formatter(fmt_k)
    ax.legend(fontsize=7, framealpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Row 3 : Geo scatter | Correlation bar ────────────────────────────────────
st.markdown('<p class="section-header">🗺️ Geographic & Correlation Insights</p>',
            unsafe_allow_html=True)
col5, col6 = st.columns([3, 2])

with col5:
    map_sample = df.sample(min(5000, len(df)), random_state=7)
    fig, ax = plt.subplots(figsize=(7, 5))
    sc = ax.scatter(map_sample["longitude"], map_sample["latitude"],
                    c=map_sample["median_house_value"] / 1_000,
                    cmap="YlOrRd", s=5, alpha=0.55, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label("House Value ($k)", fontsize=8)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("House Values Across California")
    ax.set_facecolor("#eaf0fb")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col6:
    num_cols = ["median_income", "housing_median_age", "total_rooms",
                "total_bedrooms", "population", "households"]
    corr = (df[num_cols + ["median_house_value"]]
            .corr()["median_house_value"]
            .drop("median_house_value")
            .sort_values())
    nice = {
        "median_income":      "Median Income",
        "housing_median_age": "House Age",
        "total_rooms":        "Total Rooms",
        "total_bedrooms":     "Total Bedrooms",
        "population":         "Population",
        "households":         "Households",
    }
    colors = [PALETTE[3] if v < 0 else PALETTE[0] for v in corr.values]

    fig, ax = plt.subplots(figsize=(5, 3.8))
    bars = ax.barh([nice.get(k, k) for k in corr.index], corr.values,
                   color=colors, alpha=0.88, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, corr.values):
        ax.text(val + (0.01 if val >= 0 else -0.01),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center",
                ha="left" if val >= 0 else "right", fontsize=8)
    ax.axvline(0, color="#4a5568", linewidth=0.8)
    ax.set_xlabel("Correlation with House Value")
    ax.set_title("Feature Correlations")
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE[0], label="Positive"),
        mpatches.Patch(color=PALETTE[3], label="Negative"),
    ], fontsize=7, framealpha=0.7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── Row 4 : Population scatter | Violin ──────────────────────────────────────
col7, col8 = st.columns(2)

with col7:
    pop_sample = df[df["population"] < 10000].sample(min(2000, len(df)), random_state=1)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    sc2 = ax.scatter(pop_sample["population"], pop_sample["median_house_value"] / 1_000,
                     c=pop_sample["median_income"], cmap="viridis",
                     s=8, alpha=0.55, linewidths=0)
    cbar2 = fig.colorbar(sc2, ax=ax, shrink=0.85)
    cbar2.set_label("Income (×$10k)", fontsize=8)
    ax.set_xlabel("Block Population")
    ax.set_ylabel("House Value ($k)")
    ax.set_title("Population vs. House Value\n(coloured by Income)")
    ax.yaxis.set_major_formatter(fmt_k)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with col8:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    vparts = ax.violinplot(
        [df.loc[df["ocean_proximity"] == p, "median_income"].dropna().values
         for p in PROX_ORDER],
        positions=range(len(PROX_ORDER)),
        showmedians=True, showextrema=False,
    )
    for body, color in zip(vparts["bodies"], PALETTE):
        body.set_facecolor(color)
        body.set_alpha(0.75)
    vparts["cmedians"].set_color("#f0c040")
    vparts["cmedians"].set_linewidth(2)
    ax.set_xticks(range(len(PROX_ORDER)))
    ax.set_xticklabels(PROX_ORDER, rotation=15, ha="right")
    ax.set_ylabel("Median Income (×$10k)")
    ax.set_title("Income Distribution by Ocean Proximity")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption(
    "📚 Model: Linear Regression trained on the California Housing Dataset. "
    "Predictions are estimates only and may not reflect actual market prices."
)
