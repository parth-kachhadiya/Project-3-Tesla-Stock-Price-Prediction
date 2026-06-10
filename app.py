import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title  = "TSLA Intelligence",
    page_icon   = "⚡",
    layout      = "wide",
    initial_sidebar_state = "expanded"
)

# ══════════════════════════════════════════════════════════════
# THEME CONSTANTS
# ══════════════════════════════════════════════════════════════

TESLA_RED = "#E31937"
GOLD      = "#FFD700"
CYAN      = "#00D4FF"
GREEN     = "#00FF88"
PURPLE    = "#BF5FFF"
ORANGE    = "#FF8C00"
BG        = "#0f0f0f"
CARD_BG   = "#1a1a2e"

# ══════════════════════════════════════════════════════════════
# GLOBAL STYLE
# ══════════════════════════════════════════════════════════════

st.markdown(f"""
<style>
  .stApp {{ background-color: {BG}; color: #e0e0e0; }}
  section[data-testid="stSidebar"] {{ background-color: #111122; }}

  .metric-card {{
    background: {CARD_BG};
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
  }}
  .metric-label {{ font-size: 11px; color: #888; letter-spacing: 1.5px; text-transform: uppercase; }}
  .metric-value {{ font-size: 26px; font-weight: 700; margin-top: 4px; }}

  .section-header {{
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding-bottom: 6px;
    margin-bottom: 24px;
    border-bottom: 2px solid {TESLA_RED};
    color: #ffffff;
  }}

  .graph-card {{
    background: {CARD_BG};
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
  }}

  .pred-row {{
    display: flex;
    justify-content: space-between;
    background: #16213e;
    border-radius: 8px;
    padding: 10px 16px;
    margin: 6px 0;
    border-left: 3px solid {GOLD};
  }}
  .pred-up   {{ border-left-color: {GREEN}; }}
  .pred-down {{ border-left-color: {TESLA_RED}; }}

  .model-badge {{
    display: inline-block;
    background: {TESLA_RED};
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1px;
    text-transform: uppercase;
  }}

  .tsla-divider {{
    height: 1px;
    background: linear-gradient(to right, transparent, {TESLA_RED}, transparent);
    margin: 28px 0;
  }}

  div[data-testid="stRadio"] label {{
    font-size: 14px;
    color: #ccc;
    padding: 6px 0;
  }}

  #MainMenu, footer {{ visibility: hidden; }}
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor" : BG,
    "axes.facecolor"   : CARD_BG,
    "axes.labelcolor"  : "#e0e0e0",
    "xtick.color"      : "#e0e0e0",
    "ytick.color"      : "#e0e0e0",
    "text.color"       : "#e0e0e0",
    "grid.color"       : "#2a2a4a",
    "grid.linewidth"   : 0.5,
    "figure.dpi"       : 110,
    "axes.spines.top"  : False,
    "axes.spines.right": False,
})

# ══════════════════════════════════════════════════════════════
# DATA LOADER
# ══════════════════════════════════════════════════════════════

@st.cache_data
def load_data(filepath="data/TSLA.csv"):
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df.sort_values("Date", inplace=True)
    df.set_index("Date", inplace=True)
    df["Daily_Range"]  = df["High"] - df["Low"]
    df["Daily_Return"] = df["Adj Close"].pct_change()
    df["Bullish"]      = df["Close"] > df["Open"]
    df["Year"]         = df.index.year
    df["Month"]        = df.index.month
    df["RollingStd30"] = df["Adj Close"].rolling(30).std()
    rolling_max        = df["Adj Close"].cummax()
    df["Drawdown"]     = (df["Adj Close"] - rolling_max) / rolling_max * 100
    df["Vol_MA20"]     = df["Volume"].rolling(20).mean()
    df["Vol_Spike"]    = df["Volume"] > (df["Vol_MA20"] * 1.5)
    return df

# ══════════════════════════════════════════════════════════════
# MODEL LOADER
# ══════════════════════════════════════════════════════════════

@st.cache_resource
def load_models():
    try:
        import joblib, json
        from tensorflow.keras.models import load_model
        scaler  = joblib.load("models/shared_scaler.pkl")
        model_a = load_model("models/model_a.keras")
        model_b = load_model("models/model_b.keras")
        model_c = load_model("models/model_c.keras")
        with open("models/metadata.json") as f:
            meta = json.load(f)
        return scaler, model_a, model_b, model_c, meta, True
    except Exception as e:
        return None, None, None, None, None, False

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding: 10px 0 20px'>
      <div style='font-size:42px'>⚡</div>
      <div style='font-size:20px; font-weight:800; color:{TESLA_RED}; letter-spacing:3px'>TSLA</div>
      <div style='font-size:11px; color:#888; letter-spacing:2px'>INTELLIGENCE DASHBOARD</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    section = st.radio(
        "NAVIGATE",
        ["📊  EDA", "🔮  Prediction", "🧠  Model Details"],
        label_visibility="collapsed"
    )

    st.markdown("---")
    uploaded = st.file_uploader("Upload TSLA CSV", type=["csv"])
    st.markdown(
        f"<div style='font-size:10px; color:#555; text-align:center; margin-top:20px'>"
        f"Built with ⚡ for Tesla Stock Analysis</div>",
        unsafe_allow_html=True
    )

# ── Load data ─────────────────────────────────────────────────
filepath = "data/TSLA.csv"
if uploaded:
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
    tmp.write(uploaded.read())
    tmp.close()
    filepath = tmp.name

try:
    df = load_data(filepath)
    data_ok = True
except Exception as e:
    st.error(f"Could not load data: {e}. Please upload a TSLA CSV file.")
    data_ok = False

# ══════════════════════════════════════════════════════════════
# SECTION 1 — EDA
# ══════════════════════════════════════════════════════════════

if section == "📊  EDA" and data_ok:

    st.markdown('<div class="section-header">📊 Exploratory Data Analysis</div>', unsafe_allow_html=True)

    # ── KPI strip ─────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("TOTAL DAYS",    f"{len(df):,}",                                CYAN),
        ("DATE RANGE",    f"{df.index.min().year}–{df.index.max().year}", GOLD),
        ("ALL TIME HIGH", f"${df['Adj Close'].max():.2f}",                GREEN),
        ("ALL TIME LOW",  f"${df['Adj Close'].min():.2f}",                TESLA_RED),
        ("MAX DRAWDOWN",  f"{df['Drawdown'].min():.1f}%",                 PURPLE),
    ]
    for col, (label, val, color) in zip([k1, k2, k3, k4, k5], kpis):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}">{val}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="tsla-divider"></div>', unsafe_allow_html=True)

    # ── Graph selector ────────────────────────────────────────
    graph_options = {
        "1. Adj Close Trend"       : "trend",
        "2. Yearly Boxplot"        : "boxplot",
        "3. Volume vs Price Range" : "volume_range",
        "4. Bullish vs Bearish"    : "sentiment",
        "5. Volume Spike vs Price" : "spike",
        "6. Monthly Heatmap"       : "heatmap",
        "7. Correlation Heatmap"   : "corr",
        "8. Drawdown Analysis"     : "drawdown",
        "9. Rolling Volatility"    : "volatility",
        "10. Quarterly Analysis"   : "quarterly",
    }

    selected_graphs = st.multiselect(
        "Select graphs to display",
        list(graph_options.keys()),
        default=["1. Adj Close Trend", "4. Bullish vs Bearish", "6. Monthly Heatmap"]
    )

    st.markdown("---")

    for graph_name in selected_graphs:
        key = graph_options[graph_name]
        st.markdown('<div class="graph-card">', unsafe_allow_html=True)

        # ── G1 ────────────────────────────────────────────────
        if key == "trend":
            st.markdown(f"**{graph_name}** — How did Tesla's Adj Close price grow?")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Adj Close"],
                fill='tozeroy', fillcolor='rgba(227,25,55,0.12)',
                line=dict(color=TESLA_RED, width=1.8), name="Adj Close"
            ))
            max_idx = df["Adj Close"].idxmax()
            min_idx = df["Adj Close"].idxmin()
            fig.add_annotation(x=max_idx, y=df["Adj Close"].max(),
                text=f"Peak ${df['Adj Close'].max():.1f}",
                showarrow=True, arrowcolor=GOLD, font=dict(color=GOLD))
            fig.add_annotation(x=min_idx, y=df["Adj Close"].min(),
                text=f"Low ${df['Adj Close'].min():.2f}",
                showarrow=True, arrowcolor=CYAN, font=dict(color=CYAN))
            fig.update_layout(
                title="Tesla (TSLA) — Adj Close Price",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=400,
                xaxis=dict(gridcolor="#2a2a4a"),
                yaxis=dict(gridcolor="#2a2a4a", title="Price (USD)")
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G2 ────────────────────────────────────────────────
        elif key == "boxplot":
            st.markdown(f"**{graph_name}** — Which years had extreme price volatility?")
            fig    = go.Figure()
            years  = sorted(df["Year"].unique())
            colors = px.colors.sequential.Plasma
            for i, yr in enumerate(years):
                yr_data = df[df["Year"] == yr]["Adj Close"].dropna()
                fig.add_trace(go.Box(
                    y=yr_data, name=str(yr),
                    marker_color=colors[i % len(colors)],
                    line_color=GOLD
                ))
            fig.update_layout(
                title="Price Distribution Per Year",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=420,
                yaxis=dict(gridcolor="#2a2a4a", title="Adj Close (USD)"),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G3 ────────────────────────────────────────────────
        elif key == "volume_range":
            st.markdown(f"**{graph_name}** — Does high volume = bigger price swing?")
            clean = df[["Volume", "Daily_Range", "Adj Close"]].dropna()
            corr  = clean["Volume"].corr(clean["Daily_Range"])
            fig   = go.Figure()
            fig.add_trace(go.Scatter(
                x=clean["Volume"], y=clean["Daily_Range"],
                mode='markers',
                marker=dict(
                    color=clean["Adj Close"], colorscale='Plasma',
                    size=4, opacity=0.5,
                    colorbar=dict(title="Adj Close")
                ), name="Days"
            ))
            z      = np.polyfit(clean["Volume"], clean["Daily_Range"], 1)
            x_line = np.linspace(clean["Volume"].min(), clean["Volume"].max(), 200)
            fig.add_trace(go.Scatter(
                x=x_line, y=np.poly1d(z)(x_line),
                line=dict(color=GOLD, dash='dash', width=2),
                name=f"Trend (r={corr:.3f})"
            ))
            fig.update_layout(
                title=f"Volume vs Daily Price Range  (r = {corr:.3f})",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=420,
                xaxis=dict(gridcolor="#2a2a4a", title="Volume"),
                yaxis=dict(gridcolor="#2a2a4a", title="Daily Range (USD)")
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G4 ────────────────────────────────────────────────
        elif key == "sentiment":
            st.markdown(f"**{graph_name}** — Market sentiment overview")
            c1, c2 = st.columns(2)
            counts  = df["Bullish"].value_counts()
            with c1:
                fig_pie = go.Figure(go.Pie(
                    labels=["Bullish", "Bearish"],
                    values=[counts.get(True, 0), counts.get(False, 0)],
                    marker=dict(colors=[GREEN, TESLA_RED]),
                    hole=0.4
                ))
                fig_pie.update_layout(
                    title="Overall Sentiment",
                    paper_bgcolor=BG, font_color="#e0e0e0", height=350
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            with c2:
                yearly = df.groupby(["Year", "Bullish"]).size().unstack(fill_value=0)
                fig_bar = go.Figure()
                if False in yearly.columns:
                    fig_bar.add_trace(go.Bar(
                        x=yearly.index, y=yearly[False],
                        name="Bearish", marker_color=TESLA_RED
                    ))
                if True in yearly.columns:
                    fig_bar.add_trace(go.Bar(
                        x=yearly.index, y=yearly[True],
                        name="Bullish", marker_color=GREEN
                    ))
                fig_bar.update_layout(
                    barmode='stack', title="Yearly Sentiment",
                    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                    font_color="#e0e0e0", height=350,
                    yaxis=dict(gridcolor="#2a2a4a")
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        # ── G5 ────────────────────────────────────────────────
        elif key == "spike":
            st.markdown(f"**{graph_name}** — Did volume spikes precede price moves?")
            c1, c2 = st.columns(2)
            min_d   = df.index.min().date()
            max_d   = df.index.max().date()
            with c1:
                start = st.date_input("Start date",
                    value=pd.to_datetime("2018-01-01").date(),
                    min_value=min_d, max_value=max_d, key="spike_start")
            with c2:
                end = st.date_input("End date",
                    value=max_d,
                    min_value=min_d, max_value=max_d, key="spike_end")
            plot_df = df.loc[str(start):str(end)]
            if not plot_df.empty:
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                fig.add_trace(go.Scatter(
                    x=plot_df.index, y=plot_df["Adj Close"],
                    line=dict(color=TESLA_RED, width=1.8), name="Adj Close"
                ), secondary_y=False)
                fig.add_trace(go.Bar(
                    x=plot_df.index, y=plot_df["Volume"],
                    marker_color=np.where(plot_df["Vol_Spike"], ORANGE, CYAN),
                    opacity=0.4, name="Volume"
                ), secondary_y=True)
                spike_count = plot_df["Vol_Spike"].sum()
                fig.update_layout(
                    title=f"Volume Spikes vs Price  |  {spike_count} spike days",
                    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                    font_color="#e0e0e0", height=420
                )
                st.plotly_chart(fig, use_container_width=True)

        # ── G6 ────────────────────────────────────────────────
        elif key == "heatmap":
            st.markdown(f"**{graph_name}** — Seasonal price patterns month by month")
            pivot = df.groupby(["Year", "Month"])["Adj Close"].mean().unstack()
            pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                             "Jul","Aug","Sep","Oct","Nov","Dec"]
            fig = go.Figure(go.Heatmap(
                z=pivot.values,
                x=pivot.columns.tolist(),
                y=pivot.index.tolist(),
                colorscale="RdYlGn",
                text=np.round(pivot.values, 0),
                texttemplate="%{text}",
                textfont=dict(size=10),
                colorbar=dict(title="Avg Adj Close")
            ))
            fig.update_layout(
                title="Monthly Avg Price Heatmap (Year × Month)",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=420,
                xaxis=dict(title="Month"),
                yaxis=dict(title="Year")
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G7 ────────────────────────────────────────────────
        elif key == "corr":
            st.markdown(f"**{graph_name}** — Feature correlation matrix")
            features    = ["Open","High","Low","Close","Adj Close","Volume","Daily_Range"]
            corr_matrix = df[features].corr()
            fig = go.Figure(go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns.tolist(),
                y=corr_matrix.columns.tolist(),
                colorscale="RdBu", zmid=0,
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                textfont=dict(size=11),
                colorbar=dict(title="Correlation")
            ))
            fig.update_layout(
                title="Feature Correlation Heatmap",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=480
            )
            st.plotly_chart(fig, use_container_width=True)
            vol_corr   = corr_matrix.loc["Volume", "Adj Close"]
            range_corr = corr_matrix.loc["Daily_Range", "Adj Close"]
            st.info(
                f"💡 Volume ↔ Adj Close: **{vol_corr:.3f}**  "
                f"|  Daily_Range ↔ Adj Close: **{range_corr:.3f}**"
            )

        # ── G8 ────────────────────────────────────────────────
        elif key == "drawdown":
            st.markdown(f"**{graph_name}** — How far did Tesla fall from its peak?")
            rolling_max = df["Adj Close"].cummax()
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.6, 0.4], vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"],
                line=dict(color=TESLA_RED, width=1.3), name="Adj Close"),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=rolling_max,
                line=dict(color=GOLD, dash='dash', width=1), name="Rolling Peak"),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["Drawdown"],
                fill='tozeroy', fillcolor='rgba(191,95,255,0.2)',
                line=dict(color=PURPLE, width=1), name="Drawdown %"),
                row=2, col=1)
            max_dd_idx = df["Drawdown"].idxmin()
            fig.add_annotation(
                x=max_dd_idx, y=df["Drawdown"].min(),
                text=f"Max {df['Drawdown'].min():.1f}%",
                showarrow=True, arrowcolor=ORANGE, font=dict(color=ORANGE)
            )
            fig.update_layout(
                title="Drawdown Analysis — Crash & Recovery",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=480
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G9 ────────────────────────────────────────────────
        elif key == "volatility":
            st.markdown(f"**{graph_name}** — Calm vs turbulent regimes")
            threshold = df["RollingStd30"].quantile(0.75)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                row_heights=[0.5, 0.5], vertical_spacing=0.05)
            fig.add_trace(go.Scatter(x=df.index, y=df["Adj Close"],
                fill='tozeroy', fillcolor='rgba(227,25,55,0.1)',
                line=dict(color=TESLA_RED, width=1.2), name="Adj Close"),
                row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df["RollingStd30"],
                fill='tozeroy', fillcolor='rgba(191,95,255,0.15)',
                line=dict(color=PURPLE, width=1.2), name="30D Volatility"),
                row=2, col=1)
            fig.add_hline(y=threshold,
                line=dict(color=GOLD, dash='dash'),
                annotation_text=f"75th pct: {threshold:.2f}",
                row=2, col=1)
            fig.update_layout(
                title="Price vs Rolling 30-Day Volatility",
                paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                font_color="#e0e0e0", height=480
            )
            st.plotly_chart(fig, use_container_width=True)

        # ── G10 ───────────────────────────────────────────────
        elif key == "quarterly":
            st.markdown(f"**{graph_name}** — Quarter by quarter performance")
            available_years = sorted(df["Year"].unique().tolist())
            selected_years  = st.multiselect(
                "Select years", available_years,
                default=available_years, key="qtr_years"
            )
            if selected_years:
                filtered            = df[df["Year"].isin(selected_years)].copy()
                filtered["Quarter"] = filtered.index.quarter
                filtered["Q_Label"] = (filtered["Year"].astype(str)
                                       + " Q" + filtered["Quarter"].astype(str))
                qtr = filtered.groupby("Q_Label").agg(
                    Avg_Close   = ("Adj Close", "mean"),
                    Avg_Volume  = ("Volume",    "mean"),
                    Close_Price = ("Adj Close", "last"),
                ).reset_index()
                qtr["QoQ_Return"] = qtr["Close_Price"].pct_change() * 100
                qtr["Q_Num"]      = qtr["Q_Label"].str[-1].astype(int)
                Q_COLORS   = {1: CYAN, 2: GOLD, 3: GREEN, 4: TESLA_RED}
                bar_colors = qtr["Q_Num"].map(Q_COLORS).tolist()

                fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                                    row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.06)
                fig.add_trace(go.Bar(x=qtr["Q_Label"], y=qtr["Avg_Close"],
                    marker_color=bar_colors, name="Avg Close"), row=1, col=1)
                fig.add_trace(go.Bar(x=qtr["Q_Label"], y=qtr["Avg_Volume"] / 1e6,
                    marker_color=bar_colors, name="Volume (M)", opacity=0.8), row=2, col=1)
                qoq_colors = [GREEN if v >= 0 else TESLA_RED
                              for v in qtr["QoQ_Return"].fillna(0)]
                fig.add_trace(go.Bar(x=qtr["Q_Label"], y=qtr["QoQ_Return"].fillna(0),
                    marker_color=qoq_colors, name="QoQ Return %"), row=3, col=1)
                fig.update_layout(
                    title="Quarterly Analysis  |  Q1=Cyan  Q2=Gold  Q3=Green  Q4=Red",
                    paper_bgcolor=BG, plot_bgcolor=CARD_BG,
                    font_color="#e0e0e0", height=600, showlegend=False
                )
                st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECTION 2 — PREDICTION
# ══════════════════════════════════════════════════════════════

elif section == "🔮  Prediction" and data_ok:

    st.markdown('<div class="section-header">🔮 Price Prediction</div>', unsafe_allow_html=True)

    scaler, model_a, model_b, model_c, meta, models_ok = load_models()

    if not models_ok:
        st.warning("⚠️ Model files not found in `models/` folder. Please train and save models first.")
        st.stop()

    WINDOW = meta["window_size"]

    # ── Layout ────────────────────────────────────────────────
    left, right = st.columns([1, 2], gap="large")

    with left:
        st.markdown(
            f"<div style='color:{GOLD}; font-weight:700; font-size:14px;"
            f"letter-spacing:1px'>SELECT PREDICTION TYPE</div>",
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

        model_choice = st.radio(
            "Model",
            ["⚡ Model A — Next Day", "📅 Model B — Up to 5 Days", "📆 Model C — Up to 21 Days"],
            label_visibility="collapsed"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if "Model A" in model_choice:
            selected_model = model_a
            days           = 1
            model_label    = "Model A"
            model_color    = CYAN
            st.markdown(f"""
            <div class="model-badge" style="background:{CYAN}; color:#000">MODEL A</div>
            <div style='color:#888; font-size:12px; margin-top:8px'>
              Predicts <b style='color:{CYAN}'>next trading day</b> only.<br>
              Highest accuracy. Direct single-step output.
            </div>""", unsafe_allow_html=True)

        elif "Model B" in model_choice:
            selected_model = model_b
            model_label    = "Model B"
            model_color    = GOLD
            st.markdown(f"""
            <div class="model-badge" style="background:{GOLD}; color:#000">MODEL B</div>
            <div style='color:#888; font-size:12px; margin-top:8px'>
              Predicts up to <b style='color:{GOLD}'>5 trading days</b> ahead.<br>
              Trained for multi-step direct output.
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            days = st.slider("Days to predict", min_value=1, max_value=5, value=3, key="days_b")

        else:
            selected_model = model_c
            model_label    = "Model C"
            model_color    = PURPLE
            st.markdown(f"""
            <div class="model-badge" style="background:{PURPLE}; color:#fff">MODEL C</div>
            <div style='color:#888; font-size:12px; margin-top:8px'>
              Predicts up to <b style='color:{PURPLE}'>21 trading days</b> ahead.<br>
              Hypothetical long-term view.
            </div>""", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            days = st.slider("Days to predict", min_value=1, max_value=21, value=10, key="days_c")

        # ── Display style selector ────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='color:{GOLD}; font-weight:700; font-size:12px;"
            f"letter-spacing:1px'>FORECAST DISPLAY STYLE</div>",
            unsafe_allow_html=True
        )
        display_style = st.radio(
            "display",
            ["📍 Points", "📈 Line", "〰️ Smooth Line"],
            label_visibility="collapsed",
            horizontal=True,
            key="display_style"
        )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡  PREDICT", use_container_width=True, type="primary")

    # ── Base chart builder ────────────────────────────────────
    def draw_base_chart(forecast_dates=None, forecast_prices=None,
                        f_color=CYAN, display_style="📍 Points"):
        fig = go.Figure()

        # Historical line
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Adj Close"],
            line=dict(color=TESLA_RED, width=1.5),
            fill='tozeroy', fillcolor='rgba(227,25,55,0.08)',
            name="Historical"
        ))

        if forecast_dates and forecast_prices:
            last_date  = df.index[-1]
            last_price = df["Adj Close"].iloc[-1]
            connect_x  = [last_date] + forecast_dates
            connect_y  = [last_price] + forecast_prices

            if display_style == "📍 Points":
                # Dashed connector from last actual to first point
                fig.add_trace(go.Scatter(
                    x=[last_date, forecast_dates[0]],
                    y=[last_price, forecast_prices[0]],
                    mode='lines',
                    line=dict(color=f_color, width=1, dash='dot'),
                    showlegend=False
                ))
                # Forecast points
                fig.add_trace(go.Scatter(
                    x=forecast_dates, y=forecast_prices,
                    mode='markers',
                    marker=dict(
                        size=10, color=f_color,
                        symbol='circle',
                        line=dict(color='white', width=1.5)
                    ),
                    name="Forecast Points"
                ))

            elif display_style == "📈 Line":
                # Solid distinct color line
                fig.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    mode='lines+markers',
                    line=dict(color=f_color, width=2.5),
                    marker=dict(size=6, color=f_color,
                                line=dict(color='white', width=1)),
                    name="Forecast Line"
                ))

            elif display_style == "〰️ Smooth Line":
                # Spline smooth line
                fig.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    mode='lines',
                    line=dict(
                        color=f_color,
                        width=2.5,
                        shape='spline',
                        smoothing=1.3
                    ),
                    name="Forecast Smooth"
                ))

        fig.update_layout(
            title="TSLA — Adj Close  |  Full Dataset + Forecast",
            paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font_color="#e0e0e0", height=460,
            xaxis=dict(gridcolor="#2a2a4a", title="Date"),
            yaxis=dict(gridcolor="#2a2a4a", title="Adj Close (USD)"),
            legend=dict(bgcolor='rgba(0,0,0,0)', x=0.01, y=0.99),
            margin=dict(l=10, r=10, t=50, b=10)
        )
        return fig

    # ── Right panel — initial chart ───────────────────────────
    with right:
        chart_placeholder = st.empty()
        chart_placeholder.plotly_chart(
            draw_base_chart(display_style=display_style),
            use_container_width=True
        )

    # ── Prediction logic ──────────────────────────────────────
    if predict_btn:
        with left:
            with st.spinner("Running model..."):
                price_data  = df[["Adj Close"]].values
                last_window = price_data[-WINDOW:]
                scaled      = scaler.transform(last_window)
                X           = scaled.reshape(1, WINDOW, 1)

                pred_scaled = selected_model.predict(X, verbose=0)
                pred_all    = scaler.inverse_transform(
                                  pred_scaled.reshape(-1, 1)
                              ).flatten()
                predicted   = pred_all[:days]

            # Future trading dates
            from datetime import timedelta
            last_date    = df.index[-1]
            future_dates = []
            current      = last_date
            while len(future_dates) < days:
                current += timedelta(days=1)
                if current.weekday() < 5:
                    future_dates.append(current)

            last_price = df["Adj Close"].iloc[-1]

        # ── Animated chart ────────────────────────────────────
        with right:
            anim_dates  = []
            anim_prices = []
            for fdate, fprice in zip(future_dates, predicted):
                anim_dates.append(fdate)
                anim_prices.append(round(float(fprice), 2))
                chart_placeholder.plotly_chart(
                    draw_base_chart(
                        anim_dates, anim_prices,
                        f_color=model_color,
                        display_style=display_style
                    ),
                    use_container_width=True
                )
                time.sleep(0.35)

        # ── Results table ─────────────────────────────────────
        with left:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(
                f"<div style='color:{GOLD}; font-weight:700; font-size:13px;"
                f"letter-spacing:1px'>FORECAST RESULTS</div>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<div style='color:#666; font-size:11px; margin-bottom:12px'>"
                f"Last close: ${last_price:.2f}</div>",
                unsafe_allow_html=True
            )

            prev = last_price
            for i, (fdate, fprice) in enumerate(zip(future_dates, predicted), 1):
                fprice = round(float(fprice), 2)
                change = round(fprice - prev, 2)
                pct    = round((change / prev) * 100, 2)
                arrow  = "▲" if change >= 0 else "▼"
                clr    = GREEN if change >= 0 else TESLA_RED
                cls    = "pred-up" if change >= 0 else "pred-down"
                sign   = "+" if change >= 0 else ""
                st.markdown(f"""
                <div class="pred-row {cls}">
                  <span style='color:#888; font-size:12px'>Day {i} &nbsp; {fdate.strftime('%b %d')}</span>
                  <span style='font-weight:700'>${fprice}</span>
                  <span style='color:{clr}; font-size:13px'>{arrow} {sign}{change} ({sign}{pct}%)</span>
                </div>""", unsafe_allow_html=True)
                prev = fprice

            total     = round(float(predicted[-1]) - last_price, 2)
            total_pct = round((total / last_price) * 100, 2)
            sign      = "+" if total >= 0 else ""
            clr       = GREEN if total >= 0 else TESLA_RED
            st.markdown(f"""
            <div style='margin-top:12px; padding:10px 16px; background:#0d0d1a;
                        border-radius:8px; border:1px solid #2a2a4a;'>
              <span style='color:#888; font-size:11px'>TOTAL CHANGE</span>
              <span style='color:{clr}; font-weight:700; float:right'>
                {sign}{total} ({sign}{total_pct}%)
              </span>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# SECTION 3 — MODEL DETAILS
# ══════════════════════════════════════════════════════════════

elif section == "🧠  Model Details" and data_ok:

    st.markdown(
        '<div class="section-header">🧠 Model Architecture & Performance</div>',
        unsafe_allow_html=True
    )

    _, _, _, _, meta, models_ok = load_models()

    # ── Architecture cards ────────────────────────────────────
    st.markdown("#### Model Architecture")
    m1, m2, m3 = st.columns(3)

    arch_info = [
        ("Model A", "Next Day",   CYAN,   1,  "Highest accuracy\nDirect 1-step output"),
        ("Model B", "Up to 5D",   GOLD,   5,  "Medium horizon\nDirect 5-step output"),
        ("Model C", "Up to 21D",  PURPLE, 21, "Long horizon\nDirect 21-step output"),
    ]

    for col, (name, horizon, color, steps, note) in zip([m1, m2, m3], arch_info):
        txt_color = '#000' if color == GOLD else '#fff'
        col.markdown(f"""
        <div class="metric-card" style="border-color:{color}40; text-align:left">
          <div class="model-badge" style="background:{color}; color:{txt_color}">{name}</div>
          <div style='color:{color}; font-size:13px; margin-top:10px; font-weight:600'>{horizon}</div>
          <div style='color:#888; font-size:11px; margin-top:8px; line-height:1.8'>
            LSTM(64) → return_sequences=True<br>
            LSTM(32) → return_sequences=False<br>
            Dense(32, relu)<br>
            Dense(<b style='color:{color}'>{steps}</b>) ← output steps<br><br>
            <span style='color:#555'>Optimizer: Adam &nbsp;|&nbsp; Loss: MSE</span>
          </div>
          <div style='color:#666; font-size:10px; margin-top:10px; white-space:pre-line'>{note}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="tsla-divider"></div>', unsafe_allow_html=True)

    # ── Performance table ─────────────────────────────────────
    st.markdown("#### Performance Metrics")

    if models_ok and meta:
        model_map = {
            "model_a": ("Model A", "1 Day",   CYAN),
            "model_b": ("Model B", "5 Days",  GOLD),
            "model_c": ("Model C", "21 Days", PURPLE),
        }
        rows = []
        for mkey, (name, horizon, color) in model_map.items():
            r = meta["models"][mkey]["results"]
            rows.append({
                "Model": name, "Horizon": horizon,
                "R²": r["R2"], "MAE": r["MAE"],
                "RMSE": r["RMSE"], "MAPE%": r["MAPE"]
            })
        perf_df = pd.DataFrame(rows)
        st.dataframe(
            perf_df.style
                .highlight_max(subset=["R²"], color="#1a3a1a")
                .highlight_min(subset=["RMSE", "MAE", "MAPE%"], color="#1a3a1a")
                .format({"R²": "{:.4f}", "MAE": "{:.4f}",
                         "RMSE": "{:.4f}", "MAPE%": "{:.2f}"}),
            use_container_width=True, hide_index=True
        )

        # ── R2 bar chart ──────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        fig = go.Figure()
        for mkey, (name, horizon, color) in model_map.items():
            r = meta["models"][mkey]["results"]
            fig.add_trace(go.Bar(
                x=[name], y=[r["R2"]],
                name=name, marker_color=color,
                text=[f"R²={r['R2']:.4f}"],
                textposition='outside'
            ))
        fig.update_layout(
            title="R² Score Comparison (higher = better)",
            paper_bgcolor=BG, plot_bgcolor=CARD_BG,
            font_color="#e0e0e0", height=350,
            yaxis=dict(gridcolor="#2a2a4a", range=[0, 1.1]),
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('<div class="tsla-divider"></div>', unsafe_allow_html=True)

    # ── Training config ───────────────────────────────────────
    st.markdown("#### Training Configuration")
    c1, c2, c3, c4 = st.columns(4)
    configs = [
        ("Window Size", "30 days",          CYAN),
        ("Epochs",      "100 (early stop)", GOLD),
        ("Batch Size",  "32",               GREEN),
        ("Val Split",   "10%",              PURPLE),
    ]
    for col, (label, val, color) in zip([c1, c2, c3, c4], configs):
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value" style="color:{color}; font-size:18px">{val}</div>
        </div>""", unsafe_allow_html=True)