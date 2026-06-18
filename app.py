from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "socal2.csv"
IMAGE_DIR = BASE_DIR / "socal2" / "socal_pics"
FEATURES = ["city_encoded", "bedrooms", "bathrooms", "sqft", "Total_rooms", "sqft_per_room"]


st.set_page_config(
    page_title="SoCal House Price Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_hero_ui_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --hero-bg: #050712;
            --hero-surface: #0f172a;
            --hero-surface-2: #111827;
            --hero-foreground: #f8fafc;
            --hero-muted: #a7b0c0;
            --hero-border: rgba(148, 163, 184, 0.22);
            --hero-primary: #006fee;
            --hero-secondary: #7828c8;
            --hero-success: #17c964;
            --hero-warning: #f5a524;
            --hero-danger: #f31260;
            --hero-radius: 14px;
            --hero-shadow: 0 18px 50px rgba(0, 0, 0, 0.38);
        }

        .stApp {
            background:
                radial-gradient(circle at 12% 10%, rgba(0, 111, 238, 0.20), transparent 28%),
                radial-gradient(circle at 88% 0%, rgba(120, 40, 200, 0.22), transparent 30%),
                var(--hero-bg);
            color: var(--hero-foreground);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1280px;
        }

        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.86);
            border-right: 1px solid var(--hero-border);
            backdrop-filter: blur(18px);
        }

        .hero-shell {
            background: linear-gradient(135deg, rgba(0, 111, 238, 0.95), rgba(120, 40, 200, 0.95));
            color: white;
            border-radius: 22px;
            padding: 34px 36px;
            box-shadow: var(--hero-shadow);
            margin-bottom: 22px;
        }

        .hero-title {
            font-size: 42px;
            line-height: 1.08;
            font-weight: 780;
            letter-spacing: 0;
            margin: 0 0 10px 0;
        }

        .hero-subtitle {
            font-size: 17px;
            line-height: 1.65;
            max-width: 780px;
            opacity: 0.93;
            margin: 0;
        }

        .hero-chip-row {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 18px;
        }

        .hero-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            padding: 7px 12px;
            font-size: 13px;
            font-weight: 650;
            background: rgba(255, 255, 255, 0.18);
            border: 1px solid rgba(255, 255, 255, 0.24);
        }

        .hero-card {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid var(--hero-border);
            border-radius: var(--hero-radius);
            box-shadow: var(--hero-shadow);
            padding: 18px 20px;
            height: 100%;
        }

        .hero-card h3 {
            font-size: 15px;
            line-height: 1.3;
            color: var(--hero-muted);
            margin: 0 0 8px 0;
            font-weight: 700;
        }

        .hero-metric {
            font-size: 28px;
            line-height: 1.15;
            font-weight: 780;
            color: var(--hero-foreground);
            margin: 0;
        }

        .hero-note {
            color: var(--hero-muted);
            font-size: 14px;
            line-height: 1.6;
        }

        .hero-section-title {
            font-size: 23px;
            font-weight: 760;
            letter-spacing: 0;
            margin: 12px 0 8px 0;
        }

        .hero-callout {
            border-left: 4px solid var(--hero-primary);
            background: rgba(0, 111, 238, 0.14);
            border-radius: 12px;
            padding: 14px 16px;
            color: var(--hero-foreground);
            line-height: 1.6;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 999px;
            padding: 8px 16px;
            background: rgba(15, 23, 42, 0.82);
            border: 1px solid var(--hero-border);
            color: var(--hero-muted);
        }

        .stTabs [aria-selected="true"] {
            color: white;
            background: var(--hero-primary);
            border-color: var(--hero-primary);
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid var(--hero-border);
            border-radius: var(--hero-radius);
            padding: 14px 16px;
            box-shadow: var(--hero-shadow);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--hero-border);
            border-radius: var(--hero-radius);
            overflow: hidden;
        }

        label, .stMarkdown, [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
            color: var(--hero-foreground);
        }

        .stButton > button {
            border-radius: 12px;
            border: 1px solid var(--hero-primary);
            background: var(--hero-primary);
            color: white;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    df = df.rename(
        columns={
            "citi": "city",
            "n_citi": "city_encoded",
            "bed": "bedrooms",
            "bath": "bathrooms",
        }
    )
    df["bathrooms"] = df["bathrooms"].astype("int64")
    df["Total_rooms"] = df["bathrooms"] + df["bedrooms"]
    df["Price per Square Foot"] = df["price"] / df["sqft"]
    df["sqft_per_room"] = df["sqft"] / df["Total_rooms"]
    df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
    df["image_path"] = df["image_id"].apply(lambda value: IMAGE_DIR / f"{int(value)}.jpg")
    return df


@st.cache_resource
def train_model(df: pd.DataFrame):
    x = df[FEATURES]
    y = df["price"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=20,
                    min_samples_leaf=1,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    metrics = {
        "r2": r2_score(y_test, pred),
        "mae": mean_absolute_error(y_test, pred),
        "test_rows": len(y_test),
    }
    return model, metrics


def money(value: float) -> str:
    return f"${value:,.0f}"


def render_metric_card(title: str, value: str, note: str) -> None:
    st.markdown(
        f"""
        <div class="hero-card">
            <h3>{title}</h3>
            <p class="hero-metric">{value}</p>
            <p class="hero-note">{note}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero-shell">
            <p class="hero-title">SoCal House Price Intelligence</p>
            <p class="hero-subtitle">
                A Kaggle-ready Streamlit deployment for exploring Southern California housing data,
                evaluating tabular price signals, browsing property images, and predicting prices
                with a cached Random Forest baseline.
            </p>
            <div class="hero-chip-row">
                <span class="hero-chip">HeroUI-inspired interface</span>
                <span class="hero-chip">EDA dashboard</span>
                <span class="hero-chip">Tabular prediction</span>
                <span class="hero-chip">Image + tabular design notes</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def overview_tab(df: pd.DataFrame) -> None:
    st.markdown('<p class="hero-section-title">Dataset Overview</p>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Houses", f"{len(df):,}", "Clean rows used in the app")
    with c2:
        render_metric_card("Cities", f"{df['city'].nunique():,}", "Unique city labels")
    with c3:
        render_metric_card("Median Price", money(df["price"].median()), "Central house price")
    with c4:
        render_metric_card("Median Sqft", f"{df['sqft'].median():,.0f}", "Typical property size")

    st.markdown('<p class="hero-section-title">Top Cities by Listing Count</p>', unsafe_allow_html=True)
    city_counts = df["city"].value_counts().head(15).reset_index()
    city_counts.columns = ["city", "houses"]
    fig = px.bar(
        city_counts,
        x="houses",
        y="city",
        orientation="h",
        color="houses",
        color_continuous_scale=["#e4f1ff", "#006fee"],
        labels={"houses": "Number of houses", "city": "City"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)


def eda_tab(df: pd.DataFrame) -> None:
    st.markdown('<p class="hero-section-title">EDA Answers</p>', unsafe_allow_html=True)
    top_total = df.groupby("city")["price"].sum().sort_values(ascending=False).head(5)
    top_average = df.groupby("city")["price"].mean().sort_values(ascending=False).head(10)
    top_streets = df.groupby("street")["image_id"].count().sort_values(ascending=False).head(10)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Top 5 cities by total house prices")
        st.dataframe(top_total.map(money).reset_index(name="total_price"), use_container_width=True, hide_index=True)
    with col2:
        st.markdown("#### Average house price in each city")
        st.dataframe(top_average.map(money).reset_index(name="avg_price"), use_container_width=True, hide_index=True)

    st.markdown("#### Streets with the largest number of houses")
    st.dataframe(top_streets.reset_index(name="house_count"), use_container_width=True, hide_index=True)

    st.markdown("#### Size and room relationships")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(
            df.sample(min(3500, len(df)), random_state=42),
            x="sqft",
            y="price",
            color="bedrooms",
            opacity=0.55,
            color_continuous_scale=["#006fee", "#17c964", "#f5a524"],
            labels={"sqft": "Square feet", "price": "Price"},
        )
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        room_stats = (
            df.groupby("bedrooms", as_index=False)["price"]
            .mean()
            .rename(columns={"price": "avg_price"})
        )
        fig = px.line(room_stats, x="bedrooms", y="avg_price", markers=True, labels={"avg_price": "Average price"})
        fig.update_traces(line_color="#7828c8", marker=dict(size=8))
        fig.update_layout(height=450, margin=dict(l=10, r=10, t=20, b=10))
        st.plotly_chart(fig, use_container_width=True)

    bath_stats = df.groupby("bathrooms", as_index=False)["price"].mean().rename(columns={"price": "avg_price"})
    fig = px.bar(
        bath_stats,
        x="bathrooms",
        y="avg_price",
        color="avg_price",
        color_continuous_scale=["#e8faf0", "#17c964"],
        labels={"avg_price": "Average price"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=20, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        """
        <div class="hero-callout">
            <b>Summary:</b> Sqft has the clearest relationship with price, bathrooms are also useful,
            and bedrooms are helpful but weaker by themselves. City and street remain important because
            houses with similar size can have very different prices depending on location.
        </div>
        """,
        unsafe_allow_html=True,
    )


def predictor_tab(df: pd.DataFrame, model, metrics: dict) -> None:
    st.markdown('<p class="hero-section-title">Tabular Price Predictor</p>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Random Forest R2", f"{metrics['r2']:.3f}")
    c2.metric("Random Forest MAE", money(metrics["mae"]))
    c3.metric("Test Rows", f"{metrics['test_rows']:,}")

    st.markdown(
        """
        <div class="hero-callout">
            The deployed predictor uses the tabular Random Forest baseline because it is lightweight,
            fast to run on Streamlit, and does not require shipping a large TensorFlow model file.
        </div>
        """,
        unsafe_allow_html=True,
    )

    cities = df.sort_values("city")["city"].unique().tolist()
    city = st.selectbox("City", cities, index=cities.index("San Diego, CA") if "San Diego, CA" in cities else 0)
    city_encoded = int(df.loc[df["city"] == city, "city_encoded"].mode().iloc[0])

    col1, col2, col3 = st.columns(3)
    with col1:
        bedrooms = st.number_input("Bedrooms", min_value=1, max_value=12, value=3, step=1)
    with col2:
        bathrooms = st.number_input("Bathrooms", min_value=1, max_value=12, value=2, step=1)
    with col3:
        sqft = st.number_input("Square feet", min_value=250, max_value=12000, value=1800, step=50)

    total_rooms = bedrooms + bathrooms
    sqft_per_room = sqft / total_rooms
    row = pd.DataFrame(
        [
            {
                "city_encoded": city_encoded,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "sqft": sqft,
                "Total_rooms": total_rooms,
                "sqft_per_room": sqft_per_room,
            }
        ]
    )
    prediction = float(model.predict(row)[0])
    st.markdown(
        f"""
        <div class="hero-card">
            <h3>Estimated House Price</h3>
            <p class="hero-metric">{money(prediction)}</p>
            <p class="hero-note">Prediction based on city, bedrooms, bathrooms, sqft, total rooms, and sqft per room.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def images_tab(df: pd.DataFrame) -> None:
    st.markdown('<p class="hero-section-title">Property Image Gallery</p>', unsafe_allow_html=True)
    selected_city = st.selectbox("Filter by city", ["All cities"] + sorted(df["city"].unique().tolist()))
    sample_df = df if selected_city == "All cities" else df[df["city"] == selected_city]
    sample_df = sample_df[sample_df["image_path"].apply(lambda p: p.exists())]
    count = st.slider("Images to show", min_value=4, max_value=24, value=8, step=4)
    sample_df = sample_df.sample(min(count, len(sample_df)), random_state=42)

    cols = st.columns(4)
    for index, (_, row) in enumerate(sample_df.iterrows()):
        with cols[index % 4]:
            image = Image.open(row["image_path"])
            st.image(image, use_container_width=True)
            st.caption(f"{row['city']} · {money(row['price'])} · {int(row['sqft']):,} sqft")

    st.markdown(
        """
        <div class="hero-callout">
            Images can add signals like exterior quality, property condition, curb appeal, and architectural style.
            The strongest deployment approach is a two-branch model: EfficientNet for images plus dense layers
            for tabular values, joined before the final regression output.
        </div>
        """,
        unsafe_allow_html=True,
    )


def model_design_tab() -> None:
    st.markdown('<p class="hero-section-title">Image + Tabular Model Design</p>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero-card">
            <h3>Can images provide useful information?</h3>
            <p class="hero-note">
                Yes. Images may capture visual quality that is missing from the CSV, including building condition,
                renovation level, architecture, landscaping, view, lighting, and curb appeal.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            #### How to combine both data types
            1. Resize and preprocess the image.
            2. Pass the image through EfficientNet.
            3. Scale tabular features like sqft and rooms.
            4. Pass tabular values through dense layers.
            5. Concatenate both feature vectors.
            6. Predict scaled log-price, then convert back to dollars.
            """
        )
    with col2:
        st.markdown(
            """
            #### Expected challenges
            - Images are expensive to train on.
            - Location can dominate visual signals.
            - Prices have strong outliers.
            - Some street/address values are missing or noisy.
            - City distribution is uneven.
            - A large TensorFlow model needs a saved `.keras` artifact for deployment.
            """
        )

    st.markdown(
        """
        <div class="hero-callout">
            For this Streamlit app, the Random Forest baseline is deployed for speed and reliability.
            The notebook still documents the deeper EfficientNet + tabular approach for experiments.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_hero_ui_css()
    df = load_data()
    model, metrics = train_model(df)

    with st.sidebar:
        st.title("SoCal")
        st.caption("House price dashboard")
        st.divider()
        st.metric("Rows", f"{len(df):,}")
        st.metric("Cities", f"{df['city'].nunique():,}")
        st.metric("Median price", money(df["price"].median()))
        st.divider()
        st.caption("Built from socal2.csv and local property images.")

    render_hero()

    tabs = st.tabs(["Overview", "EDA", "Predictor", "Images", "Model Design"])
    with tabs[0]:
        overview_tab(df)
    with tabs[1]:
        eda_tab(df)
    with tabs[2]:
        predictor_tab(df, model, metrics)
    with tabs[3]:
        images_tab(df)
    with tabs[4]:
        model_design_tab()


if __name__ == "__main__":
    main()
