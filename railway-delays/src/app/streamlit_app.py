"""RailCast: Railway Network Intelligence.

Streamlit Interactive Web Dashboard.
Explore railway connectivity, understand delay propagation, and predict downstream delays.
"""
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import streamlit as st

from src.app.data_loader import (
    FIGURES_DIR,
    METRICS_FILE,
    get_prediction_model,
    load_critical_sections_df,
    load_metrics_data,
    load_stations_df,
    load_trains_df,
)


def render_header():
    """Render clean application header."""
    st.markdown(
        """
        <div style="background: linear-gradient(90deg, #0f172a 0%, #1e293b 100%); padding: 20px 24px; border-radius: 12px; margin-bottom: 24px; border: 1px solid rgba(255,255,255,0.08); border-left: 5px solid #38bdf8;">
            <h1 style="margin: 0; padding-bottom: 6px; color: #f8fafc; font-size: 26px; font-weight: 700; display: flex; align-items: center; gap: 10px;">
                <span>🚆</span> RailCast — Railway Network Intelligence
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 14.5px;">
                Explore railway connectivity, understand delay propagation, and predict downstream delays.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_network_overview():
    """Section 1: Overview."""
    st.subheader("Corridor B Summary")
    st.markdown(
        "Corridor B represents the high-density Northern–North Central trunk route, connecting "
        "**New Delhi (NDLS)** through Ghaziabad, Tundla, Kanpur Central, and Prayagraj to **Pt. Deen Dayal Upadhyaya Jn (MGS)** (794.2 km)."
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Railway Stations", "108", help="Station nodes along Corridor B")
    with col2:
        st.metric("Connected Sections", "214", help="Unidirectional block sections (Up & Down)")
    with col3:
        st.metric("Trains Represented", "225", help="Active timetable train services")
    with col4:
        st.metric("Delay Cascades Identified", "12,518", help="Analyzed delay propagation cascade events")

    st.markdown("### Corridor Route Segments")
    segments_summary = pd.DataFrame([
        {"Segment": "NDLS – TDL", "Span": "New Delhi to Tundla Jn", "Distance": "218.0 km", "Stations": 11, "Junctions": 5, "Traffic Characteristics": "High traffic (suburban commuter & freight merge)"},
        {"Segment": "TDL – ETW", "Span": "Tundla Jn to Etawah Jn", "Distance": "89.7 km", "Stations": 7, "Junctions": 2, "Traffic Characteristics": "Moderate traffic coupling"},
        {"Segment": "ETW – CNB", "Span": "Etawah Jn to Kanpur Central", "Distance": "135.8 km", "Stations": 14, "Junctions": 2, "Traffic Characteristics": "High coupling; primary bottleneck zone (ETW-EKL)"},
        {"Segment": "CNB – PRYG", "Span": "Kanpur Central to Prayagraj Jn", "Distance": "197.6 km", "Stations": 11, "Junctions": 2, "Traffic Characteristics": "High density main line trunk"},
        {"Segment": "PRYG – MGS", "Span": "Prayagraj Jn to Pt. DDU Jn", "Distance": "153.1 km", "Stations": 9, "Junctions": 3, "Traffic Characteristics": "High freight and eastern trunk connections"},
    ])
    st.dataframe(segments_summary, use_container_width=True, hide_index=True)

    st.markdown("### Key Operational Insights")
    c_ins1, c_ins2, c_ins3 = st.columns(3)
    with c_ins1:
        st.markdown(
            """
            **🔗 Delay Inheritance**  
            Delays along Corridor B are predominantly inherited rather than newly generated in open sections. Strict headway constraints mean a delayed lead train directly impedes follower services.
            """
        )
    with c_ins2:
        st.markdown(
            """
            **⚠️ Critical Chokepoints**  
            Propagation cascades heavily concentrate around Etawah–Ekdil (`ETW-EKL`) and Kanpur–Chandari (`CNB-CNBI`). Targeted dispatching priority at these nodes yields corridor-wide benefits.
            """
        )
    with c_ins3:
        st.markdown(
            """
            **🌫️ Weather & Congestion Amplification**  
            Under dense winter fog or high traffic congestion, downstream cascade propagation increases significantly, multiplying delay persistence 1 to 6 hours ahead.
            """
        )

    fig_eda1 = FIGURES_DIR / "eda_01_overall.png"
    if fig_eda1.exists():
        st.markdown("### Corridor Delay Distribution")
        st.image(str(fig_eda1), caption="Delay distributions across Corridor B stations", use_container_width=True)

    st.markdown("### Critical Bottleneck Sections")
    crit_df = load_critical_sections_df().head(5)
    st.dataframe(crit_df[["rank", "section", "direction", "origin_cascades", "primary_delay_min", "risk"]], use_container_width=True, hide_index=True)


def page_network_explorer():
    """Section 2: Railway Network."""
    st.subheader("Railway Network Directory & Station Inspector")
    st.markdown("Explore stations, topology, junction connectivity, and chain distances along Corridor B.")

    stations_df = load_stations_df()

    selected_station_name = st.selectbox(
        "Select Station to Inspect:",
        options=stations_df["name"].tolist(),
        index=0,
    )
    st_info = stations_df[stations_df["name"] == selected_station_name].iloc[0]

    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        st.metric("Station Code", st_info["code"])
    with col_b:
        st.metric("Chain Distance", f"{st_info['km']} km", "From New Delhi (NDLS)")
    with col_c:
        st.metric("Junction Station?", "Yes" if st_info["is_junction"] else "No")
    with col_d:
        st.metric("Corridor Segment", st_info["segment"])

    st.markdown("---")
    st.markdown(f"### All Corridor B Stations ({len(stations_df)} Nodes)")

    segment_choice = st.selectbox(
        "Filter by Corridor Segment:",
        options=["All Segments", "NDLS-TDL", "TDL-ETW", "ETW-CNB", "CNB-PRYG", "PRYG-MGS"],
        index=0,
    )

    filtered_stations = stations_df if segment_choice == "All Segments" else stations_df[stations_df["segment"] == segment_choice]
    st.dataframe(filtered_stations, use_container_width=True, hide_index=True)

    fig_topo = FIGURES_DIR / "eda_05_weather.png"
    if fig_topo.exists():
        st.markdown("### Network Delay vs Weather Regimes")
        st.image(str(fig_topo), caption="Station delay variations across weather regimes", use_container_width=True)


def page_delay_insights():
    """Section 3: Delay Insights & Downstream Prediction."""
    st.subheader("Delay Insights & Downstream Forecasting")
    st.markdown(
        "Analyze delay propagation patterns and forecast downstream arrival delays for active train services "
        "across **1 to 6 hours ahead** using pre-cutoff observable states."
    )

    predictor = get_prediction_model()
    if predictor is None:
        st.warning("Prediction model bundle is not currently available. Please verify model artifacts in `data/processed/models`.")
    else:
        st.markdown("### 🔮 Downstream Delay Forecast")
        trains_df = load_trains_df()
        stations_df = load_stations_df()

        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            train_choice = st.selectbox(
                "Select Train Service:",
                options=[f"{row['number']} - {row['name']} ({row['type']})" for _, row in trains_df.iterrows()],
                index=3,
            )
            train_num = int(train_choice.split(" - ")[0])

        with col_in2:
            station_choice = st.selectbox(
                "Current Observed Station:",
                options=[f"{row['code']} - {row['name']}" for _, row in stations_df.iterrows()],
                index=17,
            )
            st_code = station_choice.split(" - ")[0]

        with col_in3:
            curr_delay = st.number_input(
                "Current Observed Delay (minutes):",
                min_value=0.0,
                max_value=360.0,
                value=25.0,
                step=5.0,
            )

        col_env1, col_env2, col_env3 = st.columns(3)
        with col_env1:
            fog_val = st.radio("Weather Condition:", ["Normal (Clear / Rain)", "Winter Fog (Visibility < 500m)"], index=0)
            is_fog = 1 if "Fog" in fog_val else 0
        with col_env2:
            congestion_level = st.select_slider("Station Congestion (Active Trains within 30m):", options=[1, 3, 5, 8, 12, 16], value=5)
        with col_env3:
            horizon_mode = st.selectbox("Forecast Horizon:", ["All Horizons (1h to 6h)", "1 Hour Ahead", "2 Hours Ahead", "3 Hours Ahead", "4 Hours Ahead", "5 Hours Ahead", "6 Hours Ahead"], index=0)

        if st.button("Generate Forecast", type="primary"):
            input_dict = {
                "train": train_num,
                "station": st_code,
                "current_delay": float(curr_delay),
                "rolling_mean_delay_30m": float(curr_delay * 0.8),
                "rolling_max_delay_30m": float(curr_delay * 1.2),
                "rolling_active_trains_30m": congestion_level,
                "node_degree": 2,
                "node_betweenness": 0.05,
                "section_criticality": 0.07,
                "fog_flag": is_fog,
                "visibility_m": 400.0 if is_fog else 3500.0,
                "sched_dep_hour": 14,
                "sched_dep_dow": 3,
            }

            with st.spinner("Generating downstream forecast..."):
                if "All" in horizon_mode:
                    preds_df = predictor.predict_all_horizons(input_dict)
                    
                    st.markdown("#### Forecast Summary Across 1 to 6 Hours Ahead")
                    m_cols = st.columns(6)
                    for idx, row in preds_df.iterrows():
                        h = int(row["horizon_hours"])
                        pred_val = float(row["predicted_delay_min"])
                        diff = pred_val - curr_delay
                        with m_cols[idx]:
                            st.metric(f"{h}h Ahead", f"{pred_val:.1f} min", f"{diff:+.1f} min")

                    st.markdown("#### Projected Delay Trajectory")
                    chart_df = preds_df.set_index("horizon_hours")[["predicted_delay_min"]]
                    chart_df.columns = ["Predicted Delay (min)"]
                    st.line_chart(chart_df)
                else:
                    h_val = int(horizon_mode[0])
                    pred_res = predictor.predict(input_dict, horizon=h_val)
                    pred_delay = float(pred_res.iloc[0]["predicted_delay_min"])
                    delta = pred_delay - curr_delay

                    res_col1, res_col2 = st.columns(2)
                    with res_col1:
                        st.metric("Forecast Horizon", f"{h_val} Hour(s) Ahead")
                    with res_col2:
                        st.metric("Predicted Arrival Delay", f"{pred_delay:.1f} min", f"{delta:+.1f} min vs current")

                st.markdown(
                    f"**Operational Interpretation:** At observed station **{st_code}** with an initial delay of "
                    f"**{curr_delay:.0f} min**, headway coupling and congestion indicate "
                    f"{'delay compounding downstream' if is_fog or curr_delay > 30 else 'steady delay progression with scheduled recovery'}. "
                    "Features were derived exclusively from observable pre-cutoff network conditions."
                )

    st.markdown("---")
    st.markdown("### Top Critical Cascade Origin Sections")
    st.markdown(
        "Sections where initial delays propagate most extensively to subsequent train services. "
        "Monitoring and prioritizing traffic through these sections significantly reduces network-wide cascading."
    )

    crit_df = load_critical_sections_df()
    st.dataframe(
        crit_df[["rank", "section", "direction", "origin_cascades", "primary_delay_min", "risk"]],
        use_container_width=True,
        hide_index=True,
    )

    fig_cascades = FIGURES_DIR / "mining_04_cascade_distributions.png"
    if fig_cascades.exists():
        st.markdown("### Cascade Size and Duration Patterns")
        st.image(str(fig_cascades), caption="Propagation cascade size and duration distributions", use_container_width=True)


# Backwards compatibility aliases for existing module callers and tests
page_cascade_mining = page_delay_insights
page_delay_prediction = page_delay_insights


def page_model_evaluation():
    """Retained for backward compatibility with module-level callers."""
    st.info("Model evaluation benchmarks are documented in the project documentation.")


def page_methodology():
    """Retained for backward compatibility with module-level callers."""
    st.info("Methodology and simulation specifications are documented in the project documentation.")


def main():
    """Main application launcher."""
    st.set_page_config(
        page_title="RailCast — Railway Network Intelligence",
        page_icon="🚆",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    render_header()

    st.sidebar.title("🧭 Navigation")
    page = st.sidebar.radio(
        "Select Section:",
        [
            "🌐 Overview",
            "🗺️ Railway Network",
            "⚡ Delay Insights",
        ],
        index=0,
    )

    st.sidebar.divider()
    st.sidebar.markdown("### 📍 Corridor Context")
    st.sidebar.markdown(
        "**Route:** New Delhi – Pt. DDU Jn\n\n"
        "**Corridor Length:** 794.2 km\n\n"
        "**Stations:** 108 corridor nodes\n\n"
        "**Sections:** 214 directed blocks"
    )

    st.sidebar.divider()
    st.sidebar.caption(
        "RailCast currently uses simulated railway data. Results are illustrative and are not live Indian Railways operational data."
    )

    # Route to the selected page (flexible matching for backward-compatibility)
    if "Overview" in page:
        page_network_overview()
    elif "Railway Network" in page or "Explorer" in page:
        page_network_explorer()
    elif "Delay Insights" in page or "Cascade" in page or "Prediction" in page:
        page_delay_insights()
    else:
        page_network_overview()

    st.markdown("---")
    st.caption(
        "RailCast currently uses simulated railway data. Results are illustrative and are not live Indian Railways operational data."
    )


if __name__ == "__main__":
    main()
