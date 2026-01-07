import os
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime

# -----------------------------
# Load or generate data
# -----------------------------
def load_data():
    flood_file = "dire_dawa_flood_log.csv"
    river_obs_file = "river.csv"
    river_pred_file = "river_volume_prediction.csv"

    # Flood log
    if os.path.exists(flood_file):
        flood_df = pd.read_csv(flood_file)
        if "datetime" in flood_df.columns:
            flood_df["datetime"] = pd.to_datetime(flood_df["datetime"], errors="coerce")
    else:
        dates = pd.date_range(end=datetime.today(), periods=30)
        flood_df = pd.DataFrame({
            "datetime": dates,
            "flood_index_mean": np.clip(np.random.normal(0.4, 0.2, 30), 0, 1),
            "current_precip": np.random.randint(0, 50, 30),
            "forecast_precip": np.random.randint(0, 50, 30)
        })

    # River observed
    if os.path.exists(river_obs_file):
        river_obs_df = pd.read_csv(river_obs_file)
    else:
        river_obs_df = pd.DataFrame({
            "polygon_name": ["River A", "River B", "River C"],
            "discharge_estimated": np.random.randint(50, 300, 3)
        })

    # River predicted
    if os.path.exists(river_pred_file):
        river_pred_df = pd.read_csv(river_pred_file)
    else:
        river_pred_df = pd.DataFrame({
            "polygon_name": ["River A", "River B", "River C"],
            "discharge_predicted": np.random.randint(50, 300, 3)
        })

    # Merge rivers
    river_df = pd.merge(river_obs_df, river_pred_df, on="polygon_name", how="outer")
    return flood_df, river_df

flood_df, river_df = load_data()

# Images
image_files = [f for f in os.listdir(".") if f.lower().endswith(".jpg")]

# -----------------------------
# Dash app
# -----------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "Dire Dawa Flood Warning System"

# Latest flood risk
latest_risk = flood_df["flood_index_mean"].iloc[-1] if not flood_df.empty else 0
if latest_risk < 0.3:
    alert_color, status, message = "success", "SAFE – No Flood Risk", "Enjoy your day safely!"
elif latest_risk < 0.6:
    alert_color, status, message = "info", "MONITOR – Low Risk", "Some rain possible, stay alert."
elif latest_risk < 0.8:
    alert_color, status, message = "warning", "WARNING – Medium Risk", "Possible flood soon!"
else:
    alert_color, status, message = "danger", "HIGH RISK – Flood Likely!", "Flood expected! Stay safe!"

# -----------------------------
# Layout
# -----------------------------
app.layout = dbc.Container([
    html.H1("Dire Dawa Flood Warning Dashboard", className="text-center my-4 text-primary"),
    dbc.Alert([
        html.H2(status, className="text-center"),
        html.H4(message, className="text-center mt-3")
    ], color=alert_color, className="mb-5 p-5 rounded"),
    html.Div("No flood data available!" if flood_df.empty else "", className="text-center text-warning mb-4"),

    dbc.Tabs([
        dbc.Tab(label="Overview", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Flood Risk Level"),
                    dcc.Graph(id="flood-gauge")
                ])), width=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Current Rainfall (mm)"),
                    dcc.Graph(id="precip-gauge")
                ])), width=6),
            ], className="mb-4"),
            dcc.Graph(id="overview-map")
        ]),
        dbc.Tab(label="Historical Data", children=[
            dcc.Dropdown(id="metric-dropdown", options=[
                {"label": "Flood Risk Index", "value": "flood_index_mean"},
                {"label": "Current Rainfall (mm)", "value": "current_precip"},
                {"label": "Forecast Rainfall (mm)", "value": "forecast_precip"}
            ], value="flood_index_mean", className="mb-4"),
            dcc.Graph(id="time-series-chart")
        ]),
        dbc.Tab(label="River Predictions", children=[
            dcc.Graph(id="river-bar-chart")
        ]),
        dbc.Tab(label="Detailed Data Table", children=[
            html.H5("Latest Flood Log Records"),
            dash_table.DataTable(
                data=flood_df.tail(20).to_dict("records"),
                columns=[{"name": i, "id": i} for i in flood_df.columns],
                page_size=15
            )
        ]),
        dbc.Tab(label="Flood Area Photos", children=[
            html.H3("Flood Area Photos", className="text-center mb-4"),
            dbc.Row([
                dbc.Col(html.Img(src=img, style={"width":"100%","border-radius":"10px","margin-bottom":"20px"}), width=6, md=4)
                for img in image_files
            ], justify="center") if image_files else html.P("No photos found.")
        ])
    ])
], fluid=True)

# -----------------------------
# Callbacks
# -----------------------------
@app.callback(
    [Output("flood-gauge","figure"), Output("precip-gauge","figure"),
     Output("time-series-chart","figure"), Output("river-bar-chart","figure"),
     Output("overview-map","figure")],
    Input("metric-dropdown","value")
)
def update_graphs(selected_metric):
    latest = flood_df.iloc[-1].to_dict() if not flood_df.empty else {}
    flood_val = latest.get("flood_index_mean", 0)
    precip_val = latest.get("current_precip", 0)

    flood_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=flood_val,
        title={"text": "Flood Risk (0-1)"},
        gauge={"axis":{"range":[0,1]},
               "steps":[
                   {"range":[0,0.3],"color":"green"},
                   {"range":[0.3,0.6],"color":"yellow"},
                   {"range":[0.6,0.8],"color":"orange"},
                   {"range":[0.8,1],"color":"red"}
               ]}
    ))

    precip_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=precip_val,
        title={"text": "Current Rain (mm)"},
        gauge={"axis":{"range":[None,None]}}
    ))

    time_chart = px.line(flood_df, x="datetime", y=selected_metric,
                         title=selected_metric.replace("_"," ").title()) if not flood_df.empty else go.Figure()

    river_chart = px.bar(river_df, x="polygon_name", y=["discharge_estimated","discharge_predicted"],
                         barmode="group", title="River Discharge (Estimated vs Predicted)") if not river_df.empty else go.Figure()

    map_fig = px.scatter_mapbox(center={"lat":9.6009,"lon":41.8661}, zoom=11, height=500,
                                mapbox_style="open-street-map", title="Dire Dawa Area")

    return flood_gauge, precip_gauge, time_chart, river_chart, map_fig

# -----------------------------
# Run server (Render-ready)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run_server(host="0.0.0.0", port=port, debug=False)
