import os
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load data safely
def load_data():
    flood_log = 'dire_dawa_flood_log.csv'
    river_pred = 'river_volume_prediction.csv'
    
    flood_df = pd.read_csv(flood_log) if os.path.exists(flood_log) else pd.DataFrame()
    river_df = pd.read_csv(river_pred) if os.path.exists(river_pred) else pd.DataFrame()
    
    if not flood_df.empty:
        flood_df['datetime'] = pd.to_datetime(flood_df['datetime'], errors='coerce')
    
    return flood_df, river_df

flood_df, river_df = load_data()

# Find all JPG images automatically
image_files = [f for f in os.listdir('.') if f.lower().endswith('.jpg')]

# Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])
server = app.server
app.title = "Dire Dawa Flood Warning System"

# Latest risk (safe if no data)
latest_risk = flood_df['flood_index_mean'].iloc[-1] if not flood_df.empty and 'flood_index_mean' in flood_df.columns else 0

# Alert banner
if latest_risk < 0.3:
    alert_color = "success"
    status = "SAFE – No Flood Risk"
    message = "It is fine today. Enjoy your day safely!"
elif latest_risk < 0.6:
    alert_color = "info"
    status = "MONITOR – Low Risk"
    message = "Some rain possible, but currently safe."
elif latest_risk < 0.8:
    alert_color = "warning"
    status = "WARNING – Medium Risk"
    message = "Possible flood soon. Stay alert!"
else:
    alert_color = "danger"
    status = "HIGH RISK – Flood Likely!"
    message = "Flood expected! Stay safe and avoid risk areas!"

# Layout
app.layout = dbc.Container([
    html.H1("Dire Dawa Flood Warning Dashboard", className="text-center my-4 text-primary"),
    
    dbc.Alert([
        html.H2(status, className="text-center"),
        html.H4(message, className="text-center mt-3")
    ], color=alert_color, className="mb-5 p-5 rounded"),
    
    # Message if no data
    html.Div("No data available yet — run your scripts to generate CSVs!" if flood_df.empty else "", className="text-center text-warning mb-4"),
    
    dbc.Tabs([
        dbc.Tab(label="Overview", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Flood Risk Level"), dcc.Graph(id="flood-gauge")
                ])), width=6),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Current Rainfall (mm)"), dcc.Graph(id="precip-gauge")
                ])), width=6),
            ], className="mb-4"),
            dcc.Graph(id="overview-map")
        ]),
        
        dbc.Tab(label="Historical Data", children=[
            dcc.Dropdown(id="metric-dropdown", options=[
                {'label': 'Flood Risk Index', 'value': 'flood_index_mean'},
                {'label': 'Current Rainfall (mm)', 'value': 'current_precip'},
                {'label': 'Forecast Rainfall (mm)', 'value': 'forecast_precip'},
            ], value='flood_index_mean', className="mb-4"),
            dcc.Graph(id="time-series-chart"),
        ]),
        
        dbc.Tab(label="River Predictions", children=[
            dcc.Graph(id="river-bar-chart")
        ]),
        
        dbc.Tab(label="Detailed Data Table", children=[
            html.H5("Latest Flood Log Records"),
            dash_table.DataTable(
                data=flood_df.tail(20).to_dict('records') if not flood_df.empty else [],
                columns=[{"name": i, "id": i} for i in flood_df.columns] if not flood_df.empty else [],
                page_size=15
            )
        ]),
        
        dbc.Tab(label="Flood Area Photos", children=[
            html.H3("Real Photos from Dire Dawa Flood Zones", className="text-center mb-4"),
            dbc.Row([
                dbc.Col(html.Img(src=img, style={'width': '100%', 'border-radius': '10px', 'margin-bottom': '20px'}), width=6, md=4)
                for img in image_files
            ], justify="center") if image_files else html.P("No photos found.")
        ]),
    ])
], fluid=True)

# Graphs callback
@app.callback(
    [Output("flood-gauge", "figure"), Output("precip-gauge", "figure"),
     Output("time-series-chart", "figure"), Output("river-bar-chart", "figure"),
     Output("overview-map", "figure")],
    Input("metric-dropdown", "value")
)
def update_graphs(selected_metric):
    # Safe defaults
    latest = flood_df.iloc[-1].to_dict() if not flood_df.empty else {}
    flood_val = latest.get('flood_index_mean', 0)
    precip_val = latest.get('current_precip', 0)
    
    # Gauges
    flood_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=flood_val, title={'text': "Risk (0-1)"},
        gauge={'axis': {'range': [0, 1]}, 'steps': [
            {'range': [0, 0.3], 'color': "green"},
            {'range': [0.3, 0.6], 'color': "yellow"},
            {'range': [0.6, 0.8], 'color': "orange"},
            {'range': [0.8, 1], 'color': "red"}
        ]}
    ))
    
    precip_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=precip_val, title={'text': "Current Rain (mm)"},
        gauge={'axis': {'range': [None, None]}}  # Auto scale
    ))
    
    # Charts
    time_chart = go.Figure()
    if not flood_df.empty and selected_metric in flood_df.columns:
        time_chart = px.line(flood_df, x='datetime', y=selected_metric, title=selected_metric.replace('_', ' ').title())
    
    river_chart = go.Figure()
    if not river_df.empty:
        river_chart = px.bar(river_df, x='polygon_name', y=['discharge_estimated', 'discharge_predicted'],
                             barmode='group', title="River Discharge (Estimated vs Predicted)")
    
    # Map
    map_fig = px.scatter_mapbox(center={"lat": 9.6009, "lon": 41.8661}, zoom=11, height=500,
                                mapbox_style="open-street-map", title="Dire Dawa Area")
    
    return flood_gauge, precip_gauge, time_chart, river_chart, map_fig

if __name__ == "__main__":
    app.run_server(debug=True)
