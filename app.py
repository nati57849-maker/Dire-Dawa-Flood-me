import os
import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Load your data files
def load_data():
    flood_log = 'dire_dawa_flood_log.csv'
    river_pred = 'river_volume_prediction.csv'
    
    flood_df = pd.read_csv(flood_log) if os.path.exists(flood_log) else pd.DataFrame()
    river_df = pd.read_csv(river_pred) if os.path.exists(river_pred) else pd.DataFrame()
    
    if not flood_df.empty:
        flood_df['datetime'] = pd.to_datetime(flood_df['datetime'])
    
    return flood_df, river_df

flood_df, river_df = load_data()

# Automatically find all your JPG images
image_files = [f for f in os.listdir('.') if f.lower().endswith('.jpg')]

# Dash app setup
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.CYBORG])  # Dark theme
server = app.server
app.title = "Dire Dawa Flood Warning System"

# Current risk level for the big alert banner
latest_risk = flood_df['flood_index_mean'].iloc[-1] if not flood_df.empty and 'flood_index_mean' in flood_df.columns else 0

# Simple alert messages
if latest_risk < 0.3:
    alert_color = "success"
    status = "SAFE – No Flood Risk"
    message = "It is fine today. Enjoy your day safely!"
elif latest_risk < 0.6:
    alert_color = "info"
    status = "MONITOR – Low Risk"
    message = "Some rain possible, but safe for now. Keep an eye on updates."
elif latest_risk < 0.8:
    alert_color = "warning"
    status = "WARNING – Medium Risk"
    message = "Possible flood soon. Stay alert and avoid river areas."
else:
    alert_color = "danger"
    status = "HIGH RISK – Flood Likely!"
    message = "Flood expected! Stay away from rivers and low areas. Stay safe!"

# Main layout
app.layout = dbc.Container([
    html.H1("Dire Dawa Flood Warning Dashboard", className="text-center my-4 text-primary"),
    
    # Big alert banner everyone sees first
    dbc.Alert([
        html.H2(status, className="text-center"),
        html.H4(message, className="text-center mt-3")
    ], color=alert_color, className="mb-5 p-5 rounded", is_open=True),
    
    dbc.Tabs([
        # Tab 1: Simple overview
        dbc.Tab(label="Overview", children=[
            dbc.Row([
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Current Flood Risk Level"),
                    dcc.Graph(id="flood-gauge")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Rainfall (mm)"),
                    dcc.Graph(id="precip-gauge")
                ])), width=4),
                dbc.Col(dbc.Card(dbc.CardBody([
                    html.H5("Soil Moisture & River Level"),
                    dcc.Graph(id="sm-river-gauge")
                ])), width=4),
            ], className="mb-4"),
            dcc.Graph(id="overview-map")
        ]),
        
        # Tab 2: Charts for history
        dbc.Tab(label="Historical Data", children=[
            html.P("Select what to view:"),
            dcc.Dropdown(id="metric-dropdown", options=[
                {'label': 'Flood Risk Index', 'value': 'flood_index_mean'},
                {'label': 'Rainfall (mm)', 'value': 'current_precip'},
                {'label': 'Soil Moisture', 'value': 'soil_moisture_mean'},
                {'label': 'River Level', 'value': 'river_level_mean'},
            ], value='flood_index_mean', className="mb-4"),
            dcc.Graph(id="time-series-chart"),
        ]),
        
        # Tab 3: River predictions
        dbc.Tab(label="River Discharge Predictions", children=[
            dcc.Graph(id="river-bar-chart")
        ]),
        
        # Tab 4: Raw data for experts
        dbc.Tab(label="Detailed Data Table (Experts)", children=[
            html.H5("Latest 20 Records from Flood Log"),
            dash_table.DataTable(
                id="data-table",
                columns=[{"name": i, "id": i} for i in flood_df.columns] if not flood_df.empty else [],
                data=flood_df.tail(20).to_dict('records') if not flood_df.empty else [],
                page_size=15,
                style_cell={'textAlign': 'left', 'padding': '10px'},
                style_header={'backgroundColor': 'rgb(30, 30, 30)', 'color': 'white'},
                style_data={'backgroundColor': 'rgb(50, 50, 50)', 'color': 'white'},
            )
        ]),
        
        # Tab 5: All your photos
        dbc.Tab(label="Flood Area Photos", children=[
            html.H3("Real Photos from Dire Dawa Flood Zones", className="text-center mb-4"),
            dbc.Row([
                dbc.Col(
                    html.Div([
                        html.Img(src=img, style={'width': '100%', 'border-radius': '10px'}),
                        html.P(img, className="text-center mt-2")
                    ]),
                    width=6, lg=4, className="mb-4"
                )
                for img in image_files
            ], justify="center")
        ]),
    ])
], fluid=True)

# Update graphs
@app.callback(
    [Output("flood-gauge", "figure"),
     Output("precip-gauge", "figure"),
     Output("sm-river-gauge", "figure"),
     Output("time-series-chart", "figure"),
     Output("river-bar-chart", "figure"),
     Output("overview-map", "figure")],
    Input("metric-dropdown", "value")
)
def update_graphs(selected_metric):
    latest = flood_df.iloc[-1] if not flood_df.empty else {}
    flood_val = latest.get('flood_index_mean', 0)
    precip_val = latest.get('current_precip', 0)
    sm_val = latest.get('soil_moisture_mean', 0)
    river_val = latest.get('river_level_mean', 0)
    
    # Gauges
    flood_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=flood_val,
        title={'text': "Flood Risk (0-1)"},
        gauge={'axis': {'range': [0, 1]},
               'steps': [{'range': [0, 0.3], 'color': "green"},
                         {'range': [0.3, 0.6], 'color': "yellow"},
                         {'range': [0.6, 0.8], 'color': "orange"},
                         {'range': [0.8, 1], 'color': "red"}]}
    ))
    
    precip_gauge = go.Figure(go.Indicator(mode="gauge+number", value=precip_val, title={'text': "Rainfall (mm)"}))
    sm_river_gauge = go.Figure(go.Indicator(mode="gauge+number", value=(sm_val + river_val)/2 if (sm_val + river_val) > 0 else 0, title={'text': "Average Level"}))
    
    # Charts
    time_chart = px.line(flood_df, x='datetime', y=selected_metric, title=f"{selected_metric.replace('_', ' ').title()} Over Time") if not flood_df.empty and selected_metric in flood_df.columns else go.Figure()
    
    river_chart = px.bar(river_df, x='polygon_name', y=['discharge_estimated', 'discharge_predicted'], barmode='group', title="River Discharge (Estimated vs Predicted)") if not river_df.empty else go.Figure()
    
    # Map
    map_fig = px.scatter_mapbox(
        lat=[9.6009], lon=[41.8661], zoom=11, height=600,
        title="Dire Dawa Location",
        mapbox_style="open-street-map"
    )
    map_fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0})
    
    return flood_gauge, precip_gauge, sm_river_gauge, time_chart, river_chart, map_fig

if __name__ == "__main__":
    app.run_server(debug=True)
