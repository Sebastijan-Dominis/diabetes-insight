import dash_bootstrap_components as dbc
import os
import requests
from dotenv import load_dotenv
from dash_bootstrap_templates import load_figure_template
from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL")

app = Dash(__name__, external_stylesheets=[dbc.themes.QUARTZ])

load_figure_template('QUARTZ')

app.layout = dbc.Container([
    html.H1("Diabetes Insight", style={"textAlign": "center", "marginTop": 20}),
    html.Br(),
    dbc.Row([
        dbc.Col([
            html.H2("See if you have diabetes"),
            html.P("Provide the following information to assess your diabetes status:"),
            html.H3("Hemoglobin A1c (HbA1c)"),
            dbc.Card(dcc.Input(type="number", id="hba1c", placeholder="Enter your HbA1c level")),
            html.Br(),
            html.H3("Fasting Glucose"),
            dbc.Card(dcc.Input(type="number", id="fasting_glucose", placeholder="Enter your fasting glucose level")),
            html.Br(),
            dbc.Card(dbc.Button("Submit", id="submit-button-1", color="primary")),
            html.Br(),
            html.P(id="output-diabetes-assessment-text"),
            dcc.Download(id="output-diabetes-assessment"),
        ], width=5),
        dbc.Col([
            html.H2("See if you are at risk of having diabetes"),
            html.P("Provide the following information to assess your diabetes risk score:"),
            html.H3("Family History"),
            dbc.Card(dcc.Dropdown(
                options=[
                    {"label": "Yes", "value": "Yes"},
                    {"label": "No", "value": "No"},
                ],
                id="family_history",
                placeholder="Do you have a family history of diabetes?"
            )),
            html.Br(),
            html.H3("Age"),
            dbc.Card(dcc.Input(type="number", id="age", placeholder="Enter your age")),
            html.Br(),
            html.H3("BMI"),
            dbc.Card(dcc.Input(type="number", id="bmi", placeholder="Enter your BMI")),
            html.Br(),
            html.H3("Waist-to-Hip Ratio"),
            dbc.Card(dcc.Input(type="number", id="waist_to_hip_ratio", placeholder="Enter your waist-to-hip ratio")),
            html.Br(),
            html.H3("Physical Activity (minutes per week)"),
            dbc.Card(dcc.Input(type="number", id="physical_activity", placeholder="Enter the number of minutes of physical activity you get per week")),
            html.Br(),
            html.H3("Number of alcoholic drinks per week"),
            dbc.Card(dcc.Input(type="number", id="alcoholic_drinks", placeholder="Enter the number of alcoholic drinks you have per week")),
            html.Br(),
            dbc.Card(dbc.Button("Submit", id="submit-button-2", color="primary")),
            html.Br(),
            html.P(id="output-risk-assessment-text"),
            dcc.Download(id="output-risk-assessment"),
        ], width=5),
    ], justify="between", style={"marginTop": 30}),
])

# Callback for diabetes diagnosis assessment
@app.callback(
    Output("output-diabetes-assessment", "data"),
    Output("output-diabetes-assessment-text", "children"),
    Input("submit-button-1", "n_clicks"),
    State("hba1c", "value"),
    State("fasting_glucose", "value")
)
def assess_diabetes_diagnosis(n_clicks, hba1c, fasting_glucose):
    if not n_clicks:
        raise PreventUpdate
    
    if any(v is None for v in [hba1c, fasting_glucose]):
        return None, "Please fill in all fields to assess your diabetes status."

    input_data = {
        "hba1c": float(hba1c),
        "glucose_fasting": float(fasting_glucose),
    }

    try:
        response = requests.post(f"{BACKEND_URL}/diagnosis", json=input_data)
        if response.status_code == 200:
            return dcc.send_bytes(response.content, "diabetes_diagnosis_result.pdf"), "Diabetes diagnosis assessment completed. Your report is being downloaded."
        elif response.status_code == 429:
            result = response.json()
            return None, f"{result['message']}"
        else:
            return None, "Error assessing diabetes diagnosis."
    except Exception as e:
        return None, "Error assessing diabetes diagnosis."
    

# Callback for diabetes risk assessment
@app.callback(
    Output("output-risk-assessment", "data"),
    Output("output-risk-assessment-text", "children"),
    Input("submit-button-2", "n_clicks"),
    State("family_history", "value"),
    State("age", "value"),
    State("bmi", "value"),
    State("waist_to_hip_ratio", "value"),
    State("physical_activity", "value"),
    State("alcoholic_drinks", "value"),
)
def assess_diabetes_risk(n_clicks, family_history, age, bmi, waist_to_hip_ratio, physical_activity, alcoholic_drinks):
    if not n_clicks:
        raise PreventUpdate
    
    if any(v is None for v in [family_history, age, bmi, waist_to_hip_ratio, physical_activity, alcoholic_drinks]):
        return None, "Please fill in all fields to assess your diabetes risk score."
    family_history_flag = 1 if family_history == "Yes" else 0

    input_data = {
        "family_history": int(family_history_flag),
        "age": int(round(age)),
        "bmi": float(bmi),
        "waist_to_hip_ratio": float(waist_to_hip_ratio),
        "physical_activity": int(round(physical_activity)),
        "alcohol_consumption_per_week": int(round(alcoholic_drinks)),
    }

    try:
        response = requests.post(f"{BACKEND_URL}/risk_score", json=input_data)
        if response.status_code == 200:
            return dcc.send_bytes(response.content, "diabetes_risk_score_result.pdf"), "Diabetes risk score assessment completed. Your report is being downloaded."
        elif response.status_code == 429:
            result = response.json()
            return None, f"{result['message']}"
        else:
            return None, "Error assessing diabetes risk score."
    except Exception as e:
        return None, "Error assessing diabetes risk score."
    
# Disable buttons after click to avoid multiple submissions
app.clientside_callback(
    """
    function(n_clicks, disabled) {
        if (!n_clicks) {
            return false;
        }
        // Disable button immediately on click
        return true;
    }
    """,
    Output("submit-button-1", "disabled"),
    Input("submit-button-1", "n_clicks"),
    State("submit-button-1", "disabled")
)

app.clientside_callback(
    """
    function(n_clicks, disabled) {
        if (!n_clicks) {
            return false;
        }
        return true;
    }
    """,
    Output("submit-button-2", "disabled"),
    Input("submit-button-2", "n_clicks"),
    State("submit-button-2", "disabled")
)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8050)
