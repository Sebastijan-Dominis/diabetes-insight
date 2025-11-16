import joblib
import shap
import pandas as pd
from dotenv import load_dotenv
from fastapi import APIRouter
from starlette import status

load_dotenv()

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
)

thresh1 = 0.3
thresh2 = 0.77

chosen_columns_1 = ['glucose_fasting', 'hba1c']

chosen_columns_2 = ["family_history_diabetes", "age_30-39", "age_40-49", "age_50-59", "age_60-69", "age_70-79", "age_80+", "physical_activity_minutes_per_week_Moderate", "physical_activity_minutes_per_week_Active", "physical_activity_minutes_per_week_Very Active", "bmi_Normal", "bmi_Overweight", "bmi_Obese_I", "bmi_Obese_II", "waist_to_hip_ratio", "alcohol_consumption_per_week_Light", "alcohol_consumption_per_week_Moderate", "alcohol_consumption_per_week_Heavy"]

def prepare_for_diagnosis(family_history, bmi, age, waist_to_hip_ratio, physical_activity, alcohol_consumption_per_week):
    age_30_39 = 1 if 30 <= age <= 39 else 0
    age_40_49 = 1 if 40 <= age <= 49 else 0
    age_50_59 = 1 if 50 <= age <= 59 else 0
    age_60_69 = 1 if 60 <= age <= 69 else 0
    age_70_79 = 1 if 70 <= age <= 79 else 0
    age_80_plus = 1 if age >= 80 else 0

    physical_activity_Moderate = 1 if 100 <= physical_activity < 150 else 0
    physical_activity_Active = 1 if 150 <= physical_activity < 300 else 0
    physical_activity_Very_Active = 1 if physical_activity >= 300 else 0

    bmi_Normal = 1 if 18.5 <= bmi < 25 else 0
    bmi_Overweight = 1 if 25 <= bmi < 30 else 0
    bmi_Obese_I = 1 if 30 <= bmi < 35 else 0
    bmi_Obese_II = 1 if 35 <= bmi < 40 else 0

    alcohol_consumption_per_week_Light = 1 if 0 < alcohol_consumption_per_week <= 3 else 0
    alcohol_consumption_per_week_Moderate = 1 if 3 < alcohol_consumption_per_week <= 7 else 0
    alcohol_consumption_per_week_Heavy = 1 if (7 < alcohol_consumption_per_week) and (alcohol_consumption_per_week != 0) else 0

    input_data = [[family_history, age_30_39, age_40_49, age_50_59, age_60_69, age_70_79, age_80_plus, physical_activity_Moderate, physical_activity_Active, physical_activity_Very_Active, bmi_Normal, bmi_Overweight, bmi_Obese_I, bmi_Obese_II, waist_to_hip_ratio, alcohol_consumption_per_week_Light, alcohol_consumption_per_week_Moderate, alcohol_consumption_per_week_Heavy]]

    return input_data

@router.get("/diagnosis", status_code=status.HTTP_200_OK)
async def diabetes_diagnosis(hba1c: float, glucose_fasting: int):
    model = joblib.load("backend/models/model1.pkl")
    input_data = [[glucose_fasting, hba1c]]
    prediction = model.predict_proba(input_data)
    result = True if prediction[0][1] >= thresh1 else False

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_1))
    shap_df = pd.DataFrame(shap_values, columns=chosen_columns_1)

    if result:
        return {"message": f"shap values: {shap_df.to_dict()}", "diagnosis": "You probably have diabetes."}
    else:
        return {"message": f"shap values: {shap_df.to_dict()}", "diagnosis": "You probably don't have diabetes."}
    
@router.get("/risk_score", status_code=status.HTTP_200_OK)
async def diabetes_risk_score(family_history: int, bmi: float, age: int, waist_to_hip_ratio: float, physical_activity: int, alcohol_consumption_per_week: int):
    model = joblib.load("backend/models/model2.pkl")
    input_data = prepare_for_diagnosis(family_history, bmi, age, waist_to_hip_ratio, physical_activity, alcohol_consumption_per_week)
    prediction = model.predict_proba(input_data)
    result = True if prediction[0][1] >= thresh2 else False
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_2))
    shap_df = pd.DataFrame(shap_values, columns=chosen_columns_2)

    if result:
        return {"message": f"shap values: {shap_df.to_dict()}", "diagnosis": "Your risk of diabetes is high."}
    else:
        return {"message": f"shap values: {shap_df.to_dict()}", "diagnosis": "Your risk of diabetes is low."}