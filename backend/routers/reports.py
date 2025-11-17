import joblib
import shap
import pandas as pd
from fastapi import APIRouter, HTTPException
from starlette import status
from pydantic import BaseModel, validator

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
)

thresh1 = 0.3
thresh2 = 0.77

try:
    model1 = joblib.load("backend/models/model1.pkl")
    model2 = joblib.load("backend/models/model2.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found.")

chosen_columns_1 = ['glucose_fasting', 'hba1c']

chosen_columns_2 = ["family_history_diabetes", "age_30-39", "age_40-49", "age_50-59", "age_60-69", "age_70-79", "age_80+", "physical_activity_minutes_per_week_Moderate", "physical_activity_minutes_per_week_Active", "physical_activity_minutes_per_week_Very Active", "bmi_Normal", "bmi_Overweight", "bmi_Obese_I", "bmi_Obese_II", "waist_to_hip_ratio", "alcohol_consumption_per_week_Light", "alcohol_consumption_per_week_Moderate", "alcohol_consumption_per_week_Heavy"]

class DiabetesInput(BaseModel):
    hba1c: float
    glucose_fasting: float

    @validator("hba1c")
    def hba1c_range(cls, v):
        if not (3.0 <= v <= 15.0):
            raise ValueError("hba1c must be between 3.0 and 15.0")
        return v

    @validator("glucose_fasting")
    def glucose_fasting_range(cls, v):
        if not (50.0 <= v <= 300.0):
            raise ValueError("glucose_fasting must be between 50.0 and 300.0")
        return v

class RiskInput(BaseModel):
    family_history: int
    bmi: float
    age: int
    waist_to_hip_ratio: float
    physical_activity: int
    alcohol_consumption_per_week: int

    @validator("family_history")
    def family_history_range(cls, v):
        if v not in (0, 1):
            raise ValueError("family_history must be 0 or 1")
        return v
    
    @validator("bmi")
    def bmi_range(cls, v):
        if not (10.0 <= v <= 70.0):
            raise ValueError("bmi must be between 10.0 and 70.0")
        return v
    
    @validator("age")
    def age_range(cls, v):
        if not (0 <= v <= 120):
            raise ValueError("age must be between 0 and 120")
        return v
    
    @validator("waist_to_hip_ratio")
    def waist_to_hip_ratio_range(cls, v):
        if not (0.4 <= v <= 2.0):
            raise ValueError("waist_to_hip_ratio must be between 0.4 and 2.0")
        return v
    
    @validator("physical_activity")
    def physical_activity_range(cls, v):
        if not (0 <= v <= 2400):
            raise ValueError("physical_activity must be between 0 and 2400 minutes per week")
        return v
    
    @validator("alcohol_consumption_per_week")
    def alcohol_consumption_per_week_range(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("alcohol_consumption_per_week must be between 0 and 100 units")
        return v

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

@router.post("/diagnosis", status_code=status.HTTP_200_OK)
async def diabetes_diagnosis(data: DiabetesInput):
    try:
        input_data = [[data.glucose_fasting, data.hba1c]]
        prediction = model1.predict_proba(input_data)
        result = True if prediction[0][1] >= thresh1 else False
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    try:
        explainer = shap.TreeExplainer(model1)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_1))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")

    if result:
        return {"shap_values": shap_df.to_dict(), "diagnosis": "You probably have diabetes."}
    else:
        return {"shap_values": shap_df.to_dict(), "diagnosis": "You probably don't have diabetes."}
    
@router.post("/risk_score", status_code=status.HTTP_200_OK)
async def diabetes_risk_score(data: RiskInput):
    try:
        input_data = prepare_for_diagnosis(data.family_history, data.bmi, data.age, data.waist_to_hip_ratio, data.physical_activity, data.alcohol_consumption_per_week)
        prediction = model2.predict_proba(input_data)
        result = True if prediction[0][1] >= thresh2 else False  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    try:    
        explainer = shap.TreeExplainer(model2)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_2))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")
    
    if result:
        return {"shap_values": shap_df.to_dict(), "diagnosis": "Your risk of diabetes is high."}
    else:
        return {"shap_values": shap_df.to_dict(), "diagnosis": "Your risk of diabetes is low."}

# TODO : Add error handling for model loading and prediction steps