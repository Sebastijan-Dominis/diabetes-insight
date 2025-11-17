import os
from dotenv import load_dotenv
import joblib
import shap
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from starlette import status
from pydantic import BaseModel, validator
from openai import OpenAI

from ..limiter import limiter

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
)

load_dotenv()

api_url = os.getenv("API_URL")

if not api_url:
    raise HTTPException(status_code=500, detail="API URL not found in environment variables.")

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise HTTPException(status_code=500, detail="OpenAI API key not found in environment variables.")

openai = OpenAI(api_key=openai_api_key)

thresh1 = 0.3
# thresh2 = 0.77

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

##########################################
# ENDPOINTS FOR DIAGNOSIS AND RISK SCORE #
##########################################

@router.post("/diagnosis", status_code=status.HTTP_200_OK)
@limiter.limit("3/minute")
async def diabetes_diagnosis(data: DiabetesInput, request: Request):
    try:
        input_data = [[data.glucose_fasting, data.hba1c]]
        prediction = model1.predict_proba(input_data)
        result = "has diabetes" if prediction[0][1] >= thresh1 else "does not have diabetes"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    try:
        explainer = shap.TreeExplainer(model1)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_1))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")

    # system_prompt = "You are a helpful assistant that provides explanations for diabetes diagnosis based on SHAP values. You will be given raw responses and SHAP values for glucose_fasting and hba1c, along with the model's diagnosis prediction. Provide a concise explanation of how these values contribute to the diabetes diagnosis. Give actionable advice based on the values. Point to relevant medical sources as references. Reply in a way that a patient can easily understand, avoiding technical jargon. They do not need to know about SHAP values specifically. They are only interested in how their glucose_fasting and hba1c levels affect their diabetes diagnosis and what they can do to improve their health."

    # user_prompt = f"I will give you my patient's diabetes diagnosis based on my model, along with the shap values and their raw input data. I want you to explain to him/her how these values contribute to their diagnosis, in laymen's terms. Give them some actionable advice on how to improve their health, while taking these figures into consideration. Point them to relevant medical sources as references. Here is your data:\n\nTheir raw responses: {data}\n\nThe prediction is: {result}.\n\nThe SHAP values are as follows: {shap_df.to_dict()}."

    # try:
    #     response = openai.chat.completions.create(
    #         model="gpt-5-mini",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ]
    #     )
    #     explanation = response.choices[0].message.content.strip()
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"An error occurred during explanation generation: {str(e)}")

    # return {"message": explanation}

    return {"message": result}

    
@router.post("/risk_score", status_code=status.HTTP_200_OK)
@limiter.limit("3/minute")
async def diabetes_risk_score(data: RiskInput, request: Request):
    try:
        input_data = prepare_for_diagnosis(data.family_history, data.bmi, data.age, data.waist_to_hip_ratio, data.physical_activity, data.alcohol_consumption_per_week)
        prediction = model2.predict_proba(input_data)[0][1]
        if prediction < 0.1:
            result = "very low risk"
        elif 0.1 <= prediction < 0.2:
            result = "low risk"
        elif 0.2 <= prediction < 0.4:
            result = "moderate risk"
        elif 0.4 <= prediction < 0.7:
            result = "high risk"
        else:
            result = "very high risk"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    try:    
        explainer = shap.TreeExplainer(model2)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_2))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")
    
    # system_prompt = "You are a helpful assistant that provides explanations for diabetes risk score based on SHAP values. You will be given raw responses and SHAP values for family_history, bmi, age, waist_to_hip_ratio, physical_activity, and alcohol_consumption_per_week, along with the model's risk score prediction. family_history is a binary indicating one having or not having a family history of diabetes. BMI is binned into standard bins, such as bmi_Normal (18.5-24.9), bmi_Overweight (25-29.9), and bmi_Obese_I (30-34.9). Age is binned into standard bins, such as age_30-39 (ages 30 to 39) and age_40-49 (ages 40 to 49). physical_activity is expressed in minutes per week, and binned into the following bins: Sedentary for -1-30, Light for 31-100, Moderate for 100-150, Active for 150-300, and Very Active for 301+ minutes per week of physical exercise. Provide a concise explanation of how these values contribute to the diabetes risk score. Give actionable advice based on the values. Point to relevant medical sources as references. Reply in a way that a patient can easily understand, avoiding technical jargon. They do not need to know about SHAP values specifically. They are only interested in how their family history, bmi, age, waist to hip ratio, physical activity, and alcohol consumption affect their diabetes risk score and what they can do to improve their health."

    # user_prompt = f"I will give you my patient's diabetes risk score based on my model, along with the shap values and their raw input data. I want you to explain to him/her how these values contribute to their risk score, in laymen's terms. Give them some actionable advice on how to improve their health, while taking these figures into consideration. Point them to relevant medical sources as references. Here is your data:\n\nTheir raw responses: {data}\n\nThe prediction is: {result}.\n\nThe SHAP values are as follows: {shap_df.to_dict()}."

    # try:
    #     response = openai.chat.completions.create(
    #         model="gpt-5-mini",
    #         messages=[
    #             {"role": "system", "content": system_prompt},
    #             {"role": "user", "content": user_prompt}
    #         ]
    #     )
    #     explanation = response.choices[0].message.content.strip()
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"An error occurred during explanation generation: {str(e)}")

    # return {"message": explanation}

    return {"message": result}