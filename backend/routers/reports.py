import os
from dotenv import load_dotenv
import joblib
import shap
import pandas as pd
from fastapi import APIRouter, HTTPException, Request
from starlette import status
from openai import OpenAI

from ..limiter import limiter
from ..models import DiabetesInput, RiskInput
from ..utils import prepare_for_diagnosis, generate_pdf_report

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

# Thresholds optimized for f1-score during model evaluation
thresh1 = 0.3
# thresh2 = 0.77

# Load models
try:
    model1 = joblib.load("backend/models/model1.pkl")
    model2 = joblib.load("backend/models/model2.pkl")
except FileNotFoundError:
    raise HTTPException(status_code=500, detail="Model file not found.")

# Names of the columns that the models expect
chosen_columns_1 = ['glucose_fasting', 'hba1c']

chosen_columns_2 = ["family_history_diabetes", "age_30-39", "age_40-49", "age_50-59", "age_60-69", "age_70-79", "age_80+", "physical_activity_minutes_per_week_Moderate", "physical_activity_minutes_per_week_Active", "physical_activity_minutes_per_week_Very Active", "bmi_Normal", "bmi_Overweight", "bmi_Obese_I", "bmi_Obese_II", "waist_to_hip_ratio", "alcohol_consumption_per_week_Light", "alcohol_consumption_per_week_Moderate", "alcohol_consumption_per_week_Heavy"]

##########################################
# ENDPOINTS FOR DIAGNOSIS AND RISK SCORE #
##########################################

@router.post("/diagnosis", status_code=status.HTTP_200_OK)
@limiter.limit("3/minute")
async def diabetes_diagnosis(data: DiabetesInput, request: Request):
    # Prepare input data and make prediction
    try:
        input_data = [[data.glucose_fasting, data.hba1c]]
        prediction = model1.predict_proba(input_data)
        result = "has diabetes" if prediction[0][1] >= thresh1 else "does not have diabetes"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during prediction: {str(e)}")
    
    # Calculate SHAP values
    try:
        explainer = shap.TreeExplainer(model1)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_1))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_1)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")

    # Generate explanation using an LLM
    system_prompt = "You are a helpful assistant that provides explanations for diabetes diagnosis based on SHAP values. You will be given raw responses and SHAP values for glucose_fasting and hba1c, along with the model's diagnosis prediction. Provide a concise explanation of how these values contribute to the diabetes diagnosis. Give actionable advice based on the values. Point to relevant medical sources as references. Reply in a way that a patient can easily understand, avoiding technical jargon. They do not need to know about SHAP values specifically. They are only interested in how their glucose_fasting and hba1c levels affect their diabetes diagnosis and what they can do to improve their health. Respond in proper Markdown format that can be rendered in a PDF report."

    user_prompt = f"I will give you my patient's diabetes diagnosis based on my model, along with the shap values and their raw input data. I want you to explain to him/her how these values contribute to their diagnosis, in laymen's terms. Give them some actionable advice on how to improve their health, while taking these figures into consideration. Point them to relevant medical sources as references. Here is your data:\n\nTheir raw responses: {data}\n\nThe prediction is: {result}.\n\nThe SHAP values are as follows: {shap_df.to_dict()}."

    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during explanation generation: {str(e)}")

    # Generate a PDF report
    try:
        pdf = generate_pdf_report(explanation=explanation, title="Diabetes Diagnosis Report")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF generation: {str(e)}")
    
    return pdf

    
@router.post("/risk_score", status_code=status.HTTP_200_OK)
@limiter.limit("3/minute")
async def diabetes_risk_score(data: RiskInput, request: Request):
    # Prepare input data and make prediction
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
    
    # Calculate SHAP values
    try:    
        explainer = shap.TreeExplainer(model2)
        shap_values = explainer.shap_values(pd.DataFrame(input_data, columns=chosen_columns_2))
        shap_df = pd.DataFrame(shap_values, columns=chosen_columns_2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during SHAP value calculation: {str(e)}")
    
    # Generate explanation using an LLM
    system_prompt = "You are a helpful assistant that provides explanations for diabetes risk score based on SHAP values. You will be given raw responses and SHAP values for family_history, bmi, age, waist_to_hip_ratio, physical_activity, and alcohol_consumption_per_week, along with the model's risk score prediction. family_history is a binary indicating one having or not having a family history of diabetes. BMI is binned into standard bins, such as bmi_Normal (18.5-24.9), bmi_Overweight (25-29.9), and bmi_Obese_I (30-34.9). Age is binned into standard bins, such as age_30-39 (ages 30 to 39) and age_40-49 (ages 40 to 49). physical_activity is expressed in minutes per week, and binned into the following bins: Sedentary for -1-30, Light for 31-100, Moderate for 100-150, Active for 150-300, and Very Active for 301+ minutes per week of physical exercise. Provide a concise explanation of how these values contribute to the diabetes risk score. Give actionable advice based on the values. Point to relevant medical sources as references. Reply in a way that a patient can easily understand, avoiding technical jargon. They do not need to know about SHAP values specifically. They are only interested in how their family history, bmi, age, waist to hip ratio, physical activity, and alcohol consumption affect their diabetes risk score and what they can do to improve their health. Respond in proper Markdown format that can be rendered in a PDF report."

    user_prompt = f"I will give you my patient's diabetes risk score based on my model, along with the shap values and their raw input data. I want you to explain to him/her how these values contribute to their risk score, in laymen's terms. Give them some actionable advice on how to improve their health, while taking these figures into consideration. Point them to relevant medical sources as references. Here is your data:\n\nTheir raw responses: {data}\n\nThe prediction is: {result}.\n\nThe SHAP values are as follows: {shap_df.to_dict()}."

    # Generate explanation using an LLM
    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        explanation = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during explanation generation: {str(e)}")

    try:
        pdf = generate_pdf_report(explanation=explanation, title="Diabetes Risk Score Report")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred during PDF generation: {str(e)}")
    
    return pdf 