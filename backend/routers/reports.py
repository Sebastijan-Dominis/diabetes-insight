import joblib
from dotenv import load_dotenv
from fastapi import APIRouter
from starlette import status
from sklearn.preprocessing import StandardScaler

load_dotenv()

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
)

thresh1 = 0.28
scaler1 = joblib.load("backend/models/has_or_not_scaler.pkl")

scaler2 = joblib.load("backend/models/risk_score_scaler.pkl")

def prepare_for_diagnosis(hba1c: float, glucose_fasting: int):
    input_data = [[hba1c, glucose_fasting]]
    return scaler1.transform(input_data)

def prepare_for_diagnosis_2(family_history, bmi, age, waist_to_hip_ratio, physical_activity):
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

    input_data = [[family_history, age_60_69, age_50_59, physical_activity_Active, age_70_79, age_40_49, waist_to_hip_ratio, age_30_39, physical_activity_Very_Active, physical_activity_Moderate, age_80_plus, bmi_Normal, bmi_Overweight, bmi_Obese_I, bmi_Obese_II]]

    return scaler2.transform(input_data)

@router.get("/diagnosis", status_code=status.HTTP_200_OK)
async def diabetes_diagnosis(hba1c: float, glucose_fasting: int):
    model = joblib.load("backend/models/cat_1.joblib")
    input_data = prepare_for_diagnosis(hba1c, glucose_fasting)
    prediction = model.predict_proba(input_data)
    result = True if prediction[0][1] >= thresh1 else False
    if result:
        return {"message": "You probably have diabetes."}
    else:
        return {"message": "You probably do not have diabetes."}
    
@router.get("/risk_score", status_code=status.HTTP_200_OK)
async def diabetes_risk_score(family_history: int, bmi: float, age: int, waist_to_hip_ratio: float, physical_activity: int):
    model = joblib.load("backend/models/cat_2.joblib")
    input_data = prepare_for_diagnosis_2(family_history, bmi, age, waist_to_hip_ratio, physical_activity)
    result = model.predict_proba(input_data)
    print(result)
    return {"risk_score": f"{result}"}