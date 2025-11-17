from pydantic import BaseModel, validator

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