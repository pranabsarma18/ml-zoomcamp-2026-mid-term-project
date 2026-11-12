import pickle
from fastapi import FastAPI
import uvicorn
import pandas as pd
from typing import Dict, Any

from typing import Literal
from pydantic import BaseModel, Field

MODEL_FILE = 'model.bin'

app = FastAPI(title="student-depression-prediction")

@app.get("/")
def home():
    return {"message": "Student Depression Prediction API"}


def load_model():
    with open(MODEL_FILE, 'rb') as f_in:
        pipeline = pickle.load(f_in)

    return pipeline



class Student(BaseModel):

    # Demographic features
    gender: Literal["male", "female"]
    age: int = Field(..., ge=18, le=59)
    city: str
    profession: str

    # Academic and work-related features
    academic_pressure: float = Field(..., ge=0, le=5)
    work_pressure: float = Field(..., ge=0, le=5)
    cgpa: float = Field(..., ge=0, le=10)
    study_satisfaction: float = Field(..., ge=0, le=5)
    job_satisfaction: float = Field(..., ge=0, le=5)
    work_or_study_hours: float = Field(..., ge=0, le=24)

    # Lifestyle features
    sleep_duration: str

    dietary_habits: Literal[
        "healthy",
        "moderate",
        "unhealthy",
        "others"
    ]

    # Financial and family features
    financial_stress: float = Field(..., ge=0, le=5)

    family_history_of_mental_illness: Literal[
        "yes",
        "no"
    ]

    # Educational qualification
    degree: Literal[
        "class_12", "b.ed", "b.com", "b.arch",
        "bca", "msc", "b.tech", "mca", "m.tech",
        "bhm", "bsc", "m.ed", "b.pharm", "m.com",
        "mbbs", "bba", "llb", "be", "ba", "m.pharm",
        "md", "mba", "ma", "phd", "llm", "mhm",
        "me", "others"
    ]


class PredictResponse(BaseModel):
    depression_probability: float = Field(..., ge=0, le=1)
    depression_prediction: Literal[0, 1]


def prepare_data(datapoint: Student):
    df = pd.DataFrame([datapoint.model_dump()])

    # Normalize column names
    df.columns = (
        df.columns.str.lower()
        .str.replace(' ', '_')
        .str.replace('/', '_or_')
    )

    # Rename columns
    df = df.rename(columns={
        'have_you_ever_had_suicidal_thoughts_?': 'suicidal_thoughts'
    })

    # Normalize categorical values
    categorical_columns = df.select_dtypes(
        include=['object', 'string']
    ).columns

    for col in categorical_columns:
        df[col] = (
            df[col]
            .str.lower()
            .str.replace(' ', '_', regex=False)
        )

    # Convert sleep duration to numerical average
    df['sleep_duration'] = (
        df['sleep_duration']
        .astype(str)
        .str.extract(r'(\d+)(?:-(\d+))?')
        .astype(float)
        .mean(axis=1)
    )

    # Remove excluded columns
    df = df.drop(
        columns=['id', 'depression', 'suicidal_thoughts'],
        errors='ignore'
    )

    return df

@app.post("/predict")
def predict(datapoint: Student, pipeline=None) -> PredictResponse:

    if pipeline is None:
        pipeline = load_model()

    df = prepare_data(datapoint)

    probability = pipeline.predict_proba(df)[0, 1]
    prediction = int(probability >= 0.5)

    return {
        'depression_probability': float(probability),
        'depression_prediction': prediction
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)