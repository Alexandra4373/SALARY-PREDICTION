"""
app/main.py - FastAPI API + Gradio web page in one app.

Run locally from the PROJECT ROOT:
    uvicorn app.main:app --reload
Web page:  http://localhost:8000/ui      API docs:  http://localhost:8000/docs
"""

import json
from pathlib import Path

# from fastapi.responses import RedirectResponse
import gradio as gr
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent  # project root folder
model = joblib.load(
    ROOT / "model" / "salary_model.pkl"
)  # trained pipeline (made by train.py)
meta = json.loads(
    (ROOT / "model" / "metadata.json").read_text()
)  # dropdown choices, ranges

app = FastAPI(title="Salary Prediction API")  # the API


class Person(BaseModel):  # JSON the /predict endpoint expects
    age: int
    years_of_experience: float
    education_level: str
    job_title: str


def predict_salary(age, years_of_experience, education_level, job_title):
    # Column names must match the ones used in training exactly
    row = pd.DataFrame(
        [
            {
                "Age": age,
                "Years of Experience": years_of_experience,
                "Education Level": education_level,
                "Job Title": job_title,
            }
        ]
    )
    return round(float(model.predict(row)[0]), 2)  # pipeline encodes, then predicts


@app.get("/health")  # "is the app alive?" check
def health():
    return {"status": "ok"}


@app.post("/predict")  # the API endpoint
def predict(p: Person):
    return {
        "predicted_salary": predict_salary(
            p.age, p.years_of_experience, p.education_level, p.job_title
        )
    }


demo = gr.Interface(  # the web page
    fn=predict_salary,  # called when the user clicks Submit
    inputs=[  # order must match fn's arguments
        gr.Number(label="Age", value=30, precision=0),
        gr.Number(label="Years of experience", value=5),
        gr.Dropdown(
            meta["education_levels"], label="Education level", value="Bachelor's Degree"
        ),
        gr.Dropdown(meta["job_titles"], label="Job title", value="Software Engineer"),
    ],
    outputs=gr.Number(label="Predicted salary"),
    title="Salary Predictor",
    description="Estimate a salary from age, experience, education and job title.",
)
# @app.get("/", include_in_schema=False)           # the home address
# def home():
#     return RedirectResponse(url="/ui")           # send visitors to the Gradio page
app = gr.mount_gradio_app(app, demo, path="/ui")  # attach the web page at /ui
