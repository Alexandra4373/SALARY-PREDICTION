from fastapi.testclient import (
    TestClient,  # lets us call the API without starting a server
)

from app.main import app

client = TestClient(app)

def test_health():
    assert client.get("/health").json() == {"status": "ok"}   # app is alive

def test_predict():
    r = client.post("/predict", json={
        "age": 32, "years_of_experience": 5,
        "education_level": "Bachelor's Degree", "job_title": "Software Engineer"})
    assert r.status_code == 200
    assert r.json()["predicted_salary"] > 0                   # got a sensible number