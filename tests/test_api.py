import json
from api_server import app

def test_predict_api():
    client = app.test_client()
    response = client.post("/predict", json={"data": [0.1]*100})
    assert response.status_code == 200
    assert "prediction" in response.json
