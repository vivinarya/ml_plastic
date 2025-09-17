import torch
from src.water_pollution_detector import WaterPollutionDetector

def test_cnn_detector():
    detector = WaterPollutionDetector(model_type="cnn")
    dummy_input = torch.randn(1, 1, 100)
    prediction = detector.predict(dummy_input)
    assert prediction in [0, 1]
