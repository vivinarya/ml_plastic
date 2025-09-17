from flask import Flask, request, jsonify
from src.detector import PixelMicroplasticDetector
import numpy as np

app = Flask(__name__)
detector = PixelMicroplasticDetector()

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    spectra = data.get("spectrum", None)
    if spectra is None or not isinstance(spectra, list):
        return jsonify({"error": "Provide 'spectrum' as a list of floats"}), 400

    spectra = np.array(spectra, dtype=float)
    result = detector.predict(spectra)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)

