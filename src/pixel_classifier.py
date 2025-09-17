import numpy as np
import joblib

class PixelMicroplasticDetector:
    def __init__(self, model_dir="pretrained_models"):
        self.clf = joblib.load(f"{model_dir}/pixel_rf_model.joblib")
        self.pca = joblib.load(f"{model_dir}/pca_model.joblib")
        self.le = joblib.load(f"{model_dir}/label_encoder.pkl")

    def predict(self, spectra):
        """
        spectra: np.array of shape (n_pixels, bands) or single (bands,)
        Returns: dict with polymer map and ppm per polymer
        """
        single_input = False
        if spectra.ndim == 1:
            spectra = spectra.reshape(1, -1)
            single_input = True

        # Normalize
        spectra = (spectra - spectra.mean(axis=0)) / (spectra.std(axis=0)+1e-8)
        X_pca = self.pca.transform(spectra)
        y_pred = self.clf.predict(X_pca)
        polymers = self.le.inverse_transform(y_pred)

        # Compute ppm (fraction of pixels per polymer)
        unique, counts = np.unique(polymers, return_counts=True)
        total = len(polymers)
        ppm_dict = {poly: counts[i]/total for i, poly in enumerate(unique)}

        if single_input:
            return {"polymer": polymers[0], "ppm": ppm_dict[polymers[0]]}
        else:
            return {"polymers": polymers, "ppm": ppm_dict}

