#  Water Pollution Detector – Microplastic Classification

This project detects **microplastics in water** using **hyperspectral imaging** and **machine learning**.  
It takes spectral cube data (ENVI `.hdr` + `.dat` files), labels (`.ilab`), and masks (`.imsk`), then trains a classifier to identify whether each pixel contains microplastic contamination or not.

---

##Project Structure

water-pollution-detector/
│
├── data/ # Input datasets
│ ├── microplastics.hdr # Header file for spectral cube
│ ├── microplastics.dat # Raw spectral data
│ ├── microplastics.ilab # Pixel-level labels
│ ├── microplastics.imsk # Pixel mask (valid pixels)
│
├── scripts/
│ ├── train_pixel_classifier.py # Train ML model on pixel spectra
│ └── evaluate_model.py # Evaluate saved model (optional)
│
├── src/
│ ├── data_utils.py # Load, preprocess, extract features
│ └── model_utils.py # Build classifier, save/load models
│
├── models/
│ └── pixel_classifier.pkl # Saved trained classifier
│
├── requirements.txt # Python dependencies
└── README.md # Project overview

markdown
Copy code

---

##  How It Works

1. **Load Data**
   - Reads hyperspectral cube (`.hdr` + `.dat`)
   - Loads mask (`.imsk`) and labels (`.ilab`)

2. **Preprocess**
   - Normalizes pixel spectra
   - Flattens cube → (pixels × bands)
   - Selects valid pixels using mask

3. **Train Model**
   - Optionally reduces dimensions with PCA
   - Trains classifier (RandomForest by default)
   - Splits into train/test for evaluation

4. **Evaluate**
   - Prints accuracy, classification report
   - Saves trained model (`models/pixel_classifier.pkl`)

---

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/water-pollution-detector.git
cd water-pollution-detector
Install dependencies:

bash
Copy code
pip install -r requirements.txt
▶️ Usage
Train the classifier:
bash
Copy code
python -m scripts.train_pixel_classifier
Evaluate saved model:
bash
Copy code
python -m scripts.evaluate_model
