# If your detector is in data_utils.py
from src.data_utils import PixelMicroplasticDetector

# Or if you're using the training script
from src.train_pixel_classifier import main


# Load cube
cube_path = "data/microplastic.dat"
cube = load_cube(cube_path)

# Flatten cube
height, width, bands = cube.shape
spectra = cube.reshape(-1, bands)

# Load detector
detector = PixelMicroplasticDetector()
result = detector.predict(spectra)

print("PPM per polymer:")
print(result["ppm"])
