import numpy as np
from PIL import Image

IMG_SIZE = 128

def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """Resize and scale a PIL Image for model inference without requiring TensorFlow.

    Returns a NumPy array shaped (1, H, W, C) with float32 values in [0,1].
    """
    if not isinstance(image, Image.Image):
        image = Image.open(image)

    image = image.convert("RGB")
    image = image.resize(target_size)
    arr = np.array(image).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr
