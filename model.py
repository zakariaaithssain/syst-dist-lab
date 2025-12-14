from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import numpy as np 


model = VGG16()

def classify_image(img_path: str) -> str:
    image = Image.open(img_path)
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)

    yhat = model.predict(image)

    label = decode_predictions(yhat)
    label = label[0][0]
    classification = f"{label[1]}, {label[2] * 100:.2f}%"
    return classification

