from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import numpy as np 
import pickle

class Agent: 
    def __init__(self):
        try: 
            with open("model.pkl") as f: 
                self.model = pickle.load(f)
        except FileNotFoundError: 
            print("no pickle file found, regression endpoint won't be available.")
            

    def classify_img(self, img_path: str) -> str:
        model = VGG16()
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

    def regression(self, input: list[float]): 
        if self.model: 
            X = np.array(input).reshape(1, -1)
            yhat = self.model.predict(X)
            return float(yhat[0])
        else: 
            return None
            
        

