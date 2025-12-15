from PIL import Image

from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import torch 
from transformers import AutoTokenizer, AutoModelForCausalLM

import numpy as np 
import pickle

class Agent: 
    def __init__(self):
        try: 
            with open("model.pkl", 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e: 
            print(f"regression endpoint won't be available due to: {str(e)}")
        
        try: 
            self.img_model = VGG16()
        except Exception as e: 
            print(f"img classification endpoint won't be available due to: {str(e)}")

        try: 
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.llm = AutoModelForCausalLM.from_pretrained("gpt2")
            self.llm.eval()  # inference mode
        except Exception as e: 
                        print(f"chat endpoint won't be available due to: {str(e)}")




            

    def classify_img(self, img_path: str) -> str:
        image = Image.open(img_path)
        image = image.resize((224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis = 0)
        image = preprocess_input(image)

        yhat = self.img_model.predict(image)

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
        



    def generate_text(self, req):
        inputs = self.tokenizer(req.prompt, return_tensors="pt")

        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=req.max_new_tokens,
                do_sample=True,
                temperature=0.9,
                top_p=0.95,
                repetition_penalty=1.2
            )

            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )

            return generated_text 



        

