from fastapi import FastAPI, UploadFile, File
from PIL import Image

import io
import uvicorn

from model import classify_image



app = FastAPI()

@app.post("/predict/")
async def predict(img: UploadFile = File(...)): 
    img_bytes = await img.read()
    PIL_img = Image.open(io.BytesIO(img_bytes))
    
    return {"prediction" : classify_image(PIL_img)}



if __name__ == "__main__": 
    uvicorn.run(app, port=3000)

