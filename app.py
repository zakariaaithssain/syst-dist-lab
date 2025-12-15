from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel

from pathlib import Path

import uvicorn
import shutil
import os 

from agent import Agent

from werkzeug.utils import secure_filename


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()
agent = Agent()


@app.post("/predict/")
def predict(img: UploadFile = File(...)): 
    safe_name = secure_filename(img.filename)
    file_path = UPLOAD_DIR.joinpath(safe_name)
   
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(img.file, buffer)

    prediction = agent.classify_img(file_path)
    os.remove(file_path)

    return {"prediction" : prediction}



class RegRequest(BaseModel): 
    features: list[float]

@app.post("/regpredict")
def predict(features: RegRequest):
    X = features.features
    yhat = agent.regression(X)

    return {"prediction": yhat}



class GenerationRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 50


@app.post("/textgen")
def text(prompt: GenerationRequest): 
    generated_text = agent.generate_text(prompt)

    return {
            "prompt": prompt.prompt,
            "generated_text": generated_text
        }





if __name__ == "__main__": 
    uvicorn.run(app, port=3000)

