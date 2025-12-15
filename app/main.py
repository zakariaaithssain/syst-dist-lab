from fastapi import FastAPI, UploadFile, File, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from pathlib import Path

import uvicorn
import shutil
import os 

from agent import Agent

from werkzeug.utils import secure_filename

TEMPLATES_DIR = Path("app/templates")
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI(title="ML Deployment App")
agent = Agent()

#models endpoints (post methods with json returns)
@app.post("/predict")
def predict(file: UploadFile = File(...)): 
    safe_name = secure_filename(file.filename)
    file_path = UPLOAD_DIR.joinpath(safe_name)
   
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    prediction = agent.classify_img(file_path)
    os.remove(file_path)

    return {"prediction" : prediction}



class RegRequest(BaseModel): 
    x1: float
    x2: float
    x3: float

@app.post("/regpredict")
def predict(
            x1: float = Form(...),
            x2: float = Form(...),
            x3: float = Form(...)):
    X = [x1, x2, x3]
    yhat = agent.regression(X)
    return {"prediction": yhat}



@app.post("/textgen")
def text(prompt: str = Form(...), max_new_tokens: int = Form(...)): 
    generated_text = agent.generate_text(prompt, max_new_tokens)

    return {
            "prompt": prompt,
            "generated_text": generated_text
        }


#ui endpoints (get methods with html return)
templates = Jinja2Templates(TEMPLATES_DIR)

@app.get("/home", response_class=HTMLResponse)
def home(request: Request): 
    return templates.TemplateResponse(request=request, name="home.html")


@app.get("/predict-ui", response_class=HTMLResponse)
def predict_ui(request: Request): 
    return templates.TemplateResponse(request=request, name="predict.html")

@app.get("/regpredict-ui", response_class=HTMLResponse)
def regpredict_ui(request: Request): 
    return templates.TemplateResponse(request=request, name="regpredict.html")

@app.get("/textgen-ui", response_class=HTMLResponse)
def textgen_ui(request: Request): 
    return templates.TemplateResponse(request=request, name="textgen.html")





if __name__ == "__main__": 
    uvicorn.run(app, port=3000)

