from fastapi import FastAPI, UploadFile, File

from pathlib import Path

import uvicorn
import shutil

from model import classify_image
from werkzeug.utils import secure_filename


UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

app = FastAPI()

@app.post("/predict/")
def predict(img: UploadFile = File(...)): 
    safe_name = secure_filename(img.filename)
    file_path = UPLOAD_DIR.joinpath(safe_name)
    
    with file_path.open("wb") as buffer:
        shutil.copyfileobj(img.file, buffer)
    
    return {'prediction' : classify_image(file_path)}

    







if __name__ == "__main__": 
    uvicorn.run(app, port=3000)

