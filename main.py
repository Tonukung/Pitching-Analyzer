"""Backend"""

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse
from datetime import datetime
from fastapi import Request
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="file")

@app.get("/")
async def index(request: Request):
    """index"""
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request}
    )

@app.get("/result.html")
async def result(request: Request, filename: str | None = None):
    """Result"""
    current_date = datetime.now().strftime("%d %B %Y")
    display_filename = filename if filename else "pitching_file.mp3"

    context_data = {
        "request": request,
        "filename": display_filename,
        "analysis_date": current_date,
        "score": "ERROR"
    }
    return templates.TemplateResponse(
        name="result.html",
        context=context_data
    )

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    """Upload file"""
    sanitized_filename = file.filename.replace(" ", "_")
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    print(f"File '{file.filename}' Upload Complete as '{sanitized_filename}'")

    redirect_url = f"/result.html?filename={sanitized_filename}"
    return JSONResponse(content={
        "message": "Upload successful", 
        "filename": file.filename, 
        "redirect": redirect_url
    })