"""Backend"""

from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from starlette.responses import FileResponse
import shutil
import os

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI()

# Mount static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    """Index"""
    return FileResponse('index.html')

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    # You can access file details like filename and content type
    file_location = os.path.join(UPLOAD_DIR, file.filename)
    
    # Save the file to disk asynchronously
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    finally:
        # It's a good practice to close the UploadFile object
        await file.close()

    return {"filename": file.filename, "message": "File uploaded successfully"}
