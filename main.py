import os
import shutil
import httpx
from typing import Optional
from datetime import datetime
from fastapi import Request
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="file")

ANALYSIS_API_URL = os.environ.get("ANALYSIS_API_URL", "http://localhost:8000/uploadfile/")

async def fetch_analysis_from_api(filename: str) -> dict:
    """Call the analysis API with a filename query param and return JSON dict.

    Returns a dict with an "error" key on failure.
    """
    params = {"filename": filename}
    timeout = httpx.Timeout(10.0, connect=5.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(ANALYSIS_API_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"error": "Invalid response format from analysis API"}
            return data
        except httpx.HTTPError as e:
            return {"error": f"HTTP error when calling analysis API: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to call analysis API: {str(e)}"}


async def send_file_to_analysis_api(file_path: str, filename: str) -> dict:
    """Forward a saved file to the external analysis API as multipart/form-data.

    Returns the parsed JSON from the API or a dict with an "error" key on failure.
    """
    timeout = httpx.Timeout(30.0, connect=10.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            with open(file_path, "rb") as f:
                files = {"file": (filename, f, "audio/mpeg")}
                resp = await client.post(ANALYSIS_API_URL, files=files)
                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, dict):
                    return {"error": "Invalid response format from analysis API"}
                return data
        except httpx.HTTPError as e:
            return {"error": f"HTTP error when sending file to analysis API: {str(e)}"}
        except OSError as e:
            return {"error": f"File I/O error when reading file to send: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error when sending file to analysis API: {str(e)}"}

@app.get("/")
async def index(request: Request):
    """index"""
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request}
    )


@app.get("/result.html")
async def result(request: Request, filename: Optional[str] = None):
    """Result - now fetches analysis from external API and passes it into template"""
    display_filename = filename if filename else "pitching_file.mp3"

    api_result = await fetch_analysis_from_api(display_filename)

    # default/fallback
    current_date = datetime.now().strftime("%d %B %Y")
    context_data = {
        "request": request,
        "filename": display_filename,
        "analysis_date": current_date,
        "score": "ERROR",
        "strengths": [],
        "improvements": []
    }

    if "error" in api_result:
        print("Analysis API error:", api_result["error"])
        context_data["api_error"] = api_result["error"]
    else:
        context_data["score"] = api_result.get("score", context_data["score"])
        context_data["analysis_date"] = api_result.get("analysis_date", context_data["analysis_date"])
        strengths = api_result.get("strengths", [])
        improvements = api_result.get("improvements", [])
        context_data["strengths"] = strengths if isinstance(strengths, list) else [str(strengths)]
        context_data["improvements"] = improvements if isinstance(improvements, list) else [str(improvements)]

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
    api_result = await send_file_to_analysis_api(file_path, sanitized_filename)

    print("Analysis API response:", api_result)

    message = "Analysis failed"
    redirect_url = None

    if "error" in api_result:
        api_result.setdefault("error_detail", api_result.get("error"))
        print("API returned error:", api_result["error"])
    else:
        api_filename = api_result.get("filename")
        if api_filename:
            if api_filename == sanitized_filename:
                message = "Upload and analysis successful"
                redirect_url = f"/result.html?filename={sanitized_filename}"
            else:
                message = "Filename mismatch"
                api_result["error"] = f"Filename mismatch: expected '{sanitized_filename}', got '{api_filename}'"
                print(api_result["error"])
        else:
            message = "Analysis failed"
            api_result["error"] = api_result.get("error") or "Analysis API did not return filename"

    return JSONResponse(content={
        "message": message,
        "filename": file.filename,
        "redirect": redirect_url,
        "api_result": api_result
    })
