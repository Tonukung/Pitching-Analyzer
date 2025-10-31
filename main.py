import os
import httpx
from typing import Optional
from datetime import datetime
from fastapi import Request
from fastapi import FastAPI, File, UploadFile
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.responses import JSONResponse, RedirectResponse

app = FastAPI()

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="file")

ANALYSIS_API_BASE_URL = os.environ.get("ANALYSIS_API_BASE_URL", "http://localhost:3000")
ANALYSIS_UPLOAD_URL = f"{ANALYSIS_API_BASE_URL}/uploadfile/"
ANALYSIS_STATUS_URL = f"{ANALYSIS_API_BASE_URL}/status/"

async def fetch_analysis_from_api(filename: str) -> dict:
    """Call the analysis API with a filename query param and return JSON dict."""
    params = {"filename": filename}
    timeout = httpx.Timeout(30.0, connect=5.0) # ลด Timeout ลงเพราะเราแค่ดึงข้อมูลที่ cache ไว้
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(ANALYSIS_UPLOAD_URL, params=params)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"error": "Invalid response format from analysis API"}
            return data
        except httpx.HTTPError as e:
            return {"error": f"HTTP error when calling analysis API: {str(e)}"}
        except Exception as e:
            return {"error": f"Failed to call analysis API: {str(e)}"}

# ✅ แก้ไขตรงนี้ ให้รับ UploadFile โดยตรง
async def send_file_to_analysis_api(file: UploadFile) -> dict:
    """Send uploaded file directly to external analysis API."""
    timeout = httpx.Timeout(300.0, connect=5.0) # Increased timeout to 5 minutes
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            resp = await client.post(ANALYSIS_UPLOAD_URL, files=files)
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, dict):
                return {"error": "Invalid response format from analysis API"}
            return data
        except httpx.HTTPError as e:
            return {"error": f"HTTP Error calling Analysis API: {str(e)}"}
        except Exception as e:
            return {"error": f"Unexpected error: {str(e)}"}

@app.get("/")
async def index(request: Request):
    """index"""
    return templates.TemplateResponse(
        name="index.html",
        context={"request": request}
    )

@app.get("/processing")
async def processing(request: Request, filename: str):
    """Displays a page that polls for the analysis result."""
    return templates.TemplateResponse("processing.html", {
        "request": request, "filename": filename
    })

@app.get("/result.html")
async def result(request: Request, filename: Optional[str] = None):
    """Result - fetches analysis from external API and passes it into template"""
    display_filename = filename if filename else "pitching_file.mp3"
    api_result = await fetch_analysis_from_api(display_filename)

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
    """
    Receives a file from the user and forwards it to the analysis API.
    Then, redirects the user to the result page.
    """
    api_result = await send_file_to_analysis_api(file)

    print("Analysis API response:", api_result)

    if "error" in api_result:
        return JSONResponse(status_code=500, content={"detail": api_result["error"]})
    
    api_filename = api_result.get("filename")
    if not api_filename:
        return JSONResponse(status_code=500, content={"detail": "Analysis API did not return a filename."})
    
    # Return the filename in a JSON response so the frontend can start polling.
    return JSONResponse(content={
        "message": "File uploaded successfully. Analysis has started.",
        "filename": api_filename
    })

@app.get("/check_status")
async def check_status(filename: str):
    """
    Frontend polls this endpoint. This endpoint polls the analysis service.
    """
    async with httpx.AsyncClient() as client:
        try:
            # Call the new /status/{filename} endpoint on the analysis service
            resp = await client.get(f"{ANALYSIS_STATUS_URL}{filename}")
            resp.raise_for_status()
            data = resp.json()
            return JSONResponse(content=data)
        except httpx.HTTPStatusError as e:
            return JSONResponse(status_code=e.response.status_code, content={"status": "error", "detail": f"Analysis service error: {e.response.text}"})
        except Exception as e:
            return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
