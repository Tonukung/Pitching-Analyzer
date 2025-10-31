import os
import json
import librosa
import shutil
from dotenv import load_dotenv
from pydub import AudioSegment
from groq import Groq
from faster_whisper import WhisperModel
from datetime import datetime
from fastapi import FastAPI, Query, HTTPException, UploadFile, File, BackgroundTasks
from starlette.responses import JSONResponse

load_dotenv()
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

api_key = os.getenv("GROQ_API_KEY")
print("🔑 Loaded key from env:", api_key[:10] if api_key else None)  # ตรวจว่ามาไหม

if not api_key:
    raise RuntimeError("❌ GROQ_API_KEY not found. Check .env location or syntax")

client = Groq(api_key=api_key)
print("📁 .env path:", os.path.join(os.path.dirname(__file__), ".env"))
print("🔍 Current working dir:", os.getcwd())

ASR_MODEL_NAME = "base"
ASR_COMPUTE_TYPE = "int8" # "float16" หรือ "int8" เพื่อความเร็ว

LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# Prompt เอาไว้ให้ Whisper ถอดคำฟุ่มเฟือยออกมา
PROMPT_FOR_FILLERS = """
ต่อไปนี้คือการถอดเสียงการนำเสนอแบบคำต่อคำ
กรุณาคงคำฟุ่มเฟือยทั้งหมดไว้ เช่น อ่า เอ่อ แบบว่า นะครับ นะคะ
"""

# รายการคำฟุ่มเฟือย (Filler Words) ที่ข่อยจะนับ
THAI_FILLER_WORDS = [
    "อ่า", "เอ่อ", "แบบว่า", "คือ", "แบบ", "ก็นะ",
    "ที่แบบ", "ใช่ไหม", "ใช่ป่ะ", "นะครับ", "นะคะ", "อะครับ", "อะค่ะ"
]

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Global Model Cache ---
asr_model = None

def get_asr_model():
    """Loads the ASR model into memory if it's not already loaded."""
    global asr_model
    if asr_model is None:
        print(f"กำลังโหลดโมเดล ASR ({ASR_MODEL_NAME}) เป็นครั้งแรก...")
        asr_model = WhisperModel(ASR_MODEL_NAME, device="cpu", compute_type=ASR_COMPUTE_TYPE)
    return asr_model

def convert_to_wav(source_path: str) -> str:
    """
    Converts an audio/video file to a temporary WAV file for analysis.
    Returns the path to the new WAV file.
    """
    print(f"Converting {source_path} to WAV format...")
    try:
        audio = AudioSegment.from_file(source_path)
        # Create a temporary path for the WAV file
        base, _ = os.path.splitext(source_path)
        wav_path = f"{base}_temp.wav"
        # Export as WAV
        audio.export(wav_path, format="wav")
        print(f"Successfully converted to {wav_path}")
        return wav_path
    except Exception as e:
        print(f"Error during audio conversion: {e}")
        # Re-raise the exception to be caught by the background task handler
        raise

def transcribe_audio(file_path, model: WhisperModel):
    """
    ถอดเสียงไฟล์ audio เป็น text โดยใช้ faster-whisper (Quantized)
    """
    print(f"กำลังถอดเสียงไฟล์: {file_path}...")

    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        language="th",
        initial_prompt=PROMPT_FOR_FILLERS
    )

    full_transcript = "".join([seg.text for seg in segments])

    # ใช้ librosa.get_duration เฉพาะกับไฟล์ .wav
    try:
        duration_seconds = librosa.get_duration(path=file_path)
    except Exception as e:
        print(f"Librosa failed to get duration for {file_path}: {e}")
        duration_seconds = info.duration if hasattr(info, "duration") else 0

    print("--- Transcript (ผลดิบจาก Whisper) ---")
    print(full_transcript)
    print("--------------------------------------")

    return full_transcript, duration_seconds

def analyze_presentation(transcript, duration_seconds):
    """
    วิเคราะห์ Transcript โดยใช้ LLM (Groq) เพื่อให้คะแนนและ Feedback
    """
    print("กำลังส่ง Transcript ไปให้ LLM วิเคราะห์...")

    words = transcript.split()
    total_words = len(words)
    duration_minutes = duration_seconds / 60
    words_per_minute = total_words / duration_minutes if duration_minutes > 0 else 0

    # นับคำฟุ่มเฟือย
    filler_counts = {}
    total_fillers = 0
    for filler in THAI_FILLER_WORDS:
        count = transcript.lower().count(filler)
        if count > 0:
            filler_counts[filler] = count
            total_fillers += count

    # 2. เพิ่มวันที
    current_date_str = datetime.now().strftime("%d %B %Y")

    # 3. แก้ไข System Prompt ให้ตรง Format ใหม่
    system_prompt = f"""
คุณคือโค้ชด้านการนำเสนอ (Presentation Coach) ที่เชี่ยวชาญและให้คำแนะนำที่สร้างสรรค์
หน้าที่ของคุณคือวิเคราะห์สคริปต์การนำเสนอและสถิติที่ได้รับ
และให้ผลลัพธ์กลับมาเป็น JSON Object เท่านั้น ห้ามมีข้อความอื่นใดๆ นอก JSON

รูปแบบ JSON ที่คุณต้องส่งกลับมา:
{{
    "score": <float, 0-100>,
    "analysis_date": "<string, วันที่วิเคราะห์, รูปแบบ 'DD Month YYYY'>",
    "strengths": [
    "<string, จุดแข็งที่ 1>",
    "<string, จุดแข็งที่ 2>"
    ],
    "improvements": [
    "<string, ข้อควรปรับปรุงที่ 1>",
    "<string, ข้อควรปรับปรุงที่ 2>"
    ]
}}

เกณฑ์การให้คะแนน (score):
- 100: สมบูรณ์แบบ, ชัดเจน, ไม่มีคำฟุ่มเฟือย, ความเร็วพอดี
- 85-95: ดีมาก, อาจมีคำฟุ่มเฟือยเล็กน้อย
- 70-84: ปานกลาง, พูดเร็ว/ช้าไป, มีคำฟุ่มเฟือย
- < 70: ต้องปรับปรุงด่วน, จับใจความยาก
    """

    user_prompt = f"""
ช่วยวิเคราะห์การนำเสนอครั้งนี้ให้หน่อย

--- สถิติ (สำหรับใช้ประกอบการวิเคราะห์) ---
วันที่ปัจจุบัน: {current_date_str}
ความเร็วในการพูด (WPM): {words_per_minute:.2f} คำต่อนาที
(เกณฑ์ WPM ที่ดีสำหรับภาษาไทย: 130-160 WPM)
จำนวนคำฟุ่มเฟือยทั้งหมด: {total_fillers} คำ
รายละเอียดคำฟุ่มเฟือย: {json.dumps(filler_counts, ensure_ascii=False)}

--- Transcript ---
{transcript}
---

โปรดวิเคราะห์และส่งผลลัพธ์เป็น JSON ตามรูปแบบที่กำหนดเท่านั้น
(ต้องมี keys: "score", "analysis_date", "strengths", "improvements")
    """

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=LLM_MODEL_NAME,
            temperature=0.3,
            response_format={"type": "json_object"}
        )

        response_json_str = chat_completion.choices[0].message.content

        analysis_result = json.loads(response_json_str)

        return analysis_result

    except Exception as e:
        print(f"เรียก LLM ไม่ได้อะ dude : {e}")
        # คืนค่า error ใน format ที่ frontend อาจจะพออ่านได้
        return {
            "error": str(e),
            "score": 0,
            "analysis_date": current_date_str,
            "strengths": [],
            "improvements": ["เกิดข้อผิดพลาดในการวิเคราะห์ LLM"]
        }

def run_analysis_in_background(file_path: str, sanitized_filename: str):
    """
    ฟังก์ชันนี้จะถูกรันใน background เพื่อทำการถอดเสียงและวิเคราะห์
    จากนั้นจะบันทึกผลลัพธ์เป็นไฟล์ .json
    """
    print(f"Background task started for: {sanitized_filename}")
    model = get_asr_model()
    wav_file_path = None
    try:
        # Convert the uploaded file to a processable WAV format first
        wav_file_path = convert_to_wav(file_path)

        transcript, duration = transcribe_audio(wav_file_path, model)
        if transcript:
            analysis_data = analyze_presentation(transcript, duration)
            cache_path = os.path.join(UPLOAD_DIR, f"{sanitized_filename}.json")
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_data, f, ensure_ascii=False, indent=4)
            print(f"Background task finished for: {sanitized_filename}")

    except Exception as e:
        print(f"Error during background analysis for {sanitized_filename}: {e}")
        # If an error occurs, create a JSON file with the error message
        # This helps the frontend know that the process failed.
        error_data = {
            "status": "error",
            "detail": str(e)
        }
        cache_path = os.path.join(UPLOAD_DIR, f"{sanitized_filename}.json")
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(error_data, f, ensure_ascii=False, indent=4)

@app.post("/uploadfile/")
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    รับไฟล์เสียง, บันทึก, ถอดเสียง, วิเคราะห์ และส่งผลลัพธ์ JSON กลับไป
    """
    # Sanitize filename to prevent directory traversal issues and spaces
    sanitized_filename = os.path.basename(file.filename.replace(" ", "_"))
    file_path = os.path.join(UPLOAD_DIR, sanitized_filename)

    # บันทึกไฟล์ที่อัปโหลด
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        print(f"File '{file.filename}' uploaded and saved as '{sanitized_filename}'")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")

    # สั่งให้รัน analysis ใน background
    background_tasks.add_task(run_analysis_in_background, file_path, sanitized_filename)

    # ตอบกลับทันทีว่าเริ่มงานแล้ว
    return JSONResponse(
        content={"message": "Analysis started in background", "filename": sanitized_filename}
    )

@app.get("/status/{filename}")
async def get_analysis_status(filename: str):
    """
    ตรวจสอบสถานะของไฟล์วิเคราะห์
    ถ้าเจอไฟล์ .json แสดงว่าเสร็จแล้ว, ถ้าไม่เจอคือยังประมวลผลอยู่
    """
    cache_file_path = os.path.join(UPLOAD_DIR, f"{filename}.json")

    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        if analysis_data.get("status") == "error":
            return JSONResponse(content=analysis_data)
        else:
            return JSONResponse(content={"status": "complete", "data": analysis_data})
    else:
        return JSONResponse(content={"status": "processing"})

@app.get("/uploadfile/")
async def get_analysis_by_filename(filename: str = Query(..., description="ชื่อไฟล์ที่ต้องการดึงผลการวิเคราะห์")):
    """
    ดึงผลการวิเคราะห์จากไฟล์ที่เคยอัปโหลดและวิเคราะห์ไปแล้ว
    (ในตัวอย่างนี้จะทำการวิเคราะห์ใหม่ทุกครั้งที่เรียก)
    """
    cache_file_path = os.path.join(UPLOAD_DIR, f"{filename}.json")
    audio_file_path = os.path.join(UPLOAD_DIR, filename)

    if os.path.exists(cache_file_path):
        print(f"Found cached analysis for '{filename}'. Reading from cache.")
        with open(cache_file_path, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        return JSONResponse(content=analysis_data)

    if not os.path.exists(audio_file_path):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    raise HTTPException(status_code=425, detail="Analysis is still in progress. Please try again later.")