# !pip install faster-whisper librosa groq

import os
# from google.colab import userdata

os.environ["GROQ_API_KEY"] = userdata.get('GROQ_API_KEY')

import os
import json
import librosa
from groq import Groq
from faster_whisper import WhisperModel
from datetime import datetime # <-- 1. เพิ่ม import นี้

ASR_MODEL_NAME = "large-v3"
ASR_COMPUTE_TYPE = "float16" # "float16" หรือ "int8" เพื่อความเร็ว

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

def transcribe_audio(file_path):
    """
    ถอดเสียงไฟล์ audio เป็น text โดยใช้ faster-whisper (Quantized)
    และใช้ Prompt เพื่อบังคับให้ถอดคำฟุ่มเฟือย
    """
    print(f"กำลังโหลดโมเดล ASR ({ASR_MODEL_NAME})...")

    # TODO: ควบคุมการโหลดโมเดล (อาจจะโหลดครั้งเดียวนอกฟังก์ชัน)
    model = WhisperModel(ASR_MODEL_NAME, device="cuda", compute_type=ASR_COMPUTE_TYPE)

    print(f"กำลังถอดเสียงไฟล์: {file_path}...")
    segments, info = model.transcribe(
        file_path,
        beam_size=5,
        language="th",
        initial_prompt=PROMPT_FOR_FILLERS
    )

    full_transcript = "".join([seg.text for seg in segments])

    # ใช้ librosa.get_duration(path=...) แทน filename=...
    duration_seconds = librosa.get_duration(path=file_path)

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
        client = Groq()
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

def main():
    # เปลี่ยนเป็นชื่อไฟล์ของคุณ
    audio_file = "/content/EV Hack Video.mp4"

    if not os.path.exists(audio_file):
        print(f"!!! [Error เเล้วพรี่] !!!")
        print(f"ไม่เจอไฟล์ '{audio_file}' เช็คดีๆดิ๊")
        return

    transcript, duration = transcribe_audio(audio_file)

    if not transcript:
        print("ถอดเสียงบ่ได้")
        return

    analysis_data = analyze_presentation(transcript, duration)

    if analysis_data:
        print("\n\nผลการวิเคราะห์ (JSON Output ที่แก้ไขแล้ว)")

        # 4. แก้ไข Output JSON ให้เป็นแบบ Flat (ไม่ซ้อน)
        # โดยการรวม filename เข้ากับผลลัพธ์จาก analysis_data

        # ใช้ os.path.basename เพื่อเอาแค่ชื่อไฟล์ ไม่เอา path เต็ม
        final_json_output = {
            "filename": os.path.basename(audio_file)
        }

        # .update() จะเอากุญแจทั้งหมดจาก analysis_data มารวมใน final_json_output
        final_json_output.update(analysis_data)

        # ผลลัพธ์ที่ได้จะมีหน้าตาแบบนี้:
        # {
        #   "filename": "EV Hack Video.mp4",
        #   "score": 85.0,
        #   "analysis_date": "30 October 2025",
        #   "strengths": [...],
        #   "improvements": [...]
        # }

        print(json.dumps(final_json_output, indent=2, ensure_ascii=False))

        with open("analysis_result.json", "w", encoding="utf-8") as f:
            json.dump(final_json_output, f, indent=2, ensure_ascii=False)
        print("\n(บันทึกผลลัพธ์ลงใน 'analysis_result.json' เรียบร้อย)")

if __name__ == "__main__":
    main()