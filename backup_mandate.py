from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import json
import re
import os
import base64
from google.cloud import vision
from google.oauth2 import service_account

# ✅ FastAPI Initialization
app = FastAPI()

# ✅ Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ✅ Google Vision API Initialization
encoded_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

if encoded_credentials:
    credentials_info = json.loads(base64.b64decode(encoded_credentials).decode("utf-8"))
    credentials = service_account.Credentials.from_service_account_info(credentials_info)
    vision_client = vision.ImageAnnotatorClient(credentials=credentials)
else:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS_B64 environment variable is not set")

# ✅ Updated ROIs
rois = {
    "UMRN_Number": (480, 70, 1455, 132),
    "Date1": (1515, 70, 1955, 132),
    "Sponsor_bank_Code": (480, 132, 1045, 190),
    "Utility_Code": (1195, 132, 1955, 190),
    "I/We_Hereby_Authorize": (593, 192, 880, 230),
    "Bank_A/C_Number": (460, 230, 1955, 285),
    "With_Bank": (205, 293, 870, 339),
    "IFSC_Code": (940, 285, 1460, 340),
    "MICR_Code": (1570, 285, 1955, 340),
    "Amount_in_Words": (335, 348, 1503, 403),
    "Amount_in_Digits": (1560, 348, 1945, 403),
    "Reference_1": (240, 460, 1070, 510),
    "Phone_No": (1230, 450, 1800, 507),
    "Reference_2": (240, 510, 1070, 550),
    "Email_ID": (1220, 510, 1945, 550),
    "Date_From": (150, 610, 520, 670),
    "Date_To": (150, 680, 520, 740),
    "Signature_1": (550, 600, 1000, 745),
    "Name_1": (580, 755, 1005, 815),
    "Signature_2": (1020, 600, 1470, 745),
    "Name_2": (1055, 755, 1475, 815),
    "Signature_3": (1490, 600, 1940, 745),
    "Name_3": (1533, 755, 1945, 815),
}

# ✅ Post-Processing Functions
def format_date(date_str):
    date_str = re.sub(r'\D', '', date_str)
    date_str = date_str[:8].rjust(8, '0')
    if len(date_str) == 8:
        return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
    return date_str

# ✅ OCR and Signature Extraction
async def process_image(file_bytes):
    np_arr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    extracted_data = {}
    signature_info = []

    for label, (x1, y1, x2, y2) in rois.items():
        roi = image[y1:y2, x1:x2]
        _, encoded_image = cv2.imencode(".jpg", roi)
        vision_image = vision.Image(content=encoded_image.tobytes())
        response = vision_client.text_detection(image=vision_image)
        texts = response.text_annotations

        extracted_text = texts[0].description.strip() if texts else ""

        if "Date" in label:
            extracted_data[label] = format_date(extracted_text)
        elif "Signature" in label:
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
            signature_present = cv2.countNonZero(binary_roi) > 500

            if signature_present:
                signature_path = f"{label}.jpg"
                cv2.imwrite(signature_path, roi)
                signature_info.append(signature_path)
        else:
            extracted_data[label] = extracted_text

    return extracted_data, signature_info

# ✅ API Endpoint
@app.post("/extract")
async def extract_data(file: UploadFile = File(...)):
    file_bytes = await file.read()
    extracted_data, signatures = await process_image(file_bytes)

    return JSONResponse({
        "Extracted_Text": extracted_data,
        "Signatures": signatures
    })

@app.get("/download-signature/{filename}")
async def download_signature(filename: str):
    file_path = f"./{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return JSONResponse({"error": "File not found"}, status_code=404)
