from fastapi import FastAPI, UploadFile, File
import os
import base64
from fastapi.middleware.cors import CORSMiddleware

# Get base64-encoded credentials from the environment variable
encoded_credentials = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_B64")

if encoded_credentials:
    # Decode and save as a JSON file
    with open("client_file_mandateocr1.json", "wb") as f:
        f.write(base64.b64decode(encoded_credentials))

    # Set the path to GOOGLE_APPLICATION_CREDENTIALS
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "client_file_mandateocr1.json"
else:
    raise EnvironmentError("GOOGLE_APPLICATION_CREDENTIALS_B64 environment variable is not set")

from google.cloud import vision
import cv2
import json
import numpy as np
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import shutil

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve static files for images
app.mount("/static", StaticFiles(directory="static"), name="static")

# Define ROIs with coordinates: (x1, y1, x2, y2)
rois = {
    "Account_Number": (35, 100, 380, 130),
    "Account_Name": (430, 60, 980, 130),
    "Check_Number": (1390, 70, 1650, 100),
    "BRSTN_Number": (1750, 60, 1880, 130),
    "Date": {
        "M1": (1585, 165, 1625, 220),
        "M2": (1625, 165, 1665, 220),
        "D1": (1665, 165, 1705, 220),
        "D2": (1705, 165, 1745, 220),
        "Y1": (1745, 165, 1785, 220),
        "Y2": (1785, 165, 1825, 220),
        "Y3": (1825, 165, 1865, 220),
        "Y4": (1865, 165, 1905, 220),
    },
    "Payee_Name": (310, 280, 1350, 380),
    "Amount_In_Digits": (1440, 280, 1905, 380),
    "Amount_In_Words": (200, 390, 1905, 470),
    "Signature_1": (1500, 550, 1905, 690),
    "Signature_2": (1030, 550, 1440, 690),
    "MICR_Code": (470, 790, 725, 850),
    "Bank_Name": (130, 510, 400, 570),
    "Branch_Name": (40, 595, 400, 635),
}

# Initialize Google Vision client
vision_client = vision.ImageAnnotatorClient()

# Utility Functions
def format_date(date_str):
    if len(date_str) == 8:
        return f"{date_str[:2]}/{date_str[2:4]}/{date_str[4:]}"
    return date_str

def clean_text(text):
    return text.replace("\n", " ").strip()

def remove_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
    return cv2.bitwise_and(image, image, mask=binary)

def extract_text_with_google_vision(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")

    extracted_data = {}
    signature_info = []

    for label, coordinates in rois.items():
        if isinstance(coordinates, dict):
            date_parts = []
            for sub_label, (x1, y1, x2, y2) in coordinates.items():
                roi = image[y1:y2, x1:x2]
                _, encoded_image = cv2.imencode(".jpg", roi)
                vision_image = vision.Image(content=encoded_image.tobytes())
                response = vision_client.text_detection(image=vision_image)
                texts = response.text_annotations
                date_parts.append(texts[0].description.strip() if texts else "")
            extracted_data[label] = format_date("".join(date_parts))

        elif "Signature" in label:
            x1, y1, x2, y2 = coordinates
            roi = image[y1:y2, x1:x2]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, binary_roi = cv2.threshold(gray_roi, 128, 255, cv2.THRESH_BINARY_INV)
            signature_present = cv2.countNonZero(binary_roi) > 500

            if signature_present:
                processed_signature = remove_background(roi)
                signature_path = f"static/{label}.jpg"
                cv2.imwrite(signature_path, processed_signature)
                signature_info.append({
                    "signature_label": label,
                    "status": "Present",
                    "coordinates": (x1, y1, x2, y2),
                    "cropped_image": f"/static/{label}.jpg",
                })
            else:
                signature_info.append({
                    "signature_label": label,
                    "status": "Not Present",
                    "coordinates": (x1, y1, x2, y2),
                })
        else:
            x1, y1, x2, y2 = coordinates
            roi = image[y1:y2, x1:x2]
            _, encoded_image = cv2.imencode(".jpg", roi)
            vision_image = vision.Image(content=encoded_image.tobytes())
            response = vision_client.text_detection(image=vision_image)
            texts = response.text_annotations
            extracted_data[label] = clean_text(texts[0].description if texts else "")

    return {"Extracted_Text": extracted_data, "Signatures": signature_info}

@app.post("/process-cheque")
async def process_cheque(file: UploadFile = File(...)):
    try:
        file_location = f"temp/{file.filename}"
        os.makedirs("temp", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        result = extract_text_with_google_vision(file_location)
        os.remove(file_location)

        return JSONResponse(content=result)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
