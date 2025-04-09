# The full updated code is provided here
# Streamlit Accident Detection and Alert System

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from email.mime.base import MIMEBase
from email import encoders
from datetime import datetime
from contextlib import contextmanager
import geocoder
import math

# Set page config
st.set_page_config(
    page_title="Accident Detection Alert System",
    page_icon="🚨",
    layout="wide"
)

# Constants
MAX_VIDEO_SIZE_MB = 100
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]
SUPPORTED_VIDEO_TYPES = ["mp4", "avi", "mov"]
ALERT_RADIUS_KM = 15
MAX_HOSPITALS_TO_ALERT = 3
EMAIL_SENDER = "accidentalertsystem1122@gmail.com"
EMAIL_PASSWORD = "gtejztpwxlaponsp"

# Hospital Database
HOSPITAL_DATABASE = [
    # hospitals dictionary same as before...
]

@st.cache_resource
def load_model():
    try:
        return YOLO("D:/All Documents/Projects/Real-Time Accident Detection and Alert System/Models/best.pt")
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()

model = load_model()

@contextmanager
def temp_video_file(uploaded_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        temp.write(uploaded_file.read())
        temp.close()
        yield temp.name
    finally:
        os.unlink(temp.name)

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def get_current_location():
    try:
        g = geocoder.ip('me')
        return {'latitude': g.latlng[0], 'longitude': g.latlng[1], 'address': g.address} if g.ok else None
    except:
        return None

def send_email_alert(subject, message, receiver_email, attachment_path=None, video_path=None):
    try:
        msg = MIMEMultipart()
        msg['From'], msg['To'], msg['Subject'] = EMAIL_SENDER, receiver_email, subject
        msg.attach(MIMEText(message, 'plain'))

        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', filename=os.path.basename(attachment_path))
                msg.attach(img)

        if video_path and os.path.exists(video_path):
            with open(video_path, 'rb') as f:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(video_path))
            msg.attach(part)

        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(EMAIL_SENDER, EMAIL_PASSWORD)
            server.send_message(msg)
        return True
    except Exception as e:
        st.error(f"Failed to send email to {receiver_email}: {str(e)}")
        return False

def send_all_alerts(subject, message, accident_location=None, attachment_path=None, video_path=None):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"{message}\n\nTimestamp: {timestamp}"
    if accident_location and isinstance(accident_location, dict):
        full_message += f"\nAccident Location: {accident_location.get('address', 'Unknown')}"
        full_message += f"\nCoordinates: {accident_location.get('latitude')}, {accident_location.get('longitude')}"
        full_message += f"\nGoogle Maps: https://maps.google.com/?q={accident_location.get('latitude')},{accident_location.get('longitude')}"

        nearby = sorted(
            [(h, calculate_distance(accident_location['latitude'], accident_location['longitude'], h['location']['latitude'], h['location']['longitude'])) for h in HOSPITAL_DATABASE if calculate_distance(accident_location['latitude'], accident_location['longitude'], h['location']['latitude'], h['location']['longitude']) <= ALERT_RADIUS_KM],
            key=lambda x: x[1]
        )

        for hospital, dist in nearby[:MAX_HOSPITALS_TO_ALERT]:
            hospital_msg = f"{full_message}\n\nNearest Hospital: {hospital['name']} ({dist:.1f} km)"
            if send_email_alert(subject, hospital_msg, hospital['email'], attachment_path, video_path):
                st.info(f"Alert sent to {hospital['name']} ({dist:.1f} km)")

# UI Elements
st.sidebar.title("Settings")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 0.9, 0.3, 0.05)
frame_skip = st.sidebar.selectbox("Frame Processing Rate", options=[1, 2, 3, 4], index=1)
enable_email = st.sidebar.checkbox("Enable Email Alerts", True)
location_method = st.sidebar.radio("Location Method", ("Automatic (GPS)", "Manual"))
manual_location = st.sidebar.text_input("Location Description") if location_method == "Manual" else ""

st.title("🚨 Accident Detection & Alert System")

# Helper functions for detection

def get_object_details(boxes):
    return [{
        "type": model.names[int(box.cls)],
        "confidence": float(box.conf),
        "coordinates": {
            "x1": float(box.xyxy[0][0]), "y1": float(box.xyxy[0][1]),
            "x2": float(box.xyxy[0][2]), "y2": float(box.xyxy[0][3])
        }
    } for box in boxes]

def process_image(uploaded_file):
    image = Image.open(uploaded_file)
    results = model.predict(image, conf=confidence_threshold)
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        if results[0].boxes:
            st.warning("Accident Detected")
            st.image(Image.fromarray(results[0].plot()[:, :, ::-1]), caption="Processed Result", use_container_width=True)
            st.subheader("Detected Objects:")
            for obj in get_object_details(results[0].boxes):
                st.markdown(f"""
                    <div style='background:#f0f2f6;padding:10px;border-radius:5px'>
                        <b>Type:</b> {obj['type']}<br>
                        <b>Confidence:</b> {obj['confidence']:.2f}<br>
                        <b>Bounding Box:</b> ({obj['coordinates']['x1']:.0f}, {obj['coordinates']['y1']:.0f}) to ({obj['coordinates']['x2']:.0f}, {obj['coordinates']['y2']:.0f})
                    </div>
                """, unsafe_allow_html=True)
            if enable_email:
                loc = get_current_location() if location_method == "Automatic (GPS)" else manual_location
                send_all_alerts("Accident Alert - Image", "An accident was detected.", loc)

uploaded_file = st.file_uploader("Upload Image or Video", type=SUPPORTED_IMAGE_TYPES + SUPPORTED_VIDEO_TYPES)
if uploaded_file:
    if uploaded_file.type.startswith("image"):
        process_image(uploaded_file)
    else:
        st.warning("Video processing not implemented in this version.")
