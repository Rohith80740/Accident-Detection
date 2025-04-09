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
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
FRAME_SKIP = 2
ALERT_RADIUS_KM = 100  # Alert hospitals within this radius
MAX_HOSPITALS_TO_ALERT = 3  # Maximum hospitals to notify

# Email Configuration using Streamlit secrets
try:
    EMAIL_SENDER = st.secrets["email"]["sender"]
    EMAIL_PASSWORD = st.secrets["email"]["password"]
    st.sidebar.success("✅ Email credentials loaded from secrets")
except Exception as e:
    st.sidebar.error("❌ Error loading email credentials from secrets")
    logger.error(f"Failed to load email secrets: {str(e)}")
    # Fallback to environment variables if needed
    EMAIL_SENDER = os.environ.get("EMAIL_SENDER", "")
    EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")

# Hospital Database
HOSPITAL_DATABASE = [
    {
        "name": "Eluru Hospital",
        "email": "ragalarohithkumar@gmail.com",
        "phone": "8074013857",
        "location": {
            "latitude": 16.709285,
            "longitude": 81.081122,
            "address": "Eluru, Andhra Pradesh"
        }
    },
    {
        "name": "Asram General & Super Speciality Hospital",
        "email": "adithyakrishna148@gmail.com",
        "phone": "8341636388",
        "location": {
            "latitude": 16.7364531,
            "longitude": 81.1435241,
            "address": "NH 5, Asram Rd, Eluru"
        }
    },
    {
        "name": "Andhra Hospitals Eluru",
        "email": "sowjanyakanagala13@gmail.com",
        "phone": "9338123459",
        "location": {
            "latitude": 16.7123182,
            "longitude": 81.1009435,
            "address": "Eluru, Andhra Pradesh"
        }
    },
    {
        "name": "Aayush Hospitals",
        "email": "vamsisamarla@gmail.com",
        "phone": "7036446608",
        "location": {
            "latitude": 16.7201937,
            "longitude": 81.0921321,
            "address": "Eluru, Andhra Pradesh"
        }
    },
    {
        "name": "Rohit's Hospital",
        "email": "rrkips2003@gmail.com",
        "phone": "8074013857",
        "location": {
            "latitude": 17.4431917,
            "longitude": 78.433787,
            "address": "Hyderabad, Telangana"
        }
    }
]

# Load model with caching
@st.cache_resource
def load_model():
    try:
        # Try to load from relative path first (for deployment)
        model_paths = [
            "Models/best.pt",  # Relative path for deployed app
            "best.pt",         # Direct in current directory
            "./best.pt",       # Explicitly in current directory
        ]
        
        # Add the original path as fallback
        model_paths.append("C:/Users/rishi/Downloads/Accident Detection/Models/best.pt")
        
        for path in model_paths:
            try:
                if os.path.exists(path):
                    model = YOLO(path)
                    st.sidebar.success(f"✅ Model loaded from: {path}")
                    return model
            except Exception as e:
                continue
                
        # If we get here, none of the paths worked
        st.error("🚨 Could not find model file in any location")
        st.stop()
        
    except Exception as e:
        st.error(f"🚨 Model loading failed: {str(e)}")
        st.stop()

model = load_model()

# Context manager for temp files
@contextmanager
def temp_video_file(uploaded_file):
    temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    try:
        temp.write(uploaded_file.read())
        temp.close()
        yield temp.name
    finally:
        try:
            os.unlink(temp.name)
        except:
            pass

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in kilometers"""
    R = 6371.0  # Earth radius in km
    
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def get_current_location():
    """Get current GPS coordinates"""
    try:
        g = geocoder.ip('me')
        if g.ok:
            return {
                'latitude': g.latlng[0],
                'longitude': g.latlng[1],
                'address': g.address
            }
        return None
    except Exception as e:
        st.warning(f"Could not get location: {str(e)}")
        return None

def send_email_alert(subject, message, receiver_email, attachment_path=None, video_path=None):
    """Send email notification with attachments"""
    # Skip email sending if credentials are not available
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        st.warning("⚠️ Email credentials not configured. Skipping email alert.")
        return False
        
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        msg.attach(MIMEText(message, 'plain'))
        
        if attachment_path and os.path.exists(attachment_path):
            with open(attachment_path, 'rb') as f:
                img = MIMEImage(f.read())
                img.add_header('Content-Disposition', 'attachment', 
                             filename=os.path.basename(attachment_path))
                msg.attach(img)
        
        if video_path and os.path.exists(video_path):
            part = MIMEBase('application', 'octet-stream')
            with open(video_path, 'rb') as f:
                part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition',
                           'attachment',
                           filename=os.path.basename(video_path))
            msg.attach(part)
        
        email_debug = st.sidebar.checkbox("Enable Email Debug Logs", False)
        
        with st.expander("Email Sending Log", expanded=email_debug):
            status_log = st.empty()
            
            try:
                status_log.info(f"Connecting to SMTP server...")
                server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
                
                status_log.info(f"Logging in as {EMAIL_SENDER}...")
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                
                status_log.info(f"Sending email to {receiver_email}...")
                server.send_message(msg)
                
                status_log.success(f"✅ Email sent successfully to {receiver_email}")
                server.quit()
                return True
            except Exception as e:
                status_log.error(f"❌ Email sending failed: {str(e)}")
                logger.error(f"Email sending error: {str(e)}")
                return False
        
    except Exception as e:
        st.error(f"Failed to send email to {receiver_email}: {str(e)}")
        logger.error(f"Email preparation error: {str(e)}")
        return False

def send_all_alerts(subject, message, accident_location=None, attachment_path=None, video_path=None):
    """Unified alert function with hospital proximity check"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    full_message = f"{message}\n\nTimestamp: {timestamp}"
    if accident_location:
        if isinstance(accident_location, dict):
            full_message += (f"\nAccident Location: {accident_location.get('address', 'Unknown')}"
                           f"\nCoordinates: {accident_location.get('latitude')}, {accident_location.get('longitude')}"
                           f"\nGoogle Maps: https://maps.google.com/?q={accident_location.get('latitude')},{accident_location.get('longitude')}")
            
            # Find nearby hospitals
            nearby_hospitals = []
            for hospital in HOSPITAL_DATABASE:
                distance = calculate_distance(
                    accident_location['latitude'],
                    accident_location['longitude'],
                    hospital['location']['latitude'],
                    hospital['location']['longitude']
                )
                if distance <= ALERT_RADIUS_KM:
                    nearby_hospitals.append((hospital, distance))
            
            # Sort by distance (nearest first)
            nearby_hospitals.sort(key=lambda x: x[1])
            
            if nearby_hospitals:
                st.success(f"🏥 Found {len(nearby_hospitals)} nearby hospitals within {ALERT_RADIUS_KM} km")
                alert_count = 0
                
                for hospital, distance in nearby_hospitals:
                    if alert_count >= MAX_HOSPITALS_TO_ALERT:
                        break
                        
                    hospital_msg = f"{full_message}\n\nNearest Hospital: {hospital['name']} ({distance:.1f} km)"
                    
                    # Send email to hospital
                    if send_email_alert(
                        subject,
                        hospital_msg,
                        hospital['email'],
                        attachment_path,
                        video_path
                    ):
                        st.info(f"📧 Alert sent to {hospital['name']} ({distance:.1f} km)")
                        alert_count += 1
            else:
                st.warning("⚠️ No hospitals found within alert radius")
        else:
            full_message += f"\nLocation: {accident_location}"
            
            # Send to all hospitals without distance check
            alert_count = 0
            for hospital in HOSPITAL_DATABASE:
                if alert_count >= MAX_HOSPITALS_TO_ALERT:
                    break
                    
                if send_email_alert(
                    subject,
                    full_message,
                    hospital['email'],
                    attachment_path,
                    video_path
                ):
                    st.info(f"📧 Alert sent to {hospital['name']}")
                    alert_count += 1

# Sidebar Configuration
st.sidebar.title("Settings")

# Model Settings
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 0.1, 0.9, 0.3, 0.05
)

frame_skip = st.sidebar.selectbox(
    "Frame Processing Rate",
    options=[1, 2, 3, 4],
    index=1,
    help="Higher values process fewer frames"
)

# Alert Settings
st.sidebar.subheader("Alert Settings")
enable_email = st.sidebar.checkbox("Enable Email Alerts", True)

# Location Settings
st.sidebar.subheader("Location Settings")
location_method = st.sidebar.radio(
    "Location Method",
    ("Automatic (GPS)", "Manual"),
    help="Automatic requires location permissions"
)

manual_location = ""
if location_method == "Manual":
    manual_location = st.sidebar.text_input(
        "Location Description", 
        "Enter accident location"
    )

# Main Application UI
st.title("🚨 Accident Detection & Alert System")
st.markdown("""
    <style>
    .object-info {
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

def get_object_details(boxes):
    """Extract detailed information about detected objects"""
    details = []
    for box in boxes:
        obj = {
            "type": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "coordinates": {
                "x1": float(box.xyxy[0][0]),
                "y1": float(box.xyxy[0][1]),
                "x2": float(box.xyxy[0][2]),
                "y2": float(box.xyxy[0][3])
            }
        }
        details.append(obj)
    return details

def process_image(uploaded_file):
    """Process image with detailed object information"""
    try:
        image = Image.open(uploaded_file)
        
        with st.spinner("🔍 Analyzing image..."):
            results = model.predict(image, conf=confidence_threshold)
            
            col1, col2 = st.columns(2)
            with col1:
                st.image(image, caption="Original Image", use_container_width=True)
            
            with col2:
                if len(results[0].boxes) > 0:
                    st.warning("🚨 ACCIDENT DETECTED")
                    result_img = Image.fromarray(results[0].plot()[:, :, ::-1])
                    st.image(result_img, caption="Processed Result", use_container_width=True)
                    
                    st.subheader("Detected Objects:")
                    objects = get_object_details(results[0].boxes)
                    for obj in objects:
                        st.markdown(f"""
                            <div class="object-info">
                                <b>Type:</b> {obj['type']}<br>
                                <b>Confidence:</b> {obj['confidence']:.2f}<br>
                                <b>Bounding Box:</b> ({obj['coordinates']['x1']:.0f}, {obj['coordinates']['y1']:.0f}) to ({obj['coordinates']['x2']:.0f}, {obj['coordinates']['y2']:.0f})
                            </div>
                        """, unsafe_allow_html=True)
                    
                    location = get_current_location() if location_method == "Automatic (GPS)" else manual_location
                    
                    if enable_email:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        alert_msg = f"Accident detected in image at {timestamp}"
                        
                        result_path = "accident_detected.jpg"
                        result_img.save(result_path)
                        
                        send_all_alerts(
                            "🚨 Accident Detected",
                            alert_msg,
                            accident_location=location,
                            attachment_path=result_path
                        )
                        
                        try:
                            os.unlink(result_path)
                        except:
                            pass
                else:
                    st.success("✅ NO ACCIDENT DETECTED")
                    st.image(image, caption="No detections", use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        logger.error(f"Image processing error: {str(e)}")

def process_video(uploaded_file):
    """Process video with frame-by-frame object information"""
    try:
        if uploaded_file.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            st.error(f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB")
            return
        
        st.video(uploaded_file)
        
        if st.button("Process Video with Details", type="primary"):
            with temp_video_file(uploaded_file) as input_path:
                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    st.error("Error opening video file")
                    return
                
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                output_path = "processed_video.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                result_placeholder = st.empty()
                details_placeholder = st.empty()
                
                frame_count = 0
                processed_count = 0
                accident_frames = []
                
                try:
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        frame_count += 1
                        
                        if frame_count % frame_skip != 0:
                            continue
                            
                        results = model.predict(frame, conf=confidence_threshold)
                        annotated_frame = results[0].plot()
                        out.write(annotated_frame)
                        processed_count += 1
                        
                        if len(results[0].boxes) > 0:
                            objects = get_object_details(results[0].boxes)
                            accident_frames.append({
                                "frame_number": frame_count,
                                "time_seconds": frame_count/fps,
                                "objects": objects
                            })
                        
                        progress = frame_count / total_frames
                        progress_bar.progress(progress)
                        status_text.text(f"📊 Processing frame {frame_count}/{total_frames}")
                    
                    cap.release()
                    out.release()
                    
                    if accident_frames:
                        result_placeholder.warning(f"🚨 ACCIDENTS DETECTED IN {len(accident_frames)} FRAMES")
                        
                        with st.expander("📝 Detailed Accident Report", expanded=True):
                            st.write(f"Total frames with accidents: {len(accident_frames)}")
                            
                            for accident in accident_frames:
                                st.markdown(f"""
                                    ### Frame {accident['frame_number']} (Time: {accident['time_seconds']:.1f}s)
                                """)
                                
                                for obj in accident['objects']:
                                    st.markdown(f"""
                                        <div class="object-info">
                                            <b>Object Type:</b> {obj['type']}<br>
                                            <b>Confidence:</b> {obj['confidence']:.2f}<br>
                                            <b>Bounding Box:</b> ({obj['coordinates']['x1']:.0f}, {obj['coordinates']['y1']:.0f}) to ({obj['coordinates']['x2']:.0f}, {obj['coordinates']['y2']:.0f})
                                        </div>
                                    """, unsafe_allow_html=True)
                                st.markdown("---")
                        
                        location = get_current_location() if location_method == "Automatic (GPS)" else manual_location
                        
                        if enable_email:
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            time_list = [f"{a['time_seconds']:.1f}s" for a in accident_frames]
                            alert_msg = (
                                f"Accidents detected in video at {timestamp}\n"
                                f"Occurred at: {', '.join(time_list)}"
                            )
                            
                            send_all_alerts(
                                "🚨 Multiple Accidents Detected",
                                alert_msg,
                                accident_location=location,
                                video_path=output_path
                            )
                    else:
                        result_placeholder.success("✅ NO ACCIDENTS DETECTED IN VIDEO")
                    
                    st.video(output_path)
                    
                    with open(output_path, "rb") as f:
                        st.download_button(
                            label="Download Processed Video",
                            data=f,
                            file_name="processed_video.mp4",
                            mime="video/mp4"
                        )
                
                finally:
                    if 'cap' in locals() and cap.isOpened():
                        cap.release()
                    if 'out' in locals():
                        out.release()
                    try:
                        os.unlink(output_path)
                    except:
                        pass
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")
        logger.error(f"Video processing error: {str(e)}")

# Main Application Flow
st.write("""
## 📋 Application Overview
This system detects accidents in images and videos using YOLOv8 and sends automatic alerts to nearby hospitals.
""")

# Model information
with st.expander("ℹ️ About the Model"):
    st.write("""
    This application uses YOLOv8, a state-of-the-art object detection model trained specifically for accident detection.
    The model can identify various types of accidents in real-time with high accuracy.
    """)

option = st.radio(
    "Select input type:",
    ("Image", "Video"),
    horizontal=True
)

if option == "Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=SUPPORTED_IMAGE_TYPES,
        help="Supported formats: JPG, JPEG, PNG"
    )
    if uploaded_file is not None:
        process_image(uploaded_file)
else:
    uploaded_file = st.file_uploader(
        "Upload a video",
        type=SUPPORTED_VIDEO_TYPES,
        help=f"Supported formats: MP4, AVI, MOV (Max {MAX_VIDEO_SIZE_MB}MB)"
    )
    if uploaded_file is not None:
        process_video(uploaded_file)

# Add troubleshooting section
with st.expander("🔧 Troubleshooting"):
    st.write("""
    ### Common Issues
    
    #### Email Alerts Not Working
    - Check that email secrets are properly configured
    - Verify the app has permission to send emails
    - If on Streamlit Cloud, make sure secrets are added in the dashboard
    
    #### Model Not Loading
    - Check that the model file is included in your deployment
    - The model file should be in a 'models' folder in your repository
    
    #### Location Detection Issues
    - Try using manual location if automatic detection fails
    - Some browsers or environments may block location services
    """)

# Footer
st.markdown("---")
st.markdown("Accident Detection Alert System v1.0 | Created for safety and quick emergency response")
