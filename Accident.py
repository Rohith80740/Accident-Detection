import streamlit as st
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
FRAME_SKIP = 2
ALERT_RADIUS_KM = 15  # Alert hospitals within this radius
MAX_HOSPITALS_TO_ALERT = 3  # Maximum hospitals to notify

# Email Configuration
EMAIL_SENDER = "accidentalertsystem1122@gmail.com"
EMAIL_PASSWORD = "wvczuqhhhtlhwbmk"  # App-specific password
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
        "name": "Govt Hospital",
        "email": "nithin9231@gmail.com",
        "phone": "9999999999",
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
        model = YOLO("Models/best.pt")
        return model
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
    """
    Send email alert with optional attachments (image or video)
    
    Args:
        subject (str): Email subject line
        message (str): Email body text
        receiver_email (str): Recipient email address
        attachment_path (str, optional): Path to image attachment
        video_path (str, optional): Path to video attachment
        
    Returns:
        bool: True if email sent successfully, False otherwise
    """
    try:
        # Validate email parameters
        if not all([subject, message, receiver_email]):
            st.error("Missing required email parameters")
            return False
            
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        # Attach message body
        msg.attach(MIMEText(message, 'plain'))
        
        # Attach image if provided
        if attachment_path and os.path.exists(attachment_path):
            try:
                with open(attachment_path, 'rb') as f:
                    img = MIMEImage(f.read())
                    img.add_header('Content-Disposition', 'attachment', 
                                 filename=os.path.basename(attachment_path))
                    msg.attach(img)
            except Exception as e:
                st.warning(f"Could not attach image: {str(e)}")
        
        # Attach video if provided
        if video_path and os.path.exists(video_path):
            try:
                part = MIMEBase('application', 'octet-stream')
                with open(video_path, 'rb') as f:
                    part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header('Content-Disposition',
                               'attachment',
                               filename=os.path.basename(video_path))
                msg.attach(part)
            except Exception as e:
                st.warning(f"Could not attach video: {str(e)}")
        
        # Send email with error handling
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
                server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                server.send_message(msg)
            st.success(f"Email successfully sent to {receiver_email}")
            return True
            
        except smtplib.SMTPAuthenticationError:
            st.error("Authentication failed. Please check your email credentials.")
            return False
            
        except smtplib.SMTPException as e:
            error_code = e.smtp_code if hasattr(e, 'smtp_code') else 'N/A'
            error_message = e.smtp_error.decode() if hasattr(e, 'smtp_error') else str(e)
            st.error(f"SMTP Error {error_code}: {error_message}")
            return False
            
        except Exception as e:
            st.error(f"Unexpected error sending email: {str(e)}")
            return False
            
    except Exception as e:
        st.error(f"Failed to prepare email: {str(e)}")
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
            full_message += f"\nLocation: {accident_location}"

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
                        
                        os.unlink(result_path)
                else:
                    st.success("✅ NO ACCIDENT DETECTED")
                    st.image(image, caption="No detections", use_container_width=True)
    
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")

def process_video(uploaded_file):
    """Process video with frame-by-frame object information using Streamlit"""
    try:
        if uploaded_file.size > MAX_VIDEO_SIZE_MB * 1024 * 1024:
            st.error(f"File too large. Maximum size is {MAX_VIDEO_SIZE_MB}MB")
            return
        
        # Display original video
        st.video(uploaded_file)
        
        if st.button("Process Video with Details", type="primary"):
            with temp_video_file(uploaded_file) as input_path:
                # Create a temporary directory for processed frames
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Process video using YOLO directly
                    results = model.predict(
                        source=input_path,
                        conf=confidence_threshold,
                        save=True,
                        project=temp_dir,
                        name="processed",
                        exist_ok=True
                    )
                    
                    # Find the processed video file
                    processed_video_path = os.path.join(temp_dir, "processed", os.path.basename(input_path))
                    
                    if os.path.exists(processed_video_path):
                        # Display processed video
                        st.video(processed_video_path)
                        
                        # Check for accidents
                        accident_frames = []
                        for i, result in enumerate(results):
                            if len(result.boxes) > 0:
                                accident_frames.append({
                                    "frame_number": i,
                                    "time_seconds": i / 30,  # Assuming 30 FPS
                                    "objects": get_object_details(result.boxes)
                                })
                        
                        if accident_frames:
                            st.warning(f"🚨 ACCIDENTS DETECTED IN {len(accident_frames)} FRAMES")
                            
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
                                    video_path=processed_video_path
                                )
                            
                            # Download button for processed video
                            with open(processed_video_path, "rb") as f:
                                st.download_button(
                                    label="Download Processed Video",
                                    data=f,
                                    file_name="processed_video.mp4",
                                    mime="video/mp4"
                                )
                        else:
                            st.success("✅ NO ACCIDENTS DETECTED IN VIDEO")
                    else:
                        st.error("Failed to process video")
    
    except Exception as e:
        st.error(f"Error processing video: {str(e)}")

# Main Application Flow
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
        
