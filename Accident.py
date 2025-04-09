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
FRAME_SKIP = 2
ALERT_RADIUS_KM = 500  # Alert hospitals within this radius
MAX_HOSPITALS_TO_ALERT = 3  # Maximum hospitals to notify

# Email Configuration
EMAIL_SENDER = "accidentalertsystem1122@gmail.com"
EMAIL_PASSWORD = "wvczuqhhhtlhwbmk"   # App-specific password

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
    """Send email notification with attachments using multiple SMTP methods"""
    try:
        # Create message container
        msg = MIMEMultipart()
        msg['From'] = EMAIL_SENDER
        msg['To'] = receiver_email
        msg['Subject'] = subject
        
        # Attach the message body
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
                st.error(f"Failed to attach image: {str(e)}")
        
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
                st.error(f"Failed to attach video: {str(e)}")
        
        # Try multiple ports and connection methods
        ports_to_try = [465, 587]  # SSL and TLS ports
        last_exception = None
        
        for port in ports_to_try:
            try:
                if port == 465:
                    # Try SSL connection
                    with smtplib.SMTP_SSL('smtp.gmail.com', port, timeout=10) as server:
                        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                        server.send_message(msg)
                        st.success(f"Email sent successfully via SSL (port {port})")
                        return True
                elif port == 587:
                    # Try STARTTLS connection
                    with smtplib.SMTP('smtp.gmail.com', port, timeout=10) as server:
                        server.ehlo()
                        server.starttls()
                        server.ehlo()
                        server.login(EMAIL_SENDER, EMAIL_PASSWORD)
                        server.send_message(msg)
                        st.success(f"Email sent successfully via STARTTLS (port {port})")
                        return True
            except smtplib.SMTPAuthenticationError as e:
                st.error(f"Authentication failed on port {port}: {str(e)}")
                last_exception = e
                break  # No point trying other ports if auth is failing
            except smtplib.SMTPException as e:
                last_exception = e
                st.warning(f"Failed to send via port {port}: {str(e)}")
                continue
            except Exception as e:
                last_exception = e
                st.warning(f"Unexpected error with port {port}: {str(e)}")
                continue
        
        # If we get here, all attempts failed
        if last_exception:
            st.error(f"All email sending attempts failed. Last error: {str(last_exception)}")
        else:
            st.error("Email sending failed with unknown error")
        
        return False
        
    except Exception as e:
        st.error(f"Email sending failed completely: {str(e)}")
        return False
        
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
