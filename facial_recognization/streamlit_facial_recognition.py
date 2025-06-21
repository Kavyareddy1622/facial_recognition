import streamlit as st
import numpy as np
import cv2
import os
import time
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Facial Recognition System",
    layout="wide"
)

# Title and description
st.title("Facial Recognition System")
st.markdown("A simple and easy-to-use facial recognition system")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["Home", "Add Person", "Train Model", "Face Recognition", "Test Camera"]
)

def create_folders():
    """Create necessary folders"""
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    if not os.path.exists('trainer'):
        os.makedirs('trainer')

def add_person_streamlit():
    """Add a new person using Streamlit interface"""
    st.header("Add New Person")
    
    # Create folders
    create_folders()
    
    # Get person's name
    name = st.text_input("Enter person's name:")
    
    if name:
        st.write(f"Ready to capture photos of: **{name}**")
        
        # Instructions
        st.info("""
        **Instructions:**
        - Look at your camera
        - Stay still and well-lit
        - The system will capture 30 photos automatically
        - You can stop early by clicking the stop button
        """)
        
        # Camera capture
        if st.button("Start Camera"):
            st.write("Starting camera...")
            
            # Create a placeholder for the camera feed
            camera_placeholder = st.empty()
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Initialize camera
            cap = cv2.VideoCapture(0)
            face_detector = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
            photo_count = 0
            
            # Create stop button outside the loop
            stop_col1, stop_col2 = st.columns([1, 3])
            with stop_col1:
                stop_button = st.button("Stop Capture")
            
            while photo_count < 30 and not stop_button:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_detector.detectMultiScale(gray_frame, 1.3, 5)
                
                # Draw instructions on frame
                cv2.putText(frame, f"Capturing: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Photos: {photo_count}/30", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if len(faces) == 0:
                    cv2.putText(frame, "No face detected", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    for (x, y, w, h) in faces[:1]:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        photo_count += 1
                        
                        # Save photo
                        photo_name = f"datasets/User.{photo_count}.{name}.jpg"
                        cv2.imwrite(photo_name, gray_frame[y:y+h, x:x+w])
                        
                        cv2.putText(frame, f"Captured {photo_count}/30", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Convert frame to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update progress
                progress = photo_count / 30
                progress_bar.progress(progress)
                status_text.text(f"Captured {photo_count}/30 photos")
                
                time.sleep(0.1)
                
                # Check for stop button
                if stop_button:
                    break
            
            cap.release()
            st.success(f"Captured {photo_count} photos of {name}")

def train_model_streamlit():
    """Train the model using Streamlit interface"""
    st.header("Train the Model")
    
    # Check if datasets exist
    if not os.path.exists('datasets') or len(os.listdir('datasets')) == 0:
        st.error("No photos found! Please add people first.")
        return
    
    dataset_count = len(os.listdir('datasets'))
    st.info(f"Found {dataset_count} photos to train with")
    
    if st.button("Start Training"):
        with st.spinner("Training the model... Please wait..."):
            try:
                # Create recognizer
                recognizer = cv2.face.LBPHFaceRecognizer_create()
                detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
                
                # Get images and labels
                def get_images_and_labels():
                    image_paths = [os.path.join('datasets', f) for f in os.listdir('datasets')]
                    face_samples = []
                    ids = []
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, image_path in enumerate(image_paths):
                        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                        if image is None:
                            continue
                        
                        img_numpy = np.array(image, 'uint8')
                        person_id = int(os.path.split(image_path)[-1].split(".")[1])
                        faces = detector.detectMultiScale(img_numpy)
                        
                        for (x, y, w, h) in faces:
                            face_samples.append(img_numpy[y:y+h, x:x+w])
                            ids.append(person_id)
                        
                        # Update progress
                        progress = (i + 1) / len(image_paths)
                        progress_bar.progress(progress)
                        status_text.text(f"Processing image {i+1}/{len(image_paths)}")
                    
                    return face_samples, ids
                
                faces, ids = get_images_and_labels()
                
                if len(faces) == 0:
                    st.error("No valid face samples found!")
                    return
                
                # Train the model
                recognizer.train(faces, np.array(ids))
                
                # Save the model
                recognizer.write('trainer/trainer.yml')
                
                unique_faces = len(np.unique(ids))
                st.success(f"Training completed!")
                st.info(f"Trained on {len(faces)} face samples")
                st.info(f"Recognized {unique_faces} different people")
                
            except Exception as e:
                st.error(f"Error during training: {e}")

def face_recognition_streamlit():
    """Face recognition using Streamlit interface"""
    st.header("Face Recognition")
    
    # Check if trained model exists
    if not os.path.exists('trainer/trainer.yml'):
        st.error("No trained model found! Please train the model first.")
        return
    
    st.info("""
    **Instructions:**
    - Look at your camera
    - The system will show your name and confidence
    - Press 'Stop' to end recognition
    """)
    
    if st.button("Start Recognition"):
        st.write("Starting face recognition...")
        
        # Load model
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read('trainer/trainer.yml')
        face_detector = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
        
        # Camera placeholder
        camera_placeholder = st.empty()
        info_placeholder = st.empty()
        
        # Start camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Camera not found!")
            return
        
        cap.set(3, 640)
        cap.set(4, 480)
        
        # Create stop button outside the loop
        stop_col1, stop_col2 = st.columns([1, 3])
        with stop_col1:
            stop_button = st.button("Stop Recognition")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                continue
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.2, 5)
            
            # Add title to frame
            cv2.putText(frame, "Face Recognition", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            recognition_info = []
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                try:
                    person_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 100:
                        name = get_person_name(person_id)
                        confidence_text = f"{round(100 - confidence)}%"
                        color = (0, 255, 0)
                    else:
                        name = "Unknown"
                        confidence_text = f"{round(100 - confidence)}%"
                        color = (0, 0, 255)
                    
                    cv2.putText(frame, f"Name: {name}", (x+5, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frame, f"Confidence: {confidence_text}", (x+5, y+h+25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    recognition_info.append(f"**{name}**: {confidence_text} confidence")
                    
                except Exception as e:
                    cv2.putText(frame, "Error", (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            # Show recognition info
            if recognition_info:
                info_placeholder.markdown("**Recognition Results:**")
                for info in recognition_info:
                    info_placeholder.markdown(f"- {info}")
            else:
                info_placeholder.markdown("No faces detected")
            
            time.sleep(0.1)
            
            # Check for stop button
            if stop_button:
                break
        
        cap.release()
        st.success("Recognition stopped")

def get_person_name(person_id):
    """Get person's name from their ID"""
    if os.path.exists('datasets'):
        for filename in os.listdir('datasets'):
            if filename.startswith(f'User.{person_id}.'):
                parts = filename.split('.')
                if len(parts) >= 3:
                    return parts[2]
    return f"Person_{person_id}"

def test_camera_streamlit():
    """Test camera using Streamlit interface"""
    st.header("Test Camera")
    
    st.info("This will test if your camera is working properly")
    
    if st.button("Start Camera Test"):
        st.write("Testing camera...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Camera not found or not accessible!")
            return
        
        st.success("Camera opened successfully!")
        
        # Camera placeholder
        camera_placeholder = st.empty()
        
        # Create stop button outside the loop
        stop_col1, stop_col2 = st.columns([1, 3])
        with stop_col1:
            stop_button = st.button("Stop Camera Test")
        
        while not stop_button:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame!")
                break
            
            # Add text to frame
            cv2.putText(frame, "Camera Test", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert frame to RGB for Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
            
            time.sleep(0.1)
            
            # Check for stop button
            if stop_button:
                break
        
        cap.release()
        st.success("Camera test completed")

def home_page():
    """Home page with instructions"""
    st.header("Welcome to Facial Recognition System")
    
    st.markdown("""
    This is a simple and beginner-friendly facial recognition system.
    
    ### How to use:
    
    1. **Add Person**: Take photos of people you want to recognize
    2. **Train Model**: Teach the system to recognize the faces
    3. **Face Recognition**: Start recognizing faces in real-time
    4. **Test Camera**: Check if your camera is working
    
    ### Requirements:
    - A working webcam
    - Good lighting
    - Clear face images
    
    ### Tips for best results:
    - Take photos in good lighting
    - Look directly at the camera
    - Take photos from different angles
    - Make sure your face is clearly visible
    """)
    
    # Check system status
    st.subheader("System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if os.path.exists('haarcascade_frontalface_alt.xml'):
            st.success("Face detection model")
        else:
            st.error("Face detection model missing")
    
    with col2:
        if os.path.exists('datasets') and len(os.listdir('datasets')) > 0:
            photo_count = len(os.listdir('datasets'))
            st.success(f"{photo_count} photos found")
        else:
            st.warning("No photos found")
    
    with col3:
        if os.path.exists('trainer/trainer.yml'):
            st.success("Trained model ready")
        else:
            st.warning("No trained model")

# Main app logic
def main():
    # Check if required file exists
    if not os.path.exists('haarcascade_frontalface_alt.xml'):
        st.error("haarcascade_frontalface_alt.xml not found!")
        st.info("Please make sure this file is in the same folder as this program")
        return
    
    # Navigation
    if page == "Home":
        home_page()
    elif page == "Add Person":
        add_person_streamlit()
    elif page == "Train Model":
        train_model_streamlit()
    elif page == "Face Recognition":
        face_recognition_streamlit()
    elif page == "Test Camera":
        test_camera_streamlit()

if __name__ == "__main__":
    main() 