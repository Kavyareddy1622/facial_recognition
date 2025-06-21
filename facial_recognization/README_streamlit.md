# 🤖 Facial Recognition System with Streamlit

A beginner-friendly facial recognition system with a beautiful web interface built using Streamlit.

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements_streamlit.txt
```

### 2. Run the Streamlit App
```bash
streamlit run streamlit_facial_recognition.py
```

### 3. Open Your Browser
The app will automatically open in your default browser at `http://localhost:8501`

## 📋 Features

- **🏠 Home Page**: Overview and system status
- **📸 Add Person**: Take photos of people to recognize
- **🧠 Train Model**: Train the AI to recognize faces
- **👁️ Face Recognition**: Real-time face recognition
- **📹 Test Camera**: Check if your camera is working

## 🎯 How to Use

### Step 1: Add People
1. Go to "📸 Add Person" page
2. Enter the person's name
3. Click "🎥 Start Camera"
4. Look at the camera and stay still
5. The system will capture 30 photos automatically
6. Click "⏹️ Stop Capture" to finish early

### Step 2: Train the Model
1. Go to "🧠 Train Model" page
2. Click "🚀 Start Training"
3. Wait for the training to complete
4. You'll see a success message when done

### Step 3: Start Recognition
1. Go to "👁️ Face Recognition" page
2. Click "🎥 Start Recognition"
3. Look at the camera
4. The system will show names and confidence levels
5. Click "⏹️ Stop Recognition" to stop

## 🔧 Requirements

- Python 3.7+
- Webcam
- Good lighting
- `haarcascade_frontalface_alt.xml` file in the same directory

## 📁 File Structure

```
facial_recognization/
├── streamlit_facial_recognition.py    # Main Streamlit app
├── requirements_streamlit.txt         # Python dependencies
├── haarcascade_frontalface_alt.xml   # Face detection model
├── datasets/                          # Stored face photos
├── trainer/                           # Trained model
└── README_streamlit.md               # This file
```

## 💡 Tips for Best Results

- **Good Lighting**: Make sure your face is well-lit
- **Clear Background**: Use a simple background
- **Multiple Angles**: Take photos from different angles
- **Clear Face**: Make sure your face is clearly visible
- **No Glasses**: Remove glasses if possible for better recognition

## 🐛 Troubleshooting

### Camera Not Working
- Check if another app is using the camera
- Restart your computer
- Check camera permissions

### Recognition Not Working
- Make sure you've trained the model
- Check if photos are clear and well-lit
- Try taking more photos of the person

### Installation Issues
- Make sure you have Python 3.7+
- Install dependencies: `pip install -r requirements_streamlit.txt`
- Check if `haarcascade_frontalface_alt.xml` is in the same folder

## 🎉 Enjoy!

This Streamlit version makes facial recognition super easy and beginner-friendly. Just follow the steps and you'll have a working facial recognition system in no time! 