import numpy as np
import cv2
import os
import logging
import sys
from datetime import datetime
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('facial_recognition.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class FacialRecognitionSystem:
    def __init__(self):
        self.face_detector = None
        self.recognizer = None
        self.datasets_path = 'datasets'
        self.trainer_path = 'trainer'
        self.cascade_file = 'haarcascade_frontalface_alt.xml'
        self.exit_flag = False
        
        # Initialize face detector
        if os.path.exists(self.cascade_file):
            self.face_detector = cv2.CascadeClassifier(self.cascade_file)
            logger.info("Face detector initialized successfully")
        else:
            logger.error(f"Cascade file {self.cascade_file} not found!")
            raise FileNotFoundError(f"Cascade file {self.cascade_file} not found!")
    
    def show_menu(self):
        """Display the main menu"""
        print("\n" + "="*50)
        print("           FACIAL RECOGNITION SYSTEM")
        print("="*50)
        print("1. Add New Person (Take Photos)")
        print("2. Train the System")
        print("3. Start Recognition")
        print("4. Test Camera")
        print("5. Exit")
        print("="*50)
    
    def add_person(self):
        """Add a new person to the system by taking photos"""
        try:
            print("\n" + "="*30)
            print("ADDING NEW PERSON")
            print("="*30)
            
            # Create datasets folder if it doesn't exist
            if not os.path.exists(self.datasets_path):
                os.makedirs(self.datasets_path)
                logger.info("Created datasets folder")
            
            # Get person's name
            name = input("Enter person's name: ").strip()
            if not name:
                print("Name cannot be empty!")
                return
            
            # Get next available ID
            existing_ids = set()
            for filename in os.listdir(self.datasets_path):
                if filename.startswith('User.'):
                    parts = filename.split('.')
                    if len(parts) >= 2:
                        try:
                            existing_ids.add(int(parts[1]))
                        except ValueError:
                            continue
            
            person_id = 1
            while person_id in existing_ids:
                person_id += 1
            
            print(f"\nTaking photos of {name} (ID: {person_id})...")
            print("Instructions:")
            print("• Look at the camera")
            print("• Stay still")
            print("• Press 'q' to stop early")
            print("• Wait for 30 photos to be taken")
            
            # Start camera
            cam = cv2.VideoCapture(0)
            if not cam.isOpened():
                logger.error("Failed to open camera")
                print("Error: Camera not found or not accessible!")
                return
            
            # Create window and set it as active
            window_name = 'Taking Photos - Press Q to stop'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            photo_count = 0
            max_photos = 30
            
            while photo_count < max_photos:
                success, frame = cam.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    continue
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find faces in the frame
                faces = self.face_detector.detectMultiScale(gray_frame, 1.3, 5)
                
                if len(faces) == 0:
                    # Show message when no face detected
                    cv2.putText(frame, "No face detected", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # Take photo of the first face found
                    for face in faces[:1]:
                        x, y, w, h = face
                        # Draw green box around face
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        photo_count += 1
                        # Save the face photo
                        photo_name = f"{self.datasets_path}/User.{person_id}.{name}.jpg"
                        cv2.imwrite(photo_name, gray_frame[y:y+h, x:x+w])
                        
                        # Show photo count
                        cv2.putText(frame, f"Photos: {photo_count}/{max_photos}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(frame, f"Person: {name}", 
                                  (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show the camera feed
                cv2.imshow(window_name, frame)
                
                # Check for key press - improved key detection
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Q key pressed - stopping photo capture...")
                    break
                elif key == 27:  # ESC key as alternative
                    print("ESC key pressed - stopping photo capture...")
                    break
            
            # Clean up
            cam.release()
            cv2.destroyWindow(window_name)
            cv2.destroyAllWindows()
            
            logger.info(f"Finished taking {photo_count} photos of {name}")
            print(f"Finished! Took {photo_count} photos of {name}")
            
        except Exception as e:
            logger.error(f"Error in add_person: {str(e)}")
            print(f"Error: {str(e)}")
            # Ensure cleanup even on error
            try:
                cam.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def train_system(self):
        """Train the facial recognition system"""
        try:
            print("\n" + "="*30)
            print("TRAINING THE SYSTEM")
            print("="*30)
            
            # Check if we have photos to train with
            if not os.path.exists(self.datasets_path) or len(os.listdir(self.datasets_path)) == 0:
                print("No photos found! Please add people first")
                return
            
            print("Training the system...")
            
            # Create the recognizer
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Get all photos and their labels
            photos, labels = self._get_photos_and_labels()
            
            if len(photos) == 0:
                print("No valid photos found for training!")
                return
            
            # Train the system
            self.recognizer.train(photos, np.array(labels))
            
            # Create trainer folder and save the trained model
            if not os.path.exists(self.trainer_path):
                os.makedirs(self.trainer_path)
            
            trainer_file = f"{self.trainer_path}/trainer.yml"
            self.recognizer.write(trainer_file)
            
            logger.info(f"Training complete! Model saved to {trainer_file}")
            print(f"Training complete!")
            print(f"Model saved to {trainer_file}")
            print(f"Trained with {len(photos)} photos from {len(set(labels))} people")
            
        except Exception as e:
            logger.error(f"Error in train_system: {str(e)}")
            print(f"Error: {str(e)}")
    
    def _get_photos_and_labels(self):
        """Get photos and their corresponding labels for training"""
        photos = []
        labels = []
        
        try:
            # Go through all photos in datasets folder
            for filename in os.listdir(self.datasets_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                    
                photo_path = os.path.join(self.datasets_path, filename)
                
                # Read the photo
                image = cv2.imread(photo_path, cv2.IMREAD_GRAYSCALE)
                if image is None:
                    logger.warning(f"Could not read image: {photo_path}")
                    continue
                
                # Get the ID from filename (User.ID.name.jpg)
                parts = filename.split('.')
                if len(parts) >= 2:
                    try:
                        person_id = int(parts[1])
                        
                        # Find faces in the photo
                        detected_faces = self.face_detector.detectMultiScale(image)
                        
                        for (x, y, w, h) in detected_faces:
                            photos.append(image[y:y+h, x:x+w])
                            labels.append(person_id)
                    except ValueError:
                        logger.warning(f"Invalid filename format: {filename}")
                        continue
        
        except Exception as e:
            logger.error(f"Error in _get_photos_and_labels: {str(e)}")
        
        return photos, labels
    
    def start_recognition(self):
        """Start face recognition"""
        camera = None
        try:
            print("\n" + "="*30)
            print("STARTING FACE RECOGNITION")
            print("="*30)
            
            # Check if we have a trained model
            trainer_file = f"{self.trainer_path}/trainer.yml"
            if not os.path.exists(trainer_file):
                print("No trained model found! Please train the system first (option 2)")
                return
            
            print("Starting recognition...")
            print("Exit options:")
            print("- Press 'ESC' key")
            print("- Press 'Q' key") 
            print("- Press 'Enter' key")
            print("- Press 'Space' key")
            print("- Close the window")
            
            # Load the trained model
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.recognizer.read(trainer_file)
            
            # Start camera
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                logger.error("Failed to open camera for recognition")
                print("Camera not found!")
                return
            
            # Set camera size
            camera.set(3, 640)  # width
            camera.set(4, 480)  # height
            
            # Create window
            window_name = 'Face Recognition - Multiple Exit Options'
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            frame_count = 0
            while True:
                frame_count += 1
                
                # Get frame from camera
                success, frame = camera.read()
                if not success:
                    logger.warning("Failed to read frame from camera")
                    break
                
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Find faces
                faces = self.face_detector.detectMultiScale(gray_frame, 1.2, 5)
                
                # For each face found
                for (x, y, w, h) in faces:
                    # Draw box around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Try to recognize the face
                    try:
                        person_id, confidence = self.recognizer.predict(gray_frame[y:y+h, x:x+w])
                        
                        # Get the person's name
                        person_name = self._get_person_name(person_id)
                        
                        # Show confidence (lower is better)
                        if confidence < 100:
                            confidence_text = f"{round(100 - confidence)}%"
                            color = (0, 255, 0)  # Green for recognized
                        else:
                            person_name = "Unknown"
                            confidence_text = f"{round(100 - confidence)}%"
                            color = (0, 0, 255)  # Red for unknown
                        
                        # Show name and confidence on screen
                        cv2.putText(frame, person_name, (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                        cv2.putText(frame, confidence_text, (x+5, y+h+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                    except Exception as e:
                        logger.error(f"Error in face recognition: {str(e)}")
                        cv2.putText(frame, "Error", (x+5, y-5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Add exit instructions on frame
                cv2.putText(frame, "Exit: ESC, Q, Enter, Space, or close window", (10, frame.shape[0] - 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Show the camera feed
                cv2.imshow(window_name, frame)
                
                # Key detection with multiple methods
                key = cv2.waitKey(1) & 0xFF
                
                # Debug: Print key codes (only occasionally to avoid spam)
                if key != 255 and frame_count % 30 == 0:
                    print(f"Key pressed: {key}")
                
                # Check for exit keys
                if key == 27:  # ESC
                    print("ESC key detected - exiting...")
                    break
                elif key in [ord('q'), ord('Q')]:  # Q
                    print("Q key detected - exiting...")
                    break
                elif key == 13:  # Enter
                    print("Enter key detected - exiting...")
                    break
                elif key == 32:  # Space
                    print("Space key detected - exiting...")
                    break
                
                # Check for window close
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        print("Window closed - exiting...")
                        break
                except:
                    pass
            
            # Clean up
            if camera is not None:
                camera.release()
            cv2.destroyAllWindows()
            logger.info("Recognition stopped")
            print("Recognition stopped")
            
        except Exception as e:
            logger.error(f"Error in start_recognition: {str(e)}")
            print(f"Error: {str(e)}")
            # Ensure cleanup even on error
            try:
                if camera is not None:
                    camera.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def _get_person_name(self, person_id):
        """Get person name from ID"""
        try:
            if os.path.exists(self.datasets_path):
                for filename in os.listdir(self.datasets_path):
                    if filename.startswith(f'User.{person_id}.'):
                        # Extract name from filename (User.id.name.jpg)
                        parts = filename.split('.')
                        if len(parts) >= 3:
                            return parts[2]  # Return the name
        except Exception as e:
            logger.error(f"Error in _get_person_name: {str(e)}")
        
        return f"Person_{person_id}"
    
    def test_camera(self):
        """Test camera functionality"""
        try:
            print("\n" + "="*30)
            print("TESTING CAMERA")
            print("="*30)
            print("Press 'q' or 'ESC' to stop")
            
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                logger.error("Failed to open camera for testing")
                print("Camera not found or not working!")
                return
            
            # Create window and set it as active
            window_name = "Camera Test - Press Q or ESC to stop"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
            
            while True:
                success, frame = camera.read()
                
                if not success:
                    logger.warning("Failed to get camera frame during test")
                    print("Failed to get camera frame!")
                    break
                
                # Show camera feed
                cv2.imshow(window_name, frame)
                
                # Check for 'q' or ESC key to exit - improved key detection
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Q key pressed - stopping camera test...")
                    break
                elif key == 27:  # ESC key
                    print("ESC key pressed - stopping camera test...")
                    break
            
            camera.release()
            cv2.destroyWindow(window_name)
            cv2.destroyAllWindows()
            logger.info("Camera test finished")
            print("Camera test finished")
            
        except Exception as e:
            logger.error(f"Error in test_camera: {str(e)}")
            print(f"Error: {str(e)}")
            # Ensure cleanup even on error
            try:
                camera.release()
                cv2.destroyAllWindows()
            except:
                pass
    
    def _keyboard_monitor(self):
        """Monitor keyboard input in a separate thread"""
        try:
            import msvcrt  # Windows-specific
            while not self.exit_flag:
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    if key == b'\x1b':  # ESC key
                        print("\nESC detected via keyboard monitor - exiting...")
                        self.exit_flag = True
                        break
                    elif key in [b'q', b'Q']:  # Q key
                        print("\nQ detected via keyboard monitor - exiting...")
                        self.exit_flag = True
                        break
                time.sleep(0.01)
        except ImportError:
            # Fallback for non-Windows systems
            pass
        except Exception as e:
            logger.error(f"Error in keyboard monitor: {str(e)}")

def main():
    """Main function"""
    try:
        print("Welcome to Facial Recognition System!")
        print("Initializing...")
        
        # Initialize the system
        system = FacialRecognitionSystem()
        
        logger.info("Facial Recognition System started")
        
        # Main program loop
        while True:
            system.show_menu()
            
            try:
                choice = input("Enter your choice (1-5): ").strip()
                
                if choice == '1':
                    system.add_person()
                elif choice == '2':
                    system.train_system()
                elif choice == '3':
                    system.start_recognition()
                elif choice == '4':
                    system.test_camera()
                elif choice == '5':
                    print("Thank you for using Facial Recognition System!")
                    logger.info("Facial Recognition System stopped")
                    break
                else:
                    print("Invalid choice! Please enter 1, 2, 3, 4, or 5")
                    
            except KeyboardInterrupt:
                print("\n\nProgram interrupted by user")
                logger.info("Program interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                print(f"Error: {str(e)}")
                
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        print(f"Critical error: {str(e)}")
        print("Please check the log file for more details")

if __name__ == "__main__":
    main() 