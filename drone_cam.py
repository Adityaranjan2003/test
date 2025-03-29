import cv2
import time
from picamera2 import Picamera2

# Initialize the Raspberry Pi Camera
picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()

# Define the save path
SAVE_PATH = "img/"  # Change this path if needed

# Capture image function
def capture_and_save_image():
    timestamp = time.strftime("%Y%m%d-%H%M%S")  # Generate a timestamp
    image_filename = f"{SAVE_PATH}image_{timestamp}.jpg"
    
    # Capture image
    frame = picam2.capture_array()  # Capture image as an array
    cv2.imwrite(image_filename, frame)  # Save image locally
    
    print(f"Image saved: {image_filename}")

# Capture and save image every 5 seconds (or adjust as needed)
try:
    while True:
        capture_and_save_image()
        time.sleep(5)  # Adjust interval for capturing images

except KeyboardInterrupt:
    print("Image capturing stopped.")

finally:
    picam2.close()
