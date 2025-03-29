import cv2
import numpy as np
import time
from picamera2 import Picamera2
from pymavlink import mavutil

# Initialize Raspberry Pi Camera
picam2 = Picamera2()
picam2.preview_configuration.main.size = (640, 480)
picam2.preview_configuration.main.format = "RGB888"
picam2.configure("preview")
picam2.start()

# YOLO Model Setup
yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Connect to APM 2.8 via MAVLink
vehicle = mavutil.mavlink_connection("/dev/serial0", baud=57600)
vehicle.wait_heartbeat()
print("Connected to APM 2.8")

# Function to send autonomous movement commands
def send_override(roll, pitch, throttle, yaw):
    vehicle.mav.rc_channels_override_send(
        vehicle.target_system, vehicle.target_component,
        roll, pitch, throttle, yaw, 0, 0, 0, 0
    )

while True:
    # Capture Image from Pi Camera
    frame = picam2.capture_array()
    height, width, _ = frame.shape

    # YOLO Preprocessing
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    yolo_net.setInput(blob)
    detections = yolo_net.forward(output_layers)

    obstacle_detected = False
    move_command = None

    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                obj_x, obj_y, obj_w, obj_h = (obj[0] * width, obj[1] * height, obj[2] * width, obj[3] * height)
                x_center, y_center = int(obj_x), int(obj_y)

                # Draw bounding box
                x, y, w, h = int(obj_x - obj_w / 2), int(obj_y - obj_h / 2), int(obj_w), int(obj_h)
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{classes[class_id]} ({confidence:.2f})", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Determine Obstacle Position
                if width // 3 < x_center < 2 * width // 3:
                    move_command = "stop"
                elif x_center < width // 3:
                    move_command = "right"
                elif x_center > 2 * width // 3:
                    move_command = "left"

                obstacle_detected = True

    # Override manual control if an obstacle is detected
    if obstacle_detected:
        if move_command == "stop":
            send_override(1500, 1500, 1000, 1500)  # Stop the drone
            print("Obstacle ahead! Stopping.")
        elif move_command == "left":
            send_override(1300, 1500, 1200, 1500)  # Move left
            print("Obstacle right! Moving left.")
        elif move_command == "right":
            send_override(1700, 1500, 1200, 1500)  # Move right
            print("Obstacle left! Moving right.")
    else:
        send_override(1500, 1500, 1200, 1500)  # Maintain stable flight

    # Display Output
    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
