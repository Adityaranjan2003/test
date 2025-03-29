import cv2
import numpy as np
import RPi.GPIO as GPIO
import time
from pymavlink import mavutil

# YOLO model setup
yolo_net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = yolo_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo_net.getUnconnectedOutLayers()]

# Load class labels (COCO dataset)
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Connect to APM 2.8 via MAVLink (Assuming /dev/serial0)
vehicle = mavutil.mavlink_connection("/dev/serial0", baud=57600)
vehicle.wait_heartbeat()

# GPIO Pin Configuration (PWM Read from FlySky FS-i6 Receiver)
GPIO.setmode(GPIO.BCM)
CHANNELS = {"roll": 17, "pitch": 27, "throttle": 22, "yaw": 23}
for pin in CHANNELS.values():
    GPIO.setup(pin, GPIO.IN)

# Function to measure PWM signal width
def read_pwm(channel):
    start = time.time()
    while GPIO.input(channel) == GPIO.LOW:
        if time.time() - start > 0.02:  # Timeout for failsafe
            return 1500
    start = time.time()
    while GPIO.input(channel) == GPIO.HIGH:
        if time.time() - start > 0.02:
            return 1500
    return (time.time() - start) * 1_000_000  # Convert to microseconds

# Function to send autonomous movement commands
def send_override(roll, pitch, throttle, yaw):
    vehicle.mav.rc_channels_override_send(
        vehicle.target_system,
        vehicle.target_component,
        roll, pitch, throttle, yaw, 0, 0, 0, 0
    )

# Camera Setup
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    height, width, _ = frame.shape
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

    # Read manual control inputs
    roll = int(read_pwm(CHANNELS["roll"]))
    pitch = int(read_pwm(CHANNELS["pitch"]))
    throttle = int(read_pwm(CHANNELS["throttle"]))
    yaw = int(read_pwm(CHANNELS["yaw"]))

    # Override manual commands if an obstacle is detected
    if obstacle_detected:
        if move_command == "stop":
            send_override(1500, 1500, 1000, 1500)  # Stop the drone
            print("Obstacle ahead! Stopping.")
        elif move_command == "left":
            send_override(1300, 1500, throttle, yaw)  # Move left
            print("Obstacle right! Moving left.")
        elif move_command == "right":
            send_override(1700, 1500, throttle, yaw)  # Move right
            print("Obstacle left! Moving right.")
    else:
        send_override(roll, pitch, throttle, yaw)  # Allow manual control

    # Display Output
    cv2.imshow("Obstacle Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
GPIO.cleanup()
