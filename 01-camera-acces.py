import cv2
from datetime import datetime
import time

# Text parameters
font_position = (10, 30)
font_style = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 1
font_color = (0, 255, 0)
font_thickness = 2
font_line_type = cv2.LINE_AA

# Open the camera
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
  print("Error: Cannot open camera")
  exit()

# Initialize time for FPS calculation
prev_time = 0

# If camera opened successfully
while True:
  # Read frame-by-frame
  ret, frame = cap.read()

  # Check if frame read successfully
  if not ret:
    print("Error: Cannot read frame")
    break

  # Calculate FPS
  current_time = time.time()
  fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
  prev_time = current_time

  # Draw FPS on frame
  cv2.putText(
    frame,
    f"{fps:.2f} FPS",
    font_position,
    font_style,
    font_scale,
    font_color,
    font_thickness,
    font_line_type
  )

  # Display the frame
  cv2.imshow("Camera Feed", frame)

  # Wait for key click for 1 ms
  key = cv2.waitKey(1) & 0xFF

  # Quit if "q" is pressed
  if key == ord("q"):
    print("Quitting...")
    break

  # Save the frame if "s" is pressed
  elif key == ord("s"):
    date = datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
    print(f"Saving frame {date}.png ...")
    cv2.imwrite(f"{date}.png", frame)

# Release the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()