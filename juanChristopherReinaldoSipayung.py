import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

prev_time = 0

def get_shape(contour):
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
    vertices = len(approx)
    
    if vertices == 3:
        return "Triangle", approx
    elif vertices == 4:
        x, y, w, h = cv2.boundingRect(approx)
        ratio = float(w) / h
        if 0.85 <= ratio <= 1.15:
            return "Square", approx
        else:
            return "Rectangle", approx
    elif vertices == 5:
        return "Pentagon", approx
    elif vertices == 6:
        return "Hexagon", approx
    else:
        area = cv2.contourArea(contour)
        if area > 0 and peri > 0:
            circularity = (4 * np.pi * area) / (peri * peri)
            if 0.7 < circularity < 1.3:
                return "Circle", approx
    
    return "Unknown", approx

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if 1000 < area < 50000:
            shape_name, approx = get_shape(contour)
            
            if shape_name != "Unknown":
                cv2.drawContours(frame, [approx], -1, (0, 255, 0), 2)
                
                M = cv2.moments(approx)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                else:
                    cX, cY = approx.ravel()[0], approx.ravel()[1]
                
                cv2.putText(frame, shape_name, (cX-50, cY-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time
    
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow("Shape Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("shape_detection_frame.png", frame)
        print("Frame saved!")
    elif key == ord('1'):
        cv2.imshow("Binary", binary)

cap.release()
cv2.destroyAllWindows()