import cv2
import json
import numpy as np
from ultralytics import YOLO

# --- Setup ---
input_video = 'test_case_1/frame_1.mp4'
output_data = 'output.json'

start_time, end_time = 0, 7
color = (255, 0, 0)
detect_every = 10
count2=0 

# Load YOLO model
model = YOLO("yolo11s.pt").to('cuda')

# Read video
cap = cv2.VideoCapture(input_video)
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
ret, prev_frame = cap.read()
prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

count = 0
last_boxes = []
data = []

while cap.isOpened() and count < end_time * fps:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    vis = frame.copy()

    # Step 1: Detect with YOLO every N frames
    if count % detect_every == 0 and count2 <14 :
        results = model(frame)[0]
        last_boxes = results.boxes.xyxy.cpu().numpy().astype(int)
        count2+=1

    # Step 2: Optical Flow
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)

    step = 10
    s = 1
    frame_data = []
    updated_boxes = []

    # Step 3: Update each box using flow
    for box in last_boxes:
        x1, y1, x2, y2 = box
        dx_sum, dy_sum, count_valid = 0, 0, 0

        for y in range(y1, y2, step):
            for x in range(x1, x2, step):
                if y >= flow.shape[0] or x >= flow.shape[1]: continue
                dx, dy = flow[y, x]
                if np.sqrt(dx**2 + dy**2) > 0.5:
                    dx_sum += dx
                    dy_sum += dy
                    count_valid += 1

        if count_valid > 0:
            avg_dx = dx_sum / count_valid
            avg_dy = dy_sum / count_valid
        else:
            avg_dx = avg_dy = 0

        # Shift box
        new_x1 = int(x1 + avg_dx)
        new_y1 = int(y1 + avg_dy)
        new_x2 = int(x2 + avg_dx)
        new_y2 = int(y2 + avg_dy)

        # Save and draw updated box
        updated_boxes.append([new_x1, new_y1, new_x2, new_y2])
        cv2.rectangle(vis, (new_x1, new_y1), (new_x2, new_y2), (0, 255, 0), 2)

        # Save flow data inside box
        for y in range(new_y1, new_y2, step):
            for x in range(new_x1, new_x2, step):
                if y >= flow.shape[0] or x >= flow.shape[1]: continue
                dx, dy = flow[y, x]
                if np.sqrt(dx**2 + dy**2) > 1.0:
                    pt1 = (x, y)
                    pt2 = (int(x + s*dx), int(y + s*dy))
                    cv2.line(vis, pt1, pt2, color, 1)
                    cv2.circle(vis, pt1, 1, color, -1)
                    frame_data.append({'x': x, 'y': y, 'dx': round(float(dx), 3), 'dy': round(float(dy), 3)})

    if frame_data:
        data.append({'frame': count, 'data': frame_data})

    # Update the tracked boxes
    last_boxes = updated_boxes

    # Show the frame
    cv2.imshow("Tracked Boxes + Optical Flow", vis)
    if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
        break

    prev_gray = gray.copy()
    count += 1


cap.release()
cv2.destroyAllWindows()