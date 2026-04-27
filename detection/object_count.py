import cv2
import numpy as np
try:
    from ultralytics import YOLO
    # Load model on module import to keep it in memory
    yolo_model = YOLO("yolov8n.pt")
except ImportError:
    yolo_model = None
    print("Warning: ultralytics package not found. YOLO object counting will fallback to basic OpenCV.")

def count_and_draw_products(image, 
                            min_contour_area=1500,
                            min_width=25,
                            min_height=25,
                            max_aspect_ratio=8.0):
    """
    Detect and count products in an image. Uses YOLOv8 if available for robust detection,
    otherwise falls back to OpenCV contour detection.
    Returns:
    - image: annotated frame with bounding boxes
    - count: number of items detected
    - boxes: list of bounding boxes (startX, startY, endX, endY) for tracking
    """
    if yolo_model is not None:
        # Define a Region of Interest (ROI) to "zoom in" and maximize accuracy
        # This focuses the model only on the center of the conveyor belt where objects pass
        h, w = image.shape[:2]
        zoom_factor = 0.6  # 60% of the center frame
        
        roi_w = int(w * zoom_factor)
        roi_h = int(h * zoom_factor)
        
        x_offset = (w - roi_w) // 2
        y_offset = (h - roi_h) // 2
        
        # Crop the ROI
        roi_image = image[y_offset:y_offset+roi_h, x_offset:x_offset+roi_w]
        
        # Draw the Counting Zone on the original image so the user sees it
        cv2.rectangle(image, (x_offset, y_offset), (x_offset + roi_w, y_offset + roi_h), (255, 0, 0), 2)
        cv2.putText(image, "Detection Zone", (x_offset, y_offset - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Run YOLO inference ONLY on the zoomed ROI
        results = yolo_model(roi_image, verbose=False, conf=0.4)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                # get box coordinates in the ROI space
                b = box.xyxy[0].cpu().numpy().astype(int)
                
                # Shift coordinates back to the original full-frame space
                orig_x1 = b[0] + x_offset
                orig_y1 = b[1] + y_offset
                orig_x2 = b[2] + x_offset
                orig_y2 = b[3] + y_offset
                
                boxes.append((orig_x1, orig_y1, orig_x2, orig_y2))
                
                # Draw bounding box on the original image
                cv2.rectangle(image, (orig_x1, orig_y1), (orig_x2, orig_y2), (0, 255, 0), 2)
        
        return image, len(boxes), boxes
    
    # Fallback to OpenCV if YOLO not installed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    thresh = cv2.adaptiveThreshold(morph, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    filtered_contours = []
    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area:
            continue
        
        x, y, w, h = cv2.boundingRect(cnt)
        if w < min_width or h < min_height:
            continue
        
        aspect_ratio = max(w, h) / min(w, h)
        if aspect_ratio > max_aspect_ratio:
            continue
        
        filtered_contours.append(cnt)
        boxes.append((x, y, x + w, y + h))

    for idx, contour in enumerate(filtered_contours):
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # cv2.putText(image, str(idx + 1), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return image, len(filtered_contours), boxes

def process_realtime_video(source):
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        processed_frame, count, _ = count_and_draw_products(frame)
        
        cv2.putText(processed_frame, f'Total Items: {count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Real-Time Object Detection", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Commented out to prevent automatic camera access when imported
# Uncomment the line below if you want to run real-time video processing
# process_realtime_video(0)

if __name__ == "__main__":
    # Only run real-time video when script is executed directly
    process_realtime_video(0)
