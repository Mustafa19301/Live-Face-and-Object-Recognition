import cv2
import dlib
import imutils
import time
import numpy as np

# Load YOLO configuration and weights
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

face_detector = dlib.get_frontal_face_detector()

# Function to perform face and object detection
def live_face_and_object_detection_with_fps():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Press 'q' to quit the live stream.")

    fps_start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        frame_count += 1

        frame_resized = imutils.resize(frame, width=600)

        gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        faces = face_detector(gray, 1)

        blob = cv2.dnn.blobFromImage(frame_resized, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        detections = net.forward(output_layers)

        # Process YOLO detections
        for detection in detections:
            for obj in detection:
                scores = obj[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    if class_id == 0:  
                        continue  

                    center_x = int(obj[0] * frame_resized.shape[1])
                    center_y = int(obj[1] * frame_resized.shape[0])
                    w = int(obj[2] * frame_resized.shape[1])
                    h = int(obj[3] * frame_resized.shape[0])

                    cv2.rectangle(frame_resized, (center_x - w // 2, center_y - h // 2), 
                                  (center_x + w // 2, center_y + h // 2), (0, 255, 0), 2)

        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            cv2.rectangle(frame_resized, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face_count = len(faces)
        cv2.putText(frame_resized, f"Faces: {face_count}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        fps_end_time = time.time()
        fps = frame_count / (fps_end_time - fps_start_time)

        cv2.putText(frame_resized, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Live Face and Object Detection", frame_resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_face_and_object_detection_with_fps()
