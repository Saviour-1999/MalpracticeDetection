import os
import cv2
import numpy as np
import tensorflow as tf


def preprocess_frame(frame):
    """
    Preprocess the captured frame for prediction.
    """
    img = cv2.resize(frame, (128, 128))  # Resize for model input
    img = img / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img


def detect_cheating(frame, model, prediction):
    """
    Detect if a cheating is present in the frame.
    Simulate bounding box coordinates for demo purposes.
    """
    preprocessed = preprocess_frame(frame)
    prediction = model.predict(preprocessed)
    if np.argmax(prediction) == 1:  # Pothole detected
        # Simulate bounding box coordinates
        height, width, _ = frame.shape
        x1, y1, x2, y2 = int(0.3 * width), int(0.3 * height), int(0.7 * width), int(0.7 * height)
        return True, (x1, y1, x2, y2), prediction
    return False, None, prediction


def draw_bounding_box(frame, box_coords):
    """
    Draw a bounding box around the detected pothole region.
    """
    x1, y1, x2, y2 = box_coords
    color = (0, 0, 255)  # Red bounding box
    thickness = 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)


def real_time_detection():
    """
    Perform real-time pothole detection using webcam.
    """
    model = tf.keras.models.load_model("cheating_model.keras")
    prediction = 0
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Failed to capture frame.")
            break

        try:
            # Detect pothole and get bounding box
            detected, box_coords , prediction= detect_cheating(frame, model, prediction)
            p = prediction*100
            if detected:
                cv2.putText(frame, f"CHEATING DETECTED! : {p} %", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                draw_bounding_box(frame, box_coords)  # Draw the bounding box

            # Display the frame
            cv2.imshow("Cheating Detection", frame)

            # Add a break condition (press 'q' to quit)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except Exception as e:
            print(f"Error processing frame: {e}")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    real_time_detection()
