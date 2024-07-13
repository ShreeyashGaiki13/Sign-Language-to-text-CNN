import cv2
import numpy as np
from keras.models import load_model
from filter import filter

# Load the trained model
model = load_model("D:\\Mini Project\\Project folder\\sign_language_model.h5")

vid = cv2.VideoCapture(0)
# vid.set(3, 960) # Uncomment this line to set the frame width

# Define class labels
class_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'blank']

while True:
    _, frame = vid.read()

    # Mirror image
    frame = cv2.flip(frame, 1)

    # Resizing the frame window
    frame = cv2.resize(frame, (700, 500))

    # Draw the rectangle on the frame (430, 14), (680, 260)
    cv2.rectangle(frame, (430, 10), (680, 260), (124, 252, 0), 3)

    # Define the region of interest (ROI)
    roi_frame = frame[15:260, 430:670]

    # Apply filter function
    processed_frame = filter(roi_frame)
    
    # Showing filtered frames
    cv2.imshow("Filtered frames", processed_frame)
    
    # Preprocessing for model
    processed_frame = np.array(processed_frame)
    processed_frame = processed_frame.reshape(1, 48, 48, 1)
    processed_frame = processed_frame/255.0
    
    # Make predictions using the model
    prediction = model.predict(processed_frame)
    predicted_class = np.argmax(prediction)
    
    # Replace class label "10" with "blank"
    if predicted_class == 10:
        predicted_class = "blank"
    else:
        predicted_class = class_labels[predicted_class]
 
    # Write predicted class on the video frame
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f'Predicted Class: {predicted_class}', (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Place filtered ROI back into the frame
    # frame[9:319, 350:680] = filtered_roi

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) == ord('q'):  # Press 'q' to quit
        break

vid.release()
cv2.destroyAllWindows()
