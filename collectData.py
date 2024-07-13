import cv2
import os
from filter import filter  # Filter function is imported from a separate file

# Directory where the collected data will be stored
directory = 'D:\\Mini Project\\collected_data'

# Create the directory if it doesn't exist
if not os.path.exists(directory):
    os.mkdir(directory)

# Create subdirectories for each label (0 to 9 and 'blank')
for i in range(11):  
    if i == 10:
        label = 'blank'
    else:
        label = str(i)
    if not os.path.exists(f'{directory}/{label}'):
        os.mkdir(f'{directory}/{label}')

# Open the webcam
vid = cv2.VideoCapture(0)

# Initialize count for each label
count = {str(i): 0 for i in range(10)}
count['blank'] = 0 # Start the count from last image number

while True:
    # Read frame from the webcam
    _, frame = vid.read()
    
    # Mirror the image horizontally
    frame = cv2.flip(frame, 1)
    
    # Resize the frame window
    frame = cv2.resize(frame, (700, 500))

    # Draw a rectangle on the frame to mark the region of interest (ROI)
    cv2.rectangle(frame, (430, 10), (680, 260), (124, 252, 0), 3)
    
    # Display the frame with ROI
    cv2.imshow("data", frame)
    
    # Extract the ROI from the frame
    roi_frame = frame[9:259, 429:670]
    cv2.imshow("ROI", roi_frame)
    
    # Process the ROI using the filter function
    processed_frame = filter(roi_frame, (48, 48))  
    
    # Wait for user input
    interrupt = cv2.waitKey(10)
    
    # Save the processed frame based on user input
    for i in range(10):
        if interrupt & 0xFF == ord(str(i)):
            cv2.imwrite(os.path.join(directory, str(i), f'{count[str(i)]}.jpg'), processed_frame)
            count[str(i)] += 1

    # Save the processed frame as 'blank' if '.' key is pressed
    if interrupt & 0xFF == ord('.'):
        cv2.imwrite(os.path.join(directory, 'blank', f'{count["blank"]}.jpg'), processed_frame)
        count['blank'] += 1

    # Exit the loop if 'q' key is pressed
    if interrupt & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
vid.release()
cv2.destroyAllWindows()
