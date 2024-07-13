import cv2

# target_size=(128, 128)
def filter(frame, target_size=(48, 48)):    
    # Convert ROI to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    # blur = cv2.GaussianBlur(gray, (5, 5), 2)
    
    # Apply adaptive thresholding
    # th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Apply Otsu's thresholding
    # ret, filtered_frame = cv2.threshold(th3, minValue, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Resize the filtered frame to match the target size
    filtered_frame = cv2.resize(gray, target_size)
    
    return filtered_frame
