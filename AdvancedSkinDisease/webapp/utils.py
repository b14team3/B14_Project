import cv2
import numpy as np
import os
from django.conf import settings

def process_image_dimensions(image_path):
    """
    Process the image using OpenCV and calculate dimensions
    Returns a dictionary with image dimensions and properties
    """
    img = cv2.imread(image_path)
    
    if img is None:
        return None
        
    # Get basic dimensions
    height, width, channels = img.shape
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply threshold to get binary image
    _, threshold = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (assuming it's the main object)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Calculate aspect ratio
        aspect_ratio = w / h if h != 0 else 0
        
        # Calculate area
        area = w * h
        
        # Get object dimensions in pixels
        object_width = w
        object_height = h
        
        return {
            'image_width': width,
            'image_height': height,
            'object_width': object_width,
            'object_height': object_height,
            'aspect_ratio': round(aspect_ratio, 2),
            'area': area
        }
    
    return None

def save_processed_image(image_path):
    """
    Save a processed version of the image with dimensions overlay
    Returns the path to the processed image
    """
    # Read the image
    img = cv2.imread(image_path)
    
    if img is None:
        return None
        
    # Get dimensions
    dimensions = process_image_dimensions(image_path)
    
    if dimensions:
        # Draw dimensions on image
        cv2.putText(img, f"Width: {dimensions['object_width']}px", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"Height: {dimensions['object_height']}px", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Create processed images directory if it doesn't exist
        processed_dir = os.path.join(settings.MEDIA_ROOT, 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save processed image with a unique name to avoid conflicts
        filename = os.path.basename(image_path)
        base_name, ext = os.path.splitext(filename)
        processed_filename = f"processed_{base_name}{ext}"
        processed_path = os.path.join(processed_dir, processed_filename)
        cv2.imwrite(processed_path, img)
        
        # Return the URL-friendly path that matches Django's MEDIA_URL structure
        return os.path.join('processed', processed_filename)  # Changed this line
    
    return None