


import numpy as np


# Load camera matrix and distortion coefficients
camera_mtx = np.load('camera_mtx.npy')
dist_coeffs = np.load('dist_coeffs.npy')
distance_to_object = 360

latest_measurements = {'width': 0, 'height': 0}

import cv2
import numpy as np

def measure_dimensions(frame,camera_mtx,dist_coeffs,distance_to_object):
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise and improve edge detection
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding to handle different lighting conditions
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Morphological operations to close gaps in between object edges
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Find contours from the thresholded image
    contours, _ = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]

    # Filter and draw contours
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)


        if len(approx)  >2:
            minRect = cv2.minAreaRect(c)
            box = cv2.boxPoints(minRect)
            box = np.int0(box)  # Convert the box points to integer values

            # Draw the rotated rectangle
            cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)

            # Compute the width and height of the rectangle
            edge1 = np.linalg.norm(box[0] - box[1])
            edge2 = np.linalg.norm(box[1] - box[2])
            width_pixels = max(edge1, edge2)
            height_pixels = min(edge1, edge2)



            # Compute the bounding box of the contour and use it to compute the object dimensions
            (x, y, w, h) = cv2.boundingRect(approx)
            # Calculate real-world dimensions
            width_mm = (width_pixels * distance_to_object) / camera_mtx[0, 0]  # Using f_x from the camera matrix
            height_mm = (height_pixels * distance_to_object) / camera_mtx[1, 1]  # Using f_y from the camera matrix

            latest_measurements['width'] = width_mm
            latest_measurements['height'] = height_mm
            #print(f"Width: {width_mm:.2f} mm, Height: {height_mm:.2f} mm")  # Debug print

            # Display the dimensions on the image
            cv2.putText(frame, f"Width: {width_mm:.2f} mm", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"Height: {height_mm:.2f} mm", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 2)
    return frame
