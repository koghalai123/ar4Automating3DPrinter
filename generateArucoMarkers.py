import cv2
import numpy as np

# Define the Aruco dictionary
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

# Generate markers with IDs from 0 to 20
for marker_id in range(21):
    # Create a blank image
    marker_image = np.zeros((50, 50), dtype=np.uint8)
    
    # Draw the marker on the image
    cv2.aruco.drawMarker(dictionary, marker_id, 50, marker_image, 1)
    
    # Save the marker as a PNG file
    filename = f'marker4x4_{marker_id}.png'
    cv2.imwrite(filename, marker_image)
    print(f'Saved {filename}')

dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)

# Generate markers with IDs from 0 to 20
for marker_id in range(21):
    # Create a blank image
    marker_image = np.zeros((50, 50), dtype=np.uint8)
    
    # Draw the marker on the image
    cv2.aruco.drawMarker(dictionary, marker_id, 50, marker_image, 1)
    
    # Save the marker as a PNG file
    filename = f'marker6x6_{marker_id}.png'
    cv2.imwrite(filename, marker_image)
    print(f'Saved {filename}')

print('All Aruco markers generated and saved.')
