import numpy as np
from skimage import io, color, filters, measure
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import cv2
import tempfile

def crop_image(image_path):
    global crop_coords
    crop_coords = None

    def line_select_callback(eclick, erelease):
        global crop_coords
        x1, y1 = int(eclick.xdata), int(eclick.ydata)
        x2, y2 = int(erelease.xdata), int(erelease.ydata)
        crop_coords = (x1, y1, x2, y2)
        plt.close()

    image = io.imread(image_path)
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    toggle_selector.RS = RectangleSelector(ax, line_select_callback,
                                           useblit=True,
                                           button=[1],  # left mouse button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    return crop_coords

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)

def mark_area_white(image, mark_coords):
    if mark_coords:
        x1, y1, x2, y2 = mark_coords
        image[y1:y2, x1:x2] = 255  # Paint the selected area white
    return image

def extract_data_points(image_path, crop_coords):
    # Load the image
    image = io.imread(image_path)
    
    # Check if the image is already in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image

    # Crop the image if crop coordinates are provided
    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        gray = gray[y1:y2, x1:x2]

    # Apply Gaussian filter to smooth the image
    blurred = filters.gaussian(gray, sigma=1)

    # Use Sobel filter to detect edges
    edges = filters.sobel(blurred)

    # Find contours in the edge-detected image
    contours = measure.find_contours(edges, level=0.2)

    data_points = []

    for contour in contours:
        for point in contour:
            x, y = point
            data_points.append((int(x), int(y)))

    # Plot the results (optional)
    fig, ax = plt.subplots()
    ax.imshow(gray, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.savefig('output_with_points.png')
    plt.show()

    return data_points

# Example usage
image_path = "C:\\Users\\NicklasKleemann\\Downloads\\km_curves_almashhadi.jpg"

# Crop the image
crop_coords = crop_image(image_path)

# Load the image for marking area white
image = cv2.imread(image_path)

# Mark area to be painted white
mark_coords = crop_image(image_path)  # Reuse the crop_image function for marking
image_with_white_area = mark_area_white(image, mark_coords)

# Save the modified image to a temporary file
with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
    temp_image_path = temp_file.name
    cv2.imwrite(temp_image_path, image_with_white_area)

# Use the modified image for data extraction
data_points = extract_data_points(temp_image_path, crop_coords)
print("Extracted data points:", data_points)