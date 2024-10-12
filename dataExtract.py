import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import io, color, filters, measure

# Global variables to store the ROI coordinates
crop_coords = None
exclude_coords = []

def line_select_callback(eclick, erelease):
    global crop_coords
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    crop_coords = [int(x1), int(y1), int(x2), int(y2)]

def exclude_select_callback(eclick, erelease):
    global exclude_coords
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata
    exclude_coords.append([int(x1), int(y1), int(x2), int(y2)])

def toggle_selector(event):
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        toggle_selector.RS.set_active(True)

def crop_image(image_path):
    global crop_coords
    crop_coords = None

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

def exclude_text_regions(image):
    global exclude_coords
    exclude_coords = []

    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    toggle_selector.RS = RectangleSelector(ax, exclude_select_callback,
                                           useblit=True,
                                           button=[1],  # left mouse button
                                           minspanx=5, minspany=5,
                                           spancoords='pixels',
                                           interactive=True)
    plt.connect('key_press_event', toggle_selector)
    plt.show()

    for coords in exclude_coords:
        x1, y1, x2, y2 = coords
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 255, 255), -1)  # Draw white rectangle over the selected region

    return image

def extract_curve_points(image_path):
    # Get the crop coordinates using the crop_image function
    print("Select the region of interest (ROI) to extract curve points:")
    crop_coords = crop_image(image_path)
    
    # Load the image
    image = io.imread(image_path)
    
    # Check if crop coordinates are valid
    if crop_coords is None or len(crop_coords) != 4:
        raise ValueError("Invalid crop coordinates")

    # Crop the image based on the selected ROI
    x1, y1, x2, y2 = crop_coords
    if x1 >= x2 or y1 >= y2:
        raise ValueError("Invalid crop coordinates: x1 >= x2 or y1 >= y2")
    cropped_image = image[y1:y2, x1:x2]

    # Allow the user to exclude text regions
    print("Select the text regions to exclude:")
    cropped_image = exclude_text_regions(cropped_image)

    # Convert the cropped image to HSV color space
    hsv_image = cv2.cvtColor(cropped_image, cv2.COLOR_RGB2HSV)

    # Scale the HSV image to 8-bit range
    hsv_image = (hsv_image * 255).astype(np.uint8)

    # Create a mask to exclude white and gray colors
    lower_bound = np.array([0, 0, 0])
    upper_bound = np.array([255, 51, 230])
    mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # Apply the mask to the image
    masked_image = cv2.bitwise_and(cropped_image, cropped_image, mask=mask)

    # Convert the masked image to grayscale
    gray_image = color.rgb2gray(masked_image)

    # Convert the grayscale image to high-contrast black and white
    threshold = filters.threshold_otsu(gray_image)
    binary_image = gray_image > threshold

    # Apply Gaussian filter to smooth the image
    blurred = filters.gaussian(binary_image, sigma=1)

    # Use Sobel filter to detect edges
    edges = filters.sobel(blurred)

    # Find contours in the edge-detected image
    contours = measure.find_contours(edges, level=0.2)

    data_points = []

    for contour in contours:
        for point in contour:
            x, y = point
            data_points.append((int(x), int(y)))

    return data_points, binary_image, contours

# Example usage
image_path = "C:\\Users\\NicklasKleemann\\Downloads\\km_curves_almashhadi.jpg"
try:
    points, binary_image, contours = extract_curve_points(image_path)
    
    # Plot the results
    fig, ax = plt.subplots()
    ax.imshow(binary_image, cmap=plt.cm.gray)

    for contour in contours:
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    plt.title("Detected Curve Points")
    plt.savefig('output_with_points.png')
    plt.show()
except FileNotFoundError as e:
    print(e)
except ValueError as e:
    print(e)