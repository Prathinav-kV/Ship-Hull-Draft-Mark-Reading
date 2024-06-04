import cv2
import numpy as np
import pytesseract
import re
from matplotlib import pyplot as plt

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

# Computes the homography matrix to transform the frames so that draft marks are horizontal
def compute_homography_matrix(input_image):
    lower_color_threshold = np.array([0, 80, 150])  # BGR color lower threshold
    upper_color_threshold = np.array([80, 150, 255])  # BGR color upper threshold
    num_columns, num_rows, _ = np.shape(input_image)  # Get image dimensions
    roi_x1, roi_y1, roi_x2, roi_y2 = 1130, 0, 1500, 300  # Region of interest coordinates
    mask = np.zeros((num_columns, num_rows), dtype=np.uint8)  # Initialize mask
    rect_mask = cv2.rectangle(mask, (roi_x1, roi_y1), (roi_x2, roi_y2), 255, -1)  # Draw rectangle on mask
    image_with_rect = cv2.bitwise_and(input_image, input_image, mask=rect_mask)  # Apply mask to input image
    blurred_image = cv2.GaussianBlur(image_with_rect, (7, 7), 0)  # Blur the masked image
    color_mask = cv2.inRange(blurred_image, lower_color_threshold, upper_color_threshold)  # Threshold the blurred image
    filtered_image = cv2.bitwise_and(blurred_image, blurred_image, mask=color_mask)  # Apply color mask
    edges = cv2.Canny(filtered_image, 50, 200)  # Detect edges
    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=60, minLineLength=0, maxLineGap=50)  # Detect lines
    point1 = [lines[3][0][0], lines[3][0][1]]  # Extract points from lines
    point2 = [lines[3][0][2], lines[3][0][3]]
    point3 = [lines[4][0][0], lines[4][0][1]]
    point4 = [lines[4][0][2], lines[4][0][3]]
    dest_point1 = [point1[0] + 100, 190]  # Destination points for homography
    dest_point2 = [point2[0] + 100, 190]
    dest_point3 = [point3[0] + 97, 145 + 1]
    dest_point4 = [point4[0] + 100, 145]
    src_points = np.array([point1, point2, point3, point4])  # Source points for homography
    dest_points = np.array([dest_point1, dest_point2, dest_point3, dest_point4])
    homography_matrix, _ = cv2.findHomography(src_points, dest_points)  # Compute homography matrix
    return homography_matrix

# Estimates the water level based on draft marks
def estimate_water_level(draft_marks, level, pixel_average):
    mark_to_height = {}  # Dictionary to map y-coordinates to height values
    height_values = []
    initial_height = 10.5
    for i in range(len(draft_marks)):  # Create height values from 10.5, decrementing by 0.1
        height_values.append(initial_height)
        initial_height = round((initial_height - 0.1), 2)
    height_values.reverse()  # Reverse the list to match y-coordinate order
    for i in range(len(draft_marks)):  # Map each y-coordinate to its height value
        mark_to_height[draft_marks[i][1]] = height_values[i]
    y_values = [pt[1] for pt in draft_marks]  # Extract y-coordinates from draft marks
    nearest_mark = -1
    if level > draft_marks[0][1]:  # Ensure the level is within the range of markings
        level = draft_marks[0][1] - 1
    while nearest_mark < level:  # Find the nearest y-coordinate below the current water level
        nearest_mark = min(y_values, key=lambda x: abs(x - level))
        y_values.remove(nearest_mark)
    remaining_distance = np.abs(level - nearest_mark)  # Calculate the remaining distance in pixels
    distance_in_meters = (remaining_distance / pixel_average) / 100  # Convert to meters
    estimated_height = round((mark_to_height[nearest_mark] + distance_in_meters), 2)  # Estimate the height
    return estimated_height

# Resizes an image for easier viewing on screen
def resize_image_for_viewing(input_image, scale):
    width_scaled = int(input_image.shape[1] * scale)  # Calculate new width
    height_scaled = int(input_image.shape[0] * scale)  # Calculate new height
    dimensions = (width_scaled, height_scaled)  # Create new dimensions tuple
    resized_image = cv2.resize(input_image, dimensions, interpolation=cv2.INTER_AREA)  # Resize the image
    return resized_image

# Initialize video capture
video_capture = cv2.VideoCapture("hull.mp4")

if not video_capture.isOpened():  # Check if video opened successfully
    print("Error: Could not open video.")
    exit()

ret, initial_frame = video_capture.read()  # Read the first frame
if not ret:
    print("Error: Could not read frame.")
    exit()

num_columns, num_rows, _ = np.shape(initial_frame)  # Get the dimensions of the frame
homography_matrix = compute_homography_matrix(initial_frame)  # Compute the homography matrix
initial_frame = cv2.warpPerspective(initial_frame, homography_matrix, (num_rows, num_columns + 200))  # Warp the frame
background_subtractor = cv2.createBackgroundSubtractorMOG2()  # Initialize the background subtractor
initial_gray_frame = cv2.cvtColor(initial_frame, cv2.COLOR_BGR2GRAY)  # Convert the frame to grayscale

frame_count = 0
total_frames = 0
water_levels = []  # List to store water levels
frame_indices = []  # List to store frame indices
frame_index = 1
max_height = 0

output_video_name = 'project5_final_video'
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Define the codec for output video
output_video = cv2.VideoWriter(str(output_video_name) + ".mp4", fourcc, 30, (960, 640))  # Create VideoWriter object

while video_capture.isOpened():  # Loop through frames
    ret, current_frame = video_capture.read()
    if not ret:
        break
    
    warped_frame = cv2.warpPerspective(current_frame, homography_matrix, (num_rows, num_columns + 200))  # Warp the frame
    current_gray_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

    water_level = 0

    # Compute the dense optical flow using the Farneback method
    flow = cv2.calcOpticalFlowFarneback(initial_gray_frame, current_gray_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Calculate the magnitude and angle of the flow vectors
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Threshold the magnitude to get the regions with significant motion
    magnitude_threshold = np.where(magnitude > 1, 255, 0).astype(np.uint8)

    # Find contours in the thresholded magnitude image
    contours, _ = cv2.findContours(magnitude_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the contour with the maximum area
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)  # Get bounding rectangle
        cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw bounding rectangle
        # print('Water level frame (pixels):', y)
        water_level = y

    # Visualize the flow
    hsv_representation = np.zeros_like(initial_frame)
    hsv_representation[..., 1] = 255
    hsv_representation[..., 0] = angle * 180 / np.pi / 2
    hsv_representation[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb_flow = cv2.cvtColor(hsv_representation, cv2.COLOR_HSV2BGR)  # Convert HSV to BGR for display
    resized_rgb_flow = resize_image_for_viewing(rgb_flow, 0.5)  # Resize for viewing

    # Update the previous frame
    initial_gray_frame = current_gray_frame.copy()

    _, binary_thresh_frame = cv2.threshold(current_gray_frame, 215, 255, cv2.THRESH_BINARY)  # Threshold the frame
    water_level_height = water_level  # Update water level height

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 21, -1], [-1, -1, -1]])  # Define sharpening kernel
    sharpened_frame = cv2.filter2D(binary_thresh_frame, -1, sharpening_kernel)  # Apply sharpening filter
    inverted_frame = 255 - sharpened_frame  # Invert the frame
    detected_digits = pytesseract.image_to_string(inverted_frame, lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=0124689M')  # OCR
    bounding_boxes = pytesseract.image_to_boxes(inverted_frame, lang='eng', config='--psm 11 --oem 3 -c tessedit_char_whitelist=0124689M')  # OCR bounding boxes
    extracted_digit_values = re.findall("\w+", detected_digits)  # Extract digits from OCR result
    extracted_M_values = re.findall("\d+M", detected_digits)  # Extract M values from OCR result

    m_locations = []
    for i in range(len(bounding_boxes)):
        if bounding_boxes[i] == 'M':
            m_locations.append(i)
    gray_hull_frame = cv2.cvtColor(warped_frame, cv2.COLOR_BGR2GRAY)
    _, hull_thresh = cv2.threshold(gray_hull_frame, 200, 255, cv2.THRESH_BINARY)
    hull_contours, hull_hierarchy = cv2.findContours(hull_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    valid_contours = []
    draft_marks = []
    for contour in hull_contours:
        bounding_rect = cv2.boundingRect(contour)
        x, y, w, h = bounding_rect
        contour_area = cv2.contourArea(contour)
        if 50 < contour_area < 900 and 0.8 * w > h and 750 < x < 1150:
            valid_contours.append(contour)
            point = (x, y)
            draft_marks.append(point)
    total_difference = 0
    for i in range(len(draft_marks) - 1):
        difference = draft_marks[i][1] - draft_marks[i + 1][1]
        total_difference = difference + total_difference
    average_difference = total_difference / len(draft_marks)
    pixels_to_cm = average_difference / 10  # Convert pixel difference to centimeters
    if not m_locations:
        level = (int(x), int(y))
        if water_level_height:
            estimated_level = estimate_water_level(draft_marks, water_level_height, pixels_to_cm)
            print("The estimated water level is %.2f meters" % estimated_level)
            water_levels.append(estimated_level)
            frame_indices.append(frame_index)
            frame_index += 1
    if len(extracted_M_values) >= 1:
        last_M_value = extracted_M_values[-1]
        leading_digit = int(last_M_value.replace('M', ''))
        leading_digit -= 1
        trailing_digit = extracted_digit_values[-1]
        if 'M' in trailing_digit:
            trailing_digit = 0
            leading_digit += 1
        combined_digits = str(leading_digit) + "." + str(trailing_digit)
        estimated_level = float(combined_digits)
        print(f"Water level : {estimated_level} meters")
        water_levels.append(estimated_level)
        frame_indices.append(frame_index)
        frame_index += 1
    total_frames += 1
    resized_warped_frame = resize_image_for_viewing(warped_frame, 0.5)
    water_level_text = "Water level in m: " + str(estimated_level)
    cv2.putText(resized_warped_frame, water_level_text, (30, 40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)  # Add text
    cv2.imshow('Water Level Detection', resized_warped_frame)
    cv2.imshow('Optical Flow', resized_rgb_flow)  # Display the optical flow visualization
    output_video.write(resized_warped_frame)
    if cv2.waitKey(25) & 0xFF == ord('e'):
        break

# Post-process water levels to remove outliers
water_levels_array = np.array(water_levels)
mean_level = np.mean(water_levels_array)
std_dev_level = np.std(water_levels_array)
lower_bound = mean_level - 2 * std_dev_level
upper_bound = mean_level + 2 * std_dev_level

final_water_levels = []
for level in water_levels:
    if level < lower_bound or level > upper_bound:
        frame_indices.pop()
    else:
        final_water_levels.append(level)

average_water_level = sum(final_water_levels) / len(final_water_levels)
print("Water level : %.2f meters" % average_water_level)

output_video.release()
video_capture.release()
cv2.destroyAllWindows()

# Plot the water level over time
plt.figure()
plt.title('Average Waterline Height vs. Time')
plt.ylabel('Average Waterline (m)')
plt.xlabel('Frame Number')
plt.scatter(frame_indices, final_water_levels, color='red', label='Average Frame Waterline')
plt.ylim(6, 12)
plt.legend()
plt.show()
