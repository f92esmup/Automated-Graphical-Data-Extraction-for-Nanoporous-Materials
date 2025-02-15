# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

# Image processing and system libraries
import cv2
import json
import numpy as np
import matplotlib.pyplot as plt

# Google Generative AI library
import google.generativeai as genai


def draw_and_save_rois(image_path, rois_dict, thickness=2):
    # Load the image
    image = cv2.imread(image_path)
    # Draw each ROI on the image with different colors for each group
    for group, rois in rois_dict.items():
        color = rois_dict[group]['color']
        for roi in rois['rois']:
            start_point = (int(roi[0]), int(roi[1]))
            end_point = (int(roi[2]), int(roi[3]))
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
    
    return image

def filter_rois_by_confidence(rois, threshold=0.9):
    filtered_rois = [roi[:-1] for roi in rois if roi[-1] > threshold]
    return filtered_rois

def calculate_midpoints(rois, axis='x'):
    # Filter ROIs by confidence
    filtered_rois = filter_rois_by_confidence(rois)

    # Calculate the midpoint for each ROI
    if axis == 'x':
        midpoints = [(roi[0] + roi[2]) / 2 for roi in filtered_rois]
    elif axis == 'y':
        midpoints = [(roi[1] + roi[3]) / 2 for roi in filtered_rois]
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    return midpoints

def draw_lines(image, midpoints, axis='x', thickness=2):
    # Set color based on axis
    if axis == 'x':
        color = (0, 255, 0)  # Green for vertical lines
    elif axis == 'y':
        color = (255, 0, 0)  # Blue for horizontal lines
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    # Draw lines at each midpoint
    for midpoint in midpoints:
        if axis == 'x':
            start_point = (int(midpoint), 0)
            end_point = (int(midpoint), image.shape[0])
        elif axis == 'y':
            start_point = (0, int(midpoint))
            end_point = (image.shape[1], int(midpoint))
        image = cv2.line(image, start_point, end_point, color, thickness)
        # Put the value of the number on the line
        if axis == 'x':
            position = (int(midpoint), 20)  # Position for x-axis numbers
            cv2.putText(image, str(midpoints.index(midpoint)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        elif axis == 'y':
            position = (20, int(midpoint))  # Position for y-axis numbers
            cv2.putText(image, str(midpoints.index(midpoint)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return image


def calculate_scale(midpoints_dict):
    # Extract the numbers and midpoints
    numbers = list(midpoints_dict.keys())
    midpoints = list(midpoints_dict.values())

    # Calculate the differences between consecutive numbers and midpoints
    number_diffs = np.diff(numbers)
    midpoint_diffs = np.diff(midpoints)

    # Calculate the scale as the ratio between the differences
    scales = midpoint_diffs / number_diffs

    # Calculate the average scale
    average_scale = np.mean(scales)

    return np.abs(scales), np.abs(average_scale)

def get_zero_or_min_position(x_midpoints_dict, y_midpoints_dict):
    def get_position(midpoints_dict):
        if 0 in midpoints_dict:
            return midpoints_dict[0]
        else:
            min_key = min(midpoints_dict.keys())
            return midpoints_dict[min_key]

    x_value = get_position(x_midpoints_dict)
    y_value = get_position(y_midpoints_dict)
    return x_value, y_value


def draw_lines(image, midpoints, axis='x', thickness=2, origin=None):
    # Set color based on axis
    if axis == 'x':
        color = (0, 255, 0)  # Green for vertical lines
    elif axis == 'y':
        color = (255, 0, 0)  # Blue for horizontal lines
    else:
        raise ValueError("Axis must be 'x' or 'y'")

    # Draw lines at each midpoint
    for midpoint in midpoints:
        line_color = (0, 0, 255) if midpoint == origin else color  # Red for origin line
        if axis == 'x':
            start_point = (int(midpoint), 0)
            end_point = (int(midpoint), image.shape[0])
        elif axis == 'y':
            start_point = (0, int(midpoint))
            end_point = (image.shape[1], int(midpoint))
        image = cv2.line(image, start_point, end_point, line_color, thickness)
        # Put the value of the number on the line
        if axis == 'x':
            position = (int(midpoint), 20)  # Position for x-axis numbers
            cv2.putText(image, str(midpoints.index(midpoint)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
        elif axis == 'y':
            position = (20, int(midpoint))  # Position for y-axis numbers
            cv2.putText(image, str(midpoints.index(midpoint)), position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1, cv2.LINE_AA)
    return image


def rescale_line_dataseries(x_average_scale, y_average_scale, x_origin, y_origin,x_midpoints_dict,y_midpoints_dict, line_dataseries):
    line_dataseries_scaled = []
    for series in line_dataseries:
        scaled_series = []
        for point in series:
            x, y = point
            
            
            # Rescale x and y values
            x_scaled = (x - x_origin) / x_average_scale
            y_scaled = (y_origin - y) / y_average_scale
            
            # Adjust for reference values if zero is not in the keys
            if 0 not in x_midpoints_dict.keys():
                x_scaled += min(x_midpoints_dict.keys())
            if 0 not in y_midpoints_dict.keys():
                y_scaled += min(y_midpoints_dict.keys())
            
            # Convert to negative if less than or equal to origin
            if x <= x_origin:
                x_scaled = -abs(x_scaled)
            if y >= y_origin:
                y_scaled = -abs(y_scaled)
            scaled_series.append([x_scaled, y_scaled])
        line_dataseries_scaled.append(scaled_series)
    return line_dataseries_scaled


