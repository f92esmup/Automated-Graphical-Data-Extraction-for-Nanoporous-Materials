import itertools  # Efficient functions for working with iterators in loops.
import numpy as np  # Fundamental library for scientific computing.
from scipy.interpolate import CubicSpline, interp1d  # Functions for data interpolation.
import os
import sys

# Configure PYTHONPATH to import `mmdet`
project_root = './Line_detection'
sys.path.append(os.path.join(project_root, 'mmdetection'))

# Imports from mmdet and utils
from mmdetection.mmdet.apis import inference_detector, init_detector
import utils  # Custom module with useful functions for working with lines in images.

class LineInference:
    """
    Class for performing inference and processing instance detection results in images.
    """

    def __init__(self, config, ckpt, device='cpu'):
        """
        Initialize the detection model with the configuration file and checkpoint.
        
        Parameters:
        - config: Path to the model configuration file.
        - ckpt: Path to the model checkpoint file.
        - device: Device on which the model will be run (e.g., 'cuda:0' or 'cpu').
        """
        # Declare the 'model' variable as global to access it outside the function
        self.model = init_detector(config, ckpt, device=device)

    def do_instance(self, img, score_thr=0.3):
        """
        Perform instance detection on the input image using the provided model.

        Parameters:
        - img: Input image in numpy array format.
        - score_thr: Score threshold for filtering detections.

        Returns:
        - inst_masks: Instance masks filtered according to the score threshold.
        """
        # Perform inference on the input image using the model
        result = inference_detector(self.model, img)
        
        # Process the detection results using the specified score threshold and return the filtered results
        return self.parse_result(result, score_thr)

    @staticmethod
    def parse_result(result, score_thresh=0.3):
        """
        Process detection results and filter masks according to the score threshold.

        Parameters:
        - result: Detection results returned by the model.
        - score_thresh: Score threshold for filtering masks.

        Returns:
        - inst_masks: Filtered instance masks.
        """
        # Assign the result to the variable linea_data
        linea_data = result

        # Extract bounding boxes and masks from the result
        bbox, masks = linea_data[0][0], linea_data[1][0]

        # Filter masks based on the score threshold
        # Only include masks where the score of the corresponding bounding box is greater than the score threshold
        inst_masks = list(itertools.compress(masks, ((bbox[:, 4] > score_thresh).tolist())))
        
        # Return the filtered instance masks
        return inst_masks

    @staticmethod
    def interpolate(line_ds, inter_type='linear'):
        """
        Interpolate a data series.

        Parameters:
        - line_ds: Predicted data series.
        - inter_type: Type of interpolation ('linear' or 'cubic_spline').

        Returns:
        - inter_line_ds: List of interpolation objects for each line in the mask.
        """
        # Initialize lists to store x and y coordinates
        x, y = [], []

        # Extract x and y coordinates from the data series
        for pt in line_ds:
            x.append(pt['x'])
            y.append(pt['y'])

        # Remove duplicates
        unique_x, unique_y = [], []
        for i in range(len(x)):
            if x.count(x[i]) == 1:
                unique_x.append(int(x[i]))
                unique_y.append(int(y[i]))

        # If there are fewer than 2 unique x coordinates, return the original data series
        if len(unique_x) < 2:
            return line_ds

        # Perform interpolation
        if inter_type == 'linear':
            # Linear interpolation
            inter = interp1d(unique_x, unique_y)
        elif inter_type == 'cubic_spline':
            # Cubic spline interpolation
            inter = CubicSpline(unique_x, unique_y)

        # Initialize the list to store the interpolated data series
        inter_line_ds = []
        x_min, x_max = min(unique_x), max(unique_x)

        # Generate interpolated points for each x in the range from x_min to x_max
        for x in range(x_min, x_max + 1):
            inter_line_ds.append({"x": x, "y": int(inter(x))})

        # Return the interpolated data series
        return inter_line_ds

    def get_dataseries(self, img, mask_kp_sample_interval=10,inter_type='linear' ,return_masks=False):
        """
        Extract line data series from a chart image.

        Parameters:
        - img: Chart image in numpy array format (3 channels).
        - mask_kp_sample_interval: Interval for sampling points from the predicted line mask to obtain the data series.
        - return_masks: If True, return the masks along with the data series.

        Returns:
        - pred_ds: Line data series in the specified format (list of lines, each as a list of points {x:, y:}).
        - inst_masks (optional): Instance masks if return_masks is True.
        """
        # Get instance masks using inference
        inst_masks = self.do_instance(img, score_thr=0.3)
        
        # Convert instance masks to uint8 format and multiply by 255
        inst_masks = [line_mask.astype(np.uint8) * 255 for line_mask in inst_masks]

        # Extract the line data series from the masks
        pred_ds = []
        for line_mask in inst_masks:
            # Get the x range of the line from the mask
            x_range = utils.get_xrange(line_mask)
            
            # Get keypoints from the line mask using the specified interval
            line_ds = utils.get_kp(line_mask, interval=mask_kp_sample_interval, x_range=x_range, get_num_lines=False, get_center=True)
            
            # Perform interpolation on the line data series
            line_ds = self.interpolate(line_ds, inter_type=inter_type)

            # Add the interpolated data series to the list of predicted data series
            pred_ds.append(line_ds)

        # Return the predicted data series and, optionally, the instance masks
        return (pred_ds, inst_masks) if return_masks else pred_ds
