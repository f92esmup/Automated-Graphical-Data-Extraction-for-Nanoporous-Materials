import warnings
warnings.filterwarnings('ignore')

import sys
import os
import subprocess
import numpy as np
import Image_detection.utilities as utilities
import argparse
from tqdm import tqdm

# Add path to the system
sys.path.append('./Image_detection/ChartDete')
import google.generativeai as genai
from mmdet.apis import init_detector, inference_detector

sys.path.append('./Image_detection/Line_detection')
from Image_detection.Line_detection.utils import process_image, save_and_plot_data
from Image_detection.Line_detection.BB_Inference import LineInference
#from Image_detection.Line_detection.mmdetection.mmdet.apis import init_detector, inference_detector
from AI.Gemini.CEImage import GeminiImageProcessor

class ImageProcessor:
    def __init__(self, cd_config_path, cd_weights_path, lf_config_path, lf_weights_path, device, debug=False):
        self.device = device
        self.debug = debug
        self.gemini_processor = GeminiImageProcessor(debug=debug)
        
        # Load models once
        self.cd_model = init_detector(cd_config_path, cd_weights_path, device=device)
        self.lf_inference = LineInference(config=lf_config_path, ckpt=lf_weights_path, device=device)
        
        if debug: print("Models loaded successfully.")

    def process_image(self, image_path, output_path):
        try:
            # Run inference on the image
            result = inference_detector(self.cd_model, image_path)
            if self.debug: print(f"Inference completed for {image_path}")

            X_label_ROI, Y_label_ROI, plot_area_ROI, _, x_numbers_ROI, y_numbers_ROI, title_ROI, x_ticks_ROI, y_ticks_ROI, legend_points_ROI, legend_text_ROI, _, legend_area_ROI, _, _, y_area_ROI, x_area_ROI, _ = result

            # Filter ROIs by confidence and define colors for each group
            rois_dict = {
                'x_numbers': {'rois': utilities.filter_rois_by_confidence(x_numbers_ROI), 'color': (0, 255, 0)},
                'y_numbers': {'rois': utilities.filter_rois_by_confidence(y_numbers_ROI), 'color': (255, 0, 0)},
                'x_label': {'rois': utilities.filter_rois_by_confidence(X_label_ROI), 'color': (0, 0, 255)},
                'y_label': {'rois': utilities.filter_rois_by_confidence(Y_label_ROI), 'color': (255, 255, 0)},
                'title': {'rois': utilities.filter_rois_by_confidence(title_ROI), 'color': (255, 0, 255)},
                'x_ticks': {'rois': utilities.filter_rois_by_confidence(x_ticks_ROI), 'color': (0, 255, 255)},
                'y_ticks': {'rois': utilities.filter_rois_by_confidence(y_ticks_ROI), 'color': (128, 0, 128)},
                'legend_points': {'rois': utilities.filter_rois_by_confidence(legend_points_ROI), 'color': (128, 128, 0)},
                'legend_text': {'rois': utilities.filter_rois_by_confidence(legend_text_ROI), 'color': (0, 128, 128)},
                'legend_area': {'rois': utilities.filter_rois_by_confidence(legend_area_ROI), 'color': (128, 128, 128)},
                'y_area': {'rois': utilities.filter_rois_by_confidence(y_area_ROI), 'color': (64, 64, 64)},
                'x_area': {'rois': utilities.filter_rois_by_confidence(x_area_ROI), 'color': (192, 192, 192)},
            }

            result_json = self.gemini_processor.process_image_with_gemini(image_path)
            if self.debug: print(f"Processed image with Gemini for {image_path}")

            # Calculate midpoints for both x and y numbers
            x_midpoints = utilities.calculate_midpoints(x_numbers_ROI, axis='x')
            y_midpoints = utilities.calculate_midpoints(y_numbers_ROI, axis='y')
            
            # Sort midpoints
            y_midpoints.sort(reverse=True)
            x_midpoints.sort()

            # Extract x_numbers and y_numbers from JSON
            x_numb = result_json['x_numbers']
            y_numb = result_json['y_numbers']

            # Relate x_numbers and y_numbers with midpoints
            x_midpoints_dict = {x: float(midpoint) for x, midpoint in zip(x_numb, x_midpoints)}
            y_midpoints_dict = {y: float(midpoint) for y, midpoint in zip(y_numb, y_midpoints)}

            # Calculate scales and average scale for x and y axes
            x_scales, x_average_scale = utilities.calculate_scale(x_midpoints_dict)
            y_scales, y_average_scale = utilities.calculate_scale(y_midpoints_dict)

            x_origin, y_origin = utilities.get_zero_or_min_position(x_midpoints_dict, y_midpoints_dict)

            # Process the image and get the line data series
            line_dataseries = process_image(self.lf_inference, image_path, mask_kp_sample_interval=10, inter_type='linear', eliminate_duplicates=True)
            if self.debug: print(f"Line data series processed for {image_path}")

            # Rescale the line data series
            line_dataseries_escal = utilities.rescale_line_dataseries(x_average_scale, y_average_scale, x_origin, y_origin, x_midpoints_dict, y_midpoints_dict, line_dataseries)

            # Save and plot the data series
            save_and_plot_data(line_dataseries_escal, image_path)
            if self.debug: print(f"Data series saved and plotted for {image_path}")
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            pass

def run_image_processing(args):
    # Clear the console
    #subprocess.run("clear" if os.name == "posix" else "cls", shell=True)

    # Set the API key in the environment variables
    os.environ["GEMINI_API_KEY"] = 'AIzaSyDFuwrnPunjaEG5WlzjycQ75km-w2MFsgc'
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)

    # Initialize ImageProcessor
    processor = ImageProcessor(
        cd_config_path=args.cdconfig,
        cd_weights_path=args.cdweights,
        lf_config_path=args.lfconfig,
        lf_weights_path=args.lfweights,
        device=args.device,
        debug=args.debug
    )

    # Process each image in the input directory
    if args.debug: print("Starting image processing...")
    for image_name in tqdm(os.listdir(args.input_path), desc="Processing images"):
        image_path = os.path.join(args.input_path, image_name)
        output_path = os.path.join(args.output_path, f"axes_{image_name}")
        processor.process_image(image_path, output_path)
    if args.debug: print("Image processing completed.")

if __name__ == "__main__":
    # Argument parser configuration
    parser = argparse.ArgumentParser(description="Run inference on input images with a pretrained DETR model.")
    parser.add_argument('--cdweights', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth', type=str, help="Path to the model ChartDete weights file.")
    parser.add_argument('--lfweights', default='./Image_detection/weights/weights.pth', type=str, help="Path to the model LineFormer weights file.")
    parser.add_argument('--cdconfig', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py', type=str, help="Path to the model ChartDete configuration file.")
    parser.add_argument('--lfconfig', default='./Image_detection/Line_detection/config.py', type=str, help="Path to the model LineFormer configuration file.")
    parser.add_argument('--input_path', default='./data/DemoImages', type=str, help="Path to the directory containing input images.")
    parser.add_argument('--output_path', default='./data/Line_output', type=str, help="Path to the directory where output will be saved.")
    parser.add_argument('--device', default='cpu', type=str, help="Device to run the model on (cpu or cuda).")
    parser.add_argument('--debug', action='store_true', default=False, help="Enable debug mode to print debugging information.")
    args = parser.parse_args()

    run_image_processing(args)
