from Search import run_search
from DataAssembler.Assembler import run_assembler
from image import ImageProcessor  # Import ImageProcessor class
import argparse
import os
import time  # Add import for time module
#from PDFExraction.T_I_extraction_PDF import extract_text_from_pdf, extract_images_from_pdf
from PDFExraction.Extract_image_from_PDF import ImageInference

# Argument parser configuration
parser = argparse.ArgumentParser(description="Run various tasks.")
parser.add_argument('--cdweights', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth', type=str, help="Path to the model ChartDete weights file.")
parser.add_argument('--lfweights', default='./Image_detection/weights/weights.pth', type=str, help="Path to the model LineFormer weights file.")
parser.add_argument('--cdconfig', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py', type=str, help="Path to the model ChartDete configuration file.")
parser.add_argument('--lfconfig', default='./Image_detection/Line_detection/config.py', type=str, help="Path to the model LineFormer configuration file.")
parser.add_argument('--input_dir', default='./data/papers', type=str, help="Path to the directory containing input files.")
parser.add_argument('--output_dir', default='./data/Line_output', type=str, help="Path to the directory where output files will be saved.")
parser.add_argument('--device', default='cpu', type=str, help="Device to run the model on (cpu or cuda).")
parser.add_argument('--debug', action='store_true', default=False, help="Enable debug mode to print debugging information.")
parser.add_argument('--query', default='Intrusion-Extrusion in nanoporous materials', type=str, help="Search query for papers.")
parser.add_argument('--scholar_results', default=10, type=int, help="Number of scholar results.")
parser.add_argument('--scholar_pages', default=1, type=int, help="Scholar pages to search.")
parser.add_argument('--num_limit', default=5, type=int, help="Number limit for downloads.")
parser.add_argument('--description', default="The document should focus on the processes of liquid intrusion and extrusion in confined media, either from a theoretical or experimental perspective. It may include analysis of physical properties such as wettability, hydrophobicity, surface tension, and bubble nucleation. The document should also discuss technological applications such as energy storage, liquid separation, or chromatography, as well as implications for biological or bioinspired systems. Relevant theoretical models could include confined classical nucleation theories (cCNT), experimental methods such as liquid porosimetry or calorimetry, and atomistic or DFT-based simulations. Keywords should include terms like 'intrusion-extrusion', 'wetting-drying', 'hydrophobicity-lyophobicity', 'nucleation', and 'nanoporous materials.'", type=str, help="Description for the search.")
parser.add_argument('--model_dir', default='./AI/FLorence-Demo/florence2-lora', type=str, help="Path to the model directory.")
args = parser.parse_args()

start_time = time.time()  # Start timing

# Prepare parameters for run_search
search_params = {
    "query": args.query,
    "scholar_results": args.scholar_results,
    "scholar_pages": '1'+'-' + str(args.scholar_pages),
    "dwn_dir": args.input_dir,
    "proxy_list": None,
    "min_date": None,
    "max_date": None,
    "num_limit": args.num_limit,
    "num_limit_type": None,
    "filter_jurnal_file": None,
    "restrict": 1,
    "DOIs": None,
    "SciHub_URL": None,
    "chrome_version": None,
    "cites": None,
    "use_doi_as_filename": False,
    "SciDB_URL": None,
    "skip_words": None,
    "single_proxy": None,
    "doi_file": None,
    #"description": args.description,
    "eliminate_false_values": False
}

# Call the run_search function with parameters
#run_search(search_params)

# @PDFExtraction

# Call the extract_text_from_pdf and extract_images_from_pdf functions for each PDF in the input directory
#for pdf_filename in os.listdir(args.input_dir):
#    if pdf_filename.endswith('.pdf'):
#        pdf_path = os.path.join(args.input_dir, pdf_filename)
#        extract_images_from_pdf(pdf_path, args.output_dir)
#        extract_text_from_pdf(pdf_path, args.output_dir)

# Initialize ImageInference
inference = ImageInference(args.model_dir, classification=True)

# Use ImageInference to process the PDF
inference.convert_pdf_to_images_and_infer(args.input_dir, args.output_dir)

# Initialize ImageProcessor
processor = ImageProcessor(
    cd_config_path=args.cdconfig,
    cd_weights_path=args.cdweights,
    lf_config_path=args.lfconfig,
    lf_weights_path=args.lfweights,
    device=args.device,
    debug=args.debug
)

# Run image processing
#processor.run_image_processing(args.input_dir, args.output_dir)

# Call the run_assembler function when needed
#run_assembler()

end_time = time.time()  # End timing
print(f"Total execution time: {end_time - start_time} seconds")  # Print the total execution time