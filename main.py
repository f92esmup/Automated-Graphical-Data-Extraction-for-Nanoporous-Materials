from Search import run_search
from DataAssembler.Assembler import run_assembler
from image import run_image_processing
import argparse
import os
from PDFExraction.T_I_extraction_PDF import extract_text_from_pdf, extract_images_from_pdf

# Argument parser configuration
parser = argparse.ArgumentParser(description="Run various tasks.")
parser.add_argument('--cdweights', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/checkpoint.pth', type=str, help="Path to the model ChartDete weights file.")
parser.add_argument('--lfweights', default='./Image_detection/weights/weights.pth', type=str, help="Path to the model LineFormer weights file.")
parser.add_argument('--cdconfig', default='./Image_detection/weights/work_dirs/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss/cascade_rcnn_swin-t_fpn_LGF_VCE_PCE_coco_focalsmoothloss.py', type=str, help="Path to the model ChartDete configuration file.")
parser.add_argument('--lfconfig', default='./Image_detection/Line_detection/config.py', type=str, help="Path to the model LineFormer configuration file.")
parser.add_argument('--input_path', default='./data/images', type=str, help="Path to the directory containing input images.")
parser.add_argument('--output_path', default='./data/Line_output', type=str, help="Path to the directory where csv file will be saved.")
parser.add_argument('--device', default='cpu', type=str, help="Device to run the model on (cpu or cuda).")
parser.add_argument('--debug', action='store_true', default=False, help="Enable debug mode to print debugging information.")
parser.add_argument('--query', default='Nanoporous materials', type=str, help="Search query for papers.")
parser.add_argument('--scholar_results', default=10, type=int, help="Number of scholar results.")
parser.add_argument('--scholar_pages', default=1, type=int, help="Scholar pages to search.")
parser.add_argument('--dwn_dir', default='./data/papers/', type=str, help="Directory to download papers.")
parser.add_argument('--num_limit', default=5, type=int, help="Number limit for downloads.")
parser.add_argument('--description', default="The document should focus on the processes of liquid intrusion and extrusion in confined media, either from a theoretical or experimental perspective. It may include analysis of physical properties such as wettability, hydrophobicity, surface tension, and bubble nucleation. The document should also discuss technological applications such as energy storage, liquid separation, or chromatography, as well as implications for biological or bioinspired systems. Relevant theoretical models could include confined classical nucleation theories (cCNT), experimental methods such as liquid porosimetry or calorimetry, and atomistic or DFT-based simulations. Keywords should include terms like 'intrusion-extrusion', 'wetting-drying', 'hydrophobicity-lyophobicity', 'nucleation', and 'nanoporous materials.'", type=str, help="Description for the search.")
parser.add_argument('--papers_folder', default='./data/papers', type=str, help="Path to the directory containing PDF papers.")
parser.add_argument('--output_folder', default='./data', type=str, help="Path to the directory where papers will be downloaded.")
args = parser.parse_args()

# Prepare parameters for run_search
search_params = {
    "query": args.query,
    "scholar_results": args.scholar_results,
    "scholar_pages": '1'+'-' + str(args.scholar_pages),
    "dwn_dir": args.dwn_dir,
    "proxy_list": None,
    "min_date": None,
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
    "description": args.description,
    "eliminate_false_values": False
}

# Call the run_search function with parameters
#run_search(search_params)

# Call the extract_text_from_pdf and extract_images_from_pdf functions for each PDF in the papers folder
for pdf_filename in os.listdir(args.papers_folder):
    if pdf_filename.endswith('.pdf'):
        pdf_path = os.path.join(args.papers_folder, pdf_filename)
        extract_images_from_pdf(pdf_path, args.output_folder)
        extract_text_from_pdf(pdf_path, args.output_folder)

# Call the run_image_processing function for each directory in the input_path
for subdir in os.listdir(args.input_path):
    subdir_path = os.path.join(args.input_path, subdir)
    if os.path.isdir(subdir_path):
        args.input_path = subdir_path
        #run_image_processing(args)

# Call the run_assembler function when needed
#run_assembler()