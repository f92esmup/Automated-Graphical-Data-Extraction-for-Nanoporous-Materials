import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import supervision as sv
import os
import time

# Set Hugging Face API key
#os.environ["HF_API_KEY"] = "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"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def inference_from_image(model_path, image_path):
    try:
        start_time = time.time()  # Start timing

        # Load the fine-tuned model and processor
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(DEVICE)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        # Load and preprocess the image
        image = Image.open(image_path)
        task = "<OD>"
        text = "<OD>"

        inputs = processor(text=text, images=image, return_tensors="pt").to(DEVICE)

        # Perform inference
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3
            )

        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        response = processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        detections = sv.Detections.from_lmm(sv.LMM.FLORENCE_2, response, resolution_wh=image.size)

        # Annotate and display the image
        bounding_box_annotator = sv.BoundingBoxAnnotator(color_lookup=sv.ColorLookup.INDEX)
        label_annotator = sv.LabelAnnotator(color_lookup=sv.ColorLookup.INDEX)

        image = bounding_box_annotator.annotate(image, detections)
        image = label_annotator.annotate(image, detections)
        image.thumbnail((600, 600))
        image.show()

        end_time = time.time()  # End timing
        print(f"Execution time: {end_time - start_time} seconds")

        return response
    except Exception as e:
        print(f"An error occurred during inference: {e}")
        return None

if __name__ == "__main__":
    start_time = time.time()  # Start timing

    # Example usage: Replace with your model path and image path
    model_path = "./FLorence/florence2-lora" # Replace with your model path
    image_path = "./FLorence/page3.png" # Replace with your image path
    inference_from_image(model_path, image_path)

    end_time = time.time()  # End timing
    print(f"Total execution time: {end_time - start_time} seconds")

