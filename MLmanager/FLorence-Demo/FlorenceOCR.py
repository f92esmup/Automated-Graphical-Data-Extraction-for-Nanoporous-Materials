import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import os

# Disable tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "microsoft/Florence-2-large-ft"
# Load the fine-tuned model and processor
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(DEVICE)
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

# Load and preprocess the image
image_path = "/Users/f92esmup/CICProject/AI/FLorence-Demo/image.png"
image = Image.open(image_path).convert("RGB")
prompt = "<OCR>"

# Prepare inputs with padding
inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(DEVICE)

# Generate text
generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=4096,
      num_beams=3
    )

# Decode the generated text
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

# Post-process the generated text
parsed_answer = processor.post_process_generation(generated_text, task=prompt, image_size=(image.width, image.height))

# Print the parsed answer
print(parsed_answer['<OCR>'])