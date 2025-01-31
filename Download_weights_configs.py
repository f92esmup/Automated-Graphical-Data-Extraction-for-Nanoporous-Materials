from huggingface_hub import snapshot_download, login
import os

# Define the repository and the download directory
repo_id = "pedro12g/endpoint"
download_dir = "weights"

# Check for Hugging Face authentication token
token = os.getenv('HUGGINGFACE_TOKEN')
if not token:
    print("No Hugging Face token found. Please log in.")
    login()

# Download the entire repository snapshot
snapshot_download(repo_id=repo_id, local_dir=download_dir, token=token)

# Function to sort files (example: sort by file name)
def sort_files(directory):
    files = os.listdir(directory)
    files.sort()
    return files

# Sort the downloaded files
sorted_files = sort_files(download_dir)
print("Sorted files:", sorted_files)
