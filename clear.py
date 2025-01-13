import os
import shutil

def clear_directory(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if item not in ['DemoImages', 'DemoPapers']:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)

if __name__ == "__main__":
    data_directory = os.path.join(os.path.dirname(__file__), 'data')
    clear_directory(data_directory)