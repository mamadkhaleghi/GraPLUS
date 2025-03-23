import os
import sys
import torch
import shutil

'''
It will process each .pth file(except the file with the biggest epoch value) in the directory, 
remove the 'optimizer' key, and replace the reduced files in the directory.

Example usage:
python save_reduced_model.py /path/to/your/directory

'''


def save_reduced_model(input_path, output_path):
    """Remove 'optimizer' from the model and save the reduced model."""
    model_dict = torch.load(input_path)
    
    # Remove 'optimizer' keys if they exist
    keys_to_remove = ['optimizer']
    for key in keys_to_remove:
        if key in model_dict:
            del model_dict[key]
    
    # Save the reduced dictionary to a new .pth file
    torch.save(model_dict, output_path)
    print(f"Reduced model saved to {output_path}")


def process_directory(input_dir):
    """Reduce all .pth files in the directory except the one with the largest number."""
    
    # List all .pth files in the directory
    files = [f for f in os.listdir(input_dir) if f.endswith('.pth')]
    
    # Extract integer part of filenames (assuming they are integers like 1.pth, 2.pth, ...)
    files = [f for f in files if f[:-4].isdigit()]
    files = sorted(files, key=lambda x: int(x[:-4]))  # Sort by the integer part of the filename
    
    # If there are no valid .pth files, return
    if not files:
        print("No .pth files found in the directory.")
        return

    # Create a temporary directory to store reduced models
    temp_dir = os.path.join(input_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)

    # Process all files except the largest one
    largest_file = files[-1]  # The last file in the sorted list is the largest
    for file in files[:-1]:  # Exclude the largest file
        input_path = os.path.join(input_dir, file)
        output_path = os.path.join(temp_dir, file)
        
        # Apply the save_reduced_model function to reduce the model
        save_reduced_model(input_path, output_path)

    # Remove original .pth files (except the largest one)
    for file in files[:-1]:
        os.remove(os.path.join(input_dir, file))
    
    # Move the reduced models from the temporary directory to the main directory
    for file in os.listdir(temp_dir):
        shutil.move(os.path.join(temp_dir, file), os.path.join(input_dir, file))

    # Remove the temporary directory
    shutil.rmtree(temp_dir)

    print(f"All models reduced except for the largest one ({largest_file})")





if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python save_reduced_model.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    process_directory(directory)
