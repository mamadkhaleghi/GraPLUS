#!/usr/bin/env python3
import os
import sys
import shutil
import tarfile
import subprocess
import argparse
import gdown

def setup_opa_dataset(data_dir="dataset"):
    """
    Download, extract, and preprocess the OPA dataset.
    """
    opa_dir = os.path.join(data_dir, "OPA")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    # Google Drive file ID for OPA dataset
    file_id = "133Wic_nSqfrIajDnnxwvGzjVti-7Y6PF"
    opa_zip = "OPA_dataset.zip"
    
    print("Downloading OPA dataset...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, opa_zip, quiet=False)
    
    # Create temporary directory for extraction
    temp_dir = "temp_opa"
    os.makedirs(temp_dir, exist_ok=True)
    
    print("Extracting OPA dataset...")
    import zipfile
    with zipfile.ZipFile(opa_zip, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    
    # Find the actual OPA directory path within the extracted content
    # This handles potential subdirectories in the archive
    opa_extracted_dir = None
    for root, dirs, files in os.walk(temp_dir):
        if "background" in dirs and "foreground" in dirs and "composite" in dirs:
            opa_extracted_dir = root
            break
    
    if not opa_extracted_dir:
        print("Error: Could not find expected OPA directory structure in the extracted files.")
        return False
    
    # Create final OPA directory
    os.makedirs(opa_dir, exist_ok=True)
    
    # Move all files from extracted location to final location
    for item in os.listdir(opa_extracted_dir):
        src = os.path.join(opa_extracted_dir, item)
        dst = os.path.join(opa_dir, item)
        if os.path.exists(dst):
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            else:
                os.remove(dst)
        shutil.move(src, dst)
    
    # Clean up
    shutil.rmtree(temp_dir)
    os.remove(opa_zip)
    
    # Run preprocessing script
    print("Preprocessing OPA dataset...")
    subprocess.run(["python", "tool/preprocess.py", "--data_root", opa_dir])
    
    print("OPA dataset setup complete!")
    return True

def setup_scene_graphs(data_dir="dataset"):
    """
    Download, extract, and set up the scene graph data for GraPLUS.
    """
    sg_dir = os.path.join(data_dir, "OPA_SG")
    
    # Create directory for scene graph data
    os.makedirs(sg_dir, exist_ok=True)
    
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1xxxxxxxxxxxxx"
    sg_archive = "OPA_SG.tar.gz"
    
    print("Downloading scene graph data...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, sg_archive, quiet=False)
    
    print("Extracting scene graph files...")
    with tarfile.open(sg_archive) as tar:
        tar.extractall(path=sg_dir)
    
    print("Cleaning up...")
    os.remove(sg_archive)
    
    print("Scene graph setup complete!")
    print(f"Data is located at: {sg_dir}/sg_opa_background_n20")
    return True

def setup_gpt2_embeddings():
    """
    Download and setup the GPT-2 embeddings for GraPLUS.
    """
    embeddings_dir = "gpt2_embeddings"
    
    # Create directory
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Google Drive file ID (replace with your actual file ID)
    file_id = "1xxxxxxxxxxxxx"
    embeddings_zip = "gpt2_embeddings.zip"
    
    print("Downloading GPT-2 embeddings...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, embeddings_zip, quiet=False)
    
    print("Extracting embeddings...")
    import zipfile
    with zipfile.ZipFile(embeddings_zip, 'r') as zip_ref:
        zip_ref.extractall(embeddings_dir)
    
    print("Cleaning up...")
    os.remove(embeddings_zip)
    
    print("GPT-2 embeddings setup complete!")
    return True

def main():
    parser = argparse.ArgumentParser(description="Setup datasets for GraPLUS")
    parser.add_argument('--all', action='store_true', help='Setup all datasets and embeddings')
    parser.add_argument('--opa', action='store_true', help='Setup OPA dataset')
    parser.add_argument('--sg', action='store_true', help='Setup Scene Graph data')
    parser.add_argument('--gpt2', action='store_true', help='Setup GPT-2 embeddings')
    parser.add_argument('--data_dir', type=str, default="dataset", help='Base directory for datasets')
    
    args = parser.parse_args()
    
    # If no specific arguments provided, setup everything
    if not (args.all or args.opa or args.sg or args.gpt2):
        args.all = True
    
    if args.all or args.opa:
        setup_opa_dataset(args.data_dir)
    
    if args.all or args.sg:
        setup_scene_graphs(args.data_dir)
    
    if args.all or args.gpt2:
        setup_gpt2_embeddings()
    
    print("Setup complete! All data is ready for GraPLUS training.")

if __name__ == "__main__":
    main()
