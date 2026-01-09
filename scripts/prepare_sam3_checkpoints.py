import os
import sys
from huggingface_hub import snapshot_download, login

def prepare_sam3():
    print("Preparing SAM 3 models...")
    
    token = os.environ.get("HUGGING_FACE_TOKEN")
    if not token:
        print("Warning: HUGGING_FACE_TOKEN not set in environment.")
    else:
        print("Logging in to Hugging Face Hub...")
        login(token=token)

    try:
        # Download the model checkpoints from Hugging Face Hub
        # This will cache them in ~/.cache/huggingface/hub (which we mount)
        print("Downloading facebook/sam3 from Hugging Face Hub...")
        # Downloading the main model files. 
        # The library likely loads them from cache or requires a path.
        # If we just download to cache, the library should be able to find it if it uses standard HF methods,
        # or we can point to the cache snapshot.
        snapshot_download(repo_id="facebook/sam3") 
        print("SAM 3 models downloaded successfully to cache.")
        
    except Exception as e:
        print(f"Error downloading SAM 3 models: {e}")
        print("Please ensure your HUGGING_FACE_TOKEN has access to the gated repo facebook/sam3.")

if __name__ == "__main__":
    prepare_sam3()
