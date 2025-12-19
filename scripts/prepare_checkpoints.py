import os
import sys
import urllib.request
from pathlib import Path

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Done.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

def prepare_hmr2(ckpt_dir):
    print("Preparing HMR2 models...")
    hmr2_target_dir = ckpt_dir / "hmr2_data"
    hmr2_target_dir.mkdir(exist_ok=True)
    
    # Default HMR2 cache location
    cache_dir = Path.home() / ".cache" / "4DHumans"
    
    # If we already have data in our target dir, just ensure the symlink exists
    if any(hmr2_target_dir.iterdir()):
        print(f"Found HMR2 models in {hmr2_target_dir}. Linking to {cache_dir}...")
        if cache_dir.is_symlink():
            os.unlink(cache_dir)
        elif cache_dir.exists():
            import shutil
            shutil.rmtree(cache_dir)
            
        cache_dir.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(hmr2_target_dir.absolute(), cache_dir)
        print("Symlink created.")
        return

    try:
        from hmr2.models import download_models
        # This downloads to ~/.cache/4DHumans by default
        print("Downloading HMR2 models (this may take a while)...")
        download_models()
        print("HMR2 models downloaded successfully.")
        
        # Move to persistent directory
        if cache_dir.exists() and not cache_dir.is_symlink():
            print(f"Moving models to {hmr2_target_dir}...")
            import shutil
            for item in cache_dir.iterdir():
                shutil.move(str(item), str(hmr2_target_dir))
            
            # Remove original dir and link
            shutil.rmtree(cache_dir)
            os.symlink(hmr2_target_dir.absolute(), cache_dir)
            print("Models moved and symlinked.")
            
    except ImportError:
        print("Could not import hmr2. Make sure you are running this inside the Docker container or have hmr2 installed.")
    except Exception as e:
        print(f"Error downloading HMR2 models: {e}")

def prepare_segformer(ckpt_dir):
    print("Preparing SegFormer models...")
    # We use the Hugging Face Hub to download the model
    # This will cache it in ~/.cache/huggingface/hub
    # We can also download it to a local directory if needed, but HF handles caching well.
    try:
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        model_name = "nvidia/segformer-b5-finetuned-cityscapes-1024-1024"
        print(f"Downloading {model_name}...")
        
        # Save to a local directory so we don't rely on the hidden ~/.cache
        local_model_dir = ckpt_dir / "segformer"
        
        processor = SegformerImageProcessor.from_pretrained(model_name)
        processor.save_pretrained(local_model_dir)
        
        # Force use_safetensors=True to avoid torch.load vulnerability error on older torch versions
        model = SegformerForSemanticSegmentation.from_pretrained(model_name, use_safetensors=True)
        model.save_pretrained(local_model_dir)
        
        print(f"SegFormer model saved to {local_model_dir}")
    except ImportError:
        print("Could not import transformers. Make sure you have it installed.")
    except Exception as e:
        print(f"Error downloading SegFormer model: {e}")

if __name__ == "__main__":
    print("Starting preparation of model checkpoints...")
    
    # Create checkpoints directory in the current working directory (or workspace root)
    ckpt_dir = Path("pedestrian_checkpoints")
    ckpt_dir.mkdir(exist_ok=True)
    
    prepare_hmr2(ckpt_dir)
    prepare_segformer(ckpt_dir)
    print("Preparation complete.")
