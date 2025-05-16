import os
import numpy as np
import pandas as pd 
from PIL import Image
import torch
import nibabel as nib
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

def get_nii_embeddings(nii_file_path, model=None, processor=None, temp_dir="temp_nii_slices_2", 
                       slice_range=None, device=None):
    """
    Extract visual embeddings from a NII.GZ file using Qwen2.5-VL model.
    
    Args:
        nii_file_path (str): Path to the NII.GZ file
        model: Pre-loaded Qwen2.5-VL model (will load if None)
        processor: Pre-loaded processor (will load if None)
        temp_dir (str): Directory to store temporary image slices
        slice_range (tuple, optional): Range of slices to use (start, end, step)
        device (str, optional): Device to use for inference ('cuda' or 'cpu')
        
    Returns:
        torch.Tensor: Visual embeddings from the model
    """
    # Setup device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load model and processor if not provided
    if model is None:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct", 
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="flash_attention_2"
        )
    
    if processor is None:
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    
    # Load NII.GZ file
    nii_img = nib.load(nii_file_path)
    nii_data = nii_img.get_fdata()
    print(f"NII data shape: {nii_data.shape}")
    if nii_data.shape[0] == 160:
        print(f"NII data shape is 160, which is correct")
    else:
        nii_data = nii_data.transpose(2,0,1)
        print(f"NII data shape is not 160, which is incorrect, transposed to {nii_data.shape}")

    # Create temporary directory for frames
    os.makedirs(temp_dir, exist_ok=True)
    
    # Determine which slices to use
    if slice_range is None:
        # Use all slices in the first dimension
        slices = range(nii_data.shape[0])
    else:
        start, end, step = slice_range
        slices = range(start, min(end, nii_data.shape[0]), step)
    
    frame_paths = []
    for i in slices:
        # Extract slice and normalize to 8-bit range
        slice_data = nii_data[i, :, :]
        # Clip to reasonable HU values (adjust as needed)
        slice_data = np.clip(slice_data, 0, 400)
        # Normalize to 0-255
        slice_data = (slice_data - slice_data.min()) / (max(slice_data.max() - slice_data.min(), 1e-8)) * 255
        slice_data = slice_data.astype(np.uint8)
        
        # Save as image
        frame_path = os.path.join(temp_dir, f"slice_{i:04d}.png")
        Image.fromarray(slice_data).save(frame_path)
        frame_paths.append(f"file://{os.path.abspath(frame_path)}")
    
    # Create a simple message to extract embeddings (without any specific prompt)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,
                    "fps": 4.0,
                }
            ]
        }
    ]
    
    # Process the message
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    
    # Move to specified device
    inputs = {k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)}
    model = model.to(device)
    
    # Extract embeddings - we need a different approach to get embeddings
    with torch.no_grad():
        # Method 1: Use the model's forward pass but get hidden states directly
        # Instead of passing all inputs directly, we'll handle the visual inputs separately
        
        # First, get the visual embeddings directly from the model's visual component
        if 'pixel_values_videos' in inputs:
            # Process videos through the visual encoder
            pixel_values_videos = inputs.pop('pixel_values_videos')
            video_grid_thw = inputs.pop('video_grid_thw') if 'video_grid_thw' in inputs else None
            visual_embeddings = model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        
        # If we have regular hidden states, we can get them this way
        outputs = model.model(
            input_ids=inputs.get('input_ids'),
            attention_mask=inputs.get('attention_mask'),
            position_ids=inputs.get('position_ids', None),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True
        )
        
        # Get the hidden states from the last layer
        text_embeddings = outputs.last_hidden_state
    
    # Clean up temporary files
    for frame_path in frame_paths:
        file_path = frame_path.replace("file://", "")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove temp directory if empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    # Return either visual_embeddings or text_embeddings depending on what worked
    if 'visual_embeddings' in locals():
        return visual_embeddings
    else:
        return text_embeddings

# Example usage:
if __name__ == "__main__":
    train_csv_file = "../../data/split_files/test_split.csv"
    train_df = pd.read_csv(train_csv_file)
    # image_filepath
    image_filepaths = train_df['image_filepath'].tolist()

    # For better efficiency, load model and processor once
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"
    )
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    

    save_embeddings_dir = "./data/pre_encoded_qwen25vl_3b_embeddings"
    for nii_file_path in image_filepaths:
        save_embeddings_path = os.path.join(save_embeddings_dir, os.path.basename(nii_file_path).replace(".nii.gz", ".pt"))
        if os.path.exists(save_embeddings_path):
            print(f"Skipping {nii_file_path} because it already exists")
            continue
        try:
            embeddings = get_nii_embeddings(
                nii_file_path, 
                model=model, 
                processor=processor,
                slice_range=(0, 160, 1)
            )
            torch.save(embeddings, save_embeddings_path)
            print(f"Saved embeddings to {save_embeddings_path} {embeddings.shape}")
        except Exception as e:
            print(f"Error processing {nii_file_path}: {e}")
            continue
