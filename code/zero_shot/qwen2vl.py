import os
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import time
import nibabel as nib
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

def load_nii_image(image_path):
    """Load a NIfTI image and return the numpy array and affine transformation."""
    nii_img = nib.load(image_path)
    data = nii_img.get_fdata()
    return data

def call_qwen_model(model,processor, question, image_path):
    nii_data = load_nii_image(image_path)
    
    # Create temporary directory for frames
    temp_dir = "temp_frames_zeroshot"
    os.makedirs(temp_dir, exist_ok=True)
    
    # # Select 32 slices from the nii_data (160 slices)
    # slice_indices = np.linspace(32, 128, 96, dtype=int)
    frame_paths = []
    
    for i in range(nii_data.shape[0]):
        # Get slice from the nii_data and convert to 8 bit png
        middle_slice = nii_data[i, :, :]
        # Clip to 0-400
        middle_slice = np.clip(middle_slice, 0, 400)
        middle_slice = (middle_slice - middle_slice.min()) / (middle_slice.max() - middle_slice.min()) * 255
        middle_slice = middle_slice.astype(np.uint8)
        # Save middle_slice as frame
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        Image.fromarray(middle_slice).save(frame_path)
        frame_paths.append(f"file://{os.path.abspath(frame_path)}")

    # Build the messages list with video format
    messages = [
        {
            "role": "system",
            "content": "You are a radiologist. You are given a question and a series of knee MRI images presented as a video. Please provide your answer in JSON format."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": frame_paths,  # List of frame paths as video
                    "fps": 4.0,  # Frame rate for temporal understanding
                },
                {
                    "type": "text", 
                    "text": f"{question}"
                }
            ]
        }
    ]

    # Prepare for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
        **video_kwargs,
    )
    
    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = inputs.to(device)
    model = model.to(device)

    # Generate response
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print(output_text)
    
    # Clean up temporary files
    for frame_path in frame_paths:
        file_path = frame_path.replace("file://", "")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove temp directory if empty
    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
        os.rmdir(temp_dir)
    
    # Since Qwen2.5-VL doesn't directly provide token counts, we need to estimate them
    total_tokens = len(inputs.input_ids[0]) + len(generated_ids_trimmed[0])
    completion_tokens = len(generated_ids_trimmed[0])
    prompt_tokens = len(inputs.input_ids[0])
    reasoning_tokens = 0  # Not available for this model

    try:
        content_json = json.loads(output_text.replace("```json", "").replace("```", ""))
    except json.JSONDecodeError:
        content_json = {"response": output_text}  # Wrap plain text in JSON if parsing fails
    
    return content_json, total_tokens, completion_tokens, prompt_tokens, reasoning_tokens

if __name__ == "__main__":
    # File paths
    input_csv_file = "../../data/split_files/test_split.csv"
    test_df = pd.read_csv(input_csv_file)
    patient_id_list = test_df['patient_id'].tolist()
    time_list = test_df['time'].tolist()
    study_folder_list = test_df['study_folder'].tolist()
    series_folder_list = test_df['series_folder'].tolist()
    knee_side_list = test_df['knee_side'].tolist()
    image_filepath_list = test_df['image_filepath'].tolist()
    label_filepath_list = test_df['label_filepath'].tolist()
    question_list = json.load(open('../../data/question_files/questions_nobbox_guided.json', "r"))
    save_file = '../result/zero_shot/results_qwen25vl_zeroshot.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # Read save_file if exists
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            results = json.load(f)
    else:
        results = {}


    """Call Qwen2.5-VL model with the given question and images as video"""
    # Load the model
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct", 
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2"  # If you have flash attention installed
    )
    
    # Load the processor
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    
    # Process each item
    for patient_id, time, study_folder, series_folder, knee_side, image_filepath, label_filepath in tqdm(zip(
        patient_id_list, time_list, study_folder_list, series_folder_list, knee_side_list, 
        image_filepath_list, label_filepath_list)):
        
        if image_filepath in results:
            continue
        save_key = f"{patient_id}_{time}_{study_folder}_{series_folder}_{knee_side}"
        
        for question_id in question_list.keys():
            if save_key + '_' + str(question_id) in results:
                continue
            else:
                results[save_key + '_' + str(question_id)] = {}

            question = question_list[question_id]
            # Call the model
            try:
                result_json, total_tokens, completion_tokens, prompt_tokens, reasoning_tokens = call_qwen_model(model, processor, question, image_filepath)
            except Exception as e:
                print(f"Error: {e}")
                result_json = {"error": str(e)}
                total_tokens = 0
                completion_tokens = 0
                prompt_tokens = 0
                reasoning_tokens = 0

            # Store the result
            results[save_key + '_' + str(question_id)] = {
                "question": question,
                "qwen25vl_result": result_json,
                "qwen25vl_total_tokens": total_tokens,
                "qwen25vl_completion_tokens": completion_tokens,
                "qwen25vl_prompt_tokens": prompt_tokens,
                "qwen25vl_reasoning_tokens": reasoning_tokens,
            }

            # Save the results after each item to prevent data loss
            with open(save_file, "w") as f:
                json.dump(results, f, indent=2)
            
    print(f"Results saved to {save_file}")