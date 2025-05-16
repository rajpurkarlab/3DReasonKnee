import os
import json
import base64
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from openai import AzureOpenAI
from tqdm import tqdm
import time
import nibabel as nib

def load_nii_image(image_path):
    """Load a NIfTI image and return the numpy array and affine transformation."""
    nii_img = nib.load(image_path)
    data = nii_img.get_fdata()
    return data

def encode_image(image_path):
    """Encode an image to base64 string"""
    # read image_path as 8 bit png and then encode to base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_azure_model(question, image_path):
    nii_data = load_nii_image(image_path)

    """Call Azure OpenAI's API with the given question, options, and images"""
    client = AzureOpenAI(
        azure_endpoint="https://azure-ai-dev.hms.edu",
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version="2024-12-01-preview"
    )
    

    # Build the messages list
    messages = [
        {
            "role": "system",
            "content": "You are a radiologist. You are given a question and a set of knee MRI images."
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"{question}"
                }
            ]
        }
    ]

    # select 32 slices from the nii_data(160 slices)
    slice_indices = np.linspace(32, 128, 32, dtype=int)
    for i in range(32):
        # get middle slice of the nii_data and convert to 8 bit png
        middle_slice = nii_data[slice_indices[i], :, :]
        # clip to 0-400
        middle_slice = np.clip(middle_slice, 0, 400)
        middle_slice = (middle_slice - nii_data.min()) / (nii_data.max() - nii_data.min()) * 255
        middle_slice = middle_slice.astype(np.uint8)
        # save middle_slice to tmp_guided.png
        Image.fromarray(middle_slice).save("tmp_guided.png")
        tmp_image_path = "tmp_guided.png"
        
        base64_image = encode_image(tmp_image_path)
        messages[1]["content"].append({
            "type": "image_url",
            "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
            }
        }
        )

    response = client.chat.completions.create(
        model="o1",
        messages=messages,
        response_format={"type": "json_object"},
    )
    print(response)
    
    try:
        total_tokens = response.usage.total_tokens
        completion_tokens = response.usage.completion_tokens
        prompt_tokens = response.usage.prompt_tokens
        reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    except Exception as e:
        print(f"Error: {e}")
        total_tokens = 0
        completion_tokens = 0
        prompt_tokens = 0
        reasoning_tokens = 0

    content = response.choices[0].message.content
    return content, total_tokens, completion_tokens, prompt_tokens, reasoning_tokens

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
    save_file = '../result/zero_shot/results_o1_zeroshot.json'
    os.makedirs(os.path.dirname(save_file), exist_ok=True)
    
    # read save_file if exists
    if os.path.exists(save_file):
        with open(save_file, "r") as f:
            results = json.load(f)
    else:
        results = {}
    
    # Process each item
    for patient_id, time, study_folder, series_folder, knee_side, image_filepath, label_filepath in tqdm(zip(patient_id_list, time_list, study_folder_list, series_folder_list, knee_side_list, image_filepath_list, label_filepath_list)):
        if image_filepath in results:
            continue
        save_key = f"{patient_id}_{time}_{study_folder}_{series_folder}_{knee_side}"
        
        for question_id in question_list.keys():
            if save_key + '_' + str(question_id) in results:
                continue
            else:
                results[save_key + '_' + str(question_id)] = {}
        
            # Call the model
            try:
                question = question_list[question_id]
                result, total_tokens, completion_tokens, prompt_tokens, reasoning_tokens = call_azure_model(question, image_filepath)
                result_json = json.loads(result)
            except Exception as e:
                print(f"Error: {e}")
                reasoning = f"Error: {str(e)}"
                answer = "Error"

            # Store the result
            results[save_key + '_' + str(question_id)] = {
                "question": question,
                "o1_result": result_json,
                "o1_total_tokens": total_tokens,
                "o1_completion_tokens": completion_tokens,
                "o1_prompt_tokens": prompt_tokens,
                "o1_reasoning_tokens": reasoning_tokens,
            }

            # Save the results after each item to prevent data loss
            with open(save_file, "w") as f:
                json.dump(results, f, indent=2)
            
    print(f"Results saved to {save_file}")