import os
import json
import argparse
import torch
import numpy as np
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer
)
from peft import LoraConfig, TaskType, get_peft_model
import tqdm

# Custom model forwarding to use pre-computed embeddings
class CustomQwen2_5_VLForInference:
    def __init__(self, model_path, processor_path=None, checkpoint_path=None, device="cuda"):
        """
        Initialize the model for inference
        
        Args:
            model_path (str): Path to the base model
            processor_path (str, optional): Path to the processor. If None, will use model_path
            checkpoint_path (str, optional): Path to the fine-tuned LoRA checkpoint to load. If None, will use model_path
            device (str): Device to run inference on
        """
        self.device = device
        
        # Load the base model and processor
        if processor_path is None:
            processor_path = model_path
            
        print(f"Loading processor from {processor_path}")
        self.processor = AutoProcessor.from_pretrained(processor_path)
        self.tokenizer = AutoTokenizer.from_pretrained(processor_path)
        
        print(f"Loading base model from {model_path}")
        # Load the base model
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        
        # Create LoRA config with the same parameters used during training
        if checkpoint_path:
            print(f"Setting up LoRA configuration")
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=16,  # Same as training
                lora_alpha=32,  # Same as training
                lora_dropout=0.05,  # Same as training
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj",
                ],
            )
            
            # Apply LoRA to the model
            self.model = get_peft_model(self.model, lora_config)
            
            # Load LoRA weights directly from the checkpoint
            checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")
            print(f"Loading weights from {checkpoint_file}")
            state_dict = torch.load(checkpoint_file, map_location=self.device)
            # change name model.base_model.model to base_model.model
            state_dict = {k.replace("model.base_model.model", "base_model.model"): v for k, v in state_dict.items()}
            # Load only the LoRA weights into the model
            msg = self.model.load_state_dict(state_dict, strict=False)
            print(msg)
        
        self.model.eval()
        
        # Get image token ID
        self.image_token_id = self.model.config.image_token_id
        
    def prepare_inputs(self, prompt, image_embedding):
        """
        Prepare inputs for inference
        
        Args:
            prompt (str): The text prompt
            image_embedding (torch.Tensor): Pre-computed image embedding
        
        Returns:
            dict: Prepared inputs for the model
        """
        # Process the text inputs
        text_inputs = self.processor(
            text=prompt,
            return_tensors="pt",
        )
        
        input_ids = text_inputs.input_ids.to(self.device)
        attention_mask = text_inputs.attention_mask.to(self.device)
        
        # Get model's embeddings
        inputs_embeds = self.model.model.model.embed_tokens(input_ids)
        
        # Find positions of image tokens in the prompt
        mask = (input_ids[0] == self.image_token_id)
        
        if torch.any(mask):
            # Get the image embedding
            image_embedding = image_embedding.to(self.device)
            
            # Create mask for embedding replacement
            mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds[0])
            
            # Replace embeddings at image token positions with pre-computed embeddings
            inputs_embeds[0] = inputs_embeds[0].masked_scatter(mask_expanded, image_embedding)
        
        return {
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds
        }
    
    def generate(self, prompt, image_embedding, max_length=256):
        """
        Generate a response from the model
        
        Args:
            prompt (str): The text prompt
            image_embedding (torch.Tensor): Pre-computed image embedding
            max_length (int): Maximum length of the generated text
            temperature (float): Sampling temperature
            top_p (float): Top-p sampling parameter
            num_beams (int): Number of beams for beam search
        
        Returns:
            str: Generated text
        """
        # Prepare inputs
        inputs = self.prepare_inputs(prompt, image_embedding)
        
        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                inputs_embeds=inputs["inputs_embeds"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True
            )
        
        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Remove the prompt from the generated text
        prompt_tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_text = self.tokenizer.decode(prompt_tokens, skip_special_tokens=True)
        
        if generated_text.startswith(prompt_text):
            generated_text = generated_text[len(prompt_text):].strip()
        
        return generated_text

def load_image_embedding(embedding_path):
    """
    Load a pre-computed image embedding
    
    Args:
        embedding_path (str): Path to the pre-computed embedding
    
    Returns:
        torch.Tensor: The image embedding
    """
    image_embedding = torch.load(embedding_path, weights_only=True, map_location='cpu')
    
    # Sample tokens from the embedding as done in training
    image_embedding_num = image_embedding.shape[0]
    selected_indices = np.linspace(0, image_embedding_num - 1, 4096*2, dtype=int)
    image_embedding = image_embedding[selected_indices]
    
    return image_embedding

def main():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Qwen2.5-VL model")
    
    # Model and data arguments
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Path to the base model")
    parser.add_argument("--processor_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Path to the processor. If None, will use model_path")
    parser.add_argument("--checkpoint_path", type=str, default=None, 
                        help="Path to the directory containing fine-tuned checkpoint weights")
    parser.add_argument("--embedding_dir", type=str, required=True, 
                        help="Directory containing pre-computed embeddings")
    parser.add_argument("--input_file", type=str, required=True, 
                        help="Path to input data JSON file")
    parser.add_argument("--output_file", type=str, required=True, 
                        help="Path to output JSON file")
    
    # Generation arguments
    parser.add_argument("--index", type=int, default=1, 
                        help="Index of the item to generate")
    parser.add_argument("--max_length", type=int, default=256, 
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, 
                        help="Top-p sampling parameter")
    parser.add_argument("--num_beams", type=int, default=1, 
                        help="Number of beams for beam search")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run inference on")
    
    args = parser.parse_args()
    
    if args.processor_path is None:
        args.processor_path = args.model_path
    
    # Initialize the model
    model = CustomQwen2_5_VLForInference(
        model_path=args.model_path,
        processor_path=args.processor_path,
        checkpoint_path=args.checkpoint_path,
        device=args.device
    )
    
    # Load the input data
    with open(args.input_file, 'r') as f:
        input_data = json.load(f)
    
    len_input_data = len(input_data)//8
    input_data = input_data[args.index*len_input_data:(args.index+1)*len_input_data]
    args.output_file = args.output_file.replace('.json', f'_{args.index}.json')
    # Generate outputs
    outputs = []
    
    for item in tqdm.tqdm(input_data, desc="Generating"):
        # Get the image path and corresponding embedding path
        image_path = item.get('image', '')
        final_diagnosis = item.get('final_diagnosis', '')
        
        if not image_path:
            print(f"Warning: No image path found for item: {item}")
            continue
        
        image_filename = os.path.basename(image_path).replace('.nii.gz', '.pt')
        image_embedding_path = os.path.join(args.embedding_dir, image_filename)
        
        if not os.path.exists(image_embedding_path):
            print(f"Warning: Embedding not found at {image_embedding_path}")
            continue
        
        # Load the pre-computed embedding
        image_embedding = load_image_embedding(image_embedding_path)
        
        # Prepare the prompt
        prompt = item['prompt']
        
        # Generate text
        generated_text = model.generate(
            prompt=prompt,
            image_embedding=image_embedding,
            max_length=args.max_length
        )

        print('Predicted: ', generated_text.split('Final diagnosis:')[-1])
        print('Ground truth: ', final_diagnosis)
        
        # Save the results
        output_item = {
            'image': image_path,
            'prompt': prompt,
            'generated_text': generated_text,
            'final_diagnosis': final_diagnosis
        }
        
        outputs.append(output_item)
        
    
        # Save the outputs
        with open(args.output_file, 'w') as f:
            json.dump(outputs, f, indent=2)
        
        print(f"Results saved to {args.output_file}")

if __name__ == "__main__":
    main()