import os
import json
import argparse
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoProcessor,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    PeftModel,
    prepare_model_for_kbit_training
)
from tqdm import tqdm

# Define a dataset for medical MRI data with pre-computed embeddings
class MedicalMRIEmbeddingDataset(Dataset):
    def __init__(self, json_file, processor, embedding_dir, max_length=2048):
        """
        Dataset for medical MRI data using pre-computed embeddings
        
        Args:
            json_file (str): Path to JSON file with data
            processor: Processor for Qwen2.5-VL
            embedding_dir (str): Directory containing pre-computed embeddings
            max_length (int): Maximum sequence length
        """
        self.data = []
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.processor = processor
        self.max_length = max_length
        self.embedding_dir = embedding_dir
        
        # Filter out examples without images
        self.data = [item for item in self.data if 'image' in item and item['image']]
        print(f"Loaded {len(self.data)} examples from {json_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get the image path and corresponding embedding path
        image_path = item['image']
        image_filename = os.path.basename(image_path).replace('.nii.gz', '.pt')
        image_embedding_path = os.path.join(self.embedding_dir, image_filename)
        if not os.path.exists(image_embedding_path):
            return self.__getitem__(np.random.randint(0, len(self.data)))
        # Load the pre-computed embedding
        image_embedding = torch.load(image_embedding_path, weights_only=True, map_location='cpu')
        # sample 2048 tokens from the embedding
        image_embedding_num = image_embedding.shape[0]
        selected_indices = np.linspace(0, image_embedding_num - 1, 4096*2, dtype=int)
        image_embedding = image_embedding[selected_indices]
        
        # Prepare input and output text
        prompt = item['prompt']
        answer = item.get('answer', '')  # Get answer if available, empty string otherwise
        
        # Prepare complete text with both prompt and answer
        if "instruction:" in prompt.lower():
            complete_text = prompt + "\n" + answer + self.processor.tokenizer.eos_token
        else:
            # Add a separator between prompt and answer if needed
            complete_text = prompt + "\nResponse: " + answer + self.processor.tokenizer.eos_token
        
        # Process the text inputs
        text_inputs = self.processor(
            text=complete_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )

        labels = text_inputs.input_ids[0].clone() 
        # find eos token in labels
        eos_token_index = (labels == self.processor.tokenizer.eos_token_id).nonzero()
        # all token after eos token should be set to pad token
        labels[eos_token_index+1:] = self.processor.tokenizer.pad_token_id


        # Create a dict with all the inputs
        inputs = {
            "input_ids": text_inputs.input_ids[0],
            "attention_mask": text_inputs.attention_mask[0],
            "labels": labels,
            "image_embedding": image_embedding,
        }
        
        return inputs

# Collate function to handle batch preparation
def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    
    # Collate image embeddings
    image_embeddings = [item["image_embedding"] for item in batch]
    
    # Create batch dictionary
    batch_dict = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "image_embeddings": image_embeddings,
    }
    
    return batch_dict

# Set up LoRA configuration for Qwen2.5-VL
def setup_lora_for_qwen(model, lora_rank, lora_alpha, lora_dropout):
    # Prepare the model for k-bit training if needed
    model = prepare_model_for_kbit_training(model)
    
    # Define the LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    
    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model

# Custom model forwarding to use pre-computed embeddings
class CustomQwen2_5_VLForConditionalGeneration(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, input_ids, attention_mask, labels, image_embeddings):
        # Get original model's embeddings
        inputs_embeds = self.model.model.model.embed_tokens(input_ids)
        
        # Find the positions of image tokens
        image_token_id = self.model.config.image_token_id
        batch_size, seq_length = input_ids.shape
        
        # Process each item in the batch
        for i in range(batch_size):
            # Find positions of image tokens in the current batch item
            mask = (input_ids[i] == image_token_id)
            if torch.any(mask):
                # Get the embedding for the current batch item
                image_embedding = image_embeddings[i]
                
                # Create mask for embedding replacement
                mask_expanded = mask.unsqueeze(-1).expand_as(inputs_embeds[i])
                
                # Replace embeddings at image token positions with pre-computed embeddings
                inputs_embeds[i] = inputs_embeds[i].masked_scatter(mask_expanded, image_embedding)
        
        # Forward pass without using the vision encoder
        outputs = self.model.model(
            input_ids=None,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        

        logits = outputs.logits
        
        loss = None
        if labels is not None:
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        return {"loss": loss, "logits": logits}

def save_model_with_tied_weights(model, output_dir):
    # Create a state dict without duplicate references
    state_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:  # Only save the trainable parameters for LoRA
            state_dict[name] = param.data.cpu()
    
    # Save using torch.save instead of safetensors
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state_dict, os.path.join(output_dir, "pytorch_model.bin"))
    
    # Save the model configuration
    if hasattr(model, 'config'):
        model.config.save_pretrained(output_dir)

def train_model(args):
    # Load model, processor, and tokenizer
    print("Loading model, processor, and tokenizer...")
    
    processor = AutoProcessor.from_pretrained(args.model_name_or_path)
    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() and args.bf16 else torch.float32,
        device_map="auto"
    )
    
    # Wrap the model to use our custom forward method
    model = CustomQwen2_5_VLForConditionalGeneration(base_model)
    
    # Apply LoRA to the model if requested
    if args.use_lora:
        print("Applying LoRA to the model")
        model.model = setup_lora_for_qwen(model.model, args.lora_rank, args.lora_alpha, args.lora_dropout)
    
    # Create datasets
    train_dataset = MedicalMRIEmbeddingDataset(
        args.train_file, 
        processor, 
        args.embedding_dir,
        max_length=args.max_length
    )
    
    eval_dataset = None
    # if args.eval_file:
    #     eval_dataset = MedicalMRIEmbeddingDataset(
    #         args.eval_file, 
    #         processor, 
    #         args.embedding_dir,
    #         max_length=args.max_length
    #     )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size if eval_dataset else None,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps if eval_dataset else None,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,  # Important for custom datasets
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_num_workers=args.num_workers,
        report_to=args.report_to,
        save_safetensors = False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    original_save = trainer.save_model
    
    def custom_save_model(output_dir, _internal_call=False):
        
        save_model_with_tied_weights(model, output_dir)
    
    trainer.save_model = custom_save_model
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the final model
    print(f"Saving model to {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_model(args.output_dir)
    print("Training complete!")

def main():
    # Set multiprocessing start method to 'spawn'
    import torch.multiprocessing as mp
    try:
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to spawn")
    except RuntimeError:
        print("Multiprocessing start method already set or couldn't be changed")
    
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL with pre-computed image embeddings")
    
    # Model and data arguments
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", 
                        help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--train_file", type=str, default="../../data/generate_data/train_balanced.json", 
                        help="Path to training data JSON file")
    parser.add_argument("--eval_file", type=str, default="../../data/generate_data/test_balanced.json", 
                        help="Path to evaluation data JSON file")
    parser.add_argument("--output_dir", type=str, default="./output/balanced", 
                        help="Directory to save the model")
    parser.add_argument("--embedding_dir", type=str, default="./data/pre_encoded_qwen25vl_3b_embeddings", 
                        help="Directory containing pre-computed embeddings")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=2, 
                        help="Batch size per device for training and evaluation")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, 
                        help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=1e-5, 
                        help="Initial learning rate")
    
    parser.add_argument("--weight_decay", type=float, default=0.01, 
                        help="Weight decay")
    parser.add_argument("--num_epochs", type=int, default=2, 
                        help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048, 
                        help="Maximum sequence length")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use FP16 precision")
    parser.add_argument("--bf16", action="store_true", 
                        help="Use BF16 precision")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    
    # Logging and saving arguments
    parser.add_argument("--logging_steps", type=int, default=5, 
                        help="Log every X steps")
    parser.add_argument("--save_steps", type=int, default=1000, 
                        help="Save checkpoint every X steps")
    parser.add_argument("--eval_steps", type=int, default=1000, 
                        help="Evaluate every X steps")
    parser.add_argument("--save_total_limit", type=int, default=5, 
                        help="Maximum number of saved checkpoints")
    parser.add_argument("--report_to", type=str, default="wandb", 
                        help="Report to (wandb, tensorboard, etc.)")
    
    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for fine-tuning")
    parser.add_argument("--lora_rank", type=int, default=16, 
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, 
                        help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, 
                        help="LoRA dropout rate")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Train the model
    train_model(args)

if __name__ == "__main__":
    main()