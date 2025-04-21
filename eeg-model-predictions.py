import os
import numpy as np
import torch
import pickle
from transformers import BartTokenizer
from model_decoding import BrainTranslator, BrainTranslatorNaive
from transformers import BartForConditionalGeneration
from data import normalize_1d, get_input_sample  # Import functions from data.py

def load_model(checkpoint_path, model_name, eeg_bands_count):
    """
    Load the trained BrainTranslator model from checkpoint
    """
    pretrained_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
    
    # Calculate input feature size (105 features per band × number of bands)
    in_feature = 105 * eeg_bands_count
    
    if model_name == 'BrainTranslator':
        model = BrainTranslator(
            pretrained_bart, 
            in_feature=in_feature, 
            decoder_embedding_size=1024, 
            additional_encoder_nhead=8, 
            additional_encoder_dim_feedforward=2048
        )
    elif model_name == 'BrainTranslatorNaive':
        model = BrainTranslatorNaive(
            pretrained_bart, 
            in_feature=in_feature, 
            decoder_embedding_size=1024, 
            additional_encoder_nhead=8, 
            additional_encoder_dim_feedforward=2048
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Load model weights
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu')))
    return model

def process_subject_data(dataset, subject, eeg_type, bands, tokenizer):
    """
    Process data for a specific subject and return a list of samples
    """
    samples = []
    for sample_idx, sent_obj in enumerate(dataset[subject]):
        input_sample = get_input_sample(sent_obj, tokenizer, eeg_type, bands)
        if input_sample is not None:
            samples.append({
                'eeg_data': input_sample['input_embeddings'],
                'sequence_length': input_sample['seq_len'],
                'input_mask': input_sample['input_attn_mask'],
                'input_mask_invert': input_sample['input_attn_mask_invert'],
                'target_ids': input_sample['target_ids'],  # Add target IDs
                'original_sentence': sent_obj['content']
            })
    return samples

def run_inference(model, tokenizer, sample, device, num_beams=5):
    """
    Run inference with the model to get text prediction from EEG data
    """
    model.eval()
    model = model.to(device)
    
    # Extract data from sample
    input_embeddings = sample['eeg_data'].unsqueeze(0).to(device).float()
    input_mask = sample['input_mask'].unsqueeze(0).to(device)
    input_mask_invert = sample['input_mask_invert'].unsqueeze(0).to(device)
    
    # Create a dummy target_ids tensor
    target_ids = sample['target_ids'].unsqueeze(0).to(device)
    target_ids_converted = target_ids.clone()
    target_ids_converted[target_ids == tokenizer.pad_token_id] = -100
    
    with torch.no_grad():
        # Generate prediction with the required parameters
        predictions = model.generate(
            input_embeddings, 
            input_masks_batch=input_mask, 
            input_masks_invert=input_mask_invert,
            target_ids_batch_converted=target_ids_converted,
            max_length=100,
            num_beams=num_beams
        )

        # Decode prediction
        predicted_text = tokenizer.decode(predictions.sequences[0], skip_special_tokens=True)
    
    return predicted_text

def main():
    # Configuration
    checkpoint_path =  "C:\\Users\\PC\\Desktop\\EEG-To-Text-main\\checkpoints\\decoding\\best\\task1_task2_finetune_BrainTranslator_skipstep1_b32_20_30_5e-05_5e-07_unique_sent.pt"
    model_name = "BrainTranslator"  # or "BrainTranslatorNaive"

    # EEG configuration
    subject = "ZAB"  # Example subject
    eeg_type = "GD"  # Options: "GD" (gaze duration), "FFD" (first fixation duration), "TRT" (total reading time)
    
    # Define frequency bands as in data.py
    bands = ['_t1', '_t2', '_a1', '_a2', '_b1', '_b2', '_g1', '_g2']  # theta, alpha, beta, gamma bands
    
    # Load sample EEG data for testing
    sample_path =  "C:\\Users\\PC\\Desktop\\EEG-To-Text-main\\datasets\\ZuCo\\task1-SR\\pickle\\task1-SR-dataset.pickle"
    with open(sample_path, 'rb') as handle:
        dataset = pickle.load(handle)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
    model = load_model(checkpoint_path, model_name, len(bands))
    
    # Process data for the subject - FIXED: added tokenizer parameter
    subject_samples = process_subject_data(dataset, subject, eeg_type, bands, tokenizer)
    
    # Print summary
    print(f"\nFound {len(subject_samples)} valid samples for subject {subject}")
    
    # Run inference on the first few samples
    num_samples_to_test = min(1, len(subject_samples))
    
    print("\n" + "="*50)
    print("EEG-To-Text Inference Results")
    print("="*50)
    print(f"Subject: {subject}")
    print(f"EEG type: {eeg_type}")
    print(f"EEG bands: {', '.join(bands)}")
    print(f"Model: {model_name}")
    print("-"*50)
    
    for i in range(num_samples_to_test):
        sample = subject_samples[i]
        try:
            predicted_text = run_inference(model, tokenizer, sample, device)
        except Exception as e:
            print(f"Error during inference on sample {i+1}: {e}")
            continue
        print(f"\nSample {i+1}:")
        print(f"Original sentence: {sample['original_sentence']}")
        print(f"Predicted text: {predicted_text}")
        print("-"*50)
    
    print("="*50)

if __name__ == "__main__":
    main()