import argparse
import keras
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import tensorflow as tf

def load_opt_model(model_name="facebook/opt-125m"):
    """Load the original OPT model without quantization"""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
    return model, tokenizer

def load_dataset_safe(dataset_name, split="train", nsamples=128):
    """Safely load dataset with fallback options"""
    try:
        if dataset_name == 'wikitext2':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        elif dataset_name == 'ptb':
            dataset = load_dataset("ptb_text_only", "penn_treebank", split=split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        # Use a safe approach to select samples
        try:
            if hasattr(dataset, 'select'):
                return dataset.select(range(nsamples))
            else:
                return list(dataset)[:nsamples]
        except Exception:
            return list(dataset)[:nsamples]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback dataset approach...")
        from datasets import Dataset
        texts = ["This is a sample text for evaluation."] * nsamples
        return Dataset.from_dict({"text": texts})

def prepare_calib_data(dataset, tokenizer, nsamples=128, seqlen=128):
    """Prepare calibration data (tokenize and batch)"""
    # Try 'text', then 'sentence', else raise error
    sample = dataset[0]
    if 'text' in sample:
        texts = [x['text'] for x in dataset]
    elif 'sentence' in sample:
        texts = [x['sentence'] for x in dataset]
    else:
        raise KeyError("Neither 'text' nor 'sentence' found in dataset sample keys.")
    encodings = tokenizer(texts, return_tensors="np", padding="max_length", truncation=True, max_length=seqlen)
    return encodings["input_ids"]

def make_dataloader(encodings, batch_size=8):
    """Create dataloader generator"""
    for i in range(0, encodings.shape[0], batch_size):
        yield encodings[i:i+batch_size]

def evaluate_original_model(model, testloader, args, tokenizer=None):
    """Evaluate the original model without quantization"""
    print('Evaluating original model...')
    nsamples = 0
    nlls = []
    total_tokens = 0
    seqlen = args.seqlen
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0
    
    # Add metrics tracking
    batch_losses = []
    batch_token_counts = []

    for i, batch in enumerate(testloader):
        print(f"Processing batch {i}")
        batch = np.array(batch)
        batch_size = batch.shape[0]
        nsamples += batch_size
        outputs = model(batch)
        
        # Extract logits tensor
        if hasattr(outputs, "logits"):
            logits_tensor = outputs.logits
        elif isinstance(outputs, (tuple, list)):
            logits_tensor = outputs[0]
        else:
            logits_tensor = outputs

        shift_logits = logits_tensor[:, :-1, :]
        shift_labels = batch[:, 1:]

        # Mask out padding tokens
        mask = (shift_labels != pad_token_id)
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_fn(shift_labels, shift_logits)  # shape: (batch, seqlen-1)
        loss = loss * mask  # zero out loss for padding tokens
        nll = np.sum(loss)
        nlls.append(nll)
        batch_tokens = np.sum(mask)
        total_tokens += batch_tokens
        
        # Store metrics for analysis
        batch_losses.append(nll)
        batch_token_counts.append(batch_tokens)
        
        print(f"Batch {i}: NLL = {nll:.2f}, tokens = {batch_tokens}")
        if i < 3:  # Only print details for first few batches
            print("First few shift_labels:", shift_labels[:2])
            print("First few mask values:", mask[:2])
        if np.isnan(loss).any():
            print("NaN detected in loss!")
    
    total_nll = np.sum(nlls)
    print(f"Total NLL: {total_nll}, Total tokens: {total_tokens}")
    if total_tokens == 0:
        print("No valid tokens to evaluate! Check your mask and data.")
        return float('inf')
    avg_loss = total_nll / total_tokens
    print(f"Average loss per token: {avg_loss}")
    if np.isnan(avg_loss):
        print("NaN detected in average loss!")
    ppl = np.exp(avg_loss)
    print(f'Perplexity: {ppl:.2f}')
    
    # Additional metrics
    if len(batch_losses) > 1:
        avg_batch_loss = np.mean(batch_losses)
        std_batch_loss = np.std(batch_losses)
        print(f"Average batch loss: {avg_batch_loss:.2f} ± {std_batch_loss:.2f}")
        print(f"Loss range: [{np.min(batch_losses):.2f}, {np.max(batch_losses):.2f}]")
    
    return ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="facebook/opt-125m", help='OPT model to load')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb'], help='Dataset for evaluation')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of evaluation samples')
    parser.add_argument('--seqlen', type=int, default=128, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for evaluation')
    args = parser.parse_args()

    print(f"Loading original model: {args.model}")
    model, tokenizer = load_opt_model(args.model)
    
    print(f"Loading dataset: {args.dataset}")
    dataset = load_dataset_safe(args.dataset, split="test", nsamples=args.nsamples)
    
    print("Preparing evaluation data...")
    test_data = prepare_calib_data(dataset, tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
    testloader = make_dataloader(test_data, batch_size=args.batch_size)
    
    print(f"\n=== Evaluating Original Model ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Samples: {args.nsamples}")
    print(f"Sequence length: {args.seqlen}")
    print(f"Batch size: {args.batch_size}")
    
    # Evaluate original model
    original_ppl = evaluate_original_model(model, testloader, args, tokenizer)
    
    print(f"\n=== Final Results ===")
    print(f"Original model perplexity on {args.dataset}: {original_ppl:.2f}")
    
    # Model size information
    total_params = sum([np.prod(w.shape) for w in model.weights])
    print(f"Total parameters: {total_params:,}")
    print(f"Model size (estimated): {total_params * 4 / (1024**3):.2f} GB (FP32)") 