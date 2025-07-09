import argparse
import keras
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from gptqkeras import GPTQ
from quantkeras import Quantizer
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

def find_layers(module):
    # Recursively find all Dense layers in the module (equivalent to Linear layers in PyTorch)
    layers = {}
    def _find_layers_recursive(module, name=''):
        if isinstance(module, keras.layers.Dense):
            layers[name] = module
        for i, child in enumerate(module.submodules):
            child_name = f"{name}.{i}" if name else str(i)
            _find_layers_recursive(child, child_name)
    _find_layers_recursive(module)
    return layers

# ActivationCatcher for Keras (equivalent to Catcher in PyTorch)
class ActivationCatcher(keras.layers.Layer):
    def __init__(self, module, cache):
        super().__init__()
        self.module = module
        self.cache = cache
    def call(self, inputs, **kwargs):
        # Store the input directly in the cache
        self.cache['current_input'] = inputs
        if 'attention_mask' in kwargs:
            self.cache['attention_mask'] = kwargs['attention_mask']
        raise ValueError("Catcher activated")

def opt_sequential_keras(model, dataloader, args, quantization_type='gptq'):
    print('Starting ...')

    # Disable cache for quantization
    use_cache = getattr(model.config, 'use_cache', False)
    model.config.use_cache = False
    
    # For TensorFlow models, we need to find the transformer layers
    # For OPT models, the layers are in model.model.decoder.layers
    layers = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        layers = model.model.decoder.layers
    else:
        # Fallback: look for layers with attention mechanisms
        for layer in model.submodules:
            if hasattr(layer, 'attention') or hasattr(layer, 'self_attn') or hasattr(layer, 'multi_head_attention'):
                layers.append(layer)
    
    if not layers:
        print("Warning: Could not find transformer layers, using all submodules")
        layers = list(model.submodules)

    # Create input cache
    dtype = tf.float32  # Default dtype for TensorFlow
    cache = {'attention_mask': None, 'current_input': None}

    # Set up activation catcher for first layer
    original_first_layer = layers[0]
    layers[0] = ActivationCatcher(original_first_layer, cache)
    
    # Collect activations
    print('Calibrating on token IDs...')
    for batch in dataloader:
        batch = batch.astype('int32')
        try:
            _ = model(batch)
        except ValueError:
            pass
    print('Calibration complete.')
    
    # Restore first layer
    layers[0] = original_first_layer

    # Get the collected input
    inps = cache['current_input']
    attention_mask = cache['attention_mask']

    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)
        gptq = {}
        
        for name in subset:
            gptq[name] = GPTQ(subset[name])
            quantizer = Quantizer()
            quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=getattr(args, 'trits', False)
            )
            gptq[name].quantizer = quantizer

        # For Keras, we need to use a different approach since there's no register_forward_hook
        # We'll use a custom layer wrapper
        class HookLayer(keras.layers.Layer):
            def __init__(self, layer, gptq_dict):
                super().__init__()
                self.layer = layer
                self.gptq_dict = gptq_dict
            def call(self, inputs, **kwargs):
                outputs = self.layer(inputs, **kwargs)
                for name, gptq_obj in self.gptq_dict.items():
                    gptq_obj.add_batch(inputs, outputs)
                return outputs
        
        # Apply hooks
        hooked_layer = HookLayer(layer, gptq)
        
        # Process the input through the hooked layer
        try:
            outs = hooked_layer(inps, attention_mask=attention_mask)
        except Exception as e:
            print(f"Error processing layer {i}: {e}")
            continue

        # Quantize layers
        for name in subset:
            print(f"Layer {i}, {name}")
            print('Quantizing ...')
            if quantization_type == 'gptq':
                gptq[name].fasterquant(
                    blocksize=getattr(args, 'blocksize', 128),
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=getattr(args, 'act_order', False),
                    static_groups=getattr(args, 'static_groups', False)
                )
                quantizers[f'layer_{i}.{name}'] = gptq[name].quantizer
            elif quantization_type == 'simple':
                # Simple quantization: just round weights
                W = subset[name].weights[0].numpy()
                w_min = np.min(W)
                w_max = np.max(W)
                max_val = (2 ** args.wbits) - 1
                scale = (w_max - w_min) / max_val
                zero_point = w_min
                quantized = np.round((W - zero_point) / scale)
                quantized = np.clip(quantized, 0, max_val)
                dequantized = quantized.astype(np.float32) * scale + zero_point
                subset[name].weights[0].assign(dequantized)
                # Store quantization params for analysis
                quantizers[f'layer_{i}.{name}'] = {
                    'scale': scale,
                    'zero': zero_point,
                    'maxq': max_val
                }
            gptq[name].free()
        
        # Process outputs again after quantization
        try:
            outs = layer(inps, attention_mask=attention_mask)
        except Exception as e:
            print(f"Error processing layer {i} after quantization: {e}")
            continue

        # Swap inputs and outputs for next layer
        inps = outs

    # Restore cache setting
    model.config.use_cache = use_cache
    
    print('Quantization complete.')
    return quantizers

# 1. Download OPT-125M model and tokenizer (TensorFlow version)
def load_opt_model(model_name="facebook/opt-125m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
    return model, tokenizer

# 2. Download WikiText-2 dataset
def load_wikitext(nsamples=128):
    wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # Use a safe approach to select samples
    return wikitext.select(range(nsamples))

# 3. Prepare calibration data (tokenize and batch)
def prepare_calib_data(dataset, tokenizer, nsamples=128, seqlen=128):
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

# 4. Dataloader generator
def make_dataloader(encodings, batch_size=1):
    for i in range(0, encodings.shape[0], batch_size):
        yield encodings[i:i+batch_size]

# --- Evaluation loop, ported to Keras 3.0 ---
def opt_eval_keras(model, testloader, args, tokenizer=None):
    print('Evaluating ...')
    nsamples = 0
    nlls = []
    total_tokens = 0
    seqlen = args.seqlen
    pad_token_id = tokenizer.pad_token_id if tokenizer else 0

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
        print(f"Batch {i}: NLL = {nll:.2f}, tokens = {batch_tokens}")
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
    return ppl

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str, default="facebook/opt-125m", help='OPT model to load')
    parser.add_argument('--dataset', type=str, default='wikitext2', choices=['wikitext2', 'ptb'], help='Dataset for calibration/evaluation')
    parser.add_argument('--wbits', type=int, default=4, help='Number of bits for quantization')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples')
    parser.add_argument('--seqlen', type=int, default=128, help='Sequence length')
    parser.add_argument('--percdamp', type=float, default=0.01, help='Percent of average Hessian diagonal for dampening')
    parser.add_argument('--groupsize', type=int, default=-1, help='Groupsize for quantization')
    parser.add_argument('--sym', action='store_true', help='Symmetric quantization')
    parser.add_argument('--act_order', action='store_true', help='Activation order heuristic')
    parser.add_argument('--static_groups', action='store_true', help='Use static groups')
    parser.add_argument('--trits', action='store_true', help='Use trits for quantization')
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer = load_opt_model(args.model)
    # Load dataset
    if args.dataset == 'wikitext2':
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif args.dataset == 'ptb':
        dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    # Use a safe approach to select samples
    dataset = dataset.select(range(args.nsamples))
    # Prepare calibration data
    calib_data = prepare_calib_data(dataset, tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
    # Create dataloader
    dataloader = make_dataloader(calib_data, batch_size=1)
    # Add hidden_size to args
    args.hidden_size = model.config.hidden_size
    # Call opt_sequential_keras
    quantizers = opt_sequential_keras(model, dataloader, args, quantization_type='gptq')
    print("Quantization complete. Quantizers:", quantizers)

    datasets = ['wikitext2', 'ptb']
    for dataset_name in datasets:
        if dataset_name == 'wikitext2':
            testset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        elif dataset_name == 'ptb':
            testset = load_dataset("ptb_text_only", "penn_treebank", split="test")
        else:
            continue
        # testset = testset.select(range(100))  # or testset = testset[:100]
        test_data = prepare_calib_data(testset, tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
        testloader = make_dataloader(test_data, batch_size=8)
        print(dataset_name)
        opt_eval_keras(model, testloader, args, tokenizer) 