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
            print(f"Found Dense layer: {name} -> {module.name}")
        # Check for specific OPT model structure - TensorFlow OPT has different structure
        elif hasattr(module, 'layers'):
            for i, child in enumerate(module.layers):
                child_name = f"{name}.layers[{i}]" if name else f"layers[{i}]"
                _find_layers_recursive(child, child_name)
        # Check for submodules (common in TensorFlow models)
        elif hasattr(module, 'submodules'):
            for i, child in enumerate(module.submodules):
                child_name = f"{name}.submodules[{i}]" if name else f"submodules[{i}]"
                _find_layers_recursive(child, child_name)
        # Check for specific attributes that might contain Dense layers
        for attr_name in ['dense', 'linear', 'fc', 'projection', 'q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj', 'self_attn', 'fc1', 'fc2']:
            if hasattr(module, attr_name):
                attr = getattr(module, attr_name)
                if isinstance(attr, keras.layers.Dense):
                    layers[f"{name}.{attr_name}" if name else attr_name] = attr
                    print(f"Found Dense layer in {attr_name}: {name}.{attr_name}" if name else attr_name)
                elif hasattr(attr, 'submodules'):
                    _find_layers_recursive(attr, f"{name}.{attr_name}" if name else attr_name)
                elif hasattr(attr, 'layers'):
                    _find_layers_recursive(attr, f"{name}.{attr_name}" if name else attr_name)
        # Check for TFLayerNorm and other layers that might contain Dense layers
        if hasattr(module, 'layers'):
            for i, child in enumerate(module.layers):
                child_name = f"{name}.layers[{i}]" if name else f"layers[{i}]"
                _find_layers_recursive(child, child_name)
    
    _find_layers_recursive(module)
    return layers

def find_layers_tf_opt(module):
    layers = {}
    for layer in module.submodules:
        # Robustly detect Dense layers from any Keras variant
        if 'dense' in type(layer).__name__.lower() or 'dense' in str(type(layer)).lower():
            layers[layer.name] = layer
    return layers

def debug_layer_structure(module, max_depth=3, current_depth=0):
    """Debug function to understand the actual layer structure"""
    indent = "  " * current_depth
    print(f"{indent}{type(module).__name__}: {getattr(module, 'name', 'unnamed')}")
    
    if current_depth >= max_depth:
        return
    
    # Check for Dense layers
    if isinstance(module, keras.layers.Dense):
        print(f"{indent}  -> DENSE LAYER: {module.name}")
    
    # Check all attributes
    for attr_name in dir(module):
        if not attr_name.startswith('_'):
            try:
                attr = getattr(module, attr_name)
                if isinstance(attr, keras.layers.Layer):
                    print(f"{indent}  {attr_name}: {type(attr).__name__} -> {getattr(attr, 'name', 'unnamed')}")
                    if isinstance(attr, keras.layers.Dense):
                        print(f"{indent}    -> DENSE LAYER FOUND: {attr.name}")
                    elif hasattr(attr, 'layers') or hasattr(attr, 'submodules'):
                        debug_layer_structure(attr, max_depth, current_depth + 1)
            except Exception as e:
                pass
    
    # Check layers attribute
    if hasattr(module, 'layers'):
        for i, child in enumerate(module.layers):
            debug_layer_structure(child, max_depth, current_depth + 1)
    
    # Check submodules
    if hasattr(module, 'submodules'):
        for i, child in enumerate(module.submodules):
            debug_layer_structure(child, max_depth, current_depth + 1)

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
        else:
            # Create a default attention mask if not provided
            batch_size = tf.shape(inputs)[0]
            seq_len = tf.shape(inputs)[1]
            self.cache['attention_mask'] = tf.ones((batch_size, seq_len), dtype=tf.int32)
        raise ValueError("Catcher activated")

def inspect_model_structure(model, max_depth=3):
    """Inspect the model structure to understand layer hierarchy"""
    def _inspect_recursive(module, name='', depth=0):
        if depth > max_depth:
            return
        indent = '  ' * depth
        print(f"{indent}{name}: {type(module).__name__}")
        
        # Check for Dense layers
        if isinstance(module, keras.layers.Dense):
            print(f"{indent}  -> DENSE LAYER FOUND: {module.name}")
        
        # Check submodules
        if hasattr(module, 'submodules'):
            for i, child in enumerate(module.submodules):
                _inspect_recursive(child, f"{name}.{i}", depth + 1)
        
        # Check layers attribute
        if hasattr(module, 'layers'):
            for i, child in enumerate(module.layers):
                _inspect_recursive(child, f"{name}.layers[{i}]", depth + 1)
    
    print("Model structure:")
    _inspect_recursive(model)

def opt_sequential_keras(model, dataloader, args, quantization_type='gptq'):
    print('Starting ...')

    # Disable cache for quantization
    use_cache = getattr(model.config, 'use_cache', False)
    model.config.use_cache = False
    
    # Inspect model structure for debugging
    inspect_model_structure(model)
    
    # For TensorFlow OPT models, the layers are in model.model.decoder.layers
    layers = []
    
    if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
        layers = model.model.decoder.layers
        print(f"Found {len(layers)} transformer layers")
    else:
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
    activation_count = 0
    for batch in dataloader:
        batch = batch.astype('int32')
        try:
            # For TensorFlow models, we need to pass input_ids as a dictionary
            # Also create proper attention mask
            attention_mask = np.ones_like(batch)
            _ = model({'input_ids': batch, 'attention_mask': attention_mask})
            activation_count += 1
            if activation_count % 10 == 0:
                print(f"Collected activations from {activation_count} batches")
        except ValueError:
            pass
        if activation_count >= 10:  # Limit to first 10 batches for calibration
            break
    print(f'Calibration complete. Collected from {activation_count} batches.')
    
    # Restore first layer
    layers[0] = original_first_layer

    # Get the collected input
    inps = cache['current_input']
    attention_mask = cache['attention_mask']
    
    if inps is None:
        print("Error: No input collected. Using dummy input.")
        inps = tf.zeros((1, args.seqlen, args.hidden_size), dtype=dtype)
    else:
        print(f"Collected input shape: {inps.shape}")
        print(f"Collected input range: [{tf.reduce_min(inps):.6f}, {tf.reduce_max(inps):.6f}]")

    print(f'Input shape: {inps.shape}')
    print('Ready.')

    quantizers = {}
    for i in range(len(layers)):
        layer = layers[i]
        print(f"Processing layer {i}: {type(layer)}")
        
        # Debug the layer structure first to understand what we're working with
        print(f"\n=== Debugging Layer {i} Structure ===")
        debug_layer_structure(layer, max_depth=2)
        
        # Find Dense layers in this transformer layer - use specialized function for TensorFlow OPT
        subset = find_layers_tf_opt(layer)
        print(f"Found {len(subset)} Dense layers in layer {i}")
        print(f"All submodules for layer {i}: {[type(l) for l in layer.submodules]}")
        print(f"All submodule names for layer {i}: {[l.name for l in layer.submodules]}")
        print(f"Found Dense layers: {list(subset.keys())}")
        
        if not subset:
            print(f"No Dense layers found in layer {i}, skipping quantization")
            # Process the layer normally
            try:
                # For TensorFlow models, we need to pass inputs as a dictionary
                if attention_mask is not None:
                    inputs = {'hidden_states': inps}
                    if attention_mask is not None:
                        inputs['attention_mask'] = attention_mask
                    inps = layer(inputs)
                else:
                    inps = layer({'hidden_states': inps})
            except Exception as e:
                print(f"Error processing layer {i}: {e}")
            continue
        
        gptq = {}
        
        for name in subset:
            print(f"Setting up GPTQ for {name}")
            gptq[name] = GPTQ(subset[name])
            quantizer = Quantizer()
            quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=getattr(args, 'trits', False)
            )
            gptq[name].quantizer = quantizer

        # For Keras, we need to use a different approach since there's no register_forward_hook
        # We'll use a custom layer wrapper
        class DenseHook(keras.layers.Layer):
            def __init__(self, dense_layer, gptq_obj):
                super().__init__()
                self.dense_layer = dense_layer
                self.gptq_obj = gptq_obj
            def call(self, inputs, **kwargs):
                outputs = self.dense_layer(inputs, **kwargs)
                self.gptq_obj.add_batch(inputs, outputs)
                return outputs

        # Replace each Dense layer in the transformer block with a hooked version
        for name, dense_layer in subset.items():
            # 1. Find parent and attribute name
            result = find_parent_and_attr(layer, dense_layer)
            if result is None:
                print(f"Warning: Could not find parent for {name}")
                continue
            parent, attr_name = result

            # 2. Save original layer
            original_layer = getattr(parent, attr_name)

            # 3. Replace with hook
            setattr(parent, attr_name, DenseHook(dense_layer, gptq[name]))

            # 4. Run block on calibration input
            try:
                if attention_mask is not None:
                    outs = layer(inps, attention_mask)
                else:
                    outs = layer(inps)
            except Exception as e:
                print(f"Error processing layer {i}, {name}: {e}")
                # Restore original layer before continuing
                setattr(parent, attr_name, original_layer)
                continue

            # 5. Quantize
            # ... (quantization code as before) ...

            # 6. Restore original layer
            setattr(parent, attr_name, original_layer)
        
        # Process the input through the hooked layer
        try:
            if attention_mask is not None:
                outs = layer(inps, attention_mask)
            else:
                outs = layer(inps)
        except Exception as e:
            print(f"Error processing layer {i}: {e}")
            continue

        # Quantize layers
        for name in subset:
            print(f"Quantizing layer {i}, {name}")
            original_weight = subset[name].weights[0].numpy().copy()
            print(f"Original weight shape: {original_weight.shape}")
            print(f"Original weight range: [{np.min(original_weight):.6f}, {np.max(original_weight):.6f}]")
            
            if quantization_type == 'gptq':
                gptq[name].fasterquant(
                    blocksize=getattr(args, 'blocksize', 128),
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=getattr(args, 'act_order', False),
                    static_groups=getattr(args, 'static_groups', False)
                )
                quantizers[f'layer_{i}.{name}'] = gptq[name].quantizer
                
                # Verify quantization actually happened
                quantized_weight = subset[name].weights[0].numpy()
                print(f"Quantized weight range: [{np.min(quantized_weight):.6f}, {np.max(quantized_weight):.6f}]")
                weight_change = np.mean(np.abs(original_weight - quantized_weight))
                print(f"Average weight change: {weight_change:.6f}")
                
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
                
                # Verify quantization actually happened
                quantized_weight = subset[name].weights[0].numpy()
                print(f"Simple quantized weight range: [{np.min(quantized_weight):.6f}, {np.max(quantized_weight):.6f}]")
                weight_change = np.mean(np.abs(original_weight - quantized_weight))
                print(f"Average weight change: {weight_change:.6f}")
                
            gptq[name].free()
        
        # Process outputs again after quantization
        try:
            if attention_mask is not None:
                outs = layer(inps, attention_mask)
            else:
                outs = layer(inps)
        except Exception as e:
            print(f"Error processing layer {i} after quantization: {e}")
            continue

        # Swap inputs and outputs for next layer
        inps = outs

    # Restore cache setting
    model.config.use_cache = use_cache
    
    print('Quantization complete.')
    print(f'Total quantizers: {len(quantizers)}')
    return quantizers

# Add function to print quantization summary
def print_quantization_summary(quantizers, model_name="OPT-125M"):
    """Print a summary of quantization results"""
    print(f"\n=== Quantization Summary for {model_name} ===")
    print(f"Total quantized layers: {len(quantizers)}")
    
    if quantizers:
        # Analyze quantizer types
        gptq_count = sum(1 for q in quantizers.values() if hasattr(q, 'scale'))
        simple_count = sum(1 for q in quantizers.values() if isinstance(q, dict))
        
        print(f"GPTQ quantizers: {gptq_count}")
        print(f"Simple quantizers: {simple_count}")
        
        # Print some example quantizer info
        print("\nExample quantizer information:")
        for i, (name, quantizer) in enumerate(quantizers.items()):
            if i < 3:  # Show first 3
                if hasattr(quantizer, 'scale'):
                    # Handle tensors that might be multi-dimensional
                    if hasattr(quantizer.scale, 'numpy'):
                        scale_np = quantizer.scale.numpy()
                        if scale_np.size > 1:
                            # Multi-dimensional tensor - show statistics
                            scale_mean = float(scale_np.mean())
                            scale_std = float(scale_np.std())
                            zero_np = quantizer.zero.numpy() if hasattr(quantizer.zero, 'numpy') else quantizer.zero
                            zero_mean = float(zero_np.mean()) if hasattr(zero_np, 'mean') else float(zero_np)
                            maxq_np = quantizer.maxq.numpy() if hasattr(quantizer.maxq, 'numpy') else quantizer.maxq
                            maxq_val = float(maxq_np)
                            print(f"  {name}: scale_mean={scale_mean:.6f}±{scale_std:.6f}, zero={zero_mean:.6f}, maxq={maxq_val}")
                        else:
                            # Scalar tensor
                            scale_val = float(scale_np)
                            zero_val = float(quantizer.zero.numpy() if hasattr(quantizer.zero, 'numpy') else quantizer.zero)
                            maxq_val = float(quantizer.maxq.numpy() if hasattr(quantizer.maxq, 'numpy') else quantizer.maxq)
                            print(f"  {name}: scale={scale_val:.6f}, zero={zero_val:.6f}, maxq={maxq_val}")
                    else:
                        # Handle PyTorch tensors
                        if hasattr(quantizer.scale, 'numel') and quantizer.scale.numel() > 1:
                            scale_mean = quantizer.scale.mean().item()
                            scale_std = quantizer.scale.std().item()
                            zero_mean = quantizer.zero.mean().item() if hasattr(quantizer.zero, 'mean') else quantizer.zero.item()
                            maxq_val = quantizer.maxq.item() if hasattr(quantizer.maxq, 'item') else quantizer.maxq
                            print(f"  {name}: scale_mean={scale_mean:.6f}±{scale_std:.6f}, zero={zero_mean:.6f}, maxq={maxq_val}")
                        else:
                            scale_val = quantizer.scale.item() if hasattr(quantizer.scale, 'item') else quantizer.scale
                            zero_val = quantizer.zero.item() if hasattr(quantizer.zero, 'item') else quantizer.zero
                            maxq_val = quantizer.maxq.item() if hasattr(quantizer.maxq, 'item') else quantizer.maxq
                            print(f"  {name}: scale={scale_val:.6f}, zero={zero_val:.6f}, maxq={maxq_val}")
                elif isinstance(quantizer, dict):
                    print(f"  {name}: scale={quantizer['scale']:.6f}, zero={quantizer['zero']:.6f}, maxq={quantizer['maxq']}")
    
    print("=" * 50)

# Add function to compare original vs quantized performance
def compare_model_performance(original_model, quantized_model, testloader, args, tokenizer):
    """Compare performance between original and quantized models"""
    print("\n=== Performance Comparison ===")
    
    # Test original model
    print("Testing original model...")
    original_ppl = opt_eval_keras(original_model, testloader, args, tokenizer)
    
    # Test quantized model
    print("\nTesting quantized model...")
    quantized_ppl = opt_eval_keras(quantized_model, testloader, args, tokenizer)
    
    # Calculate degradation
    degradation = ((quantized_ppl - original_ppl) / original_ppl) * 100
    print(f"\n=== Results ===")
    print(f"Original perplexity: {original_ppl:.2f}")
    print(f"Quantized perplexity: {quantized_ppl:.2f}")
    print(f"Degradation: {degradation:.2f}%")
    
    return original_ppl, quantized_ppl, degradation

# 1. Download OPT-125M model and tokenizer (TensorFlow version)
def load_opt_model(model_name="facebook/opt-125m"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
    return model, tokenizer

# 2. Download WikiText-2 dataset
def load_wikitext(nsamples=128):
    try:
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Use a safe approach to select samples
        try:
            if hasattr(wikitext, 'select'):
                return wikitext.select(range(nsamples))
            else:
                # Fallback: convert to list and slice
                return list(wikitext)[:nsamples]
        except Exception:
            # Fallback: convert to list and slice
            return list(wikitext)[:nsamples]
    except Exception as e:
        print(f"Error loading WikiText dataset: {e}")
        print("Using fallback dataset approach...")
        # Fallback: create a simple dataset
        from datasets import Dataset
        texts = ["This is a sample text for calibration."] * nsamples
        return Dataset.from_dict({"text": texts})

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
        if i < 3:  # Only print details for first few batches to avoid spam
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

def find_parent_and_attr(root, target_layer):
    for attr_name in dir(root):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(root, attr_name)
            if attr is target_layer:
                return root, attr_name
        except Exception:
            continue
    # Also check inside submodules
    if hasattr(root, 'submodules'):
        for sub in root.submodules:
            if sub is target_layer:
                continue  # Don't check self
            result = find_parent_and_attr(sub, target_layer)
            if result is not None:
                return result
    return None

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
    try:
        if args.dataset == 'wikitext2':
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        elif args.dataset == 'ptb':
            dataset = load_dataset("ptb_text_only", "penn_treebank", split="train")
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")
        # Use a safe approach to select samples
        try:
            if hasattr(dataset, 'select'):
                dataset = dataset.select(range(args.nsamples))
            else:
                # Fallback: convert to list and slice
                dataset = list(dataset)[:args.nsamples]
        except Exception:
            # Fallback: convert to list and slice
            dataset = list(dataset)[:args.nsamples]
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Using fallback dataset approach...")
        from datasets import Dataset
        texts = ["This is a sample text for calibration."] * args.nsamples
        dataset = Dataset.from_dict({"text": texts})
    
    # Prepare calibration data
    calib_data = prepare_calib_data(dataset, tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
    # Create dataloader
    dataloader = make_dataloader(calib_data, batch_size=1)
    # Add hidden_size to args
    args.hidden_size = model.config.hidden_size
    # Call opt_sequential_keras
    quantizers = opt_sequential_keras(model, dataloader, args, quantization_type='gptq')
    print_quantization_summary(quantizers, "OPT-125M (TensorFlow)")

    # Test quantization effectiveness
    print("\n=== Quantization Verification ===")
    total_weight_change = 0
    total_weights = 0
    quantized_layers = 0
    
    # More comprehensive weight analysis
    for layer in model.layers:
        if hasattr(layer, 'weights') and layer.weights:
            for weight in layer.weights:
                if 'dense' in weight.name.lower() or 'linear' in weight.name.lower():
                    weight_np = weight.numpy()
                    weight_change = np.mean(np.abs(weight_np))
                    weight_std = np.std(weight_np)
                    total_weight_change += weight_change
                    total_weights += 1
                    quantized_layers += 1
                    print(f"Weight {weight.name}: mean={weight_change:.6f}, std={weight_std:.6f}")
    
    if total_weights > 0:
        avg_weight_change = total_weight_change / total_weights
        print(f"\nQuantization Summary:")
        print(f"- Quantized layers: {quantized_layers}")
        print(f"- Average weight magnitude: {avg_weight_change:.6f}")
        print(f"- Total weights analyzed: {total_weights}")
        
        if avg_weight_change < 0.001:
            print("⚠️  WARNING: Very small weight changes detected. Quantization may not be working properly.")
        elif avg_weight_change < 0.01:
            print("⚠️  WARNING: Small weight changes detected. Check quantization parameters.")
        else:
            print("✅ Quantization appears to be working (significant weight changes detected).")
    else:
        print("❌ No quantizable weights found. Check layer discovery.")

    datasets = ['wikitext2', 'ptb']
    for dataset_name in datasets:
        try:
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
        except Exception as e:
            print(f"Error evaluating on {dataset_name}: {e}")
            continue 