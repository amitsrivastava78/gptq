import argparse
import keras
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from gptqkeras import GPTQ
from quantkeras import Quantizer
import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))

# Helper to robustly extract tensor from dicts

def get_tensor(x):
    # Helper to extract tensor from dicts
    if isinstance(x, dict):
        if 'hidden_states' in x:
            return get_tensor(x['hidden_states'])
        # Try common keys
        for k in ['output', 'outputs', 'last_hidden_state', 'logits']:
            if k in x:
                return get_tensor(x[k])
        # If dict has only one value, return it
        if len(x) == 1:
            return get_tensor(list(x.values())[0])
        return None
    return x

# ActivationCatcher for Keras (equivalent to Catcher in PyTorch)
class ActivationCatcher(keras.layers.Layer):
    # Class variable to store cache
    cache = {}
    
    def __init__(self, module):
        print('📌 ENTRY: ActivationCatcher.__init__')
        super().__init__()
        self.module = module
        print('📌 EXIT: ActivationCatcher.__init__')
    def call(self, inputs, **kwargs):
        print('📌 ENTRY: ActivationCatcher.call')
        print("ActivationCatcher triggered!")
        ActivationCatcher.cache['current_input'] = inputs
        print("Cache after assignment:", ActivationCatcher.cache)
        if 'attention_mask' in kwargs:
            ActivationCatcher.cache['attention_mask'] = kwargs['attention_mask']
        else:
            # Create a default attention mask if not provided
            # Use tf.shape(inputs) safely
            tensor_inp = get_tensor(inputs)
            if tensor_inp is not None:
                shape = tf.shape(tensor_inp)
                # Try to get static shape as tuple
                static_shape = tf.get_static_value(shape)
                if static_shape is not None and len(static_shape) >= 2:
                    batch_size = int(static_shape[0])
                    seq_len = int(static_shape[1])
                else:
                    batch_size = 1
                    seq_len = 1
            else:
                batch_size = 1
                seq_len = 1
            ActivationCatcher.cache['attention_mask'] = tf.ones((batch_size, seq_len), dtype=tf.int32)
        print('📌 EXIT: ActivationCatcher.call')
        raise ValueError("Catcher activated")

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
    print('📌 ENTRY: find_layers_tf_opt')
    layers = {}
    for layer in module.submodules:
        if 'dense' in type(layer).__name__.lower() or 'dense' in str(type(layer)).lower():
            layers[layer.name] = layer
    print(f'📌 EXIT: find_layers_tf_opt - found {len(layers)} layers')
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

# === Helper Class ===
class DenseHook(keras.layers.Layer):
    def __init__(self, dense_layer, gptq_obj):
        print('📌 ENTRY: DenseHook.__init__')
        super().__init__()
        self.dense_layer = dense_layer
        self.gptq_obj = gptq_obj
        self.called = False
        print('📌 EXIT: DenseHook.__init__')
    def call(self, inputs, **kwargs):
        print('📌 ENTRY: DenseHook.call')
        if self.called:
            return self.dense_layer(inputs, **kwargs)
        self.called = True
        print(f"[DenseHook] CALL: id={id(self)}, layer={self.dense_layer.name}")
        layer_name = self.dense_layer.name
        if inputs is None:
            print(f"[DenseHook] {self.dense_layer.name} received None as input, skipping.")
            return None
        # Always extract tensor from dicts
        inputs = get_tensor(inputs)
        if inputs is None:
            print(f"[DenseHook] {layer_name} inputs could not be extracted as tensor, skipping.")
            return None
        print(f"[DenseHook] {layer_name} input shape: {inputs.shape}")
        if layer_name in ['k_proj', 'q_proj', 'v_proj', 'out_proj']:
            outputs = self.dense_layer(inputs, **kwargs)
            outputs = get_tensor(outputs)
            if outputs is None:
                print(f"[DenseHook] {layer_name} outputs could not be extracted as tensor, skipping.")
                return None
            print(f"[DenseHook] {layer_name} output shape: {outputs.shape}")
            in_shape = inputs.shape
            flat_inputs = tf.reshape(inputs, [-1, in_shape[-1]])
            out_shape = outputs.shape
            flat_outputs = tf.reshape(outputs, [-1, out_shape[-1]])
            self.gptq_obj.add_batch(flat_inputs, flat_outputs)
        else:
            input_shape = inputs.shape
            rank = len(input_shape)
            if rank == 3:
                batch, seq, hidden = input_shape
                flat_inputs = tf.reshape(inputs, [-1, hidden])
                print(f"[DenseHook] {layer_name} flat_inputs shape: {flat_inputs.shape}")
                outputs = self.dense_layer(flat_inputs, **kwargs)
                outputs = get_tensor(outputs)
                if outputs is None:
                    print(f"[DenseHook] {layer_name} outputs could not be extracted as tensor, skipping.")
                    return None
                print(f"[DenseHook] Rank3 {layer_name} dense output shape: {outputs.shape}")
                out_shape = outputs.shape
                outputs = tf.reshape(outputs, [batch, seq, out_shape[-1]])
                print(f"[DenseHook] {layer_name} reshaped output shape: {outputs.shape}")
                self.gptq_obj.add_batch(flat_inputs, tf.reshape(outputs, [-1, out_shape[-1]]))
            elif rank == 2:
                outputs = self.dense_layer(inputs, **kwargs)
                outputs = get_tensor(outputs)
                if outputs is None:
                    print(f"[DenseHook] {layer_name} outputs could not be extracted as tensor, skipping.")
                    return None
                print(f"[DenseHook] Rank2 {layer_name} output shape: {outputs.shape}")
                out_shape = outputs.shape
                print("before call to add_batch")
                self.gptq_obj.add_batch(inputs, outputs)
            else:
                raise ValueError(f"DenseHook: Unexpected input rank {rank}, shape {input_shape}")
        # Final defensive check before returning
        if outputs is None:
            print(f"[DenseHook] {layer_name} final outputs is None, returning zeros tensor.")
            # Return a zero tensor with appropriate shape as fallback
            if hasattr(inputs, 'shape') and len(inputs.shape) == 2:
                return tf.zeros((inputs.shape[0], self.dense_layer.units), dtype=inputs.dtype)
            elif hasattr(inputs, 'shape') and len(inputs.shape) == 3:
                return tf.zeros((inputs.shape[0], inputs.shape[1], self.dense_layer.units), dtype=inputs.dtype)
            else:
                return None
        
        # Add defensive check before calling add_batch
        if hasattr(self.gptq_obj, 'H') and self.gptq_obj.H is not None:
            try:
                if layer_name in ['k_proj', 'q_proj', 'v_proj', 'out_proj']:
                    in_shape = inputs.shape
                    flat_inputs = tf.reshape(inputs, [-1, in_shape[-1]])
                    out_shape = outputs.shape
                    flat_outputs = tf.reshape(outputs, [-1, out_shape[-1]])
                    self.gptq_obj.add_batch(flat_inputs, flat_outputs)
                else:
                    input_shape = inputs.shape
                    rank = len(input_shape)
                    if rank == 3:
                        batch, seq, hidden = input_shape
                        flat_inputs = tf.reshape(inputs, [-1, hidden])
                        print(f"[DenseHook] {layer_name} flat_inputs shape: {flat_inputs.shape}")
                        out_shape = outputs.shape
                        outputs = tf.reshape(outputs, [batch, seq, out_shape[-1]])
                        print(f"[DenseHook] {layer_name} reshaped output shape: {outputs.shape}")
                        self.gptq_obj.add_batch(flat_inputs, tf.reshape(outputs, [-1, out_shape[-1]]))
                    elif rank == 2:
                        print("before call to add_batch")
                        self.gptq_obj.add_batch(inputs, outputs)
                    else:
                        raise ValueError(f"DenseHook: Unexpected input rank {rank}, shape {input_shape}")
            except Exception as e:
                print(f"[DenseHook] Error in add_batch for {layer_name}: {e}")
                # Continue without adding batch if there's an error
        else:
            print(f"[DenseHook] Skipping add_batch for {layer_name} - GPTQ object not properly initialized")
        
        print('📌 EXIT: DenseHook.call')
        return outputs

def reset_all_densehook_flags(module):
    """Recursively reset the .called flag on all DenseHook instances in the model."""
    print('📌 ENTRY: reset_all_densehook_flags')
    if hasattr(module, 'submodules'):
        for submodule in module.submodules:
            if isinstance(submodule, DenseHook):
                submodule.called = False
            reset_all_densehook_flags(submodule)
    print('📌 EXIT: reset_all_densehook_flags')

def opt_sequential_keras(model, dataloader, args, quantization_type='gptq'):
    """
    Quantize an OPT model in TensorFlow/Keras using GPTQ, with a single calibration phase.
    Steps:
      1. Patch layers for calibration
      2. Collect calibration input
      3. For each transformer block:
         a. Replace Dense layers with hooks
         b. Run calibration
         c. Restore original layers
         d. Quantize
      4. Remove all DenseHook instances from the model
    """
    print('🚀 ENTRY: opt_sequential_keras')
    print('Starting ...')
    print(f'[DEBUG] nsamples: {getattr(args, "nsamples", "unknown")}')

    # === 1. Patch model layers for calibration ===
    def patch_all_decoder_layers(model):
        print('📌 ENTRY: patch_all_decoder_layers')
        if hasattr(model, 'model') and hasattr(model.model, 'decoder') and hasattr(model.model.decoder, 'layers'):
            layers = model.model.decoder.layers
            print(f"Found {len(layers)} transformer layers")
        else:
            print("Warning: Could not find transformer layers, using all submodules")
            layers = list(model.submodules)
        for layer in layers:
            patch_decoder_layer(layer)
        print('📌 EXIT: patch_all_decoder_layers')
        return layers

    layers = patch_all_decoder_layers(model)

    # === 2. Collect calibration input ===
    def collect_calibration_input(model, dataloader, args, layers):
        print('📌 ENTRY: collect_calibration_input')
        ActivationCatcher.cache = {'attention_mask': None, 'current_input': None}
        original_first_layer = layers[0]
        layers[0] = ActivationCatcher(original_first_layer)
        for batch in dataloader:
            batch = batch.astype('int32')
            try:
                attention_mask = np.ones_like(batch)
                _ = model({'input_ids': batch, 'attention_mask': attention_mask})
            except ValueError:
                break  # Only need one batch for calibration
            break
        layers[0] = original_first_layer
        inps = ActivationCatcher.cache['current_input']
        attention_mask = ActivationCatcher.cache['attention_mask']
        if inps is None:
            print("Warning input after the calibration was ZERO")
            inps = tf.zeros((1, args.seqlen, args.hidden_size), dtype=tf.float32)
        print('📌 EXIT: collect_calibration_input')
        return inps, attention_mask

    inps, attention_mask = collect_calibration_input(model, dataloader, args, layers)

    # === 3. Quantize each transformer block ===
    quantizers = {}
    for i, layer in enumerate(layers):
        print(f"\n=== Quantizing Layer {i} ===")
        # a. Find Dense layers
        subset = find_layers_tf_opt(layer)
        if not subset:
            inps = run_layer(layer, inps, attention_mask)
            continue
        # b. Replace Dense layers with hooks
        gptq, hook_instances = setup_gptq_and_hooks(subset, args)
        replace_dense_with_hooks(layer, subset, hook_instances)
        if hasattr(layer, 'self_attn'):
            patch_attention_module(layer.self_attn)
        # Reset hook flags before calibration
        reset_all_densehook_flags(layer)
        # c. Run calibration
        inps = run_layer(layer, inps, attention_mask)
        # d. Restore original layers
        restore_dense_layers(layer, subset)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, '_original_call'):
            layer.self_attn.call = layer.self_attn._original_call
        # e. Quantize
        quantize_dense_layers(subset, gptq, quantizers, args, quantization_type)
        # Reset hook flags before post-quantization run (shouldn't matter, but for safety)
        reset_all_densehook_flags(layer)
        inps = run_layer(layer, inps, attention_mask)
    print('[DEBUG] Quantization complete.')
    print(f'Total quantizers: {len(quantizers)}')
    # Remove all DenseHook instances from the model
    remove_all_dense_hooks(model)
    print('🏁 EXIT: opt_sequential_keras')
    return quantizers

# === Helper Functions ===
def run_layer(layer, inps, attention_mask):
    print('📌 ENTRY: run_layer')
    _inps = get_tensor(inps)
    inputs = {'hidden_states': inps}
    if attention_mask is not None:
        inputs['attention_mask'] = attention_mask
    outs = layer(inputs)
    if isinstance(outs, (tuple, list)):
        result = outs[0]
    elif isinstance(outs, dict) and 'hidden_states' in outs:
        result = outs['hidden_states']
    else:
        result = outs
    print('📌 EXIT: run_layer')
    return result

def setup_gptq_and_hooks(subset, args):
    print('📌 ENTRY: setup_gptq_and_hooks')
    gptq = {}
    hook_instances = {}
    for name, dense_layer in subset.items():
        gptq[name] = GPTQ(dense_layer)
        quantizer = Quantizer()
        quantizer.configure(
            args.wbits, perchannel=True, sym=args.sym, mse=False, trits=getattr(args, 'trits', False)
        )
        gptq[name].quantizer = quantizer
        hook = DenseHook(dense_layer, gptq[name])
        hook_instances[name] = hook
    print('📌 EXIT: setup_gptq_and_hooks')
    return gptq, hook_instances

def replace_dense_with_hooks(layer, subset, hook_instances):
    print('📌 ENTRY: replace_dense_with_hooks')
    for name, dense_layer in subset.items():
        result = find_parent_and_attr(layer, dense_layer)
        if result is not None:
            parent, attr_name = result
            setattr(parent, attr_name, hook_instances[name])
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, name):
            setattr(layer.self_attn, name, hook_instances[name])
    print('📌 EXIT: replace_dense_with_hooks')

def restore_dense_layers(layer, subset):
    print('📌 ENTRY: restore_dense_layers')
    for name, dense_layer in subset.items():
        result = find_parent_and_attr(layer, dense_layer)
        if result is not None:
            parent, attr_name = result
            setattr(parent, attr_name, dense_layer)
        if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, name):
            setattr(layer.self_attn, name, dense_layer)
    
    # More thorough restoration - find and replace all DenseHook instances
    def restore_hooks_recursive(module):
        if hasattr(module, 'submodules'):
            for submodule in module.submodules:
                if isinstance(submodule, DenseHook):
                    # Replace DenseHook with its original dense_layer
                    original_layer = getattr(submodule, 'dense_layer', None)
                    if original_layer is not None:
                        # Find the parent module and attribute name
                        for attr_name in dir(module):
                            if getattr(module, attr_name, None) is submodule:
                                setattr(module, attr_name, original_layer)
                                print(f"[CLEANUP] Restored {attr_name} in {module.__class__.__name__} to original Dense layer (id={id(original_layer)})")
                restore_hooks_recursive(submodule)
    
    restore_hooks_recursive(layer)
    print('📌 EXIT: restore_dense_layers')

def quantize_dense_layers(subset, gptq, quantizers, args, quantization_type):
    print('📌 ENTRY: quantize_dense_layers')
    for name, dense_layer in subset.items():
        try:
            if quantization_type == 'gptq':
                gptq[name].fasterquant(
                    blocksize=getattr(args, 'blocksize', 128),
                    percdamp=args.percdamp,
                    groupsize=args.groupsize,
                    actorder=getattr(args, 'act_order', False),
                    static_groups=getattr(args, 'static_groups', False)
                )
                quantizers[name] = gptq[name].quantizer
            elif quantization_type == 'simple':
                W = dense_layer.weights[0].numpy()
                w_min = np.min(W)
                w_max = np.max(W)
                max_val = (2 ** args.wbits) - 1
                scale = (w_max - w_min) / max_val
                zero_point = w_min
                quantized = np.round((W - zero_point) / scale)
                quantized = np.clip(quantized, 0, max_val)
                dequantized = quantized.astype(np.float32) * scale + zero_point
                dense_layer.weights[0].assign(dequantized)
                quantizers[name] = {
                    'scale': scale,
                    'zero': zero_point,
                    'maxq': max_val
                }
            gptq[name].free()
        except Exception as e:
            print(f"Error quantizing {name}: {e}")
    print('📌 EXIT: quantize_dense_layers')

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
    print('📌 ENTRY: load_opt_model')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForCausalLM.from_pretrained(model_name, from_pt=True)
    print('📌 EXIT: load_opt_model')
    return model, tokenizer

# 2. Download WikiText-2 dataset
def load_wikitext(nsamples=128):
    try:
        wikitext = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        # Use a safe approach to select samples
        from datasets import Dataset
        if isinstance(wikitext, Dataset):
            return wikitext.select(range(nsamples))
        else:
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
    print('📌 ENTRY: prepare_calib_data')
    # Try 'text', then 'sentence', else raise error
    sample = dataset[0]
    if 'text' in sample:
        texts = [x['text'] for x in dataset]
    elif 'sentence' in sample:
        texts = [x['sentence'] for x in dataset]
    else:
        raise KeyError("Neither 'text' nor 'sentence' found in dataset sample keys.")
    encodings = tokenizer(texts, return_tensors="np", padding="max_length", truncation=True, max_length=seqlen)
    print('📌 EXIT: prepare_calib_data')
    return encodings["input_ids"]

# 4. Dataloader generator
def make_dataloader(encodings, batch_size=1):
    print('📌 ENTRY: make_dataloader')
    for i in range(0, encodings.shape[0], batch_size):
        yield encodings[i:i+batch_size]
    print('📌 EXIT: make_dataloader')

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
    print('📌 ENTRY: find_parent_and_attr')
    for attr_name in dir(root):
        if attr_name.startswith('_'):
            continue
        try:
            attr = getattr(root, attr_name)
            if attr is target_layer:
                print('📌 EXIT: find_parent_and_attr - found')
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
                print('📌 EXIT: find_parent_and_attr - found in submodule')
                return result
    print('📌 EXIT: find_parent_and_attr - not found')
    return None

def patch_decoder_layer(layer):
    print('📌 ENTRY: patch_decoder_layer')
    def flatten_dense_call(dense_layer, x, **kwargs):
        tensor_x = get_tensor(x)
        static_shape = getattr(tensor_x, 'shape', None)
        if static_shape is not None and len(static_shape) == 3 and None not in static_shape:
            batch, seq, hidden = static_shape
            x_flat = tf.reshape(tensor_x, [-1, static_shape[-1]])
            out = dense_layer(x_flat, **kwargs)
            out = tf.reshape(out, [batch, seq, -1])
            return out
        else:
            # Try dynamic shape
            shape = tf.shape(tensor_x)
            static_shape = tf.get_static_value(shape)
            if static_shape is not None and len(static_shape) == 3:
                batch, seq, hidden = static_shape
                x_flat = tf.reshape(tensor_x, [-1, hidden])
                out = dense_layer(x_flat, **kwargs)
                out = tf.reshape(out, [batch, seq, -1])
                return out
            else:
                return dense_layer(tensor_x, **kwargs)

    def new_call(self, inputs, *args, **kwargs):
        print("[DEBUG] Patched call for TFOPTDecoderLayer")
        if isinstance(inputs, dict):
            hidden_states = inputs['hidden_states']
            attention_mask = inputs.get('attention_mask', None)
        else:
            hidden_states = inputs
            attention_mask = None

        x = hidden_states
        print("[DEBUG] input to self_attn_layer_norm:", x.shape)
        x = self.self_attn_layer_norm(x)
        print("[DEBUG] after self_attn_layer_norm:", x.shape)
        attn_outputs = self.self_attn(x, attention_mask=attention_mask, training=kwargs.get('training', False))
        x = attn_outputs[0] if isinstance(attn_outputs, (tuple, list)) else attn_outputs
        print("[DEBUG] after self_attn:", x.shape)
        x = self.dropout(x, training=kwargs.get('training', False))
        print("[DEBUG] after dropout:", x.shape)
        x = x + hidden_states
        print("[DEBUG] after residual add:", x.shape)

        y = self.final_layer_norm(x)
        print("[DEBUG] after final_layer_norm:", y.shape)
        y = flatten_dense_call(self.fc1, y)
        print("[DEBUG] after fc1:", y.shape)
        y = flatten_dense_call(self.fc2, y)
        print("[DEBUG] after fc2:", y.shape)
        y = self.dropout(y, training=kwargs.get('training', False))
        print("[DEBUG] after dropout2:", y.shape)
        if y.shape == x.shape:
            y = y + x
            print("[DEBUG] after MLP residual add:", y.shape)
        else:
            print(f"[WARNING] Skipping residual addition: y.shape={y.shape}, x.shape={x.shape}")
        return {'hidden_states': y}
    layer.call = new_call.__get__(layer, layer.__class__)
    print('📌 EXIT: patch_decoder_layer')

def patch_attention_module(attn_module):
    """
    Monkey-patch the call method of TFOPTAttention to always use the current
    k_proj, q_proj, v_proj, out_proj attributes (which may be hooks).
    During calibration, call all projections to trigger hooks and collect data, but skip actual attention computation.
    """
    print('📌 ENTRY: patch_attention_module')
    # Save the original call method
    if not hasattr(attn_module, '_original_call'):
        attn_module._original_call = attn_module.call

    def new_call(self, hidden_states, attention_mask=None, **kwargs):
        print("[DEBUG] Patched call for TFOPTAttention")
        print("  k_proj type:", type(self.k_proj))
        print("  q_proj type:", type(self.q_proj))
        print("  v_proj type:", type(self.v_proj))
        print("  out_proj type:", type(self.out_proj))
        # --- Calibration logic: call all projections to trigger hooks ---
        # This matches PyTorch GPTQ calibration logic
        k = self.k_proj(hidden_states)
        print("[DEBUG] k_proj output shape:", getattr(k, 'shape', None))
        q = self.q_proj(hidden_states)
        print("[DEBUG] q_proj output shape:", getattr(q, 'shape', None))
        v = self.v_proj(hidden_states)
        print("[DEBUG] v_proj output shape:", getattr(v, 'shape', None))
        out = self.out_proj(hidden_states)
        print("[DEBUG] out_proj output shape:", getattr(out, 'shape', None))
        # Skip actual attention computation for calibration
        print("[DEBUG] Skipping attention computation for calibration, returning hidden_states")
        return hidden_states

    attn_module.call = new_call.__get__(attn_module, attn_module.__class__)
    print('📌 EXIT: patch_attention_module')

def remove_all_dense_hooks(module):
    """Recursively replace all DenseHook instances in the model with their original dense_layer."""
    print('📌 ENTRY: remove_all_dense_hooks')
    if hasattr(module, 'submodules'):
        for submodule in module.submodules:
            if isinstance(submodule, DenseHook):
                original_layer = getattr(submodule, 'dense_layer', None)
                if original_layer is not None:
                    for attr_name in dir(module):
                        if getattr(module, attr_name, None) is submodule:
                            setattr(module, attr_name, original_layer)
                            print(f"[GLOBAL CLEANUP] Restored {attr_name} in {module.__class__.__name__} to original Dense layer (id={id(original_layer)})")
            remove_all_dense_hooks(submodule)
    print('📌 EXIT: remove_all_dense_hooks')

if __name__ == "__main__":
    print('🚀 ENTRY: main')
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
        from datasets import Dataset
        if isinstance(dataset, Dataset):
            dataset = dataset.select(range(args.nsamples))
        else:
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
    print('📌 About to call opt_sequential_keras')
    quantizers = opt_sequential_keras(model, dataloader, args, quantization_type='gptq')
    print('📌 Returned from opt_sequential_keras')
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
    print('🏁 EXIT: main') 