import argparse
import keras
import numpy as np
from transformers import TFAutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from gptqkeras import GPTQ
from quantkeras import Quantizer
from tensorflow import keras as tf_keras  # For compatibility with HuggingFace


def find_layers(module):
    # Recursively find all Dense layers in the module
    return {f"dense_{i}": l for i, l in enumerate(module.submodules) if isinstance(l, keras.layers.Dense)}

# ActivationCatcher as before
class ActivationCatcher(keras.layers.Layer):
    def __init__(self, layer, gptq_obj, **kwargs):
        super().__init__(**kwargs)
        self.layer = layer
        self.gptq_obj = gptq_obj
    def call(self, inputs, **kwargs):
        outputs = self.layer(inputs, **kwargs)
        self.gptq_obj.add_batch(inputs, outputs)
        return outputs

def opt_sequential_keras(model, dataloader, args, quantization_type='gptq'):
    print('Starting ...')
    print('Calibrating on token IDs...')
    for batch in dataloader:
        batch = batch.astype('int32')
        _ = model(batch)
    print('Calibration complete.')

    # Now quantize all Dense layers
    quantizers = {}
    for i, layer in enumerate(model.submodules):
        if isinstance(layer, keras.layers.Dense):
            gptq = GPTQ(layer)
            gptq.quantizer = Quantizer()
            gptq.quantizer.configure(
                args.wbits, perchannel=True, sym=args.sym, mse=False, trits=getattr(args, 'trits', False)
            )
            print(f"Quantizing layer {i} ({layer.name}) ...")
            gptq.fasterquant(
                blocksize=getattr(args, 'blocksize', 128),
                percdamp=args.percdamp,
                groupsize=args.groupsize,
                actorder=getattr(args, 'act_order', False),
                static_groups=getattr(args, 'static_groups', False)
            )
            quantizers[layer.name] = gptq.quantizer
            gptq.free()
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
    return wikitext.select(range(nsamples))

# 3. Prepare calibration data (tokenize and batch)
def prepare_calib_data(dataset, tokenizer, nsamples=128, seqlen=128):
    texts = [x['text'] for x in dataset]
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
    seqlen = args.seqlen
    for batch in testloader:
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
        loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        loss = loss_fn(shift_labels, shift_logits)
        nll = np.sum(loss)
        nlls.append(nll)
    total_tokens = nsamples * (seqlen - 1)
    total_nll = np.sum(nlls)
    ppl = np.exp(total_nll / total_tokens)
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
        test_data = prepare_calib_data(testset, tokenizer, nsamples=args.nsamples, seqlen=args.seqlen)
        testloader = make_dataloader(test_data, batch_size=1)
        print(dataset_name)
        opt_eval_keras(model, testloader, args, tokenizer) 