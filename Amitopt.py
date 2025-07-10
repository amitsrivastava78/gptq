# main.py
import tensorflow as tf
from datasets import load_dataset
from transformers import AutoTokenizer, TFOPTForCausalLM

def get_wikitext2(tokenizer, sequence_length=128, batch_size=8):
    """
    Loads and processes the wikitext-2-raw-v1 dataset.

    Args:
        tokenizer: The tokenizer to use for encoding the text.
        sequence_length (int): The fixed length of sequences.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        A tf.data.Dataset object ready for training.
    """
    print("Loading wikitext-2 dataset...")
    # Load the training split
    train_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Filter out empty lines
    train_dataset = train_dataset.filter(lambda example: example['text'] != '')
    print(f"Number of examples after filtering: {len(train_dataset)}")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="tf", padding='max_length', truncation=True, max_length=sequence_length)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Convert to a TensorFlow DataLoader (tf.data.Dataset)
    # For language modeling, the input_ids are used as both input and label.
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['input_ids'], # Use input_ids as the label
        shuffle=True,
        batch_size=batch_size,
        collate_fn=None # Use default collation
    )

    print("Wikitext-2 dataset converted to TensorFlow DataLoader.")
    return tf_dataset

def get_ptb(tokenizer, sequence_length=128, batch_size=8):
    """
    Loads and processes the Penn Treebank (PTB) dataset directly from its source URL.

    Args:
        tokenizer: The tokenizer to use for encoding the text.
        sequence_length (int): The fixed length of sequences.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        A tf.data.Dataset object ready for training.
    """
    print("\nLoading PTB dataset...")
    # We load the data directly from its source URL using the generic 'text' loader.
    data_files = {"train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt"}
    train_dataset = load_dataset("text", data_files=data_files, split="train")

    # Filter out empty lines (the 'text' loader creates a 'text' column)
    train_dataset = train_dataset.filter(lambda example: example['text'] != '')
    print(f"Number of examples after filtering: {len(train_dataset)}")

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["text"], return_tensors="tf", padding='max_length', truncation=True, max_length=sequence_length)

    tokenized_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Convert to a TensorFlow DataLoader (tf.data.Dataset)
    tf_dataset = tokenized_dataset.to_tf_dataset(
        columns=['input_ids', 'attention_mask'],
        label_cols=['input_ids'], # Use input_ids as the label
        shuffle=True,
        batch_size=batch_size,
        collate_fn=None # Use default collation
    )

    print("PTB dataset converted to TensorFlow DataLoader.")
    return tf_dataset

def get_opt_125m_tf():
    """
    Loads the facebook/opt-125m model and tokenizer for TensorFlow.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    print("\nLoading facebook/opt-125m for TensorFlow...")
    model_name = "facebook/opt-125m"
    # Note the use of TFOPTForCausalLM for TensorFlow
    model = TFOPTForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Model and tokenizer loaded.")
    return model, tokenizer

if __name__ == "__main__":
    # Define a batch size
    BATCH_SIZE = 4

    # 1. Load the TensorFlow model and tokenizer
    opt_model, opt_tokenizer = get_opt_125m_tf()

    # 2. Load and process the datasets into TensorFlow DataLoaders
    wikitext_dataloader = get_wikitext2(opt_tokenizer, batch_size=BATCH_SIZE)
    ptb_dataloader = get_ptb(opt_tokenizer, batch_size=BATCH_SIZE)

    # 3. Print some information to verify
    print("\n--- Verification ---")
    print(f"Model Class: {opt_model.__class__.__name__}")
    print(f"Tokenizer Class: {opt_tokenizer.__class__.__name__}")

    # Take one batch from each dataloader to show the structure
    print("\nSample batch from Wikitext-2 DataLoader:")
    for inputs, labels in wikitext_dataloader.take(1):
        print("Inputs (input_ids) shape:", inputs['input_ids'].shape)
        print("Inputs (attention_mask) shape:", inputs['attention_mask'].shape)
        print("Labels shape:", labels.shape)

    print("\nSample batch from PTB DataLoader:")
    for inputs, labels in ptb_dataloader.take(1):
        print("Inputs (input_ids) shape:", inputs['input_ids'].shape)
        print("Inputs (attention_mask) shape:", inputs['attention_mask'].shape)
        print("Labels shape:", labels.shape)