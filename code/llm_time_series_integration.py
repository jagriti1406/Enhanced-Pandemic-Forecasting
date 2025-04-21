import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os

# Set the device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load and Preprocess Data
def load_data(case_file):
    case_data = pd.read_csv(case_file, index_col=0)
    time_series_data = case_data.iloc[:, 1:].values
    return time_series_data

def quantize_time_series(data, n_bins=10):
    discretizer = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="uniform")
    data_reshaped = data.reshape(-1, 1)
    quantized_data = discretizer.fit_transform(data_reshaped).flatten()
    return quantized_data, discretizer

def create_sequences(data, sequence_length=30):
    inputs, targets = [], []
    for i in range(len(data) - sequence_length):
        inputs.append(data[i:i + sequence_length])
        targets.append(data[i + sequence_length])
    return np.array(inputs), np.array(targets)

# Step 2: Define Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, inputs, targets):
        assert len(inputs) == len(targets), "Inputs and targets must have the same length"
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.inputs[idx], dtype=torch.long),
            "labels": torch.tensor(self.targets[idx], dtype=torch.long),
        }

# Step 3: Fine-Tune LLaMA 2
def fine_tune_llama2(X_train, y_train, X_val, y_val, vocab_size):
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token for padding

    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf").to(device)
    model.resize_token_embeddings(vocab_size)  # Resize embeddings

    # Ensure all tokens are within range
    assert X_train.max() < vocab_size, "X_train contains invalid tokens!"
    assert X_val.max() < vocab_size, "X_val contains invalid tokens!"

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Disable masked language modeling
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,  # Lower learning rate for LLaMA 2
        per_device_train_batch_size=4,  # Adjust batch size for larger model
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=10,
        gradient_accumulation_steps=4,  # To handle larger models
        fp16=True,  # Mixed precision training for speed
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    return model


# Step 4: Generate Predictions
def generate_predictions(model, tokenizer, input_sequence, num_predictions):
    model.eval()
    input_ids = torch.tensor(input_sequence, dtype=torch.long).unsqueeze(0).to(device)

    predictions = []
    with torch.no_grad():
        for _ in range(num_predictions):
            outputs = model.generate(
                input_ids,
                max_new_tokens=1,
                temperature=0.7,
                top_k=50,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,  # Use tokenizer.eos_token_id here
            )
            next_token = outputs[0, -1].item()
            predictions.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], device=device)], dim=-1)

    return predictions

# Step 5: Evaluate Model
def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return mse, mae

def main():
    # Define the data directory
    data_dir = "C:/Users/amanl/Downloads/pandemic_tgnn-master/pandemic_tgnn-master/data"
    sequence_length = 50
    n_bins = 100
    test_size = 0.2
    random_state = 42

    # Initialize the tokenizer once
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token to eos token

    # Iterate through each country folder dynamically
    for country in ["Italy", "France", "England", "Spain"]:
        country_dir = os.path.join(data_dir, country)
        label_file = os.path.join(country_dir, f"{country.lower()}_labels.csv")
        
        if not os.path.exists(label_file):
            print(f"Label file not found for {country}: {label_file}")
            continue

        print(f"Processing data for {country}...")

        # Step 1: Load and preprocess data
        time_series_data = load_data(label_file)
        quantized_data, discretizer = quantize_time_series(time_series_data.flatten(), n_bins=n_bins)

        # Step 2: Create sequences
        X, y = create_sequences(quantized_data, sequence_length=sequence_length)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=random_state)

        print(f"{country} - X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"{country} - X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

        # Step 3: Fine-tune the model
        vocab_size = int(X_train.max()) + 1
        print(f"Max value in X_train: {X_train.max()}")
        print(f"Vocab size: {vocab_size}")
        print(X_train[:5])
        model = fine_tune_llama2(X_train, y_train, X_val, y_val, vocab_size)
        
        # Step 4: Generate predictions
        last_sequence = X_val[-1]
        num_predictions = 10
        predicted_tokens = generate_predictions(model, tokenizer, last_sequence, num_predictions)

        # Decode predictions
        predicted_values = discretizer.inverse_transform(np.array(predicted_tokens).reshape(-1, 1))
        print(f"{country} - Predicted values for the next 10 steps:", predicted_values.flatten())

        # Evaluate the model
        y_val_decoded = discretizer.inverse_transform(y_val.reshape(-1, 1))
        mse, mae = evaluate_model(y_val_decoded[:num_predictions], predicted_values)
        print(f"{country} - Mean Squared Error: {mse}")
        print(f"{country} - Mean Absolute Error: {mae}")

if __name__ == "__main__":
    main()
