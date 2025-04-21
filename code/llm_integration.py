import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from transformers import (
    BertTokenizer,
    BertModel,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import mean_squared_error, mean_absolute_error
import glob
from context_data_fetcher import generate_contextual_embeddings

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Custom BERT model to integrate contextual embeddings
class CustomBertModel(nn.Module):
    def __init__(self, bert_model_name, feature_dim, num_labels):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.feature_dim = feature_dim
        self.hidden_dim = self.bert.config.hidden_size  # Get hidden size from BERT
        self.num_labels = num_labels

        # Define the classifier layer
        self.classifier = nn.Linear(self.hidden_dim + feature_dim, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, features=None, labels=None):
        # Get BERT outputs
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output  # Shape: [batch_size, hidden_dim]

        # Process additional features
        if features is not None:
            features = features.mean(dim=1)  # Average pooling over sequence dimension

        # Concatenate pooled output and features
        combined = torch.cat((pooled_output, features), dim=1)  # Shape: [batch_size, hidden_dim + feature_dim]

        # Pass through the classifier
        logits = self.classifier(combined)  # Shape: [batch_size, num_labels]

        # Compute loss if labels are provided
        loss = None
        if labels is not None:
            loss_fn = nn.MSELoss()
            loss = loss_fn(logits.squeeze(), labels.squeeze())  # Ensure shapes match

        return {"logits": logits, "loss": loss} if labels is not None else logits

# Dataset Class
class TimeSeriesDataset(Dataset):
    def __init__(self, input_ids, attention_masks, features, targets):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attention_masks[idx], dtype=torch.long),
            "features": torch.tensor(self.features[idx], dtype=torch.float),
            "labels": torch.tensor(self.targets[idx], dtype=torch.float),
        }

# Prepare data
def prepare_data(label_file, tokenizer, contextual_embeddings, sequence_length=30):
    time_series_data = pd.read_csv(label_file, index_col=0).iloc[:, 1:].values.flatten()
    quantized_data = KBinsDiscretizer(n_bins=100, encode="ordinal", strategy="uniform").fit_transform(time_series_data.reshape(-1, 1)).flatten()

    # Combine contextual embeddings with time-series data
    enriched_data = np.hstack([quantized_data.reshape(-1, 1), contextual_embeddings.repeat(len(quantized_data), axis=0)])

    # Create sequences
    inputs, targets = [], []
    for i in range(len(enriched_data) - sequence_length):
        inputs.append(enriched_data[i:i + sequence_length])
        targets.append(enriched_data[i + sequence_length][0])  # Target is the first column

    # Tokenize input sequences for BERT
    input_ids, attention_masks = [], []
    for sequence in inputs:
        text_sequence = " ".join(map(str, sequence[:, 0]))  # Use the first column for text representation
        tokens = tokenizer(text_sequence, truncation=True, padding="max_length", max_length=sequence_length, return_tensors="pt")
        input_ids.append(tokens["input_ids"].squeeze().tolist())
        attention_masks.append(tokens["attention_mask"].squeeze().tolist())

    features = np.array([seq[:, 1:] for seq in inputs])  # Use remaining columns as features
    targets = np.array(targets)
    return input_ids, attention_masks, features, targets

# Train the model
def fine_tune_model(train_dataset, val_dataset, feature_dim, num_labels, bert_model_name="bert-base-uncased", epochs=3, batch_size=8):
    model = CustomBertModel(bert_model_name=bert_model_name, feature_dim=feature_dim, num_labels=num_labels).to(device)

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.squeeze()
        labels = labels.squeeze()
        mse = mean_squared_error(labels, logits)
        mae = mean_absolute_error(labels, logits)
        return {"mse": mse, "mae": mae}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return model

def main():
    data_dir = "/content/drive/My Drive/pandemic_tgnn-master/data/Italy"
    label_file = os.path.join(data_dir, "italy_labels.csv")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    sequence_length = 30

    print(f"Processing file: {label_file}")

    # Generate contextual embeddings
    region_name = "Italy"
    contextual_embeddings = generate_contextual_embeddings(
        region_names=[region_name],
        start_dates=["2020-02-24"],
        end_dates=["2020-06-01"],
    )

    # Prepare data
    input_ids, attention_masks, features, targets = prepare_data(label_file, tokenizer, contextual_embeddings, sequence_length)

    # Train-test split
    X_train_ids, X_val_ids, X_train_masks, X_val_masks, X_train_features, X_val_features, y_train, y_val = train_test_split(
        input_ids, attention_masks, features, targets, test_size=0.2, random_state=42
    )

    train_dataset = TimeSeriesDataset(X_train_ids, X_train_masks, X_train_features, y_train)
    val_dataset = TimeSeriesDataset(X_val_ids, X_val_masks, X_val_features, y_val)

    print(f"X_train shape: {X_train_features.shape}, y_train shape: {y_train.shape}")
    print(f"X_val shape: {X_val_features.shape}, y_val shape: {y_val.shape}")

    # Fine-tune model
    feature_dim = X_train_features.shape[-1]
    num_labels = 1  # Assuming regression task
    model = fine_tune_model(train_dataset, val_dataset, feature_dim, num_labels, bert_model_name="bert-base-uncased")

if __name__ == "__main__":
    main()
