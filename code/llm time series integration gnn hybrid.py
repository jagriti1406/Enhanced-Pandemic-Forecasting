import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from transformers import AutoTokenizer, AutoModel
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data as GeoData
from sklearn.model_selection import train_test_split, KFold
import matplotlib.pyplot as plt
from tqdm import tqdm

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def load_case_counts(case_file):
    """
    Load case counts and prepare data per region per date.
    """
    case_data = pd.read_csv(case_file)
    case_data = case_data.loc[:, ~case_data.columns.str.contains('^Unnamed')]
    date_columns = case_data.columns.drop(['name', 'id'])
    case_data_melted = case_data.melt(
        id_vars=['name', 'id'],
        value_vars=date_columns,
        var_name='date',
        value_name='cases'
    )
    
    case_data_melted['date'] = pd.to_datetime(case_data_melted['date'], format='%Y-%m-%d')  
    case_data_melted['cases'] = pd.to_numeric(case_data_melted['cases'], errors='coerce').fillna(0)
    
    print(f"Loaded case counts data with shape: {case_data_melted.shape}")
    return case_data_melted  

def load_movement_data(graph_folder):
    """
    Load movement data and prepare inflow and outflow per region per date.
    """
    movement_data_list = []
    for graph_file in sorted(os.listdir(graph_folder)):
        if graph_file.endswith(".csv") and graph_file.startswith("IT_"):
            date_str = graph_file[3:-4] 
            graph_path = os.path.join(graph_folder, graph_file)
            edges = pd.read_csv(graph_path, header=None, names=['source', 'destination', 'weight'])
            edges = edges.loc[:, ~edges.columns.str.contains('^Unnamed')]
            try:
                edges['date'] = pd.to_datetime(date_str, format='%Y-%m-%d')  # Adjust format as needed
            except ValueError:
                edges['date'] = pd.to_datetime(date_str, infer_datetime_format=True)
            
            movement_data_list.append(edges)
    if movement_data_list:
        movement_data = pd.concat(movement_data_list, ignore_index=True)
    else:
        movement_data = pd.DataFrame(columns=['source', 'destination', 'weight', 'date'])
    print(f"Loaded movement data with shape: {movement_data.shape}")
    return movement_data 

def compute_inflow_outflow(movement_data):
    """
    Compute inflow and outflow per region per date.
    """
    inflow = movement_data.groupby(['destination', 'date'])['weight'].sum().reset_index()
    inflow.rename(columns={'destination': 'name', 'weight': 'inflow'}, inplace=True)
    outflow = movement_data.groupby(['source', 'date'])['weight'].sum().reset_index()
    outflow.rename(columns={'source': 'name', 'weight': 'outflow'}, inplace=True)
    flow_data = pd.merge(inflow, outflow, on=['name', 'date'], how='outer').fillna(0)
    print(f"Computed flow data with shape: {flow_data.shape}")
    return flow_data  

def merge_case_flow_data(case_data, flow_data):
    """
    Merge case counts and flow data per region per date.
    """
    data = pd.merge(case_data, flow_data, on=['name', 'date'], how='left').fillna(0)
    data = data.drop_duplicates(subset=['name', 'date'])
    data['cases'] = data['cases'].astype(float)
    data['inflow'] = data['inflow'].astype(float)
    data['outflow'] = data['outflow'].astype(float)
    
   
    data = data.sort_values(['name', 'date']).reset_index(drop=True)
    print(data)
    
    print(f"Merged case and flow data with shape: {data.shape}")
    return data  

def prepare_features_enhanced(data, lag_days=3, window=7):
    """
    Prepare features and target variable for modeling.
    """
    data = data.sort_values(['name', 'date'])
    for lag in range(1, lag_days + 1):
        data[f'cases_lag{lag}'] = data.groupby('name')['cases'].shift(lag).fillna(0)
        data[f'inflow_lag{lag}'] = data.groupby('name')['inflow'].shift(lag).fillna(0)
        data[f'outflow_lag{lag}'] = data.groupby('name')['outflow'].shift(lag).fillna(0)
    
    
    data[f'cases_roll_mean_{window}'] = data.groupby('name')['cases'].transform(lambda x: x.rolling(window=window).mean().fillna(0))
    data[f'inflow_roll_mean_{window}'] = data.groupby('name')['inflow'].transform(lambda x: x.rolling(window=window).mean().fillna(0))
    data[f'outflow_roll_mean_{window}'] = data.groupby('name')['outflow'].transform(lambda x: x.rolling(window=window).mean().fillna(0))
    
    feature_columns = [f'cases_lag{lag}' for lag in range(1, lag_days + 1)] + \
                      [f'inflow_lag{lag}' for lag in range(1, lag_days + 1)] + \
                      [f'outflow_lag{lag}' for lag in range(1, lag_days + 1)] + \
                      [f'cases_roll_mean_{window}', f'inflow_roll_mean_{window}', f'outflow_roll_mean_{window}']
    
    data['inflow_outflow_interaction'] = data['inflow'] * data['outflow']
    feature_columns.append('inflow_outflow_interaction')
    
    X = data[feature_columns]
    y = data['cases']
    y = np.log1p(y)
    
    print(f"Prepared features with shape: {X.shape} and target with shape: {y.shape}")
    return X, y, data['name'], data['date']

class GNN_Autoencoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(GNN_Autoencoder, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, embedding_dim)
        self.conv3 = GCNConv(embedding_dim, hidden_channels)
        self.conv4 = GCNConv(hidden_channels, in_channels)
        self.activation = nn.ReLU()
        
    def encode(self, x, edge_index, edge_weight=None):
        x = self.conv1(x, edge_index, edge_weight)
        x = self.activation(x)
        x = self.conv2(x, edge_index, edge_weight)
        return x
    
    def decode(self, z, edge_index, edge_weight=None):
        z = self.conv3(z, edge_index, edge_weight)
        z = self.activation(z)
        z = self.conv4(z, edge_index, edge_weight)
        return z
    
    def forward(self, x, edge_index, edge_weight=None):
        z = self.encode(x, edge_index, edge_weight)
        out = self.decode(z, edge_index, edge_weight)
        return out

def train_gnn_autoencoder(movement_data, num_epochs=200, learning_rate=0.01):
    """
    Train the GNN Autoencoder and return node embeddings.
    """
    aggregated = movement_data.groupby(['source', 'destination'])['weight'].sum().reset_index()
    regions = np.unique(np.concatenate((aggregated['source'].values, aggregated['destination'].values)))
    region_to_idx = {region: idx for idx, region in enumerate(regions)}
    aggregated['source_idx'] = aggregated['source'].map(region_to_idx)
    aggregated['destination_idx'] = aggregated['destination'].map(region_to_idx)
    
    edge_index = torch.tensor(aggregated[['source_idx', 'destination_idx']].values, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(aggregated['weight'].values, dtype=torch.float)
    
    num_nodes = len(regions)
    in_channels = 1 
    x = torch.ones((num_nodes, in_channels), dtype=torch.float)
    
    data_geo = GeoData(x=x, edge_index=edge_index, edge_weight=edge_weight)
    hidden_channels = 16
    embedding_dim = 8
    model = GNN_Autoencoder(in_channels, hidden_channels, embedding_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_geo.x.to(device), data_geo.edge_index.to(device), data_geo.edge_weight.to(device))
        loss = criterion(out, data_geo.x.to(device))
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item():.4f}")
    
   
    model.eval()
    with torch.no_grad():
        embeddings = model.encode(data_geo.x.to(device), data_geo.edge_index.to(device), data_geo.edge_weight.to(device))
        embeddings = embeddings.cpu().numpy()
    node_emb_df = pd.DataFrame(embeddings, index=regions)
    node_emb_df.columns = [f'gnn_emb_{i}' for i in range(embeddings.shape[1])]
    
    print(f"GNN Embeddings generated with shape: {node_emb_df.shape}")
    return node_emb_df


def merge_gnn_embeddings(data, node_emb_df):
    """
    Merge GNN embeddings into the main DataFrame based on region names.
    """
    data = data.merge(node_emb_df, left_on='name', right_index=True, how='left')
    gnn_emb_cols = [col for col in data.columns if col.startswith('gnn_emb_')]
    data[gnn_emb_cols] = data[gnn_emb_cols].fillna(0)
    print(f"Merged GNN embeddings with data. New shape: {data.shape}")
    return data



def quantize_features(X, n_clusters=10):
    """
    Quantize multivariate features into discrete tokens using K-means clustering.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    return cluster_labels, kmeans

def serialize_quantized_data(cluster_labels, names, dates):
    """
    Serialize quantized data into strings for LLM input, one per region-date pair.
    """
    serialized_data = []
    for label, name, date in zip(cluster_labels, names, dates):
        date_str = date.strftime('%Y-%m-%d') if isinstance(date, pd.Timestamp) else str(date)
        serialized_data.append(f"Date: {date_str}, Region: {name}, Cluster Label: {label}")
    return serialized_data

def generate_llm_embeddings(serialized_data, model_name, batch_size=16, max_length=64):
    """
    Generate dense embeddings from a specified LLM model for serialized data.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token 

    embeddings = []
    for i in tqdm(range(0, len(serialized_data), batch_size), desc=f"Embedding {model_name}"):
        batch = serialized_data[i:i+batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            last_hidden_state = outputs.last_hidden_state 
            mean_embeddings = last_hidden_state.mean(dim=1)  
            embeddings.append(mean_embeddings.cpu().numpy())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings  

def save_embeddings(embeddings, filename):
    """
    Save embeddings to a file using pickle.
    """
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)

def load_embeddings(filename):
    """
    Load embeddings from a pickle file.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def generate_or_load_llm_embeddings(serialized_data_train, serialized_data_test, model_name, save_dir="embeddings", batch_size=16, max_length=64):
    """
    Generate LLM embeddings or load from cache if available.
    """
    os.makedirs(save_dir, exist_ok=True)
    train_filename = os.path.join(save_dir, f"{model_name}_train_embeddings.pkl")
    test_filename = os.path.join(save_dir, f"{model_name}_test_embeddings.pkl")
    
    if os.path.exists(train_filename) and os.path.exists(test_filename):
        print(f"Loading cached embeddings for {model_name}...")
        llm_embeddings_train = load_embeddings(train_filename)
        llm_embeddings_test = load_embeddings(test_filename)
    else:
        print(f"Generating embeddings for {model_name}...")
        llm_embeddings_train = generate_llm_embeddings(serialized_data_train, model_name, batch_size=batch_size, max_length=max_length)
        llm_embeddings_test = generate_llm_embeddings(serialized_data_test, model_name, batch_size=batch_size, max_length=max_length)
        save_embeddings(llm_embeddings_train, train_filename)
        save_embeddings(llm_embeddings_test, test_filename)
    
    return llm_embeddings_train, llm_embeddings_test

def combine_features_embeddings(llm_embeddings, gnn_embeddings, X_scaled):
    """
    Combine LLM embeddings, GNN embeddings, and scaled features into a single feature set.
    """
    combined_features = np.hstack((llm_embeddings, gnn_embeddings, X_scaled))
    return combined_features



class HybridModel(nn.Module):
    def __init__(self, input_dim):
        super(HybridModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(p=0.3)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.dropout(x)
        x = self.fc5(x)
        return x


def train_hybrid_model_with_validation(model, X_train, y_train, X_val, y_val, num_epochs=200, batch_size=64):
    """
    Train the hybrid model with a validation set and implement early stopping.
    """
    model.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, verbose=True)
    train_losses = []
    val_losses = []
    
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                                   torch.tensor(y_train, dtype=torch.float32).unsqueeze(1))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(X_val, dtype=torch.float32),
                                                 torch.tensor(y_val, dtype=torch.float32).unsqueeze(1))
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    early_stop = False
    patience = 30  
    
    for epoch in range(1, num_epochs + 1):
        
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in train_dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        average_train_loss = epoch_loss / len(train_dataloader)
        train_losses.append(average_train_loss)
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_dataloader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        average_val_loss = val_loss / len(val_dataloader)
        val_losses.append(average_val_loss)
        scheduler.step(average_val_loss)

        print(f"Epoch [{epoch}/{num_epochs}], Train Loss: {average_train_loss:.4f}, Val Loss: {average_val_loss:.4f}")

        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_hybrid_model.pt')
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                early_stop = True
                break

    if os.path.exists('best_hybrid_model.pt'):
        model.load_state_dict(torch.load('best_hybrid_model.pt'))

    return train_losses, val_losses


def evaluate_model(model, X_test, y_test, threshold=1):
    """
    Evaluate the hybrid model and compute metrics.
    """
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X_test, dtype=torch.float32).to(device)
        predictions = model(inputs).cpu().numpy().flatten()
        y_test_np = y_test.flatten()
        predictions_exp = np.expm1(predictions)
        y_test_exp = np.expm1(y_test_np)

        # y_test_exp_safe = np.where(y_test_exp < threshold, threshold, y_test_exp)
        mse = mean_squared_error(y_test_exp, predictions_exp)
        mae = mean_absolute_error(y_test_exp, predictions_exp)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_exp, predictions_exp)
        print(f"Test MSE: {mse:.4f}")
        print(f"Test MAE: {mae:.4f}")
        print(f"Test RMSE: {rmse:.4f}")
        print(f"R-squared Score: {r2:.4f}")
       

        return mse, mae, rmse, r2, predictions_exp, y_test_exp

def validate_features(df, feature_columns):
    """
    Validate that all required feature columns are present in the DataFrame.
    """
    missing_cols = [col for col in feature_columns if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in DataFrame: {missing_cols}")
    print("All required feature columns are present.")

def validate_combined_features(X_train_combined, X_test_combined):
    """
    Validate that combined feature sets have no missing or infinite values.
    """
    if np.isnan(X_train_combined).any():
        raise ValueError("NaN values found in training features.")
    if np.isnan(X_test_combined).any():
        raise ValueError("NaN values found in test features.")
    if np.isinf(X_train_combined).any():
        raise ValueError("Infinite values found in training features.")
    if np.isinf(X_test_combined).any():
        raise ValueError("Infinite values found in test features.")
    print("Combined feature sets are free from NaNs and infinite values.")

def main():
   
    case_file = "/Users/jagritibhandari/Desktop/Pandemic-Forecasting/data/Italy/italy_labels.csv"
    graph_folder = "/Users/jagritibhandari/Desktop/Pandemic-Forecasting/data/Italy/graphs"

    print("Loading and preprocessing data...")
    case_data = load_case_counts(case_file)
    movement_data = load_movement_data(graph_folder)
    flow_data = compute_inflow_outflow(movement_data)
    data = merge_case_flow_data(case_data, flow_data)

    print("Training GNN Autoencoder to generate node embeddings...")
    node_emb_df = train_gnn_autoencoder(movement_data, num_epochs=200, learning_rate=0.01)

    print("Merging GNN embeddings with main data...")
    data = merge_gnn_embeddings(data, node_emb_df)
    print("Preparing features...")
    X, y, names, dates = prepare_features_enhanced(data, lag_days=3, window=7)

   
    X_values = X.values
    y_values = y.values
    names_values = names.values
    dates_values = dates.values
    test_size = 0.2
    split_index = int(len(X_values) * (1 - test_size))
    X_train, X_test = X_values[:split_index], X_values[split_index:]
    y_train, y_test = y_values[:split_index], y_values[split_index:]
    names_train, names_test = names_values[:split_index], names_values[split_index:]
    dates_train, dates_test = dates_values[:split_index], dates_values[split_index:]

    print("Scaling features...")
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)
    
    print("Quantizing features for LLM...")
    cluster_labels_train, kmeans = quantize_features(X_train_scaled)
    serialized_data_train = serialize_quantized_data(cluster_labels_train, names_train, dates_train)
    cluster_labels_test = kmeans.predict(X_test_scaled)
    serialized_data_test = serialize_quantized_data(cluster_labels_test, names_test, dates_test)
    llm_models = [
        "bert-base-uncased",
        "all-MiniLM-L6-v2",
        "paraphrase-MiniLM-L6-v2",
        "distilbert-base-uncased",
        "all-distilroberta-v1"
    ]
    metrics_list = []
    feature_columns = [f'cases_lag{lag}' for lag in range(1, 4)] + \
                      [f'inflow_lag{lag}' for lag in range(1, 4)] + \
                      [f'outflow_lag{lag}' for lag in range(1, 4)] + \
                      ['cases_roll_mean_7', 'inflow_roll_mean_7', 'outflow_roll_mean_7', 'inflow_outflow_interaction']
    try:
        validate_features(X, feature_columns)
    except KeyError as e:
        print(e)
        return  

    for model_name in llm_models:
        print(f"\nProcessing LLM Model: {model_name}")
        try:
            llm_embeddings_train, llm_embeddings_test = generate_or_load_llm_embeddings(
                serialized_data_train, serialized_data_test, model_name, save_dir="embeddings", batch_size=64, max_length=64
            )
        except Exception as e:
            print(f"Error generating embeddings for {model_name}: {e}")
            continue  

        print("Combining LLM and GNN embeddings...")
        try:
            gnn_emb_cols = [col for col in data.columns if col.startswith('gnn_emb_')]
            gnn_embeddings = data[gnn_emb_cols].values[:split_index]  
            gnn_embeddings_test = data[gnn_emb_cols].values[split_index:]  
            X_train_combined = combine_features_embeddings(llm_embeddings_train, gnn_embeddings, X_train_scaled)
            X_test_combined = combine_features_embeddings(llm_embeddings_test, gnn_embeddings_test, X_test_scaled)
        except ValueError as e:
            print(f"Error in combining embeddings for {model_name}: {e}")
            continue  
        try:
            validate_combined_features(X_train_combined, X_test_combined)
        except ValueError as e:
            print(e)
            continue  
        input_dim_hybrid = X_train_combined.shape[1]
        hybrid_model = HybridModel(input_dim_hybrid).to(device)
        print("Splitting into training and validation sets...")
        X_train_final, X_val, y_train_final, y_val = train_test_split(
            X_train_combined, y_train, test_size=0.2, random_state=42
        )

        print("Training Hybrid Model with Validation...")
        train_losses, val_losses = train_hybrid_model_with_validation(
            hybrid_model, X_train_final, y_train_final, X_val, y_val,
            num_epochs=200, batch_size=64
        )

        print("Evaluating Hybrid Model...")
        mse, mae, rmse, r2, mape, smape_value, predictions_exp, y_test_exp = evaluate_model(
            hybrid_model, X_test_combined, y_test, threshold=1
        )

        # Store metrics in the list
        metrics_list.append({
            "LLM_Model": model_name,
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "R2_Score": r2,
            "MAPE": mape,
            "SMAPE": smape_value
        })

    # Convert the list of metrics to a DataFrame
    metrics_df = pd.DataFrame(metrics_list)

    # Display the metrics for all LLM models
    print("\n=== Performance Metrics for All LLM Models ===")
    print(metrics_df)

    # Optionally, save the metrics to a CSV file for future reference
    metrics_df.to_csv("llm_performance_metrics.csv", index=False)

    # Identify the best model based on R²
    if not metrics_df.empty:
        best_model = metrics_df.loc[metrics_df['R2_Score'].idxmax()]
        print(f"\nBest LLM Model based on R²: {best_model['LLM_Model']}")
        print(best_model)
        
        # Plotting the comparison of R² across models
        plt.figure(figsize=(12, 6))
        plt.bar(metrics_df['LLM_Model'], metrics_df['R2_Score'], color='skyblue')
        plt.xlabel('LLM Models')
        plt.ylabel('R-squared Score')
        plt.title('R-squared Comparison Across LLM Models')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print("No metrics to display. All LLM models encountered errors.")

if __name__ == "__main__":
    main()
