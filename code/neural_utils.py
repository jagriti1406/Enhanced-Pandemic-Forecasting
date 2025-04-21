import torch
import numpy as np
from math import ceil

import torch.nn.functional as F
import torch.optim as optim

from preprocess import AverageMeter, generate_batches, generate_batches_lstm, generate_new_batches
from models import MPNN_LSTM, LSTM, MPNN, GATModel
import time 

def train(model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          adj: torch.nn.Module,
          features: torch.nn.Module,
          y: torch.nn.Module):
    """
    Train the model

    Parameters:
    model (torch.nn.Module): Model to train
    adj (torch.Tensor): Adjacency matrix
    features (torch.Tensor): Features matrix
    y (torch.Tensor): Labels matrix

    Returns:
    output (torch.Tensor): Output predictions of the model
    loss_train (torch.Tensor): Loss of the model
    """
    optimizer.zero_grad()
    output = model(adj, features)
    
    # # Print shapes of predictions and labels to verify alignment
    # print("Train - Prediction shape:", output.shape)
    # print("Train - Label shape:", y.shape)

    # Reshape `y` if shapes don't match
    if output.size(0) != y.size(0):  
        y = y.view(-1).repeat((output.size(0) // y.size(0)) + 1)[:output.size(0)]
    
    loss_train = F.mse_loss(output, y)
    loss_train.backward(retain_graph=True)
    optimizer.step()
    return output, loss_train

def test(model: torch.nn.Module,
         adj: torch.Tensor,
         features: torch.Tensor,
         y: torch.Tensor):
    """
    Test the model
    
    Parameters:
    model (torch.nn.Module): Model to test
    adj (torch.Tensor): Adjacency matrix
    features (torch.Tensor): Features matrix
    y (torch.Tensor): Labels matrix

    Returns:
    output (torch.Tensor): Output predictions of the model
    loss_test (torch.Tensor): Loss of the model
    """    
    output = model(adj, features)
    
    # # Print shapes of predictions and labels to verify alignment during testing
    # print("Test - Prediction shape:", output.shape)
    # print("Test - Label shape:", y.shape)

    # Reshape `y` if shapes don't match
    if output.size(0) != y.size(0):
        y = y.view(-1).repeat((output.size(0) // y.size(0)) + 1)[:output.size(0)]
    
    loss_test = F.mse_loss(output, y)
    return output, loss_test

def run_neural_model(model: str,
                     n_nodes: int,
                     early_stop: int,
                     idx_train: list,
                     window: int,
                     shift: int,
                     batch_size: int,
                     y: list,
                     device: torch.device,
                     test_sample: int,
                     graph_window: int,
                     recur: bool,
                     gs_adj: list,
                     features: list,
                     idx_val: list,
                     hidden, dropout: float,
                     lr: float,
                     nfeat: int,
                     epochs: int,
                     print_epoch: int=50):
    """
    Derive batches from the data, train the model, and test it.
    """

    # Generate batches based on model type
    if model == "LSTM":
        lstm_features = 1 * n_nodes
        adj_train, features_train, y_train = generate_batches_lstm(n_nodes, y, idx_train, window, shift, batch_size, device, test_sample)
        adj_val, features_val, y_val = generate_batches_lstm(n_nodes, y, idx_train, window, shift, batch_size, device, test_sample)
        adj_test, features_test, y_test = generate_batches_lstm(n_nodes, y, [test_sample], window, shift, batch_size, device, test_sample)
    elif model == "GAT":
        # Use edge_index for GAT model
        edge_index_train, features_train, y_train = generate_new_batches(gs_adj, features, y, idx_train, graph_window, shift, batch_size, device, test_sample, use_edge_index=True)
        edge_index_val, features_val, y_val = generate_new_batches(gs_adj, features, y, idx_val, graph_window, shift, batch_size, device, test_sample, use_edge_index=True)
        edge_index_test, features_test, y_test = generate_new_batches(gs_adj, features, y, [test_sample], graph_window, shift, batch_size, device, test_sample, use_edge_index=True)
    else:
        adj_train, features_train, y_train = generate_batches(gs_adj, features, y, idx_train, graph_window, shift, batch_size, device, test_sample)
        adj_val, features_val, y_val = generate_batches(gs_adj, features, y, idx_val, graph_window, shift, batch_size, device, test_sample)
        adj_test, features_test, y_test = generate_batches(gs_adj, features, y, [test_sample], graph_window, shift, batch_size, device, test_sample)

    # Print data shapes after batching
    print("Batch Shapes - features_train:", features_train[0].shape, "y_train:", y_train[0].shape)
    print("Batch Shapes - features_val:", features_val[0].shape, "y_val:", y_val[0].shape)
    print("Batch Shapes - features_test:", features_test[0].shape, "y_test:", y_test[0].shape)

    # Initialize the model based on selection
    if model == "LSTM":
        model = LSTM(nfeat=lstm_features, nhid=hidden, n_nodes=n_nodes, window=window, dropout=dropout, batch_size=batch_size, recur=recur).to(device)
    elif model == "MPNN_LSTM":
        model = MPNN_LSTM(nfeat=nfeat, nhid=hidden, nout=1, n_nodes=n_nodes, window=graph_window, dropout=dropout).to(device)
    elif model == "MPNN":
        model = MPNN(nfeat=nfeat, nhid=hidden, nout=1, dropout=dropout).to(device)
    elif model == "GAT":
        model = GATModel(nfeat=nfeat, nhid=hidden, nout=1, dropout=dropout, heads=4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    # Training
    n_train_batches = ceil(len(idx_train) / batch_size)
    best_val_acc = 1e8
    val_among_epochs = []
    train_among_epochs = []
    stop = False

    while not stop:
        for epoch in range(epochs):
            model.train()
            train_loss = AverageMeter()

            # Train for one epoch
            for batch in range(n_train_batches):
                if model == "GAT":
                    output, loss = train(model, optimizer, edge_index_train[batch], features_train[batch], y_train[batch])
                else:
                    output, loss = train(model, optimizer, adj_train[batch], features_train[batch], y_train[batch])
                
                train_loss.update(loss.data.item(), output.size(0))

            # Evaluate on validation set
            model.eval()
            if model == "GAT":
                output, val_loss = test(model, edge_index_val[0], features_val[0], y_val[0])
            else:
                output, val_loss = test(model, adj_val[0], features_val[0], y_val[0])

            val_loss = float(val_loss.detach().cpu().numpy())

            # Print results
            if epoch % print_epoch == 0:
                print(f"Epoch: {epoch + 1}, train_loss={train_loss.avg:.5f}, val_loss={val_loss:.5f}")

            train_among_epochs.append(train_loss.avg)
            val_among_epochs.append(val_loss)

            # Early stopping logic
            if epoch > early_stop and len(set([round(v) for v in val_among_epochs[-int(early_stop / 2):]])) == 1:
                print("Break early stop")
                stop = True
                break

            # Save best model
            if val_loss < best_val_acc:
                best_val_acc = val_loss
                torch.save({
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, 'model_best.pth.tar')

            scheduler.step(val_loss)

    # Testing
    checkpoint = torch.load('model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    if model == "GAT":
        output, loss = test(model, edge_index_test[0], features_test[0], y_test[0])
    else:
        output, loss = test(model, adj_test[0], features_test[0], y_test[0])

    o = output.cpu().detach().numpy()
    l = y_test[0].cpu().numpy()
    
    # Diagnostic print statements
    print("Final Prediction shape:", o.shape)
    print("Final Target shape:", l.shape)

    # Ensure shapes match before error calculation
    if o.shape[0] != l.shape[0]:
        min_size = min(o.shape[0], l.shape[0])
        o = o[:min_size]
        l = l[:min_size]

    error = np.sum(abs(o - l)) / n_nodes
    return error
