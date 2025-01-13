import sys

sys.path.append("../python")
from needle.data import MNISTDataset, DataLoader
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    block_1 = nn.Sequential(
        nn.Linear(dim, hidden_dim),
        norm(hidden_dim), 
        nn.ReLU(),
        nn.Dropout(drop_prob),
        nn.Linear(hidden_dim, dim),
        norm(dim))
    res_block = nn.Residual(block_1)
    return nn.Sequential(
        res_block, 
        nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    input_layer = nn.Linear(dim, hidden_dim)
    blocks = [ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)]
    output_layer = nn.Linear(hidden_dim, num_classes)
    return nn.Sequential(
        input_layer,
        nn.ReLU(),
        *blocks,
        output_layer)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    criterion = nn.SoftmaxLoss()
    model.train() if opt else model.eval()
    num_batches = 0
    total_loss = 0
    total_correct = 0
    for X, y in dataloader:        
        logits = model(X) # forward pass
        loss = criterion(logits, y)
        if opt != None:
            opt.reset_grad()
            loss.backward()
            opt.step()
          
        logits_data = logits.realize_cached_data()
        y_data = y.realize_cached_data()

        preds_data = np.argmax(logits_data, axis=1)
        bool_mask = (preds_data==y_data)
        # print(f"Preds inside Epoch: {preds_data}")
        # print(f"Labels inside Epoch: {y_data}")
        # print(f"inside epoch{bool_mask.shape}")
        # print(f"inside epoch{y_data.shape}")
        total_correct += bool_mask.sum().item() # tensors have no == implemented ?????? not a differentiable function. where is this implemented?
        total_loss += loss.realize_cached_data().item()
        num_batches += 1
    return 1-total_correct/len(dataloader.dataset), total_loss/num_batches


    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    
    # download data, split it train/test
    train_dataset = MNISTDataset(
        image_filename=f"{data_dir}/train-images-idx3-ubyte.gz", 
        label_filename=f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dataset = MNISTDataset(
        image_filename=f"{data_dir}/t10k-images-idx3-ubyte.gz", 
        label_filename=f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    
    # init data loaders with batch_size (train, test)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    # init model with hidden_dim and image dim as input_dim 28*28
    model = MLPResNet(dim=28*28, hidden_dim=hidden_dim)

    # init optimizer with lr, decay
    opt = optimizer(params=model.parameters(), lr=lr, weight_decay=weight_decay)

    # loop over all epochs and call epoch helper function
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, model, opt)
    
    # compute test loss
    test_err, test_loss = epoch(test_loader, model, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="../data")