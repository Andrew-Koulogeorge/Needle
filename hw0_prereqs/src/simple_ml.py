import struct
import numpy as np
import gzip
try:
    from simple_ml_ext import *
except:
    pass

BIG_ENDIAN = ">" # string format for reading bytes in big endian style
MAX_INTENSITY = 255 # largest intensity of pixel in image

def add(x, y):
    """ A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0 
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE

    # get the rest of the data
    images = []
    with gzip.open(image_filename, "rb") as file:
      header = file.read(16)
      magic_number, num_images, n_rows, n_cols = struct.unpack(">IIII", header)
      print(magic_number)
      if(magic_number != 2051):
        print("Invalid Magic Number\n")
        return 
      image_size = n_rows*n_cols

      for _ in range(num_images):
        # read in each pixel and append it to an image
        image = list(file.read(image_size))
        images.append(image)
      
          
    with gzip.open(label_filename, "rb") as file:
      header = file.read(8)
      magic_number, num_images = struct.unpack(BIG_ENDIAN + "II", header)
      labels = list(file.read(num_images))
      

    images_np = np.array(images, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.uint8)

    # ensure types in X and Y and return
    images_np /= MAX_INTENSITY

    return (images_np, labels_np)
    ### END YOUR CODE


def softmax_loss(Z, y):
    """ Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE    
    S = np.exp(Z)/(np.sum(np.exp(Z), axis=1).reshape(-1,1)) # (B x k) / (B,1)
    log_liklihoods = -np.log(S)[np.arange(y.shape[0]), y]
    return np.mean(log_liklihoods)
    ### END YOUR CODE


def softmax_regression_epoch(X, y, theta, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    
    # compute the gradient of the parameter matrix
    n = X.shape[0]
    start = 0
    while start < n:
      batch_size = min(batch, n - start)

      X_batch = X[start:start + batch_size, :]
      y_batch = y[start:start+batch_size]
      preds = X_batch @ theta # (B x n) (n x k) --> Forward pass to get logits
      Z = np.exp(preds)/(np.sum(np.exp(preds), axis=1).reshape(-1,1)) # (B x k) / (B,1)
      I_y = np.zeros_like(Z)
      I_y[np.arange(batch_size), y_batch] = 1
      grad = (1/batch_size)*(X_batch.T @ (Z - I_y))
      theta -= lr*grad
      
      start += batch_size

    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr = 0.1, batch=100):
    """ Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    n = X.shape[0]
    start = 0
    while start < n:
      batch_size = min(batch, n - start)
      X_batch = X[start:start + batch_size, :]
      y_batch = y[start:start+batch_size]
      
      A = X_batch @ W1 # activation for this batch
      Z1 = np.where(A > 0, A, 0) # relu with numpy
      I_y = np.zeros_like(Z1 @ W2)
      I_y[np.arange(batch_size), y_batch] = 1
      G2 = np.exp(Z1 @ W2)/(np.sum(np.exp(Z1 @ W2), axis=1).reshape(-1,1)) - I_y
      mask = np.where(Z1 > 0, 1, 0)
      G1 = mask * (G2 @ W2.T)

      W1_grad = (1/batch_size) * (X_batch.T @ G1)
      W2_grad = (1/batch_size) * (Z1.T @ G2)

      W1 -= lr * W1_grad
      W2 -= lr * W2_grad

      start += batch

    pass
    ### END YOUR CODE



### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT

def loss_err(h,y):
    """ Helper funciton to compute both loss and error"""
    return softmax_loss(h,y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100,
                  cpp=False):
    """ Example function to fully train a softmax regression classifier """
    theta = np.zeros((X_tr.shape[1], y_tr.max()+1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim = 500,
             epochs=10, lr=0.5, batch=100):
    """ Example function to train two layer neural network """
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr@W1,0)@W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te@W1,0)@W2, y_te)
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |"\
              .format(epoch, train_loss, train_err, test_loss, test_err))



if __name__ == "__main__":
    X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz",
                             "data/train-labels-idx1-ubyte.gz")
    X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                             "data/t10k-labels-idx1-ubyte.gz")

    print("Training softmax regression")
    train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.1)

    print("\nTraining two layer neural network w/ 100 hidden units")
    train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=100, epochs=20, lr = 0.2)
