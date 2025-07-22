import numpy as np
import itertools
from numpy.fft import rfft2
import itertools

def dict_combiner(mydict):
    """
    Combines the values of a dictionary into a list of dictionaries,
    where each dictionary represents a combination of the values.

    Args:
        mydict (dict): The input dictionary containing keys and lists of values.

    Returns:
        list: A list of dictionaries, where each dictionary represents a combination
              of the values from the input dictionary.

    Example:
        >>> mydict = {'A': [1, 2], 'B': [3, 4]}
        >>> dict_combiner(mydict)
        [{'A': 1, 'B': 3}, {'A': 1, 'B': 4}, {'A': 2, 'B': 3}, {'A': 2, 'B': 4}]
    """
    if mydict:
        keys, values = zip(*mydict.items())
        experiment_list = [dict(zip(keys, v))
                           for v in itertools.product(*values)]
    else:
        experiment_list = [{}]
    return experiment_list

class UnitGaussianNormalizer(object):
    """
    A class for normalizing data using unit Gaussian normalization.

    Attributes:
        mean (numpy.ndarray): The mean values of the input data.
        std (numpy.ndarray): The standard deviation values of the input data.
        eps (float): A small value added to the denominator to avoid division by zero.

    Methods:
        encode(x): Normalize the input data using unit Gaussian normalization.
        decode(x): Denormalize the input data using unit Gaussian normalization.
    """

    def __init__(self, x, eps=1e-5):
        super(UnitGaussianNormalizer, self).__init__()

        self.mean = np.mean(x, 0)
        self.std = np.std(x, 0)
        self.eps = eps

    def encode(self, x):
        """
        Normalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The input data to be normalized.

        Returns:
            numpy.ndarray: The normalized data.
        """
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x):
        """
        Denormalize the input data using unit Gaussian normalization.

        Args:
            x (numpy.ndarray): The normalized data to be denormalized.

        Returns:
            numpy.ndarray: The denormalized data.
        """
        return (x * (self.std + self.eps)) + self.mean


def subsample_and_flatten(matrix, stride):
    """
    Subsamples a matrix and flattens it into a 1D array.
    The subsampling is done by extracting the first and last rows and columns,
    and then extracting the interior elements based on the stride.
    The extracted elements are then flattened into a 1D array.
    The function returns the indices of the extracted elements and the flattened array.
    """
    # matrix: a 3D numpy array of shape (N, rows, cols)
    # stride: an integer representing the stride

    # Get the dimensions of the input matrix
    N, rows, cols = matrix.shape

    # Create a list to store the indices of the elements to be extracted
    indices = []

    # Add indices for the first row (left to right)
    indices.extend((0, j) for j in range(cols))

    # Add indices for the last row (left to right)
    if rows > 1:
        indices.extend((rows - 1, j) for j in range(cols))

    # Add indices for the first column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, 0) for i in range(1, rows - 1))

    # Add indices for the last column (top to bottom, excluding corners)
    if rows > 2:
        indices.extend((i, cols - 1) for i in range(1, rows - 1))

    # print(indices)
    # Generate indices for the interior elements based on the stride
    counter = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if counter % stride == 0:
                indices.append((i, j))
                counter = 0
            counter += 1

    # sort the indices
    indices.sort(key=lambda x: (x[0], x[1]))

    # Extract the elements from the matrix using the sorted indices
    result = [matrix[:, i, j] for i, j in indices]

    return np.array(indices).astype('float32'), np.array(result).T.astype('float32')

def patch_coords(matrix):
    """
    Generate coordinates for each element in the matrix.

    Args:
        matrix (ndarray): Input matrix.

    Returns:
        ndarray: Array of coordinates for each element in the matrix.
    """
    N, rows, cols = matrix.shape

    nx = np.linspace(0,1,num=cols)
    ny = np.linspace(0,1,num=rows)

    coords = np.array(np.meshgrid(nx,ny))
    
    return coords

# plotting functions for Lorentz system predictions

import numpy as np
import matplotlib.pyplot as plt
from tslearn.metrics import dtw
from scipy.stats import pearsonr

def plot_lorenz_prediction_vs_truth(
    t, x_true, y_true, z_true, y_pred, z_pred, 
    title_prefix='', 
    model_hparams=''
):
    """
    Plot Lorenz system predictions vs ground truth for a single trajectory,
    and compute DTW distance and Pearson correlation for y and z.

    Args:
        t (np.ndarray): Time array.
        x_true (np.ndarray): True x values.
        y_true (np.ndarray): True y values.
        z_true (np.ndarray): True z values.
        y_pred (np.ndarray): Predicted y values.
        z_pred (np.ndarray): Predicted z values.
        title_prefix (str): Optional prefix for the plot title.
        model_hparams (str): Model hyperparameters to display in the title.
    """
    # ——— Compute new similarity metrics ———
    # 1) DTW distance
    dtw_y = dtw(y_true, y_pred)
    dtw_z = dtw(z_true, z_pred)

    # 2) Pearson correlation coefficient
    corr_y, _ = pearsonr(y_true, y_pred)
    corr_z, _ = pearsonr(z_true, z_pred)

    # Print them
    print(f"{title_prefix}DTW y: {dtw_y:.4e}, Pearson r y: {corr_y:.4f}")
    print(f"{title_prefix}DTW z: {dtw_z:.4e}, Pearson r z: {corr_z:.4f}")

    # ——— Plotting ———
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t, x_true, label='True x')
    axs[0].set_ylabel('x')
    axs[0].legend()

    axs[1].plot(t, y_true, label='True y')
    axs[1].plot(t, y_pred, '--', label='Pred y')
    axs[1].set_ylabel('y')
    axs[1].legend()

    axs[2].plot(t, z_true, label='True z')
    axs[2].plot(t, z_pred, '--', label='Pred z')
    axs[2].set_xlabel('Time')
    axs[2].set_ylabel('z')
    axs[2].legend()

    # Super title with hyperparameters
    plt.suptitle(
        f"{title_prefix}Prediction vs Ground Truth\n"
        f"Model: {model_hparams}",
        fontsize=14
    )

    # Text box with the new metrics
    metrics_text = (
        f"DTW y: {dtw_y:.2e}, Pearson r y: {corr_y:.2f}\n"
        f"DTW z: {dtw_z:.2e}, Pearson r z: {corr_z:.2f}"
    )
    plt.gcf().text(
        0.99, 0.01, metrics_text, fontsize=12, ha='right', va='bottom',
        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray')
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()