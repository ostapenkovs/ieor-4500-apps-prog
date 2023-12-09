import numpy as np
np.random.seed(1)

from functools import partial
import multiprocessing as mp
from tqdm import tqdm

import pickle
import matplotlib.pyplot as plt
import seaborn as sns

def get_batches(n: int, batch_size: int) -> np.ndarray:
    '''Get random batch of indices of size batch_size.'''
    k = n // batch_size
    # shuffle the indices
    ind = list(range(n))
    np.random.shuffle(ind)
    # trimming the sample indices to a multiple of batch_size
    ind = ind[:k*batch_size]
    # yield a batch of indices one by one
    for i in range(0, k, batch_size):
        yield ind[i: i+batch_size]

def f(x: np.ndarray, theta: float, pi: float, delta_bar: np.ndarray, delta_centered: np.ndarray) -> np.ndarray:
    '''Function which we intend to minimize. Vectorized.'''
    t = delta_centered.shape[0]
    return -np.dot(delta_bar, x) + theta/t * np.linalg.norm(delta_centered @ x, pi)**pi

def g(x: np.ndarray, theta: float, pi: float, delta_bar: np.ndarray, delta_centered: np.ndarray) -> np.ndarray:
    '''Gradient of function f. Vectorized.'''
    t = delta_centered.shape[0]
    delta_centered_at_x = delta_centered @ x
    return -delta_bar + (theta / t * pi) * ((delta_centered_at_x**(pi-1)).T @ delta_centered)

def gradient_descent(x_0: np.ndarray, theta: float, pi: float, alpha: float, beta: float, \
                     num_iter: int, batch_size: int, tolerance: float, delta_bar: np.ndarray, delta_centered: np.ndarray) -> tuple:
    '''Gradient descent function. Using gradient normalization, momentum, clipping, and batches. Vectorized.'''
    converged = False
    n = delta_centered.shape[0]
    x = y = x_0
    # evaluating function at initial value
    # seeing if we can minimize this!
    f_val = f(x=x_0, theta=theta, pi=pi, delta_bar=delta_bar, delta_centered=delta_centered)
    hist = [f_val]

    num_workers = mp.cpu_count()
    pool = mp.Pool(num_workers)

    for iter in tqdm(range(num_iter), leave=True, desc='Iterations'):
        # computing gradient IN PARALLEL
        frozen_grad = partial(g, x, theta, pi, delta_bar)
        grad = pool.map(
            frozen_grad,
            [delta_centered[index_batch, :] for index_batch in get_batches(n=n, batch_size=batch_size)]
        )
        # averaging the gradient from all the processes
        grad = sum(grad) / len(grad)
        # norming gradient: why?
        #   1) we are able to control the step-size purely with beta and 
        #       alpha without being influenced by varying magnitude of gradient
        #   2) we are able to eliminate "slow crawling" gradient descent and
        #       push out of flat regions of the surface faster
        grad /= np.linalg.norm(grad)
        # computing next step using momentum
        y = beta*y + (1-beta)*grad
        # clipping the gradient to enforce the problem's constraint
        x = np.clip(x - alpha*y, -1, 1)
        # evaluating new function value
        f_new = f(x=x, theta=theta, pi=pi, delta_bar=delta_bar, delta_centered=delta_centered)
        hist.append(f_new)
        # checking convergence
        if f_val - f_new < tolerance:
            converged = True
            break
        # new function value is the old function value in the next iteration
        f_val = f_new
    
    pool.close()
    pool.join()
    # returning tuple of
    #   bool: did we converge?
    #   np.ndarray: our portfolio positions at optimality
    #   list: function value at each step of gradient descent
    return converged, x, hist

def main() -> None:
    '''Testing our gradient descent (parallelized + vectorized).'''
    ### READING IN THE DATA ####
    data_folder = './data'
    index_name = 'closeRussell1000'

    # stuff that is used as input to the gradient descent function
    data = dict()
    file_endings = ['delta_bar', 'delta_centered']
    for end in file_endings:
        with open(f'{data_folder}/{index_name}_{end}.pkl', 'rb') as f:
            data[end] = pickle.load(f)

    n, p = data['delta_centered'].shape

    with open(f'{data_folder}/{index_name}_pair_names.pkl', 'rb') as f:
        pair_names = pickle.load(f)
    ############################

    ### TESTING OUR GRADIENT DESCENT ###
    # initial feasible guess
    x_0 = np.random.uniform(-1, 1, p)

    converged, x, hist = gradient_descent(
            x_0=x_0,
            theta=10, pi=2, alpha=1e-3, beta=0.9,
            num_iter=5, batch_size=32, tolerance=1e-6,
            **data
        )
    ####################################

    ### INSPECTING / PLOTTING THE RESULTS ###
    print(converged)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(hist, ax=ax[0])
    sns.histplot(x, ax=ax[1])
    plt.tight_layout()
    plt.show()
    #########################################

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
