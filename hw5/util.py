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
                     num_iter: int, batch_size: int, tolerance: float, num_workers: int, delta_bar: np.ndarray, delta_centered: np.ndarray) -> tuple:
    '''Gradient descent function. Using multiprocessed batches, gradient normalization, momentum, clipping. Vectorized.'''
    # initial parameters to start gradient descent
    converged = False
    x = y = x_0
    hist = list()
    
    # parameter for getting batches and process pool initialization
    n = delta_centered.shape[0]
    pool = mp.Pool(num_workers)
    
    for _ in tqdm(range(num_iter), leave=True, desc='Iterations'):
        f_curr = f(x=x, theta=theta, pi=pi, delta_bar=delta_bar, delta_centered=delta_centered)
        hist.append( f_curr )
        # computing gradient IN PARALLEL
        frozen_grad = partial(g, x, theta, pi, delta_bar)
        grad = pool.map(
            frozen_grad,
            [delta_centered[index_batch, :] for index_batch in get_batches(n=n, batch_size=batch_size)]
        )
        # averaging the gradient from all the processes
        grad = sum(grad) / len(grad)
        # checking convergence right here
        norm = np.linalg.norm(grad)
        if norm < tolerance:
            converged = True
            break
        # if we didn't converge, we take a step
        # norming gradient: why?
        #   1) we are able to control the step-size purely with beta and 
        #       alpha without being influenced by varying magnitude of gradient
        #   2) we are able to eliminate "slow crawling" gradient descent and
        #       push out of flat regions of the surface faster
        grad /= norm
        # computing next step using momentum
        y = beta*y + (1-beta)*grad
        # clipping the gradient to enforce the problem's constraint
        x = np.clip(x - alpha*y, -1, 1)
    
    pool.close()
    pool.join()
    # returning tuple of
    #   bool: did we converge?
    #   np.ndarray: our portfolio positions at optimality
    #   list: function value at each step of gradient descent
    return converged, x, hist

def main() -> None:
    '''Testing our gradient descent (parallelized + vectorized).'''
    ### READING IN THE DATA #################
    data_folder = './data'
    index_name = 'closeRussell1000'

    data = dict()
    file_endings = ['delta_bar', 'delta_centered']
    for end in file_endings:
        with open(f'{data_folder}/{index_name}_{end}.pkl', 'rb') as f:
            data[end] = pickle.load(f)
    #########################################

    ### TESTING OUR GRADIENT DESCENT ########
    # initial feasible guess
    p = data['delta_centered'].shape[1]
    # using all available cores
    num_workers = mp.cpu_count()
    print(f'Using {num_workers} workers.')

    converged, x, hist = gradient_descent(
            x_0=np.random.uniform(-1, 1, p),
            theta=10, pi=2, alpha=1e-1, beta=0.95,
            # 128 is the fastest empirically tested batch size
            num_iter=200, batch_size=128, tolerance=1e-8,
            num_workers=num_workers,
            **data
        )
    
    data = None
    #########################################

    ### INSPECTING / PLOTTING THE RESULTS ###
    print(f'Did we converge? {converged}. Final function value: {hist[-1]}')

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # For the lineplot, create an array for the x-axis (e.g., iterations)
    iterations = np.arange(len(hist))
    sns.lineplot(x=iterations, y=hist, ax=ax[0])
    ax[0].set_title('Function Value Over Iterations')
    ax[0].set_xlabel('Iteration')
    ax[0].set_ylabel('Function Value')

    # For the histogram
    sns.histplot(x, ax=ax[1])
    ax[1].set_title('Histogram of x Values')
    ax[1].set_xlabel('x Value')
    ax[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.show()


    with open(f'{data_folder}/{index_name}_pair_names.pkl', 'rb') as f:
        pair_names = np.array( pickle.load(f) )
    
    cutoff = 0.75
    mask = np.abs(x) > cutoff
    # print( pair_names[mask] )
    print(f'Num. pairs passing the cutoff: {sum(mask)}')
    #########################################

if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()
