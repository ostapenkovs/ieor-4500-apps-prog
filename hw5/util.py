import numpy as np

def get_batches(data: np.ndarray, batch_size: int) -> np.ndarray:
    '''Get random batch of size batch_size from data.'''
    n = data.shape[0]
    k = n // batch_size
    # shuffle the indices
    ind = list(range(n))
    np.random.shuffle(ind)
    # trimming the samples to a multiple of batch_size
    ind = ind[:k*batch_size]
    # yield a batch one by one
    for i in range(0, k, batch_size):
        yield data[ind[i: i+batch_size], :]

def f(x: np.ndarray, theta: float, pi: float, **kwargs) -> np.ndarray:
    '''Function which we intend to minimize. Vectorized.'''
    t = kwargs['delta_centered'].shape[0]
    return -np.dot(kwargs['delta_bar'], x) + theta/t * np.linalg.norm(kwargs['delta_centered'] @ x, pi)**pi

def g(x: np.ndarray, theta: float, pi: float, **kwargs) -> np.ndarray:
    '''Gradient of function f. Vectorized.'''
    t = kwargs['delta_centered'].shape[0]
    delta_centered_at_x = kwargs['delta_centered'] @ x
    return -kwargs['delta_bar'] + (theta / t * pi) * ((delta_centered_at_x**(pi-1)).T @ kwargs['delta_centered'])

def gradient_descent(x_0: np.ndarray, theta: float, pi: float, alpha: float, beta: float, \
                     num_iter: int, batch_size: int, tolerance: float, notebook=False, **kwargs) -> tuple:
    '''Gradient descent function. Using gradient normalization, momentum, clipping, and batches. Vectorized.'''
    converged = False
    x = y = x_0
    # evaluating function at initial value
    # seeing if we can minimize this!
    f_val = f(x=x_0, theta=theta, pi=pi, delta_bar=kwargs['delta_bar'], delta_centered=kwargs['delta_centered'])
    hist = [f_val]

    if not notebook: from tqdm import tqdm
    else:            from tqdm.notebook import tqdm
    for iter in tqdm(range(num_iter), leave=True, desc='Iterations'):
        for batch in get_batches(data=kwargs['delta'], batch_size=batch_size):
            # computing gradient
            grad = g(x=x, theta=theta, pi=pi, delta_centered=kwargs['delta_centered'], delta_bar=kwargs['delta_bar'])
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
            f_new = f(x=x, theta=theta, pi=pi, delta_bar=kwargs['delta_bar'], delta_centered=kwargs['delta_centered'])
            hist.append(f_new)
            # checking convergence
            if f_val - f_new < tolerance:
                converged = True
                break
            # new function value is the old function value in the next iteration
            f_val = f_new
        
    # returning tuple of
    #   bool: did we converge?
    #   np.ndarray: our portfolio positions at optimality
    #   list: function value at each step of gradient descent
    return converged, x, hist

def main() -> None:
    '''@TODO'''
    pass

if __name__ == '__main__':
    main()
