# Optimal schedule algorithm given market impact model.
# Fastest runtime as of 10.03.2023.

import numpy as np

def optimal_algo(z: float, alpha: float, pi: float, T: int, N: int | float) -> list:
    '''Computes optimal share sales schedule for the given market impact model.
    M(t) = ceil[ 0.1*M(t-1) + 0.9*N(t) ];
    S(t) = ceil[ (1 - alpha*M(t)^pi) * N(t) ].'''

    N = int(N)
    S = np.zeros((T+1, N+1, N+1), dtype='i')
    H = np.zeros((T, N+1, N+1), dtype='i')
    
    shares = np.arange(N+1, dtype='i')
    rev_shares = N - shares
    res_shares = shares.reshape(-1, 1)
    impact = 1-alpha*shares**pi

    for t in range(T-1, -1, -1):
        for n in range(0, N+1, 1):
            temp_m = np.ceil(z*res_shares + (1-z)*shares[: n+1]).astype(int)
            temp_sell = np.ceil( impact[ temp_m ] * shares[: n+1] ) + S[t+1, temp_m, rev_shares[N-n: ]]

            idx_max = np.argmax( temp_sell, axis=1 )
            S[t, :, n] = temp_sell[shares, idx_max]
            H[t, :, n] = idx_max
            
            temp_m = temp_sell = idx_max = None
    S = shares = rev_shares = res_shares = impact = None

    schedule = list()
    m, remaining = 0, N
    for t in range(T):
        nt = H[t, m, remaining]
        schedule.append(nt)
        m = np.ceil(z*m + (1-z)*nt).astype(int)
        remaining = int(remaining - nt)
    return schedule

def run_sim(z: float, alpha: float, T: int, N: int | float, several_pi: list | np.ndarray, notebook=False, F=None) -> dict:
    '''Run algo simulation for a range of pi values.'''
    
    if not notebook: from tqdm import tqdm
    else:            from tqdm.notebook import tqdm
    
    schedules = dict()
    pbar = tqdm(several_pi, leave=True)
    
    for pi in pbar:
        pi = round(pi, 2)
        pbar.set_description(f'PI = { pi }')
        
        if F is None:
            schedules[pi] = optimal_algo(z=z, alpha=alpha, pi=pi, T=T, N=N)
        else:
            schedules[pi] = [ n*(10**F) for n in optimal_algo(z=z, alpha=alpha*10**(F*pi), pi=pi, T=T, N=N/(10**F)) ]
    
    pbar.close()
    return schedules

def main() -> None:
    '''Testing the algorithm via simulation for some parameters.'''

    print('### EXACT ALGO ###')
    
    schedules = run_sim(z=0.1, alpha=1e-2, T=5, N=1e3, several_pi=[0.3, 0.5, 0.7], notebook=False, F=None)
    
    print()
    
    for pi, schedule in schedules.items():
        print(f'{ pi }: { str(schedule) }')
        print()

    print('### APPROX ALGO USING F = 1 ###')

    schedules = run_sim(z=0.1, alpha=1e-2, T=5, N=1e3, several_pi=[0.3, 0.5, 0.7], notebook=False, F=1)

    print()
    
    for pi, schedule in schedules.items():
        print(f'{ pi }: { str(schedule) }')
        print()

if __name__ == '__main__':
    main()
