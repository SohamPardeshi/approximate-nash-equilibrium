import math
import numpy as np
import itertools


def approximate_equilibrium(row_payoff, col_payoff, epsilon, max_k=30):
    """
    Returns an tuple representing a strategy pair that is an epsilon-approximate
    Nash equilibrium solution based on the given payoff matrices. 
    """
    # Do some error checking
    if isinstance(row_payoff, list) and isinstance(col_payoff, list):
        row_payoff = np.array(row_payoff)
        col_payoff = np.array(col_payoff)

    if row_payoff.shape != col_payoff.shape:
        raise ValueError("Payoff matrices differ in shape!")

    row_shape = row_payoff.shape
    if len(row_shape) != 2 or row_shape[0] != row_shape[1]:
        raise ValueError("Payoff matrix is not a square matrix")

    # Calculate our parameters
    n = row_shape[0]
    k = math.ceil((12 * math.log(n)) / (epsilon * epsilon))
    print("Chose k = ", k)

    # Normalize our payoff matrices and epsilon
    row_payoff = normalize(row_payoff)
    col_payoff = normalize(col_payoff)

    if max_k is not None and k > max_k:
        print(f"k={k} has too long of an expected runtime. Switching to k={max_k}")
        k = max_k

    # Calculate the k multisets
    indices = np.arange(n)
    row_multisets = itertools.product(indices, repeat=k)
    col_multisets = itertools.product(indices, repeat=k)

    # Iterate over the (k^2) strategies profiles
    for row_multiset in row_multisets:
        for col_multiset in col_multisets:
            
            # Convert the pair of multisets into mixed strategies
            row_strategy = _to_mixed_strategy(row_multiset, k, n)
            col_strategy = _to_mixed_strategy(col_multiset, k, n)

            # Spend O(n)  to check whether our solution is correct
            if _is_approximate(row_strategy, col_strategy, row_payoff, epsilon, n):
                if _is_approximate(col_strategy, row_strategy, col_payoff.T, epsilon, n):          
                    return (row_strategy, col_strategy)

    # Should never happen
    return None

def _is_approximate(row_strategy, col_strategy, payoff, epsilon, n, debug=False):
    """
    Returns True if the given k-uniform mixed strategy is an epsilon-approximate solution.
    Returns False otherwise.
    """
    col = np.dot(payoff, col_strategy)
    row_reward = np.inner(row_strategy, col)

    deviations = np.identity(n)
    pure_rewards = np.inner(deviations, col)

    if debug:
        print(pure_rewards - row_reward)

    return np.all(np.abs(pure_rewards - row_reward) <= epsilon)

def normalize(matrix):
    """
    Returns the given numpy array with values scaled to [0, 1]
    """
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    scaling_factor = (max_val - min_val)

    normalized = (matrix - min_val) / scaling_factor
    return normalized


def _to_mixed_strategy(multiset, k, size):
    """
    Returns a k-uniform mixed strategy based on the given multiset
    """
    # Convert to numpy array for speed
    multiset = np.array(multiset)

    # Track the probability of events
    values, alpha = np.unique(multiset, return_counts=True)

    # Convert to mixed strategy
    strategy = np.zeros(size)
    strategy[values] = alpha / k

    return strategy


def main():
    print("Try running the test function instead!")

if __name__ == '__main__':
    main()