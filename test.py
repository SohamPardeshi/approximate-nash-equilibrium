import nashpy as nash
import numpy as np
from lmm import approximate_equilibrium, normalize

def test1():
    """
    Tests against the class rock-paper-scissors example. It can be easily shown that 
    the best mixed strategy is to choose each option uniformly at random. This provides
    an expected payout of 
    """
    print("Test 1")
    # Use existing library to benchmark results on RPS
    row_payoff = normalize(np.array([[0, -1, 1], [1, 0, -1], [-1, 1, 0]]))
    col_payoff = row_payoff.T
    rps = nash.Game(row_payoff)

    # Calculate my approximation
    epsilon = 0.5
    row_strat, col_strat = approximate_equilibrium(row_payoff, col_payoff, epsilon, max_k=None)

    # What is our expected reward?
    reward = rps[row_strat, col_strat]
    print("Approx.:", reward, row_strat, col_strat)

    # What is the true Nash equilibria reward? We are close to one of them
    for row_opt, col_opt in list(rps.support_enumeration()):
        opt_reward = rps[row_opt, col_opt]
        print("Exact:  ", opt_reward, row_opt, col_opt)
        if np.all(np.abs(reward - opt_reward) <= epsilon):
            return

    # Uh oh! We were close to none of them
    assert False

def test2():
    """
    Tests against the class rock-paper-scissors example. It can be easily shown that 
    the best mixed strategy is to choose each option uniformly at random. This provides
    an expected payout of 
    """
    print()
    print("Test 2")
    # Use existing library to benchmark results on RPS
    row_payoff = normalize(np.array([[3, 0], [5, 1]]))
    col_payoff = normalize(np.array([[3, 5], [0, 1]]))
    rps = nash.Game(row_payoff, col_payoff)

    # Calculate my approximation
    epsilon = 0.4
    row_strat, col_strat = approximate_equilibrium(row_payoff, col_payoff, epsilon, max_k=None)

    # What is our expected reward?
    reward = rps[row_strat, col_strat]
    print("Approx.:", reward, row_strat, col_strat)

    # What is the true Nash equilibria reward? We are close to one of them
    for row_opt, col_opt in list(rps.support_enumeration()):
        opt_reward = rps[row_opt, col_opt]
        print("Exact:  ", opt_reward, row_opt, col_opt)

        if np.all(np.abs(reward - opt_reward) <= epsilon):
            return
    
    # Uh oh! We were close to none of them
    assert False


def main():
    test1()
    test2()

if __name__ == '__main__':
    main()