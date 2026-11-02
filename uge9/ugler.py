import numpy as np
import matplotlib.pyplot as plt

def owl_population(time_steps, p0, change_matrix):
    """
    Given timesteps, a starting population and a matrix of how the population
    changes with respect to each time step, compute the population recursively.

    :param time_steps: discretized time steps
    :param p0: initial population
    :param change_matrix: matrix of how each time step changes

    """
    n = len(time_steps)
    population = np.zeros((3, n))
    population[:, 0] = p0

    # Compute recursively

    for i in range(1, n):
        population[:, i] = change_matrix @ population[:, i - 1]

    # Compute using "Advanced techniques" (eigenvalues and eigenvectors)
    eig_val, eig_vec = np.linalg.eig(change_matrix)
    # Do your analysis and find dominating real eigenvalues
    eig_val = sorted(eig_val, key=lambda x: np.abs(np.real(x)), reverse=True)[0]
    pop_func = lambda t: eig_val**t*p0[:,None]
    population_eig = pop_func(time_steps)

    # Plot the series

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, population[0, :], label='Juvenile', color='blue')
    plt.plot(time_steps, population[1, :], label='Subadult', color='orange')
    plt.plot(time_steps, population[2, :], label='Adult', color='green')
    plt.plot(time_steps, population_eig[0,:], label='EigenJuvenile', color='blue', linestyle='dashed')
    plt.plot(time_steps, population_eig[1,:], label='EigenSubadult', color='orange', linestyle='dashed')
    plt.plot(time_steps, population_eig[2,:], label = 'EigenAdult', color='green', linestyle='dashed')
    plt.title('Owl Population Over Time')
    plt.xlabel('Time Step')
    plt.ylabel('Population')
    plt.legend()
    plt.show()
    return population

M = np.array([[0, 0, 0.33],
              [0.18, 0, 0],
              [0, 0.71, 0.94]])

print(np.linalg.eig(M))

p0 = np.array([900, 500, 2600])

time_steps = np.arange(0, 100, 1)

owl_population = owl_population(time_steps, p0, M)


