# PSARROS FILIPPOS
import numpy as np

# Definition of the Sphere function
# This function computes the sum of squares of a vector's elements
# It is one of the simplest optimization benchmark functions
def sphere_function(x):
    return np.sum(x**2)

# Definition of the Sum of Squares function
# Each term has a different weight depending on its position
def sum_of_squares_function(x):
    return np.sum([(i + 1) * x[i]**2 for i in range(len(x))])

# Definition of the Matyas function
# A smooth function with a unique global minimum at (0, 0)
def matyas_function(x):
    return 0.26 * (x[0]**2 + x[1]**2) - 0.48 * x[0] * x[1]

# Definition of the Exponential Decay function
# A simple exponential function for optimization
def exponential_decay_function(x):
    return np.sum(np.exp(x))

# Definition of the Booth function
# A simple function with parabolic characteristics
def booth_function(x):
    return (x[0] + 2 * x[1] - 7)**2 + (2 * x[0] + x[1] - 5)**2

# Gradient Descent implementation for local minimization
# Bounds are enforced to keep the solution within the valid range
# func: the function to minimize
# grad_func: function returning the gradient
# x_init: initial solution
# learning_rate: step size
# max_iter: maximum number of iterations
# tol: stopping tolerance
# bounds: search bounds
def gradient_descent(func, grad_func, x_init, bounds, learning_rate=0.01, max_iter=100, tol=1e-6):
    x = x_init.copy()  # Copy of the initial solution
    for _ in range(max_iter):
        grad = grad_func(x)  # Compute gradient
        x -= learning_rate * grad  # Update position
        x = np.clip(x, [b[0] for b in bounds], [b[1] for b in bounds])  # Clip within bounds
        if np.linalg.norm(grad) < tol:  # Check stopping condition
            break
    return x

# Genetic Algorithm implementation
# func: function to optimize
# bounds: parameter bounds
# pop_size: population size
# generations: number of generations
# mutation_rate: mutation probability
# epsilon: stopping threshold for fitness difference
def genetic_algorithm(func, bounds, pop_size=50, generations=100, mutation_rate=0.1, epsilon=1e-6):
    num_params = len(bounds)  # Number of parameters

    # Initialize population within bounds
    population = np.random.uniform(
        [b[0] for b in bounds], [b[1] for b in bounds], (pop_size, num_params)
    )
    velocity = np.zeros_like(population)  # Initialize velocity for PSO-inspired mutation
    best_solution = None  # Store best solution
    best_fitness = float('inf')  # Initialize best fitness

    for generation in range(generations):
        # Compute fitness of each chromosome
        fitness = np.array([func(ind) for ind in population])
        best_gen_fitness = np.min(fitness)  # Best fitness of current generation
        best_gen_solution = population[np.argmin(fitness)]  # Best chromosome of current generation

        # Update global best solution
        if best_gen_fitness < best_fitness:
            best_fitness = best_gen_fitness
            best_solution = best_gen_solution

        # Stop if fitness difference is below threshold
        if np.abs(np.max(fitness) - np.min(fitness)) < epsilon:
            break

        # Select parents based on probabilities
        probabilities = 1 / (fitness + 1e-6)  # Inverted fitness (lower is better)
        probabilities /= probabilities.sum()  # Normalize
        parents_indices = np.random.choice(range(pop_size), size=pop_size, p=probabilities)
        parents = population[parents_indices]

        # Crossover to generate children
        children = []
        for i in range(0, pop_size, 2):
            p1, p2 = parents[i], parents[(i + 1) % pop_size]  # Parent pair
            alpha = np.random.uniform(0, 1, size=num_params)  # Random coefficient for weighted avg
            child1 = alpha * p1 + (1 - alpha) * p2
            child2 = alpha * p2 + (1 - alpha) * p1
            children.extend([child1, child2])

        children = np.array(children)  # Convert to NumPy array

        # PSO-inspired mutation
        for i in range(pop_size):
            if np.random.rand() < mutation_rate:  # Mutation chance
                c1, c2 = 1.5, 1.5  # Acceleration coefficients
                r1, r2 = np.random.rand(), np.random.rand()  # Random values
                personal_best = children[i]  # Assumed personal best
                velocity[i] = (
                    velocity[i]
                    + c1 * r1 * (personal_best - children[i])  # Influence of personal best
                    + c2 * r2 * (best_solution - children[i])  # Influence of global best
                )
                children[i] += velocity[i]  # Update position

        # Update population with clipped values within bounds
        population = np.clip(children, [b[0] for b in bounds], [b[1] for b in bounds])

    return best_solution, best_fitness  # Return best solution and fitness

# Example usage
if __name__ == "__main__":
    # List of test functions
    test_functions = [
        ("Sphere", sphere_function, lambda x: 2 * x, [(-5, 5)] * 2),
        ("Sum of Squares", sum_of_squares_function, lambda x: np.array([(i + 1) * 2 * x[i] for i in range(len(x))]), [(-10, 10)] * 2),
        ("Matyas", matyas_function, lambda x: np.array([0.52 * x[0] - 0.48 * x[1], 0.52 * x[1] - 0.48 * x[0]]), [(-10, 10)] * 2),
        ("Exponential Decay", exponential_decay_function, lambda x: np.exp(x), [(-5, 5)] * 2),
        ("Booth", booth_function, lambda x: np.array([
            2 * (x[0] + 2 * x[1] - 7) + 4 * (2 * x[0] + x[1] - 5),
            4 * (x[0] + 2 * x[1] - 7) + 2 * (2 * x[0] + x[1] - 5)
        ]), [(-10, 10)] * 2),
    ]

    for name, func, grad_func, bounds in test_functions:
        print(f"\nTesting {name} Function")
        # Run Genetic Algorithm
        best_sol, best_fit = genetic_algorithm(func, bounds, pop_size=100, generations=200)
        print(f"Best solution from GA: {best_sol}, fitness: {best_fit}")

        # Apply Gradient Descent if gradient exists
        if grad_func:
            optimized_sol = gradient_descent(func, grad_func, best_sol, bounds, learning_rate=0.1)
            print(f"Optimized solution after Gradient Descent: {optimized_sol}, fitness: {func(optimized_sol)}")
