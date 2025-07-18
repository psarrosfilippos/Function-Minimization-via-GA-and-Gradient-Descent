## Function Minimization via GA and Gradient Descent

This project implements a hybrid optimization framework that combines a Genetic Algorithm (GA) with Gradient Descent (GD) for solving continuous unconstrained optimization problems. It supports multiple benchmark functions including Sphere, Sum of Squares, Matyas, Exponential Decay, and Booth functions. The GA uses a PSO-inspired mutation strategy to enhance exploration, while GD refines the best-found solution locally.

The implementation is written in Python using NumPy, and demonstrates an efficient approach to balancing global and local search in numerical optimization tasks.

## Features

**Implementation of multiple benchmark optimization functions:**
  - Sphere
  - Sum of Squares
  - Matyas
  - Exponential Decay
  - Booth

**Two optimization algorithms included:**
  - Genetic Algorithm (GA) with crossover, mutation (inspired by PSO), and adaptive selection
  - Gradient Descent (GD) with boundary constraints and convergence checks

**Hybrid optimization pipeline:**

  - Use GA to find a good initial solution
  - Refine it further using GD

**Built-in test suite to evaluate performance on each benchmark function**

**Modular and extensible code structure, suitable for experimenting with other functions or optimization strategies**

## Technologies Used

- **Python 3.10+ – Primary programming language**

- **NumPy – Efficient numerical and matrix operations**

- **Evolutionary Computation – Genetic Algorithm for global optimization**

- **Gradient-Based Optimization – Gradient Descent for local refinement**

- **Benchmark Functions – Sphere, Sum of Squares, Matyas, Booth, Exponential Decay**
## How to use

**1. Clone the repository or download the** **`.py` file:**

    bash
    git clone https://github.com/your-username/optimization-algorithms.git
    cd optimization-algorithms

**2. Install dependencies (only NumPy is required):**

    bash
    pip install numpy

**3. Run the script to test all included optimization functions:**

    bash
    python optimization.py

**4. Observe the results printed in the terminal, showing:**

- Best solution found by the Genetic Algorithm

- Final refined solution after Gradient Descent

- Corresponding fitness values

## Examples

This script demonstrates how to use a Genetic Algorithm combined with Gradient Descent to optimize several benchmark functions.

Example output for the Booth function:

    Testing Booth Function
    Best solution from GA: [1.00002 2.99999], fitness: 1.2e-05
    Optimized solution after Gradient Descent: [1. 3.], fitness: 0.0

You can modify the list of test functions or add your own custom objective function inside the __main__ section:

    def custom_function(x):
    return x[0]**2 + x[1]**2 + 10

    test_functions.append((
    "Custom", 
    custom_function, 
    lambda x: np.array([2*x[0], 2*x[1]]), 
    [(-10, 10)] * 2
    ))

Run the script again to include your function in the optimization pipeline.
## Acknowledgements

This project was developed as part of my final-year coursework during the 4th year of my undergraduate studies in Computer Science.

I would like to express my sincere gratitude to my professor for his guidance, support, and valuable feedback throughout the development of this work. His input was instrumental in refining both the implementation and the theoretical approach.

Special appreciation also goes to the academic staff and fellow students whose discussions helped shape this project, as well as to the open-source community and the maintainers of the NumPy library, whose tools enabled the creation of this optimization framework.




## Authors

Filippos Psarros

informatics and telecommunications Student

GitHub: psarrosfilippos
[README.md](https://github.com/user-attachments/files/21315293/README.md)[Uploading READM
E.md…]()
