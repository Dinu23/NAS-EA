# Neural Architecture Search - Evolutionary Algorithms

## Project Overview

This project explores the use of Evolutionary Algorithms for Neural Architecture Search (NAS) using the NAS-Bench-101 dataset. The goal is to automatically identify optimal neural network architectures for specific tasks by leveraging evolutionary computation techniques. The NAS-Bench-101 dataset provides a comprehensive mapping of neural architectures to their respective training and evaluation metrics, allowing for efficient evaluation of different architectures.

## Features

- **Genetic Algorithm**: Implements a genetic algorithm to optimize neural architectures.
- **Evolution Strategy**: Employs evolution strategies to find the best architecture.
- **NAS-Bench-101 Integration**: Utilizes the NAS-Bench-101 dataset for evaluating the performance of different architectures.

## Algorithms

### 1. Genetic Algorithm
The Genetic Algorithm involves:
- **Initialization**: Generating an initial population of valid architectures.
- **Selection**: Using tournament selection to choose parents for crossover based on their fitness.
- **Crossover**: Implementing both uniform and n-point crossover methods.
- **Mutation**: Flipping bits in the binary representation or adjusting the value of architecture parameters.
- **Environment Selection**: Using either comma (µ, λ) or plus (µ + λ) selection to create a new population.

### 2. Evolution Strategy
The Evolution Strategy includes:
- **Representation**: Encoding architectures in continuous space for optimization.
- **Recombination**: Using intermediary recombination to generate offspring.
- **Mutation**: Modifying the encoded representations with either single or individual step sizes.
- **Environment Selection**: Similar to the Genetic Algorithm, selecting new populations using comma or plus strategies.

## Requirements

- Python 3.x
- Libraries: numpy, scipy, NAS-Bench-101 dataset library

## Usage

1. Run the Genetic Algorithm:
   ```bash
   python genetic_algorithm.py
   ```
2. Run the Evolution Strategy:
   ```bash
   python evolution_strategy.py
   ```

## Files

- **genetic_algorithm.py**: Implementation of the Genetic Algorithm.
- **evolution_strategy.py**: Implementation of the Evolution Strategy.
- **README.md**: Project overview and instructions.

## Contact

For any inquiries, please contact:
- **Name**: Dinu Catalin Viorel
- **Email**: viorel.dinu00@gmail.com
