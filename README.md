# Effects of Genetic Algorithm Hyperparameters on Neural Network Evolution for 2D Rocket Landing Control

**Team Members:** Tyler Garriott, David Long, Gavin Picard

This project serves as our final project for our COSC420/527 course, Introduction to Biologically-Inspired Computing, at The University of Tennessee, Knoxville.

### Research Question
**RQ1:** How do genetic algorithm hyperparameters (population size, tournament size, crossover rate, mutation rate, mutation sigma, and elitism count) affect the convergence speed, landing success rate, and evolved policy quality of a neuroevolution system for 2D rocket landing?

### Description
This project investigates how the hyperparameters of a genetic algorithm affect its ability to evolve a neural network that produces control policies for a 2D rocket landing task. A physics simulation with gravity, drag, wind, and fuewl constraints provides the environment. Each genome encodes the weights of a feedforward neural network; the GA evolves this population using tournament selection, uniform crossover, Gaussian mutation, and elitism. A novelty search bonus encourages behavioral diversity. We vary the GA parameters and measure their effect on fitness convergence, landing success rate, and fuel efficiency.

### Parameters Under Investigation (Default Values)
- POPULATION_SIZE = 100
- NUM_GENERATIONS = 300
- TOURNAMENT_SIZE = 5
- CROSSOVER_RATE = 0.7
- MUTATION_RATE = 0.1         
- MUTATION_SIGMA = 0.1        
- ELITISM_COUNT = 5
