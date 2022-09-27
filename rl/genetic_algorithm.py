from attrs import define, field
from typing import Callable
import numpy as np

@define
class GeneticAlgorithm:
    '''
        Genetic Algorithm

        Creates a population of size `population_size`, where each individual is 
        a segment of `chromosone_length` numbers, where the numbers range from 0 to `n_gene_variants`.

        Fitness is determined by passing in the function `fitness_fn`, which will take in the entire population 
        and compute the fitness of the current population.

    
    '''
    population_size : int
    n_gene_variants : int
    n_parents : int  
    n_mutations : int
    chromosone_length : int
    max_iterations : int = field(default=100)
    fitness_fn : Callable[[list[int]], np.ndarray] = field(factory=lambda x: np.array(x))
    crossover_point : int = field()
    rng : np.random.Generator = field(factory=lambda seed=None: np.random.default_rng(seed))
    _population : np.ndarray = field()
    _fitness : np.ndarray = field(init=False)
    _iterations : int = field(init=False, default=0)

    @crossover_point.default
    def _check_valid_crossover_point(self, crossover_point=None):
        if crossover_point is None:
            crossover_point = self.chromosone_length // 2
        elif crossover_point >= self.chromosone_length:
            raise ValueException("Crossover point cannot be greater than chromosone length!")
        return crossover_point


    @_population.default
    def generate_population(self):
        return self.rng.integers(self.n_gene_variants, size=(self.population_size, self.chromosone_length))

    @_fitness.default
    def compute_initial_fitness(self):
        return self.compute_fitness(self._population)

    def compute_fitness(self, population):
        return self.fitness_fn(population)


    def selection(self, population, weights):
        return self.rng.choice(population, p=weights/weights.sum(), size=self.n_parents, axis=1)
    
    # Breed between two pairs of parents
    def crossover(self, parents):
        new_generation = np.zeros_like([self.n_parents, population.shape[1]])
        left_indices = np.shuffle(np.arange(self.n_parents))
        right_indices = np.shuffle(np.arange(self.n_parents))
        new_generation[:, :self.crossover_point] = parents[left_indices, :self.crossover_point]
        new_generation[:, self.crossover_point:] = parents[right_indices, self.crossover_point:]
        return new_generation 

    def mutation(self, new_generation):
        mutated_indices = self.rng.integers(self.population_size, size=(self.n_mutations,))         
        mutated_gene_indices = self.rng.integers(self.chromosone_length, size=(self.n_mutations,))
        mutated_values = self.rng.integers(self.n_gene_variants, size=(self.n_mutations,))
        new_generation[mutated_indices, mutated_gene_indices] = mutated_values
        
    def check_convergence(self):
        return self._iterations < self.max_iterations

    def training_loop(self):
        parents = self.selection(self._population, self._fitness)
        new_generation = self.crossover(parents)
        self.mutation(new_generation)
        fitness = self.compute_fitness(new_generation)
        self._population = new_generation
        self._iterations += 1
        return fitness

    def initialize(self):
        self._population = self.generate_population()
        self._fitness = self.compute_fitness(self._population)
        self._iterations = 0

    def train(self):
        while self.check_convergence():
            self._fitness = self.training_loop()


def random_fitness_function(population):
    return np.array(sum(np.array(population) < i) for i in population)

if __name__ == "__main__":
    g = GeneticAlgorithm(
            population_size=50, 
            n_gene_variants=5, 
            n_parents=10,
            n_mutations=10,
            chromosone_length=6, 
            fitness_fn=random_fitness_function)
    print(random_fitness_function(np.array([1,2,3,4])))
    g.train()
