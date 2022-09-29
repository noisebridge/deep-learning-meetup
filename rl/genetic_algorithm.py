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
        return self.rng.integers(self.n_gene_variants-1, size=(self.population_size, self.chromosone_length))

    @_fitness.default
    def compute_initial_fitness(self):
        return self.compute_fitness(self._population)

    def compute_fitness(self, population):
        return self.fitness_fn(population)


    def selection(self, population, weights):
        return self.rng.choice(population, p=weights/weights.sum(), size=self.n_parents, axis=0)
    
    # Breed between two pairs of parents
    def crossover(self, parents):
        new_generation = np.zeros([self.population_size, parents.shape[1]])
        left_indices = self.rng.integers(self.n_parents, size=(self.population_size,))
        right_indices = self.rng.integers(self.n_parents, size=(self.population_size,))
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

    def training_loop(self, add_mutations=True):
        parents = self.selection(self._population, self._fitness)
        #print("Parents")
        #print(parents)
        new_generation = self.crossover(parents)
        #print("Spliced generation")
        #print(new_generation)
        nonmutated = np.copy(new_generation)
        if add_mutations:
            self.mutation(new_generation)
        #print("mutations")
        #print(new_generation - nonmutated)
        fitness = self.compute_fitness(new_generation)
        self._population = new_generation
        self._iterations += 1
        return fitness

    def initialize(self):
        self._population = self.generate_population()
        self._fitness = self.compute_fitness(self._population)
        self._iterations = 0


    def represent(self, population):
        print("Population:")
        population_ints = population.astype(int)
        print(population_ints)
        for chromosone in population_ints:
            word = str([chr(i) for i in chromosone])
            print(word)

    def train(self):
        while self.check_convergence():
            self._fitness = self.training_loop()
            self.represent(self._population)
        self.training_loop(add_mutations=False)
        print("Fitness")
        print(self._fitness)
        print("Final population")
        self.represent(self._population)
        return self._fitness


def random_fitness_function(population):
    return np.square(np.sum(population, axis=1, keepdims=False))


# Make sure the word is as close to "Be Excellent" as possible
expected_string = "Be Excellent"
expected_string_as_int_array = np.array([ord(c) for c in expected_string])
eps=1e-3
def excellent_fitness_function(population):
    return 1/((np.sum(np.square(population - expected_string_as_int_array), axis=1))+eps)


if __name__ == "__main__":
    g = GeneticAlgorithm(
            population_size=20, 
            max_iterations=10000,
            n_gene_variants=256, 
            n_parents=15,
            n_mutations=15,
            chromosone_length=12, 
            fitness_fn=excellent_fitness_function)

    g.train()
    print("Expected:")
    print(expected_string_as_int_array)
