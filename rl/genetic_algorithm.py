from attrs import define, field
from typing import Callable
import numpy as np
from functools import partial

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
    min_iterations : int = field(default=10)
    max_iterations : int = field(default=100)
    n_convergence_iterations : int = field(default=100)
    fitness_fn : Callable[[list[int]], np.ndarray] = field(factory=lambda x: np.array(x))
    crossover_point : int = field()
    rng : np.random.Generator = field(factory=lambda seed=None: np.random.default_rng(seed))
    _population : np.ndarray = field()
    _fitness : np.ndarray = field(init=False)
    _iterations : int = field(init=False, default=0)
    _fitness_history : np.ndarray = field(init=False)

    @crossover_point.default
    def _check_valid_crossover_point(self, crossover_point=None):
        if crossover_point is None:
            crossover_point = self.chromosone_length // 2
        elif crossover_point >= self.chromosone_length:
            raise ValueException("Crossover point cannot be greater than chromosone length!")
        return crossover_point


    @_population.default
    def _generate_population(self):
        return self.rng.integers(self.n_gene_variants-1, size=(self.population_size, self.chromosone_length))

    @_fitness.default
    def compute_initial_fitness(self):
        return self.compute_fitness(self._population)

    @_fitness_history.default
    def _create_fitness_history(self):
        return np.zeros(self.n_convergence_iterations)

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
        CONVERGENCE_THRESHOLD = 1e-6
        normalized_history = self._fitness_history / np.sum(self._fitness_history)
        fitness_diff = np.sqrt(np.max(np.square(normalized_history[1:] - normalized_history[:-1])))
        print(self._fitness_history)
        print(fitness_diff)
        return (
                self._iterations > self.min_iterations and fitness_diff < CONVERGENCE_THRESHOLD 
        )    or self._iterations >= self.max_iterations


    def _update_fitness_history(self, fitness):
        average_relative_fitness = np.average(fitness)
        if self._iterations < self.n_convergence_iterations:
            self._fitness_history[self._iterations] = average_relative_fitness
        else:
            self._fitness_history[:-1] = np.copy(self._fitness_history[1:])
            self._fitness_history[-1] = average_relative_fitness


    def training_loop(self, add_mutations=True):
        parents = self.selection(self._population, self._fitness)
        #print("Parents")
        #print(parents)
        new_generation = self.crossover(parents)
        #print("Spliced generation")
        #print(new_generation)
        convergence_fitness = self.compute_fitness(new_generation)
        if add_mutations:
            self.mutation(new_generation)
        #print("mutations")
        #print(new_generation - nonmutated)
        self._fitness = self.compute_fitness(new_generation)
        self._population = new_generation
        return convergence_fitness

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
        while not self.check_convergence():
            convergence_fitness = self.training_loop()
            self.represent(self._population)
            self._update_fitness_history(convergence_fitness)
            self._iterations += 1
        self.training_loop(add_mutations=False)
        print("Fitness")
        print(self._fitness)
        print(f"Converged in {self._iterations} iterations")
        self.represent(self._population)
        return self._fitness


def random_fitness_function(population):
    return np.square(np.sum(population, axis=1, keepdims=False))


# Make sure the word is as close to "Be Excellent" as possible
@define
class ExcellentFitness:
    expected_string : str = field(default="Be Excellent")
    expected_string_as_int_array : np.ndarray = field()
    eps : float = field(default=1e-3)

    @expected_string_as_int_array.default
    def init_int_array(self):
        return np.array([ord(c) for c in self.expected_string])

    def max_value(self):
        return 1 / self.eps

    # Can realistically get there
    def max_threshold(self):
        return self.max_value()

    def excellent_fitness_function(self, population):
        return 1/((np.sum(np.square(population - self.expected_string_as_int_array), axis=1))+self.eps)


if __name__ == "__main__":
    fitness = ExcellentFitness()
    fn = fitness.excellent_fitness_function
    g = GeneticAlgorithm(
            population_size=20, 
            max_iterations=10000,
            min_iterations=1000,
            n_gene_variants=256, 
            n_parents=15,
            n_mutations=15,
            chromosone_length=12, 
            fitness_fn=fn)

    g.train()
    print("Expected:")
    print(fitness.expected_string_as_int_array)
