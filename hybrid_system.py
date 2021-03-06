import data as dt
from penalty_function import penalty
import genetic_algorithm as ga
import simulated_annealing as sa
import numpy as np
from timeit import default_timer as timer


# hybrid system using genetic algorithm and simulated annealing
def hybrid_system():
    # load data and functions
    slot_match, match_match, match_referee, referee_preference = dt.load()

    # initialize matrices
    slot_no = slot_match.shape[0]
    match_no = slot_match.shape[1]
    population_size = 10
    population = np.empty([population_size, slot_no, match_no], dtype=np.int8)
    penalty_points = np.empty(population_size, dtype=int)

    # create initial population
    for i in range(population_size):
        chromosome = ga.generate_chromosome(slot_match)
        population[i] = chromosome
        penalty_point = \
            penalty(chromosome, match_match, match_referee, referee_preference)[0]
        penalty_points[i] = penalty_point

    # sort initial population based on penalty points
    population = population[penalty_points.argsort()]
    penalty_points = penalty_points[penalty_points.argsort()]

    # run genetic algorithm for 100 generations
    ga_max_generations = 100
    population, penalty_points, ga_plot_data = \
        ga.reproduction(ga_max_generations, population, penalty_points, match_match,
                        match_referee, referee_preference)

    # run simulated annealing after running genetic algorithm
    temperature = penalty_points[population_size - 1] - penalty_points[0]
    candidate = population[0]
    penalty_point = penalty_points[0]
    best_candidate, best_penalty_point, sa_plot_data = \
        sa.anneal(temperature, candidate, penalty_point, match_match,
                  match_referee, referee_preference)

    # write result data
    constraint_counts = \
        penalty(best_candidate, match_match, match_referee, referee_preference)
    plot_data = np.concatenate([ga_plot_data, sa_plot_data])
    dt.write(best_candidate,referee_preference, constraint_counts, plot_data)


start = timer()
hybrid_system()
print("\nExecution Time of Hybrid System:", round(timer() - start, 2), "seconds")