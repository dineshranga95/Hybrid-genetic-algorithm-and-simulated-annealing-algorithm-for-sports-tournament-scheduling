from penalty_function import penalty
import numpy as np


# generate initial population where all hard constraints have been solved except HC02
def generate_chromosome(slot_match):
    chromosome = np.copy(slot_match)
    slot_no = chromosome.shape[0]
    match_no = chromosome.shape[1]

    for match in range(match_no):
        while True:
            random_slot = np.random.randint(slot_no)
            # if the slot is available and empty
            if chromosome[random_slot][match] == 0 and np.count_nonzero(chromosome[random_slot] == 1) == 0:
                chromosome[random_slot][match] = 1
                break

    return chromosome


# select 2 chromosomes based on tournament selection
def selection(population, penalty_points):
    tournament_size = 2

    # select 1st chromosome based on 1st tournament selection
    t1, t2 = np.random.choice(range(population.shape[0]), tournament_size)
    first = t1 if penalty_points[t1] <= penalty_points[t2] else t2

    # ensure 2 chromosomes selected are not identical
    while True:
        # select 2nd chromosome based on 2nd tournament selection
        t1, t2 = np.random.choice(range(population.shape[0]), tournament_size)
        second = t1 if penalty_points[t1] <= penalty_points[t2] else t2

        if second != first:
            break

    return population[first], population[second]


# perform 2-point crossover
def crossover(first_parent, second_parent):
    first_child = np.copy(first_parent)
    second_child = np.copy(second_parent)
    match_no = first_parent.shape[1]
    cutpoint1, cutpoint2 = np.random.choice(range(match_no), 2)

    if cutpoint1 > cutpoint2:
        cutpoint1, cutpoint2 = cutpoint2, cutpoint1

    # swap matches from cutpoint1 to cutpoint2 between 2 parents
    first_child[:, cutpoint1:cutpoint2], second_child[:, cutpoint1:cutpoint2] = \
        second_child[:, cutpoint1:cutpoint2], np.copy(first_child[:, cutpoint1:cutpoint2])
    first_child = repair(first_child, cutpoint1, cutpoint2)
    second_child = repair(second_child, cutpoint1, cutpoint2)
    return first_child, second_child


# repair chromosome after crossover
def repair(chromosome, cutpoint1, cutpoint2):
    slot_no = chromosome.shape[0]

    for match in range(cutpoint1, cutpoint2):
        slot = np.where(chromosome[:, match] == 1)[0][0]

        # more than 1 match scheduled for a slot
        if np.count_nonzero(chromosome[slot] == 1) > 1:
            chromosome[slot][match] = 0

            # schedule match for another random slot
            while True:
                random_slot = np.random.randint(slot_no)

                if chromosome[random_slot][match] == 0 and np.count_nonzero(chromosome[random_slot] == 1) == 0:
                    chromosome[random_slot][match] = 1
                    break

    return chromosome


# swap mutation of chromosome after crossover
def mutation(chromosome):
    match_no = chromosome.shape[1]
    random_match1 = np.random.randint(match_no)
    slot1 = np.where(chromosome[:, random_match1] == 1)[0][0]

    while True:
        random_match2 = np.random.randint(match_no)
        slot2 = np.where(chromosome[:, random_match2] == 1)[0][0]

        # 2 matches can be scheduled on slots to be exchanged, hence swap 2 matches
        if chromosome[slot1][random_match2] == 0 and chromosome[slot2][random_match1] == 0:
            chromosome[slot1][random_match1] = chromosome[slot2][random_match2] = 0
            chromosome[slot1][random_match2] = chromosome[slot2][random_match1] = 1
            break

    return chromosome


# Steady-State Genetic Algorithm - replace 2 chromosomes in population
def replacement(population, penalty_points, first_child, second_child, first_penalty_point, second_penalty_point):
    # replace 2 chromosomes of highest penalty points with 2 new chromosomes
    population_size = len(population)
    population[population_size - 1], population[population_size - 2] = first_child, second_child
    penalty_points[population_size - 1], penalty_points[population_size - 2] = first_penalty_point, second_penalty_point

    # sort population based on penalty points
    population = population[penalty_points.argsort()]
    penalty_points = penalty_points[penalty_points.argsort()]

    return population, penalty_points


# reproduce new chromosomes in new generation
def reproduction(max_generations, population, penalty_points, match_match,
                 match_referee, referee_preference):
    plot_data = []

    for generation in range(max_generations):
        first_parent, second_parent = selection(population, penalty_points)
        first_child, second_child = crossover(first_parent, second_parent)
        first_child = mutation(first_child)
        second_child = mutation(second_child)
        first_penalty_point = \
            penalty(first_child, match_match, match_referee, referee_preference)[0]
        second_penalty_point = \
            penalty(second_child, match_match, match_referee, referee_preference)[0]
        population, penalty_points = \
            replacement(population, penalty_points, first_child, second_child,
                        first_penalty_point, second_penalty_point)
        plot_data.append(penalty_points[0])

        if (generation + 1) % 5 == 0:
            print("[Iteration ", generation + 1, "] Penalty Point: ", penalty_points[0], sep="")

    return population, penalty_points, plot_data
