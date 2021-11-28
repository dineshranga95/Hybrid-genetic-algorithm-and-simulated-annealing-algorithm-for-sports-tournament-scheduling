from penalty_function import penalty
import numpy as np
from numba import njit


# interchange two slots of a professor
@njit(cache=True)
def neighbourhood_structure1(candidate, match_referee):
    referee_no = match_referee.shape[1]

    while True:
        random_referee = np.random.randint(referee_no)
        referied_matches = np.where(match_referee[:, random_referee] == 1)[0]

        if len(referied_matches) > 1:
            # get 2 random presentations supervised by the supervisor
            match1 = referied_matches[np.random.randint(len(referied_matches))]
            slot1 = np.where(candidate[:, match1] == 1)[0][0]
            match2 = referied_matches[np.random.randint(len(referied_matches))]
            slot2 = np.where(candidate[:, match2] == 1)[0][0]

            # interchange the slots of the two presentations
            if candidate[slot2][match1] == 0 and candidate[slot1][match2] == 0:
                candidate[slot1][match1] = candidate[slot2][match2] = 0
                candidate[slot2][match1] = candidate[slot1][match2] = 1
                break


# change venue of presentation (time-slot remains the same)
@njit(cache=True)
def neighbourhood_structure2(candidate):
    match_no = candidate.shape[1]
    venue_no = 4
    time_slot_no = 5
    day_slot_no = venue_no * time_slot_no

    while True:
        random_match = np.random.randint(match_no)
        slot = np.where(candidate[:, random_match] == 1)[0][0]
        min_concurrent_slot = (slot % time_slot_no) + (slot // day_slot_no) * day_slot_no
        max_concurrent_slot = min_concurrent_slot + day_slot_no

        # find a concurrent slot that is available and empty
        for concurrent_slot in range(min_concurrent_slot, max_concurrent_slot, time_slot_no):
            if candidate[concurrent_slot][random_match] == 0 and \
                    np.count_nonzero(candidate[concurrent_slot] == 1) == 0:
                candidate[slot][random_match] = 0
                candidate[concurrent_slot][random_match] = 1
                return


# assign presentation to a random empty slot
@njit(cache=True)
def neighbourhood_structure3(candidate):
    slot_no = candidate.shape[0]
    match_no = candidate.shape[1]
    random_match = np.random.randint(match_no)
    original_slot = np.where(candidate[:, random_match] == 1)[0][0]

    while True:
        random_slot = np.random.randint(slot_no)

        # if the slot is available and empty (other presentations are not using the slot)
        if candidate[random_slot][random_match] == 0 and np.count_nonzero(candidate[random_slot] == 1) == 0:
            candidate[original_slot][random_match] = 0
            candidate[random_slot][random_match] = 1
            break


# find a random presentation and assign a presentation that has the same supervisor
# to the slot next to the random presentation
# aims to increase the number of consecutive presentations
@njit(cache=True)
def neighbourhood_structure4(candidate, match_match):
    match_no = candidate.shape[1]
    venue_no = 4
    time_slot_no = 5
    day_slot_no = venue_no * time_slot_no

    while True:
        current_match = np.random.randint(match_no)
        current_slot = np.where(candidate[:, current_match] == 1)[0][0]
        adjacent_slot = (current_slot - 1) % 100 if np.random.random() < 0.5 else (current_slot + 1) % 100

        # find all time slots after the current slot
        # the slot with the same venue as the current slot is given priority
        min_concurrent_slot = (adjacent_slot % time_slot_no) + (adjacent_slot // day_slot_no) * day_slot_no
        max_concurrent_slot = min_concurrent_slot + day_slot_no
        adjacent_concurrent_slots = [adjacent_slot]

        for concurrent_slot in range(min_concurrent_slot, max_concurrent_slot, time_slot_no):
            if concurrent_slot != adjacent_slot:
                adjacent_concurrent_slots.append(concurrent_slot)

        for adjacent_concurrent_slot in adjacent_concurrent_slots:
            # check if the slot is available (if the slot is unavailable, the row is all -1)
            if np.all(candidate[adjacent_concurrent_slot]):
                continue

            # check if the slot is empty
            if np.count_nonzero(candidate[adjacent_concurrent_slot] == 1) == 0:
                overlapping_matches = np.where(match_match[current_match])[0]

                if len(overlapping_matches) > 0:
                    chosen_match = overlapping_matches[np.random.randint(len(overlapping_matches))]

                    if candidate[adjacent_concurrent_slot][chosen_match] == 0:
                        original_slot = np.where(candidate[:, chosen_match] == 1)[0][0]
                        candidate[original_slot][chosen_match] = 0
                        candidate[adjacent_concurrent_slot][chosen_match] = 1
                        return


# Simulated-Annealing Algorithm - perform random walk in space until temperature drops below a limit
def anneal(initial_temperature, initial_candidate, penalty_point, match_match,
           match_referee, referee_preference):
    temperature = initial_temperature
    final_temperature = 0.0001 * initial_temperature
    alpha = 0.9999  # annealing schedule to decrease temperature
    current_candidate = initial_candidate
    best_candidate = initial_candidate
    current_penalty_point = penalty_point
    best_penalty_point = penalty_point
    iteration = 100  # initial iteration after performing genetic algorithm
    neighbourhood_structure_no = 4
    plot_data = []

    while temperature >= final_temperature:
        new_candidate = np.copy(current_candidate)
        neighbourhood_structure = np.random.randint(neighbourhood_structure_no)

        if neighbourhood_structure == 0:
            neighbourhood_structure1(new_candidate, match_referee)
        elif neighbourhood_structure == 1:
            neighbourhood_structure2(new_candidate)
        elif neighbourhood_structure == 2:
            neighbourhood_structure3(new_candidate)
        else:
            neighbourhood_structure4(new_candidate, match_match)

        new_penalty_point = \
            penalty(new_candidate, match_match, match_referee, referee_preference)[0]
        difference = new_penalty_point - current_penalty_point

        if difference < 0 or np.random.random() < np.exp((-1 * difference) / temperature):
            current_candidate = new_candidate
            current_penalty_point = new_penalty_point

        if current_penalty_point < best_penalty_point:
            best_candidate = current_candidate
            best_penalty_point = current_penalty_point

        temperature *= alpha
        iteration += 1
        plot_data.append(best_penalty_point)

        if iteration % 50 == 0:
            print("[Iteration ", iteration, "] Penalty Point: ", best_penalty_point, sep="")

    return best_candidate, best_penalty_point, plot_data
