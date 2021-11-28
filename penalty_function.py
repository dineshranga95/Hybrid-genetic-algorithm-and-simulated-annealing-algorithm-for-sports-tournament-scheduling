import numpy as np
from numba import njit


# calculate penalty points based on hard and soft constraints
@njit(cache=True)  # decorate function to be compiled instead of interpreted, speed up code execution by 2-3 times
def penalty(slot_match, match_match, match_referee, referee_preference):
    penalty_point = 0
    hc_count = 0
    sc_count = 0
    match_no = slot_match.shape[1]
    referee_no = referee_preference.shape[0]
    venue_no = 4
    time_slot_no = 5
    day_slot_no = venue_no * time_slot_no
    day_no = 5
    match_match = np.copy(match_match)
    referee_preference[:, 3:6].fill(0)

    # HC02: no staff can attend more than 1 presentations concurrently
    for match in range(match_no):
        slot = np.where(slot_match[:, match] == 1)[0][0]
        min_concurrent_slot = (slot % time_slot_no) + (slot // day_slot_no) * day_slot_no
        max_concurrent_slot = min_concurrent_slot + day_slot_no

        for concurrent_slot in range(min_concurrent_slot, max_concurrent_slot, time_slot_no):
            concurrent_match = np.where(slot_match[concurrent_slot] == 1)[0]

            if len(concurrent_match) != 0:
                concurrent_match = concurrent_match[0]

                if match_match[match][concurrent_match] == 1:
                    match_match[match][concurrent_match] = -1
                    match_match[concurrent_match][match] = -1
                    penalty_point += 1000
                    hc_count += 1

    # 5(day) × 5(time slot) matrix storing venue for each presentation
    day_time_slot = np.zeros((day_no, time_slot_no + 1), dtype=np.int8)  # extra last column to handle last time slot

    for referee in range(referee_no):  # most time-consuming loop
        referied_matches = np.where(match_referee[:, referee] == 1)[0]
        day_time_slot.fill(0)

        for referied_match in referied_matches:
            referied_slot = np.where(slot_match[:, referied_match] == 1)[0][0]
            referied_day = referied_slot // day_slot_no
            referied_time_slot = referied_slot % time_slot_no
            referied_venue = (referied_slot // time_slot_no) % venue_no + 1  # add 1 to avoid conflict with 0
            day_time_slot[referied_day][referied_time_slot] = referied_venue

        consecutive_preference = referee_preference[referee][0]  # SC01: consecutive presentations
        day_count = 0
        venue_changes = 0

        for day in range(day_no):
            is_consecutive = False
            is_this_day = False
            consecutive_count = 0
            previous_venue = 0

            for time_slot in range(time_slot_no + 1):
                if day_time_slot[day][time_slot] != 0:
                    venue = day_time_slot[day][time_slot]

                    # if current presentation is consecutive with previous presentation
                    # this check is ignored if it is the first presentation of group of consecutive presentations
                    if is_consecutive and venue != previous_venue:
                        venue_changes += 1

                    is_consecutive = True
                    is_this_day = True
                    consecutive_count += 1
                    previous_venue = venue
                else:
                    # calculate penalty points for a group of consecutive presentations (might has only 1 presentation)
                    if is_consecutive:
                        if consecutive_count < consecutive_preference:  # encourage presentations to be consecutive
                            penalty_point += (consecutive_preference - consecutive_count) * 1
                            referee_preference[referee][3] += 1
                        elif consecutive_count > consecutive_preference:  # exceeds maximum consecutive preference
                            penalty_point += (consecutive_count - consecutive_preference) * 10
                            referee_preference[referee][3] += 1
                            sc_count += 1

                    is_consecutive = False

            if is_this_day:  # a presentation takes place on this day
                day_count += 1

        days_preference = referee_preference[referee][1]  # SC02: number of days
        referee_preference[referee][4] = day_count

        if day_count > days_preference:
            penalty_point += (day_count - days_preference) * 10
            sc_count += 1

        venue_preference = referee_preference[referee][2]  # SC03: change of venue
        referee_preference[referee][5] = venue_changes

        if venue_preference == 1 and venue_changes > 0:  # supervisor does not want to change venue
            penalty_point += venue_changes * 10
            sc_count += 1

    return penalty_point, hc_count, sc_count
