import csv
import numpy as np
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from datetime import datetime as date


#  load data from csv files
def load():
    slot_no = 100
    referee_no = 15
    match_no = 30
    preference_no = 3
    match_referee = np.zeros([match_no, referee_no], dtype=np.int8)
    referee_slot = np.zeros([referee_no, slot_no], dtype=np.int8)
    referee_preference = np.zeros([referee_no, 2 * preference_no], dtype=np.int8)

    # read supExaAssign.csv
    with open('input_files\SupExaAssign.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')
        next(csv_reader)

        for row in csv_reader:
            i = int(row[0][1:]) - 1  # only underscores in M___ will be considered

            for col in range(1, 4):
                j = int(row[col][2:]) - 1  # only underscores in R0__ will be considered
                match_referee[i][j] = 1

    match_match = np.dot(match_referee, match_referee.transpose())
    # presentations supervised by same examiners are marked with 1
    match_match[match_match >= 1] = 1
    np.fill_diagonal(match_match, 0)  # mark diagonal with 0 so penalty points can be calculated correctly

    # read HC04.csv (staff unavailability)
    with open('input_files\HC04.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            i = int(row[0][2:]) - 1  # only underscores in R0__ will be considered
            j = [int(_) - 1 for _ in row[1:]]
            referee_slot[i][j] = 1

    slot_match = np.dot(referee_slot.transpose(), match_referee.transpose())
    slot_match[slot_match >= 1] = -1  # unavailable slots for presentation are marked with -1

    # read HC03.csv (venue unavailability)
    with open('input_files\HC03.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            i = [int(_) - 1 for _ in row[1:]]
            slot_match[i, :] = -1  # unavailable slots for presentation are marked with -1

    # read SC01.csv (consecutive presentations)
    with open('input_files\SC01.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            i = int(row[0][2:]) - 1  # only underscores in R0__ will be considered
            referee_preference[i][0] = int(row[1])

    # read SC02.csv (number of days)
    with open('input_files\SC02.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            i = int(row[0][2:]) - 1  # only underscores in R0__ will be considered
            referee_preference[i][1] = int(row[1])

    # read SC03.csv (change of venue)
    with open('input_files\SC03.csv') as file:
        csv_reader = csv.reader(file, delimiter=',')

        for row in csv_reader:
            i = int(row[0][2:]) - 1  # only underscores in R0__ will be considered
            referee_preference[i][2] = 1 if row[1] == "yes" else 0

    return slot_match, match_match, match_referee, referee_preference


# write result to csv file with timestamp
def write(slot_match, referee_preference, constraints_count, plot_data):
    timestamp = date.now().strftime("[%Y-%m-%d %H-%M-%S]")

    # plot graph
    title = (f"Improvement of match Scheduling over Iterations\n"
             f"[Hard Constraints Violated:] {constraints_count[1]} "
             f"[Soft Constraints Violated:] {constraints_count[2]}\n"
             f"[Final Penalty Points:] {constraints_count[0]}")
    plt.title(title)
    plt.xlabel("Number of Iterations")
    plt.ylabel("Penalty Points")
    plt.axis([0, len(plot_data), 0, max(plot_data)])
    plt.plot(plot_data, "r--")
    plt.grid(True)
    plt.ioff()
    plt.show()
    graph_name = f"graph {timestamp}"
    plt.savefig(graph_name)

    # draw schedule
    venue_no = 4
    time_slot_no = 5
    day_slot_no = venue_no * time_slot_no
    day_no = 5
    slot_no = day_slot_no * day_no
    venues = ["G1", "G2", "G3", "G4"]
    days = ["Mon", "Tues", "Wed", "Thu", "Fri"]

    schedule = PrettyTable()
    schedule.field_names = ["Day", "Venue",
                            "0800-0930", "1000-1130", "1200-0130",
                            "0200-0330", "0400-0530"]

    venue = 0
    day = 0

    for first_slot in range(0, slot_no, time_slot_no):
        row = []

        if venue == 0:
            row.append(days[day])
        else:
            row.append("")

        row.append(venues[venue])

        for slot in range(first_slot, first_slot + time_slot_no):
            match = np.where(slot_match[slot] == 1)[0]

            if len(match) == 0:
                row.append("")
            else:
                match = match[0] + 1
                row.append("M" + str(match))

        schedule.add_row(row)
        venue += 1

        if venue == venue_no:
            venue = 0
            day += 1
            schedule.add_row([""] * (2 + time_slot_no))

    print("\n", schedule, "\n")

    # print supervisor-related data
    referee_no = referee_preference.shape[0]

    for referee in range(referee_no):
        venue_preference = "No" if referee_preference[referee][2] else "Yes"

        print(f"[Referee R{str(referee + 1).zfill(3)}] "
              f"[No. of Continuous matches: {referee_preference[referee][3]}] "
              f"[Day Preference: {referee_preference[referee][1]}] "
              f"[Days: {referee_preference[referee][4]}] "
              f"[Venue Change Preference: {venue_preference}] "
              f"[Venue Changes: {referee_preference[referee][5]}]")

    # write result data to csv file with timestamp
    filename = f"result {timestamp}.csv"

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)

        for slot in range(slot_match.shape[0]):
            match = np.where(slot_match[slot] == 1)[0]

            if len(match) == 0:  # empty if no presentation is found for the slot
                writer.writerow(["null", ""])
            else:
                match = match[0] + 1  # Access x in array([x])
                writer.writerow(["M" + str(match), ""])
