#!/usr/bin/env python3

import sys
import numpy as np

np.set_printoptions(suppress=True)

def read(filename, num_of_match_outcome_factors, num_of_match_statistics):
    labels = []
    match_outcome = []
    match_stats = []

    with open("./data/{}" .format(filename), "r") as f:
        # read labels
        labels = np.array(f.readline()[:-1].split(","))
        # read data from each line
        for line in f.readlines()[:][:-1]:
            data = []
            for val in line.split(","):
                data.append(float(val))
            # separate match outcome factors and match statistics
            match_outcome.append(data[:num_of_match_outcome_factors])
            match_stats.append(data[-num_of_match_statistics:])

    match_outcome = np.array(match_outcome)
    match_stats = np.array(match_stats)

    print("\nLabels:\n{}\n" .format(labels))
    print("Match Outcome ({}x{}):\n{}\n" .format(match_outcome.shape[0], match_outcome.shape[1], match_outcome))
    print("Match Statistics ({}x{}):\n{}\n" .format(match_stats.shape[0], match_stats.shape[1], match_stats))

    return labels, match_outcome, match_stats
