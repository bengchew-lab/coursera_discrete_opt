#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import pandas as pd
import numpy as np
Item = namedtuple("Item", ['index', 'value', 'weight'])

# def solve_it(input_data):
#     # Modify this code to run your optimization algorithm

#     # parse the input
#     lines = input_data.split('\n')

#     firstLine = lines[0].split()
#     item_count = int(firstLine[0])
#     capacity = int(firstLine[1])

#     items = []

#     for i in range(1, item_count+1):
#         line = lines[i]
#         parts = line.split()
#         items.append(Item(i-1, int(parts[0]), int(parts[1])))

#     # a trivial algorithm for filling the knapsack
#     # it takes items in-order until the knapsack is full
#     value = 0
#     weight = 0
#     taken = [0]*len(items)

#     for item in items:
#         if weight + item.weight <= capacity:
#             taken[item.index] = 1
#             value += item.value
#             weight += item.weight
    
#     # prepare the solution in the specified output format
#     output_data = str(value) + ' ' + str(0) + '\n'
#     output_data += ' '.join(map(str, taken))
#     return output_data

# Algorithm 1: take every item until it becomes full 
def take_until_full(available_item_df, max_capa):
    value = 0
    weight = 0
    taken = [0]*len(available_item_df)
    

    for index, row in available_item_df.iterrows():
        if weight + available_item_df['weight'][index] <= max_capa:
            taken[index] = 1
            value += available_item_df['values'][index]
            weight += available_item_df['weight'][index]

    return value, taken

# Algorithm 2: sort by highest value first, take until full
def sort_highest_value_tuf(available_item_df, max_capa):
    value = 0
    weight = 0
    taken = [0]*len(available_item_df)
    

    available_item_df.sort_values("values", ascending=False, inplace=True)

    for index, row in available_item_df.iterrows():
        if weight + available_item_df['weight'][index] <= max_capa:
            taken[index] = 1
            value += available_item_df['values'][index]
            weight += available_item_df['weight'][index]

    return value, taken

# Algorithm 3: sort by highest value-weight ratio first, take until full
def sort_highest_vwratio_tuf(available_item_df, max_capa):
    value = 0
    weight = 0
    taken = [0]*len(available_item_df)
    
    available_item_df["value_weight_ratio"] = available_item_df['values']/available_item_df['weight']
    available_item_df.sort_values("weight", ascending=True, inplace=True)
    available_item_df.sort_values("value_weight_ratio", ascending=False, inplace=True)

    for index, row in available_item_df.iterrows():
        if weight + available_item_df['weight'][index] <= max_capa:
            taken[index] = 1
            value += available_item_df['values'][index]
            weight += available_item_df['weight'][index]

    return value, taken

def solve_it(input_data, mode='3'):
    # Modify this code to run your optimization algorithm

    # Step 1: Preprocessing & convert to df format
    lines = input_data.split('\n')
    lines

    # get number of item available & max_capa
    item_count = int(lines[0].split()[0])
    max_capa = int(lines[0].split()[1])

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        if i == 1: # initialise new variables
            values = [int(parts[0])]
            weight = [int(parts[1])]
        else:
            values.append(int(parts[0]))
            weight.append(int(parts[1]))

    available_item_df = pd.DataFrame({"values":values, "weight":weight})

    # optimization solver
    if mode == "1":
        print(f"Use algorithm {mode}")
        value, taken = take_until_full(available_item_df, max_capa)
    if mode == "2":
        print(f"Use algorithm {mode}")
        value, taken = sort_highest_value_tuf(available_item_df, max_capa)
    if mode == "3":
        print(f"Use algorithm {mode}")
        value, taken = sort_highest_vwratio_tuf(available_item_df, max_capa)
    
    # prepare the solution in the specified output format
    output_data = str(value) + ' ' + str(0) + '\n'
    output_data += ' '.join(map(str, taken))
    return output_data


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        file_location = sys.argv[1].strip()
        print(file_location)
        # mode = str(sys.argv[2])
        with open(file_location, 'r') as input_data_file:
            input_data = input_data_file.read()
        print(solve_it(input_data, mode='2'))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

