#!/usr/bin/python
# -*- coding: utf-8 -*-

from collections import namedtuple
import pandas as pd
import numpy as np
from queue import Queue
from collections import deque
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

# Algorithm 4: dynamic programming - bottom-up approach (filling up the table)
def dp_bottom_up(available_item_df, max_capa):
    value_list = available_item_df['values'].to_list()
    weight_list = available_item_df['weight'].to_list()
    number_of_item = len(weight_list)

    value = 0
    weight = 0
    taken = [0]*len(available_item_df)


    dp_tbl = pd.DataFrame(np.zeros((number_of_item+1, max_capa+1)))
    dp_picked = pd.DataFrame(np.zeros((number_of_item+1, max_capa+1)))

    for i in range(1, number_of_item+1):
        for w in range(0, max_capa+1):

            if weight_list[i-1] <= w:
                dp_tbl.iloc[i, w] = max(dp_tbl.iloc[i-1, w], value_list[i-1] + dp_tbl.iloc[i-1, (w - weight_list[i-1])] )
                dp_picked.iloc[i, w] = 1
            else:
                dp_tbl.iloc[i, w] = dp_tbl.iloc[i-1, w] 
                dp_picked.iloc[i, w] = 0


    w = max_capa
    taken_index =list()
    for i in range(number_of_item,0,-1):
        if dp_picked.iloc[i, w] == True:
            taken_index.append(i-1)
            w = w - weight_list[i-1]

    for n in taken_index:
        value += available_item_df['values'][n]
        weight += available_item_df['weight'][n]
        taken[n] = 1

    return value, taken

# Algorithm 5: branch and bound - https://www.geeksforgeeks.org/implementation-of-0-1-knapsack-using-branch-and-bound/
class Node:
    def __init__(self):
        self.level = None
        self.profit = None
        self.bound = None
        self.weight = None
        self.contains = []

    def __str__(self):
        return "Level: %s Profit: %s Bound: %s Weight: %s" % (self.level, self.profit, self.bound, self.weight)

def bound(node, n, W, items):
    if(node.weight >= W):
        return 0

    profit_bound = int(node.profit)
    j = node.level + 1
    totweight = int(node.weight)

    while ((j < n) and (totweight + items[j].weight) <= W):
        totweight += items[j].weight
        profit_bound += items[j].value
        j += 1

    if(j < n):
        profit_bound += (W - totweight) * items[j].value / float(items[j].weight)

    return profit_bound

def KnapSackBranchNBound(weight, items, total_items):
    Q = deque([])
    items = sorted(items, key=lambda x: x.value/float(x.weight), reverse=True)

    u = Node()

    u.level = -1
    u.profit = 0
    u.weight = 0

    Q.append(u)
    maxProfit = 0
    bestItems = []

    while (len(Q) != 0):

        u = Q[0]
        Q.popleft()
        v = Node()

        if u.level == -1:
            v.level = 0

        if u.level == total_items - 1:
            continue

        v.level = u.level + 1
        v.weight = u.weight + items[v.level].weight
        v.profit = u.profit + items[v.level].value
        v.contains = list(u.contains)
        v.contains.append(items[v.level].index)

        if (v.weight <= weight and v.profit > maxProfit):
            maxProfit = v.profit
            bestItems = v.contains

        v.bound = bound(v, total_items, weight, items)
        if (v.bound > maxProfit):
            # print v
            Q.append(v)

        v = Node()
        v.level = u.level + 1
        v.weight = u.weight
        v.profit = u.profit
        v.contains = list(u.contains)

        v.bound = bound(v, total_items, weight, items)
        if (v.bound > maxProfit):
            # print v
            Q.append(v)

    taken = [0] * len(items)
    for i in range(len(bestItems)):
        taken[bestItems[i]] = 1

    return maxProfit, taken
      

def solve_it(input_data, mode='5'):
    # Modify this code to run your optimization algorithm

    # Step 1: Preprocessing & convert to df format
    lines = input_data.split('\n')

    # get number of item available & max_capa
    item_count = int(lines[0].split()[0])
    max_capa = int(lines[0].split()[1])

    items = []

    for i in range(1, item_count+1):
        line = lines[i]
        parts = line.split()
        items.append(Item(i-1, int(parts[0]), int(parts[1])))
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
        optimal_value, taken = take_until_full(available_item_df, max_capa)
    if mode == "2":
        print(f"Use algorithm {mode}")
        optimal_value, taken = sort_highest_value_tuf(available_item_df, max_capa)
    if mode == "3":
        print(f"Use algorithm {mode}")
        optimal_value, taken = sort_highest_vwratio_tuf(available_item_df, max_capa)
    if mode == "4":
        print(f"Use algorithm {mode}")
        optimal_value, taken = dp_bottom_up(available_item_df, max_capa)
    if mode == "5":
        print(f"Use algorithm {mode}")
        optimal_value, taken = KnapSackBranchNBound(max_capa, items, item_count)
    
    # prepare the solution in the specified output format
    output_data = str(optimal_value) + ' ' + str(0) + '\n'
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
        print(solve_it(input_data))
    else:
        print('This test requires an input file.  Please select one from the data directory. (i.e. python solver.py ./data/ks_4_0)')

