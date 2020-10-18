import copy
import random
import queue
from time import sleep

# Returns a tuple of tuples representation of a state from a pile
def LoadFromFile(filepath):
    with open(filepath, "r") as f:
        data = f.readlines()
        print(data)
        N = int(data[0][0])
        if len(data) != N + 1:
            print("Error"); return
        table = []
        for i in range(1,N+1):
            convert = lambda x: int(x) if x != "*" else 0
            row = [convert(num) for num in data[i].strip().split('\t')]
            if len(row) != N:
                print("Error"); return
            table.append(row)
        return tuple(tuple(row) for row in table)

# Prints state
def DebugPrint(state):
    for row in state:
        line = ""
        for num in row:
            line += str(num) + '\t'
        print(line)

# Takes in state and positions of tiles to be swapped
# Returns new state
def swap_tiles(state, r1, c1, r2, c2):
    new_state = list(list(row) for row in copy.deepcopy(state))
    new_state[r1][c1] = new_state[r2][c2]
    new_state[r2][c2] = state[r1][c1]
    return tuple(tuple(row) for row in new_state)

# Computes neighbors of a state
# Return a collection of pairs: (tile moved into hole, new state reached)
def ComputeNeighbors(state):
    N = len(state)
    # find 0
    for r in range(N):
        for c in range(N):
            if state[r][c] == 0:
                hole_r = r
                hole_c = c
    
    # find neighbor tiles of 0
    neighbors = []
    if hole_r < N - 1:
        neighbors.append((state[hole_r + 1][hole_c], swap_tiles(state, hole_r, hole_c, hole_r + 1, hole_c)))
    if hole_r > 0:
        neighbors.append((state[hole_r - 1][hole_c], swap_tiles(state, hole_r, hole_c, hole_r - 1, hole_c)))
    if hole_c < N - 1:
        neighbors.append((state[hole_r][hole_c + 1], swap_tiles(state, hole_r, hole_c, hole_r, hole_c + 1)))
    if hole_c > 0:
        neighbors.append((state[hole_r][hole_c - 1], swap_tiles(state, hole_r, hole_c, hole_r, hole_c - 1)))
    return neighbors

# Returns flattened version of nested list
def flatten(nested_list):
    current_list = []
    def flatten_rec(nested_list):
        if not isinstance(nested_list, list):
            current_list.append(nested_list)
        else:
            for i in nested_list:
                flatten_rec(i)
        return current_list
    return flatten_rec(nested_list)

# Return True if state is goal and False otherwise
def IsGoal(state):
    return flatten(list(list(i) for i in state)) == list(range(1, len(state)**2)) + [0]

# Use AStar algorithm to find solution to puzzle.
# Returns: a sequence of tiles to reach goal, None if no solution found
def BFS(state):
    frontier = [(0, state)]
    discovered = set([state])
    parents = {(0, state): None}
    path = []
    while len(frontier) != 0:
        current_state = frontier.pop(0)
        discovered.add(current_state[1])
        if IsGoal(current_state[1]):
            while parents.get((current_state[0], current_state[1])) != None:
                path.insert(0, current_state[0])
                current_state = parents.get((current_state[0], current_state[1]))
            return path
        for neighbor in ComputeNeighbors(current_state[1]):
            # print(neighbor[0])
            if neighbor[1] not in discovered:
                frontier.append(neighbor)
                discovered.add(neighbor[1])
                parents.update({(neighbor[0], neighbor[1]): current_state})
    print("FAIL")
    return None

# Use AStar algorithm to find solution to puzzle.
# Returns: a sequence of tiles to reach goal, None if no solution found
def DFS(state):
    frontier = [(0, state)]
    discovered = set([state])
    parents = {(0, state): None}
    path = []
    while len(frontier) != 0:
        current_state = frontier.pop(0)
        discovered.add(current_state[1])
        if IsGoal(current_state[1]):
            while parents.get((current_state[0], current_state[1])) != None:
                path.insert(0, current_state[0])
                current_state = parents.get((current_state[0], current_state[1]))
            return path
        for neighbor in ComputeNeighbors(current_state[1]):
            if neighbor[1] not in discovered:
                frontier.insert(0, neighbor)
                discovered.add(neighbor[1])
                parents.update({(neighbor[0], neighbor[1]): current_state})
    print("FAIL")
    return None

# Provides heuristic for AStar function
# Returns: sum of L1 distances to goal for each block
def h(state):
    dist = 0
    for r in range(len(state)):
        for c in range(len(state)):
            i = state[r][c]
            if i == 0:
                goal_r = len(state) - 1
                goal_c = len(state) - 1
            else:
                goal_r = (i-1)//len(state)
                goal_c = (i%len(state)-1)%len(state)
            dist += abs(goal_r-r) + abs(goal_c-c)
    return dist

# Use AStar algorithm to find solution to puzzle.
# Returns: a sequence of tiles to reach goal, None if no solution found
def AStar(state):
    frontier = queue.PriorityQueue()
    frontier.put((0, (0, state, 0)))           # Heuristic, (tile, state, cost from start to state) 
    discovered = set([state])
    parents = {(0, state): None}
    path = []
    while not frontier.empty():
        current_state = frontier.get()
        discovered.add(current_state[1][1])
        if IsGoal(current_state[1][1]):
            while parents.get(current_state[1][:2]) != None:
                path.insert(0, current_state[1][0])
                current_state = parents.get(current_state[1][:2])
            return path
        for neighbor in ComputeNeighbors(current_state[1][1]):
            # print(neighbor[0])
            if neighbor[1] not in discovered:
                steps_taken = current_state[1][2] + 1
                frontier.put( ( h(neighbor[1]) + steps_taken, tuple(list(neighbor) + [steps_taken]) ) )
                discovered.add(neighbor[1])
                parents.update({(neighbor[0], neighbor[1]): current_state})
    print("FAIL")
    return None

# Method for creating test cases by randomly moving to a neighbor.
# Returns: new state after making (steps) random steps
def shuffle(state, steps=30):
    current_state = state
    for i in range(steps):
        current_state = random.choice(ComputeNeighbors(current_state))[1]
    return current_state

table = LoadFromFile("test_puzzle.txt")
DebugPrint(table)
print(IsGoal(table))
print(h(table))
# print(20*"-")
# table = swap_tiles(table, 3, 3, 3, 2)
# table = swap_tiles(table, 3, 2, 3, 1)
table = shuffle(table)
DebugPrint(table)
# print(h(table))
# print("Neighbors: " + 40*"-")
# n = ComputeNeighbors(table)
# for i in n:
#     print(i[0])
#     DebugPrint(i[1])
#     print(40*"-")
result = AStar(table)
print(result)
print(len(result))
