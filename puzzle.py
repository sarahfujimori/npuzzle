import copy
import random
import queue
from time import sleep
import time

def profile(f):
    def g(arg):
        start = time.time()
        ans = f(arg)
        end = time.time()
        print(f.__name__ + " " + str(arg) + " took " + str(end-start) + " seconds")
        return ans
    return g

def memoize(f):
    memo = {}
    def helper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]
    return helper

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
@memoize
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

# Returns True if the first row and column are correct
def IsPGoal(state, iterations):
    original = target(len(state) + iterations)
    original_row = original[iterations]
    original_col = tuple([original[i][iterations] for i in range(len(state)+iterations)])
    goal_row = original_row[iterations:]
    goal_col = original_col[iterations:]
    state_row = state[0]
    state_col = tuple([state[i][0] for i in range(len(state))])
    if state_row == goal_row and state_col == goal_col:
        return True
    return False

# Check if the target can be reached through the current state
# Source: https://www.geeksforgeeks.org/check-instance-15-puzzle-solvable/
# Returns true if solvable, false otherwise
def isSolvable(state):
    N = len(state)
    l = flatten(list(list(i) for i in state))
    l.remove(0)

    # count inversions
    count = 0
    for i in range(len(l)):
        for j in range(len(l)):
            if i < j and l[i] > l[j]:       
                count += 1
    if N % 2 == 1 and count % 2 == 0:
        return True
    elif N % 2 == 0:
        # find hole coords
        for r in range(N):
            for c in range(N):
                if state[r][c] == 0:
                    hole_r = r
                    hole_c = c
        row_count_bottom = N - hole_r       # Returns 1 if last row, 2 if second to last row, etc.
        if count % 2 == 1 and row_count_bottom % 2 == 0:
            return True
        elif count % 2 == 0 and row_count_bottom % 2 == 1:
            return True
    return False

# Use BFS algorithm to find solution to puzzle.
# Returns: a sequence of tiles to reach goal, None if no solution found

def BFS(state, iterations=0):
    if iterations == 0 and not isSolvable(state):
        return None
    if len(state) == 1:
        return []
    frontier = [(0, state)]
    discovered = set([state])
    parents = {(0, state): None}
    path = []
    while len(frontier) != 0:
        current_state = frontier.pop(0)
        discovered.add(current_state[1])
        if IsPGoal(current_state[1], iterations):
            new_state = current_state[1][1:]
            new_state = tuple([row[1:] for row in new_state])
            while parents.get((current_state[0], current_state[1])) != None:
                path.insert(0, current_state[0])
                current_state = parents.get((current_state[0], current_state[1]))
            return path + BFS(new_state, iterations+1)
        for neighbor in ComputeNeighbors(current_state[1]):
            if neighbor[1] not in discovered:
                frontier.append(neighbor)
                discovered.add(neighbor[1])
                parents.update({(neighbor[0], neighbor[1]): current_state})
    print("FAIL")
    return None

# Use DFS algorithm to find solution to puzzle.
# Returns: a sequence of tiles to reach goal, None if no solution found
def DFS(state, iterations=0):
    if iterations == 0 and not isSolvable(state):
        return None
    if len(state) == 1:
        return []
    frontier = [(0, state)]
    discovered = set([state])
    parents = {(0, state): None}
    path = []
    while len(frontier) != 0:
        current_state = frontier.pop(0)
        discovered.add(current_state[1])
        if IsPGoal(current_state[1], iterations):
            new_state = current_state[1][1:]
            new_state = tuple([row[1:] for row in new_state])
            while parents.get((current_state[0], current_state[1])) != None:
                path.insert(0, current_state[0])
                current_state = parents.get((current_state[0], current_state[1]))
            return path + DFS(new_state, iterations+1)
        for neighbor in ComputeNeighbors(current_state[1]):
            if neighbor[1] not in discovered:
                frontier.insert(0, neighbor)
                discovered.add(neighbor[1])
                parents.update({(neighbor[0], neighbor[1]): current_state})
    print("FAIL")
    return None

#Returns state representation of target 
def target(n):
    l = [list(range(n * i + 1, n * (i + 1) + 1)) for i in range(n)]
    l[-1][-1] = 0
    return tuple(tuple(row) for row in l)

# Use Bidirectional Search to find a path from state to target
# Get each row of nodes for both frontier and backtier; compare all the nodes from that row
# If there are any commonalities, then you're done; otherwise, continue to the next row of nodes
# Returns: None if not solvable, list of tiles to target otherwise
@profile
def BidirectionalSearch(state):
    if not isSolvable(state):
        return None
    target_1 = target(len(state))

    frontier = [(0, state)]
    backtier = [(0, target_1)]

    discovered_frontier = set([state])
    discovered_backtier = set([target_1])

    parents_frontier = {state: []}
    parents_backtier = {target_1: []}

    path = []

    while len(frontier) != 0 and len(backtier) != 0: 
        current_state_frontier = frontier.pop(0)
        current_state_backtier = backtier.pop(0)
        
        discovered_frontier.add(current_state_frontier[1])
        discovered_backtier.add(current_state_backtier[1])

        if len(discovered_frontier.intersection(discovered_backtier)) > 0: 
            intersect = list(discovered_frontier.intersection(discovered_backtier))[0]
            path_1 = parents_frontier[intersect]
            path_2 = list(reversed(parents_backtier[intersect]))
            return path_1 + path_2


        for neighbor in ComputeNeighbors(current_state_frontier[1]):
            if neighbor[1] not in discovered_frontier:
                frontier.append(neighbor)
                discovered_frontier.add(neighbor[1])
                parents_frontier.update({neighbor[1]: parents_frontier[current_state_frontier[1]] + [neighbor[0]]})

        for neighbor in ComputeNeighbors(current_state_backtier[1]):
            if neighbor[1] not in discovered_backtier:
                backtier.append(neighbor)
                discovered_backtier.add(neighbor[1])
                parents_backtier.update({neighbor[1]: parents_backtier[current_state_backtier[1]] + [neighbor[0]]})

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
@profile
def AStar(state):
    if not isSolvable(state):
        return None
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
table = shuffle(table, 20)
# table = ((1, 3, 4, 8), (6, 2, 7, 0), (5, 9, 10, 12), (13, 14, 11, 15))
print(table)
DebugPrint(table)
# print(h(table))
# print("Neighbors: " + 40*"-")
# n = ComputeNeighbors(table)
# for i in n:
#     print(i[0])
#     DebugPrint(i[1])
#     print(40*"-")
print(DFS(table))
print(BidirectionalSearch(table))
print(AStar(table))
start = time.time()
print(BFS(table))
print(time.time() - start)
# state = ((3, 9, 1, 15), (14, 11, 4, 6), (13, 0, 10, 12), (2, 7, 8, 5))
# print(isSolvable(state))
s = ((6, 7, 8), (6, 11, 12), (14, 15, 0))
print(IsPGoal(s, 1))