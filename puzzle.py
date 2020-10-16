import copy

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
        return (table)

def DebugPrint(state):
    for row in state:
        line = ""
        for num in row:
            line += str(num) + '\t'
        print(line)

# Takes in state and positions of tiles to be swapped
# Returns new state
def swap_tiles(state, r1, c1, r2, c2):
    new_state = copy.deepcopy(state)
    new_state[r1][c1] = new_state[r2][c2]
    new_state[r2][c2] = state[r1][c1]
    return new_state

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

def IsGoal(state):
    return flatten(state) == list(range(1, len(state)**2)) + [0]

def BFS(state):
    frontier = [(0, state)]
    discovered = set(tuple(flatten(state)))
    parents = {(0, tuple(flatten(state))): None}
    path = []
    while len(frontier) != 0:
        current_state = frontier.pop(0)
        discovered.add(tuple(flatten(current_state[1])))
        if IsGoal(current_state[1]):
            while parents.get((current_state[0], tuple(flatten(current_state[1])))) != None:
                path.insert(0, current_state[0])
                current_state = parents.get((current_state[0], tuple(flatten(current_state[1]))))
            return path
        for neighbor in ComputeNeighbors(current_state[1]):
            if tuple(flatten(neighbor[1])) not in discovered:
                frontier.append(neighbor)
                discovered.add(tuple(flatten(neighbor[1])))
                parents.update({(neighbor[0], tuple(flatten(neighbor[1]))): current_state})
    print("FAIL")
    return None

table = LoadFromFile("test_puzzle.txt")
DebugPrint(table)
# print(20*"-")
table = swap_tiles(table, 3, 3, 2, 2)
DebugPrint(table)
# DebugPrint(table)
# print("Neighbors: " + 40*"-")
# n = ComputeNeighbors(table)
# for i in n:
#     print(i[0])
#     DebugPrint(i[1])
#     print(40*"-")
print(BFS(table))