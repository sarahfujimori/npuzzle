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
        print(row)

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
        neighbors.append(swap_tiles(state, hole_r, hole_c, hole_r + 1, hole_c))
    if hole_r > 0:
        neighbors.append(swap_tiles(state, hole_r, hole_c, hole_r - 1, hole_c))
    if hole_c < N - 1:
        neighbors.append(swap_tiles(state, hole_r, hole_c, hole_r, hole_c + 1))
    if hole_c > 0:
        neighbors.append(swap_tiles(state, hole_r, hole_c, hole_r, hole_c - 1))
    return neighbors

table = LoadFromFile("test_puzzle.txt")
DebugPrint(table)
print(20*"-")
# DebugPrint(swap_tiles(table, 3, 3, 2, 3))
# print("State: " + 20*"-")
# DebugPrint(table)
table = swap_tiles(table, 3, 3, 2, 2)
DebugPrint(table)
print("Neighbors: " + 40*"-")
n = ComputeNeighbors(table)
for i in n:
    DebugPrint(i)
    print(40*"-")
