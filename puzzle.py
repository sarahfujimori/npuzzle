def LoadFromFile(filepath):
    with open(filepath, "r") as f:
        data = f.readlines()
        print(data)
        N = int(data[0][0])
        print(N)
        table = []
        for i in range(1,N+1):
            convert = lambda x: int(x) if x != "*" else 0
            table.append([convert(num) for num in data[i].strip().split('\t')])
        return (table)

def DebugPrint(state):
    for row in state:
        print(row)

# Takes in state and positions of tiles to be swapped
# Returns new state
def swap_tiles(state, r1, c1, r2, c2):
    temp = state[r1][c1]
    state[r1][c1] = state[r2][c2]
    state[r2][c2] = temp
    return state

def ComputeNeighbors(state):
    # find 0
    for r in range(len(state)):
        for c in range(len(state[r])):
            if state[r][c] == 0:
                hole_r = r
                hole_c = c
    
    # find neighbor tiles of 0
    neighbors = []
    if hole_r + 1 < len(state):
        print("s")

table = LoadFromFile("test_puzzle.txt")
DebugPrint(table)
ComputeNeighbors(table)