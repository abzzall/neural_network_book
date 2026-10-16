"""A 3x3 grid world: reach the goal (+1), avoid the pit (-1)."""

ROWS, COLS = 3, 3
GOAL, PIT, START = (0, 2), (1, 2), (2, 0)
ACTIONS = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}
STATES = [(r, c) for r in range(ROWS) for c in range(COLS)]
TERMINAL = {GOAL, PIT}


def step(state, action):
    """Deterministic move; walking into a wall leaves the agent in place."""
    dr, dc = ACTIONS[action]
    r, c = state[0] + dr, state[1] + dc
    nxt = (r, c) if 0 <= r < ROWS and 0 <= c < COLS else state
    reward = 1.0 if nxt == GOAL else -1.0 if nxt == PIT else 0.0
    return nxt, reward, nxt in TERMINAL


def show(values):
    """Print a value table row by row; terminal cells are marked G and P."""
    for r in range(ROWS):
        cells = []
        for c in range(COLS):
            s = (r, c)
            cells.append('   G  ' if s == GOAL else '   P  ' if s == PIT
                         else f'{values[s]:6.3f}')
        print(' '.join(cells))
