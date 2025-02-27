import numpy as np
import matplotlib.pyplot as plt
import random
import dask.array as da
from dask import delayed, compute
from dask.distributed import Client
import time

# Constants
GRID_SIZE = 800
FIRE_SPREAD_PROB = 0.3
BURN_TIME = 3
DAYS = 60
SIMULATIONS = 8  # Number of parallel simulations

# State definitions
EMPTY = 0
TREE = 1
BURNING = 2
ASH = 3


def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)

    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1

    return forest, burn_time


def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors


@delayed
def simulate_wildfire():
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()
    fire_spread = []

    for day in range(DAYS):
        new_forest = forest.copy()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1

                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH

                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1

        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))

        if np.sum(forest == BURNING) == 0:
            break

    return fire_spread

if __name__ == '__main__':
    client = Client()

    start_time = time.time()
    # Run multiple simulations in parallel
    simulations = [simulate_wildfire() for _ in range(SIMULATIONS)]
    results = compute(*simulations)
    print(f"{SIMULATIONS} parallel simulations completed in {time.time() - start_time:.2f} seconds.")
    print(f"Average execution time per simulation: {(time.time() - start_time) / SIMULATIONS:.2f} seconds.")

    max_days = max(len(res) for res in results)
    padded_results = [res + [0] * (max_days - len(res)) for res in results]

    # Convert to Dask array for efficient aggregation
    fire_spread_results = da.from_array(padded_results, chunks=(1, DAYS))
    mean_fire_spread = fire_spread_results.mean(axis=0).compute()

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mean_fire_spread)), mean_fire_spread, label="Avg Burning Trees")
    plt.xlabel("Day")
    plt.ylabel("Number of Burning Trees")
    plt.title("Wildfire Spread Over Time (Parallelized with Dask)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Close Dask client
    client.close()
