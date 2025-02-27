import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
import time

# Constants
GRID_SIZE = 800  # 800x800 forest grid
FIRE_SPREAD_PROB = 0.3  # Probability that fire spreads to a neighboring tree
BURN_TIME = 3  # Time before a tree turns into ash
DAYS = 60  # Maximum simulation time

# State definitions
EMPTY = 0  # No tree
TREE = 1  # Healthy tree
BURNING = 2  # Burning tree
ASH = 3  # Burned tree


def initialize_forest():
    """Creates a forest grid with all trees and ignites one random tree."""
    forest = np.ones((GRID_SIZE, GRID_SIZE), dtype=int)  # All trees
    burn_time = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)  # Tracks how long a tree burns

    # Ignite a random tree
    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    forest[x, y] = BURNING
    burn_time[x, y] = 1  # Fire starts burning

    return forest, burn_time


def get_neighbors(x, y):
    """Returns the neighboring coordinates of a cell in the grid."""
    neighbors = []
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            neighbors.append((nx, ny))
    return neighbors


def simulate_wildfire(_):
    """Simulates wildfire spread over time."""
    forest, burn_time = initialize_forest()

    fire_spread = []  # Track number of burning trees each day

    for day in range(DAYS):
        new_forest = forest.copy()

        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                if forest[x, y] == BURNING:
                    burn_time[x, y] += 1  # Increase burn time

                    # If burn time exceeds threshold, turn to ash
                    if burn_time[x, y] >= BURN_TIME:
                        new_forest[x, y] = ASH

                    # Spread fire to neighbors
                    for nx, ny in get_neighbors(x, y):
                        if forest[nx, ny] == TREE and random.random() < FIRE_SPREAD_PROB:
                            new_forest[nx, ny] = BURNING
                            burn_time[nx, ny] = 1

        forest = new_forest.copy()
        fire_spread.append(np.sum(forest == BURNING))

        if np.sum(forest == BURNING) == 0:  # Stop if no more fire
            break

    return fire_spread


if __name__ == '__main__':
    num_simulations = 8  # Number of parallel simulations to run

    start_time = time.time()
    with multiprocessing.Pool() as pool:
        # Run simulations in parallel
        results = pool.map(simulate_wildfire, range(num_simulations))
    print(f"{num_simulations} parallel simulations completed in {time.time() - start_time:.2f} seconds.")
    print(f"Average execution time per simulation: {(time.time() - start_time)/num_simulations:.2f} seconds.")

    max_days = max(len(res) for res in results)
    padded_results = [res + [0] * (max_days - len(res)) for res in results]

    # Calculate the average number of burning trees per day
    padded_array = np.array(padded_results)
    average_spread = np.mean(padded_array, axis=0)

    # Plotting the average fire spread
    plt.figure(figsize=(10, 6))
    plt.plot(average_spread, label='Average Burning Trees')
    plt.xlabel('Day')
    plt.ylabel('Number of Burning Trees')
    plt.title(f'Average Wildfire Spread Over {num_simulations} Simulations')
    plt.legend()
    plt.grid(True)
    plt.show()