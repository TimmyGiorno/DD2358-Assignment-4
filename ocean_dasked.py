import numpy as np
import matplotlib.pyplot as plt

# TASK B1
import dask.array as da
import time

# TASK B2.1
# pip install dask distributed
# from dask.distributed import Client

# TASK B2.4
from pyevtk.hl import imageToVTK

# Grid size
grid_size = 200
TIME_STEPS = 100
chunk_size = (100, 100)
output_interval = 10

# Initialize Dask arrays
temperature = da.random.uniform(5, 30, size=(grid_size, grid_size), chunks=chunk_size)
u_velocity = da.random.uniform(-1, 1, size=(grid_size, grid_size), chunks=chunk_size)
v_velocity = da.random.uniform(-1, 1, size=(grid_size, grid_size), chunks=chunk_size)
wind = da.random.uniform(-0.5, 0.5, size=(grid_size, grid_size), chunks=chunk_size)


def laplacian(field):
    """Computes the discrete Laplacian of a 2D field using finite differences with ghost cells."""
    return da.map_overlap(
        lambda f: (
            np.roll(f, shift=1, axis=0) + np.roll(f, shift=-1, axis=0) +
            np.roll(f, shift=1, axis=1) + np.roll(f, shift=-1, axis=1) - 4 * f
        ),
        field,
        depth=1,
        boundary='reflect',  # Reflecting boundaries to maintain continuity
    )


def update_ocean(u, v, temperature, wind, alpha=0.1, beta=0.02):
    """Updates ocean velocity and temperature fields using a simplified flow model."""
    u_new = u + alpha * laplacian(u) + beta * wind
    v_new = v + alpha * laplacian(v) + beta * wind
    temperature_new = temperature + 0.01 * laplacian(temperature)  # Small diffusion
    return u_new, v_new, temperature_new


def save_vtk(filename, temperature, u, v):

    temp_3d = temperature.T.reshape(grid_size, grid_size, 1)
    u_3d = u.T.reshape(grid_size, grid_size, 1)
    v_3d = v.T.reshape(grid_size, grid_size, 1)
    zeros = np.zeros_like(u_3d)

    point_data = {
        "temperature": temp_3d,
        "velocity": (u_3d, v_3d, zeros)
    }

    imageToVTK(
        filename,
        origin=(0, 0, 0),
        spacing=(1, 1, 1),
        pointData=point_data
    )
    print(f"Saved VTK file: {filename}.vtr")


if __name__ == "__main__":

    # B2.1
    # client = Client()
    # print("Dask Dashboard running at:", client.dashboard_link)

    # Run the simulation
    start_time = time.time()
    for t in range(TIME_STEPS):
        u_velocity, v_velocity, temperature = update_ocean(
            u_velocity, v_velocity, temperature, wind
        )

        if t % output_interval == 0 or t == TIME_STEPS - 1:
            u_np, v_np, temp_np = da.compute(u_velocity, v_velocity, temperature)
            filename = f"ocean_sim_{t:04d}"
            save_vtk(filename, temp_np, u_np, v_np)

    print(f"Simulation completed in {time.time() - start_time:.2f} seconds.")


