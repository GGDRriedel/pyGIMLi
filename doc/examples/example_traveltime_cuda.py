#!/usr/bin/env python
"""
Example: Travel Time Inversion with CUDA/GPU Acceleration
==========================================================

This example demonstrates how to use CUDA/GPU acceleration for travel time
inversion in pyGIMLi. It shows:

1. How to check if CUDA is available
2. How to enable CUDA acceleration
3. Performance comparison between CPU and GPU
4. Automatic fallback when CUDA is not available

Requirements:
- pyGIMLi built with CUDA support (cmake -DUSE_CUDA=ON)
- NVIDIA GPU with CUDA drivers installed
"""

import time
import numpy as np
import pygimli as pg
from pygimli.physics import TravelTimeManager
from pygimli.physics import traveltime as tt

# %%
# Check CUDA Availability
# -----------------------
# First, let's check if CUDA is available on this system

print("=" * 60)
print("CUDA Availability Check")
print("=" * 60)

cuda_available = TravelTimeManager.cudaAvailable()
print(f"CUDA available: {cuda_available}")

if cuda_available:
    print("✓ GPU acceleration can be used")
else:
    print("✗ GPU acceleration not available, will use CPU")
    print("  This is normal if:")
    print("  - pyGIMLi was not built with CUDA support")
    print("  - No NVIDIA GPU is present")
    print("  - CUDA drivers are not installed")

print()

# %%
# Create Synthetic Data
# ---------------------
# Create a simple 2D mesh and synthetic travel time data

print("=" * 60)
print("Creating Synthetic Problem")
print("=" * 60)

# Create mesh
x = np.arange(0, 50, 2.0)
y = np.arange(0, 30, 2.0)
mesh = pg.createGrid(x=x, y=y)
print(f"Mesh: {mesh.cellCount()} cells, {mesh.nodeCount()} nodes")

# Create measurement scheme
sensors = [(xi, 0) for xi in x]
scheme = tt.createRAData(sensors, shotDistance=10)
print(f"Scheme: {scheme.size()} measurements, {scheme.sensorCount()} sensors")

# Define velocity model (simple two-layer model)
velocity_model = pg.Vector(mesh.cellCount())
for cell in mesh.cells():
    if cell.center().y() > -15:
        velocity_model[cell.id()] = 1000  # Upper layer: 1000 m/s
    else:
        velocity_model[cell.id()] = 2000  # Lower layer: 2000 m/s

print(f"Velocity model: {min(velocity_model)} - {max(velocity_model)} m/s")
print()

# %%
# Simulation with CPU
# -------------------
# First, simulate with CPU to get baseline performance

print("=" * 60)
print("Simulation: CPU")
print("=" * 60)

mgr_cpu = TravelTimeManager(useCuda=False, verbose=True)
mgr_cpu.fop.setVerbose(False)

t_start = time.time()
data_cpu = mgr_cpu.simulate(mesh=mesh, scheme=scheme, vel=velocity_model, 
                            secNodes=3, noiseLevel=0.01)
t_cpu = time.time() - t_start

print(f"CPU Time: {t_cpu:.4f} seconds")
print(f"Travel times: {min(data_cpu['t']):.6f} - {max(data_cpu['t']):.6f} s")
print()

# %%
# Simulation with GPU (if available)
# ----------------------------------
# Now simulate with CUDA acceleration

print("=" * 60)
print("Simulation: GPU (CUDA)")
print("=" * 60)

if cuda_available:
    mgr_gpu = TravelTimeManager(useCuda=True, verbose=True)
    mgr_gpu.fop.setVerbose(False)
    
    t_start = time.time()
    data_gpu = mgr_gpu.simulate(mesh=mesh, scheme=scheme, vel=velocity_model,
                                secNodes=3, noiseLevel=0.01)
    t_gpu = time.time() - t_start
    
    print(f"GPU Time: {t_gpu:.4f} seconds")
    print(f"Travel times: {min(data_gpu['t']):.6f} - {max(data_gpu['t']):.6f} s")
    
    # Check if results match
    diff = np.max(np.abs(data_cpu['t'] - data_gpu['t']))
    print(f"Maximum difference (CPU vs GPU): {diff:.2e} s")
    
    if diff < 1e-6:
        print("✓ GPU results match CPU results")
    else:
        print("✗ GPU results differ from CPU results")
    
    # Performance comparison
    speedup = t_cpu / t_gpu
    print(f"\nSpeedup: {speedup:.2f}x")
    
    if speedup > 1.0:
        print(f"✓ GPU is {speedup:.2f}x faster than CPU")
    else:
        print(f"Note: GPU is slower than CPU for this small problem")
        print("     GPU benefits are typically seen for larger problems")
else:
    print("CUDA not available, skipping GPU simulation")

print()

# %%
# Inversion with Automatic GPU Selection
# ---------------------------------------
# The manager will automatically use GPU if available

print("=" * 60)
print("Inversion with Automatic GPU Selection")
print("=" * 60)

# Create manager (automatically uses GPU if available)
mgr = TravelTimeManager(useCuda=cuda_available)

# Use the CPU-generated data for inversion
mgr.fop.setData(scheme)

print(f"Using CUDA: {mgr.useCuda()}")

# Run inversion
t_start = time.time()
velocity_inv = mgr.invert(data_cpu['t'], mesh=mesh, secNodes=3, 
                          lam=100, verbose=False)
t_inv = time.time() - t_start

print(f"Inversion time: {t_inv:.4f} seconds")
print(f"Chi-squared: {mgr.inv.chi2():.4f}")
print()

# %%
# Summary
# -------

print("=" * 60)
print("Summary")
print("=" * 60)
print(f"CUDA available: {cuda_available}")
print(f"Used GPU: {mgr.useCuda()}")
print(f"Mesh size: {mesh.cellCount()} cells")
print(f"Number of measurements: {scheme.size()}")
if cuda_available:
    print(f"Speedup (simulation): {speedup:.2f}x")
print()
print("For best GPU performance:")
print("- Use larger meshes (>10,000 nodes)")
print("- Use more measurements (>100 shots)")
print("- GPU benefits increase with problem size")
print()

# %%
# Plot Results
# ------------
# Visualize the inversion results

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True model
    ax = axes[0]
    pg.show(mesh, velocity_model, ax=ax, cMap='plasma', 
            colorBar=True, label='Velocity (m/s)')
    ax.set_title('True Velocity Model')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    # Inverted model
    ax = axes[1]
    pg.show(mesh, velocity_inv, ax=ax, cMap='plasma',
            colorBar=True, label='Velocity (m/s)')
    ax.set_title(f'Inverted Model ({"GPU" if mgr.useCuda() else "CPU"})')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    
    plt.tight_layout()
    plt.savefig('traveltime_cuda_example.png', dpi=150)
    print("Results saved to: traveltime_cuda_example.png")
    
except ImportError:
    print("Matplotlib not available, skipping visualization")

print("\nExample completed successfully!")
