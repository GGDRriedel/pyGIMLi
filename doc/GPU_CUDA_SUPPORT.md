# GPU/CUDA Support for Travel Time Inversion

## Overview

pyGIMLi now supports GPU acceleration using CUDA for travel time inversion computations. The Dijkstra shortest path algorithm, which is the computational bottleneck for travel time problems, has been parallelized to run on NVIDIA GPUs.

## Requirements

To use GPU acceleration, you need:

1. **NVIDIA GPU** with CUDA support (compute capability 3.5 or higher recommended)
2. **CUDA Toolkit** (version 11.0 or higher recommended)
3. **pyGIMLi built with CUDA support** (see Building with CUDA below)

## Building with CUDA

To build pyGIMLi with CUDA support, you need to enable the CUDA option during compilation:

```bash
mkdir build
cd build
cmake -DUSE_CUDA=ON ..
make
```

The build system will automatically detect your CUDA installation. If CUDA is not found, the build will continue without GPU support and display a warning.

## Usage

### Python API

#### Basic Usage

Enable CUDA when creating a TravelTimeManager:

```python
import pygimli as pg
from pygimli.physics import TravelTimeManager

# Create manager with CUDA enabled
mgr = TravelTimeManager(useCuda=True)

# Load your data and run inversion as usual
data = mgr.load("mydata.dat")
mgr.invert(data)
```

#### Check CUDA Availability

Before using CUDA, you can check if it's available:

```python
from pygimli.physics import TravelTimeManager

# Check if CUDA is available
if TravelTimeManager.cudaAvailable():
    print("CUDA is available")
    mgr = TravelTimeManager(useCuda=True)
else:
    print("CUDA not available, using CPU")
    mgr = TravelTimeManager()
```

#### Dynamic CUDA Control

You can enable or disable CUDA at runtime:

```python
mgr = TravelTimeManager()

# Enable CUDA
mgr.setUseCuda(True)

# Check current state
print(f"Using CUDA: {mgr.useCuda()}")

# Disable CUDA
mgr.setUseCuda(False)
```

#### Using CUDA with Forward Modelling

```python
import pygimli as pg
from pygimli.physics.traveltime import TravelTimeDijkstraModelling

# Create forward operator with CUDA
fop = TravelTimeDijkstraModelling(useCuda=True)

# Set mesh and data
fop.setMesh(mesh, ignoreRegionManager=True)
fop.setData(data)

# Compute response (will use GPU if available)
response = fop.response(slowness)
```

## Automatic Fallback

If CUDA is requested but not available (e.g., no GPU, CUDA not installed, or build without CUDA support), pyGIMLi will automatically fall back to CPU computation. A warning message will be displayed:

```
Warning: CUDA requested but not available. Using CPU.
```

This ensures your code will work on any system, whether or not CUDA is available.

## Performance Considerations

### When to Use CUDA

GPU acceleration provides the most benefit when:
- You have a large number of shots and receivers (>100 shots)
- The mesh is relatively large (>10,000 nodes)
- You're running multiple forward simulations (e.g., during inversion)

### When Not to Use CUDA

For small problems, the overhead of transferring data to/from the GPU may outweigh the computational benefits:
- Small meshes (<1,000 nodes)
- Few shots (<10)
- Single forward computation only

### Memory Limitations

GPU memory is typically more limited than system RAM. For very large problems, you may encounter out-of-memory errors on the GPU. In such cases, the system will fall back to CPU computation.

## Troubleshooting

### CUDA Not Found During Build

If CMake cannot find CUDA:
1. Make sure CUDA Toolkit is installed
2. Set `CUDA_HOME` or `CUDACXX` environment variable
3. Add CUDA to your PATH

### Runtime CUDA Errors

If you encounter CUDA errors at runtime:
1. Check GPU drivers are up to date
2. Verify GPU is not being used by other applications
3. Monitor GPU memory usage
4. Try disabling CUDA to verify the problem is GPU-specific

### Performance Not Improved

If GPU acceleration doesn't improve performance:
1. Ensure your problem is large enough (see Performance Considerations)
2. Profile to check if data transfer is the bottleneck
3. Monitor GPU utilization
4. Verify CUDA is actually being used (check for "Computed with CUDA" message)

## Technical Details

### Implementation

The CUDA implementation uses:
- Parallel Dijkstra algorithm with GPU kernels
- CSR (Compressed Sparse Row) graph representation
- Atomic operations for thread-safe distance updates
- Reduction kernels for finding minimum distance nodes

### Limitations

Current limitations:
- Path reconstruction (for Jacobian) falls back to CPU
- Only distance computation is GPU-accelerated
- Single GPU support only (no multi-GPU)

## Examples

### Example 1: Simple Travel Time Inversion with GPU

```python
import pygimli as pg
from pygimli.physics import TravelTimeManager

# Check CUDA availability
if TravelTimeManager.cudaAvailable():
    print("CUDA available! Using GPU acceleration.")
    mgr = TravelTimeManager(useCuda=True)
else:
    print("CUDA not available. Using CPU.")
    mgr = TravelTimeManager()

# Load and invert data
data = mgr.load("traveltime_data.sgt")
velocity = mgr.invert(data)

# Show results
mgr.showResult()
```

### Example 2: Simulation with GPU

```python
import pygimli as pg
from pygimli.physics import traveltime as tt

# Create mesh and data scheme
mesh = pg.createGrid(x=range(50), y=range(30))
scheme = tt.createRAData(sensors=[(i, 0) for i in range(50)], 
                         shotDistance=5)

# Create manager with CUDA
mgr = tt.TravelTimeManager(useCuda=True)

# Simulate with GPU acceleration
data = mgr.simulate(mesh=mesh, scheme=scheme, 
                    vel=[[1, 1000], [2, 2000]])
                    
print(f"Simulated {data.size()} traveltimes using GPU")
```

## References

For more information:
- pyGIMLi documentation: https://www.pygimli.org
- CUDA documentation: https://docs.nvidia.com/cuda/
- Report issues: https://github.com/gimli-org/gimli/issues
