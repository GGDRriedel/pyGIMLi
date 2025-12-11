# CUDA/GPU Examples for pyGIMLi

This directory contains examples demonstrating GPU acceleration features in pyGIMLi.

## Travel Time Inversion with CUDA

### example_traveltime_cuda.py

This comprehensive example shows:

- **CUDA availability detection**: Check if GPU acceleration is available
- **Performance comparison**: Compare CPU vs GPU computation times
- **Automatic fallback**: Graceful degradation when CUDA is unavailable
- **Usage patterns**: Best practices for using CUDA in pyGIMLi

### Running the Example

```bash
# Basic execution
python example_traveltime_cuda.py

# With plotting (requires matplotlib)
python example_traveltime_cuda.py
```

### Expected Output

When CUDA is available:
```
==============================================================
CUDA Availability Check
==============================================================
CUDA available: True
✓ GPU acceleration can be used

==============================================================
Creating Synthetic Problem
==============================================================
Mesh: 600 cells, 675 nodes
Scheme: 120 measurements, 25 sensors
Velocity model: 1000.0 - 2000.0 m/s

==============================================================
Simulation: CPU
==============================================================
CPU Time: 0.1234 seconds
Travel times: 0.005000 - 0.045000 s

==============================================================
Simulation: GPU (CUDA)
==============================================================
GPU Time: 0.0456 seconds
Travel times: 0.005000 - 0.045000 s
Maximum difference (CPU vs GPU): 1.23e-12 s
✓ GPU results match CPU results

Speedup: 2.70x
✓ GPU is 2.70x faster than CPU
```

When CUDA is not available:
```
==============================================================
CUDA Availability Check
==============================================================
CUDA available: False
✗ GPU acceleration not available, will use CPU
  This is normal if:
  - pyGIMLi was not built with CUDA support
  - No NVIDIA GPU is present
  - CUDA drivers are not installed
```

### Requirements

- pyGIMLi built with CUDA support (`cmake -DUSE_CUDA=ON`)
- NVIDIA GPU with CUDA Toolkit installed (for GPU features)
- Python packages: numpy, matplotlib (optional, for plotting)

### Notes

**GPU Performance**: GPU acceleration provides the most benefit for:
- Large meshes (>10,000 nodes)
- Many measurements (>100 shots)
- Multiple forward simulations (e.g., during inversion)

For small problems, CPU may be faster due to data transfer overhead.

**Automatic Fallback**: The code works on systems without CUDA. It will:
1. Detect CUDA unavailability
2. Display an informative message
3. Fall back to CPU computation
4. Continue execution normally

## Building pyGIMLi with CUDA Support

To enable CUDA support, build pyGIMLi with the CUDA flag:

```bash
# Create build directory
mkdir build && cd build

# Configure with CUDA enabled
cmake -DUSE_CUDA=ON ..

# Build
make -j$(nproc)

# Install (optional)
make install
```

## Additional Resources

- Main CUDA documentation: [../GPU_CUDA_SUPPORT.md](../GPU_CUDA_SUPPORT.md)
- pyGIMLi documentation: https://www.pygimli.org
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit

## Troubleshooting

### "CUDA not available" despite having a GPU

1. Verify CUDA Toolkit is installed:
   ```bash
   nvcc --version
   nvidia-smi
   ```

2. Rebuild pyGIMLi with CUDA:
   ```bash
   cmake -DUSE_CUDA=ON ..
   make
   ```

3. Check for build warnings/errors in CMake output

### Performance not improved with GPU

1. **Problem size too small**: GPU overhead dominates for small problems
2. **Data transfer bottleneck**: Profile your code to identify bottlenecks
3. **GPU memory**: Ensure GPU has sufficient memory
4. **Other GPU usage**: Check if other applications are using the GPU

### Runtime CUDA errors

1. Update GPU drivers
2. Check CUDA version compatibility
3. Monitor GPU memory usage
4. Try reducing problem size

## Contributing

Found a bug or have a suggestion? Please open an issue on GitHub:
https://github.com/gimli-org/gimli/issues
