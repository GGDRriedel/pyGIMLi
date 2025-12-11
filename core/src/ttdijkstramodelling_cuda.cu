/******************************************************************************
 *   Copyright (C) 2006-2024 by the GIMLi development team                    *
 *   Carsten Rücker carsten@resistivity.net                                   *
 *   Thomas Günther thomas@resistivity.net                                    *
 *                                                                            *
 *   Licensed under the Apache License, Version 2.0 (the "License");          *
 *   you may not use this file except in compliance with the License.         *
 *   You may obtain a copy of the License at                                  *
 *                                                                            *
 *       http://www.apache.org/licenses/LICENSE-2.0                           *
 *                                                                            *
 *   Unless required by applicable law or agreed to in writing, software      *
 *   distributed under the License is distributed on an "AS IS" BASIS,        *
 *   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. *
 *   See the License for the specific language governing permissions and      *
 *   limitations under the License.                                           *
 *                                                                            *
 ******************************************************************************/

#include "ttdijkstramodelling_cuda.h"

#ifdef GIMLI_USE_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <limits>
#include <iostream>
#include <algorithm>
#include "matrix.h"
#include "vector.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " \
                      << cudaGetErrorString(err) << std::endl; \
            return false; \
        } \
    } while(0)

namespace GIMLI {
namespace cuda {

bool isCudaAvailable() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    return (error == cudaSuccess && deviceCount > 0);
}

int getCudaDeviceCount() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    return deviceCount;
}

// CUDA kernel for parallel Dijkstra initialization
__global__ void initDijkstraKernel(double* distances, bool* visited, int* parent,
                                     int numNodes, int sourceNode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numNodes) {
        distances[tid] = (tid == sourceNode) ? 0.0 : INFINITY;
        visited[tid] = false;
        parent[tid] = -1;
    }
}

// Device function for atomic min with double precision
__device__ double atomicMinDouble(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(fmin(val, __longlong_as_double(assumed))));
    } while (assumed != old);
    
    return __longlong_as_double(old);
}

// CUDA kernel for relaxation step in Dijkstra's algorithm
__global__ void relaxEdgesKernel(const int* rowPtr, const int* colIdx, 
                                  const double* values, double* distances,
                                  bool* visited, int* parent, bool* updated,
                                  int numNodes, int currentNode) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (currentNode >= numNodes) return;
    
    int start = rowPtr[currentNode];
    int end = rowPtr[currentNode + 1];
    
    if (tid >= end - start) return;
    
    int edgeIdx = start + tid;
    int neighbor = colIdx[edgeIdx];
    double edgeWeight = values[edgeIdx];
    
    if (!visited[neighbor]) {
        double newDist = distances[currentNode] + edgeWeight;
        double oldDist = atomicMinDouble(&distances[neighbor], newDist);
        
        if (oldDist > newDist) {
            parent[neighbor] = currentNode;
            *updated = true;
        }
    }
}

// CUDA kernel to find minimum distance node
__global__ void findMinDistanceKernel(const double* distances, const bool* visited,
                                       int numNodes, double* minDist, int* minNode) {
    extern __shared__ char sharedMem[];
    double* sMinDist = (double*)sharedMem;
    int* sMinNode = (int*)&sMinDist[blockDim.x];
    
    int tid = threadIdx.x;
    int globalTid = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory
    sMinDist[tid] = INFINITY;
    sMinNode[tid] = -1;
    
    if (globalTid < numNodes && !visited[globalTid]) {
        sMinDist[tid] = distances[globalTid];
        sMinNode[tid] = globalTid;
    }
    
    __syncthreads();
    
    // Reduction to find minimum
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (sMinDist[tid + stride] < sMinDist[tid]) {
                sMinDist[tid] = sMinDist[tid + stride];
                sMinNode[tid] = sMinNode[tid + stride];
            }
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        atomicMinDouble(minDist, sMinDist[0]);
        if (*minDist == sMinDist[0]) {
            *minNode = sMinNode[0];
        }
    }
}

// Host function to run Dijkstra on GPU
bool runDijkstraOnGpu(const int* d_rowPtr, const int* d_colIdx, const double* d_values,
                       int numNodes, int sourceNode, double* d_distances) {
    
    bool* d_visited;
    int* d_parent;
    bool* d_updated;
    double* d_minDist;
    int* d_minNode;
    
    CUDA_CHECK(cudaMalloc(&d_visited, numNodes * sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_parent, numNodes * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_updated, sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&d_minDist, sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_minNode, sizeof(int)));
    
    // Initialize
    int blockSize = 256;
    int numBlocks = (numNodes + blockSize - 1) / blockSize;
    initDijkstraKernel<<<numBlocks, blockSize>>>(d_distances, d_visited, d_parent,
                                                   numNodes, sourceNode);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Main Dijkstra loop
    for (int iter = 0; iter < numNodes; iter++) {
        // Find minimum distance unvisited node
        double h_minDist = INFINITY;
        int h_minNode = -1;
        
        CUDA_CHECK(cudaMemcpy(d_minDist, &h_minDist, sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_minNode, &h_minNode, sizeof(int), cudaMemcpyHostToDevice));
        
        int sharedMemSize = blockSize * (sizeof(double) + sizeof(int));
        findMinDistanceKernel<<<numBlocks, blockSize, sharedMemSize>>>(
            d_distances, d_visited, numNodes, d_minDist, d_minNode);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        
        CUDA_CHECK(cudaMemcpy(&h_minNode, d_minNode, sizeof(int), cudaMemcpyDeviceToHost));
        
        if (h_minNode == -1) break;  // All reachable nodes visited
        
        // Mark node as visited
        bool visited = true;
        CUDA_CHECK(cudaMemcpy(&d_visited[h_minNode], &visited, sizeof(bool), cudaMemcpyHostToDevice));
        
        // Relax edges
        bool h_updated = false;
        CUDA_CHECK(cudaMemcpy(d_updated, &h_updated, sizeof(bool), cudaMemcpyHostToDevice));
        
        relaxEdgesKernel<<<numBlocks, blockSize>>>(d_rowPtr, d_colIdx, d_values,
                                                     d_distances, d_visited, d_parent,
                                                     d_updated, numNodes, h_minNode);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    cudaFree(d_visited);
    cudaFree(d_parent);
    cudaFree(d_updated);
    cudaFree(d_minDist);
    cudaFree(d_minNode);
    
    return true;
}

bool computeDijkstraDistancesCuda(
    const std::vector<int>& graphRowPtr,
    const std::vector<int>& graphColIdx,
    const std::vector<double>& graphValues,
    int graphSize,
    const IndexArray& shotNodes,
    const IndexArray& recNodes,
    RMatrix& distances) {
    
    if (!isCudaAvailable()) {
        std::cerr << "CUDA is not available" << std::endl;
        return false;
    }
    
    // Allocate device memory for graph
    int* d_rowPtr;
    int* d_colIdx;
    double* d_values;
    double* d_distances;
    
    size_t rowPtrSize = graphRowPtr.size() * sizeof(int);
    size_t colIdxSize = graphColIdx.size() * sizeof(int);
    size_t valuesSize = graphValues.size() * sizeof(double);
    
    CUDA_CHECK(cudaMalloc(&d_rowPtr, rowPtrSize));
    CUDA_CHECK(cudaMalloc(&d_colIdx, colIdxSize));
    CUDA_CHECK(cudaMalloc(&d_values, valuesSize));
    CUDA_CHECK(cudaMalloc(&d_distances, graphSize * sizeof(double)));
    
    // Copy graph to device
    CUDA_CHECK(cudaMemcpy(d_rowPtr, graphRowPtr.data(), rowPtrSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colIdx, graphColIdx.data(), colIdxSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_values, graphValues.data(), valuesSize, cudaMemcpyHostToDevice));
    
    // Compute distances for each shot
    std::vector<double> h_distances(graphSize);
    
    for (size_t shotIdx = 0; shotIdx < shotNodes.size(); shotIdx++) {
        int sourceNode = shotNodes[shotIdx];
        
        if (!runDijkstraOnGpu(d_rowPtr, d_colIdx, d_values, graphSize, 
                               sourceNode, d_distances)) {
            cudaFree(d_rowPtr);
            cudaFree(d_colIdx);
            cudaFree(d_values);
            cudaFree(d_distances);
            return false;
        }
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances, 
                              graphSize * sizeof(double), cudaMemcpyDeviceToHost));
        
        // Extract distances to receivers
        for (size_t recIdx = 0; recIdx < recNodes.size(); recIdx++) {
            int targetNode = recNodes[recIdx];
            distances[shotIdx][recIdx] = h_distances[targetNode];
        }
    }
    
    // Free device memory
    cudaFree(d_rowPtr);
    cudaFree(d_colIdx);
    cudaFree(d_values);
    cudaFree(d_distances);
    
    return true;
}

bool computeDijkstraPathsCuda(
    const std::vector<int>& graphRowPtr,
    const std::vector<int>& graphColIdx,
    const std::vector<double>& graphValues,
    int graphSize,
    const IndexArray& shotNodes,
    const IndexArray& recNodes,
    std::vector<std::vector<IndexArray>>& wayMatrix) {
    
    // For now, path reconstruction is done on CPU after computing distances on GPU
    // Full GPU path reconstruction would require more complex memory management
    std::cerr << "GPU path reconstruction not yet implemented, using CPU fallback" << std::endl;
    return false;
}

} // namespace cuda
} // namespace GIMLI

#endif // GIMLI_USE_CUDA
