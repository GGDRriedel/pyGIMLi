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

#ifndef _GIMLI_TTDIJKSTRAMODDELING_CUDA__H
#define _GIMLI_TTDIJKSTRAMODDELING_CUDA__H

#ifdef GIMLI_USE_CUDA

#include "gimli.h"
#include <vector>

namespace GIMLI {

// Forward declarations
class RVector;
class RMatrix;
typedef std::vector<size_t> IndexArray;

namespace cuda {

/*! Check if CUDA is available at runtime */
bool isCudaAvailable();

/*! Get CUDA device count */
int getCudaDeviceCount();

/*! CUDA-accelerated Dijkstra shortest path computation
 * 
 * Computes shortest paths from multiple source nodes to multiple target nodes
 * using GPU parallelization.
 * 
 * @param graph Adjacency information as arrays
 * @param graphSize Number of nodes in the graph
 * @param shotNodes Array of source node indices
 * @param nShots Number of source nodes
 * @param recNodes Array of target node indices  
 * @param nRecs Number of target nodes
 * @param distances Output matrix of distances [nShots x nRecs]
 * @return true if successful, false if CUDA execution failed
 */
bool computeDijkstraDistancesCuda(
    const std::vector<int>& graphRowPtr,
    const std::vector<int>& graphColIdx,
    const std::vector<double>& graphValues,
    int graphSize,
    const IndexArray& shotNodes,
    const IndexArray& recNodes,
    RMatrix& distances
);

/*! CUDA-accelerated shortest path computation
 * 
 * Computes shortest paths and returns the actual path as node sequences.
 * 
 * @param graph Adjacency information as arrays
 * @param graphSize Number of nodes in the graph
 * @param shotNodes Array of source node indices
 * @param recNodes Array of target node indices
 * @param wayMatrix Output matrix of paths [nShots x nRecs], each containing node sequences
 * @return true if successful, false if CUDA execution failed
 */
bool computeDijkstraPathsCuda(
    const std::vector<int>& graphRowPtr,
    const std::vector<int>& graphColIdx,
    const std::vector<double>& graphValues,
    int graphSize,
    const IndexArray& shotNodes,
    const IndexArray& recNodes,
    std::vector<std::vector<IndexArray>>& wayMatrix
);

} // namespace cuda

} // namespace GIMLI

#endif // GIMLI_USE_CUDA

#endif // _GIMLI_TTDIJKSTRAMODDELING_CUDA__H
