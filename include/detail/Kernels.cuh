#pragma once

#include "BinarySearch.cuh"
#include "BlockMemMove.cuh"
#include "Utils.cuh"
#include <cstdint>
#include <cub/cub.cuh>

template <std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
__global__ void MemMoveSimple(std::uint8_t* dest,
                              const std::uint8_t* src,
                              std::size_t num,
                              cub::ScanTileState<bool> scanTileState)
{
  static constexpr std::uint32_t TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  // Determine offset (global, since only a single src)
  auto offset = static_cast<std::size_t>(blockIdx.x) * TILE_ITEMS;

  // Invoke the MemMove block primitive
  BlockMemMove<BLOCK_THREADS, ITEMS_PER_THREAD>(dest, src, num, offset, scanTileState);
}

template <std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
__global__ void MemMoveComplex(std::uint8_t* dest,
                               const std::uint32_t* blockStarts,
                               const SourceTriple* srcTriples,
                               std::int32_t numSources,
                               cub::ScanTileState<bool> scanTileState)
{
  static constexpr std::uint32_t TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD;

  __shared__ std::uint32_t blockStartSmem;
  __shared__ SourceTriple srcTripleSmem;

  // Do binary search to find src index
  if (threadIdx.x == 0)
  {
    std::uint32_t idx = BinarySearch(blockStarts, blockIdx.x, 0, numSources);
    blockStartSmem    = blockStarts[idx];
    srcTripleSmem     = srcTriples[idx];
  }
  __syncthreads();

  // Load srcs, blockStarts, and writeOffsets into SMEM
  auto offset = static_cast<std::size_t>(blockIdx.x - blockStartSmem) * TILE_ITEMS;

  // Invoke the MemMove block primitive
  BlockMemMove<BLOCK_THREADS, ITEMS_PER_THREAD>(dest + srcTripleSmem.WriteOffset,
                                                srcTripleSmem.Src,
                                                srcTripleSmem.Count,
                                                offset,
                                                scanTileState);
}