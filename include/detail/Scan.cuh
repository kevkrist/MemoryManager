#pragma once

// The scan op for MemMove with decoupled look-back
struct MemMoveScanOp
{
  __device__ __forceinline__ bool operator()(bool l, bool r) const
  {
    return l && r;
  }
};

// Initializes state for decoupled look-back
template<typename ScanTileStateT>
__global__ void ScanTileStateInit(ScanTileStateT scanTileState, std::size_t numThreadBlocks)
{
	scanTileState.InitializeStatus(numThreadBlocks);
}