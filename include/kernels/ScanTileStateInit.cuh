#pragma once

// Initializes state for decoupled look-back
template<typename ScanTileStateT>
__global__ void ScanTileStateInit(ScanTileStateT scanTileState, std::size_t numThreadBlocks)
{
	scanTileState.InitializeStatus(numThreadBlocks);
}