#pragma once

#include "kernels/ScanTileStateInit.cuh"
#include "kernels/MemMoveSimple.cuh"
#include <cstdint>
#include <cub/cub.cuh>
#include <thrust/host_vector.h>
#include <thrust/scan.h>
#include <thrust/tuple.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <memory>

struct SourcePair
{
	const std::uint8_t* Src;
	std::size_t Count;
};

template<std::size_t Alignment, typename ByteAllocatorT>
class MemMove
{
	using scan_tile_state_t = cub::ScanTileState<bool>;
	using byte_allocator_t = ByteAllocatorT;
	using chunk_t = std::uint32_t;

	static constexpr std::int32_t alignment = Alignment;

	// Constructor
	explicit __host__ MemMove(std::unique_ptr<byte_allocator_t> byteAllocator) : ByteAllocator{
			std::move(byteAllocator) }
	{
	}

	// Destructor
	~MemMove()
	{
		if (CopyStream)
		{
			auto error = CubDebug(cudaStreamDestroy(CopyStream));
			if (error != cudaSuccess)
			{
				std::cerr << "Error: MemMove destructor failed to destroy CopyStream. " << cudaGetErrorString(error)
						  << "\n";
			}
			CopyStream = nullptr;
		}
		CleanUp();
	}

	template<std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
	__host__ cudaError_t Move(std::uint8_t* dest, const std::uint8_t* src, std::size_t count)
	{
		// If there is no memory overlap, default to cudaMemcpy
		if (dest + count < src)
		{
			return CubDebug(cudaMemcpy(dest, src, count, cudaMemcpyDeviceToDevice));
		}

		// We need the memmove kernel. Prepare decoupled look-back.
		NumTotalThreadBlocks = GetNumThreadBlocks<BLOCK_THREADS>(count);
		auto error = InitScanTileState<BLOCK_THREADS>();
		if (error != cudaSuccess) return error;

		// Synchronize stream
		error = CubDebug(cudaStreamSynchronize(InitStream));
		if(error != cudaSuccess) return error;

		// Launch kernel
	}

	template<std::uint32_t BLOCK_THREADS, std::uint32_t ITEMS_PER_THREAD>
	__host__ cudaError_t Move(std::uint8_t* dest, const thrust::host_vector<SourcePair>& srcPairs)
	{
		ScanSrcPairs<BLOCK_THREADS>(srcPairs);
		auto error = CopyToDevice<BLOCK_THREADS>();
		if (error != cudaSuccess) return error;
		error = InitScanTileState<BLOCK_THREADS>();
		if (error != cudaSuccess) return error;

		// Synchronize streams
		error = CubDebug(cudaStreamSynchronize(CopyStream));
		if(error != cudaSuccess) return error;
		error = CubDebug(cudaStreamSynchronize(InitStream));
		if(error != cudaSuccess) return error;

		// Launch kernel
	}

	__host__ std::unique_ptr<byte_allocator_t> ReleaseAllocator()
	{
		if (!ByteAllocator)
		{
			std::cerr << "Error: No allocator. Release failed.";
			return nullptr;
		}

		// Clean up before returning the allocator
		CleanUp();
		return std::move(ByteAllocator);
	}

	__host__ void CleanUp()
	{
		// Erase allocated memory, if necessary
		if (ByteAllocator)
		{
			ByteAllocator->deallocate(BytesAllocated);
			BytesAllocated = 0;
		}

		// Reset device pointers
		WriteOffsetsDevice = nullptr;
		BlockStartsDevice = nullptr;
		SrcsDevice = nullptr;
		ScanTileStateStorage = nullptr;
	}

private:
	thrust::host_vector<std::size_t> WriteOffsetsHost{};
	thrust::host_vector<std::size_t> BlockStartsHost{};
	thrust::host_vector<const std::uint8_t*> SrcsHost{};
	std::size_t* WriteOffsetsDevice = nullptr;
	std::size_t* BlockStartsDevice = nullptr;
	std::uint8_t** SrcsDevice = nullptr;
	std::size_t NumTotalThreadBlocks = 0;
	std::size_t NumSources = 0;
	std::unique_ptr<byte_allocator_t> ByteAllocator;
	std::size_t BytesAllocated = 0;
	cudaStream_t CopyStream{};
	scan_tile_state_t ScanTileState{};
	std::uint8_t* ScanTileStateStorage = nullptr;
	cudaStream_t InitStream{};

	template<std::uint32_t BLOCK_THREADS>
	static __host__ __forceinline__ std::size_t GetNumThreadBlocks(std::size_t count)
	{
		return cuda::ceil_div(count / sizeof(chunk_t), BLOCK_THREADS);
	}

	static __host__ __forceinline__ std::size_t GetAlignedCount(std::size_t count)
	{
		return cuda::ceil_div(count, Alignment) * Alignment;
	}

	template<typename T>
	__host__ __forceinline__ T* Allocate(std::size_t num)
	{
		if (!ByteAllocator)
		{
			std::cerr << "Error: No allocator available. Allocation failed.\n";
			return nullptr;
		}

		std::uint8_t* placeholder = ByteAllocator->allocate(sizeof(T) * num);
		if (!placeholder) return nullptr;

		BytesAllocated += sizeof(T) * num;
		return reinterpret_cast<T*>(placeholder);
	}

	template<std::uint32_t BLOCK_THREADS>
	__host__ __forceinline__ void ScanSrcPairs(const thrust::host_vector<SourcePair>& srcPairs)
	{
		using tuple_t = thrust::tuple<std::size_t, std::size_t>; // <aligned count, num thread blocks>

		// Define the operators
		auto transformOp = [](const SourcePair& srcPair) -> tuple_t
		{
			return { GetAlignedCount(srcPair.Count), GetNumThreadBlocks<BLOCK_THREADS>(srcPair.Count), srcPair.Src };
		};
		auto scanOp = [](const tuple_t& l, const tuple_t& r) -> tuple_t
		{
			return { thrust::get<0>(l) + thrust::get<0>(r), thrust::get<1>(l) + thrust::get<1>(r) };
		};
		auto sliceOp = [](const SourcePair& srcPair) -> const std::uint8_t*
		{
			return srcPair.Src;
		};

		// Execute the scan
		auto inputIterator = thrust::make_transform_iterator(srcPairs.begin(), transformOp);
		auto outputIterator = thrust::make_zip_iterator(WriteOffsetsHost.begin(), BlockStartsHost.begin());
		thrust::exclusive_scan(inputIterator, inputIterator + NumSources, outputIterator, { 0, 0 }, scanOp);
		NumTotalThreadBlocks = BlockStartsHost.back() + GetNumThreadBlocks<BLOCK_THREADS>(srcPairs.back().Count);

		// Extract the sources
		thrust::transform(srcPairs.begin(), srcPairs.end(), SrcsHost.begin(), sliceOp);
	}

	template<std::uint32_t BLOCK_THREADS>
	__host__ __forceinline__ cudaError_t
	CopyToDevice()
	{
		// Initialize the copy stream
		auto error = CubDebug(cudaStreamCreate(&CopyStream));
		if (error != cudaSuccess) return error;

		// Determine the number of required bytes
		std::size_t requiredBytes = sizeof(std::size_t) * NumSources;

		// Write offsets
		WriteOffsetsDevice = Allocate<std::size_t>(NumSources);
		if (!WriteOffsetsDevice) return cudaErrorMemoryAllocation;
		error = CubDebug(
				cudaMemcpyAsync(WriteOffsetsDevice, thrust::raw_pointer_cast(WriteOffsetsHost.data()), requiredBytes,
						cudaMemcpyHostToDevice, CopyStream));
		if (error != cudaSuccess) return error;

		// Block starts
		BlockStartsDevice = Allocate<std::size_t>(NumSources);
		if (!BlockStartsDevice) return cudaErrorMemoryAllocation;
		error = CubDebug(
				cudaMemcpyAsync(BlockStartsDevice, thrust::raw_pointer_cast(BlockStartsHost.data()), requiredBytes,
						cudaMemcpyHostToDevice, CopyStream));
		if (error != cudaSuccess) return error;

		// Source pointers
		requiredBytes = sizeof(std::uint8_t*) * NumSources;
		SrcsDevice = Allocate<std::uint8_t*>(NumSources);
		if (!SrcsDevice) return cudaErrorMemoryAllocation;
		return CubDebug(cudaMemcpyAsync(SrcsDevice, thrust::raw_pointer_cast(SrcsHost.data()), requiredBytes,
				cudaMemcpyHostToDevice,
				CopyStream));
	}

	template<std::uint32_t BLOCK_THREADS>
	__host__ __forceinline__ cudaError_t InitScanTileState()
	{
		// Determine temporary storage requirements and allocate
		std::size_t scanTileStateStorageBytes = 0;
		scan_tile_state_t::AllocationSize(NumTotalThreadBlocks, scanTileStateStorageBytes);
		ScanTileStateStorage = Allocate(scanTileStateStorageBytes);
		if (!ScanTileStateStorage) return cudaErrorMemoryAllocation;

		// Initialize the temporary storage
		auto error = CubDebug(
				ScanTileState.Init(NumTotalThreadBlocks, ScanTileStateStorage, scanTileStateStorageBytes));
		if (error != cudaSuccess) return error;

		// Initialize the init stream
		error = CubDebug(cudaStreamCreate(&InitStream));
		if (error != cudaSuccess) return error;

		// Invoke the initialization kernel
		ScanTileStateInit<<<cuda::ceil_div(NumTotalThreadBlocks, BLOCK_THREADS), BLOCK_THREADS, 0, InitStream>>>(
				ScanTileState, NumTotalThreadBlocks
		);

		return CubDebug(cudaGetLastError());
	}
};
