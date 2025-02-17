#pragma once

#include <cub/cub.cuh>

template <typename T, typename IndexT>
__device__ __forceinline__ IndexT
BinarySearch(const T* searchData, const T& searchKey, IndexT start, IndexT end)
{
  IndexT idx;
  T currentKey;

  while (start < end)
  {
    idx        = cub::MidPoint(start, end);
    currentKey = searchData[idx];
    if (searchKey < currentKey)
    {
      end = idx;
    }
    else
    {
      start = idx + 1;
    }
  }
  return start;
}