#pragma once

#include <cstdint>

#define RETURN_ERROR(e)                                                                            \
  Error = (e);                                                                                     \
  if (Error != cudaSuccess)                                                                        \
  return Error
#define RETURN_ERROR_ALLOCATE(e)                                                                   \
  if (!(e))                                                                                        \
  return cudaErrorMemoryAllocation

struct SourcePair
{
  const std::uint8_t* Src;
  std::size_t Count;
};

struct SourceTriple
{
  const std::uint8_t* Src;
  std::size_t Count;
  std::size_t WriteOffset;
};