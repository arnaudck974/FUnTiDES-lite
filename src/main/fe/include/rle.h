/**
 * @file rle.h
 * @brief Lossless Run-Length Encoding (RLE) for raw or quantized fields.
 *
 * Consecutive identical values are stored as (value, count) pairs:
 *   0 0 0 0 0 7 7 3  ->  (0,5) (7,2) (3,1)
 *
 * Header-only and generic: works on float pressure fields as well as on
 * integer codes produced by quantization (uint8_t, uint16_t, ...).
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <ostream>

/// A run of identical consecutive values.
template <class T>
struct Run
{
  T value;
  std::uint32_t count;
};

/**
 * @brief Bitwise equality.
 *
 * Unlike operator==, distinguishes 0.0f from -0.0f, which keeps the
 * encoding strictly lossless for floating-point data.
 */
template <class T>
inline bool sameBits(const T& a, const T& b)
{
  return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/**
 * @brief Encodes n values into a list of runs.
 *
 * Single pass, O(n). A run is split if its count would overflow uint32_t.
 */
template <class T>
std::vector<Run<T>> rleEncode(const T* data, std::size_t n)
{
  std::vector<Run<T>> runs;
  if (n == 0) return runs;

  Run<T> current{data[0], 1};
  for (std::size_t i = 1; i < n; ++i)
  {
    const bool canExtend =
        current.count < std::numeric_limits<std::uint32_t>::max();
    if (sameBits(data[i], current.value) && canExtend)
    {
      ++current.count;
    }
    else
    {
      runs.push_back(current);
      current = Run<T>{data[i], 1};
    }
  }
  runs.push_back(current);  // flush the last run
  return runs;
}

/// @brief Restores the original values from a list of runs.
template <class T>
std::vector<T> rleDecode(const std::vector<Run<T>>& runs)
{
  std::vector<T> data;
  for (const Run<T>& run : runs) data.insert(data.end(), run.count, run.value);
  return data;
}

/**
 * @brief Encoded size in bytes: one value and one 32-bit count per run.
 *
 * sizeof(Run<T>) is not used because it includes padding
 * (e.g. Run<uint8_t> is 8 bytes in memory but 5 bytes once written).
 */
template <class T>
std::size_t rleBytes(const std::vector<Run<T>>& runs)
{
  return runs.size() * (sizeof(T) + sizeof(std::uint32_t));
}

/**
 * @brief Compression ratio = original size / encoded size.
 *
 * A ratio below 1 means the encoding is larger than the input
 * (for float data, runs must average more than 2 values to pay off).
 * Returns 1 for empty input.
 */
template <class T>
double rleRatio(std::size_t n, const std::vector<Run<T>>& runs)
{
  const std::size_t compressed = rleBytes(runs);
  return compressed == 0 ? 1.0
                         : static_cast<double>(n * sizeof(T)) / compressed;
}

/**
 * @brief Writes runs in binary: run count (uint64), then value/count pairs.
 *
 * Fields are written one by one, so the file holds no padding and its size
 * matches rleBytes() plus the 8-byte header.
 */
template <class T>
void rleWrite(std::ostream& out, const std::vector<Run<T>>& runs)
{
  const std::uint64_t nRuns = runs.size();
  out.write(reinterpret_cast<const char*>(&nRuns), sizeof(nRuns));
  for (const Run<T>& run : runs)
  {
    out.write(reinterpret_cast<const char*>(&run.value), sizeof(T));
    out.write(reinterpret_cast<const char*>(&run.count), sizeof(run.count));
  }
}