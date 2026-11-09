/**
 * @file test_rle.cpp
 * @brief Standalone unit tests for rle.h (no Kokkos, no simulation).
 *
 * Build and run from the repository root:
 *   clang++ -std=c++17 -O2 -Wall -Wextra -Isrc/main/fe/include \
 *           tests/test_rle.cpp -o test_rle && ./test_rle
 */

#include <rle.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what)
{
  std::printf("[%s] %s\n", ok ? " OK " : "FAIL", what);
  if (!ok) ++failures;
}

/// True if decode(encode(x)) is byte-identical to x.
template <class T>
static bool roundTrip(const std::vector<T>& x)
{
  const auto runs = rleEncode(x.data(), x.size());
  const std::vector<T> y = rleDecode(runs);
  return y.size() == x.size() &&
         (x.empty() ||
          std::memcmp(x.data(), y.data(), x.size() * sizeof(T)) == 0);
}

int main()
{
  // Reference example
  const std::vector<int> ex = {0, 0, 0, 0, 0, 7, 7, 3};
  const auto runs = rleEncode(ex.data(), ex.size());
  check(runs.size() == 3 && runs[0].value == 0 && runs[0].count == 5 &&
            runs[1].value == 7 && runs[1].count == 2 &&
            runs[2].value == 3 && runs[2].count == 1,
        "0 0 0 0 0 7 7 3 -> (0,5) (7,2) (3,1)");

  // Edge cases
  check(roundTrip(std::vector<float>{}), "empty input");
  check(roundTrip(std::vector<float>{1.5f}), "single value");
  check(roundTrip(std::vector<float>{2.f, 2.f, 2.f}), "constant input");

  // Signed zeros must not be merged
  const std::vector<float> zeros = {0.0f, -0.0f, 0.0f};
  check(rleEncode(zeros.data(), zeros.size()).size() == 3 && roundTrip(zeros),
        "0.0 and -0.0 preserved");

  // Integer codes, as produced by quantization
  check(roundTrip(std::vector<std::uint16_t>{5, 5, 9, 9, 9, 0}), "uint16_t");
  check(roundTrip(std::vector<std::uint8_t>{1, 1, 1, 200, 200}), "uint8_t");

  // Synthetic pressure field: zero cube with a non-zero ball at the centre,
  // as in the early time steps of the simulation
  const int n = 64;
  std::vector<float> field(n * n * n, 0.0f);
  for (int k = 0; k < n; ++k)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
      {
        const float r = std::sqrt(float((i - 32) * (i - 32) +
                                        (j - 32) * (j - 32) +
                                        (k - 32) * (k - 32)));
        if (r < 10) field[i + j * n + k * n * n] = std::sin(r);
      }
  const auto fieldRuns = rleEncode(field.data(), field.size());
  check(roundTrip(field), "64^3 field round trip");
  std::printf("       64^3 field: %zu values -> %zu runs, ratio = %.1f\n",
              field.size(), fieldRuns.size(),
              rleRatio(field.size(), fieldRuns));

  // Worst case: no repetition, the encoding doubles the size
  std::vector<float> noRepeat(1000);
  for (int i = 0; i < 1000; ++i) noRepeat[i] = float(i);
  const auto worst = rleEncode(noRepeat.data(), noRepeat.size());
  check(rleRatio(noRepeat.size(), worst) < 1.0, "worst case: ratio < 1");
  std::printf("       worst case: ratio = %.2f\n",
              rleRatio(noRepeat.size(), worst));

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "ALL PASSED",
              failures);
  return failures ? 1 : 0;
}