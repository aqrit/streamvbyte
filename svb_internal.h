/** @file svb_internal.h StreamVByte helpers -
 *  filter transforms, bounds calculations, and unaligned access.
 */

#ifndef INCLUDE_SVB_INTERNAL_H_
#define INCLUDE_SVB_INTERNAL_H_

/**
 * @defgroup Delta_Transpose
 *
 * --------------------------
 * 00 01 02 03 -> 00 10 20 30
 * 04 05 06 07    01 11 21 31
 * 08 09 0A 0B    02 12 22 32
 * 0C 0D 0E 0F    03 13 23 33
 * 10 11 12 13 -> 04 14 24 34
 * 14 15 16 17    05 15 25 35
 * 18 19 1A 1B    06 16 26 36
 * 1C 1D 1E 1F    07 17 27 37
 * 20 21 22 23 -> 08 18 28 38
 * 24 25 26 27    09 19 29 39
 * 28 29 2A 2B    0A 1A 2A 3A
 * 2C 2D 2E 2F    0B 1B 2B 3B
 * 30 31 32 33 -> 0C 1C 2C 3C
 * 34 35 36 37    0D 1D 2D 3D
 * 38 39 3A 3B    0E 1E 2E 3E
 * 3C 3D 3E 3F    0F 1F 2F 3F
 */

#include <stddef.h>  // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t
#include <string.h> // memcpy


/**
 * Get size of the key block.
 * The key block is at the beginning of the stream and immediately
 * precedes the data block.
 *
 * Four 2-bit keys per byte, rounded up to byte boundry.
 *
 * @param [in] count number of elements in a stream
 * @return size in bytes
 */
static inline
size_t _svb_key_block_len (size_t count) {
	return (count + 3) >> 2;
}


static inline
uint32_t _zigzag_encode_32 (uint32_t val) {
	return (val + val) ^ ((int32_t)val >> 31);
}


static inline
uint32_t _zigzag_decode_32 (uint32_t val) {
	return (val >> 1) ^ -(val & 1);
}


static inline
uint32_t _delta_encode_32 (uint32_t val, uint32_t previous) {
	return val - previous;
}


static inline
uint32_t _delta_decode_32 (uint32_t val, uint32_t previous) {
	return val + previous;
}


static inline
uint32_t _delta_zigzag_encode_32 (uint32_t val, uint32_t previous) {
	return _zigzag_encode_32(_delta_encode_32(val, previous));
}


static inline
uint32_t _delta_zigzag_decode_32 (uint32_t val, uint32_t previous) {
	return _delta_decode_32(_zigzag_decode_32(val), previous);
}


#if defined(__SSE4_1__) || defined(__AVX__)
# include <smmintrin.h> // SSE4.1 intrinsics
# include <string.h> // memcpy

static inline
__m128i _mm_zigzag_encode_epi32 (__m128i v) {
	__m128i signmask =_mm_cmpgt_epi32(_mm_setzero_si128(), v);
	return _mm_xor_si128(_mm_add_epi32(v, v), signmask);
}


static inline
__m128i _mm_zigzag_decode_epi32 (__m128i v) {
	const __m128i m = _mm_set1_epi32(1);
	__m128i signmask =_mm_cmpeq_epi32(_mm_and_si128(m, v), m);
	return _mm_xor_si128(_mm_srli_epi32(v, 1), signmask);
}


/**
 * Difference each 32-bit subword with the preceding subword.
 * @param [in] v `[A, B, C, D]` (A in low order subword)
 * @param [in] prev `[?, ?, ?, P]` (P in high order subword)
 * @return deltas `[A-P, B-A, C-B, D-C]`
 */
static inline
__m128i _mm_delta_encode_epi32 (__m128i v, __m128i prev) {
	return _mm_sub_epi32(v, _mm_alignr_epi8(v, prev, 12));
}


/**
 * Sums the delta values to recover the orginal values.
 * @param [in] v `[A, B, C, D]` (A in low order subword)
 * @param [in] prev `[?, ?, ?, P]` (P in high order subword)
 * @return sums `[P+A, P+A+B, P+A+B+C, P+A+B+C+D]`
 */
static inline
__m128i _mm_delta_decode_epi32 (__m128i v, __m128i prev) {
	prev = _mm_shuffle_epi32(prev, _MM_SHUFFLE(3,3,3,3)); // [P P P P]
	v = _mm_add_epi32(v, _mm_slli_si128(v, 4)); // [A AB BC CD]
	prev = _mm_add_epi32(prev, v); // [PA PAB PBC PCD]
	v = _mm_slli_si128(v, 8); // [0 0 A AB]
	return _mm_add_epi32(prev, v); // [PA PAB PABC PABCD]
}


static inline
__m128i _mm_delta_zigzag_encode_epi32 (__m128i v, __m128i prev) {
	return _mm_zigzag_encode_epi32(_mm_delta_encode_epi32(v, prev));
}


static inline
__m128i _mm_delta_zigzag_decode_epi32 (__m128i v, __m128i prev) {
	return _mm_delta_decode_epi32(_mm_zigzag_decode_epi32(v), prev);
}


/**
 * transpose a 4x4 matrix of 32-bit subwords
 * (_MM_TRANSPOSE4_PS for integers)
 * ~~~
 * w0 x0 y0 z0       w0 w1 w2 w3
 * w1 x1 y1 z1  <->  x0 x1 x2 x3
 * w2 x2 y2 z2       y0 y1 y2 y3
 * w3 x3 y3 z3       z0 z1 z2 z3
 * ~~~
 */
#define SVB_TRANSPOSE(row0, row1, row2, row3) \
do { \
	__m128i t0 = _mm_unpacklo_epi32(row0, row1); \
	__m128i t1 = _mm_unpacklo_epi32(row2, row3); \
	__m128i t2 = _mm_unpackhi_epi32(row0, row1); \
	__m128i t3 = _mm_unpackhi_epi32(row2, row3); \
	row0 = _mm_unpacklo_epi64(t0, t1); \
	row1 = _mm_unpackhi_epi64(t0, t1); \
	row2 = _mm_unpacklo_epi64(t2, t3); \
	row3 = _mm_unpackhi_epi64(t2, t3); \
} while (0)


/**
 * @defgroup Unaligned
 * Memory Access Helpers
 *
 * C has no support for dereferencing misaligned pointers. Though, there
 * are a few magic intrinsics for working with unaligned xmmwords. For
 * everything else, unaligned word access is done with memcpy -- which
 * should be safe. It is performance critical that these memcpy
 * operations optimize to a single load or store instruction. If your
 * compiler fails to optimize these you may need to replace the memcpy
 * operations with something unsafe like `(*((aligned_word_type*)ptr))`.
 */

/// @ingroup Unaligned
/// Store 16-bit word to an unaligned memory location
static inline
void svb_storeu_u16 (void *dst, uint16_t src) {
	memcpy(dst, &src, sizeof(src));
}

/// @ingroup Unaligned
/// Store 32-bit word to an unaligned memory location
static inline
void svb_storeu_u32 (void *dst, uint32_t src) {
	memcpy(dst, &src, sizeof(src));
}

/// @ingroup Unaligned
/// Load 16-bit word from an unaligned memory location
static inline
uint16_t svb_loadu_u16 (const void *ptr) {
	uint16_t data;
	memcpy(&data, ptr, sizeof(data));
	return data;
}

/// @ingroup Unaligned
/// Load 32-bit word from an unaligned memory location
static inline
uint32_t svb_loadu_u32 (const void *ptr) {
	uint32_t data;
	memcpy(&data, ptr, sizeof(data));
	return data;
}

/// @ingroup Unaligned
/// Load 64-bit word from an unaligned memory location
static inline
uint64_t svb_loadu_u64 (const void *ptr) {
	uint64_t data;
	memcpy(&data, ptr, sizeof(data));
	return data;
}

#endif // defined(__SSE4_1__) || defined(__AVX__)

#endif /* INCLUDE_SVB_INTERNAL_H_ */
