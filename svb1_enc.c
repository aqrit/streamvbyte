/**
 * @file svb1_enc.c
 * StreamVByte routines for encoding "1234 format" streams.
 *
 * ~~~
 * svb1 2-bit key values:
 * 0b00 == 1 byte  stored, 3 leading zero bytes stripped.
 * 0b01 == 2 bytes stored, 2 leading zero bytes stripped.
 * 0b10 == 3 bytes stored, 1 leading zero byte  stripped.
 * 0b11 == 4 bytes stored, 0 leading zero bytes stripped.
 * ~~~
 *
 * generic routines assume:
 * unknown endian, no unaligned memory access, two's complement.
 *
 * x64 routines assume:
 * SSE4.1, 64-bit mode, fast pshufb, fast unaligned memory access.
 *
 */

#include <stddef.h> // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#include "svb.h"
#include "svb_internal.h"

#if defined(__SSE4_1__) || defined(__AVX__)

#include <smmintrin.h> // SSE4.1 intrinsics


/// Lists how many bytes are retained from a 16-byte chunk.
/// `size_t length = svb1_len_table[key_byte];`
static const uint8_t svb1_len_table[256] = {
	 4,  5,  6,  7,  5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10,
	 5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10,  8,  9, 10, 11,
	 6,  7,  8,  9,  7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12,
	 7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13,
	 5,  6,  7,  8,  6,  7,  8,  9,  7,  8,  9, 10,  8,  9, 10, 11,
	 6,  7,  8,  9,  7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12,
	 7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13,
	 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14,
	 6,  7,  8,  9,  7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12,
	 7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13,
	 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14,
	 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15,
	 7,  8,  9, 10,  8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13,
	 8,  9, 10, 11,  9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14,
	 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15,
	10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15, 13, 14, 15, 16
};


/// Encoder permutation table.
/// `__m128* shuf = (__m128i*)&svb1_enc_table[(key_byte*16) & 0x03F0];`
static const uint8_t svb1_enc_table[64*16] = {
	 0,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  4,  5,  6,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  8, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,
	 0,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  4,  5,  6,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  8,  9, 12, 13, 14, 15,  0,  0,  0,
	 0,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,
	 0,  4,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  4,  5,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,
	 0,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 12, 13, 14, 15,  0,  0,
	 0,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 13, 14, 15,  0,
	 0,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,
	 0,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  2,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  8,  9, 10, 11, 12, 13, 14, 15,  0,
	 0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,
	 0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
};


/**
 * Compress two xmmwords into a "1234 format" stream. Unroll helper.
 * @param [in,out] key_ptr current position in the stream's key block.
 * @param [in,out] data_ptr current position in the stream's data block.
 * @param [in] src_a, src_b xmmwords to encode.
 */
#define SVB1_ENCODE8(key_ptr, data_ptr, src_a, src_b) \
do { \
	const __m128i mask_01 = _mm_set1_epi8(0x01); \
	const __m128i mask_7F00 = _mm_set1_epi16(0x7F00); \
	\
	/* normalize each byte to 0 or 1 */ \
	__m128i a = _mm_min_epu8(mask_01, src_a); \
	__m128i b = _mm_min_epu8(mask_01, src_b); \
	\
	/* generate keys */ \
	a = _mm_packus_epi16(a, b); \
	a = _mm_min_epi16(a, mask_01); /* convert 0x01FF to 0x0101 */ \
	a = _mm_adds_epu16(a, mask_7F00);/*0x0101->0x8001, 0xFF01->0xFFFF*/\
	size_t keys = (size_t)_mm_movemask_epi8(a); \
	\
	/* In-register shuffle to transfer non-leading-zero bytes to \
	 * contiguous low order byte positions. To reduce table size, \
	 * the 2-bit key of the highest 32-bit subword of an xmmword is \
	 * ignored. (unwanted high bytes are overwritten, later) */ \
	__m128i *shuf_a = (__m128i*)&svb1_enc_table[(keys << 4) & 0x03F0]; \
	__m128i *shuf_b = (__m128i*)&svb1_enc_table[(keys >> 4) & 0x03F0]; \
	__m128i data_a = _mm_shuffle_epi8(src_a, _mm_loadu_si128(shuf_a)); \
	__m128i data_b = _mm_shuffle_epi8(src_b, _mm_loadu_si128(shuf_b)); \
	\
	/* Store an entire xmmword then advance the pointer by how ever \
	 * many of those bytes are actually wanted. Any trailing \
	 * garbage will be overwritten by the next store. \
	 * note: little endian byte memory order */ \
	_mm_storeu_si128((__m128i *)data_ptr, data_a); \
	data_ptr += svb1_len_table[keys & 0xFF]; \
	_mm_storeu_si128((__m128i *)data_ptr, data_b); \
	data_ptr += svb1_len_table[keys >> 8]; \
	\
	/* save keys so the stream can be decoded at some point... */ \
	svb_storeu_u16(key_ptr, (uint16_t)keys); \
	key_ptr += 2; \
} while (0)


/**
 * @param [in] dst current position in the stream's data block.
 * @param [in] dw value to encode
 * @return 2-bit code
 *
 * @todo BitScanReverse() would be a win, ONLY if natively available.
 * _lzcnt_u32() is probably faster than bsr on AMD.
 * `key = bsr(val | 1) >> 3;`
 */
static inline
uint32_t _svb1_encode1_x64 (uint8_t* dst, uint32_t dw)
{
	/* copy input directly into the output stream. Unwanted
	 * bytes (zeros) will be overwritten by next copy.
	 * note: little endian byte memory order */
	svb_storeu_u32(dst, dw);

	// Detect leading zero bytes and map that to a key code
	dw >>= 8;
	uint32_t a = dw + 0x00FFFF00;
	uint32_t b = dw & 0x00FF0000;
	a |= 0x00FFFFFF;
	dw += a;
	dw |= 0x00FFFFFF;
	dw += b;
	dw >>= 24;

	return dw;
}


/**
 * StreamVByte compress an array of 32-bit integers to a "1234 format"
 * stream. Idealy, array elements should be unordered and have small
 * non-zero values.
 *
 * @param [in] in array of elements to be compressed.
 *
 * @param [in] count number of array elements to be compressed.
 * The caller is responsible for recording `count` out-of-band, it is
 * not saved as part of the stream.
 *
 * @param [in] out buffer into which the compressed stream will be
 * written. The buffer must be at least `svb_compress_bound(count)`
 * bytes in size.
 *
 * @return pointer to just past-the-end of the stream.
 */
uint8_t* svb1_enc_x64 (const uint32_t* in, size_t count, uint8_t* out)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) { // use chunking
		const uint32_t* end = &in[(count & ~7)];
		do {
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);
			in += 8;
			SVB1_ENCODE8(key_ptr, data_ptr, src_a, src_b);
		} while (in < end);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t code = _svb1_encode1_x64(data_ptr, in[i]);
			data_ptr += 1 + code;
			keys |= code << i*2;
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


uint8_t* svb1z_enc_x64 (const uint32_t* in, size_t count, uint8_t* out)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) { // use chunking
		const uint32_t* end = &in[(count & ~7)];
		do {
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);
			in += 8;
			src_a = _mm_zigzag_encode_epi32(src_a);
			src_b = _mm_zigzag_encode_epi32(src_b);
			SVB1_ENCODE8(key_ptr, data_ptr, src_a, src_b);
		} while (in < end);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t dw = _zigzag_encode_32(in[i]);
			uint32_t code = _svb1_encode1_x64(data_ptr, dw);
			data_ptr += 1 + code;
			keys |= code << i*2;
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


/**
 * Delta & StreamVByte encode a unsigned 32-bit array to a
 * "1234 format" stream. Ideally, array values should occur in
 * ascending order, with few repeats.
 *
 * @param [in] in array of 32-bit elements to be compressed.
 *
 * @param [in] count number of array elements to be compressed.
 * The caller is responsible for recording `count` out-of-band, it is
 * not saved as part of the stream.
 *
 * @param [in] out buffer into which the compressed stream will be
 * written. The buffer must be at least `svb_compress_bound(count)`
 * bytes in size.
 *
 * @param [in] previous value to diff against the first array element.
 * This value is not worth storing in the stream, it could save at most
 * 3 bytes. Recommended value is zero.
 *
 * @return pointer to just past-the-end of the stream.
 */
uint8_t* svb1d_enc_x64 (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) { // use chunking
		__m128i prev = _mm_undefined_si128();
		prev = _mm_insert_epi32(prev, previous, 3);
		const uint32_t* end = &in[(count & ~7)];
		do {
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);
			in += 8;

			// delta transform
			__m128i diff_a = _mm_delta_encode_epi32(src_a, prev);
			__m128i diff_b = _mm_delta_encode_epi32(src_b, src_a);
			prev = src_b;

			SVB1_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
		} while (in < end);
		previous = _mm_extract_epi32(prev, 3);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint32_t code = _svb1_encode1_x64(data_ptr, delta);
			data_ptr += 1 + code;
			keys |= code << i*2;
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


uint8_t* svb1dz_enc_x64 (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) { // use chunking
		__m128i prev = _mm_undefined_si128();
		prev = _mm_insert_epi32(prev, previous, 3);
		const uint32_t* end = &in[(count & ~7)];
		do {
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);
			in += 8;

			__m128i diff_a = _mm_delta_zigzag_encode_epi32(src_a, prev);
			__m128i diff_b = _mm_delta_zigzag_encode_epi32(src_b,src_a);
			prev = src_b;

			SVB1_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
		} while (in < end);
		previous = _mm_extract_epi32(prev, 3);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_zigzag_encode_32(in[i], previous);
			previous = in[i];
			uint32_t code = _svb1_encode1_x64(data_ptr, delta);
			data_ptr += 1 + code;
			keys |= code << i*2;
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


/**
 * Delta & Transpose & StreamVByte encode an array to a "1234 format"
 * stream. Ideally, array values should occur in ascending order,
 * with few repeats.
 *
 * @param [in] in array of elements to be compressed.
 *
 * @param [in] count number of array elements to be compressed.
 * The caller is responsible for recording `count` out-of-band, it is
 * not saved as part of the stream.
 *
 * @param [in] out buffer into which the compressed stream will be
 * written. The buffer must be at least `svb_compress_bound(count)`
 * bytes in size.
 *
 * @param [in] previous value to diff against the first array element.
 * This value is not worth storing in the stream, it could save at most
 * 3 bytes. Recommended value is zero.
 *
 * @return pointer to just past-the-end of the stream.
 */
uint8_t* svb1dt_enc_x64 (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	uint32_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) {
		__m128i prev = _mm_undefined_si128();
		prev = _mm_insert_epi32(prev, previous, 3);

		// for chunks of 64 elements
		// delta transpose and encode
		for (const uint32_t* end = &in[(count & ~63)];
			in != end; in += 64)
		{
			__m128i r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,rA,rB,rC,rD,rE,rF;
			__m128i temp;

			// Note: The bottom row is needed for the delta transform of
			// the first row.  It loaded out of sequence so we can start
			// encoding & storing and thereby relieve register pressure.
			r3 = _mm_loadu_si128((__m128i*)&in[12]);
			r7 = _mm_loadu_si128((__m128i*)&in[28]);
			rB = _mm_loadu_si128((__m128i*)&in[44]);
			rF = _mm_loadu_si128((__m128i*)&in[60]);
			SVB_TRANSPOSE(r3,r7,rB,rF);
			prev = _mm_alignr_epi8(rF, prev, 12);

			r0 = _mm_loadu_si128((__m128i*)&in[0]);
			r4 = _mm_loadu_si128((__m128i*)&in[16]);
			r8 = _mm_loadu_si128((__m128i*)&in[32]);
			rC = _mm_loadu_si128((__m128i*)&in[48]);
			SVB_TRANSPOSE(r0,r4,r8,rC);
			temp = rC;
			rC = _mm_sub_epi32(rC, r8);
			r8 = _mm_sub_epi32(r8, r4);
			r4 = _mm_sub_epi32(r4, r0);
			r0 = _mm_sub_epi32(r0, prev);
			SVB1_ENCODE8(key_ptr, data_ptr, r0, r4);
			SVB1_ENCODE8(key_ptr, data_ptr, r8, rC);

			r1 = _mm_loadu_si128((__m128i*)&in[4]);
			r5 = _mm_loadu_si128((__m128i*)&in[20]);
			r9 = _mm_loadu_si128((__m128i*)&in[36]);
			rD = _mm_loadu_si128((__m128i*)&in[52]);
			SVB_TRANSPOSE(r1,r5,r9,rD);
			prev = rD;
			rD = _mm_sub_epi32(rD, r9);
			r9 = _mm_sub_epi32(r9, r5);
			r5 = _mm_sub_epi32(r5, r1);
			r1 = _mm_sub_epi32(r1, temp);
			SVB1_ENCODE8(key_ptr, data_ptr, r1, r5);
			SVB1_ENCODE8(key_ptr, data_ptr, r9, rD);

			r2 = _mm_loadu_si128((__m128i*)&in[8]);
			r6 = _mm_loadu_si128((__m128i*)&in[24]);
			rA = _mm_loadu_si128((__m128i*)&in[40]);
			rE = _mm_loadu_si128((__m128i*)&in[56]);
			SVB_TRANSPOSE(r2,r6,rA,rE);
			temp = rE;
			rE = _mm_sub_epi32(rE, rA);
			rA = _mm_sub_epi32(rA, r6);
			r6 = _mm_sub_epi32(r6, r2);
			r2 = _mm_sub_epi32(r2, prev);
			SVB1_ENCODE8(key_ptr, data_ptr, r2, r6);
			SVB1_ENCODE8(key_ptr, data_ptr, rA, rE);

			// finish the last four rows (already loaded and transposed)
			prev = rF;
			rF = _mm_sub_epi32(rF, rB);
			rB = _mm_sub_epi32(rB, r7);
			r7 = _mm_sub_epi32(r7, r3);
			r3 = _mm_sub_epi32(r3, temp);
			SVB1_ENCODE8(key_ptr, data_ptr, r3, r7);
			SVB1_ENCODE8(key_ptr, data_ptr, rB, rF);
		}


		// for any remaining chunks of 8 elements
		// delta transform (no transpose) and encode.
		for (const uint32_t* end = &in[(count & (63 ^ 7))];
			in != end; in += 8)
		{
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);

			// delta transform
			__m128i diff_a = _mm_delta_encode_epi32(src_a, prev);
			__m128i diff_b = _mm_delta_encode_epi32(src_b, src_a);
			prev = src_b;

			SVB1_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
		}
		previous = _mm_extract_epi32(prev, 3);
	}
	count &= 7; // all chunks of 8 are done

	// scalar tail loop
	// delta transform (no transpose) and encode
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint32_t code = _svb1_encode1_x64(data_ptr, delta);
			data_ptr += 1 + code;
			keys |= code << i*2;
		} while(++i < count);
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


#else // not __SSE4_1__


static inline
uint8_t _svb1_encode1_generic (uint8_t* dst, uint32_t dw) {
	uint8_t code = (dw > 0xFF);
	dst[0] = (uint8_t)dw;
	dst[1] = (uint8_t)(dw >> 8);
	dw >>= 16;
	if (dw != 0) { // hopefully worthwhile to banch here...
		code += 1 + (dw > 0xFF);
		dst[2] = (uint8_t)dw;
		dst[3] = (uint8_t)(dw >> 8);
	}
	return code;
}


uint8_t* svb1_enc_generic (const uint32_t* in, size_t count,
	uint8_t* out)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			uint8_t code = _svb1_encode1_generic(data_ptr, in[i]);
			data_ptr += 1 + code;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys; // flush buffer
	}
	return data_ptr;
}


uint8_t* svb1z_enc_generic (const uint32_t* in, size_t count,
	uint8_t* out)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			uint32_t dw = _zigzag_encode_32(in[i]);
			uint8_t code = _svb1_encode1_generic(data_ptr, dw);
			data_ptr += 1 + code;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys; // flush buffer
	}
	return data_ptr;
}


uint8_t* svb1d_enc_generic (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb1_encode1_generic(data_ptr, delta);
			data_ptr += 1 + code;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys; // flush buffer
	}
	return data_ptr;
}


uint8_t* svb1dz_enc_generic (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_zigzag_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb1_encode1_generic(data_ptr, delta);
			data_ptr += 1 + code;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys; // flush buffer
	}
	return data_ptr;
}


uint8_t* svb1dt_enc_generic (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	// delta transpose chunks of 64 elements
	if ((count & ~63) != 0) {
		const uint32_t* end = &in[(count & ~63)];
		do {
			// 1st pass
			uint32_t tile[64]; // temp buffer
			size_t j = 0;
			for (size_t j = 0; j < 4; j++) {
				for (size_t i = 0; i < 16; i++) {
					uint32_t dw = in[i];
					tile[i*4+j] = _delta_encode_32(dw, previous);
					previous = dw;
				}
				in += 16;
			}

			// 2nd pass
			uint8_t keys = 0;
			uint8_t shift = 0;
			for (size_t i = 0; i < 64; i++) {
				uint8_t code = _svb1_encode1_generic(data_ptr, tile[i]);
				data_ptr += 1 + code;
				if (shift != 8) {
					keys |= code << shift;
					shift += 2;
				} else { // filled a byte
					*key_ptr++ = keys;
					keys = code;
					shift = 2;
				}
			}
			*key_ptr++ = keys;
		} while (in != end);
		previous = *(in - 1);
		count &= 63;
	}

	// delta no transpose
	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb1_encode1_generic(data_ptr, delta);
			data_ptr += 1 + code;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys; // flush buffer
	}
	return data_ptr;
}


#endif // defined(__SSE4_1__) || defined(__AVX__)
