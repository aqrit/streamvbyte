/**
 * ~~~
 * The "0124" format.
 * A 2-bit key value of:
 * 0b00 == 0 bytes stored, 4 leading zero bytes stripped.
 * 0b01 == 1 byte  stored, 3 leading zero bytes stripped.
 * 0b10 == 2 bytes stored, 2 leading zero byte  stripped.
 * 0b11 == 4 bytes stored, 0 leading zero bytes stripped.
 * ~~~
 */

#include <stddef.h>  // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#include "svb.h"
#include "svb_internal.h"

#if defined(__SSE4_1__) || defined(__AVX__)

#include <smmintrin.h> // SSE4.1 intrinsics

/// Lists how many bytes are retained from a 16-byte chunk.
/// `size_t length = svb0_len_table[key_byte];`
static const uint8_t svb0_len_table[256] = {
	 0,  1,  2,  4,  1,  2,  3,  5,  2,  3,  4,  6,  4,  5,  6,  8,
	 1,  2,  3,  5,  2,  3,  4,  6,  3,  4,  5,  7,  5,  6,  7,  9,
	 2,  3,  4,  6,  3,  4,  5,  7,  4,  5,  6,  8,  6,  7,  8, 10,
	 4,  5,  6,  8,  5,  6,  7,  9,  6,  7,  8, 10,  8,  9, 10, 12,
	 1,  2,  3,  5,  2,  3,  4,  6,  3,  4,  5,  7,  5,  6,  7,  9,
	 2,  3,  4,  6,  3,  4,  5,  7,  4,  5,  6,  8,  6,  7,  8, 10,
	 3,  4,  5,  7,  4,  5,  6,  8,  5,  6,  7,  9,  7,  8,  9, 11,
	 5,  6,  7,  9,  6,  7,  8, 10,  7,  8,  9, 11,  9, 10, 11, 13,
	 2,  3,  4,  6,  3,  4,  5,  7,  4,  5,  6,  8,  6,  7,  8, 10,
	 3,  4,  5,  7,  4,  5,  6,  8,  5,  6,  7,  9,  7,  8,  9, 11,
	 4,  5,  6,  8,  5,  6,  7,  9,  6,  7,  8, 10,  8,  9, 10, 12,
	 6,  7,  8, 10,  7,  8,  9, 11,  8,  9, 10, 12, 10, 11, 12, 14,
	 4,  5,  6,  8,  5,  6,  7,  9,  6,  7,  8, 10,  8,  9, 10, 12,
	 5,  6,  7,  9,  6,  7,  8, 10,  7,  8,  9, 11,  9, 10, 11, 13,
	 6,  7,  8, 10,  7,  8,  9, 11,  8,  9, 10, 12, 10, 11, 12, 14,
	 8,  9, 10, 12,  9, 10, 11, 13, 10, 11, 12, 14, 12, 13, 14, 16
};


/// Encoder permutation table.
/// `__m128* shuf = (__m128i*)&svb0_enc_table[(key_byte*16) & 0x03F0];`
static const uint8_t svb0_enc_table[64*16] = {
	12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 4, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 4,  5, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  5, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 4,  5,  6,  7, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  6,  7, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7, 12, 13, 14, 15,  0,  0,  0,  0,
	 8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8, 12, 13, 14, 15,  0,  0,  0,
	 8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,
	 4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 12, 13, 14, 15,  0,  0,
	 8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,  0,
	 0,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  1,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,  0,
	 0,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  1,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,  0,
	 0,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,  0,
	 0,  1,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  1,  2,  3,  4,  5,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,
	 4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,  0,
	 0,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,  0,
	 0,  1,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15,  0,  0,
	 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15
};


/**
 * Compress two xmmwords into a "0124 format" stream. Unroll helper.
 * @param [in,out] key_ptr current position in the stream's key block.
 * @param [in,out] data_ptr current position in the stream's data block.
 * @param [in] src_a, src_b xmmwords to encode.
 */
#define SVB0_ENCODE8(key_ptr, data_ptr, src_a, src_b) \
do { \
	const __m128i mask_01 = _mm_set1_epi8(0x01); \
	const __m128i mask_0100 = _mm_slli_epi16(mask_01, 8); \
	\
	/* normalize each byte to 0 or 1 */ \
	__m128i a = _mm_min_epu8(mask_01, src_a); \
	__m128i b = _mm_min_epu8(mask_01, src_b); \
	\
	/* generate keys */ \
	a = _mm_packus_epi32(a, b); \
	a = _mm_min_epi16(a, mask_0100); /* convert 0x0101 to 0x0100 */ \
	a = _mm_slli_epi16(a, 7); \
	size_t keys = (size_t)_mm_movemask_epi8(a); \
	\
	/* In-register shuffle to transfer non-leading-zero bytes to \
	 * contiguous low order byte positions. To reduce table size, \
	 * the 2-bit key of the highest 32-bit subword of an xmmword is \
	 * ignored. (unwanted high bytes are overwritten, later) */ \
	__m128i *shuf_a = (__m128i*)&svb0_enc_table[(keys << 4) & 0x03F0]; \
	__m128i *shuf_b = (__m128i*)&svb0_enc_table[(keys >> 4) & 0x03F0]; \
	__m128i data_a = _mm_shuffle_epi8(src_a, _mm_loadu_si128(shuf_a)); \
	__m128i data_b = _mm_shuffle_epi8(src_b, _mm_loadu_si128(shuf_b)); \
	\
	/* Store an entire xmmword then advance the pointer by how ever \
	 * many of those bytes are actually wanted. Any trailing \
	 * garbage will be overwritten by the next store. \
	 * note: little endian byte memory order */ \
	_mm_storeu_si128((__m128i *)data_ptr, data_a); \
	data_ptr += svb0_len_table[keys & 0xFF]; \
	_mm_storeu_si128((__m128i *)data_ptr, data_b); \
	data_ptr += svb0_len_table[keys >> 8]; \
	\
	/* save keys so the stream can be decoded at some point... */ \
	svb_storeu_u16(key_ptr, (uint16_t)keys); \
	key_ptr += 2; \
} while (0)


/**
 * Compress an unsigned 32-bit integer into a "0124 format" stream.
 * Unroll helper.
 * @param [in,out] keys bit buffer to hold key output
 * @param [in,out] data_ptr current position in the stream's data block.
 * @param [in] val unsigned 32-bit integer to compress.
 * @param [in] shift offset in keys bit buffer
 */
#define SVB0_ENCODE1(keys, data_ptr, val, shift) \
do { \
	uint32_t x = (val); \
	\
	/* copy input directly into the output stream. Unwanted \
	 * bytes (zeros) will be overwritten by next copy. \
	 * note: little endian byte memory order */ \
	svb_storeu_u32(data_ptr, x); \
	\
	/* generate key */ \
	size_t k = (x != 0) + (x > 0x000000FF) + (x > 0x0000FFFF); \
	\
	/* advance pointer by number of bytes actually wanted */ \
	data_ptr += k + (x > 0x0000FFFF); \
	\
	/* place key in the bit buffer */ \
	keys |= k << (shift); \
} while (0)


/**
 * StreamVByte compress an array of unsigned 32-bit integers to a
 * "0124 format" stream.
 *
 * @param [in] in array of elements to be compressed.
 * Ideally elements would be unordered, small, with many zeros.
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
uint8_t* svb0_enc_x64 (const uint32_t* in, size_t count, uint8_t* out)
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
			SVB0_ENCODE8(key_ptr, data_ptr, src_a, src_b);
		} while (in < end);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			SVB0_ENCODE1(keys, data_ptr, in[i], i*2);
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


uint8_t* svb0z_enc_x64 (const uint32_t* in, size_t count, uint8_t* out)
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
			SVB0_ENCODE8(key_ptr, data_ptr, src_a, src_b);
		} while (in < end);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t dw = _zigzag_encode_32(in[i]);
			SVB0_ENCODE1(keys, data_ptr, dw, i*2);
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
 * Delta StreamVByte compress a unsigned 32-bit array to a
 * "0124 format" stream.
 *
 * @param [in] in array of elements to be compressed.
 * Ideally elements would be mostly in ascending ordered with many
 * back-to-back repeats.
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
uint8_t* svb0d_enc_x64 (const uint32_t* in, size_t count,
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

			SVB0_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
		} while (in < end);
		previous = _mm_extract_epi32(prev, 3);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t dw = _delta_encode_32(in[i], previous);
			SVB0_ENCODE1(keys, data_ptr, dw, i*2);
			previous = in[i];
		} while (++i < count);

		// store keys
		key_ptr[0] = (uint8_t)keys;
		if (count > 4) {
			key_ptr[1] = (uint8_t)(keys >> 8);
		}
	}
	return data_ptr;
}


uint8_t* svb0dz_enc_x64 (const uint32_t* in, size_t count,
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
			__m128i diff_b = _mm_delta_zigzag_encode_epi32(src_b, src_a);
			prev = src_b;

			SVB0_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
		} while (in < end);
		previous = _mm_extract_epi32(prev, 3);
	}
	count &= 7; // any chunks of 8 are already done

	// scalar tail loop - do any remaining elements
	if (count != 0) {
		uint32_t keys = 0;
		size_t i = 0;
		do {
			uint32_t dw = _delta_zigzag_encode_32(in[i], previous);
			SVB0_ENCODE1(keys, data_ptr, dw, i*2);
			previous = in[i];
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
 * Delta Transpose StreamVByte compress a unsigned 32-bit array to a
 * "0124 format" stream.
 *
 * @param [in] in array of elements to be compressed.
 * Ideally elements would be mostly in ascending ordered with many
 * back-to-back repeats.
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
uint8_t* svb0dt_enc_x64 (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	uint32_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if (count >= 8) {
		__m128i prev = _mm_insert_epi32(_mm_undefined_si128(), previous, 3);

		// for "large" chunks of 64 elements
		// delta transpose and encode
		for (const uint32_t* end = &in[(count & ~63)]; in != end; in += 64) {
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
			SVB0_ENCODE8(key_ptr, data_ptr, r0, r4);
			SVB0_ENCODE8(key_ptr, data_ptr, r8, rC);

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
			SVB0_ENCODE8(key_ptr, data_ptr, r1, r5);
			SVB0_ENCODE8(key_ptr, data_ptr, r9, rD);

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
			SVB0_ENCODE8(key_ptr, data_ptr, r2, r6);
			SVB0_ENCODE8(key_ptr, data_ptr, rA, rE);

			// finish the last four rows (already loaded and transposed)
			prev = rF;
			rF = _mm_sub_epi32(rF, rB);
			rB = _mm_sub_epi32(rB, r7);
			r7 = _mm_sub_epi32(r7, r3);
			r3 = _mm_sub_epi32(r3, temp);
			SVB0_ENCODE8(key_ptr, data_ptr, r3, r7);
			SVB0_ENCODE8(key_ptr, data_ptr, rB, rF);
		}

		// for any remaining "small" chunks of 8 elements
		// delta transform (no transpose) and encode
		for (const uint32_t* end = &in[(count & (63 ^ 7))]; in != end; in += 8) {
			__m128i src_a = _mm_loadu_si128((__m128i*)&in[0]);
			__m128i src_b = _mm_loadu_si128((__m128i*)&in[4]);

			// delta transform
			__m128i diff_a = _mm_delta_encode_epi32(src_a, prev);
			__m128i diff_b = _mm_delta_encode_epi32(src_b, src_a);
			prev = src_b;

			SVB0_ENCODE8(key_ptr, data_ptr, diff_a, diff_b);
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
			uint32_t dw = _delta_encode_32(in[i], previous);
			SVB0_ENCODE1(keys, data_ptr, dw, i*2);
			previous = in[i];
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
uint8_t _svb0_encode1_generic (size_t* len, uint8_t* dst, uint32_t dw) {
	uint8_t code;
	dst[0] = (uint8_t)dw;
	dst[1] = (uint8_t)(dw >> 8);
	if (dw <= 0x0000FFFF) {
		*len = code = (dw != 0) + (dw > 0xFF);
	} else {
		dw >>= 16;
		dst[2] = (uint8_t)dw;
		dst[3] = (uint8_t)(dw >> 8);
		code = 3;
		*len = 4;
	}
	return code;
}

uint8_t* svb0_enc_generic (const uint32_t* in, size_t count,
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
			size_t len;
			uint8_t code = _svb0_encode1_generic(&len, data_ptr, in[i]);
			data_ptr += len;
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


uint8_t* svb0z_enc_generic (const uint32_t* in, size_t count,
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
			size_t len;
			uint32_t dw = _zigzag_encode_32(in[i]);
			uint8_t code = _svb0_encode1_generic(&len, data_ptr, dw);
			data_ptr += len;
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


uint8_t* svb0d_enc_generic (const uint32_t* in, size_t count,
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
			size_t len;
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb0_encode1_generic(&len, data_ptr, delta);
			data_ptr += len;
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


uint8_t* svb0dz_enc_generic (const uint32_t* in, size_t count,
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
			size_t len;
			uint32_t delta = _delta_zigzag_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb0_encode1_generic(&len, data_ptr, delta);
			data_ptr += len;
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


uint8_t* svb0dt_enc_generic (const uint32_t* in, size_t count,
	uint8_t* out, uint32_t previous)
{
	size_t key_block_len = _svb_key_block_len(count);
	uint8_t *restrict key_ptr = out;
	uint8_t *restrict data_ptr = &out[key_block_len];

	if ((count & ~63) != 0) {
		const uint32_t* end = &in[(count & ~63)];
		do {
			// delta transpose a block of 64 elements
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
			// encode block
			uint8_t keys = 0;
			uint8_t shift = 0;
			for (size_t i = 0; i < 64; i++) {
				size_t len;
				uint8_t code;
				code = _svb0_encode1_generic(&len, data_ptr, tile[i]);
				data_ptr += len;
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

	// no transpose
	if (count != 0) {
		uint8_t keys = 0;
		uint8_t shift = 0;
		size_t i = 0;
		do {
			size_t len;
			uint32_t delta = _delta_encode_32(in[i], previous);
			previous = in[i];
			uint8_t code = _svb0_encode1_generic(&len, data_ptr, delta);
			data_ptr += len;
			if (shift != 8) {
				keys |= code << shift;
				shift += 2;
			} else { // filled a byte
				*key_ptr++ = keys;
				keys = code;
				shift = 2;
			}
		} while (++i < count);
		*key_ptr = keys;
	}
	return data_ptr;
}

#endif // defined(__SSE4_1__) || defined(__AVX__)
