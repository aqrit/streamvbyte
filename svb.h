#ifndef INCLUDE_SVB_H_
#define INCLUDE_SVB_H_

#include <stddef.h>  // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t

#if __cplusplus
#define restrict __restrict
extern "C" {
#endif

#include "svb_internal.h" // _svb_key_block_len()

/**
 * Calc maximum size of worst case compression scenario.
 * Permits allocation of a buffer to hold output from the encoder.
 *
 * @param [in] count of array elements to be compressed
 * @return size in bytes
 */
static inline
size_t streamvbyte_compress_bound (size_t count) {
	size_t key_len = _svb_key_block_len(count); // control bytes
	size_t data_len_max = count << 2; // litteral bytes
	return key_len + data_len_max;
}


uint8_t* svb1_enc (const uint32_t* in, size_t count, uint8_t* out);
uint8_t* svb1z_enc (const uint32_t* in, size_t count, uint8_t* out);
uint8_t* svb1d_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);
uint8_t* svb1dz_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);
uint8_t* svb1dt_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);

const uint8_t* svb1_dec (const uint8_t* in, size_t count, uint32_t* out);
const uint8_t* svb1z_dec (const uint8_t* in, size_t count, uint32_t* out);
const uint8_t* svb1d_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);
const uint8_t* svb1dz_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);
const uint8_t* svb1dt_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);

uint8_t* svb0_enc (const uint32_t* in, size_t count, uint8_t* out);
uint8_t* svb0z_enc (const uint32_t* in, size_t count, uint8_t* out);
uint8_t* svb0d_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);
uint8_t* svb0dz_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);
uint8_t* svb0dt_enc (const uint32_t* in, size_t count, uint8_t* out, uint32_t previous);

const uint8_t* svb0_dec (const uint8_t* in, size_t count, uint32_t* out);
const uint8_t* svb0z_dec (const uint8_t* in, size_t count, uint32_t* out);
const uint8_t* svb0d_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);
const uint8_t* svb0dz_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);
const uint8_t* svb0dt_dec (const uint8_t* in, size_t count, uint32_t* out, uint32_t previous);



// MSVC doesn't define __SSE4_1__ so instead check for __AVX__
#if defined(__SSE4_1__) || defined(__AVX__)

# define svb1_enc_x64        svb1_enc
# define svb1z_enc_x64       svb1z_enc
# define svb1d_enc_x64       svb1d_enc
# define svb1dz_enc_x64      svb1dz_enc
# define svb1dt_enc_x64      svb1dt_enc

# define svb1_dec_x64        svb1_dec
# define svb1z_dec_x64       svb1z_dec
# define svb1d_dec_x64       svb1d_dec
# define svb1dz_dec_x64      svb1dz_dec
# define svb1dt_dec_x64      svb1dt_dec

# define svb0_enc_x64        svb0_enc
# define svb0z_enc_x64       svb0z_enc
# define svb0d_enc_x64       svb0d_enc
# define svb0dz_enc_x64      svb0dz_enc
# define svb0dt_enc_x64      svb0dt_enc

# define svb0_dec_x64        svb0_dec
# define svb0z_dec_x64       svb0z_dec
# define svb0d_dec_x64       svb0d_dec
# define svb0dz_dec_x64      svb0dz_dec
# define svb0dt_dec_x64      svb0dt_dec

#else

# define svb1_enc_generic    svb1_enc
# define svb1z_enc_generic   svb1z_enc
# define svb1d_enc_generic   svb1d_enc
# define svb1dz_enc_generic  svb1dz_enc
# define svb1dt_enc_generic  svb1dt_enc

# define svb1_dec_generic    svb1_dec
# define svb1z_dec_generic   svb1z_dec
# define svb1d_dec_generic   svb1d_dec
# define svb1dz_dec_generic  svb1dz_dec
# define svb1dt_dec_generic  svb1dt_dec

# define svb0_enc_generic    svb0_enc
# define svb0z_enc_generic   svb0z_enc
# define svb0d_enc_generic   svb0d_enc
# define svb0dz_enc_generic  svb0dz_enc
# define svb0dt_enc_generic  svb0dt_enc

# define svb0_dec_generic    svb0_dec
# define svb0z_dec_generic   svb0z_dec
# define svb0d_dec_generic   svb0d_dec
# define svb0dz_dec_generic  svb0dz_dec
# define svb0dt_dec_generic  svb0dt_dec

#endif

#if __cplusplus
}
#endif

#endif /* INCLUDE_STREAMVBYTE_H_ */
