#include <stdio.h> // printf
#include <string.h> // memcmp
#include <stddef.h>  // size_t
#include <stdint.h> // uint8_t, uint16_t, uint32_t, uint64_t


#include "svb.h"

uint8_t buf0[0x100000];
uint8_t buf1[0x140000];
uint8_t buf2[0x140000];

int test_roundtrip(void)
{
	const char* fn;
	printf("test_roundtrip...");fflush(0);

	for (size_t i = 0; i <= 0x10000 / 4; i++) {
		uint8_t *enc_end;
		const uint8_t *dec_end;
		const size_t max_len = streamvbyte_compress_bound(i);

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb1";
		enc_end = svb1_enc((uint32_t*)buf0, i, buf1);
		dec_end = svb1_dec(buf1, i, (uint32_t*)buf2);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb1z";
		enc_end = svb1z_enc((uint32_t*)buf0, i, buf1);
		dec_end = svb1z_dec(buf1, i, (uint32_t*)buf2);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb1d";
		enc_end = svb1d_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb1d_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb1dz";
		enc_end = svb1dz_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb1dz_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb1dt";
		enc_end = svb1dt_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb1dt_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb0";
		enc_end = svb0_enc((uint32_t*)buf0, i, buf1);
		dec_end = svb0_dec(buf1, i, (uint32_t*)buf2);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb0z";
		enc_end = svb0z_enc((uint32_t*)buf0, i, buf1);
		dec_end = svb0z_dec(buf1, i, (uint32_t*)buf2);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn ="svb0d";
		enc_end = svb0d_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb0d_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn ="svb0dz";
		enc_end = svb0dz_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb0dz_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;

		memset(buf1, 0xFE, 0x140000);
		memset(buf2, 0xFD, 0x140000);

		fn = "svb0dt";
		enc_end = svb0dt_enc((uint32_t*)buf0, i, buf1, 42);
		dec_end = svb0dt_dec(buf1, i, (uint32_t*)buf2, 42);
		if (enc_end != dec_end) goto fail_size;
		if (memcmp(buf0, buf2, i*4) != 0) goto fail_cmp;
		if (buf1[max_len] != 0xFE) goto fail_overflow;
		if (buf2[i*4] != 0xFD) goto fail_overflow;
	}

	printf("ok\n");
	return 0;

fail_size:
	printf("\n\n!!! %s fail - stream length disagreement.\n\n", fn);
	return -1;

fail_cmp:
	printf("\n\n!!! %s fail - round-trip.\n\n", fn);
	return -2;

fail_overflow:
	printf("\n\n!!! %s fail - wrote out of bounds.\n\n", fn);
	return -3;
}


int main(void)
{
	// fill with all possible 16-byte set/zero combinations
	uint8_t* p = buf0;
	for( int i = 0; i < 0x10000; i++ ){
		p[0]  = (uint8_t)(((i >> 0 ) & 1) - 1) & 0x80;
		p[1]  = (uint8_t)(((i >> 1 ) & 1) - 1) & 0x11;
		p[2]  = (uint8_t)(((i >> 2 ) & 1) - 1) & 0x22;
		p[3]  = (uint8_t)(((i >> 3 ) & 1) - 1) & 0x33;
		p[4]  = (uint8_t)(((i >> 4 ) & 1) - 1) & 0x44;
		p[5]  = (uint8_t)(((i >> 5 ) & 1) - 1) & 0x55;
		p[6]  = (uint8_t)(((i >> 6 ) & 1) - 1) & 0x66;
		p[7]  = (uint8_t)(((i >> 7 ) & 1) - 1) & 0x77;
		p[8]  = (uint8_t)(((i >> 8 ) & 1) - 1) & 0x88;
		p[9]  = (uint8_t)(((i >> 9 ) & 1) - 1) & 0x99;
		p[10] = (uint8_t)(((i >> 10) & 1) - 1) & 0xAA;
		p[11] = (uint8_t)(((i >> 11) & 1) - 1) & 0xBB;
		p[12] = (uint8_t)(((i >> 12) & 1) - 1) & 0xCC;
		p[13] = (uint8_t)(((i >> 13) & 1) - 1) & 0xDD;
		p[14] = (uint8_t)(((i >> 14) & 1) - 1) & 0xEE;
		p[15] = (uint8_t)(((i >> 15) & 1) - 1) & 0xFF;
		p += 16;
	}

	test_roundtrip();

	printf("done\n");
	return 0;
}

