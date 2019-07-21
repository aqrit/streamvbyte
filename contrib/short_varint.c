/**
 * Varint for uint16
 *
 * Each uint16 is represented by 1 bit in the "keys" bitmap.
 * A set bit means the uint16 uses 2 bytes.
 * A clear bit means the uint16 uses 1 byte, the HIBYTE is zero.
 */


#include <stdint.h> //  uint8_t, uint16_t, uint32_t, uint64_t
#include <string.h> // memcpy
#include <tmmintrin.h> // SSSE3


// worst case: 17 bits per uint16_t
uint64_t short_compress_bound (uint32_t count) {
    uint64_t n = count;
    return (n << 1) + ((n + 7) >> 3);
}


const uint8_t* short_dec (uint16_t *restrict out, const uint8_t *restrict in, size_t count)
{
    const uint8_t *restrict data = &in[((uint64_t)count + 7) >> 3];

    if (count >= 64) {
        const uint64_t kx01 = 0x0101010101010101;
        const uint64_t kmul = kx01 | 0x80;
        const uint64_t kx88 = kx01 * 0x88;
        const __m128i idx = _mm_cvtsi64_si128(0x0F0E0D0C0B0A0908);

        data -= 16;
        do {
            uint64_t keys;
            memcpy(&keys, in, 8); // unaligned load
            in += 8;

            for( int i = 0; i < 8; i++) {
                uint64_t rank = (keys & kx01) * kmul; // prefix sum & put keys in sign bits
                keys >>= 1;
                data += 8 + ((rank + rank) >> 57); // length
                __m128i shuf = _mm_unpacklo_epi8(idx, _mm_cvtsi64_si128(kx88 - rank)); // invert sign bits, get indices
                __m128i src = _mm_loadu_si128((__m128i*)data);
                _mm_storeu_si128((__m128i*)out, _mm_shuffle_epi8(src, shuf));
                out += 8;
            }
            count -= 64;
        } while (count >= 64);
        data += 16;
    }

    // tail loop
    uint8_t keys = 0;
    for (uint32_t i = 0; i < count; i++) {
        if ((i & 7) == 0) {
            keys = *in++;
        }
        if (keys & 1) {
            memcpy(&out[i], data, 2);
            data += 2;
        } else {
            out[i] = *data++;
        }
        keys >>= 1;
    }

    return data;
}


uint8_t* short_enc (uint8_t *restrict out, const uint16_t *restrict in, size_t count)
{
    uint8_t *restrict data = &out[((uint64_t)count + 7) >> 3];

    if (count >= 64) {
        const __m128i separate = _mm_set_epi8(
            14, 12, 10, 8, 6, 4, 2, 0,
            15, 13, 11, 9, 7, 5, 3, 1
        );
        // bits[4:0] = index -> ((trit_d * 9) + (trit_c * 3) + (trit_b * 1) + (trit_a * 0))
        // bits[15:7] = popcnt
        const __m128i sadmask = _mm_cvtsi64_si128(0x8989838381818080);
        const __m128i neg1 = _mm_cmpeq_epi8(sadmask, sadmask); // used for bitwise-not
        static const uint32_t table[27] = { // compressed shuffle control indices
            0x00000001, 0x00000103, 0x00010203, 0x00000105,
            0x00010305, 0x01020305, 0x00010405, 0x01030405,
            0x02030415, 0x00000107, 0x00010307, 0x01020307,
            0x00010507, 0x01030507, 0x02030517, 0x01040507,
            0x03040517, 0x03041527, 0x00010607, 0x01030607,
            0x02030617, 0x01050607, 0x03050617, 0x03051627,
            0x04050617, 0x04051637, 0x04152637
        };

        do{
            __m128i keys = _mm_setzero_si128();
            for (int i = 0; i < 8; i++) {
                __m128i src = _mm_shuffle_epi8(_mm_loadu_si128((__m128i*)in), separate);
                in += 8;

                __m128i mask = _mm_cmpeq_epi8(_mm_setzero_si128(), src);
                keys = _mm_avg_epu8(keys, mask);

                __m128i pack = _mm_or_si128(_mm_and_si128(_mm_slli_epi16(src, 8), mask), src); // reduce to 27 possibilities
                uint64_t desc = _mm_cvtsi128_si64(_mm_sad_epu8(_mm_and_si128(mask, sadmask), sadmask)); // get index & popcnt
                __m128i shuf = _mm_cvtsi32_si128(table[desc & 0x1F]);
                shuf = _mm_or_si128(_mm_slli_epi64(shuf, 28), shuf); // decompress

                _mm_storel_epi64((__m128i*)data, _mm_shuffle_epi8(pack, shuf));
                data += desc >> 7;
                _mm_storeh_pi((__m64*)data, _mm_castsi128_ps(src));
                data += 8;
            }
            _mm_storel_epi64((__m128i*)out, _mm_xor_si128(keys, neg1));
            out += 8;
            count -= 64;
        } while (count >= 64);
    }

    // tail loop
    uint64_t keys = 0;
    for (uint32_t i = 0; i < count; i++) {
        uint64_t w = in[i];
        memcpy(data, &w, 2);
        uint64_t k = (w + 0xFF00) >> 16;
        data += 1 + k;
        keys |= k << i;
    }
    memcpy(out, &keys, (count + 7) >> 3);

    return data;
}


#if 0
int short_test_roundtrip(void)
{
    static uint8_t buf0[0x100000];
    static uint8_t buf1[0x140000];
    static uint8_t buf2[0x140000];

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

    for (size_t i = 0; i <= 0x00100000 / 2; i++) {
        uint8_t *enc_end;
        const uint8_t *dec_end;

        memset(buf1, 0xFE, 0x140000);
        memset(buf2, 0xFD, 0x140000);
        uint64_t max_len = short_compress_bound(i);
        enc_end = short_enc(buf1, (uint16_t*)buf0, i);
        dec_end = short_dec((uint16_t*)buf2, buf1, i);
        if (enc_end != dec_end) goto fail_size;
        if (memcmp(buf0, buf2, i*2) != 0) goto fail_cmp;
        if (buf1[max_len] != 0xFE) goto fail_overflow;
        if (buf2[i*2] != 0xFD) goto fail_overflow;
    }
    return 0;

fail_size: // stream length disagreement.\n\n");
    return 1;

fail_cmp: // fail - round-trip.\n\n");
    return 2;

fail_overflow: // fail - wrote out of bounds.\n\n");
    return 3;
}
#endif
