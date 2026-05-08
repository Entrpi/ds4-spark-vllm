/* CLI driver for ds4 dot validation.
 *
 * Wire format (all little-endian binary on stdin):
 *   1 byte op:
 *     'q' = quantize Q8_K
 *     'i' = IQ2_XXS x Q8_K dot
 *     '2' = Q2_K x Q8_K dot
 *   4 bytes uint32 n_blocks
 *   then op-specific payload:
 *     'q': n_blocks * QK_K float32 inputs
 *          -> writes n_blocks * sizeof(block_q8_K) bytes to stdout
 *     'i': n_blocks * (sizeof(block_iq2_xxs) + sizeof(block_q8_K)) bytes
 *          -> writes 1 float32 to stdout
 *     '2': n_blocks * (sizeof(block_q2_K)   + sizeof(block_q8_K)) bytes
 *          -> writes 1 float32 to stdout
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define QK_K 256

typedef struct {
    uint8_t  scales[QK_K / 16];
    uint8_t  qs[QK_K / 4];
    uint16_t d;
    uint16_t dmin;
} block_q2_K;

typedef struct {
    float   d;
    int8_t  qs[QK_K];
    int16_t bsums[QK_K / 16];
} block_q8_K;

typedef struct {
    uint16_t d;
    uint16_t qs[QK_K / 8];
} block_iq2_xxs;

void ds4_quantize_row_q8_K(const float *x, block_q8_K *y, int64_t k);
void ds4_vec_dot_q2_K_q8_K(int n, float *s, const block_q2_K *x, const block_q8_K *y);
void ds4_vec_dot_iq2_xxs_q8_K(int n, float *s, const block_iq2_xxs *x, const block_q8_K *y);

static int read_exact(void *buf, size_t n) {
    size_t got = 0;
    while (got < n) {
        size_t r = fread((char *)buf + got, 1, n - got, stdin);
        if (r == 0) return -1;
        got += r;
    }
    return 0;
}

int main(void) {
    int op = fgetc(stdin);
    if (op == EOF) return 1;

    uint32_t n_blocks;
    if (read_exact(&n_blocks, sizeof(n_blocks)) < 0) return 2;

    if (op == 'q') {
        size_t bytes = (size_t)n_blocks * QK_K * sizeof(float);
        float *x = malloc(bytes);
        block_q8_K *y = malloc((size_t)n_blocks * sizeof(*y));
        if (!x || !y) return 3;
        if (read_exact(x, bytes) < 0) return 4;
        ds4_quantize_row_q8_K(x, y, (int64_t)n_blocks * QK_K);
        fwrite(y, sizeof(*y), n_blocks, stdout);
        free(x); free(y);
        return 0;
    }

    if (op == 'i') {
        block_iq2_xxs *x = malloc((size_t)n_blocks * sizeof(*x));
        block_q8_K *y = malloc((size_t)n_blocks * sizeof(*y));
        if (!x || !y) return 3;
        if (read_exact(x, (size_t)n_blocks * sizeof(*x)) < 0) return 4;
        if (read_exact(y, (size_t)n_blocks * sizeof(*y)) < 0) return 5;
        float s = 0.0f;
        ds4_vec_dot_iq2_xxs_q8_K((int)n_blocks * QK_K, &s, x, y);
        fwrite(&s, sizeof(s), 1, stdout);
        free(x); free(y);
        return 0;
    }

    if (op == '2') {
        block_q2_K *x = malloc((size_t)n_blocks * sizeof(*x));
        block_q8_K *y = malloc((size_t)n_blocks * sizeof(*y));
        if (!x || !y) return 3;
        if (read_exact(x, (size_t)n_blocks * sizeof(*x)) < 0) return 4;
        if (read_exact(y, (size_t)n_blocks * sizeof(*y)) < 0) return 5;
        float s = 0.0f;
        ds4_vec_dot_q2_K_q8_K((int)n_blocks * QK_K, &s, x, y);
        fwrite(&s, sizeof(s), 1, stdout);
        free(x); free(y);
        return 0;
    }

    fprintf(stderr, "unknown op %d\n", op);
    return 99;
}
