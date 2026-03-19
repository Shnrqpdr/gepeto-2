#include <stdlib.h>
#include <string.h>

/*
 * Aplica uma sequencia de merges BPE a um array de tokens.
 *
 * Para cada merge (a, b) -> new_id, varre o array substituindo
 * pares adjacentes (a, b) pelo new_id correspondente.
 *
 * tokens:     array de token ids (modificado in-place)
 * num_tokens: tamanho atual do array (retorna novo tamanho)
 * merges_a:   array com o primeiro elemento de cada merge
 * merges_b:   array com o segundo elemento de cada merge
 * base_id:    id do primeiro merge (new_id = base_id + i)
 * num_merges: quantidade de merges
 *
 * Retorna o novo tamanho do array de tokens.
 */
int apply_merges(
    int *tokens,
    int num_tokens,
    const int *merges_a,
    const int *merges_b,
    int base_id,
    int num_merges
) {
    for (int m = 0; m < num_merges; m++) {
        int a = merges_a[m];
        int b = merges_b[m];
        int new_id = base_id + m;

        int write = 0;
        for (int read = 0; read < num_tokens; read++) {
            if (read < num_tokens - 1 && tokens[read] == a && tokens[read + 1] == b) {
                tokens[write++] = new_id;
                read++; /* pula o proximo, ja consumido */
            } else {
                tokens[write++] = tokens[read];
            }
        }
        num_tokens = write;
    }

    return num_tokens;
}

/*
 * Aplica merges em batch: processa multiplos chunks de uma vez.
 *
 * chunks:         buffer continuo com todos os chunks concatenados
 * chunk_offsets:  inicio de cada chunk no buffer (tamanho num_chunks + 1)
 * out_tokens:     buffer de saida (mesmo tamanho que chunks)
 * out_offsets:    offsets de saida (tamanho num_chunks + 1)
 * num_chunks:     quantidade de chunks
 * merges_a/b:     arrays de merges
 * base_id:        id base dos merges
 * num_merges:     quantidade de merges
 *
 * Retorna o total de tokens de saida.
 */
int apply_merges_batch(
    const int *chunks,
    const int *chunk_offsets,
    int *out_tokens,
    int *out_offsets,
    int num_chunks,
    const int *merges_a,
    const int *merges_b,
    int base_id,
    int num_merges
) {
    int total_out = 0;
    out_offsets[0] = 0;

    for (int c = 0; c < num_chunks; c++) {
        int start = chunk_offsets[c];
        int len = chunk_offsets[c + 1] - start;

        /* Copia chunk pro buffer de saida */
        memcpy(out_tokens + total_out, chunks + start, len * sizeof(int));

        /* Aplica merges in-place no buffer de saida */
        int new_len = apply_merges(out_tokens + total_out, len, merges_a, merges_b, base_id, num_merges);

        total_out += new_len;
        out_offsets[c + 1] = total_out;
    }

    return total_out;
}
