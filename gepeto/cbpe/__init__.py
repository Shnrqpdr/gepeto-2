"""
Wrapper ctypes para o encoder BPE em C.

Compila automaticamente na primeira importacao se a .so nao existir.
"""

import ctypes
import os
import subprocess
from ctypes import c_int, POINTER

_dir = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_dir, "bpe_merge.c")
_lib = os.path.join(_dir, "bpe_merge.so")


def _compile():
    """Compila o bpe_merge.c -> bpe_merge.so"""
    cmd = ["gcc", "-O2", "-shared", "-fPIC", "-o", _lib, _src]
    subprocess.check_call(cmd)


def _load():
    if not os.path.exists(_lib) or os.path.getmtime(_src) > os.path.getmtime(_lib):
        _compile()
    lib = ctypes.CDLL(_lib)

    # apply_merges(tokens, num_tokens, merges_a, merges_b, base_id, num_merges) -> int
    lib.apply_merges.argtypes = [
        POINTER(c_int), c_int,
        POINTER(c_int), POINTER(c_int),
        c_int, c_int,
    ]
    lib.apply_merges.restype = c_int

    # apply_merges_batch(...) -> int
    lib.apply_merges_batch.argtypes = [
        POINTER(c_int), POINTER(c_int),
        POINTER(c_int), POINTER(c_int),
        c_int,
        POINTER(c_int), POINTER(c_int),
        c_int, c_int,
    ]
    lib.apply_merges_batch.restype = c_int

    return lib


_lib_handle = _load()


def apply_merges(tokens: list[int], merges_a: list[int], merges_b: list[int], base_id: int) -> list[int]:
    """Aplica merges BPE em um unico chunk de tokens."""
    n = len(tokens)
    c_tokens = (c_int * n)(*tokens)
    c_ma = (c_int * len(merges_a))(*merges_a)
    c_mb = (c_int * len(merges_b))(*merges_b)

    new_len = _lib_handle.apply_merges(c_tokens, n, c_ma, c_mb, base_id, len(merges_a))
    return list(c_tokens[:new_len])


def apply_merges_batch(
    chunks: list[list[int]],
    merges_a: list[int],
    merges_b: list[int],
    base_id: int,
) -> list[list[int]]:
    """Aplica merges BPE em batch (todos os chunks de uma vez)."""
    num_chunks = len(chunks)

    # Monta buffer continuo e offsets
    flat = []
    offsets = [0]
    for chunk in chunks:
        flat.extend(chunk)
        offsets.append(len(flat))

    total = len(flat)
    c_chunks = (c_int * total)(*flat)
    c_offsets = (c_int * (num_chunks + 1))(*offsets)
    c_out = (c_int * total)()
    c_out_offsets = (c_int * (num_chunks + 1))()
    c_ma = (c_int * len(merges_a))(*merges_a)
    c_mb = (c_int * len(merges_b))(*merges_b)

    _lib_handle.apply_merges_batch(
        c_chunks, c_offsets,
        c_out, c_out_offsets,
        num_chunks,
        c_ma, c_mb,
        base_id, len(merges_a),
    )

    # Reconstroi lista de chunks a partir do buffer de saida
    result = []
    for i in range(num_chunks):
        start = c_out_offsets[i]
        end = c_out_offsets[i + 1]
        result.append(list(c_out[start:end]))

    return result
