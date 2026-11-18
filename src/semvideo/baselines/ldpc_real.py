"""Real LDPC encode / decode + BPSK / QPSK modulation for classical baselines.

Complements the BER-threshold LDPC approximation of
``scripts/run_classical_clssim_sweep.py`` by running an actual sparse-parity-check code via
``pyldpc`` (MacKay-style irregular construction, belief-propagation decoder).

Why pyldpc?
    pyldpc gives us a self-contained, pip-installable, deterministic-by-seed
    LDPC stack that is suitable for paper-class baseline comparisons. It is NOT
    a 5G-grade NR-LDPC decoder; for production-equivalent codes one should call
    Sionna's LDPC blocks. Sionna requires TensorFlow which conflicts with our
    PyTorch venv on Windows, so we keep it optional (``[sionna]`` extra) and
    default to pyldpc here.

Code rates supported (regular (d_v, d_c) ensembles, rate = 1 - d_v/d_c):
    1/3   – ``make_ldpc(n_v, d_v=4, d_c=6)``
    1/2   – default, ``make_ldpc(n_v, d_v=4, d_c=8)``
    2/3   – ``make_ldpc(n_v, d_v=4, d_c=12)``
    3/4   – ``make_ldpc(n_v, d_v=4, d_c=16)``

The block length n must be divisible by d_c (pyldpc requirement); the helper
rounds up to the nearest multiple.

Modulation:
    BPSK  – ±1 antipodal, 1 bit/symbol.
    QPSK  – Gray-coded (I, Q) ∈ {±1, ±1}, 2 bits/symbol, complex.

SNR semantics (per the work order, "BPSK/QPSK + LDPC encode/decode at configurable
code rate (1/2 or 3/4)"):
    snr_db is interpreted as *Es/N0* (per coded symbol), matching pyldpc's
    internal convention: σ² = 10^(-snr_db/10). This makes the SNR sweep axis
    directly comparable to standard JSCC papers (DeepJSCC, SwinJSCC) which
    also use per-symbol SNR. To convert to Eb/N0 add 10·log10(1/(R·bps)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _lazy_pyldpc():
    """Import ``pyldpc`` or raise an informative error with install hint."""
    try:
        import pyldpc

        return pyldpc
    except ImportError as exc:
        raise RuntimeError("pyldpc not installed. Run: pip install pyldpc") from exc


# ─────────────────────────────────────────────────────────────────────
# Byte <-> bit utilities (MSB-first, matches pyldpc convention)
# ─────────────────────────────────────────────────────────────────────
def bytes_to_bits(data: bytes) -> np.ndarray:
    """Convert a byte string to a 1-D uint8 array of bits (MSB first).

    Args:
        data: byte string of length L.

    Returns:
        ``np.ndarray`` shape (8 * L,) of values in {0, 1}.
    """
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


def bits_to_bytes(bits: np.ndarray) -> bytes:
    """Inverse of ``bytes_to_bits``. Right-pads to a multiple of 8 with zeros.

    Args:
        bits: array of 0/1 with length divisible by 8 (or it will be zero-padded).

    Returns:
        Byte string of length ``len(bits) // 8`` (after padding).
    """
    bits = np.asarray(bits, dtype=np.uint8) & 1
    if bits.size % 8:
        pad = 8 - (bits.size % 8)
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    return np.packbits(bits).tobytes()


# ─────────────────────────────────────────────────────────────────────
# LDPC matrix cache
# ─────────────────────────────────────────────────────────────────────
_LDPC_CACHE: dict[tuple[int, int, int, int], tuple[object, object, int]] = {}


def _get_ldpc(rate: str, n_block: int, seed: int) -> tuple[object, object, int]:
    """Return ``(H, G, k)`` for the requested rate and block length.

    pyldpc.make_ldpc(n, d_v, d_c) requires d_c | n. Rate ≈ 1 − d_v/d_c.
    (d_v=4, d_c=8)  ⇒ rate ≈ 1/2.
    (d_v=4, d_c=16) ⇒ rate ≈ 3/4.

    Args:
        rate: one of ``"1/3"``, ``"1/2"``, ``"2/3"``, ``"3/4"``.
        n_block: target codeword length (rounded UP to the nearest multiple of d_c).
        seed: PRNG seed so the matrix is deterministic per call.

    Returns:
        H (parity-check), G (generator, n × k), k (number of message bits).
    """
    # rate = 1 - d_v/d_c for a regular (d_v, d_c) ensemble.
    _DEGREES = {
        "1/3": (4, 6),
        "1/2": (4, 8),
        "2/3": (4, 12),
        "3/4": (4, 16),
    }
    if rate not in _DEGREES:
        raise ValueError(f"rate must be one of {sorted(_DEGREES)}, got {rate!r}")
    d_v, d_c = _DEGREES[rate]

    n = max(d_c * 2, ((n_block + d_c - 1) // d_c) * d_c)
    key = (d_v, d_c, n, seed)
    if key in _LDPC_CACHE:
        return _LDPC_CACHE[key]

    pyldpc = _lazy_pyldpc()
    H, G = pyldpc.make_ldpc(n, d_v, d_c, systematic=True, sparse=True, seed=seed)
    k = G.shape[1]
    _LDPC_CACHE[key] = (H, G, k)
    return H, G, k


# ─────────────────────────────────────────────────────────────────────
# BPSK / QPSK + LDPC + AWGN
# ─────────────────────────────────────────────────────────────────────
@dataclass
class LDPCResult:
    """Round-trip output of ``bpsk_qpsk_ldpc``.

    Attributes:
        decoded_bits: recovered message bits (length matches input).
        ber: bit-error-rate measured between input and output bits.
        n_blocks: number of LDPC code blocks used.
        n: codeword length per block.
        k: message length per block.
    """

    decoded_bits: np.ndarray
    ber: float
    n_blocks: int
    n: int
    k: int


def bpsk_qpsk_ldpc(
    message_bits: np.ndarray,
    snr_db: float,
    modulation: str = "bpsk",
    code_rate: str = "1/2",
    n_block: int = 240,
    bp_iter: int = 30,
    seed: int = 0,
) -> LDPCResult:
    """Encode bits with LDPC, modulate, push through AWGN, demap, BP-decode.

    For BPSK we use ``pyldpc.encode`` directly (it does the (G·v) mod 2 step,
    BPSK maps 0/1 to +1/−1, and adds Gaussian noise with σ² = 10^(−snr_db/10)).
    For QPSK we Gray-code consecutive bit pairs onto (I, Q) ∈ {±1}², halve the
    noise variance per real dimension, and stack the soft I/Q LLRs back into a
    real-valued y of the same length as the codeword for ``pyldpc.decode``.

    Args:
        message_bits: 1-D ``{0,1}`` array. Padded with zeros at the tail to a
            multiple of the LDPC message length ``k``.
        snr_db: per-coded-symbol Es/N0 in dB (matches pyldpc's convention).
        modulation: ``"bpsk"`` (1 bps, real) or ``"qpsk"`` (2 bps, complex).
        code_rate: ``"1/2"`` or ``"3/4"``.
        n_block: target codeword length (rounded up to a multiple of d_c).
        bp_iter: max belief-propagation iterations.
        seed: master PRNG seed (for both matrix construction and AWGN draws).

    Returns:
        LDPCResult with decoded bits, BER, and block-geometry metadata.
    """
    if modulation not in ("bpsk", "qpsk"):
        raise ValueError(f"modulation must be 'bpsk' or 'qpsk', got {modulation!r}")
    pyldpc = _lazy_pyldpc()

    H, G, k = _get_ldpc(code_rate, n_block, seed=seed)
    n = G.shape[0]

    bits = np.asarray(message_bits, dtype=np.uint8) & 1
    if bits.size == 0:
        return LDPCResult(np.array([], dtype=np.uint8), 0.0, 0, n, k)

    pad = (-bits.size) % k
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    n_blocks = bits.size // k
    msgs = bits.reshape(n_blocks, k).astype(np.int64)
    decoded = np.empty_like(msgs)

    sigma_bpsk = 10 ** (-snr_db / 20.0)  # σ for BPSK at Es/N0
    rng = np.random.default_rng(seed + 7)

    for b in range(n_blocks):
        if modulation == "bpsk":
            # pyldpc.encode applies BPSK + AWGN at sigma = 10**(-snr/20).
            y = pyldpc.encode(G, msgs[b], snr=snr_db, seed=int(seed + 1000 * b + 1))
        else:
            # QPSK Gray: pair bits (b0, b1) on (I, Q). Equal-energy per bit:
            # split codeword over I/Q with σ_I = σ_Q = sigma_bpsk (so total Es
            # per QPSK symbol = 2). Then re-stack LLRs for pyldpc.decode which
            # expects a real (n,) array of +1/−1 + noise samples.
            d = pyldpc.utils.binaryproduct(G, msgs[b]).astype(np.int64)
            x = 1.0 - 2.0 * d  # ±1 BPSK per coded bit
            # Pair every 2 bits onto an I,Q symbol; if odd, pad with +1 (zero-bit).
            if x.size % 2:
                x = np.concatenate([x, np.ones(1, dtype=x.dtype)])
            pairs = x.reshape(-1, 2)
            tx_c = (pairs[:, 0] + 1j * pairs[:, 1]) / np.sqrt(2.0)
            noise_c = (
                rng.normal(scale=sigma_bpsk, size=tx_c.shape)
                + 1j * rng.normal(scale=sigma_bpsk, size=tx_c.shape)
            ) / np.sqrt(2.0)
            rx_c = tx_c + noise_c
            # Recover the per-bit soft samples for the BP decoder. The BPSK
            # equivalent is y_i = ±1 + N(0, σ²); rx_c.real/sqrt(2)*sqrt(2) = ±1 + n.
            soft = np.empty(pairs.size, dtype=np.float64)
            soft[0::2] = rx_c.real * np.sqrt(2.0)
            soft[1::2] = rx_c.imag * np.sqrt(2.0)
            y = soft[:n]  # trim padding

        try:
            codeword_hat = pyldpc.decode(H, y, snr=snr_db, maxiter=bp_iter)
        except Exception:
            # BP can diverge at extreme SNR; fall back to hard decision so the
            # caller still gets a BER and the sweep CSV stays schema-stable.
            codeword_hat = (y < 0).astype(np.int64)

        msg_hat = pyldpc.get_message(G, codeword_hat)
        decoded[b] = msg_hat[:k]

    decoded_bits = decoded.reshape(-1)[: message_bits.size]
    ref = np.asarray(message_bits, dtype=np.uint8) & 1
    ref = ref[: decoded_bits.size]
    ber = float(np.mean(decoded_bits != ref)) if ref.size else 0.0

    return LDPCResult(
        decoded_bits=decoded_bits.astype(np.uint8),
        ber=ber,
        n_blocks=n_blocks,
        n=n,
        k=k,
    )
