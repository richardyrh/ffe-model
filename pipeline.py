"""Simple 1000BASE‑T (1‑Gigabit Ethernet) single‑pair baseband simulation
---------------------------------------------------------------------------
This script models the **TX** and **RX** signal processing chain for one
four‑level PAM‑5 wire pair in an ideal (noise‑free) channel.

Implemented pipeline (per IEEE 802.3‑2005, simplified):

TX
~~
bits → 11‑bit self‑synchronising scrambler  
    → 2‑bit‑to‑PAM5 mapper (‑2,‑1,+1,+2)  
    → root‑raised‑cosine (RRC) pulse shaper  
    → analogue waveform (ideal channel)

RX
~~
waveform → matched RRC filter  
        → symbol‑rate sampling & scaling  
        → PAM5 slicer → bits  
        → descrambler → recovered bits

Given an ideal channel the bit‑error‑rate (BER) is **0**.  All parameters
(sps, roll‑off, span, scrambler seed) are configurable.

Test‑bench code at the bottom runs a 1000‑bit simulation and prints the BER.

Author: ChatGPT (April 2025) (and Richard)
"""

import numpy as np
import matplotlib.pyplot as plt
import fir

###############################################################################
# Scrambler / Descrambler (x¹¹ + x⁹ + 1) – specified in IEEE 802.3‑2005 40.6.1
###############################################################################

def _lfsr_step(state: int) -> int:
    """Compute next feedback bit for the 11‑bit LFSR (taps 10 & 8)."""
    return ((state >> 10) ^ (state >> 8)) & 1


def scramble(bits, seed: int = 0x7FF):
    """Self‑synchronising scrambler – identical logic is used for RX."""
    state = seed & 0x7FF
    out = []
    for b in bits:
        fb   = _lfsr_step(state)
        sbit = b ^ fb
        out.append(sbit)
        state = ((state << 1) & 0x7FF) | sbit   # update with scrambled bit
    return out


def descramble(bits, seed: int = 0x7FF):
    """Descrambler (mirror of *scramble* – also self‑synchronising)."""
    state = seed & 0x7FF
    out   = []
    for sbit in bits:
        fb = _lfsr_step(state)
        b  = sbit ^ fb
        out.append(b)
        state = ((state << 1) & 0x7FF) | sbit
    return out


###############################################################################
# Simple 4‑level PAM‑5 mapper (we omit the '0' symbol used for control/idles)
###############################################################################

_MAP  = {(0, 0): -2,
         (0, 1): -1,
         (1, 0):  1,
         (1, 1):  2}
_DEMAP = {v: k for k, v in _MAP.items()}


def pam5_encode(bits):
    """Group input bits into 2‑bit words and map to (‑2,‑1,+1,+2)."""
    if len(bits) % 2:
        bits.append(0)           # pad for odd length
    return [_MAP[(bits[i], bits[i + 1])]
            for i in range(0, len(bits), 2)]


def pam5_decode(symbols):
    """Hard‑decision slicer back to bits (ideal channel ⇒ no errors)."""
    out = []
    for s in symbols:
        out.extend(_DEMAP[int(s)])
    return out


###############################################################################
# Pulse‑shaping – square‑root‑raised‑cosine (SRRC/RRC) filters
###############################################################################

def rrc_filter(beta: float, span: int, sps: int):
    """Generate a *unit‑energy* RRC impulse response."""
    N  = span * sps           # total taps either side (span symbols)
    t  = np.arange(-N, N + 1) / sps
    h  = np.zeros_like(t, dtype=float)

    for i, tt in enumerate(t):
        at = abs(tt)
        if at < 1e-12:
            h[i] = 1.0 + beta * (4/np.pi - 1)
        elif abs(at - 1/(4 * beta)) < 1e-12:
            h[i] = (beta / np.sqrt(2)
                    * ((1 + 2/np.pi) * np.sin(np.pi / (4*beta))
                       + (1 - 2/np.pi) * np.cos(np.pi / (4*beta))))
        else:
            num = (np.sin(np.pi * tt * (1 - beta)) +
                   4 * beta * tt * np.cos(np.pi * tt * (1 + beta)))
            den = (np.pi * tt * (1 - (4 * beta * tt) ** 2))
            h[i] = num / den
    # normalise to unit energy so TX–RX pair ⇒ Nyquist response (no ISI)
    h /= np.sqrt(np.sum(h * h))
    return h


###############################################################################
# Transmitter / Receiver classes
###############################################################################

class Tx1000BaseT:
    """Very‑small‑signal 1000BASE‑T transmitter – *one* wire pair."""

    def __init__(self, sps=8, beta=0.25, span=6, seed=0x7FF):
        self.sps   = sps
        self.seed  = seed
        self.rrc   = rrc_filter(beta, span, sps)

    # --------------------------------------------------------------------- #

    def transmit(self, bits):
        """Return floating‑point baseband waveform (numpy array)."""
        scr_bits = scramble(bits.copy(), self.seed)
        symbols  = pam5_encode(scr_bits)

        # Generate up‑sampled (PAM) sequence
        up = np.zeros(len(symbols) * self.sps, dtype=float)
        up[::self.sps] = symbols

        # Convolve with SRRC pulse
        waveform = np.convolve(up, self.rrc, mode="full")
        return waveform, len(symbols)


class Rx1000BaseT:
    """Matched RX for *Tx1000BaseT* – assumes ideal, noise‑free channel."""

    def __init__(self, sps=8, beta=0.25, span=6, seed=0x7FF):
        self.sps    = sps
        self.seed   = seed
        self.rrc    = rrc_filter(beta, span, sps)
        self.scale  = np.sum(self.rrc * self.rrc)    # TX‑RX energy gain
        self.levels = np.array([-2, -1, 1, 2])

    # --------------------------------------------------------------------- #

    def receive(self, waveform, n_syms):
        """Recover **n_bits = n_syms × 2** descrambled bits from waveform."""
        # Matched filter
        y = np.convolve(waveform, self.rrc[::-1], mode="full")

        # Symbol‑spaced sampling (account for filter delay)
        delay  = len(self.rrc) - 1
        idx    = np.arange(delay, delay + n_syms * self.sps, self.sps)
        slices = y[idx] / self.scale

        # Hard decision to nearest PAM‑5 level
        decisions = np.array([self.levels[np.argmin(abs(self.levels - v))]
                              for v in slices])

        scr_bits  = pam5_decode(decisions)
        rx_bits   = descramble(scr_bits, self.seed)

        return rx_bits[:n_syms * 2]      # drop pad bit (if any)


###############################################################################
# Quick self‑test / demonstration
###############################################################################

def adaptive_ffe(rx_signal, tx_symbols, taps=8, mu=0.01, n_iters=1, init_w=None):
    """LMS equalizer."""
    if init_w is None:
        w = np.zeros(taps)
        w[taps//2] = 1.0  # start with middle tap active
    else:
        w = init_w[:]
    x = np.zeros(taps)
    y_out = []
    for _ in range(5 if mu > 0 else 1):
      for _ in range(n_iters):
          for i in range(len(rx_signal)):
              x = np.roll(x, -1)
              x[-1] = rx_signal[i]
              y = np.dot(w, x)
              d = tx_symbols[i] if i < len(tx_symbols) else 0
              e = d - y
              w = w + mu * e * x  # LMS update
              y_out.append(y)
      mu *= 0.1
    return np.array(y_out), w

def debug(label, arr):
    print(label, arr[:20], np.min(arr), np.max(arr))

if __name__ == "__main__":
    N_BITS = 10_000
    rng    = np.random.default_rng(seed=42)
    tx     = Tx1000BaseT(sps=1)
    rx     = Rx1000BaseT(sps=1)

    np.set_printoptions(precision=4, suppress=True)

    bits_tx       = rng.integers(0, 2, N_BITS).tolist()
    # bits_tx       = np.zeros(N_BITS, dtype=int)
    # bits_tx[::2] = 1
    # bits_tx = bits_tx.tolist()
    debug("tx", bits_tx)
    debug("tx", bits_tx[-20:])
    waveform, Ns  = tx.transmit(bits_tx)
    debug("waveform", waveform)

    plt.figure(figsize=(15, 5))
    plt.plot(waveform[:100])
    plt.plot(bits_tx[:100])
    plt.savefig("test4.png")

    # waveform = waveform[1::2]

    max_amp = np.max(np.abs(waveform))
    channel_rx = fir.simulate_channel(waveform, cable_length=100)
    debug("channel", channel_rx)

    


    # taps = fir.gen_taps(cable_length=10, num_taps=9, fir_type=1)
    # # taps = fir.gen_taps(cable_length=1, num_taps=8, fir_type=4)
    # # taps = np.array([0, 0, 0, 1, 0, 0, 0, 0])
    # equalized_rx = fir.manual_filter(taps, channel_tx / max_amp)
    # equalized_rx *= (max_amp / np.max(np.abs(equalized_rx)))
    # equalized_rx = equalized_rx[1:]
    # debug("equalized", equalized_rx)

    delay = 1
    num_taps = 8

    def train(delay, num_taps, channel_rx, waveform):
      _, taps = adaptive_ffe(channel_rx[delay:], waveform[:-delay], taps=num_taps, n_iters=2, mu=0.04)
      equalized_rx, taps = adaptive_ffe(channel_rx, waveform, taps=num_taps, n_iters=1, mu=0, init_w=taps)
      return equalized_rx[delay:], taps

    min_taps = 4
    max_taps = 8
    results = np.zeros((5, max_taps + 1))
    final_taps = None
    for delay in range(1,4):
      for ntaps in range(min_taps, max_taps + 1):
        equalized_rx, taps = train(delay, ntaps, channel_rx, waveform)
        bits_rx       = rx.receive(equalized_rx, Ns)
        ber           = np.mean(np.not_equal(bits_tx, bits_rx))
        print(f"ber[{delay}][{ntaps}] = {ber}")
        results[delay][ntaps] = ber
        if delay == 1 and ntaps == 4:
          final_taps = taps

    # baseline
    equalized_rx = channel_rx
    bits_rx      = rx.receive(equalized_rx, Ns)
    ber          = np.mean(np.not_equal(bits_tx, bits_rx))
    print(f"baseline ber = {ber}")


    print(final_taps)
    
    # shmoo plot of results
    plt.figure(figsize=(16, 12))
    plt.imshow(results[:, min_taps:], origin='lower', cmap='viridis', aspect='auto', 
               extent=[min_taps - 0.5, results.shape[1] - 0.5, -0.5, results.shape[0] - 0.5])
    plt.colorbar(label='Result Value')
    plt.xlabel('num taps')
    plt.ylabel('delay')
    plt.title('Schmoo Plot')

    nrows, ncols = results.shape
    print(nrows, ncols)
    for i in range(nrows):
      for j in range(ncols):
        plt.text(j, i, f"{results[i, j]:.4f}", color='white', ha='center', va='center', fontsize=18)


    plt.savefig("schmoo.png")




    print("taps", taps)
    # debug("equalized", equalized_rx)



    # bits_rx       = rx.receive(waveform, Ns)
    # bits_rx       = rx.receive(channel_tx, Ns)
    bits_rx       = rx.receive(equalized_rx[:], Ns)

    debug("rx", bits_rx)
    debug("rx", bits_rx[-20:])

    for offset in range(14):
      if offset > 0:
        ber = np.mean(np.not_equal(bits_tx[:-offset], bits_rx[offset:]))
      else:
        ber = np.mean(np.not_equal(bits_tx, bits_rx))

      print(f"Simulated {N_BITS} bits – BER = {ber * 100}%")

