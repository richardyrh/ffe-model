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
import torch
import torch.nn.functional as F
import scipy.signal as signal

np.set_printoptions(precision=4, suppress=True)

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
        # waveform = np.convolve(up, self.rrc, mode="full")
        waveform = np.convolve(up, self.rrc, mode="same")

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
        # y = np.convolve(waveform, self.rrc[::-1], mode="full")
        y = np.convolve(waveform, self.rrc[::-1], mode="same")

        # Symbol‑spaced sampling (account for filter delay)
        delay  = 0 # len(self.rrc) - 1
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

def adaptive_ffe(rx_signal, tx_symbols, taps=8, mu=0.01, n_iters=1):
    """LMS equalizer."""
    w = np.zeros(taps)
    # w[(taps - 1)//2] = 1.0
    x = np.zeros(taps)
    for _ in range(5 if mu > 0 else 1):
      for _ in range(n_iters):
          for i in range(len(rx_signal)):
              x = np.roll(x, -1)
              x[-1] = rx_signal[i]
              y = np.dot(w, x)
              d = tx_symbols[i] if i < len(tx_symbols) else 0
              e = d - y
              w = w + mu * e * x  # LMS update
      mu *= 0.1
    return w


def debug(label, arr):
    print(label, arr[:20], np.min(arr), np.max(arr))

def sanity(rx, tx):
    bits_tx        = np.random.default_rng().integers(0, 2, 100_000).tolist()
    wave, n_syms   = tx.transmit(bits_tx)

    lowpass_taps = signal.firwin2(5, [0, 46/62.5, 49/62.5, 1], [1, 1, 0, 0], fs=2.0, antisymmetric=False)
    lowpass_delay = (len(lowpass_taps) - 1) // 2
    wave_ = signal.lfilter(lowpass_taps, 1.0, wave)[lowpass_delay:]
    wave = wave_ * np.max(np.abs(wave)) / np.max(np.abs(wave_))

    print(len(bits_tx), len(wave))
    bits_rx        = rx.receive(wave, n_syms - lowpass_delay)

    ber = np.mean(np.not_equal(bits_tx[:-lowpass_delay * 2], bits_rx))
    print("BER =", ber)          # prints 0.0
    exit()

if __name__ == "__main__":
    N_BITS = 10_000
    SPS = 1
    rng    = np.random.default_rng(seed=42)
    tx     = Tx1000BaseT(sps=SPS, span=6)
    rx     = Rx1000BaseT(sps=SPS, span=6)
    # sanity(rx, tx)


    # ----- generate signal and transmit waveform----------------------------------
    bits_tx       = rng.integers(0, 2, N_BITS).tolist()
    waveform, Ns  = tx.transmit(bits_tx)

    # ----- pass through channel --------------------------------------------------
    channel_rx = fir.simulate_channel(waveform, freq=125e6*SPS, cable_length=100)
    # channel_rx = waveform
    # channel_rx = channel_rx[1::4]
    debug("channel", channel_rx)

    # ----- low pass filter to eliminate isi --------------------------------------

    # vvv this enables isi elimination
    lowpass_taps = signal.firwin2(7, [0, 45/62.5, 50/62.5, 1], [1, 1, 0, 0], fs=2.0, antisymmetric=False)
    # vvv this disables isi elimination but keeps the delay
    # lowpass_taps = [0, 1, 0]

    lowpass_delay = (len(lowpass_taps) - 1) // 2
    channel_rx_ = signal.lfilter(lowpass_taps, 1.0, channel_rx)[lowpass_delay:]
    channel_rx = channel_rx_ * np.max(np.abs(channel_rx)) / np.max(np.abs(channel_rx_))

    # ----- normalize tx & rx waveform to -1~1 to simulate adc --------------------
    channel_rx = channel_rx / np.max(np.abs(channel_rx))
    waveform_factor = np.max(np.abs(waveform))
    normalized_waveform = waveform / waveform_factor

    #####################
    # FILTER TRAINING
    #####################
    def train(delay, num_taps, channel_rx, waveform):
        # taps = adaptive_ffe_torch(channel_rx[delay:], waveform[:-delay], taps=num_taps, n_iters=100, mu=1e-3)
        taps = adaptive_ffe(channel_rx[delay:], waveform[:-delay], taps=num_taps, n_iters=2, mu=0.03)
        max_tap = np.max(np.abs(taps))
        return taps / max_tap, max_tap

    def eval_(taps, channel_rx):
        padded_rx = np.hstack([[0] * (len(taps) - 1), channel_rx])
        return np.convolve(channel_rx, taps, mode='same')
        # x = np.zeros_like(taps)
        # w = taps
        # y_out = []
        # for i in range(len(channel_rx)):
        #     x = np.roll(x, -1)
        #     x[-1] = channel_rx[i]
        #     y = np.dot(w, x)
        #     y_out.append(y)
        # return np.array(y_out)

    #####################
    # SWEEP
    #####################
    min_taps = 3
    max_taps = 8
    max_delay = 4

    results = np.zeros((max_delay + 1, max_taps + 1)) + 0.5
    final_taps = None
    for delay in range(1, max_delay + 1):
        for ntaps in range(max(min_taps, delay * 2), max_taps + 1):
            taps, filter_factor = train(delay, ntaps, channel_rx, normalized_waveform[:-lowpass_delay])
            actual_delay      = max(0, int(ntaps // 2) - delay)
            equalized_rx      = eval_(taps, channel_rx)[actual_delay:]
            actual_delay      += lowpass_delay
            # equalized_rx      = eval_(taps, channel_rx)[delay:]
            bits_rx           = rx.receive(equalized_rx * waveform_factor * filter_factor, Ns - actual_delay)
            ber               = np.mean(np.not_equal(
                bits_tx[:-actual_delay*2] if actual_delay else bits_tx[:], 
                bits_rx[:]))
            print("weights:", ", ".join([f"{t:.3}" for t in taps]))
            print(f"ber[{delay}][{ntaps}] = {ber}")
            results[delay][ntaps] = ber
            if delay == 1 and ntaps == 6:
                final_taps = taps

    #####################
    # BASELINE
    #####################
    equalized_rx = channel_rx * waveform_factor
    bits_rx      = rx.receive(equalized_rx, Ns - lowpass_delay)
    ber          = np.mean(np.not_equal(bits_tx[:-lowpass_delay * 2], bits_rx))
    print(f"baseline ber = {ber}")


    print(final_taps)

    #####################
    # SHMOOOO
    #####################
    plt.figure(figsize=(16, 12))
    plt.imshow(results[:, min_taps:], origin='lower', cmap='viridis', aspect='auto', 
               extent=[min_taps - 0.5, results.shape[1] - 0.5, -0.5, results.shape[0] - 0.5])
    plt.colorbar(label='Result Value')
    plt.xlabel('num taps')
    plt.ylabel('delay')
    plt.title('Shmoo Plot')

    nrows, ncols = results.shape
    print(nrows, ncols)
    for i in range(nrows):
        for j in range(ncols):
            plt.text(j, i, f"{results[i, j]:.4f}", color='white', ha='center', va='center', fontsize=18)


    plt.savefig("shmoo.png")
