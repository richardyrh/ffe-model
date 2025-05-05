import numpy as np
import matplotlib.pyplot as plt
import fir
import torch
import torch.nn.functional as F
import scipy.signal as signal

np.set_printoptions(precision=4, suppress=True)
levels = np.array([-2, -1, 0, 1, 2])

#####################
# FILTER TRAINING
#####################
def ffe_train(delay, num_taps, channel_rx, waveform):
    def adaptive_ffe(rx_signal, tx_symbols, taps=8, mu=0.01, n_iters=1):
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

    # taps = adaptive_ffe_torch(channel_rx[delay:], waveform[:-delay], taps=num_taps, n_iters=100, mu=1e-3)
    taps = adaptive_ffe(channel_rx[delay:], waveform[:-delay], taps=num_taps, n_iters=2, mu=0.03)
    max_tap = np.max(np.abs(taps))
    return taps / max_tap, max_tap

def ffe_eval(taps, channel_rx):
    padded_rx = np.hstack([[0] * (len(taps) - 1), channel_rx])
    return np.convolve(channel_rx, taps, mode='same')


# trains an ffe filter and outputs equalized signal, as well as trained taps
def ffe(delay, ntaps, tx_signal, rx_signal):
    actual_delay = max(0, int(ntaps // 2) - delay)
    # first train filter
    taps, filter_factor = ffe_train(delay, ntaps, rx_signal, tx_signal)
    # equalize adc bits
    equalized_rx = ffe_eval(taps, rx_signal)[actual_delay:] * filter_factor
    return equalized_rx, taps

def sweep_test():
    def debug(label, arr):
        print(label, arr[:20], np.min(arr), np.max(arr))

    # ----- generate some symbols -------------------------------------------------
    N_BITS = 10_000
    rng = np.random.default_rng(seed=42)
    bits_tx = rng.integers(-2, 3, N_BITS).tolist()

    # ----- pass through channel --------------------------------------------------
    waveform = np.array(bits_tx, dtype="float")
    channel_rx = fir.simulate_channel(waveform, freq=125e6, cable_length=100)
    # debug("channel", channel_rx)

    # ----- normalize tx & rx waveform to -1~1 to simulate adc --------------------
    channel_rx = channel_rx / np.max(np.abs(channel_rx))
    waveform_factor = np.max(np.abs(waveform))
    normalized_waveform = waveform / waveform_factor

    # ----- calculate error rate for unequalized baseline -------------------------
    baseline_scaled_rx = channel_rx * np.max(np.abs(waveform)) / np.max(np.abs(channel_rx))
    baseline_bits_rx   = np.array([levels[np.argmin(abs(levels - v))] for v in baseline_scaled_rx])
    baseline_ber       = np.mean(np.not_equal(bits_tx, baseline_bits_rx))
    print(f"baseline ber\t= {baseline_ber}")

    # ----- sweep different number of taps and delays -----------------------------
    min_taps = 3
    max_taps = 8
    max_delay = 4

    for delay in range(1, max_delay + 1):
        for ntaps in range(max(min_taps, delay * 2), max_taps + 1):
            actual_delay       = max(0, int(ntaps // 2) - delay)
            # equalize sampled rx signal
            equalized_rx, taps = ffe(delay, ntaps, normalized_waveform, channel_rx)
            # rescale to original tx scale (-2~2)
            scaled_rx          = equalized_rx * waveform_factor
            # decide nearest level
            bits_rx            = np.array([levels[np.argmin(abs(levels - v))] for v in scaled_rx])
            # calculate error rate
            ber                = np.mean(np.not_equal(
                                     bits_tx[:-actual_delay] if actual_delay else bits_tx[:], 
                                     bits_rx[:]))

            print(f"taps[{delay}][{ntaps}]\t=", ", ".join([f"{t:.3}" for t in taps]))
            print(f"ber[{delay}][{ntaps}]\t= {ber}")

if __name__ == "__main__":
    sweep_test()
