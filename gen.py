import numpy as np
import simple
import fir

levels = np.array([-2, -1, 0, 1, 2])

def gen_ffe_eval(taps, channel_rx):
    equalized_rx = simple.ffe_eval(taps, rx_signal)
    padded_rx = np.hstack([[0] * (len(taps) - 1), channel_rx])
    return np.convolve(channel_rx, taps, mode='same')

if __name__ == "__main__":
    # ----- generate some symbols -------------------------------------------------
    N_BITS = 100_000
    rng = np.random.default_rng(seed=42)
    bits_tx = rng.integers(-2, 3, N_BITS).tolist()

    # ----- pass through channel --------------------------------------------------
    waveform = np.array(bits_tx, dtype="float")
    channel_rx = fir.simulate_channel(waveform, freq=125e6, cable_length=200)
    # debug("channel", channel_rx)

    # ----- normalize tx & rx waveform to -1~1 to simulate adc --------------------
    channel_rx = channel_rx / np.max(np.abs(channel_rx))
    waveform_factor = np.max(np.abs(waveform)) # = 2

    delay = 3
    ntaps = 7
    taps = [-8, 8, -55, 127, -55, 8, -8]
    # taps = [-0.0613, 0.0658, -0.433, 1.0, -0.433, 0.0658, -0.0614]

    actual_delay       = max(0, int(ntaps // 2) - delay)
    # equalize sampled rx signal
    equalized_rx       = simple.ffe_eval(taps, channel_rx)[actual_delay:]
    # rescale to original tx scale (-2~2)
    scaled_rx          = equalized_rx * 2.5 / np.max(np.abs(equalized_rx))
    # decide nearest level
    bits_rx            = np.array([levels[np.argmin(abs(levels - v))] for v in scaled_rx])
    # calculate error rate
    ber                = np.mean(np.not_equal(
                             bits_tx[:-actual_delay] if actual_delay else bits_tx[:], 
                             bits_rx[:]))

    print(f"taps[{delay}][{ntaps}]\t=", ", ".join([f"{t}" for t in taps]))
    print(f"ber[{delay}][{ntaps}]\t= {ber:.5f}")
