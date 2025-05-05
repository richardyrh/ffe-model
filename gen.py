import numpy as np
import simple
import fir

acc_width = 8
msb = -2
levels = np.array([-2, -1, 0, 1, 2])
fac = 1.485 / (2 ** (acc_width - msb)) * 32 * 0.400
levels_raw         = np.round(np.array(levels) / fac / 4) * 4

# def ffe_eval(taps, channel_rx):
#     padded_rx = np.hstack([[0] * (len(taps) - 1), channel_rx])
#     return np.convolve(channel_rx, taps, mode='same')

def ffe_eval(taps, channel_rx):
    taps = np.asarray(taps, dtype=np.int16)
    channel_rx = np.asarray(channel_rx, dtype=np.int16)
    
    n_taps = len(taps)
    pad = n_taps // 2
    padded_rx = np.pad(channel_rx, (pad, pad), mode='constant')

    result = np.zeros_like(channel_rx, dtype=np.int32)

    for i in range(len(channel_rx)):
        acc = 0
        for j in range(n_taps):
            product = padded_rx[i + j] * taps[j]
            quantized_product = product >> (8 + 8 - acc_width + msb)
            acc += quantized_product
        result[i] = np.clip(acc, -(1 << (acc_width - 1)), (1 << (acc_width - 1)) - 1)

    return result.astype(np.int32)

if __name__ == "__main__":
    # ----- generate some symbols -------------------------------------------------
    N_BITS = 100_000
    rng = np.random.default_rng(seed=42)
    # rng = np.random.default_rng(seed=42)
    bits_tx = rng.integers(-2, 3, N_BITS).tolist()

    # ----- pass through channel --------------------------------------------------
    waveform = np.array(bits_tx, dtype="float")
    channel_rx = fir.simulate_channel(waveform, freq=125e6, cable_length=100)

    # ----- normalize tx & rx waveform to -1~1 to simulate adc --------------------
    channel_rx = np.round((channel_rx / np.max(np.abs(channel_rx))) * 127)

    delay = 3
    ntaps = 7
    # taps = [-8, 8, -55, 127, -55, 8, -8]
    taps = [-6, 1, -32, 127, -32, 1, -6]
    # taps = [-0.0613, 0.0658, -0.433, 1.0, -0.433, 0.0658, -0.0614]

    actual_delay       = max(0, int(ntaps // 2) - delay)
    # equalize sampled rx signal
    equalized_rx       = ffe_eval(taps, channel_rx)
    time_shifted_rx    = equalized_rx[actual_delay:]
    # bit_shifted_rx     = np.round(time_shifted_rx / (1 << ((8 + 8) - (10 - 2))))
    bit_shifted_rx     = np.round(time_shifted_rx) # / (1 << ((8 + 8) - (10 - 2))))
    print(f"actual max={np.max(np.abs(bit_shifted_rx))}")
    # rescale to original tx scale (-2~2)

    # for fac in np.linspace(0.187, 0.195, 20):
    print(f"fac={fac:.5f}")
    scaled_rx          = bit_shifted_rx * fac
    print(f"scaled max={np.max(np.abs(scaled_rx))}")
    # decide nearest level
    bits_rx            = np.array([levels[np.argmin(abs(levels_raw - v))] for v in bit_shifted_rx])
    # calculate error rate
    ber                = np.mean(np.not_equal(
                             bits_tx[:-actual_delay] if actual_delay else bits_tx[:], 
                             bits_rx[:]))
    print(bits_rx[:10], bits_tx[:10])
  
    print(f"taps[{delay}][{ntaps}]\t=", ", ".join([f"{t}" for t in taps]))
    print(f"ber[{delay}][{ntaps}]\t= {ber:.5f}")
    print(f"levels raw {levels_raw}")
