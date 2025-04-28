# Final correct FIR filter implementation
import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import resample

cable_length = 100
high_sample_freq=500e6
nyquist_freq = 62.5 * 1e6

def gen_taps(cable_length=100, num_taps=8, fir_type=4):
  # freq_Hz = np.array([0, 1, 4, 8, 10, 16, 20, 25, 31.25, 62.5]) * 1e6
  # attenuation_dB = np.array([0, 2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4])
  # attenuation_dB = np.max(attenuation_dB) - attenuation_dB
  # attenuation_dB *= (cable_length / 10)
  # gain = 10 ** (-attenuation_dB / 20)

  freq_Hz = np.array([0, 25, 62.5, 100, 125])
  gain = np.array([0, 1, 0, 1, 0])

  if fir_type == 2:
    gain[-1] = 0
  elif fir_type == 3:
    freq_Hz = np.hstack([freq_Hz, [125e6]])
    gain = np.hstack([gain, [0]])
    gain[0] = 0
  elif fir_type == 4:
    gain[0] = 0
  # fir_taps = signal.firwin2(num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0)
  fir_taps = signal.firwin2(num_taps, freq_Hz / np.max(freq_Hz), gain, fs=2.0, antisymmetric=(fir_type % 2 == 0))
  return fir_taps


def upsample(signal, in_freq=125e6, out_freq=high_sample_freq):
  return resample(signal, int(len(signal) * (out_freq // in_freq)))

def downsample(signal):
  # return resample(signal, int(len(signal) // (in_freq // out_freq)))
  return signal[::4]

# Interpolated channel response
def channel_response(frequencies_Hz, cable_length=100):
  freq_MHz = np.array([1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250])
  attenuation_dB = np.array([2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8, 29.0, 32.8])
  adjusted_dB = attenuation_dB * (cable_length / 100)  # Scale attenuation with cable length
  # interp = interp1d(freq_MHz * 1e6, adjusted_dB, kind='linear', fill_value="extrapolate")
  # att_dB = interp(frequencies_Hz)
  # att_linear = 10 ** (-att_dB / 20)  # Convert dB to linear
  interp = interp1d(freq_MHz * 1e6, 10 ** (-adjusted_dB / 20), kind='linear', fill_value="extrapolate")
  att_linear = interp(frequencies_Hz)
  return att_linear

# Channel simulation
def simulate_channel(signal, freq=high_sample_freq, cable_length=100):
  fft_sig = np.fft.rfft(signal)
  freqs = np.fft.rfftfreq(len(signal), d=1/freq)
  response = channel_response(freqs, cable_length=cable_length)
  fft_sig_filtered = fft_sig * response
  signal_filtered = np.fft.irfft(fft_sig_filtered, n=len(signal))
  return signal_filtered

def fir_filter_all(fir_taps, signals):
  return np.array([signal.lfilter(fir_taps, 1.0, sig) for sig in signals])

def fir_filter(fir_taps, signals):
  return signal.lfilter(fir_taps, 1.0, signals)

#######################
# GENERATE TAPS
#######################
fir_taps = gen_taps(cable_length=cable_length, num_taps=9, fir_type=1)
fir_taps = np.array([-0.1002,-0.6199, 0.7888,-5.8325,13.2606,-5.6888])
# print(fir_taps)

#######################
# VISUALIZE FREQ RESP
#######################
if __name__ == "__main__":
  freq_response, response = signal.freqz(fir_taps, whole=True) # 8000)
  # freq_response, response = signal.freqz(fir_taps, worN=8000)
  response_phase = np.unwrap(np.angle(response))
  group_delay = -np.gradient(response_phase, freq_response)
  print("group delay", stats.mode(group_delay))

  plt.figure(figsize=(12, 5))
  ax = plt.subplot(121)
  ax.plot(((freq_response / np.pi) * nyquist_freq / 1e6)[:8000], (20 * np.log10(np.abs(response)))[:8000])
  plt.title('FIR Filter Frequency Response')
  plt.xlabel('Frequency (MHz)')
  plt.ylabel('Magnitude (dB)')
  plt.grid(True)

  print("orig taps", fir_taps)
  taps_quantized = np.clip(np.round(fir_taps * 128), -128, 127).astype(np.int8)
  print("taps_quantized\n", taps_quantized)

#######################
# SIMULATE FREQS
#######################

n_cycles = 100  # Number of sine cycles per test

sim_freqs = np.linspace(1, 124, 200)
# sim_freqs = np.arange(62.5, 63.5, 1)

# Generate original signals with respective attenuations
orig_signals = [
    np.sin(2 * np.pi * freq_MHz_test * 1e6 * np.arange(0, (n_cycles / (freq_MHz_test * 1e6)), 1/high_sample_freq))
    for freq_MHz_test in sim_freqs
]
freq_MHz = np.array([1, 4, 8, 10, 16, 20, 25, 31.25, 62.5, 100, 200, 250])
attenuation_dB = np.array([2.0, 3.8, 5.3, 6.0, 7.6, 8.5, 9.5, 10.7, 15.4, 19.8, 29.0, 32.8])
interp_atten = interp1d(freq_MHz, attenuation_dB, kind="linear")
high_freq_signals = [
    simulate_channel(sig, cable_length=cable_length)
    for sig in orig_signals
]
signals = [
    downsample(sig)
    for sig in high_freq_signals
]
# Apply FIR filter
filtered_signals = np.array([signal.lfilter(fir_taps, 1.0, sig) for sig in signals])













# manual filter
def manual_filter(taps, sig):
  taps = np.asarray(taps)
  sig = np.asarray(sig)
  return np.convolve(sig, taps, mode='full')[:len(sig)]

def manual_filter_q(taps, sig):
  taps = np.asarray(taps)
  sig = np.asarray(sig)
  fir_taps_q = quantize(taps)
  sig_q = quantize(sig)
  return manual_filter(fir_taps_q, sig_q) / 128 / 128


def quantize(x):
  return np.clip(np.round(x * 128), -128, 127)

filtered_signals_q = np.array([manual_filter_q(fir_taps, sig) for sig in signals])

special_fsig = []
# special_fsig = [manual_filter_q(fir_taps, sig.repeat(300) + np.random.rand(len(sig)*300) * 0.1) for sig in [
#     np.array([-1, 1]),
#     np.array([-1, 0, 1, 0]),
#     np.array([-2, -1.4, 0, 1.4, 2, 1.4, 0, -1.4]) / 2,
#     np.array([-2, -1.8, -1.4, 0, 1.4, 1.8, 2, 1.8, 1.4, 0, -1.4, -1.8]) / 2,
# ]]

# Measure amplitudes after filtering (steady-state)
orig_amplitudes = np.array([np.max(np.abs(sig)) for sig in signals])
filtered_amplitudes = np.array([np.max(np.abs(sig[0:])) for sig in filtered_signals])
filtered_q_amplitudes = np.array([np.max(np.abs(sig[0:])) for sig in filtered_signals_q])
filtered_special_amplitudes = np.array([np.max(np.abs(sig[300:])) for sig in special_fsig])
# print(filtered_special_amplitudes)
# for i in range(len(filtered_signals)):
#     sig = filtered_signals[i]
#     filtered_amplitudes[i] = np.max(np.abs(sig[100:]))
# Convert amplitudes to dB
orig_amplitudes_dB = 20 * np.log10(orig_amplitudes)
filtered_amplitudes_dB = 20 * np.log10(filtered_amplitudes)
filtered_q_amplitudes_dB = 20 * np.log10(filtered_q_amplitudes)
filtered_sepcial_amplitudes_dB = 20 * np.log10(filtered_special_amplitudes)

# Plot original vs. filtered attenuations
# plt.figure(figsize=(10, 6))
ax = plt.subplot(122)
ax.plot(sim_freqs, orig_amplitudes_dB,  label='Original Attenuation (dB)')
ax.plot(sim_freqs, filtered_q_amplitudes_dB,  label='Filtered Attenuation (dB)')
for i in range(len(filtered_sepcial_amplitudes_dB)):
  ax.axhline(filtered_sepcial_amplitudes_dB[i], label=str(i), color=["red", "orange", "yellow", "green"][i])
plt.title('Original vs. Filtered Signal Attenuation')
plt.xlabel('Frequency (MHz)')
plt.ylabel('Magnitude (dB)')
plt.grid(True)
plt.legend()
plt.savefig("fir.png")
# plt.show()

