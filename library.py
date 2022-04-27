# %%
import numpy as np
from math import erfc, sqrt

# %% [markdown]
# # Transmitter
# Involves a binary source generating random 0s and 1s and a pulse shape filter that fits pulses on the bitstream.

# %%
def BinarySource(n):
	return np.random.randint(0,2,n)

# %%
def binarycode_to_signal(bitstream, step):
    T = 1  # assume the pulse period is 1
    A = 1  # assume the amplitude is 1
    pulse = np.ones(int(T/step))
    pulse = pulse*A
    signal = np.zeros(len(bitstream)*len(pulse))
    # Polar nonreturn to zero
    for i in range(len(bitstream)):
        if bitstream[i] == 1:
            signal[i*len(pulse):(i+1)*len(pulse)] = 1*pulse    # take the shape of pulse along the symbol's interval
        else:
            signal[i*len(pulse):(i+1)*len(pulse)] = -1*pulse
    return signal


# %% [markdown]
# # Channel
# Only adds additive white Gaussian noise on the signal

# %%
def AWGN(n, sigma):
	return np.random.normal(0,sigma,n)

# %% [markdown]
# # Receiver
# Receive signal from the channel then pass it by receive felter, sampler and decision maker.

# %%
def receive_filter(signal_noise, filter_num, step):
   filter_num-=1
   filters = [np.ones(int(1/step)), np.ones(1), np.sqrt(3)*np.arange(0, 1, step)]
   filter = filters[filter_num]
   filter = np.concatenate((filter, np.zeros(int(1/step)-len(filter))))
   signal_noise_filter=np.convolve(signal_noise, filter)
   if (filter_num==0 or filter_num==2):
      signal_noise_filter*step
   return signal_noise_filter



# %%
def sampler(sampling_period, signal_noise_filtered, n=10):
   samples = np.zeros(n)
   for i in range(len(samples)):
      samples[i] = signal_noise_filtered[sampling_period-1+i*sampling_period]
   return samples

# %%
Q = lambda x : 0.5 * erfc(x/sqrt(2))
def decision_maker(samples, λ):
   return (samples>λ)*1