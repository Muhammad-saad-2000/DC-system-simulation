
import numpy as np
import matplotlib.pyplot as plt

# simulation parameters
n = 10000
step = 0.01
t = np.arange(0, n, step)


def AWGN(n, sigma):
    return np.random.normal(0, sigma, n)


def BinarySource(n):
    return np.random.randint(0, 2, n)


def binarycode_to_signal(bitstream, step):
    T = 1  # assume the pulse period is 1
    A = 1  # assume the amplitude is 1
    pulse = np.ones(int(T/step))
    pulse = pulse*A
    signal = np.zeros(len(bitstream)*len(pulse))
    # Polar nonreturn to zero
    for i in range(len(bitstream)):
        if bitstream[i] == 1:
            signal[i*len(pulse):(i+1)*len(pulse)] = 1*pulse
        else:
            signal[i*len(pulse):(i+1)*len(pulse)] = -1*pulse
    return signal


# generate the binary symbols
bitstream = BinarySource(n)


# generate the binary signal
signal = binarycode_to_signal(bitstream, step)


# construct the resive filter (We have 3 filters to choose from)
filter1 = np.ones(int(1/step))
filter2 = np.ones(1)
filter3 = np.sqrt(3)*np.arange(0, 1, step)


def error_rate(sigma_noise, filter_num):
    # generate the noise
    noise = AWGN(len(signal), sigma_noise)

    # add the noise to the signal
    signal_noise = signal+noise

    # select the filter
    filter = filter1 if filter_num == 1 else filter2 if filter_num == 2 else filter3
    filter = np.concatenate((filter, np.zeros(int(1/step)-len(filter))))

    # apply the filter to the signal
    signal_noise_filter = np.convolve(signal_noise, filter)

    # sample the filtered signal
    sampling_period = int(1/step)
    samples = np.zeros(n)
    for i in range(len(samples)):
        samples[i] = signal_noise_filter[(i+1)*sampling_period]

    # decode the samples
    reconstructed_bitstram = (samples > 0)*1

    # results
    return np.sum(bitstream != reconstructed_bitstram)/len(bitstream)



# plot E/No vs error rate (first filter)
x=np.arange(-10, 20, 1)
sigma_noise=2/(10**(x/10))

error_rate_1 = np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_1[i] = error_rate(sigma_noise[i], 1)

error_rate_2 = np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_2[i] = error_rate(sigma_noise[i], 2)

error_rate_3 = np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_3[i] = error_rate(sigma_noise[i], 3)


plt.plot(x, error_rate_1, 'r')
plt.plot(x, error_rate_2, 'g')
plt.plot(x, error_rate_3, 'b')

plt.xlabel('E/No (dB)')
plt.ylabel('BER')
plt.title('E/No vs BER')
plt.legend(['First filter', 'Second filter', 'Third filter'])
plt.show() 