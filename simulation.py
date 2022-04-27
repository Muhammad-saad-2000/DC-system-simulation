
import numpy as np
import matplotlib.pyplot as plt
from library import AWGN, BinarySource, binarycode_to_signal, receive_filter, Q, decision_maker, sampler
# simulation parameters
n = 100000
step = 0.05
t = np.arange(0, n, step)

# generate the binary symbols
bitstream = BinarySource(n)


# generate the binary signal
signal = binarycode_to_signal(bitstream, step)


def error_rate(sigma_noise, filter_num):
    # generate the noise
    noise = AWGN(len(signal), sigma_noise)

    # add the noise to the signal
    signal_noise = signal+noise

    # apply the filter to the signal
    signal_noise_filter = receive_filter(signal_noise,filter_num, step)

    # sample the filtered signal
    sampling_period = int(1/step)
    samples = sampler(sampling_period, signal_noise_filter,n)

    # decode the samples
    reconstructed_bitstram = decision_maker(samples, 0)
    # results
    return np.sum(bitstream != reconstructed_bitstram)/len(bitstream)



# plot E/No vs error rate (first filter)
x=np.arange(-10, 20, 1)
sigma_noise=np.sqrt(2/(10**(x/10)))

error_rate_1 = np.zeros(len(sigma_noise))
theorey_error_rate_1=np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_1[i] = error_rate(sigma_noise[i], 1)
    theorey_error_rate_1[i] = Q(1/sigma_noise[i])

error_rate_2 = np.zeros(len(sigma_noise))
theorey_error_rate_2=np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_2[i] = error_rate(sigma_noise[i], 2)
    theorey_error_rate_2[i] = Q(1/sigma_noise[i])

error_rate_3 = np.zeros(len(sigma_noise))
theorey_error_rate_3 = np.zeros(len(sigma_noise))
for i in range(len(sigma_noise)):
    error_rate_3[i] = error_rate(sigma_noise[i], 3)
    theorey_error_rate_3[i] = Q(np.sqrt(3)/2*1/sigma_noise[i])

plt.semilogy(x, error_rate_1, 'r')
plt.semilogy(x, error_rate_2, 'g')
plt.semilogy(x, error_rate_3, 'b')
plt.semilogy(x, theorey_error_rate_1, 'c')
plt.semilogy(x, theorey_error_rate_2, 'm--')
plt.semilogy(x, theorey_error_rate_3, 'y')

plt.xlabel('E/No (dB)')
plt.ylabel('BER')
plt.title('E/No vs BER')
plt.legend(['First filter', 'Second filter', 'Third filter', 'Theorey 1', 'Theorey 2', 'Theorey 3'])
plt.ylim([1/n*100, 1])
plt.savefig('./BER.png')