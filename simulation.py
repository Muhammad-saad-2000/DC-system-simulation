
import numpy as np
import matplotlib.pyplot as plt
from library import AWGN, BinarySource, binarycode_to_signal, receive_filter, Q, decision_maker, sampler
# simulation parameters
n = 100000                                                      # bitstream length
step = 0.05                                                     # 20 Samples per pulse of duration 1
T=1                                                             # pulse period

## Generate a random bitstream and cascade it through the communication system:

# generate the binary symbols (1)
input_bitstream = BinarySource(n)


# generate the binary signal (2)
g_t = binarycode_to_signal(input_bitstream, step)


# The following function simulates the channel and receiver for different noise variances/filter
# it then returns the probability of error associated with the transmission

def compute_BER(σₙ, f):                  # σₙ is the noise's variance and f is the filter's E_Nₒ. (1 or 2 or 3)
    # generate the noise
    w_t = AWGN(len(g_t), σₙ)

    # add the Nₒise to the signal (3)
    s_t = g_t + w_t

    # apply the filter to the signal (4)
    y_t = receive_filter(s_t,f, step)

    # sample the filtered signal (5)
    sampling_period = int(T/step)
    y_iT = sampler(sampling_period, y_t,n)                   # y_T has all samples 

    # decode the samples (6)
    λ = 0                                                   # holds for all 3 cases.
    bitstream_output = decision_maker(y_iT, λ)

    # compare the resulting bitstream with the original to compute the probability of error
    return np.sum(input_bitstream != bitstream_output)/len(input_bitstream)



# plot BER VS. E/No for each filter
E_Nₒ=np.arange(-10, 20, 1)                      # E_Nₒ range
Nₒ = 1/(10**(E_Nₒ/10))
σₙ = np.sqrt(Nₒ/2)                              # the corresponding range of sigma.



# Filter 1:
filter1_BER, filter1_BER_th = np.zeros(len(σₙ)), np.zeros(len(σₙ))
for i in range(len(σₙ)):
    filter1_BER[i] = compute_BER(σₙ[i], 1)
    filter1_BER_th[i] = Q(1/σₙ[i])

# Filter 2:
filter2_BER, filter2_BER_th = np.zeros(len(σₙ)), np.zeros(len(σₙ))
for i in range(len(σₙ)):
    filter2_BER[i] = compute_BER(σₙ[i], 2)
    filter2_BER_th[i] = Q(1/σₙ[i])

# Filter 3:
filter3_BER, filter3_BER_th = np.zeros(len(σₙ)), np.zeros(len(σₙ))
for i in range(len(σₙ)):
    filter3_BER[i] = compute_BER(σₙ[i], 3)
    filter3_BER_th[i] = Q(np.sqrt(3)/2*1/σₙ[i])

plt.semilogy(E_Nₒ, filter1_BER, 'r')
plt.semilogy(E_Nₒ, filter2_BER, 'g.')
plt.semilogy(E_Nₒ, filter3_BER, 'b')
plt.semilogy(E_Nₒ, filter1_BER_th, 'c')
plt.semilogy(E_Nₒ, filter2_BER_th, 'm--')
plt.semilogy(E_Nₒ, filter3_BER_th, 'y')

plt.xlabel('E/Nₒ (db)')
plt.ylabel('BER (log-scale)')
plt.title(' BER VS. E/Nₒ')
plt.legend(['Matched Filter (1)', 'No Filter (2)', 'Linear Filter (3)', 'Theory 1', 'Theory 2', 'Theory 3'])
plt.ylim([1/(100*n), 1])
plt.savefig('./BER.png')


# Why is the BER decreasing?
""" 
The BER is decreasing as a function of E/No as in the plot. We can justify this in different ways:

1 - As E/No increases (here E is constant) No decreases and thus σₙ which means the added AWGN 
involves less variations (corresponds to a thinner Gaussian distribution) and thus any noise added
corresponds to small values close to zero which do not affect the signal that much (hence BER decreases)

2 - As in the theoritical expression and knowing that Q is a decreasing function, its clear that Q(a * sqrt(E/No))
for all the cases above. Hence, BER is a decreasing function of E/No (noting that sqrt is an increasing function)

"""

#Which case has the lowest BER and why?
""" 
The matched filter case is the one with lowest BER since it uses a filter matched to the pulse
to minimize the probability of error (as we have proven in the lecture.) To accompolish this it
equivalently maximizes the peak pulse SNR at the sampling instant.

Note that in the theoritical case using a filter or not yields the same expression due to our assumptions on variance and PSD.
"""
 