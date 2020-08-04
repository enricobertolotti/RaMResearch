import numpy as np
from scipy import optimize


def sin_func(x, amp, freq, phase, offset):
    return (amp * np.sin((freq * x) + phase)) + offset


def sin_fitting(y_data):
    x_data = range(len(y_data))

    offset_estimate = np.mean(y_data)
    amp_estimate = np.max(y_data) - np.min(y_data)
    freq_estimate = 4 * np.pi / 360
    phase_estimate = 0
    initial_guess = [amp_estimate, freq_estimate, phase_estimate, offset_estimate]
    params, params_covariance = optimize.curve_fit(sin_func, x_data, y_data,
                                                   p0=initial_guess)
    print(params)
    return params

# def sin_fitting(data):
#     N = len(data)                       # number of data points
#     t = np.linspace(0, 8 * np.pi, N)    # Linear space of points for sin curve (4 cycles)
#
#
# N = 1000 # number of data points
# t = np.linspace(0, 4*np.pi, N)
# f = 1.15247 # Optional!! Advised not to use
# data = 3.0*np.sin(f*t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise
#
# guess_mean = np.mean(data)
# guess_std = 3*np.std(data)/(2**0.5)/(2**0.5)
# guess_phase = 0
# guess_freq = 1
# guess_amp = 1
#
# # we'll use this to plot our first estimate. This might already be good enough for you
# data_first_guess = guess_std*np.sin(t+guess_phase) + guess_mean
#
# # Define the function to optimize, in this case, we want to minimize the difference
# # between the actual data and our "guessed" parameters
# optimize_func = lambda x: x[0]*np.sin(x[1]*t+x[2]) + x[3] - data
# est_amp, est_freq, est_phase, est_mean = leastsq(optimize_func, [guess_amp, guess_freq, guess_phase, guess_mean])[0]
#
# # recreate the fitted curve using the optimized parameters
# data_fit = est_amp*np.sin(est_freq*t+est_phase) + est_mean
#
# # recreate the fitted curve using the optimized parameters
#
# fine_t = np.arange(0,max(t),0.1)
# data_fit=est_amp*np.sin(est_freq*fine_t+est_phase)+est_mean
#
# plt.plot(t, data, '.')
# plt.plot(t, data_first_guess, label='first guess')
# plt.plot(fine_t, data_fit, label='after fitting')
# plt.legend()
# plt.show()