import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import signal
import math
from scipy.sparse import csr_array


def ccg(bt1, bt2, ccg_win=[-10, 10], t_lags_shift=0):
    left_edge, right_edge = np.subtract(ccg_win, t_lags_shift)
    lags = np.arange(ccg_win[0], ccg_win[1] + 1)
    pad_width = min(max(-left_edge, 0), max(right_edge, 0))
    bt2_pad = np.pad(bt2, pad_width=pad_width, mode='constant')
    cross_corr = signal.fftconvolve(bt2_pad, bt1[::-1], mode="valid")
    return np.round(cross_corr), lags


def p_fast(n, lambda_):
    """
    A poisson estimation of the probability of observing n or more events
    """
    ## take log to make sure the factorial does not overflow
    # add poisson_var when x = 0, 1, take log after calculation to avoid log(0)
    if n > 1:
        poisson_01 = [np.exp(-lambda_)*lambda_**x/math.factorial(x) for x in [0, 1]]
        poisson_res = [np.exp(-lambda_ + x*math.log(lambda_) - math.log(math.factorial(x))) for x in range(2, n)]
        poisson_var = poisson_01 + poisson_res
    else:
        poisson_var = [np.exp(-lambda_)*lambda_**x/math.factorial(x) for x in range(n)]
    continuity_correction = np.exp((math.log(0.5) - lambda_ + n*math.log(lambda_)) - math.log(math.factorial(n)))
    return 1 - np.sum(poisson_var) - continuity_correction


def sparse_train(spike_train: list, bin_size=0.001):
    """
    create a sparse matrix for the input spike trains
    with a given bin size
    """
    num = len(spike_train)
    length = np.max([t[-1] for t in spike_train])
    indices = np.hstack([np.ceil(ts / bin_size) - 1
                         for ts in spike_train]).astype(int)
    units = np.hstack([0] + [len(ts) for ts in spike_train])
    indptr = np.cumsum(units)
    values = np.ones_like(indices)
    length = int(np.ceil(length / bin_size))
    np.clip(indices, 0, length - 1, out=indices)
    st = csr_array((values, indices, indptr),
                   shape=(num, length)).toarray()
    return st


def functional_pair(spike_data, binary_bin_size=0.001, ccg_win=[-50, 50],
                    f22, func_prob=0.00001, verbose=True):
    """
    Note: the input spike times are in seconds!
    """
    train = spike_data["train"]
    neuron_data = spike_data["neuron_data"]
    unit_count = len(train)
    sparse_train = sparse_train(train, bin_size=binary_bin_size)
    
    if unit_count < 2:
        return (0, 0), {}
    for i in range(unit_count-1):
        for j in range(i+1, unit_count):
            counts, lags = ccg(sparse_train[i],
                            sparse_train[j],
                            ccg_win=ccg_win)
            max_ind = np.argmax(counts)
            latency = lags[max_ind]
            if latency >= -func_latency and latency <= func_latency:
                if max_ind != np.diff(ccg_win)//2:
                    ccg_smth = gaussian_filter1d(counts, sigma=10)   
                    # ccg_smth = utils.hollow_gaussian_filter(counts, sigma=10) 
                    lambda_slow_peak = ccg_smth[max_ind]
                    ccg_peak = int(counts[max_ind])
                    # estimate p_fast
                    p_fast_est = p_fast(ccg_peak, lambda_slow_peak)
                    if verbose:
                        print(f"Putative functional pair {i}, {j}")
                        print(f"Cross correlation latency: {latency} ms, counts: {ccg_peak}, smoothed counts: {lambda_slow_peak}")
                        print(f"p_fast: {p_fast_est}")
                    if p_fast_est <= func_prob:    # test with func_prob = 10e-5
                        yield (i, j), {"latency": latency,
                                    "p_fast": p_fast_est,
                                    "ccg": counts,
                                    "lags": lags,
                                    "ccg_smth": ccg_smth}


# test
func_pairs = {}
for (i, j), value in functional_pair():
    lags, counts, ccg_smth = value["lags"], value["ccg"], value["ccg_smth"]
    latency, p_fast = value["latency"], value["p_fast"]
    ccg_peak = np.max(counts)
    fig, axs = plt.subplots(figsize=(5, 3), tight_layout=True)
    plt.suptitle(f"ccg_unit_{i}_{j}")
    axs.bar(lags, counts, label=f"p_fast={p_fast:.3g} \n count={ccg_peak}")
    axs.plot(lags, ccg_smth, color="red")
    axs.axvline(0, linestyle="--", color="black")
    axs.scatter(latency, ccg_peak, color="red", marker="x", label=f"latency={latency}")
    axs.legend()
    axs.set_xlabel("Lags (ms)")
    axs.set_ylabel("Counts")
    plt.savefig(f"ccg_unit_{i}_{j}.png")
    plt.close()
    func_pairs[(i, j)] = value

np.savez("func_pairs.npz", func_pairs=func_pairs)