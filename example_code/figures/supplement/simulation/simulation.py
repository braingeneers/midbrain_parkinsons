# hippocampus.py
#
# Replicate the dentate gyrus network of Buchin et al. 2023, but using
# NEST and Izhikevich neurons for quick and easy simulation.
import os
import tempfile
from collections import namedtuple

os.environ["PYNEST_QUIET"] = "1"
import braingeneers.analysis as ba
import matplotlib.pyplot as plt
import nest
import numpy as np
from pynestml.frontend.pynestml_frontend import generate_nest_target
from tqdm import tqdm

nest.set_verbosity("M_WARNING")
with tempfile.TemporaryDirectory() as tempdir:
    generate_nest_target(
        "models/", tempdir, module_name="hippocampusmodule", logging_level="WARNING"
    )
nest.Install("hippocampusmodule")


# %%


def reset_nest(dt, seed):
    nest.ResetKernel()
    nest.local_num_threads = 12
    nest.resolution = dt
    nest.rng_seed = seed
    np.random.seed(seed)
    nest.CopyModel(
        "izh_cond_exp2syn",
        "granule_cell",
        params=dict(C_m=65.0, tau1_exc=1.5, tau2_exc=5.5, tau1_inh=0.26, tau2_inh=1.8),
    )
    nest.CopyModel(
        "izh_cond_exp2syn",
        "basket_cell",
        params=dict(C_m=150.0, tau1_exc=0.3, tau2_exc=0.6, tau1_inh=0.16, tau2_inh=1.8),
    )


Weights = namedtuple("Weights", "EE EI II IE XE FE XI FI")
default_weights = Weights(
    EE=0.13, EI=4.7, II=7.6, IE=1.6, XE=3.3, FE=1.0, XI=1.5, FI=1.0
)


def create_dentate_gyrus(
    N_granule=500,
    N_basket=6,
    N_perforant=50,
    N_sprout_pre=0,
    N_sprout_post=10,
    w=default_weights,
):
    """
    Create a dentate gyrus network for a NEST simulation, consisting of
    N_granule granule cells and N_basket basket cells, based on the
    dentate gyrus network of Buchin et al. 2023.

    The network consists of granule cells and basket cells, but they have
    been simplified from the original paper; instead of using a set of
    explicitly optimized neuron parameters, the neurons are Izhikevich
    neurons with the same parameter distributions as the excitatory and
    inhibitory cell types from the 2003 paper that introduced the model.

    Granule cells and basket cells are arranged in concentric
    half-circular arcs of radius 800μm and 750μm, and connectivity is all
    local. GCs connect to 50 of the 100 closest GCs as well as the
    3 closest BCs. BCs connect to 100/140 closest GCs as well as their
    own two closest neighbors. There are also Poisson inputs to each
    neuron.

    Instead of randomizing the number of synapses, we just use uniformly
    distributed weights, equal to a number of synapses times the weight
    per synapse from the original paper.
    """
    r_g, r_b = 800, 750

    theta_g = np.linspace(0, np.pi, N_granule)
    pos_g = nest.spatial.free(list(zip(r_g * np.sin(theta_g), r_g * np.cos(theta_g))))
    granule = nest.Create("granule_cell", positions=pos_g)
    variate = np.random.uniform(size=N_granule) ** 2
    granule.c = -65 + 15 * variate
    granule.d = 8 - 6 * variate
    granule.V_m = -70 + 5 * variate

    theta_b = np.linspace(0, np.pi, N_basket)
    pos_b = nest.spatial.free(list(zip(r_b * np.sin(theta_b), r_b * np.cos(theta_b))))
    basket = nest.Create("basket_cell", positions=pos_b)
    variate = np.random.uniform(size=N_basket)
    basket.a = 0.02 + 0.08 * variate
    basket.b = 0.25 - 0.05 * variate
    basket.V_m = -70 + 5 * variate

    def r_kth_nearest(radius, N, k):
        "The distance to the kth nearest neighbor on a half-circle."
        angle_per_step = np.pi / (N - 1)
        return 2 * radius * np.sin(k * angle_per_step / 2)

    # Connect the granule cells to each other with a circular mask that
    # only grabs the 100 nearest neighbors, and a fixed degree of 50.
    # Note that this means going for the 50th-nearest neighbor, as the
    # radius extends in both directions.
    conn_EE = dict(
        rule="fixed_outdegree",
        outdegree=47,
        mask=dict(circular=dict(radius=r_kth_nearest(r_g, N_granule, 50))),
        allow_autapses=False,
    )
    syn_EE = dict(
        synapse_model="static_synapse",
        delay=1.0,
        weight=w.EE * (2 + nest.random.uniform_int(4)),
    )
    nest.Connect(granule, granule, conn_EE, syn_EE)

    # Likewise for the BCs, but instead of including a fixed number of
    # neighbors, the radius is fixed to capture one neighbor in the
    # original formulation with only 6 BCs.
    conn_II = dict(
        rule="pairwise_bernoulli",
        p=1.0,
        mask=dict(circular=dict(radius=r_kth_nearest(r_b, 6, 1.1))),
        allow_autapses=False,
    )
    syn_II = dict(
        synapse_model="static_synapse",
        delay=1.0,
        weight=-w.II * (2 + nest.random.uniform_int(4)),
    )
    nest.Connect(basket, basket, conn_II, syn_II)

    # Incorporate mossy fiber sprouting by selecting some GCs to be presynaptic to mossy
    # cells (not modeled) which sprout problematic recurrent fibers. Since the recurrent
    # connection is polysynaptic, we add a 5ms delay.
    if N_sprout_pre > 0:
        sprouts = np.sort(np.random.choice(granule, N_sprout_pre, replace=False))
        conn_sprout = dict(
            rule="fixed_outdegree",
            outdegree=N_sprout_post,
            allow_autapses=False,
        )
        nest.Connect(sprouts, granule, conn_sprout, syn_EE)

    # For between-population connections, find the nearest point in the
    # other population by calculating the position of the nearest neuron
    # in the other layer and using that as the anchor for the mask.
    for b, θ in zip(basket, theta_b):
        θg = np.clip(θ, theta_g[69], theta_g[-70])
        mask = nest.CreateMask(
            "circular", dict(radius=r_kth_nearest(r_g, N_granule, 70))
        )
        neighbors = nest.SelectNodesByMask(
            granule, [r_g * np.sin(θg), r_g * np.cos(θg)], mask
        )
        nest.Connect(
            b,
            neighbors,
            dict(rule="fixed_outdegree", outdegree=100),
            dict(
                synapse_model="static_synapse",
                delay=1.0,
                weight=-w.IE * (2 + nest.random.uniform_int(4)),
            ),
        )

    for g, θ in zip(granule, theta_g):
        θb = np.clip(θ, theta_b[1], theta_b[-2])
        mask = nest.CreateMask("circular", dict(radius=r_kth_nearest(r_b, 6, 1.5)))
        neighbors = nest.SelectNodesByMask(
            basket, [r_b * np.sin(θb), r_b * np.cos(θb)], mask
        )
        nest.Connect(
            g,
            neighbors,
            dict(rule="pairwise_bernoulli", p=1.0),
            dict(
                synapse_model="static_synapse",
                delay=1.0,
                weight=w.EI * (2 + nest.random.uniform_int(4)),
            ),
        )

    # Finally create the Poisson inputs to all of this...
    noise = nest.Create("poisson_generator", params=dict(rate=15.0))
    for layer, wX in ((granule, w.XE), (basket, w.XI)):
        nest.Connect(
            noise,
            layer,
            "all_to_all",
            dict(
                synapse_model="static_synapse",
                weight=wX * (5 + nest.random.uniform_int(11)),
            ),
        )

    # The focal input is required to give the simulation a kick. It comes
    # through the "perforant path", which is supposed to trigger one
    # lamella of the hippocampus at a time, in this case just N_perforant
    # adjacent cells from the middle of the granule cell layer.
    if N_perforant > 0:
        focal = nest.Create(
            "poisson_generator", params=dict(rate=100.0, start=100.0, stop=200.0)
        )
        focal_granule = nest.Create("parrot_neuron", 200)
        nest.Connect(focal, focal_granule, "all_to_all")
        n = N_perforant // 2
        nest.Connect(
            focal_granule,
            granule[N_granule // 2 - n : N_granule // 2 + n],
            dict(rule="fixed_indegree", indegree=100),
            dict(
                synapse_model="static_synapse",
                weight=w.FE * (5 + nest.random.uniform_int(11)),
            ),
        )
        nest.Connect(
            focal,
            basket,
            "all_to_all",
            dict(
                synapse_model="static_synapse",
                weight=w.FI * (10 + nest.random.uniform_int(21)),
            ),
        )

    return granule, basket


def simulate(T, seed, p_sprout):
    reset_nest(dt=0.1, seed=seed)
    granule, basket = create_dentate_gyrus(
        N_granule=1000,
        N_basket=12,
        N_perforant=0,
        N_sprout_pre=int(1000 * p_sprout),
    )
    all = np.hstack((granule, basket))

    rec = nest.Create("spike_recorder")
    nest.Connect(all, rec)
    nest.Simulate(T)

    return ba.SpikeData.from_nest(rec, all, N=len(all), length=T)


def plot_sds(f, sds):
    """
    Plot the raster and firing rates, for a list of spike rasters on the given figure.
    """
    f.clear()
    axes = f.subplots(len(sds), 1)
    for ax, sd in zip(axes, sds):
        idces, times = sd.idces_times()
        ax.plot(times, idces, "k|", ms=0.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_xlim(0, sd.length)
        ax2 = ax.twinx()
        ax2.plot(sd.binned(1), c="purple", lw=0.75)
        ax2.set_yticks([0, 300])
        ax2.set_ylim(-25, 325)
        ax2.set_ylabel("Pop. Rate (kHz)")
    ticks = np.arange(sd.length // 1e3 + 1) * 1e3
    axes[-1].set_xticks(ticks, [f"{t/1e3:0.0f}" for t in ticks])
    axes[-1].set_xlabel("Time (sec)")
    return axes


# %%
# Simluation Visual Demonstration
# The particular value of T and the start in subtime below are chosen to cherry-pick a
# 3-second chunk whose firing rates align well with longer-term averages for
# illustrative purposes.
T = 11e3
sds = []
p_sprouts = [0.0, 0.05, 0.1, 0.3]
for p_sprout in p_sprouts:
    sd = simulate(T, 1234, p_sprout).subtime(8e3, ...)
    idces, times = sd.idces_times()
    print(f"For {p_sprout = :.0%}, FR was {sd.rates('Hz').mean():.2f} Hz. ")
    sds.append(sd)

f = plt.figure("Varying Sprouting", figsize=(6.4, 6.4))
axes = plot_sds(f, sds)
plt.show()

for ax, sprout in zip(axes, p_sprouts):
    ax.set_ylabel(f"{sprout*100:.0f}\\% Sprouting")

f.savefig("sprouting-simulation.png", dpi=600)


# %%
# Actual Simulation Results
# Here we use several simulations with different random seeds to find the actual
# firing rate and burstiness of the population as a function of mossy fiber sprouting
# probability.
T = 30e3
sdses = []
p_sprouts = np.linspace(0.0, 0.3, num=11)
seeds = 25

with tqdm(total=len(p_sprouts) * seeds) as pbar:
    for p_sprout in p_sprouts:
        sdses.append([])
        for seed in range(seeds):
            sdses[-1].append(simulate(T, 1234 + seed, p_sprout).subtime(100, ...))
            pbar.update()


fr = np.array([[sd.rates("kHz").sum() for sd in sds] for sds in sdses])
bi = np.array([[sd.burstiness_index() for sd in sds] for sds in sdses])

# Calculate "burst rate," the number of burst events per second over the recording.
# Non-burst firing rates are about 50 Hz and the default bin size is 40 ms, so this
# cutoff means 15% of neurons are firing in a single bin and is about triple the usual.
# Avoid using the standard deviation because it's affected a lot by the number of
# bursts. We don't see two bins satisfying this condition in a row in practice, so be
# lazy and don't bother checking for that special case.
br = np.array([[np.mean(sd.binned() > 6000) / 0.04 for sd in sds] for sds in sdses])

f = plt.figure("Metrics")
axes = afr, abr, abi = f.subplots(3, 1)

frm, frs = fr.mean(1), fr.std(1)
afr.plot(p_sprouts, frm)
afr.fill_between(p_sprouts, frm - frs, frm + frs, alpha=0.25)

brm, brs = br.mean(1), br.std(1)
abr.plot(p_sprouts, brm)
abr.fill_between(p_sprouts, np.maximum(0, brm - brs), brm + brs, alpha=0.25)

bim, bis = bi.mean(1), bi.std(1)
abi.plot(p_sprouts, bim)
abi.fill_between(p_sprouts, bim - bis, bim + bis, alpha=0.25)

afr.set_ylabel("Population Rate (kHz)")
afr.set_xticks([])
abr.set_ylabel("Seizure Frequency (Hz)")
abr.set_xticks([])
abi.set_ylabel("Burstiness Index")
abi.set_xlabel("Sprouting Percentage")
abi.xaxis.set_major_formatter(
    plt.matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
)

f.savefig("metrics.png", dpi=600)
