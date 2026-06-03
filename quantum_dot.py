"""
Spinful quantum-dot simulation for finite-frequency response duality and R-TUR.

Model
-----
A channel-resolved four-state single-level quantum dot in the sequential
-tunneling regime.  The dot states are

    0 : empty
    u : occupied by spin up
    d : occupied by spin down
    2 : double occupation

The dot is coupled to two electronic reservoirs L and R.  Each lead defines a
separate jump channel.  For each channel-resolved undirected edge e, the plus
orientation is chosen as "electron enters the dot from the lead":

    r_e^+ = Gamma_e f_alpha(Delta E_e),
    r_e^- = Gamma_e [1 - f_alpha(Delta E_e)],

where f_alpha(E)=1/[1+exp(beta(E-mu_alpha))].  The decomposed edge parameters
are equivalently

    r_e^+ = exp(b_e + f_e/2),
    r_e^- = exp(b_e - f_e/2),

so that b_e=0.5 log(r_e^+ r_e^-) and f_e=log(r_e^+/r_e^-).

Numerical measurements
----------------------
The script samples steady-state random trajectories on a discrete time grid.
From the same trajectories it estimates

    S_Q(omega),
    R_b_e(omega) = FT_{t>=0} Cov(Qdot(t), Lambdadot_b_e(0)),
    R_f_e(omega) = FT_{t>=0} Cov(Qdot(t), Lambdadot_f_e(0)).

The observable Q is the net particle current into the right lead.  The default
perturbed edge is a right-lead channel, so the response is cleanly visible.

Figures
-------
1. R_b and R_f as functions of frequency.
2. R_b/R_f compared with 2 j_e/a_e.
3. R-TUR plot: SNR_b=|R_b|^2/S_Q compared with sigma_pseudo/2 and sigma/2.

Outputs are saved to ./data and ./figures next to this script.

Run examples
------------
Quick test:
    N_TRAJ=2000 N_STEPS=1024 BATCH_SIZE=500 python spinful_quantum_dot_frequency_response.py

Better data:
    N_TRAJ=100000 N_STEPS=4096 BATCH_SIZE=1000 FORCE_RECOMPUTE=1 python spinful_quantum_dot_frequency_response.py
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import numba as nb
import matplotlib.pyplot as plt


# =============================================================================
# Plot style
# =============================================================================

if shutil.which("latex") is not None:
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 10,
        "text.latex.preamble": "\n".join([
            r"\usepackage{amsmath}",
            r"\usepackage{amsfonts}",
        ]),
    })
else:
    plt.rcParams.update({"text.usetex": False, "font.family": "serif", "font.size": 10})


# =============================================================================
# User-level knobs
# =============================================================================

DT = float(os.environ.get("DT", "2.0e-2"))
N_STEPS = int(os.environ.get("N_STEPS", "4096"))
N_TRAJ = int(os.environ.get("N_TRAJ", "50000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "5000"))
SEED = int(os.environ.get("SEED", "111"))
FORCE_RECOMPUTE = bool(int(os.environ.get("FORCE_RECOMPUTE", "0")))

# Default: edge 3 = right-lead channel 0 <-> down, plus direction R -> dot.
TARGET_EDGE = int(os.environ.get("TARGET_EDGE", "3"))
MAX_PLOT_FREQ = float(os.environ.get("MAX_PLOT_FREQ", "30.0"))
MIN_RATIO_DENOM = float(os.environ.get("MIN_RATIO_DENOM", "1.0e-9"))


# =============================================================================
# Quantum-dot model
# =============================================================================

@dataclass
class DotModel:
    beta: float
    eps: float
    U: float
    delta_z: float
    mu_L: float
    mu_R: float
    E: np.ndarray
    R: np.ndarray
    pi: np.ndarray
    eigvals: np.ndarray
    edge_tail: np.ndarray
    edge_head: np.ndarray
    edge_lead_is_R: np.ndarray
    edge_label: list[str]
    r_plus: np.ndarray
    r_minus: np.ndarray
    b: np.ndarray
    f: np.ndarray
    a: np.ndarray
    j: np.ndarray
    activity: float
    sigma: float
    sigma_pseudo: float
    q_mean_right: float


def fermi(beta: float, energy_minus_mu: float) -> float:
    """Fermi function 1/[1+exp(beta*(E-mu))] with overflow protection."""
    x = beta * energy_minus_mu
    if x > 60.0:
        return 0.0
    if x < -60.0:
        return 1.0
    return 1.0 / (1.0 + np.exp(x))


def steady_state_from_eig(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Right null eigenvector of the column-convention generator dp/dt=R p."""
    eigvals, eigvecs = np.linalg.eig(R)
    idx = np.argmin(np.abs(eigvals))
    pi = np.real(eigvecs[:, idx])
    if np.sum(pi) < 0:
        pi = -pi
    pi[np.abs(pi) < 1e-14] = 0.0
    if np.any(pi < -1e-10):
        raise RuntimeError("Steady-state eigenvector has sizeable negative components.")
    pi = np.maximum(pi, 0.0)
    pi /= np.sum(pi)
    return pi, eigvals


def build_spinful_dot_model() -> DotModel:
    """Build the four-state spinful quantum dot with the parameters discussed above."""
    beta = 1.0
    eps = 0.0
    U = 2.0
    delta_z = 0.6
    mu_L = 2.5
    mu_R = -1.5

    # State order: 0, up, down, double.
    E0 = 0.0
    E_up = eps - delta_z / 2.0
    E_down = eps + delta_z / 2.0
    E2 = 2.0 * eps + U
    E = np.array([E0, E_up, E_down, E2], dtype=np.float64)

    # Tunnel couplings Gamma_{alpha,spin}^{(n)}, where n=0 means adding the first
    # electron and n=1 means adding the second electron.
    Gamma = {
        ("L", "up", 0): 1.00,
        ("L", "down", 0): 0.70,
        ("R", "up", 0): 0.60,
        ("R", "down", 0): 1.10,
        ("L", "up", 1): 0.80,
        ("L", "down", 1): 1.20,
        ("R", "up", 1): 1.00,
        ("R", "down", 1): 0.75,
    }
    mu = {"L": mu_L, "R": mu_R}

    # Undirected channel-resolved edges.  The plus orientation is always adding
    # one electron to the dot from the specified lead.
    # Tuple: (label, lead, spin_added, occupancy_before, tail_state, head_state, Delta E)
    edge_specs = []
    for lead in ("L", "R"):
        edge_specs.append((f"{lead}:0\\leftrightarrow\\uparrow", lead, "up", 0, 0, 1, E_up - E0))
    for lead in ("L", "R"):
        edge_specs.append((f"{lead}:0\\leftrightarrow\\downarrow", lead, "down", 0, 0, 2, E_down - E0))
    for lead in ("L", "R"):
        edge_specs.append((f"{lead}:\\uparrow\\leftrightarrow2", lead, "down", 1, 1, 3, E2 - E_up))
    for lead in ("L", "R"):
        edge_specs.append((f"{lead}:\\downarrow\\leftrightarrow2", lead, "up", 1, 2, 3, E2 - E_down))

    n_edges = len(edge_specs)
    edge_tail = np.zeros(n_edges, dtype=np.int64)
    edge_head = np.zeros(n_edges, dtype=np.int64)
    edge_lead_is_R = np.zeros(n_edges, dtype=np.int64)
    edge_label: list[str] = []
    r_plus = np.zeros(n_edges, dtype=np.float64)
    r_minus = np.zeros(n_edges, dtype=np.float64)

    R = np.zeros((4, 4), dtype=np.float64)
    for e, (label, lead, spin_added, occ_before, tail, head, dE) in enumerate(edge_specs):
        G = Gamma[(lead, spin_added, occ_before)]
        ff = fermi(beta, dE - mu[lead])
        rp = G * ff
        rm = G * (1.0 - ff)

        edge_tail[e] = tail
        edge_head[e] = head
        edge_lead_is_R[e] = 1 if lead == "R" else 0
        edge_label.append(label)
        r_plus[e] = rp
        r_minus[e] = rm

        # Column convention: R[i,j] is rate j -> i.
        R[head, tail] += rp
        R[tail, head] += rm

    for col in range(4):
        R[col, col] = -np.sum(R[:, col])

    pi, eigvals = steady_state_from_eig(R)

    a = np.zeros(n_edges, dtype=np.float64)
    j = np.zeros(n_edges, dtype=np.float64)
    sigma = 0.0
    sigma_pseudo = 0.0
    q_mean_right = 0.0
    for e in range(n_edges):
        x = r_plus[e] * pi[edge_tail[e]]
        y = r_minus[e] * pi[edge_head[e]]
        a[e] = x + y
        j[e] = x - y
        sigma += (x - y) * np.log(x / y)
        sigma_pseudo += 2.0 * (x - y) ** 2 / (x + y)
        # Q is particle current into the right lead.  Since plus means lead -> dot,
        # the right-lead current increment is -1 for plus and +1 for minus.
        if edge_lead_is_R[e] == 1:
            q_mean_right += -j[e]

    b = 0.5 * np.log(r_plus * r_minus)
    f = np.log(r_plus / r_minus)
    activity = float(np.sum(a))

    return DotModel(
        beta=beta,
        eps=eps,
        U=U,
        delta_z=delta_z,
        mu_L=mu_L,
        mu_R=mu_R,
        E=E,
        R=R,
        pi=pi,
        eigvals=eigvals,
        edge_tail=edge_tail,
        edge_head=edge_head,
        edge_lead_is_R=edge_lead_is_R,
        edge_label=edge_label,
        r_plus=r_plus,
        r_minus=r_minus,
        b=b,
        f=f,
        a=a,
        j=j,
        activity=float(activity),
        sigma=float(sigma),
        sigma_pseudo=float(sigma_pseudo),
        q_mean_right=float(q_mean_right),
    )


def build_directed_events(model: DotModel):
    """Directed events for simulation.  Plus event has sign +1; minus has sign -1."""
    n_edges = len(model.edge_tail)
    n_events = 2 * n_edges
    tails = np.zeros(n_events, dtype=np.int64)
    heads = np.zeros(n_events, dtype=np.int64)
    edge_ids = np.zeros(n_events, dtype=np.int64)
    dir_signs = np.zeros(n_events, dtype=np.int64)
    lead_is_R = np.zeros(n_events, dtype=np.int64)
    rates = np.zeros(n_events, dtype=np.float64)

    for e in range(n_edges):
        # Plus: tail -> head.
        ev = 2 * e
        tails[ev] = model.edge_tail[e]
        heads[ev] = model.edge_head[e]
        edge_ids[ev] = e
        dir_signs[ev] = 1
        lead_is_R[ev] = model.edge_lead_is_R[e]
        rates[ev] = model.r_plus[e]

        # Minus: head -> tail.
        ev = 2 * e + 1
        tails[ev] = model.edge_head[e]
        heads[ev] = model.edge_tail[e]
        edge_ids[ev] = e
        dir_signs[ev] = -1
        lead_is_R[ev] = model.edge_lead_is_R[e]
        rates[ev] = model.r_minus[e]

    n_states = model.R.shape[0]
    max_out = 0
    for s in range(n_states):
        max_out = max(max_out, int(np.sum(tails == s)))
    out_counts = np.zeros(n_states, dtype=np.int64)
    out_events = -np.ones((n_states, max_out), dtype=np.int64)
    for ev in range(n_events):
        s = tails[ev]
        k = out_counts[s]
        out_events[s, k] = ev
        out_counts[s] += 1

    escape = np.zeros(n_states, dtype=np.float64)
    for s in range(n_states):
        for k in range(out_counts[s]):
            escape[s] += rates[out_events[s, k]]

    return tails, heads, edge_ids, dir_signs, lead_is_R, rates, out_events, out_counts, escape


# =============================================================================
# Simulation and FFT estimators
# =============================================================================

@nb.njit
def _sample_initial_state(cum_pi: np.ndarray) -> int:
    u = np.random.random()
    for s in range(cum_pi.size):
        if u < cum_pi[s]:
            return s
    return cum_pi.size - 1


@nb.njit
def simulate_batch_rates(
    n_traj: int,
    n_steps: int,
    dt: float,
    seed: int,
    cum_pi: np.ndarray,
    heads: np.ndarray,
    edge_ids: np.ndarray,
    dir_signs: np.ndarray,
    lead_is_R: np.ndarray,
    rates: np.ndarray,
    out_events: np.ndarray,
    out_counts: np.ndarray,
    escape: np.ndarray,
    target_edge: int,
    target_tail: int,
    target_head: int,
    target_r_plus: float,
    target_r_minus: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample binned trajectories and return Qdot, Lambdadot_b, Lambdadot_f.

    Q is the particle current into the right lead.  For right-lead jumps, plus
    direction has increment -1 and minus direction has increment +1.
    """
    np.random.seed(seed)
    Q = np.zeros((n_traj, n_steps), dtype=np.float64)
    Lb = np.zeros((n_traj, n_steps), dtype=np.float64)
    Lf = np.zeros((n_traj, n_steps), dtype=np.float64)

    inv_dt = 1.0 / dt

    for tr in range(n_traj):
        state = _sample_initial_state(cum_pi)

        for step in range(n_steps):
            old_state = state

            # Deterministic waiting-time parts of the target-edge conjugate observables.
            lb = 0.0
            lf = 0.0
            if old_state == target_tail:
                lb -= target_r_plus
                lf -= 0.5 * target_r_plus
            if old_state == target_head:
                lb -= target_r_minus
                lf += 0.5 * target_r_minus

            q_inc = 0.0
            esc = escape[old_state]
            u = np.random.random()
            if u < esc * dt:
                # Choose one outgoing event conditional on a jump in this bin.
                v = np.random.random() * esc
                acc = 0.0
                chosen = out_events[old_state, 0]
                for kk in range(out_counts[old_state]):
                    ev = out_events[old_state, kk]
                    acc += rates[ev]
                    if v <= acc:
                        chosen = ev
                        break

                e = edge_ids[chosen]
                sgn = dir_signs[chosen]

                if lead_is_R[chosen] == 1:
                    q_inc = -float(sgn)

                if e == target_edge:
                    if sgn == 1:
                        lb += inv_dt
                        lf += 0.5 * inv_dt
                    else:
                        lb += inv_dt
                        lf -= 0.5 * inv_dt

                state = heads[chosen]

            Q[tr, step] = q_inc * inv_dt
            Lb[tr, step] = lb
            Lf[tr, step] = lf

    return Q, Lb, Lf


def accumulate_fft_estimators(
    model: DotModel,
    n_traj: int,
    n_steps: int,
    dt: float,
    batch_size: int,
    seed: int,
    target_edge: int,
) -> dict[str, np.ndarray | float]:
    """Accumulate S_Q and causal response correlations from trajectory batches."""
    (
        tails,
        heads,
        edge_ids,
        dir_signs,
        lead_is_R,
        rates,
        out_events,
        out_counts,
        escape,
    ) = build_directed_events(model)

    max_p_jump = float(np.max(escape) * dt)
    if max_p_jump > 0.10:
        print(f"WARNING: max escape_rate * dt = {max_p_jump:.3f}; decrease DT for better binning accuracy.")
    else:
        print(f"max escape_rate * dt = {max_p_jump:.4f}")

    cum_pi = np.cumsum(model.pi).astype(np.float64)
    freqs = 2.0 * np.pi * np.fft.rfftfreq(n_steps, d=dt)

    psd_sum = np.zeros(freqs.size, dtype=np.float64)
    corr_b_sum = np.zeros(n_steps, dtype=np.float64)
    corr_f_sum = np.zeros(n_steps, dtype=np.float64)

    n_done = 0
    batch_id = 0
    nfft_corr = 2 * n_steps
    norm_lags = np.arange(n_steps, 0, -1, dtype=np.float64)

    while n_done < n_traj:
        nbatch = min(batch_size, n_traj - n_done)
        Q, Lb, Lf = simulate_batch_rates(
            nbatch,
            n_steps,
            dt,
            seed + 1009 * batch_id,
            cum_pi,
            heads,
            edge_ids,
            dir_signs,
            lead_is_R,
            rates,
            out_events,
            out_counts,
            escape,
            target_edge,
            int(model.edge_tail[target_edge]),
            int(model.edge_head[target_edge]),
            float(model.r_plus[target_edge]),
            float(model.r_minus[target_edge]),
        )

        # Center Q by the exact steady-state mean.  The conjugate observables have
        # zero mean theoretically; subtracting their sample mean reduces finite-sample drift.
        q = Q - model.q_mean_right
        lb = Lb - np.mean(Lb)
        lf = Lf - np.mean(Lf)

        Fq = np.fft.rfft(q, axis=1)
        psd_sum += np.sum(np.abs(Fq) ** 2, axis=0)

        # Linear positive-lag cross-correlations C[k] = < q[m+k] lambda[m] >.
        # FFT gives sums over m; divide by the number of pairs at each lag later.
        Qfft = np.fft.fft(q, n=nfft_corr, axis=1)
        Bfft = np.fft.fft(lb, n=nfft_corr, axis=1)
        Ffft = np.fft.fft(lf, n=nfft_corr, axis=1)
        corr_b = np.fft.ifft(Qfft * np.conj(Bfft), axis=1).real[:, :n_steps]
        corr_f = np.fft.ifft(Qfft * np.conj(Ffft), axis=1).real[:, :n_steps]
        corr_b_sum += np.sum(corr_b, axis=0)
        corr_f_sum += np.sum(corr_f, axis=0)

        n_done += nbatch
        batch_id += 1
        print(f"  simulated {n_done}/{n_traj} trajectories")

    S_Q = dt / n_steps * psd_sum / n_traj
    C_b = corr_b_sum / (n_traj * norm_lags)
    C_f = corr_f_sum / (n_traj * norm_lags)

    # R(omega) = int_0^infty dt exp(+i omega t) C(t).
    R_b = dt * np.conj(np.fft.rfft(C_b, n=n_steps))
    R_f = dt * np.conj(np.fft.rfft(C_f, n=n_steps))

    return {
        "freqs": freqs,
        "S_Q": S_Q,
        "C_b": C_b,
        "C_f": C_f,
        "R_b": R_b,
        "R_f": R_f,
        "max_p_jump": max_p_jump,
    }


# =============================================================================
# Plotting
# =============================================================================

def finite_mask(freqs: np.ndarray) -> np.ndarray:
    return (freqs > 0.0) & (freqs <= MAX_PLOT_FREQ)

def set_style():
    plt.rcParams.update({
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.2,
        "ytick.labelsize": 8.2,
        "legend.fontsize": 7.6,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.minor.width": 0.6,
        "ytick.minor.width": 0.6,
        "legend.frameon": False,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.01,
    })

SINGLE_COLUMN_WIDTH_IN = 1.95
FIG_HEIGHT_IN = 1.55

def savefig(fig: plt.Figure, fig_dir: Path, stem: str) -> None:
    fig.tight_layout(pad=0.5)
    fig.savefig(fig_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{stem}.svg", bbox_inches="tight")
    plt.close(fig)

def plot_response_spectra(result: dict, model: DotModel, fig_dir: Path, target_edge: int) -> None:
    set_style()
    freqs = result["freqs"]
    R_b = result["R_b"]
    R_f = result["R_f"]
    mask = finite_mask(freqs)

    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN))

    ax.plot(
        freqs[mask],
        np.real(R_b[mask]),
        lw=0.85,
        label=r"$\mathrm{Re}\,\mathcal{R}_b$",
    )
    ax.plot(
        freqs[mask],
        np.imag(R_b[mask]),
        lw=0.85,
        ls="-.",
        label=r"$\mathrm{Im}\,\mathcal{R}_b$",
    )
    ax.plot(
        freqs[mask],
        np.real(R_f[mask]),
        lw=0.85,
        ls="--",
        label=r"$\mathrm{Re}\,\mathcal{R}_f$",
    )
    ax.plot(
        freqs[mask],
        np.imag(R_f[mask]),
        lw=0.85,
        ls=":",
        label=r"$\mathrm{Im}\,\mathcal{R}_f$",
    )

    # ax.axhline(0.0, color="0.4", ls=":", lw=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"response")
    ax.legend(frameon=False, fontsize=7.8, loc="best")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    label = model.edge_label[target_edge].replace("\\", "")

    savefig(fig, fig_dir, "spinful_dot_response_spectra")


# def plot_response_duality(result: dict, model: DotModel, fig_dir: Path, target_edge: int) -> None:
#     freqs = result["freqs"]
#     R_b = result["R_b"]
#     R_f = result["R_f"]
#     mask = finite_mask(freqs)
#     denom_mask = np.abs(R_f) > MIN_RATIO_DENOM * np.nanmax(np.abs(R_f[mask]))
#     mask = mask & denom_mask

#     ratio = np.full_like(R_b, np.nan + 1j * np.nan)
#     ratio[denom_mask] = R_b[denom_mask] / R_f[denom_mask]
#     target = 2.0 * model.j[target_edge] / model.a[target_edge]

#     fig, ax = plt.subplots(figsize=(3.35, 2.55))
#     ax.plot(freqs[mask], np.real(ratio[mask]), lw=1.15, label=r"$\mathrm{Re}\,[R_b/R_f]$")
#     ax.plot(freqs[mask], np.imag(ratio[mask]), lw=1.15, ls="-.", label=r"$\mathrm{Im}\,[R_b/R_f]$")
#     ax.axhline(target, color="k", ls="--", lw=1.0, label=r"$2j_e/a_e$")
#     ax.axhline(0.0, color="0.4", ls=":", lw=0.8)
#     ax.set_xscale("log")
#     ax.set_xlabel(r"$\omega$")
#     ax.set_ylabel(r"response ratio")
#     ax.legend(frameon=False, fontsize=8, loc="best")
#     ax.grid(True, ls="--", lw=0.4, alpha=0.6)
#     savefig(fig, fig_dir, "spinful_dot_response_duality")

def plot_response_duality(result: dict, model: DotModel, fig_dir: Path, target_edge: int) -> None:
    set_style()
    freqs = result["freqs"]
    R_b = result["R_b"]
    R_f = result["R_f"]

    base_mask = finite_mask(freqs)
    target = 2.0 * model.j[target_edge] / model.a[target_edge]

    def robust_ratio_mask(y: np.ndarray, mask: np.ndarray, expected: float) -> np.ndarray:
        """
        Remove abnormal spikes in ratio curves.

        The first mask removes non-finite values. Then we use a robust
        MAD criterion on residuals y - expected. A hard-width cutoff is
        also imposed to remove denominator-induced spikes.
        """
        m = mask & np.isfinite(y)
        if np.sum(m) < 8:
            return m

        vals = y[m]
        residual = vals - expected

        med = np.nanmedian(residual)
        mad = 1.4826 * np.nanmedian(np.abs(residual - med))

        if (not np.isfinite(mad)) or mad < 1e-14:
            q25, q75 = np.nanpercentile(residual, [25, 75])
            mad = 0.7413 * (q75 - q25)

        if (not np.isfinite(mad)) or mad < 1e-14:
            mad = np.nanstd(residual)

        if (not np.isfinite(mad)) or mad < 1e-14:
            mad = 1e-14

        # The ratio should be close to expected.
        # This minimum width avoids over-filtering when the simulation is clean.
        scale = max(1.0, abs(expected))
        soft_width = max(8.0 * mad, 0.04 * scale)

        # This hard cutoff removes large spikes caused by near-zero denominators.
        hard_width = 3.0 * scale
        width = min(soft_width, hard_width)

        return m & (np.abs((y - expected) - med) <= width)

    # ---------- Complex ratio R_b / R_f ----------
    abs_Rf_max = np.nanmax(np.abs(R_f[base_mask]))
    complex_denom_mask = np.abs(R_f) > MIN_RATIO_DENOM * abs_Rf_max

    ratio = np.full(R_b.shape, np.nan + 1j * np.nan, dtype=np.complex128)
    ratio[complex_denom_mask] = R_b[complex_denom_mask] / R_f[complex_denom_mask]

    ratio_real = np.real(ratio)
    ratio_imag = np.imag(ratio)

    complex_real_mask = robust_ratio_mask(
        ratio_real,
        base_mask & complex_denom_mask,
        expected=target,
    )
    complex_imag_mask = robust_ratio_mask(
        ratio_imag,
        base_mask & complex_denom_mask,
        expected=0.0,
    )

    # ---------- Component-wise ratios ----------
    re_Rb = np.real(R_b)
    im_Rb = np.imag(R_b)
    re_Rf = np.real(R_f)
    im_Rf = np.imag(R_f)

    re_Rf_max = np.nanmax(np.abs(re_Rf[base_mask]))
    im_Rf_max = np.nanmax(np.abs(im_Rf[base_mask]))

    re_denom_mask = np.abs(re_Rf) > MIN_RATIO_DENOM * re_Rf_max
    im_denom_mask = np.abs(im_Rf) > MIN_RATIO_DENOM * im_Rf_max

    ratio_re_parts = np.full(freqs.shape, np.nan, dtype=np.float64)
    ratio_im_parts = np.full(freqs.shape, np.nan, dtype=np.float64)

    ratio_re_parts[re_denom_mask] = re_Rb[re_denom_mask] / re_Rf[re_denom_mask]
    ratio_im_parts[im_denom_mask] = im_Rb[im_denom_mask] / im_Rf[im_denom_mask]

    re_parts_mask = robust_ratio_mask(
        ratio_re_parts,
        base_mask & re_denom_mask,
        expected=target,
    )
    im_parts_mask = robust_ratio_mask(
        ratio_im_parts,
        base_mask & im_denom_mask,
        expected=target,
    )

    # ---------- Plot ----------
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH_IN, FIG_HEIGHT_IN))

    ax.plot(
        freqs[im_parts_mask],
        ratio_im_parts[im_parts_mask],
        lw=0.75,
        ls="-.",
        color='#B2182B',
        label=r"$\frac{\mathrm{Im}\,\mathcal{R}_b}{\mathrm{Im}\,\mathcal{R}_f}$",
    )
    ax.plot(
        freqs[complex_real_mask],
        ratio_real[complex_real_mask],
        lw=0.85,
        color='blue',
        label=r"$\mathrm{Re}\,[\frac{\mathcal{R}_b}{\mathcal{R}_f}]$",
    )
    ax.plot(
        freqs[complex_imag_mask],
        ratio_imag[complex_imag_mask],
        lw=0.85,
        ls="--",
        color='orange',
        label=r"$\mathrm{Im}\,[\frac{\mathcal{R}_b}{\mathcal{R}_f}]$",
    )
    ax.plot(
        freqs[re_parts_mask],
        ratio_re_parts[re_parts_mask],
        lw=0.75,
        ls=":",
        color='lime',
        label=r"$\frac{\mathrm{Re}\,\mathcal{R}_b}{\mathrm{Re}\,\mathcal{R}_f}$",
    )

    ax.axhline(target, color="gray", ls="--", lw=0.7, label=r"$2\tanh\frac{A_e}{2}$")
    ax.axhline(2.0 * np.arctanh(target / 2.0), color="gray", lw=0.5, label=r"$A_e$")
    ax.axhline(0.0, color="0.4", ls=":", lw=0.8)

    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"response ratio")
    ax.legend(frameon=False, fontsize=7.2, loc="best")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)

    savefig(fig, fig_dir, "spinful_dot_response_duality")


def plot_rturbound(result: dict, model: DotModel, fig_dir: Path, target_edge: int) -> None:
    freqs = result["freqs"]
    R_b = result["R_b"]
    S_Q = result["S_Q"]
    mask = finite_mask(freqs)
    mask = mask & (S_Q > 0.0)

    snr = np.zeros_like(freqs)
    snr[mask] = np.abs(R_b[mask]) ** 2 / S_Q[mask]
    bound_pseudo = model.sigma_pseudo / 2.0
    bound_sigma = model.sigma / 2.0

    fig, ax = plt.subplots(figsize=(3.35, 2.55))
    ax.plot(freqs[mask], snr[mask], lw=1.2, label=r"$|R_b|^2/S_Q$")
    ax.axhline(bound_pseudo, color="0.25", lw=1.0, ls="--", label=r"$\dot\sigma_{\rm pseudo}/2$")
    ax.axhline(bound_sigma, color="k", lw=1.0, ls=":", label=r"$\dot\sigma/2$")
    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"SNR and bounds")
    ax.legend(frameon=False, fontsize=8, loc="best")
    ax.grid(True, ls="--", lw=0.4, alpha=0.6)
    savefig(fig, fig_dir, "spinful_dot_rturbound")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    fig_dir = script_dir / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)

    model = build_spinful_dot_model()
    if TARGET_EDGE < 0 or TARGET_EDGE >= len(model.edge_tail):
        raise ValueError(f"TARGET_EDGE must be in [0,{len(model.edge_tail)-1}].")

    print("\nFour-state spinful quantum dot")
    print("State order: 0, up, down, double")
    print(f"beta={model.beta}, epsilon={model.eps}, U={model.U}, Delta_Z={model.delta_z}")
    print(f"mu_L={model.mu_L}, mu_R={model.mu_R}")
    print("pi =", model.pi)
    print("generator eigenvalues =", model.eigvals)
    print(f"activity={model.activity:.6g}, sigma_pseudo={model.sigma_pseudo:.6g}, sigma={model.sigma:.6g}")
    print(f"mean right current = {model.q_mean_right:.6g}")
    print("\nChannel-resolved edges, plus direction = electron enters dot:")
    for e, lab in enumerate(model.edge_label):
        print(
            f"  e={e}: {lab:28s}  r+={model.r_plus[e]:.5g}  r-={model.r_minus[e]:.5g}  "
            f"b={model.b[e]: .4f}  f={model.f[e]: .4f}  "
            f"j={model.j[e]: .5g}  a={model.a[e]: .5g}  2j/a={2*model.j[e]/model.a[e]: .5g}"
        )
    print(f"\nTarget edge e={TARGET_EDGE}: {model.edge_label[TARGET_EDGE]}")

    cache_name = (
        f"spinful_dot_target{TARGET_EDGE}_dt{DT:g}_steps{N_STEPS}_traj{N_TRAJ}_"
        f"seed{SEED}.npz"
    )
    cache_path = data_dir / cache_name

    if cache_path.exists() and not FORCE_RECOMPUTE:
        print(f"\nLoading cached data: {cache_path}")
        loaded = np.load(cache_path, allow_pickle=False)
        result = {key: loaded[key] for key in loaded.files}
    else:
        print("\nSampling trajectories...")
        result = accumulate_fft_estimators(
            model=model,
            n_traj=N_TRAJ,
            n_steps=N_STEPS,
            dt=DT,
            batch_size=BATCH_SIZE,
            seed=SEED,
            target_edge=TARGET_EDGE,
        )
        np.savez_compressed(
            cache_path,
            freqs=result["freqs"],
            S_Q=result["S_Q"],
            C_b=result["C_b"],
            C_f=result["C_f"],
            R_b=result["R_b"],
            R_f=result["R_f"],
            max_p_jump=result["max_p_jump"],
            pi=model.pi,
            r_plus=model.r_plus,
            r_minus=model.r_minus,
            a=model.a,
            j=model.j,
            sigma=model.sigma,
            sigma_pseudo=model.sigma_pseudo,
            q_mean_right=model.q_mean_right,
            target_edge=TARGET_EDGE,
        )
        print(f"Saved data: {cache_path}")

    # Diagnostics for the target edge.
    freqs = result["freqs"]
    R_b = result["R_b"]
    R_f = result["R_f"]
    mask = finite_mask(freqs)
    ratio_mask = mask & (np.abs(R_f) > MIN_RATIO_DENOM * np.nanmax(np.abs(R_f[mask])))
    ratio = R_b[ratio_mask] / R_f[ratio_mask]
    target = 2.0 * model.j[TARGET_EDGE] / model.a[TARGET_EDGE]
    if ratio.size > 0:
        err = np.nanmedian(np.abs(ratio - target))
        print(f"Median |R_b/R_f - 2j/a| over plotted window: {err:.4g}")
    print(f"Theoretical 2j/a on target edge: {target:.8g}")

    plot_response_spectra(result, model, fig_dir, TARGET_EDGE)
    plot_response_duality(result, model, fig_dir, TARGET_EDGE)
    plot_rturbound(result, model, fig_dir, TARGET_EDGE)

    print("\nSaved figures:")
    for stem in [
        "spinful_dot_response_spectra",
        "spinful_dot_response_duality",
        "spinful_dot_rturbound",
    ]:
        print(f"  {fig_dir / (stem + '.pdf')}")
        print(f"  {fig_dir / (stem + '.svg')}")


if __name__ == "__main__":
    main()
