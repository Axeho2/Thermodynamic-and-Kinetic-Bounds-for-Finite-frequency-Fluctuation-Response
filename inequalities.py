"""
Simulate Eq. (11) finite-frequency fluctuation-response inequalities
for many randomly sampled complete-graph Markov jump processes with several
state-space sizes and many observables.

The script estimates, from steady-state stochastic trajectories,

    sum_e |R_{b_e}(omega)|^2 / a_e <= S_Q(omega),
    sum_e 4 |R_{f_e}(omega)|^2 / a_e <= S_Q(omega),

and plots only LHS/S_Q(omega) as scatter plots.

Two separate figures are generated:
  (i)  generic trajectory observables with arbitrary directed jump weights;
  (ii) state-current observables with antisymmetric jump weights.

Plot convention:
  - color  = number of states, e.g. N=3,4,5;
  - marker = Eq. (11) term, not observable index;
  - all sampled observables are pooled within each network size.

Rate convention for each undirected edge e:
    r_e^+ = exp(b_e + f_e/2),
    r_e^- = exp(b_e - f_e/2).

The master equation uses the column convention: dp/dt = R p.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ----------------------------- user controls -----------------------------
SEED = int(os.environ.get("SEED", "123"))
TRAJ_SEED = int(os.environ.get("TRAJ_SEED", "2027"))

# Comma-separated list, for example STATE_SIZES=3,4,5,6
STATE_SIZES = tuple(
    int(x.strip()) for x in os.environ.get("STATE_SIZES", "3,4,5,6").split(",") if x.strip()
)

# N_MODELS is the number of independently sampled rate matrices per size.
N_MODELS = int(os.environ.get("N_MODELS", "8"))
N_GENERIC_OBS = int(os.environ.get("N_GENERIC_OBS", "8"))
N_STATE_CURRENT_OBS = int(os.environ.get("N_STATE_CURRENT_OBS", "8"))

N_STEPS = int(os.environ.get("N_STEPS", "1024"))
N_TRAJ = int(os.environ.get("N_TRAJ", "5000"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "500"))
DT = float(os.environ.get("DT", "0.01"))
RATE_SCALE = float(os.environ.get("RATE_SCALE", "0.45"))
FORCE_RECOMPUTE = int(os.environ.get("FORCE_RECOMPUTE", "0"))

MIN_PLOT_FREQ = float(os.environ.get("MIN_PLOT_FREQ", "1e-2"))
MAX_PLOT_FREQ = float(os.environ.get("MAX_PLOT_FREQ", "2e2"))
MIN_S_DENOM = float(os.environ.get("MIN_S_DENOM", "1e-12"))
SCATTER_STRIDE = int(os.environ.get("SCATTER_STRIDE", "2"))


# ----------------------------- plot settings ------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "legend.fontsize": 7.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "mathtext.fontset": "cm",
})


@dataclass
class MarkovJumpModel:
    n_states: int
    edge_tail: np.ndarray
    edge_head: np.ndarray
    b: np.ndarray
    f: np.ndarray
    r_plus: np.ndarray
    r_minus: np.ndarray
    R: np.ndarray
    pi: np.ndarray
    j: np.ndarray
    a: np.ndarray
    activity: float
    epr: float
    pseudo_epr: float


@nb.njit
def set_numba_seed(seed: int) -> None:
    np.random.seed(seed)


def ensure_dirs() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    fig_dir = root / "figures"
    data_dir.mkdir(exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    return data_dir, fig_dir


def savefig(fig: plt.Figure, fig_dir: Path, name: str) -> None:
    fig.tight_layout(pad=0.35)
    fig.savefig(fig_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.svg", bbox_inches="tight")
    fig.savefig(fig_dir / f"{name}.png", bbox_inches="tight", dpi=800)
    plt.close(fig)


def stationary_distribution(R: np.ndarray) -> np.ndarray:
    vals, vecs = np.linalg.eig(R)
    idx = int(np.argmin(np.abs(vals)))
    pi = np.real(vecs[:, idx])
    if np.sum(pi) < 0:
        pi = -pi
    pi = np.maximum(pi, 0.0)
    pi = pi / np.sum(pi)
    return pi


def complete_graph_edges(n_states: int) -> tuple[np.ndarray, np.ndarray]:
    edge_tail = []
    edge_head = []
    for i in range(n_states):
        for j in range(i + 1, n_states):
            edge_tail.append(i)
            edge_head.append(j)
    return np.asarray(edge_tail, dtype=np.int64), np.asarray(edge_head, dtype=np.int64)


def build_rate_matrix(
    n_states: int,
    edge_tail: np.ndarray,
    edge_head: np.ndarray,
    r_plus: np.ndarray,
    r_minus: np.ndarray,
) -> np.ndarray:
    R = np.zeros((n_states, n_states), dtype=np.float64)
    for e in range(len(edge_tail)):
        tail = int(edge_tail[e])
        head = int(edge_head[e])
        R[head, tail] += r_plus[e]
        R[tail, head] += r_minus[e]
    for j in range(n_states):
        R[j, j] = -np.sum(R[:, j]) + R[j, j]
    return R


def sample_random_complete_graph_model(
    n_states: int,
    seed: int,
    rate_scale: float = RATE_SCALE,
) -> MarkovJumpModel:
    """Random complete-graph Markov jump process with n_states states."""
    rng = np.random.default_rng(seed)
    edge_tail, edge_head = complete_graph_edges(n_states)
    n_edges = len(edge_tail)

    # b controls symmetric kinetic scale; f controls edge force.
    # The additive log(rate_scale) keeps max_escape * dt modest for the
    # one-jump-per-bin approximation.
    b = rng.uniform(-0.65, 0.55, size=n_edges) + np.log(rate_scale)
    f = rng.uniform(-2.0, 2.0, size=n_edges)

    r_plus = np.exp(b + 0.5 * f)
    r_minus = np.exp(b - 0.5 * f)
    R = build_rate_matrix(n_states, edge_tail, edge_head, r_plus, r_minus)
    pi = stationary_distribution(R)

    j = r_plus * pi[edge_tail] - r_minus * pi[edge_head]
    a = r_plus * pi[edge_tail] + r_minus * pi[edge_head]
    activity = float(np.sum(a))
    edge_force_ss = np.log((r_plus * pi[edge_tail]) / (r_minus * pi[edge_head]))
    epr = float(np.sum(j * edge_force_ss))
    pseudo_epr = float(np.sum(2.0 * j * j / a))

    return MarkovJumpModel(
        n_states=n_states,
        edge_tail=edge_tail,
        edge_head=edge_head,
        b=b,
        f=f,
        r_plus=r_plus,
        r_minus=r_minus,
        R=R,
        pi=pi,
        j=j,
        a=a,
        activity=activity,
        epr=epr,
        pseudo_epr=pseudo_epr,
    )


def normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Normalize each row to O(1), avoiding almost-constant observables."""
    y = x.copy()
    y -= np.mean(y, axis=1, keepdims=True)
    scale = np.std(y, axis=1, keepdims=True)
    scale = np.maximum(scale, eps)
    return y / scale


def define_observable_ensemble(
    model: MarkovJumpModel,
    seed: int,
    n_generic: int = N_GENERIC_OBS,
    n_state_current: int = N_STATE_CURRENT_OBS,
) -> dict:
    """Sample many generic and state-current observables for one model."""
    rng = np.random.default_rng(seed)
    n_edges = len(model.edge_tail)

    # Generic trajectory observables: arbitrary state weights and arbitrary
    # directed jump weights. These are intentionally not constrained by
    # h_minus = -h_plus.
    g_generic = normalize_rows(rng.normal(0.0, 1.0, size=(n_generic, model.n_states)))
    h_plus_generic = rng.normal(0.0, 0.75, size=(n_generic, n_edges))
    h_minus_generic = rng.normal(0.0, 0.75, size=(n_generic, n_edges))

    # State-current observables: arbitrary state weights plus antisymmetric
    # jump weights on every edge.
    g_sc = normalize_rows(rng.normal(0.0, 1.0, size=(n_state_current, model.n_states)))
    h_edge_sc = rng.normal(0.0, 0.75, size=(n_state_current, n_edges))
    h_plus_sc = h_edge_sc.copy()
    h_minus_sc = -h_edge_sc.copy()

    return {
        "generic": {
            "g": g_generic.astype(np.float64),
            "h_plus": h_plus_generic.astype(np.float64),
            "h_minus": h_minus_generic.astype(np.float64),
        },
        "state_current": {
            "g": g_sc.astype(np.float64),
            "h_plus": h_plus_sc.astype(np.float64),
            "h_minus": h_minus_sc.astype(np.float64),
        },
    }


@nb.njit
def sample_initial_states(pi: np.ndarray, n_traj: int) -> np.ndarray:
    states = np.empty(n_traj, dtype=np.int64)
    cdf = np.empty(pi.shape[0], dtype=np.float64)
    acc = 0.0
    for i in range(pi.shape[0]):
        acc += pi[i]
        cdf[i] = acc
    for tr in range(n_traj):
        u = np.random.random()
        s = 0
        while s < pi.shape[0] - 1 and u > cdf[s]:
            s += 1
        states[tr] = s
    return states


@nb.njit
def simulate_batch_multi(
    edge_tail: np.ndarray,
    edge_head: np.ndarray,
    r_plus: np.ndarray,
    r_minus: np.ndarray,
    pi: np.ndarray,
    g_generic: np.ndarray,
    h_plus_generic: np.ndarray,
    h_minus_generic: np.ndarray,
    g_sc: np.ndarray,
    h_plus_sc: np.ndarray,
    h_minus_sc: np.ndarray,
    dt: float,
    n_steps: int,
    n_traj: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Binned CTMC trajectories with at most one jump in each time bin."""
    n_edges = edge_tail.shape[0]
    n_generic = g_generic.shape[0]
    n_sc = g_sc.shape[0]
    states = sample_initial_states(pi, n_traj)

    q_generic = np.zeros((n_generic, n_traj, n_steps), dtype=np.float64)
    q_sc = np.zeros((n_sc, n_traj, n_steps), dtype=np.float64)
    lambda_b = np.zeros((n_edges, n_traj, n_steps), dtype=np.float64)
    lambda_f = np.zeros((n_edges, n_traj, n_steps), dtype=np.float64)

    inv_dt = 1.0 / dt

    for tr in range(n_traj):
        s = states[tr]
        for t in range(n_steps):
            for m in range(n_generic):
                q_generic[m, tr, t] = g_generic[m, s]
            for m in range(n_sc):
                q_sc[m, tr, t] = g_sc[m, s]

            # Compensator contribution to all conjugate observables.
            escape = 0.0
            for e in range(n_edges):
                tail = edge_tail[e]
                head = edge_head[e]
                if s == tail:
                    escape += r_plus[e]
                    lambda_b[e, tr, t] -= r_plus[e]
                    lambda_f[e, tr, t] -= 0.5 * r_plus[e]
                elif s == head:
                    escape += r_minus[e]
                    lambda_b[e, tr, t] -= r_minus[e]
                    lambda_f[e, tr, t] += 0.5 * r_minus[e]

            # Jump event, if any.
            if np.random.random() < escape * dt:
                u = np.random.random() * escape
                acc = 0.0
                chosen = -1
                sign = 0  # +1 for plus jump, -1 for minus jump
                new_state = s
                for e in range(n_edges):
                    tail = edge_tail[e]
                    head = edge_head[e]
                    if s == tail:
                        acc += r_plus[e]
                        if u <= acc:
                            chosen = e
                            sign = 1
                            new_state = head
                            break
                    elif s == head:
                        acc += r_minus[e]
                        if u <= acc:
                            chosen = e
                            sign = -1
                            new_state = tail
                            break

                if chosen >= 0:
                    if sign == 1:
                        for m in range(n_generic):
                            q_generic[m, tr, t] += h_plus_generic[m, chosen] * inv_dt
                        for m in range(n_sc):
                            q_sc[m, tr, t] += h_plus_sc[m, chosen] * inv_dt
                        lambda_b[chosen, tr, t] += inv_dt
                        lambda_f[chosen, tr, t] += 0.5 * inv_dt
                    else:
                        for m in range(n_generic):
                            q_generic[m, tr, t] += h_minus_generic[m, chosen] * inv_dt
                        for m in range(n_sc):
                            q_sc[m, tr, t] += h_minus_sc[m, chosen] * inv_dt
                        lambda_b[chosen, tr, t] += inv_dt
                        lambda_f[chosen, tr, t] -= 0.5 * inv_dt
                    s = new_state

    return q_generic, q_sc, lambda_b, lambda_f


def init_accumulator(n_obs: int, n_edges: int, n_steps: int) -> dict:
    return {
        "S_sum": np.zeros((n_obs, n_steps // 2 + 1), dtype=np.float64),
        "corr_b_sum": np.zeros((n_obs, n_edges, n_steps), dtype=np.float64),
        "corr_f_sum": np.zeros((n_obs, n_edges, n_steps), dtype=np.float64),
        "n_traj": 0,
    }


def accumulate_for_observable_ensemble(
    q_all: np.ndarray,
    lambda_b: np.ndarray,
    lambda_f: np.ndarray,
    dt: float,
    accum: dict,
) -> None:
    """Accumulate periodogram S_Q and one-sided response correlations."""
    n_obs, n_traj, n_steps = q_all.shape
    n_edges = lambda_b.shape[0]
    nfft_corr = 2 * n_steps

    # These do not depend on the observable, so compute once per batch.
    lambda_b_centered = lambda_b - np.mean(lambda_b, axis=(1, 2), keepdims=True)
    lambda_f_centered = lambda_f - np.mean(lambda_f, axis=(1, 2), keepdims=True)
    lambda_b_fft = np.fft.rfft(lambda_b_centered, n=nfft_corr, axis=2)
    lambda_f_fft = np.fft.rfft(lambda_f_centered, n=nfft_corr, axis=2)

    for m in range(n_obs):
        q = q_all[m]
        q_centered = q - np.mean(q)
        q_fft = np.fft.rfft(q_centered, n=n_steps, axis=1)
        accum["S_sum"][m] += (dt / n_steps) * np.sum(np.abs(q_fft) ** 2, axis=0)

        x_fft = np.fft.rfft(q_centered, n=nfft_corr, axis=1)
        for e in range(n_edges):
            # corr[k] = < q(t+k) Lambda(t) > with biased normalization by n_steps.
            corr_b = np.fft.irfft(x_fft * np.conj(lambda_b_fft[e]), n=nfft_corr, axis=1)[:, :n_steps]
            corr_f = np.fft.irfft(x_fft * np.conj(lambda_f_fft[e]), n=nfft_corr, axis=1)[:, :n_steps]
            accum["corr_b_sum"][m, e] += np.sum(corr_b, axis=0) / n_steps
            accum["corr_f_sum"][m, e] += np.sum(corr_f, axis=0) / n_steps

    accum["n_traj"] += n_traj


def finalize_accumulator(accum: dict, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_traj = int(accum["n_traj"])
    n_steps = accum["corr_b_sum"].shape[2]
    freqs = 2.0 * np.pi * np.fft.rfftfreq(n_steps, d=dt)

    S_Q = accum["S_sum"] / n_traj
    corr_b = accum["corr_b_sum"] / n_traj
    corr_f = accum["corr_f_sum"] / n_traj

    # R(omega) = int_0^infty exp(+i omega t) C(t) dt.
    R_b = dt * np.conj(np.fft.rfft(corr_b, n=n_steps, axis=2))
    R_f = dt * np.conj(np.fft.rfft(corr_f, n=n_steps, axis=2))
    return freqs, S_Q, R_b, R_f


def run_simulation_for_model(
    model: MarkovJumpModel,
    observables: dict,
    data_path: Path,
    size_idx: int,
    model_idx: int,
) -> dict:
    n_edges = len(model.edge_tail)
    n_generic = observables["generic"]["g"].shape[0]
    n_sc = observables["state_current"]["g"].shape[0]
    accum_generic = init_accumulator(n_generic, n_edges, N_STEPS)
    accum_sc = init_accumulator(n_sc, n_edges, N_STEPS)

    # Unique trajectory seed for every size/model pair.
    set_numba_seed(TRAJ_SEED + 1_000_003 * size_idx + 100_003 * model_idx)
    n_done = 0
    while n_done < N_TRAJ:
        n_batch = min(BATCH_SIZE, N_TRAJ - n_done)
        q_generic, q_sc, lambda_b, lambda_f = simulate_batch_multi(
            model.edge_tail,
            model.edge_head,
            model.r_plus,
            model.r_minus,
            model.pi,
            observables["generic"]["g"],
            observables["generic"]["h_plus"],
            observables["generic"]["h_minus"],
            observables["state_current"]["g"],
            observables["state_current"]["h_plus"],
            observables["state_current"]["h_minus"],
            DT,
            N_STEPS,
            n_batch,
        )
        accumulate_for_observable_ensemble(q_generic, lambda_b, lambda_f, DT, accum_generic)
        accumulate_for_observable_ensemble(q_sc, lambda_b, lambda_f, DT, accum_sc)
        n_done += n_batch
        print(f"      simulated {n_done}/{N_TRAJ} trajectories", flush=True)

    freqs, S_generic, Rb_generic, Rf_generic = finalize_accumulator(accum_generic, DT)
    _, S_sc, Rb_sc, Rf_sc = finalize_accumulator(accum_sc, DT)

    result = {
        "freqs": freqs,
        "S_generic": S_generic,
        "Rb_generic": Rb_generic,
        "Rf_generic": Rf_generic,
        "S_state_current": S_sc,
        "Rb_state_current": Rb_sc,
        "Rf_state_current": Rf_sc,
    }

    np.savez_compressed(
        data_path,
        **result,
        n_states=model.n_states,
        edge_tail=model.edge_tail,
        edge_head=model.edge_head,
        b=model.b,
        f=model.f,
        r_plus=model.r_plus,
        r_minus=model.r_minus,
        R=model.R,
        pi=model.pi,
        j=model.j,
        a=model.a,
        activity=model.activity,
        epr=model.epr,
        pseudo_epr=model.pseudo_epr,
        g_generic=observables["generic"]["g"],
        h_plus_generic=observables["generic"]["h_plus"],
        h_minus_generic=observables["generic"]["h_minus"],
        g_state_current=observables["state_current"]["g"],
        h_plus_state_current=observables["state_current"]["h_plus"],
        h_minus_state_current=observables["state_current"]["h_minus"],
        dt=DT,
        n_steps=N_STEPS,
        n_traj=N_TRAJ,
        batch_size=BATCH_SIZE,
        seed=SEED,
        traj_seed=TRAJ_SEED,
        size_idx=size_idx,
        model_idx=model_idx,
        n_models=N_MODELS,
        state_sizes=np.asarray(STATE_SIZES, dtype=np.int64),
        n_generic_obs=N_GENERIC_OBS,
        n_state_current_obs=N_STATE_CURRENT_OBS,
        rate_scale=RATE_SCALE,
    )
    return result


def load_or_run_model(
    model: MarkovJumpModel,
    observables: dict,
    data_dir: Path,
    size_idx: int,
    model_idx: int,
    model_seed: int,
    obs_seed: int,
) -> dict:
    data_path = data_dir / (
        f"eq11_multi_randomN{model.n_states}_model{model_idx:03d}_"
        f"modelseed{model_seed}_obsseed{obs_seed}_traj{TRAJ_SEED}_"
        f"dt{DT:g}_N{N_STEPS}_M{N_TRAJ}_"
        f"G{N_GENERIC_OBS}_SC{N_STATE_CURRENT_OBS}_scale{RATE_SCALE:g}.npz"
    )
    if data_path.exists() and not FORCE_RECOMPUTE:
        print(f"    Loading cached data: {data_path.name}")
        data = np.load(data_path)
        return {key: data[key] for key in [
            "freqs", "S_generic", "Rb_generic", "Rf_generic",
            "S_state_current", "Rb_state_current", "Rf_state_current",
        ]}
    print("    Running new simulation...")
    return run_simulation_for_model(model, observables, data_path, size_idx, model_idx)


def ratio_arrays_for_result(
    result: dict,
    model: MarkovJumpModel,
    observable_type: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return selected omega and Eq. (11) ratios with shape (n_obs, n_freq)."""
    freqs = result["freqs"]
    if observable_type == "generic":
        S_Q = result["S_generic"]
        R_b = result["Rb_generic"]
        R_f = result["Rf_generic"]
    elif observable_type == "state_current":
        S_Q = result["S_state_current"]
        R_b = result["Rb_state_current"]
        R_f = result["Rf_state_current"]
    else:
        raise ValueError(f"Unknown observable_type={observable_type}")

    lhs_b = np.sum(np.abs(R_b) ** 2 / model.a[None, :, None], axis=1)
    lhs_f = np.sum(4.0 * np.abs(R_f) ** 2 / model.a[None, :, None], axis=1)

    mask_freq = (
        (freqs > MIN_PLOT_FREQ)
        & (freqs < MAX_PLOT_FREQ)
        & np.isfinite(freqs)
    )
    if SCATTER_STRIDE > 1:
        idx = np.where(mask_freq)[0][::SCATTER_STRIDE]
    else:
        idx = np.where(mask_freq)[0]

    S_sel = S_Q[:, idx]
    denom_mask = np.isfinite(S_sel) & (S_sel > MIN_S_DENOM * np.nanmax(S_Q, axis=1, keepdims=True))

    ratio_b = lhs_b[:, idx] / S_sel
    ratio_f = lhs_f[:, idx] / S_sel
    ratio_b = np.where(denom_mask & np.isfinite(ratio_b), ratio_b, np.nan)
    ratio_f = np.where(denom_mask & np.isfinite(ratio_f), ratio_f, np.nan)

    return freqs[idx], ratio_b, ratio_f


def append_ratio_clouds(
    ratio_data: dict,
    result: dict,
    model: MarkovJumpModel,
    observable_type: str,
) -> None:
    omega, ratio_b, ratio_f = ratio_arrays_for_result(result, model, observable_type)
    n_obs = ratio_b.shape[0]
    for obs_idx in range(n_obs):
        ratio_data[observable_type].append({
            "n_states": model.n_states,
            "obs_idx": obs_idx,
            "omega": omega.copy(),
            "barrier": ratio_b[obs_idx].copy(),
            "force": ratio_f[obs_idx].copy(),
        })


def color_map_for_sizes(state_sizes: tuple[int, ...]) -> dict[int, str]:
    base_colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9"]
    return {n: base_colors[i % len(base_colors)] for i, n in enumerate(state_sizes)}


def marker_for_obs(obs_idx: int) -> str:
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">", "h", "p"]
    return markers[obs_idx % len(markers)]


def plot_ratio_scatter(
    ratio_data: dict,
    fig_dir: Path,
    observable_type: str,
    title: str,
    file_prefix: str,
) -> None:
    clouds = ratio_data[observable_type]
    size_colors = color_map_for_sizes(STATE_SIZES)

    # Plot convention in this version:
    #   color  = network size N;
    #   marker = which Eq. (11) inequality;
    #   individual sampled observables are pooled and are not distinguished.
    # Using two genuinely different marker shapes is much clearer than
    # filled-vs-hollow markers once many semi-transparent points overlap.
    formula_styles = {
        "barrier": {
            "marker": "o",
            "s": 0.5,
            "alpha": 0.4,
            "linewidths": 0.0,
            "label": r"Eq. (11a)",
        },
        "force": {
            "marker": "x",
            "s": 8,
            "alpha": 0.4,
            "linewidths": 0.1,
            "label": r"Eq. (11b)",
        },
    }

    y_all_list = []
    fig, ax = plt.subplots(1, 1, figsize=(3, 2.5))

    for cloud in clouds:
        n_states = int(cloud["n_states"])
        omega = cloud["omega"]
        ratio_b = cloud["barrier"]
        ratio_f = cloud["force"]
        color = size_colors[n_states]

        mask_b = np.isfinite(omega) & np.isfinite(ratio_b) & (ratio_b >= 0.0)
        mask_f = np.isfinite(omega) & np.isfinite(ratio_f) & (ratio_f >= 0.0)
        if np.any(mask_b):
            y_all_list.append(ratio_b[mask_b])
        if np.any(mask_f):
            y_all_list.append(ratio_f[mask_f])

        ax.scatter(
            omega[mask_b],
            ratio_b[mask_b],
            color=color,
            marker=formula_styles["barrier"]["marker"],
            s=formula_styles["barrier"]["s"],
            alpha=formula_styles["barrier"]["alpha"],
            linewidths=formula_styles["barrier"]["linewidths"],
        )
        ax.scatter(
            omega[mask_f],
            ratio_f[mask_f],
            color=color,
            marker=formula_styles["force"]["marker"],
            s=formula_styles["force"]["s"],
            alpha=formula_styles["force"]["alpha"],
            linewidths=formula_styles["force"]["linewidths"],
        )

    ax.axhline(1.0, color="k", ls=":", lw=0.9)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"LHS$/S_Q(\omega)$")
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, ls="--", lw=0.4, alpha=0.5)

    # Two compact legends: network size/color and Eq. (11) term/marker.
    size_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=size_colors[n],
               markeredgecolor=size_colors[n], markersize=5.0, label=rf"$N={n}$")
        for n in STATE_SIZES
    ]
    formula_handles = [
        Line2D([0], [0], marker=formula_styles["barrier"]["marker"],
               color="0.20", linestyle="none", markersize=5.0,
               markerfacecolor="0.20", markeredgecolor="0.20",
               label=formula_styles["barrier"]["label"]),
        Line2D([0], [0], marker=formula_styles["force"]["marker"],
               color="0.20", linestyle="none", markersize=5.2,
               markeredgewidth=0.8,
               label=formula_styles["force"]["label"]),
    ]

    leg1 = ax.legend(handles=size_handles, frameon=False, loc="upper left", fontsize=7.0,
                     handletextpad=0.3, borderpad=0.2)
    ax.add_artist(leg1)
    ax.legend(handles=formula_handles, frameon=False, loc="lower left", fontsize=6.7,
              handletextpad=0.35, borderpad=0.2)

    if y_all_list:
        y_all = np.concatenate(y_all_list)
        ymax = max(1.05, 1.10 * float(np.nanmax(y_all)))
        ax.set_ylim(0.0, ymax)
    else:
        ax.set_ylim(0.0, 1.05)

    savefig(fig, fig_dir, file_prefix)


def print_diagnostics(model: MarkovJumpModel, size_idx: int, model_idx: int) -> None:
    print(
        f"\nRandom complete-graph model: "
        f"N={model.n_states}, size {size_idx + 1}/{len(STATE_SIZES)}, "
        f"rate matrix {model_idx + 1}/{N_MODELS}"
    )
    print("--------------------------------")
    print(f"pi = {model.pi}")
    print(f"activity = {model.activity:.6g}")
    print(f"pseudo_epr = {model.pseudo_epr:.6g}")
    print(f"epr = {model.epr:.6g}")
    max_escape = float(np.max(-np.diag(model.R)))
    print(f"max escape rate = {max_escape:.6g}")
    print(f"max escape * dt = {max_escape * DT:.6g}")
    if max_escape * DT > 0.08:
        print("WARNING: max_escape * dt is not very small. Consider reducing DT or RATE_SCALE.")


def main() -> None:
    data_dir, fig_dir = ensure_dirs()

    ratio_data = {
        "generic": [],
        "state_current": [],
    }

    for size_idx, n_states in enumerate(STATE_SIZES):
        for model_idx in range(N_MODELS):
            model_seed = SEED + 10_000 * n_states + 1009 * model_idx
            obs_seed = SEED + 20_000 * n_states + 50_021 * model_idx + 17
            model = sample_random_complete_graph_model(n_states=n_states, seed=model_seed)
            observables = define_observable_ensemble(model, seed=obs_seed)

            print_diagnostics(model, size_idx, model_idx)
            result = load_or_run_model(
                model,
                observables,
                data_dir,
                size_idx,
                model_idx,
                model_seed,
                obs_seed,
            )

            append_ratio_clouds(ratio_data, result, model, "generic")
            append_ratio_clouds(ratio_data, result, model, "state_current")

    # Two separate figures, as requested.
    plot_ratio_scatter(
        ratio_data,
        fig_dir,
        observable_type="generic",
        title="Generic trajectory observables",
        file_prefix="eq11_multi_size_generic_scatter",
    )
    plot_ratio_scatter(
        ratio_data,
        fig_dir,
        observable_type="state_current",
        title="State-current observables",
        file_prefix="eq11_multi_size_state_current_scatter",
    )

    print(f"\nFigures saved to {fig_dir}")
    print("  eq11_multi_size_generic_scatter.pdf / .svg")
    print("  eq11_multi_size_state_current_scatter.pdf / .svg")


if __name__ == "__main__":
    main()
