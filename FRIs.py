import numpy as np
import numba as nb
import matplotlib.pyplot as plt

params = {
            'text.usetex' : True,
            'font.family' : 'serif',
            'font.size' : 11,
            'text.latex.preamble' : '\n'.join([
                    r'\usepackage{amsfonts}',
                ]),
}
plt.rcParams.update(params)

@nb.njit
def set_seed(value):
    np.random.seed(value)

set_seed(111)

def pre_eq(R):
    """calculate the steady-state distribution of a Markov chain given transition rate matrix R"""
    n = R.shape[0]

    # find the eigenvector corresponding to the zero eigenvalue
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # find the index of the eigenvalue closest to zero
    idx = np.argmin(np.abs(eigenvalues))

    # extract the corresponding eigenvector and normalize it to get the steady-state distribution
    pi = np.real(eigenvectors[:, idx])
    pi = np.abs(pi)  # non-negative
    pi = pi / np.sum(pi)

    return pi

def compute_aij_jij(R, pi_ss):
    """calculate a_ij and j_ij matrices"""
    n = R.shape[0]
    a_ij = np.zeros((n, n))
    j_ij = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                a_ij[i, j] = pi_ss[j] * R[i, j] + pi_ss[i] * R[j, i]
                j_ij[i, j] = pi_ss[j] * R[i, j] - pi_ss[i] * R[j, i]

    return a_ij, j_ij

@nb.jit(nopython=True)
def generate_trajectories(R, pi_ss, T, dt, n_traj):
    """generate trajectories of a Markov chain"""
    n_states = R.shape[0]
    n_steps = int(T / dt)
    trajectories = np.zeros((n_traj, n_steps), dtype=np.int32)
    
    # calculate transition probability matrix for time step dt
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                P[i, j] = R[i, j] * dt
    for i in range(n_states):
        P[i, i] = 1 - np.sum(P[:, i])
    
    for traj in range(n_traj):
        # set initial state based on steady-state distribution
        r = np.random.rand()
        state = np.searchsorted(np.cumsum(pi_ss), r)
        for step in range(n_steps):
            trajectories[traj, step] = state
            r = np.random.rand()
            state = np.searchsorted(np.cumsum(P[:, state]), r)
    
    return trajectories

def compute_rates(trajectories, R, dt, g=None, h=None):
    """calculate Q_rate, Lambda_b_rate, and Lambda_f_rate for given trajectories"""
    n_traj, n_steps = trajectories.shape
    n_states = R.shape[0]
    
    # initialize output arrays
    Q_rate = np.zeros((n_traj, n_steps))
    noise_rate = np.zeros((n_states, n_states, n_traj, n_steps))
    
    # set default g and h if not provided
    if g is None:
        g = np.random.randn(n_states)
    if h is None:
        h = np.random.randn(n_states, n_states)
        h = (h - h.T) / 2  # anti-symmetric part for h
    
    # create one-hot encoding of trajectories
    trajectories_onehot = np.eye(n_states)[trajectories]
    
    # calculate Q_rate for each time step based on current state
    trajectories_prev_onehot = trajectories_onehot[:, :-1, :]
    trajectories_curr_onehot = trajectories_onehot[:, 1:, :]
    
    # calculate transition indicators
    trans_indicator = np.zeros((n_states, n_states, n_traj, n_steps))
    
    # calculate transition indicators for i != j
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                # create indicator for transitions from j to i
                trans_indicator[i, j, :, 1:] = (
                    trajectories_curr_onehot[:, :, i] * 
                    trajectories_prev_onehot[:, :, j]
                )
    
    # calculate Q_rate based on g and h
    Q_rate = g[trajectories]
    
    # h[i,j] * trans_indicator[i,j,traj,step] * (1/dt)
    h_contribution = np.einsum('ij,ijkl->kl', h, trans_indicator) * (1/dt)
    Q_rate[:, 1:] += h_contribution[:, 1:]  # from step 1 onwards, since step 0 has no previous state
    
    # reset Q_rate for step 0 (no transitions yet)
    Q_rate[:, 0] = 0
    
    # noise_rate[i,j,traj,step] = trans_indicator[i,j,traj,step] * (1/dt) - R[i,j] * state_indicator_j
    # first part: trans_indicator[i,j,traj,step] * (1/dt)
    noise_rate = trans_indicator * (1/dt)
    
    # second part: - R[i,j] * state_indicator_j
    for j in range(n_states):
        state_indicator_j = trajectories_onehot[:, :, j] # (n_traj, n_steps)
        for i in range(n_states):
            if i != j:
                # minus R[i,j] * state_indicator_j
                noise_rate[i, j, :, :] -= R[i, j] * state_indicator_j
    
    # calculate Lambda_b_rate and Lambda_f_rate
    Lambda_b_rate = noise_rate + noise_rate.transpose(1, 0, 2, 3)  # symmetric part
    Lambda_f_rate = (noise_rate - noise_rate.transpose(1, 0, 2, 3)) / 2  # anti-symmetric part
    
    return Q_rate, Lambda_b_rate, Lambda_f_rate


@nb.jit(nopython=True)
def compute_auto_covariance(Q_rate, dt):
    n_traj, n_steps = Q_rate.shape
    auto_cov = np.zeros(n_steps-1)
    for i in range(n_steps-1):
        auto_cov[i] = np.mean(Q_rate[:, i+1] * Q_rate[:, 1]) - np.mean(Q_rate[:, i+1]) * np.mean(Q_rate[:, 1])
    return auto_cov

@nb.jit(nopython=True)
def compute_cross_covariance(Q_rate, Lambda_rate, dt):
    n_states, _, n_traj, n_steps = Lambda_rate.shape
    cross_cov = np.zeros((n_states, n_states, n_steps-1))
    for i in range(n_states):
        for j in range(n_states):
            for k in range(n_steps-1):
                if i != j:
                    cross_cov[i, j, k] = np.mean(Q_rate[:, 1] * Lambda_rate[i, j, :, k+1]) - np.mean(Q_rate[:, 1]) * np.mean(Lambda_rate[i, j, :, k+1])
    return cross_cov

@nb.jit(nopython=True)
def compute_power_spectra(auto_cov, cross_cov_b, cross_cov_f, dt):
    """Fourier transform to compute power spectra"""

    n_states, _, n_steps = cross_cov_b.shape
    freq = np.logspace(-1.5, 2, num=150)  # frequency range for power spectrum
    S_Q = np.zeros(len(freq), dtype=np.complex128)
    S_Q_Lambda_b = np.zeros((n_states, n_states, len(freq)), dtype=np.complex128)
    S_Q_Lambda_f = np.zeros((n_states, n_states, len(freq)), dtype=np.complex128)

    for k in range(len(freq)):
        sinc = np.sinc(freq[k] * dt / 2)
        for i in range(n_steps):
            t = i * dt
            exp_factor = np.exp(1j * freq[k] * t)
            S_Q[k] += auto_cov[i] * exp_factor * dt
            for m in range(n_states):
                for n in range(n_states):
                    S_Q_Lambda_b[m, n, k] += cross_cov_b[m, n, i] * exp_factor * dt
                    S_Q_Lambda_f[m, n, k] += cross_cov_f[m, n, i] * exp_factor * dt
        S_Q[k] *= sinc ** 2
        for m in range(n_states):
            for n in range(n_states):
                S_Q_Lambda_b[m, n, k] *= sinc ** 2
                S_Q_Lambda_f[m, n, k] *= sinc ** 2
    S_Q *= 2

    # absolute value to get power spectrum
    return freq, np.abs(S_Q), np.abs(S_Q_Lambda_b), np.abs(S_Q_Lambda_f)

def compute_eta_values(freq, S_Q, S_Q_Lambda_b, S_Q_Lambda_f, a_ij, j_ij):
    n_states = a_ij.shape[0]
    eta_12a = np.zeros_like(freq)
    eta_12b = np.zeros_like(freq)
    eta_14a = np.zeros_like(freq)
    eta_14b = np.zeros_like(freq)

    for i in range(n_states):
        for j in range(n_states):
            if i < j:
                eta_12a += (S_Q_Lambda_b[i, j, :] ** 2) / (S_Q * a_ij[i, j])
                eta_12b += 4 * (S_Q_Lambda_f[i, j, :] ** 2) / (S_Q * a_ij[i, j])
                eta_14a += a_ij[i, j] * (S_Q_Lambda_b[i, j, :] ** 2) / (S_Q * j_ij[i, j] ** 2)
                eta_14b += 4 * (j_ij[i, j] ** 2) * (S_Q_Lambda_f[i, j, :] ** 2) / (S_Q * a_ij[i, j] ** 3)

    return eta_12a, eta_12b, eta_14a, eta_14b

np.random.seed(111)

# n states in the Markov chain
n_states = 3

# generate random transition rates for the Markov chain
R = np.random.exponential(1.0, size=(n_states, n_states))
np.fill_diagonal(R, 0)
# R = R / 10  # minimize transition rates to increase correlation time

# adjust diagonal elements to ensure rows sum to zero
for i in range(n_states):
    R[i, i] = -np.sum(R[:, i]) + R[i, i]

# calculate steady-state distribution
pi_ss = pre_eq(R)

a_ij, j_ij = compute_aij_jij(R, pi_ss)

# simulation parameters
T = 100
dt = 0.1
n_traj = 400000

# plotting results
fig, ax = plt.subplots(figsize=(4, 3))
n_system = 10
n_obserable = 10
alpha_scatter = 0.5
size_scatter = 5
for i in range(n_system):
    print(f"Plotting trajectory {i+1}/{n_system}")
    traj = generate_trajectories(R, pi_ss, T, dt, n_traj)
    for j in range(n_obserable):
        print(f"  Processing observable {j+1}/{n_obserable}")
        Q_rate, Lambda_b_rate, Lambda_f_rate = compute_rates(traj, R, dt)
        auto_cov_Q = compute_auto_covariance(Q_rate, dt)
        cross_cov_Q_Lambda_b = compute_cross_covariance(Q_rate, Lambda_b_rate, dt)
        cross_cov_Q_Lambda_f = compute_cross_covariance(Q_rate, Lambda_f_rate, dt)
        freq, S_Q, S_Q_Lambda_b, S_Q_Lambda_f = compute_power_spectra(auto_cov_Q, cross_cov_Q_Lambda_b, cross_cov_Q_Lambda_f, dt)
        eta_12a_i, eta_12b_i, eta_14a_i, eta_14b_i = compute_eta_values(freq, S_Q, S_Q_Lambda_b, S_Q_Lambda_f, a_ij, j_ij)

        if i == 0 and j == 0:
            ax.scatter(freq, eta_12a_i, color='blue', edgecolor='none', alpha=alpha_scatter, s=size_scatter, label=r'$\eta_{12a}$')
            ax.scatter(freq, eta_12b_i, color='red', edgecolor='none', alpha=alpha_scatter, s=size_scatter, label=r'$\eta_{12b}$')
            ax.scatter(freq, eta_14a_i, color='green', edgecolor='none', alpha=alpha_scatter, s=size_scatter, label=r'$\eta_{14a}$')
            ax.scatter(freq, eta_14b_i, color='orange', edgecolor='none', alpha=alpha_scatter, s=size_scatter, label=r'$\eta_{14b}$')
        else:
            ax.scatter(freq, eta_12a_i, color='blue', edgecolor='none', alpha=alpha_scatter, s=size_scatter)
            ax.scatter(freq, eta_12b_i, color='red', edgecolor='none', alpha=alpha_scatter, s=size_scatter)
            ax.scatter(freq, eta_14a_i, color='green', edgecolor='none', alpha=alpha_scatter, s=size_scatter)
            ax.scatter(freq, eta_14b_i, color='orange', edgecolor='none', alpha=alpha_scatter, s=size_scatter)
ax.set_xscale('log')
ax.axhline(1, color='k', linestyle='--')
ax.set_xlabel(r'$\omega$')
ax.set_ylabel(r'$\eta$ values')
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
