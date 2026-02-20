import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

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
    """calculate the steady-state distribution by finding the left eigenvector corresponding to eigenvalue 0"""
    n = R.shape[0]

    # find the left eigenvector corresponding to eigenvalue 0
    eigenvalues, eigenvectors = np.linalg.eig(R)

    # find the index of the eigenvalue closest to zero
    idx = np.argmin(np.abs(eigenvalues))

    # normalize the corresponding eigenvector to get the steady-state distribution
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
    """generate trajectories using Gillespie algorithm"""
    n_states = R.shape[0]
    n_steps = int(T / dt)
    trajectories = np.zeros((n_traj, n_steps), dtype=np.int32)
    
    # calculate transition probability matrix P for discrete-time simulation
    P = np.zeros((n_states, n_states))
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                P[i, j] = R[i, j] * dt
    for i in range(n_states):
        P[i, i] = 1 - np.sum(P[:, i])
    
    for traj in range(n_traj):
        # generate initial state based on steady-state distribution
        r = np.random.rand()
        state = np.searchsorted(np.cumsum(pi_ss), r)
        for step in range(n_steps):
            trajectories[traj, step] = state
            r = np.random.rand()
            state = np.searchsorted(np.cumsum(P[:, state]), r)
    
    return trajectories

@nb.jit(nopython=True)
def compute_rates(trajectories, R, dt, g=None, h=None):
    """calculate Q_rate, Lambda_b_rate, and Lambda_f_rate for each trajectory and time step"""
    n_traj, n_steps = trajectories.shape
    n_states = R.shape[0]

    Q_rate = np.zeros((n_traj, n_steps))
    noise_rate = np.zeros((n_states, n_states, n_traj, n_steps))
    Lambda_b_rate = np.zeros((n_states, n_states, n_traj, n_steps))
    Lambda_f_rate = np.zeros((n_states, n_states, n_traj, n_steps))

    if g is None:
        g = np.random.randn(n_states)
    if h is None:
        h = np.random.randn(n_states, n_states)
        h = (h - h.T) / 2  # anti-symmetrize h

    for traj in range(n_traj):
        for step in range(1, n_steps):
            state = trajectories[traj, step]
            Q_rate[traj, step] = g[state]
            for i in range(n_states):
                for j in range(n_states):
                    if i != j:
                        Q_rate[traj, step] += h[i, j] * (1/dt if trajectories[traj, step - 1] == j and trajectories[traj, step] == i else 0)
                        noise_rate[i, j, traj, step] = (1/dt if trajectories[traj, step - 1] == j and trajectories[traj, step] == i else 0) - R[i, j] * (1 if trajectories[traj, step] == j else 0)
            Lambda_b_rate[:, :, traj, step] = noise_rate[:, :, traj, step] + noise_rate[:, :, traj, step].T
            Lambda_f_rate[:, :, traj, step] = ( noise_rate[:, :, traj, step] - noise_rate[:, :, traj, step].T ) / 2
    
    return Q_rate, Lambda_b_rate, Lambda_f_rate

def compute_rates(trajectories, R, dt, g=None, h=None):
    """vectorized version of compute_rates function"""
    n_traj, n_steps = trajectories.shape
    n_states = R.shape[0]
    
    # initialize rate arrays
    Q_rate = np.zeros((n_traj, n_steps))
    noise_rate = np.zeros((n_states, n_states, n_traj, n_steps))
    
    # set g and h if not provided
    if g is None:
        # g = np.random.randn(n_states)
        g = np.zeros(n_states)  # set g to zero for simplicity
    if h is None:
        h = np.zeros((n_states, n_states))  # set h to zero for simplicity
        h[0, 1] = 1.0
        h[1, 0] = -1.0  # anti-symmetric h
    
    # create one-hot encoding for states: shape (n_traj, n_steps, n_states)
    trajectories_onehot = np.eye(n_states)[trajectories]
    
    trajectories_prev_onehot = trajectories_onehot[:, :-1, :]
    trajectories_curr_onehot = trajectories_onehot[:, 1:, :]
    
    # calculate transition indicator for all i, j, traj, step
    # trans_indicator[i, j, traj, step]: 1 if there's a transition from j to i at (traj, step), else 0
    trans_indicator = np.zeros((n_states, n_states, n_traj, n_steps))
    
    # vectorized computation of trans_indicator
    # step begin from 1 since step=0 has no previous state
    for i in range(n_states):
        for j in range(n_states):
            if i != j:
                # create a mask for transitions from j to i
                trans_indicator[i, j, :, 1:] = (
                    trajectories_curr_onehot[:, :, i] * 
                    trajectories_prev_onehot[:, :, j]
                )
    
    # calculate g part: state contribution
    Q_rate = g[trajectories]
    
    # calculate h part: transition contribution
    # h[i,j] * trans_indicator[i,j,traj,step] * (1/dt)
    h_contribution = np.einsum('ij,ijkl->kl', h, trans_indicator) * (1/dt)
    Q_rate[:, 1:] += h_contribution[:, 1:]  # start from step=1 since step=0 has no previous state
    
    # reset Q_rate = 0 for step=0 since no transition happens at the first step
    Q_rate[:, 0] = 0
    
    noise_rate = trans_indicator * (1/dt)
    
    # calculate R[i,j] * state_indicator_j for all i,j, traj, step
    for j in range(n_states):
        # state_indicator_j: indicates if the current state is j for each trajectory and time step
        state_indicator_j = trajectories_onehot[:, :, j]  # (n_traj, n_steps)
        
        for i in range(n_states):
            if i != j:
                noise_rate[i, j, :, :] -= R[i, j] * state_indicator_j
    
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
    '''calculate power spectra using numerical integration'''
    n_states, _, n_steps = cross_cov_b.shape
    freq = np.logspace(-1.5, 2, num=150)
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

    return freq, np.abs(S_Q), np.abs(S_Q_Lambda_b), np.abs(S_Q_Lambda_f)

def compute_Response(freq, S_Q_Lambda_b, S_Q_Lambda_f, J_b_zeta, J_f_xi):
    n_states = S_Q_Lambda_b.shape[0]
    R2_zeta = np.zeros_like(freq)
    R2_xi = np.zeros_like(freq)

    for i in range(n_states):
        for j in range(n_states):
            if i < j:
                R2_zeta += (S_Q_Lambda_b[i, j, :] * J_b_zeta[i, j] ) ** 2
                R2_xi += (S_Q_Lambda_f[i, j, :] * J_f_xi[i, j] ) ** 2

    return R2_zeta, R2_xi

np.random.seed(111)

n_states = 3
# rate matrix parameters
k_01, k_12, k_20 = 165.0, 165.0, 165.0
k_10, k_21, k_02 = 165.0, 165.0, 165.0
F_01, F_12, F_20 = -20.0, 17.5, 2.5
F_10, F_21, F_02 = 20.0, -17.5, -2.5
mu_01, W_12, mu_20 = -18.0, -17.0, 0.0
mu_10, W_21, mu_02 = 18.0, 17.0, 0.0
R = np.array([[0,                            k_01*np.exp((-F_01+mu_01)/2), k_02*np.exp((-F_02-mu_02)/2)],
              [k_10*np.exp((-F_10+mu_10)/2), 0,                            k_12*np.exp((-F_12-W_12)/2) ],
              [k_20*np.exp((-F_20-mu_20)/2), k_21*np.exp((-F_21-W_21)/2),  0                           ]], dtype=np.float64)
for i in range(n_states):
    R[i, i] = -np.sum(R[:, i]) + R[i, i]

R = R / 100.0  # scale rates to make the system slower and easier to simulate

# Jacobian matrices for response calculations
J_b_zeta1 = np.array([[0, 1.1, 1.2],
                     [1.1, 0, 1.3],
                     [1.2, 1.3, 0]], dtype=np.float64)

J_b_zeta2 = np.array([[0, 1.3, 1.3],
                     [1.3, 0, 1.3],
                     [1.3, 1.3, 0]], dtype=np.float64)

J_b_zeta3 = np.array([[0, 1.0, 1.0],
                     [1.0, 0, 1.3],
                     [1.0, 1.3, 0]], dtype=np.float64)

J_f_xi = np.array([[0, 1.1, 1.2],
                   [1.1, 0, 1.3],
                   [1.2, 1.3, 0]], dtype=np.float64)


# calculate steady-state distribution, a_ij, j_ij, activity, and entropy production
pi_ss = pre_eq(R)

a_ij, j_ij = compute_aij_jij(R, pi_ss)

activity = np.sum(a_ij) / 2
entropy_production = 0
for i in range(n_states):
    for j in range(n_states):
        if i < j:
            entropy_production += j_ij[i, j] * np.log((a_ij[i, j] + j_ij[i, j]) / (a_ij[i, j] - j_ij[i, j]))



T = 50.0
dt = 0.1
n_traj = 400000

traj = generate_trajectories(R, pi_ss, T, dt, n_traj)
Q_rate, Lambda_b_rate, Lambda_f_rate = compute_rates(traj, R, dt)
auto_cov_Q = compute_auto_covariance(Q_rate, dt)
cross_cov_Q_Lambda_b = compute_cross_covariance(Q_rate, Lambda_b_rate, dt)
cross_cov_Q_Lambda_f = compute_cross_covariance(Q_rate, Lambda_f_rate, dt)
freq, S_Q, S_Q_Lambda_b, S_Q_Lambda_f = compute_power_spectra(auto_cov_Q, cross_cov_Q_Lambda_b, cross_cov_Q_Lambda_f, dt)
R2_zeta1, R2_xi = compute_Response(freq, S_Q_Lambda_b, S_Q_Lambda_f, J_b_zeta1, J_f_xi)
R2_zeta2, R2_xi = compute_Response(freq, S_Q_Lambda_b, S_Q_Lambda_f, J_b_zeta2, J_f_xi)
R2_zeta3, R2_xi = compute_Response(freq, S_Q_Lambda_b, S_Q_Lambda_f, J_b_zeta3, J_f_xi)

SNR_zeta1 = R2_zeta1 / S_Q
SNR_zeta2 = R2_zeta2 / S_Q
SNR_zeta3 = R2_zeta3 / S_Q
SNR_xi = R2_xi / S_Q

# Savitzky-Golay filter for smoothing the noisy SNR curves
window_length = 7  # window length (must be odd and less than the size of the input)
polyorder = 2      # polynomial order for fitting (must be less than window_length)
SNR_zeta1_smooth = savgol_filter(SNR_zeta1, window_length, polyorder)
SNR_zeta2_smooth = savgol_filter(SNR_zeta2, window_length, polyorder)
SNR_zeta3_smooth = savgol_filter(SNR_zeta3, window_length, polyorder)

fig, ax = plt.subplots(figsize=(3, 2.8))
ax.plot(freq, SNR_zeta1_smooth, color='orange', label=r'SNR$_{\zeta}^{(1)}$', linestyle='-')
ax.plot(freq, SNR_zeta2_smooth, color='blue', label=r'SNR$_{\zeta}^{(2)}$', linestyle='-.')
ax.plot(freq, SNR_zeta3_smooth, color='red', label=r'SNR$_{\zeta}^{(3)}$', linestyle=':')
ax.hlines(np.max(J_b_zeta1)**2 * entropy_production / 2, freq[0], freq[-1], color='black', linestyle='--', label=r'bound')
ax.set_xlabel(r'$\omega$')
ax.set_xscale('log')
ax.grid(True, ls='--', lw=0.5)
ax.legend(frameon=False, fontsize=8)

plt.tight_layout()
plt.show()
