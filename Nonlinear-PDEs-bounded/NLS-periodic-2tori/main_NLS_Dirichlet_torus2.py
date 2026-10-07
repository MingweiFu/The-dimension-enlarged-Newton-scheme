# %% [1] Main part
# ========================================================================
# 2026/05/25  Mingwei Fu
# This main algorithm performs the Newton iteration to solve the
# 1d-NLS with Dirichlet b.c., for two chosen tori.
# ========================================================================




# 0. Clear workspace
try:
    from IPython import get_ipython
    if get_ipython() is not None:
        get_ipython().run_line_magic("reset", "-f")       # clear
        get_ipython().run_line_magic("clear", "")         # clc
        # get_ipython().run_line_magic("matplotlib", "qt")  # set interactive plotting mode to 'qt'
except Exception as e:
    print(f"IPython standard magic failed due to: {e}")




# 1. Set environment variables, directories and close figures
import os
os.environ["MKL_NUM_THREADS"] = "20"
os.environ["NUMEXPR_NUM_THREADS"] = "20"
os.environ["OMP_NUM_THREADS"] = "20"

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    SCRIPT_DIR = os.getcwd()

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "py_fig_1DNLS_D_2tori")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.close('all')




# 2. Import libraries
import numpy as np
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# Set global font and LaTeX style
plt.rcParams.update({
    'font.size': 25, 
    'mathtext.fontset': 'cm',
    'axes.labelsize': 25,
    'axes.titlesize': 25,
    'legend.fontsize': 20
})

from Newton_2d_NLS_Solver import Newton_2d_NLS_Solver
from Tensor_Expand_Padding import Tensor_Expand_Padding

# Closure factories 
def make_u_xt_2d(u_hat_r, mu_r, ks_vec):
    N, K1, K2 = np.meshgrid(ks_vec, ks_vec, ks_vec, indexing='ij')
    N_flat = N.ravel()[:, None]
    K1_flat = K1.ravel()[:, None]
    K2_flat = K2.ravel()[:, None]
    u_flat = u_hat_r.ravel()[None, :]
    omega_dot_k = K1_flat * mu_r[0] + K2_flat * mu_r[1]
    
    def u_xt(x, t):
        x_arr, t_arr = np.asarray(x), np.asarray(t)
        orig_shape = x_arr.shape
        phase = x_arr.ravel()[:, None] @ N_flat.T + t_arr.ravel()[:, None] @ omega_dot_k.T
        res = np.exp(1j * phase) @ u_flat.T
        return res.reshape(orig_shape)
    return u_xt

def make_u_n_t_2d(u_slice, mu_r, ks_vec):
    K1, K2 = np.meshgrid(ks_vec, ks_vec, indexing='ij')
    K1_flat = K1.ravel()[:, None]
    K2_flat = K2.ravel()[:, None]
    u_flat = u_slice.ravel()[None, :]
    omega_dot_k = K1_flat * mu_r[0] + K2_flat * mu_r[1]
    
    def u_n_t(t):
        t_arr = np.asarray(t)
        orig_shape = t_arr.shape
        phase = t_arr.ravel()[:, None] @ omega_dot_k.T
        res = np.exp(1j * phase) @ u_flat.T
        return res.reshape(orig_shape)
    return u_n_t

def make_u_n_t_zeros():
    return lambda t: np.zeros_like(np.asarray(t), dtype=np.complex128)




# 3. Settings
A         = 3                                  # Truncation parameter 
eps       = 0.5                                # Perturbation parameter
a         = np.array([1, 1])                   # Fixed amplitudes
R_max     = 2                                  # Maximum number of iterations
tol       = 1e-20                              # Error tolerance
torus_idx = np.array([1, 2])                   # Index of the chosen basic torus n1, n2
b         = np.atleast_1d(torus_idx).shape[0]  # Torus dimension, k \in Z^{b}, b = 2
A_0       = np.max(np.abs(torus_idx))          # Initial support parameter
m         = np.sqrt(2)                         # Parameter m to adjust the basic frequencies

def mu(n): 
    return n**2 + m                            # Basic frequency function 

# Note: If R_max = 2, the matrix size is 50653 × 50653 requiring 19.1 GB of memory




# 4. Initialization
L = int(2 * A_0 + 1)
S_term_p = np.array([
    [torus_idx[0], 1, 0],  # ( n1, e1)
    [torus_idx[1], 0, 1]   # ( n2, e2)
])

S_term_m = np.array([
    [-torus_idx[0], 1, 0], # (-n1, e1)
    [-torus_idx[1], 0, 1]  # (-n2, e2)
])

u_ini = np.zeros((L, L, L))  # u = ( u(n,k1,k2) ) 
u_ini[S_term_p[0, 0] + A_0, S_term_p[0, 1] + A_0, S_term_p[0, 2] + A_0] = a[0]
u_ini[S_term_m[0, 0] + A_0, S_term_m[0, 1] + A_0, S_term_m[0, 2] + A_0] = -a[0]

u_ini[S_term_p[1, 0] + A_0, S_term_p[1, 1] + A_0, S_term_p[1, 2] + A_0] = a[1]
u_ini[S_term_m[1, 0] + A_0, S_term_m[1, 1] + A_0, S_term_m[1, 2] + A_0] = -a[1]




# 5. Call the Newton solver
print("Starting Newton Iteration...")
u_history, fre_history, res_history, r = Newton_2d_NLS_Solver(A_0, A, eps, mu, a, R_max, tol, torus_idx, u_ini)
print(f"Converged/Finished at step: {r}")




# 6. Construct approximate functions u^{(r)}(x, t)
num_steps = r + 1                           # Number of iteration steps 
u_xt_steps       = [] 
u_hat_n1_t_steps = []
u_hat_n2_t_steps = []

fre_history_full = np.zeros((2, num_steps))
fre_history_full[:, 0] = mu(torus_idx)
fre_history_full[:, 1:] = fre_history[:, :r]

for rr in range(num_steps):
    u_hat_r = u_history[rr]     
    mu_r    = fre_history_full[:, rr] 
    
    Lr      = u_hat_r.shape[0]
    Ar_curr = (Lr - 1) // 2
    ks_curr = np.arange(-Ar_curr, Ar_curr + 1)
    
    u_xt_steps.append(make_u_xt_2d(u_hat_r, mu_r, ks_curr))
    u_hat_n1_t_steps.append(make_u_n_t_2d(u_hat_r[torus_idx[0] + Ar_curr, :, :], mu_r, ks_curr))
    u_hat_n2_t_steps.append(make_u_n_t_2d(u_hat_r[torus_idx[1] + Ar_curr, :, :], mu_r, ks_curr))

print("Data calculation completed! You can now run the plotting sections below.")










# %% [2] Figure 1: Error curve ||F(u^{r}; mu^{r+1})||
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
fig.canvas.manager.set_window_title('Figure 1: Residual Error Decay')
ax.semilogy(res_history[0, :num_steps], '-o', linewidth=2, markersize=8, clip_on=False)

ax.set_box_aspect(0.80) 
plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

# ax.grid(True, which='both', linestyle='--', linewidth=0.7, alpha=0.7)
ax.set_xlabel(r'iterating steps $r$')
ax.set_ylabel(r'residual $\| F(u^{(r)}; \omega^{(r)}) \|$')

plt.savefig(os.path.join(OUTPUT_DIR, "error_eqn.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_eqn.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_eqn.png"))
plt.close(fig)
# plt.show()










# %% [3] Figure 2: Convergence of neighboring steps ||u^{r+1} - u^{r}||
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')  
fig.canvas.manager.set_window_title('Figure 2: Neighboring Step Convergence')
x_ticks = np.arange(num_steps - 1)
y_vals = res_history[1, :num_steps - 1]
ax.semilogy(x_ticks, y_vals, '-s', linewidth=4, markersize=8, markerfacecolor='b', color='b', clip_on=False)

ax.set_box_aspect(0.80)
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(0, max(1, num_steps - 2))
ax.set_ylim(1e-3, 1e-0)
ax.set_xticks(x_ticks)
ax.set_yticks([1e-3, 1e-2, 1e-1, 1e-0])

ax.tick_params(axis='x', pad=10)  
ax.tick_params(axis='y', pad=10)

ax.set_xlabel(r'Iteration: $r$')
ax.set_ylabel(r'$\|\hat{u}^{(r+1)} - \hat{u}^{(r)}\|$')

plt.savefig(os.path.join(OUTPUT_DIR, "error_vec.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_vec.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_vec.png"))
plt.close(fig)
# plt.show()










# %% [4] Figure 3: Phase Space Trajectories of TANGENTIAL tori n1, n2 and NORMAL torus n3
T_endtime = 6                             # t in [0, 6]
t_eval = np.linspace(0, T_endtime, 2000) 
t_marks = [0, 3, 6]                       # Time instants for marking points
steps_to_plot = [0, num_steps - 1] 
labels = ['Initial', 'Final']
colors = ['b', 'r']




# =====================================================================
# 1. Phase plane of Tangential Torus n1
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
fig.canvas.manager.set_window_title('Figure 1: Phase Plane of Tangential Torus n1')
for i, rr in enumerate(steps_to_plot):
    u_eval = u_hat_n1_t_steps[rr](t_eval)
    ax.plot(np.real(u_eval), np.imag(u_eval), color=colors[i], linewidth=4, label=labels[i])
    
for i, rr in enumerate(steps_to_plot):    
    for tm in t_marks:
        u_mark = u_hat_n1_t_steps[rr](np.array([tm]))
        qm, pm = np.real(u_mark[0]), np.imag(u_mark[0])
        ax.plot(qm, pm, 'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=15)

        # ax.annotate(fr'$t = {tm:g}$', xy=(qm, pm), xytext=(10, 10), textcoords='offset points', color=colors[i], fontsize=25, weight='bold')
        
# Decoration and axis settings for tangential mode n1
ax.set_box_aspect(1) 
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(-1.5, 1.5)
ax .set_ylim(-1.5, 1.5)
ax.set_xticks(np.linspace(-1.5, 1.5, 5))
ax.set_yticks(np.linspace(-1.5, 1.5, 5))

ax.tick_params(axis='x', pad=10) 
ax.tick_params(axis='y', pad=10) 

# Mark positions at time t = 0, 3, 6 (initial)
ax.text(0.55, 0.03, r'$t = 0$', color=colors[0], fontsize=25, weight='bold')
ax.text(0.7, 0.85, r'$t = 3$', color=colors[0], fontsize=25, weight='bold')
ax.text(-0.7, 1.0, r'$t = 6$', color=colors[0], fontsize=25, weight='bold')

# Mark positions at time t = 0, 3, 6 (final)
ax.text(0.55, -0.12, r'$t = 0$', color=colors[1], fontsize=25, weight='bold')
ax.text(0.3, -1.2, r'$t = 3$', color=colors[1], fontsize=25, weight='bold')
ax.text(-1.3, -0.6, r'$t = 6$', color=colors[1], fontsize=25, weight='bold')

ax.legend(loc='upper right', fontsize=20)

plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n1_1.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n1_1.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n1_1.png"))
plt.close(fig)
# plt.show()




# =====================================================================
# 2. Phase plane of Tangential Torus n2
# =====================================================================
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
fig.canvas.manager.set_window_title('Figure 2: Phase Plane of Tangential Torus n2')
for i, rr in enumerate(steps_to_plot):
    u_eval = u_hat_n2_t_steps[rr](t_eval)
    ax.plot(np.real(u_eval), np.imag(u_eval), color=colors[i], linewidth=4, label=labels[i])
    
for i, rr in enumerate(steps_to_plot):    
    for tm in t_marks:
        u_mark = u_hat_n2_t_steps[rr](np.array([tm]))
        qm, pm = np.real(u_mark[0]), np.imag(u_mark[0])
        ax.plot(qm, pm, 'o', markerfacecolor=colors[i], markeredgecolor='k', markersize=15)
        
        # ax.annotate(fr'$t = {tm:g}$', xy=(qm, pm), xytext=(10, 10), textcoords='offset points', color=colors[i], fontsize=25, weight='bold')   

# Decoration and axis settings for tangential mode n2
ax.set_box_aspect(1)
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xticks(np.linspace(-1.5, 1.5, 5))
ax.set_yticks(np.linspace(-1.5, 1.5, 5))

ax.tick_params(axis='x', pad=10)
ax.tick_params(axis='y', pad=10)

# Mark positions at time t = 0, 3, 6 (initial)
ax.text(0.5, 0.06, r'$t = 0$', color=colors[0], fontsize=25, weight='bold')
ax.text(-1.3, -0.7, r'$t = 3$', color=colors[0], fontsize=25, weight='bold')
ax.text(0.65, 0.9, r'$t = 6$', color=colors[0], fontsize=25, weight='bold')

# Mark positions at time t = 0, 3, 6 (final)
ax.text(0.5, -0.12, r'$t = 0$', color=colors[1], fontsize=25, weight='bold')
ax.text(-0.1, 0.7, r'$t = 3$', color=colors[1], fontsize=25, weight='bold')
ax.text(-0.8, 0.3, r'$t = 6$', color=colors[1], fontsize=25, weight='bold')

ax.legend(loc='upper right', fontsize=20)

plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n2_2.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n2_2.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_tg_n2_2.png"))
plt.close(fig)
# plt.show()




# =====================================================================
# 3. Phase plane of Normal Torus n3 = 3
# =====================================================================
n3 = 3
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
fig.canvas.manager.set_window_title(f'Figure 3: Phase Plane of Normal Torus n3 = {n3}')
for i, rr in enumerate(steps_to_plot):
    u_hat_r = u_history[rr]
    mu_r    = fre_history_full[:, rr]
    Lr      = u_hat_r.shape[0]
    Ar_curr = (Lr - 1) // 2
    
    if np.abs(n3) > Ar_curr:
        func = make_u_n_t_zeros()
    else:
        ks_vec = np.arange(-Ar_curr, Ar_curr + 1)
        func = make_u_n_t_2d(u_hat_r[n3 + Ar_curr, :, :], mu_r, ks_vec)
        
    u_eval = func(t_eval)
    ax.plot(np.real(u_eval), np.imag(u_eval), color=colors[i], linewidth=4, label=labels[i])

for i, rr in enumerate(steps_to_plot):
    if i == 0:
        ax.plot(0, 0, 'o', markerfacecolor=colors[0], markeredgecolor='k', markersize=15)

        # ax.annotate(r'$t = 0, 3, 6$', xy=(0, 0), xytext=(15, 15), textcoords='offset points', color=colors[0], fontsize=25, weight='bold')
    else:
        for tm in t_marks:
            u_mark = func(np.array([tm]))
            qm, pm = np.real(u_mark[0]), np.imag(u_mark[0])
            ax.plot(qm, pm, 'o', markerfacecolor=colors[1], markeredgecolor='k', markersize=15)

            # ax.annotate(fr'$t = {tm:g}$', xy=(qm, pm), xytext=(15, 15), textcoords='offset points', color=colors[1], fontsize=25, weight='bold')
            
# Decoration and axis settings for normal mode n3
ax.set_box_aspect(1)
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(-0.2, 0.2)  
ax.set_ylim(-0.2, 0.2)
ax.set_xticks(np.linspace(-0.2, 0.2, 5))
ax.set_yticks(np.linspace(-0.2, 0.2, 5))

ax.tick_params(axis='x', pad=10)
ax.tick_params(axis='y', pad=10)

# Mark positions at time t = 0, 3, 6 (initial)
ax.text(-0.04, -0.025, r'$t = 0, 3, 6$', color=colors[0], fontsize=25, weight='bold')

# Mark positions at time t = 0, 3, 6 (final)
ax.text(-0.16, 0.0, r'$t = 0$', color=colors[1], fontsize=25, weight='bold')
ax.text(0.05, 0.1, r'$t = 3$', color=colors[1], fontsize=25, weight='bold')
ax.text(0.07, -0.085, r'$t = 6$', color=colors[1], fontsize=25, weight='bold')

ax.legend(loc='upper right', fontsize=20)

plt.savefig(os.path.join(OUTPUT_DIR, "traj_nor_n3_3.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_nor_n3_3.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "traj_nor_n3_3.png"))
plt.close(fig)
# plt.show()










# %% [5] Figure 4: 3D Spatiotemporal evolution of |u(x,t)|^2
x_grid = np.linspace(0, np.pi, 200)  # x in [0, pi]
t_grid = np.linspace(0, 6, 200)      # t in [0, 6]
X_mat, T_mat = np.meshgrid(x_grid, t_grid, indexing='xy')

u_ini_eval = u_xt_steps[0](X_mat, T_mat)
u_ini_sq_norm = np.abs(u_ini_eval)**2

u_final_eval = u_xt_steps[-1](X_mat, T_mat)
u_final_sq_norm = np.abs(u_final_eval)**2

for label, data in zip(['Initial', 'Final'], [u_ini_sq_norm, u_final_sq_norm]):
    fig = plt.figure(figsize=(10, 8), facecolor='w')
    fig.canvas.manager.set_window_title(f'3D Surface of |u(x,t)|^2 ({label})')
    ax = fig.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X_mat, T_mat, data, cmap='jet', 
                           edgecolor='none', rstride=1, cstride=1, 
                           antialiased=False, shade=False, vmin=0, vmax=14)
    
    ax.view_init(elev=15, azim=-45)
    plt.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.95)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 6)
    ax.set_zlim(0, 14)
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_yticks(np.linspace(0, 6, 5))
    ax.set_zticks(np.linspace(0, 14, 5))
    ax.set_xticklabels([f"{val:.2f}" for val in ax.get_xticks()])
    
    ax.tick_params(axis='x', pad=20)  
    ax.tick_params(axis='y', pad=5)   
    ax.tick_params(axis='z', pad=20)  
    
    ax.set_xlabel(r'$x$', fontsize=25, labelpad=35)
    ax.set_ylabel(r'$t$', fontsize=25, labelpad=15)
    ax.set_zlabel(r'$|u(x,t)|^2$', fontsize=25, labelpad=35)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_figure_{label}.eps"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_figure_{label}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_figure_{label}.png"))
    plt.close(fig)
    # plt.show()










# %% [6] Figure 5: 2D Spatiotemporal heatmap of |u(x,t)|^2 
x_grid = np.linspace(0, np.pi, 200)  # x in [0, pi]
t_grid = np.linspace(0, 6, 200)      # t in [0, 6]
X_mat, T_mat = np.meshgrid(x_grid, t_grid, indexing='xy')

u_ini_eval = u_xt_steps[0](X_mat, T_mat)
u_ini_sq_norm = np.abs(u_ini_eval)**2

u_final_eval = u_xt_steps[-1](X_mat, T_mat)
u_final_sq_norm = np.abs(u_final_eval)**2

for label, data in zip(['Initial', 'Final'], [u_ini_sq_norm, u_final_sq_norm]):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
    fig.canvas.manager.set_window_title(f'2D Heatmap of |u(x,t)|^2 ({label})')
    mesh = ax.pcolormesh(X_mat, T_mat, data, cmap='jet', shading='gouraud', vmin=0, vmax=14)
    
    ax.set_box_aspect(1)
    plt.subplots_adjust(left=0.15, right=0.90, bottom=0.15, top=0.85)
    
    cbar = fig.colorbar(mesh, ax=ax, pad=0.04)
    cbar.set_ticks([0, 3.5, 7, 10.5, 14]) 
    cbar.ax.tick_params(labelsize=20)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 6)
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_yticks(np.linspace(0, 6, 5))
    ax.set_xticklabels([f"{val:.2f}" for val in ax.get_xticks()])
    
    ax.tick_params(axis='x', pad=15)  
    ax.tick_params(axis='y', pad=5)   
    
    ax.set_xlabel(r'$x$', fontsize=25)
    ax.set_ylabel(r'$t$', fontsize=25)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_heat_{label}.eps"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_heat_{label}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"sq_norm_heat_{label}.png"))
    plt.close(fig)
    # plt.show()










# %% [7] Figure 6: Spatial profiles of |u(x,t)|^2 at different times
x_eval  = np.linspace(0, np.pi, 500)  # x in [0, pi]
t_marks = [0, 3, 6]                   # t = 0, 3, 6
colors_profile = ['k', 'b', 'r']      # color = black, blue, red

for idx, label in zip([0, -1], ['Initial', 'Final']):
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
    fig.canvas.manager.set_window_title(f'Spatial Profiles ({label})')
    
    for i, t_val in enumerate(t_marks):
        T_eval = np.full_like(x_eval, t_val)
        u_val = u_xt_steps[idx](x_eval, T_eval)
        ax.plot(x_eval, np.abs(u_val)**2, color=colors_profile[i], linewidth=4, label=fr'$t = {t_val}$')
    
    ax.set_box_aspect(0.8)
    plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 14)
    ax.set_xticks(np.linspace(0, np.pi, 5))
    ax.set_yticks(np.linspace(0, 14, 5))
    ax.set_xticklabels([f"{val:.2f}" for val in ax.get_xticks()])
    
    ax.tick_params(axis='x', pad=15)  
    ax.tick_params(axis='y', pad=5)
    
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(r'$|u(x,t)|^2$')
    ax.legend(loc='upper right', fontsize=20)
    
    plt.savefig(os.path.join(OUTPUT_DIR, f"evolve_sq_norm_{label}.eps"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"evolve_sq_norm_{label}.pdf"))
    plt.savefig(os.path.join(OUTPUT_DIR, f"evolve_sq_norm_{label}.png"))
    plt.close(fig)
    # plt.show()










# %% [8] GIF 7: Generate GIF animations for spatiotemporal evolution of |u(x,t)|^2
print("Generating GIF animations...")
x_eval = np.linspace(0, np.pi, 500)  
T_endtime = 6    
num_frames = 400                       
t_frames = np.linspace(0, T_endtime, num_frames)

for idx, label in zip([0, -1], ['Initial', 'Final']):
    fig_anim, ax_anim = plt.subplots(figsize=(10, 8), facecolor='w')
    fig_anim.canvas.manager.set_window_title(f'Animation of |u(x,t)|^2 ({label})')
    
    line_color = 'b' if label == 'Initial' else 'r'
    line_anim, = ax_anim.plot([], [], color=line_color, linewidth=4)
    
    ax_anim.set_box_aspect(1)
    ax_anim.set_position([0.11, 0.13, 0.8, 0.8]) 
    
    ax_anim.set_xlim(0, np.pi)
    ax_anim.set_ylim(0, 14)
    ax_anim.set_xticks(np.linspace(0, np.pi, 5))
    ax_anim.set_yticks(np.linspace(0, 14, 5))
    ax_anim.set_xticklabels([f"{val:.2f}" for val in ax_anim.get_xticks()])
    ax_anim.set_yticklabels([f"{val:.1f}" for val in ax_anim.get_yticks()])
    
    ax_anim.tick_params(axis='x', pad=15)  
    ax_anim.tick_params(axis='y', pad=5)   
    
    ax_anim.set_xlabel(r'$x$')
    ax_anim.set_ylabel(r'$|u(x,t)|^2$')

    def init():
        line_anim.set_data([], [])
        return line_anim,

    def update(frame):
        t_curr = t_frames[frame]
        T_eval = np.full_like(x_eval, t_curr)
        
        u_val = u_xt_steps[idx](x_eval, T_eval)
        line_anim.set_data(x_eval, np.abs(u_val)**2)
        
        ax_anim.set_title(fr'{label} Step: $t = {t_curr:.2f}$', fontsize=25, pad=15)
        return line_anim,

    ani = FuncAnimation(fig_anim, update, frames=num_frames, init_func=init, blit=False)
    gif_name = f'Evolution_u_sq_norm_{label}.gif'
    gif_path = os.path.join(OUTPUT_DIR, gif_name)
    ani.save(gif_path, writer='pillow', fps=50)
    print(f"Successfully saved: {gif_name}")
    plt.close(fig_anim)

print("All animations generated successfully!")










# %% [9] Figure 8: Difference between consecutive approximate frequencies |omega^{(r+1)} - omega^{(r)}|
fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
fre_diff = np.diff(fre_history_full, axis=1)
fre_diff_norm = np.linalg.norm(fre_diff, ord=2, axis=0)
x_ticks_diff = np.arange(num_steps - 1)
ax.semilogy(x_ticks_diff, fre_diff_norm, '-s', linewidth=4, markersize=8, markerfacecolor='b', color='b', clip_on=False)

ax.set_box_aspect(0.80) 
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(0, num_steps - 2)
ax.set_ylim(1e-2, 1e+1)
ax.set_xticks(x_ticks_diff)
ax.set_yticks([1e-2, 1e-1, 1e-0, 1e+1])

ax.tick_params(axis='x', pad=10)  
ax.tick_params(axis='y', pad=10)

ax.set_xlabel(r'Iteration: $r$', fontsize=25)
ax.set_ylabel(r'$|\omega^{(r+1)} - \omega^{(r)}|$', fontsize=25)

plt.savefig(os.path.join(OUTPUT_DIR, "error_freq.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_freq.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "error_freq.png"))
plt.close(fig)
# plt.show()









# %% [10] Figure 9: Difference of approximate solutions ||u^{(r+1)}(3) - u^{(r)}(3)||_{L^2} at fixed time t = 3
t_fixed = 3
diff_norms = np.zeros(num_steps - 1)

for r_idx in range(num_steps - 1):
    u_hat_curr = u_history[r_idx]
    mu_curr    = fre_history_full[:, r_idx]
    Ar_curr    = (u_hat_curr.shape[0] - 1) // 2
    
    u_hat_next = u_history[r_idx + 1]
    mu_next    = fre_history_full[:, r_idx + 1]
    Ar_next    = (u_hat_next.shape[0] - 1) // 2
    
    u_hat_curr_padded = Tensor_Expand_Padding(u_hat_curr, Ar_curr, Ar_next)
    
    ks_vec = np.arange(-Ar_next, Ar_next + 1)
    K1, K2 = np.meshgrid(ks_vec, ks_vec, indexing='ij')
    phase_curr = np.exp(1j * (K1 * mu_curr[0] + K2 * mu_curr[1]) * t_fixed)
    phase_next = np.exp(1j * (K1 * mu_next[0] + K2 * mu_next[1]) * t_fixed)
    
    # For each n, sum over k1, k2
    u_curr_t = np.tensordot(u_hat_curr_padded, phase_curr, axes=([1, 2], [0, 1]))
    u_next_t = np.tensordot(u_hat_next,        phase_next, axes=([1, 2], [0, 1]))
    
    diff_norms[r_idx] = np.linalg.norm(u_next_t - u_curr_t, ord=2)

fig, ax = plt.subplots(figsize=(10, 8), facecolor='w')
ax.semilogy(x_ticks_diff, diff_norms, '-s', linewidth=4, markersize=8, markerfacecolor='b', color='b', clip_on=False)

ax.set_box_aspect(0.80)
plt.subplots_adjust(left=0.17, right=0.85, bottom=0.10, top=0.90)

ax.set_xlim(0, num_steps - 2)
ax.set_ylim(1e-2, 1e+1)
ax.set_xticks(x_ticks_diff)
ax.set_yticks([1e-2, 1e-1, 1e-0, 1e+1])

ax.tick_params(axis='x', pad=10)
ax.tick_params(axis='y', pad=10)

ax.set_xlabel(r'Iteration: $r$', fontsize=25)
ax.set_ylabel(r'$\|u^{(r+1)}(3) - u^{(r)}(3)\|_{L^2}$', fontsize=25)

plt.savefig(os.path.join(OUTPUT_DIR, "error.eps"))
plt.savefig(os.path.join(OUTPUT_DIR, "error.pdf"))
plt.savefig(os.path.join(OUTPUT_DIR, "error.png"))
plt.close(fig)
# plt.show()









# %%
