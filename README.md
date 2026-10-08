# Dimension-enlarged Newton scheme

This is the repository for the source codes accompanying the following papers  

> [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  
> [Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound](https://arxiv.org/abs/2605.01864), Mingwei Fu and Bin Shi.  
> [Numerical Construction of Quasi-Periodic Solutions for Nonlinear PDEs: I. Bounded Perturbation](https://arxiv.org/abs/2610.07262), Mingwei Fu and Bin Shi.  

These codes are implementations of the Dimension-enlarged Newton scheme for:  
- Finding full-dimensional quasi-periodic solutions of 1D undamped Duffing oscillator and 2D Henon-Heiles system  
- Finding lower-dimensional quasi-periodic solutions of 2D Henon-Heiles system and 3D Fermi-Pasta-Ulam (FPU) model
- Finding quasi-periodic solutions of 1D Nonlinear Schrödinger (NLS) equation and 1D Nonlinear Wave (NLW) equation  

---  

## Requirements

### Finite-dimensional systems  

The codes for the finite-dimensional systems are implemented in MATLAB.  

- MATLAB

### Nonlinear PDEs

The codes for the nonlinear PDEs are implemented in Python.  

- Python 3
- NumPy  
- SciPy  
- Matplotlib  
- Pillow

---  

## Computational environment  

The MATLAB and Python codes were executed on different machines.  

### Personal computer  

The finite-dimensional MATLAB codes were executed on a personal computer with the following configuration:  

- Operating system: Windows 11 Home, Version 25H2, 64-bit  
- CPU: 13th Gen Intel(R) Core(TM) i9-13900HX @ 2.20 GHz  
- RAM: 16 GB  
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB  

### Computing server  

The nonlinear PDE Python codes were executed on a computing server with the following configuration:  

- Operating system: Ubuntu 22.04.5 LTS, 64-bit  
- CPU: Intel(R) Xeon(R) w7-3445, 20 cores and 40 threads  
- RAM: 512 GB  
- GPU: NVIDIA GeForce RTX 4090 D, 24 GB  

---  

## File structure

The file structure is as follows.

### 1. Full-Dimensional 

This folder contains the codes for the paper: [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  

- Duffing  
  > main_duffing.m  
  *This is the main program with the Dimension-enlarged Newton scheme for the 1D undamped Duffing oscillator.*  

  > Newton_duffing_Solver.m    
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for the 1D undamped Duffing oscillator.*  

- Henon-Heiles  
  > main_Henon.m  
  *This is the main program with the Dimension-enlarged Newton scheme for the 2D Henon-Heiles system.*  

  > Newton_HH_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for the 2D Henon-Heiles system.*

### 2. Lower-Dimensional  

This folder contains the codes for the paper: [Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound](https://arxiv.org/abs/2605.01864), Mingwei Fu and Bin Shi.  

- Lower-Henon-Heiles-1  
  > main_Henon_low_dim.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 2D Henon-Heiles system, with the first torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 2D Henon-Heiles system, with the first torus prescribed.*  

- Lower-Henon-Heiles-2  
  > main_Henon_low_dim.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 2D Henon-Heiles system, with the second torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 2D Henon-Heiles system, with the second torus prescribed.*  

- Lower-FPU-periodic-1  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the first torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the first torus prescribed.*  

- Lower-FPU-periodic-2  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the second torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the second torus prescribed.*  

- Lower-FPU-periodic-3
  > main_FPU_3nodes_1torus.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the third torus prescribed.*

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the third torus prescribed.*

- Lower-FPU-quasiperiodic-1  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the first and second tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the first and second tori prescribed.*  

- Lower-FPU-quasiperiodic-2  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the first and third tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the first and third tori prescribed.*  

- Lower-FPU-quasiperiodic-3  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with the Dimension-enlarged Newton scheme for lower-dimensional solutions of the 3D FPU model, with the second and third tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of the 3D FPU model, with the second and third tori prescribed.*  

### 3. Nonlinear-PDEs-bounded  

This folder contains the codes for the paper: [Numerical Construction of Quasi-Periodic Solutions for Nonlinear PDEs: I. Bounded Perturbation](https://arxiv.org/abs/2610.07262), Mingwei Fu and Bin Shi.  

- NLS-Dirichlet-1torus  
  > main_NLS_Dirichlet_torus1.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time periodic solutions of the 1D NLS with Dirichlet boundary conditions, taking n=1 as the tangential mode.*  

  > Newton_1d_NLS_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Matrix_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time periodic solutions of the 1D NLS with Dirichlet boundary conditions, taking n=1 as the tangential mode.*  

- NLS-Dirichlet-2tori  
  > main_NLS_Dirichlet_torus2.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time quasi-periodic solutions of the 1D NLS with Dirichlet boundary conditions, taking n=1 and n=2 as the tangential modes.*  

  > Newton_2d_NLS_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Tensor_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time quasi-periodic solutions of the 1D NLS with Dirichlet boundary conditions, taking n=1 and n=2 as the tangential modes.*

- NLS-periodic-1torus  
  > main_NLS_Periodic_torus1.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time periodic solutions of the 1D NLS with periodic boundary conditions, taking n=1 as the tangential mode.*  

  > Newton_1d_NLS_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Matrix_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time periodic solutions of the 1D NLS with periodic boundary conditions, taking n=1 as the tangential mode.*  

- NLS-periodic-2tori  
  > main_NLS_Periodic_torus2.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time quasi-periodic solutions of the 1D NLS with periodic boundary conditions, taking n=1 and n=2 as the tangential modes.*  

  > Newton_2d_NLS_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Tensor_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time quasi-periodic solutions of the 1D NLS with periodic boundary conditions, taking n=1 and n=2 as the tangential modes.*

- NLW-Dirichlet-1torus  
  > main_NLW_Dirichlet_torus1.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time periodic solutions of the 1D NLW with Dirichlet boundary conditions, taking n=1 as the tangential mode.*  

  > Newton_1d_NLW_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Matrix_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time periodic solutions of the 1D NLW with Dirichlet boundary conditions, taking n=1 as the tangential mode.*  

- NLW-Dirichlet-2tori  
  > main_NLW_Dirichlet_torus2.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time quasi-periodic solutions of the 1D NLW with Dirichlet boundary conditions, taking n=1 and n=2 as the tangential modes.*  

  > Newton_2d_NLW_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Tensor_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time quasi-periodic solutions of the 1D NLW with Dirichlet boundary conditions, taking n=1 and n=2 as the tangential modes.*

- NLW-periodic-1torus  
  > main_NLW_Periodic_torus1.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time periodic solutions of the 1D NLW with periodic boundary conditions, taking n=1 as the tangential mode.*  

  > Newton_1d_NLW_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Matrix_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time periodic solutions of the 1D NLW with periodic boundary conditions, taking n=1 as the tangential mode.*  

- NLW-periodic-2tori  
  > main_NLW_Periodic_torus2.py  
  *This is the main program with the Dimension-enlarged Newton scheme for time quasi-periodic solutions of the 1D NLW with periodic boundary conditions, taking n=1 and n=2 as the tangential modes.*  

  > Newton_2d_NLW_Solver.py  
  > P_eqn_calcu.py  
  > Q_eqn.py  
  > T_construct_opt.py  
  > Tensor_Expand_Padding.py  
  > Vectorization_Process.py  
  > Vectorization_Process_Inverse.py  
  *These are functions used in the main program for time quasi-periodic solutions of the 1D NLW with periodic boundary conditions, taking n=1 and n=2 as the tangential modes.*  

---  

## Citing

If you want to use `Dimension-enlarged Newton scheme` for acadamic proposes, please cite the main references as follows:

```
@article{Fu2026numerical,
  title={Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators},
  author={Fu, Mingwei and Shi, Bin},
  journal={arXiv preprint arXiv:2602.16275},
  year={2026}
}
```

```
@article{Fu2026numerical,
  title={Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound},
  author={Fu, Mingwei and Shi, Bin},
  journal={arXiv preprint arXiv:2605.01864},
  year={2026}
}
```

  
