# Dimension-enlarged Newton scheme

This is the repository for the source codes accompanying the following papers  

> [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  
> [Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound](https://arxiv.org/abs/2605.01864), Mingwei Fu and Bin Shi.  
> [Numerical Construction of Quasi-Periodic Solutions for Nonlinear PDEs: I. Bounded Perturbation](https://arxiv.org/abs/2610.07262), Mingwei Fu and Bin Shi.  

These codes are implementations of the Dimension-enlarged Newton scheme for:  
- Finding full-dimensional quasi-periodic solutions of 1D undamped Duffing oscillator and 2D Henon-Heiles system  
- Finding lower-dimensional quasi-periodic solutions of 2D Henon-Heiles system and 3D Fermi-Pasta-Ulam (FPU) model
- Finding quasi-periodic solutions of 1D Nonlinear Schrödinger (NLS) equation and 1D Nonlinear Wave (NLW) equation  

## Requirements

- MATLAB  
- Python  

## File structure

The file structure is as follows.

### 1. Full-Dimensional 

This folder contains codes for paper: [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  

- Duffing  
  > main_duffing.m  
  *This is the main program with Dimension-enlarged Newton scheme for 1D undamped Duffing oscillator.*  

  > Newton_duffing_Solver.m    
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for 1D undamped Duffing oscillator.*  

- Henon-Heiles  
  > main_Henon.m  
  *This is the main program with Dimension-enlarged Newton scheme for 2D Henon-Heiles system.*  

  > Newton_HH_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for 2D Henon-Heiles system.*

### 2. Lower-Dimensional  

This folder contains codes for paper: [Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound](https://arxiv.org/abs/2605.01864), Mingwei Fu and Bin Shi.  

- Lower-Henon-Heiles-1  
  > main_Henon_low_dim.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 2D Henon-Heiles system, with the first torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 2D Henon-Heiles system, with the first torus perscribed.*  

- Lower-Henon-Heiles-2  
  > main_Henon_low_dim.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 2D Henon-Heiles system, with the second torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 2D Henon-Heiles system, with the second torus perscribed.*  

- Lower-FPU-periodic-1  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the first torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the first torus perscribed.*  

- Lower-FPU-periodic-2  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the second torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the second torus perscribed.*  

- Lower-FPU-periodic-3
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the third torus prescribed.*

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the third torus perscribed.*

- Lower-FPU-quasiperiodic-1  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the first and second tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the first and second tori prescribed.*  

- Lower-FPU-quasiperiodic-2  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the first and third tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the first and third tori prescribed.*  

- Lower-FPU-quasiperiodic-3  
  > main_FPU_3nodes_2tori.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3D FPU model, with the second and third tori prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3D FPU model, with the second and third tori prescribed.*  
  
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

  
