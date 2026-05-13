# Dimension-enlarged Newton scheme

This is the repository for the source codes of the paper

> [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  
> [Numerical Construction of Elliptic Lower-Dimensional Quasi-Periodic Solutions with a Priori Bound](https://arxiv.org/abs/2605.01864), Mingwei Fu and Bin Shi.  

These codes are implementations of Dimension-enlarged Newton scheme for:  
- Finding full-dimensional quasi-periodic solutions of 1-d undamped Duffing oscillator and 2-d Henon-Heiles system  
- Finding lower-dimensional quasi-periodic solutions of 2-d Henon-Heiles system and 3-d Fermi-Pasta-Ulam (FPU) model

## Requirements

- MATLAB

## File structure

The file structure is as follows.

- Duffing  
  > main_duffing.m  
  *This is the main program with Dimension-enlarged Newton scheme for 1-d undamped Duffing oscillator.*  

  > Newton_duffing_Solver.m    
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for 1-d undamped Duffing oscillator.*  

- Henon-Heiles  
  > main_Henon.m  
  *This is the main program with Dimension-enlarged Newton scheme for 2-d Henon-Heiles system.*  

  > Newton_HH_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for 2-d Henon-Heiles system.*

- Lower-Henon-Heiles-1  
  > main_Henon_low_dim.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 2-d Henon-Heiles system, with the first torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 2-d Henon-Heiles system, with the first torus perscribed.*  

- Lower-Henon-Heiles-2  
  > main_Henon_low_dim.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 2-d Henon-Heiles system, with the second torus prescribed.*  

  > Newton_HH_low_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 2-d Henon-Heiles system, with the second torus perscribed.*  

- Lower-FPU-periodic-1  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3-d FPU model, with the first torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3-d FPU model, with the first torus perscribed.*  

- Lower-FPU-periodic-2  
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3-d FPU model, with the second torus prescribed.*  

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3-d FPU model, with the second torus perscribed.*  

- Lower-FPU-periodic-3
  > main_FPU_3nodes_1torus.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3-d FPU model, with the third torus prescribed.*

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Vector_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3-d FPU model, with the third torus perscribed.*

- Lower-FPU-quasiperiodic-1
  > main_FPU_3nodes_2tori.m  
  *This is the main program with Dimension-enlarged Newton scheme for lower-dimensional solutions of 3-d FPU model, with the first and second tori prescribed.*

  > Newton_FPU_Solver.m  
  > P_eqn_calcu.m  
  > Q_eqn.m  
  > T_construct.m  
  > Matrix_Expand_Padding.m  
  > Vectorization_Process.m  
  > Vectorization_Process_Inverse.m  
  *These are functions used in the main program for lower-dimensional solutions of 3-d FPU model, with the first and second tori prescribed.*  

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
  


  
