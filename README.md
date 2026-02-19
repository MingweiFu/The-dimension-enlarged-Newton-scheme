# Dimension-enlarged Newton scheme (full dimensional quasi-periodic solutions of finitely dimensional systems)

This is the repository for the source codes of the paper

> [Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators](https://arxiv.org/abs/2602.16275), Mingwei Fu and Bin Shi.  

These codes are implementations of Dimension-enlarged Newton scheme for finding full dimensional quasi-periodic solutions of 1-d undamped Duffing oscillator and 2-d Henon-Heiles system.

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

## Citing

If you want to use `Dimension-enlarged Newton scheme` for acadamic proposes, please cite the main references as follows:

```
@article{Fu2026numerical,
  title={Numerical Construction of Quasi-Periodic Solutions Beyond Symplectic Integrators},
  author={Fu, Mingwei and Shi, Bin},
  journal={arXiv preprint arXiv:2602.16275},
  year={2026}
```
  


  
