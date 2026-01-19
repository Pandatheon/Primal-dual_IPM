# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 1.  1. -1. 12. 12.]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $P$ Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[13.5274, 11.2226]` | `[ 1.,  1., -1.]` |
| Step 4 | `[14.8824, 10.3913]` | `[ 1.8731,  1.9968, -2.0881]` |
| Step 3 | `[15.8516,  9.3648]` | `[ 2.4621,  2.9831, -3.4911]` |
| Step 2 | `[16.4432,  7.8457]` | `[ 3.9481,  2.7725, -5.6802]` |
| Step 1 | `[16.77  ,  4.8687]` | `[-10.1975,   2.8829,   4.8719]` |

## Matrix Details (System Definition)

<details>
<summary>Click to expand A (System Dynamics)</summary>


```python
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

</details>

<details>
<summary>Click to expand B (Input Matrix)</summary>


```python
[[1.  0.5]
 [0.5 0.5]
 [0.  1. ]]
```

</details>

<details>
<summary>Click to expand Q (State Cost)</summary>


```python
[[ 1.  0.  0.]
 [ 0.  1.  0.]
 [ 0.  0. -1.]]
```

</details>

<details>
<summary>Click to expand R (Input Cost)</summary>


```python
[[12.  0.]
 [ 0. 12.]]
```

</details>

<details>
<summary>Click to expand H (Full Hessian)</summary>


```python
[[ 1.  0.  0.  0.  0.]
 [ 0.  1.  0.  0.  0.]
 [ 0.  0. -1.  0.  0.]
 [ 0.  0.  0. 12.  0.]
 [ 0.  0.  0.  0. 12.]]
```

</details>
