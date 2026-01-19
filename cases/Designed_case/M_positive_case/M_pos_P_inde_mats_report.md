# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 1.  1. -1. 12. 12.]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $B^T P B$ Contribution Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[13.5274, 11.2226]` | `[ 1.5274, -0.7774]` |
| Step 4 | `[14.8824, 10.3913]` | `[ 2.8824, -1.6087]` |
| Step 3 | `[15.8516,  9.3648]` | `[ 3.8516, -2.6352]` |
| Step 2 | `[16.4432,  7.8457]` | `[ 4.4432, -4.1543]` |
| Step 1 | `[16.77  ,  4.8687]` | `[ 4.77  , -7.1313]` |

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
