# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 2.1  1.   1.  -2.8  0.1]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $P$ Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[-0.1454,  2.5454]` | `[2.1, 1. , 1. ]` |
| Step 4 | `[1.0036, 3.0686]` | `[3.2398, 1.0991, 1.002 ]` |
| Step 3 | `[-0.4196,  2.4971]` | `[1.9168, 0.9992, 1.0026]` |
| Step 2 | `[0.2629, 2.6588]` | `[2.4538, 1.0657, 1.0019]` |
| Step 1 | `[-1.0619,  2.4093]` | `[1.5468, 0.8488, 1.002 ]` |

## Matrix Details (System Definition)

<details>
<summary>Click to expand A (System Dynamics)</summary>


```python
[[0.2 0.  0. ]
 [0.  0.2 0. ]
 [0.  0.  0.2]]
```

</details>

<details>
<summary>Click to expand B (Input Matrix)</summary>


```python
[[1. 0.]
 [0. 1.]
 [1. 1.]]
```

</details>

<details>
<summary>Click to expand Q (State Cost)</summary>


```python
[[2.1 0.  0. ]
 [0.  1.  0. ]
 [0.  0.  1. ]]
```

</details>

<details>
<summary>Click to expand R (Input Cost)</summary>


```python
[[-2.8  0. ]
 [ 0.   0.1]]
```

</details>

<details>
<summary>Click to expand H (Full Hessian)</summary>


```python
[[ 2.1  0.   0.   0.   0. ]
 [ 0.   1.   0.   0.   0. ]
 [ 0.   0.   1.   0.   0. ]
 [ 0.   0.   0.  -2.8  0. ]
 [ 0.   0.   0.   0.   0.1]]
```

</details>
