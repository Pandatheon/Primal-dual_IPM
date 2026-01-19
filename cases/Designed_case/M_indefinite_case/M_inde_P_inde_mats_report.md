# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 2.  2. -1.  1.  2.]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $B^T P B$ Contribution Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[ 1.2918, 14.7082]` | `[ 0., 13.]` |
| Step 4 | `[ 0.4836, 17.3059]` | `[-0.956 , 15.7455]` |
| Step 3 | `[  9.0769, -72.4731]` | `[  8.0769, -74.4731]` |
| Step 2 | `[ 5.6351, 60.1247]` | `[ 4.4844, 58.2754]` |
| Step 1 | `[ 5.1909, 56.3928]` | `[ 4.0043, 54.5794]` |

## Matrix Details (System Definition)

<details>
<summary>Click to expand A (System Dynamics)</summary>


```python
[[1. 0. 0.]
 [0. 2. 0.]
 [0. 0. 2.]]
```

</details>

<details>
<summary>Click to expand B (Input Matrix)</summary>


```python
[[1. 2.]
 [1. 1.]
 [0. 1.]]
```

</details>

<details>
<summary>Click to expand Q (State Cost)</summary>


```python
[[ 2.  0.  0.]
 [ 0.  2.  0.]
 [ 0.  0. -1.]]
```

</details>

<details>
<summary>Click to expand R (Input Cost)</summary>


```python
[[1. 0.]
 [0. 2.]]
```

</details>

<details>
<summary>Click to expand H (Full Hessian)</summary>


```python
[[ 2.  0.  0.  0.  0.]
 [ 0.  2.  0.  0.  0.]
 [ 0.  0. -1.  0.  0.]
 [ 0.  0.  0.  1.  0.]
 [ 0.  0.  0.  0.  2.]]
```

</details>
