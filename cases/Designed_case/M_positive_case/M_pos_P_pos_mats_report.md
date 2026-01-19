# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 1.  2.  2.  2. -1.]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $P$ Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[ 1.8769, 10.1231]` | `[1., 2., 2.]` |
| Step 4 | `[11.3572,  2.7481]` | `[0.8173, 5.8605, 7.1643]` |
| Step 3 | `[15.7066,  3.6593]` | `[ 0.7973, 10.8764, 13.2116]` |
| Step 2 | `[18.6035,  4.1828]` | `[ 0.7951, 19.2822, 12.4288]` |
| Step 1 | `[19.9661,  4.3866]` | `[ 0.7949, 22.6297, 12.7459]` |

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
[[1. 0. 0.]
 [0. 2. 0.]
 [0. 0. 2.]]
```

</details>

<details>
<summary>Click to expand R (Input Cost)</summary>


```python
[[ 2.  0.]
 [ 0. -1.]]
```

</details>

<details>
<summary>Click to expand H (Full Hessian)</summary>


```python
[[ 1.  0.  0.  0.  0.]
 [ 0.  2.  0.  0.  0.]
 [ 0.  0.  2.  0.  0.]
 [ 0.  0.  0.  2.  0.]
 [ 0.  0.  0.  0. -1.]]
```

</details>
