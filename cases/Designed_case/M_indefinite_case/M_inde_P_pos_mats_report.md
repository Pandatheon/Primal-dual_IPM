# QP Analysis Report

## Core Parameter Summary

| Parameter | Value |
| :--- | :--- |
| Prediction Horizon (N) | 5 |
| Hessian H Eigenvalues | `[ 1.   3.   1.5  2.  -2. ]` |
## Discrete Riccati Iteration Eigenvalue Evolution

| Step | Reduced Hessian $M = R + B^T P B$ Eigenvalues | $P$ Eigenvalues |
| :--- | :--- | :--- |
| Step 5 | `[ 1.2438, 11.2562]` | `[1. , 3. , 1.5]` |
| Step 4 | `[10.3047, -0.8405]` | `[0.0946, 4.3157, 9.0539]` |
| Step 3 | `[ 3.9907, 40.9938]` | `[25.9514,  1.1921, 10.041 ]` |
| Step 2 | `[16.2173, -0.027 ]` | `[ 0.1753, 17.3825,  9.6999]` |
| Step 1 | `[   5.8788, 1665.972 ]` | `[1.3686e+03, 1.3647e+00, 1.1840e+01]` |

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
[[1.  0.  0. ]
 [0.  3.  0. ]
 [0.  0.  1.5]]
```

</details>

<details>
<summary>Click to expand R (Input Cost)</summary>


```python
[[ 2.  0.]
 [ 0. -2.]]
```

</details>

<details>
<summary>Click to expand H (Full Hessian)</summary>


```python
[[ 1.   0.   0.   0.   0. ]
 [ 0.   3.   0.   0.   0. ]
 [ 0.   0.   1.5  0.   0. ]
 [ 0.   0.   0.   2.   0. ]
 [ 0.   0.   0.   0.  -2. ]]
```

</details>
