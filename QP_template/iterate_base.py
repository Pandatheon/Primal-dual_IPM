from dataclasses import dataclass, astuple
import numpy as np
from typing import List

@dataclass
class iterate_sol:
    x: List[np.ndarray] = None
    u: List[np.ndarray] = None
    pi: List[np.ndarray] = None
    lam: List[np.ndarray] = None
    s: List[np.ndarray] = None

    def __iter__(self):
        # Enable easy unpacking: x, u, pi, ... = iterate_obj
        return iter(astuple(self))

@dataclass
class iterate_rhs:
    rg : List[np.ndarray] = None
    rb : List[np.ndarray] = None
    rd : List[np.ndarray] = None
    rm : List[np.ndarray] = None
    duality_measure : float = None

    def __iter__(self):
        # Enable easy unpacking: rg, rb, rd, rm, duality_measure = rhs_dict_obj
        return iter(astuple(self))

@dataclass
class Feedback:
    P: List[np.ndarray] = None
    p: List[np.ndarray] = None
    K: List[np.ndarray] = None
    k: List[np.ndarray] = None

    def compute_control(self, index: int, state: np.ndarray) -> np.ndarray:
        '''compute the control according to the feedback law u = k + Kx'''
        return self.K[index] @ state + self.k[index]