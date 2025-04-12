# Author : Amalya Cox Johnson
# email : amalyaj@stanford.edu

import OTRaPy.RamanSolver as RS
import json
import os
import numpy as np


def test_init():
    solver = RS.RamanSolver()
    assert solver


def test_iso():
    with open("test_files/taube_in.json", "r") as f:
        taube = json.load(f)
    solver = RS.RamanSolver(**taube["init"])


# DEPRECATED
# def test_Txy_forloop():
#     solver = RS.RamanSolver()
#     error = 1
#     count = 1

#     threshold = 1e-5

#     solver.generate_qdot(1e-3)
#     g = 1.94e6
#     kx = 62.2
#     ky = 62.2
#     Ta = 300

#     T = np.ones((solver.nx, solver.ny)) * Ta
#     Tg = T.copy()

#     d2h = solver.delta**2 / solver.h

#     while error > threshold:
#         for i in range(1, solver.nx - 1):
#             for j in range(1, solver.ny - 1):
#                 T[i, j] = (
#                     kx * (T[i, j - 1] + T[i, j + 1])
#                     + ky * (T[i - 1, j] + T[i + 1, j])
#                     + d2h * (solver.qdot[i, j] + g * Ta)
#                 ) / (d2h * g + 2 * kx + 2 * ky)

#         error = np.sqrt(np.sum(np.abs(T - Tg) ** 2))

#         Tg = T.copy()

#     return T
