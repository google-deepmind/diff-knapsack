# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Official Numba implementation of algorithms in the paper: Differentiable Knapsack and Top-k Operators via Dynamic Programming."""

import numba
import numpy as np

# ==============================================================================
# Public Python Wrappers
# ==============================================================================


def dp_value(
    theta: np.ndarray,
    capacity: int,
    gamma: float,
    weights: np.ndarray,
    constraint: str = "knapsack",
    regularizer: str = "shannon",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Computes the forward dynamic programming pass.

  This function implements Algorithm 1 from the paper, supporting Knapsack and
  Top-k constraints, as well as Shannon, Gini, and Tsallis entropies as
  regularizers Omega.

  Args:
      theta: A 2D numpy array of shape (batch_size, n) representing item values.
      capacity: An integer representing the maximum weight capacity.
      gamma: A float representing the regularization strength.
      weights: A 2D numpy array of shape (batch_size, n) representing item
        weights.
      constraint: A string, either "knapsack" or "top_k", specifying the problem
        type.
      regularizer: A string, one of "shannon", "gini", or "tsallis".

  Returns:
      A tuple containing:
      - The scalar relaxed objective values of shape (batch_size,).
      - The full DP value table (v_table) of shape (batch_size, n + 1, capacity
      + 1).
      - The local choice probability table (q_table) of shape (batch_size, n +
      1, capacity + 1).
  """
  batch_size, n = theta.shape
  _validate_inputs(batch_size, n, weights, constraint, regularizer)
  return _dp_value_numba(
      theta, capacity, gamma, weights, constraint, regularizer
  )


def dp_layer(
    v_table: np.ndarray,
    q_table: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Executes the backward recursion to compute the relaxed operators.

  This function implements Algorithm 2 from the paper, supporting Knapsack and
  Top-k constraints, as well as Shannon, Gini, and Tsallis entropies as
  regularizers Omega.

  Args:
      v_table: A 3D numpy array representing DP values from the forward pass.
      q_table: A 3D numpy array representing the local choice probabilities.
      weights: A 2D numpy array of shape (batch_size, n) representing item
        weights.

  Returns:
      A tuple containing:
      - y: The relaxed item selections of shape (batch_size, n)
      - e_table: The marginal probability table of the DP states.
  """
  batch_size, n_plus_1, _ = v_table.shape
  n = n_plus_1 - 1
  _validate_inputs(batch_size, n, weights)
  return _dp_layer_numba(v_table, q_table, weights)


def dp_sample(
    q_table: np.ndarray,
    weights: np.ndarray,
    num_samples: int = 1,
) -> np.ndarray:
  """Generates valid item selections via ancestral sampling.

  This function implements Algorithm 3 from the paper, supporting Knapsack and
  Top-k constraints, as well as Shannon, Gini, and Tsallis entropies as
  regularizers Omega.

  Args:
      q_table: A 3D numpy array representing the local choice probabilities.
      weights: A 2D numpy array representing item weights.
      num_samples: An integer specifying the number of samples to draw per
        batch.

  Returns:
      A 3D numpy array of shape (batch_size, num_samples, n) containing binary
      item selections.
  """
  batch_size, n_plus_1, _ = q_table.shape
  n = n_plus_1 - 1
  _validate_inputs(batch_size, n, weights)
  return _dp_sample_numba(q_table, weights, num_samples)


def dp_vjp(
    z: np.ndarray,
    v_table: np.ndarray,
    q_table: np.ndarray,
    e_table: np.ndarray,
    gamma: float,
    weights: np.ndarray,
    regularizer: str = "shannon",
) -> np.ndarray:
  """Computes the Vector-Jacobian Product (VJP) for the relaxed operator.

  This function implements Algorithm 4 from the paper, supporting Knapsack and
  Top-k constraints, as well as Shannon, Gini, and Tsallis entropies as
  regularizers Omega.

  Args:
      z: A 2D numpy array of shape (batch_size, n) representing the cotangent
        vector.
      v_table: A 3D numpy array of the DP value table from the forward pass.
      q_table: A 3D numpy array of the local choice probabilities.
      e_table: A 3D numpy array of the marginal state probabilities (from Algo
        2).
      gamma: A float representing the regularization strength.
      weights: A 2D numpy array of shape (batch_size, n) representing item
        weights.
      regularizer: A string, one of "shannon", "gini", or "tsallis".

  Returns:
    A 2D numpy array of shape (batch_size, n) representing the computed VJP.
  """
  batch_size, n_plus_1, _ = v_table.shape
  n = n_plus_1 - 1
  _validate_inputs(batch_size, n, weights, regularizer=regularizer)
  return _dp_vjp_numba(
      z, v_table, q_table, e_table, gamma, weights, regularizer
  )


# ==============================================================================
# Validation Helper
# ==============================================================================


def _validate_inputs(
    batch_size: int,
    n: int,
    weights: np.ndarray,
    constraint: str | None = None,
    regularizer: str | None = None,
) -> None:
  """Validates inputs for DP operators."""
  if weights.shape != (batch_size, n):
    raise ValueError(f"Weights must have shape ({batch_size}, {n})!")

  if constraint is not None:
    if constraint == "top_k":
      if (weights != 1).any():
        raise ValueError("Weights must be all 1 for Top-k!")
    elif constraint != "knapsack":
      raise ValueError("Unsupported constraint!")

  if regularizer is not None:
    if regularizer not in ["shannon", "gini", "tsallis"]:
      raise ValueError("Unsupported regularizer!")


# ==============================================================================
# Private Numba Kernels
# ==============================================================================


@numba.jit(nopython=True, parallel=True)
def _dp_value_numba(
    theta: np.ndarray,
    capacity: int,
    gamma: float,
    weights: np.ndarray,
    constraint: str,
    regularizer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Numba loop for DP value computation."""
  batch_size, n = theta.shape

  v_table = np.full((batch_size, n + 1, capacity + 1), -np.inf)
  q_table = np.zeros((batch_size, n + 1, capacity + 1))

  for b in numba.prange(batch_size):

    v_table[b, 0, 0] = 0.0
    if constraint == "knapsack":
      v_table[b, 0, 1:] = 0.0
    elif constraint == "top_k":
      v_table[b, 0, 1:] = -np.inf

    for i in range(1, n + 1):
      ti = theta[b, i - 1]
      wi = weights[b, i - 1]

      for c in numba.prange(capacity + 1):

        v_skip = v_table[b, i - 1, c]

        v_pick = -np.inf
        if c >= wi:
          if v_table[b, i - 1, c - wi] > -np.inf:
            v_pick = v_table[b, i - 1, c - wi] + ti

        if v_skip <= -np.inf and v_pick <= -np.inf:
          continue

        v_val, q_ic = _omega_value_and_choice(
            v_skip, v_pick, gamma, regularizer
        )

        if v_val > -np.inf:
          v_table[b, i, c] = v_val
        q_table[b, i, c] = q_ic

  return v_table[:, n, capacity], v_table, q_table


@numba.jit(nopython=True, parallel=True)
def _dp_layer_numba(
    v_table: np.ndarray,
    q_table: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
  """Numba loop for DP relaxed operators."""
  batch_size, n_plus_1, capacity_plus_1 = v_table.shape
  n = n_plus_1 - 1
  capacity = capacity_plus_1 - 1

  e_table = np.zeros((batch_size, n + 1, capacity + 1))
  y = np.zeros((batch_size, n))

  for b in numba.prange(batch_size):
    e_table[b, n, capacity] = 1.0

    # Backward recursion
    for i in range(n - 1, -1, -1):
      wi_next = weights[b, i]
      for c in numba.prange(capacity + 1):
        # skip path
        val = e_table[b, i + 1, c] * (1.0 - q_table[b, i + 1, c])

        # pick path
        if c + wi_next <= capacity:
          val += e_table[b, i + 1, c + wi_next] * q_table[b, i + 1, c + wi_next]

        e_table[b, i, c] = val

    # Layer assembly
    for i in numba.prange(1, n + 1):
      current_sum = 0.0
      for c in numba.prange(capacity + 1):
        current_sum += e_table[b, i, c] * q_table[b, i, c]
      y[b, i - 1] = current_sum

  return y, e_table


@numba.jit(nopython=True, parallel=True)
def _dp_sample_numba(
    q_table: np.ndarray,
    weights: np.ndarray,
    num_samples: int,
) -> np.ndarray:
  """Numba loop for DP ancestral sampling."""
  batch_size, n_plus_1, capacity_plus_1 = q_table.shape
  n = n_plus_1 - 1
  capacity = capacity_plus_1 - 1

  samples = np.zeros((batch_size, num_samples, n))
  noise = np.random.random((batch_size, num_samples, n))

  for b in numba.prange(batch_size):
    for s in numba.prange(num_samples):
      current_c = capacity
      for i in range(n, 0, -1):
        wi = weights[b, i - 1]
        if current_c >= wi:
          p_pick = q_table[b, i, current_c]

          if noise[b, s, i - 1] < p_pick:
            samples[b, s, i - 1] = 1.0
            current_c -= wi
          else:
            samples[b, s, i - 1] = 0.0
        else:
          samples[b, s, i - 1] = 0.0

  return samples


@numba.jit(nopython=True, parallel=True)
def _dp_vjp_numba(
    z: np.ndarray,
    v_table: np.ndarray,
    q_table: np.ndarray,
    e_table: np.ndarray,
    gamma: float,
    weights: np.ndarray,
    regularizer: str,
) -> np.ndarray:
  """Numba loop for VJP computation."""
  batch_size, n_plus_1, capacity_plus_1 = v_table.shape
  n = n_plus_1 - 1
  capacity = capacity_plus_1 - 1

  v_dot_table = np.zeros((batch_size, n + 1, capacity + 1))
  e_dot_table = np.zeros((batch_size, n + 1, capacity + 1))
  vjp = np.zeros((batch_size, n))

  for b in numba.prange(batch_size):
    # Forward v_dot
    for i in range(1, n + 1):
      zi = z[b, i - 1]
      wi = weights[b, i - 1]
      for c in numba.prange(capacity + 1):
        q_ic = q_table[b, i, c]
        v_dot_skip = v_dot_table[b, i - 1, c]
        v_dot_pick = 0.0
        if c >= wi:
          v_dot_pick = v_dot_table[b, i - 1, c - wi] + zi

        v_dot_table[b, i, c] = v_dot_pick * q_ic + v_dot_skip * (1.0 - q_ic)

    # Backward e_dot
    for i in range(n - 1, -1, -1):
      wi_next = weights[b, i]
      zi_next = z[b, i]
      for c in numba.prange(capacity + 1):
        q_next_skip = q_table[b, i + 1, c]
        dq_dv_skip = -_omega_derivative(q_next_skip, gamma, regularizer)

        v_dot_p_prev_skip = 0.0
        if c >= wi_next:
          v_dot_p_prev_skip = v_dot_table[b, i, c - wi_next]

        dot_q_skip = dq_dv_skip * (
            v_dot_p_prev_skip - v_dot_table[b, i, c] + zi_next
        )
        val = (
            e_dot_table[b, i + 1, c] * (1.0 - q_next_skip)
            + e_table[b, i + 1, c] * dot_q_skip
        )

        if c + wi_next <= capacity:
          q_next_pick = q_table[b, i + 1, c + wi_next]
          dq_dv_pick = _omega_derivative(q_next_pick, gamma, regularizer)

          dot_q_pick = dq_dv_pick * (
              v_dot_table[b, i, c] - v_dot_table[b, i, c + wi_next] + zi_next
          )
          val += (
              e_dot_table[b, i + 1, c + wi_next] * q_next_pick
              + e_table[b, i + 1, c + wi_next] * dot_q_pick
          )

        e_dot_table[b, i, c] = val

    # VJP reduction
    for i in numba.prange(1, n + 1):
      zi = z[b, i - 1]
      wi = weights[b, i - 1]
      row_sum = 0.0
      for c in numba.prange(capacity + 1):
        q_ic = q_table[b, i, c]
        dq_dt = _omega_derivative(q_ic, gamma, regularizer)

        v_dot_p_prev = 0.0
        if c >= wi:
          v_dot_p_prev = v_dot_table[b, i - 1, c - wi]

        delta_v_dot = v_dot_p_prev - v_dot_table[b, i - 1, c] + zi
        row_sum += (
            e_dot_table[b, i, c] * q_ic + e_table[b, i, c] * delta_v_dot * dq_dt
        )
      vjp[b, i - 1] = row_sum

  return vjp


# ==============================================================================
# Regularization-Specific Helpers
# ==============================================================================


@numba.jit(nopython=True)
def _omega_value_and_choice(
    v_skip: float, v_pick: float, gamma: float, regularizer: str
) -> tuple[float, float]:
  """Computes smoothed value and choice probability for a DP state."""
  if v_skip <= -np.inf and v_pick <= -np.inf:
    return -np.inf, 0.0

  if regularizer == "shannon":
    v_max = max(v_skip, v_pick)
    v_val = v_max + gamma * np.log(
        np.exp((v_skip - v_max) / gamma) + np.exp((v_pick - v_max) / gamma)
    )
    q_ic = np.exp((v_pick - v_val) / gamma)
    return v_val, q_ic

  elif regularizer == "gini":
    if v_pick > -np.inf and v_skip > -np.inf:
      delta_ic = v_skip - v_pick
      q_val = (-delta_ic + gamma) / (2 * gamma)
      q_ic = min(max(q_val, 0.0), 1.0)
    elif v_pick > -np.inf:
      q_ic = 1.0
    else:
      q_ic = 0.0

    if q_ic <= 0.0:
      v_val = v_skip
    elif q_ic >= 1.0:
      v_val = v_pick
    else:
      delta_ic = v_skip - v_pick
      v_val = v_skip - q_ic * delta_ic + gamma * q_ic * (1.0 - q_ic)
    return v_val, q_ic

  elif regularizer == "tsallis":
    if v_pick > -np.inf and v_skip > -np.inf:
      delta_ic = v_skip - v_pick
      c_bar = min(max(-delta_ic / (2.0 * gamma), -1.0), 1.0)
      q_ic = 0.5 * (1.0 + c_bar * np.sqrt(2.0 - c_bar**2))
    elif v_pick > -np.inf:
      q_ic = 1.0
    else:
      q_ic = 0.0

    if q_ic <= 0.0:
      v_val = v_skip
    elif q_ic >= 1.0:
      v_val = v_pick
    else:
      delta_ic = v_skip - v_pick
      v_val = (
          v_skip
          - delta_ic * q_ic
          + (4.0 * gamma / 3.0) * (1.0 - q_ic**1.5 - (1.0 - q_ic) ** 1.5)
      )
    return v_val, q_ic

  return -np.inf, 0.0


@numba.jit(nopython=True)
def _omega_derivative(q: float, gamma: float, regularizer: str) -> float:
  """Computes partial derivative of choice probability dq_id/dv_ic."""
  if regularizer == "shannon":
    return (1.0 / gamma) * q * (1.0 - q)
  elif regularizer == "gini":
    if 0.0 < q < 1.0:
      return 1.0 / (2.0 * gamma)
    return 0.0
  elif regularizer == "tsallis":
    if 0.0 < q < 1.0:
      return (1.0 / gamma) * (1.0 / (1.0 / np.sqrt(q) + 1.0 / np.sqrt(1.0 - q)))
    return 0.0
  return 0.0
