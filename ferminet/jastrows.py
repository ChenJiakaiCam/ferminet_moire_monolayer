# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multiplicative Jastrow factors."""

import enum
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from ferminet import network_blocks
import jax.numpy as jnp

ParamTree = Union[jnp.ndarray, Iterable['ParamTree'], Mapping[Any, 'ParamTree']]


class JastrowType(enum.Enum):
  """Available multiplicative Jastrow factors."""

  NONE = enum.auto()
  SIMPLE_EE = enum.auto()
  MONOLAYER = enum.auto()


def safe_norm(ee: jnp.ndarray):
  n = ee.shape[0]
  return (
      jnp.linalg.norm(ee + jnp.eye(n)[..., None], axis=-1) * (1.0 - jnp.eye(n))
  )[
      ..., None
  ]  # extra dim for consistency with previous code versions


def make_periodic_r_ee(lattice: jnp.ndarray):
  """Util function for transforming r_ee into the periodic version.

  Some of this could be refactored with ferminet.pbc.feature_layer

  Args:
      lattice: Matrix whose columns are the primitive lattice vectors of the
        system, shape (ndim, ndim).

  Returns:
      Function mapping electron-electron displacement vectors (ee) to periodic
      distances of shape (..., 1).
  """

  # Calculate reciprocal vectors, factor 2pi omitted
  reciprocal_vecs = jnp.linalg.inv(lattice)
  lattice_metric = lattice.T @ lattice

  def apply(ee: jnp.ndarray):
    s_ee = jnp.einsum('il,jkl->jki', reciprocal_vecs, ee)

    n = ee.shape[0]
    eye = jnp.eye(n, dtype=s_ee.dtype)

    a = 1 - jnp.cos(2 * jnp.pi * s_ee)
    b = jnp.sin(2 * jnp.pi * s_ee)
    cos_term = jnp.einsum('...m,mn,...n->...', a, lattice_metric, a)
    sin_term = jnp.einsum('...m,mn,...n->...', b, lattice_metric, b)
    periodic_r_ee = (
        (1 / (2 * jnp.pi)) * jnp.sqrt(cos_term + sin_term + eye) * (1.0 - eye)
    )
    return periodic_r_ee[..., None]

  return apply


def _jastrow_ee(
    r_ee: jnp.ndarray,
    params: ParamTree,
    nspins: tuple[int, int],
    ndim: int,
    jastrow_fun: Callable[[jnp.ndarray, float, jnp.ndarray], jnp.ndarray],
) -> jnp.ndarray:
  """Jastrow factor for electron-electron cusps."""
  r_ees = [
      jnp.split(r, nspins[0:1], axis=1)
      for r in jnp.split(r_ee, nspins[0:1], axis=0)
  ]
  r_ees_parallel = jnp.concatenate([
      r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
      r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
  ])

  if r_ees_parallel.shape[0] > 0:
    jastrow_ee_par = jnp.sum(
        jastrow_fun(r_ees_parallel, 1 / (ndim + 1), params['ee_par'])  # pyrefly: ignore[bad-argument-type, bad-index]
    )
  else:
    jastrow_ee_par = jnp.asarray(0.0)

  if r_ees[0][1].shape[0] > 0:
    jastrow_ee_anti = jnp.sum(
        jastrow_fun(r_ees[0][1], 1 / (ndim - 1), params['ee_anti'])  # pyrefly: ignore[bad-argument-type, bad-index]
    )
  else:
    jastrow_ee_anti = jnp.asarray(0.0)

  return jastrow_ee_anti + jastrow_ee_par


def make_simple_ee_jastrow(lattice: jnp.ndarray | None = None, ndim: int = 3):
  """Creates a Jastrow factor for electron-electron cusps."""

  # If working in PBC, use periodic distance for the Jastrow
  if lattice is not None:
    norm = make_periodic_r_ee(lattice)
  else:
    norm = safe_norm

  def simple_ee_cusp_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    """Jastrow function satisfying electron cusp condition."""
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    params = {}
    params['ee_par'] = jnp.ones(
        shape=1,
    )
    params['ee_anti'] = jnp.ones(
        shape=1,
    )
    return params

  def apply(
      ee: jnp.ndarray,
      params: ParamTree,
      nspins: tuple[int, int],
  ) -> jnp.ndarray:
    """Jastrow factor for electron-electron cusps."""
    r_ee = norm(ee)
    return _jastrow_ee(
        r_ee, params, nspins, ndim, jastrow_fun=simple_ee_cusp_fun
    )

  return init, apply


def make_monolayer_jastrow(
    lattice: jnp.ndarray | None = None,
    ndim: int = 2,
    mass: float = 1.0,
    dielectric_constant: float = 1.0,
):
  """Creates a 2D effective-mass/dielectric-screened Jastrow factor for monolayers.

  Pair function in J(R) is u(r) = -(c * alpha^2) / (alpha + r)
  where alpha is trainable and c is fixed by the 2D Kato cusp condition using
  the effective mass m* and relative dielectric constant epsilon_r:
    - parallel spin/valley: c_par = m* / (epsilon_r * (ndim + 1))
      (i.e. m* / (3 * epsilon_r) in 2D),
    - antiparallel spin/valley: c_anti = m* / (epsilon_r * (ndim - 1))
      (i.e. m* / epsilon_r in 2D).
  """
  if mass <= 0.0:
    raise ValueError(f'Effective mass must be positive, got mass={mass}.')
  if dielectric_constant <= 0.0:
    raise ValueError(
        f'dielectric_constant must be positive, got {dielectric_constant}.'
    )

  if lattice is not None:
    norm = make_periodic_r_ee(lattice)
  else:
    norm = safe_norm

  denom_same = float(ndim + 1.0)
  denom_diff = float(max(ndim - 1.0, 1.0))
  scale = float(mass) / float(dielectric_constant)
  cusp_par = scale / denom_same
  cusp_anti = scale / denom_diff

  def jastrow_fun(
      r: jnp.ndarray, cusp: float, alpha: jnp.ndarray
  ) -> jnp.ndarray:
    return -(cusp * alpha**2) / (alpha + r)

  def init() -> Mapping[str, jnp.ndarray]:
    return {
        'ee_par': jnp.ones(shape=1),
        'ee_anti': jnp.ones(shape=1),
    }

  def apply(
      ee: jnp.ndarray,
      params: ParamTree,
      nspins: Sequence[int],
  ) -> jnp.ndarray:
    if len(nspins) != 2:
      raise ValueError(
          f'Monolayer Jastrow requires 2 spin channels, got {nspins}.'
      )
    r_ee = norm(ee)
    split_idx = network_blocks.array_partitions(nspins)
    r_ees = [
        jnp.split(r, split_idx, axis=1)
        for r in jnp.split(r_ee, split_idx, axis=0)
    ]
    r_ees_parallel = jnp.concatenate([
        r_ees[0][0][jnp.triu_indices(nspins[0], k=1)],
        r_ees[1][1][jnp.triu_indices(nspins[1], k=1)],
    ])

    if r_ees_parallel.shape[0] > 0:
      jastrow_ee_par = jnp.sum(
          jastrow_fun(r_ees_parallel, cusp_par, params['ee_par'])  # pyrefly: ignore[bad-argument-type, bad-index]
      )
    else:
      jastrow_ee_par = jnp.asarray(0.0, dtype=ee.dtype)

    if r_ees[0][1].size > 0:
      jastrow_ee_anti = jnp.sum(
          jastrow_fun(r_ees[0][1], cusp_anti, params['ee_anti'])  # pyrefly: ignore[bad-argument-type, bad-index]
      )
    else:
      jastrow_ee_anti = jnp.asarray(0.0, dtype=ee.dtype)

    return jastrow_ee_anti + jastrow_ee_par

  return init, apply


def get_jastrow(
    jastrow: JastrowType,
    lattice: jnp.ndarray | None = None,
    ndim: int = 3,
    jastrow_kwargs: Mapping[str, Any] | None = None,
):
  jastrow_init, jastrow_apply = None, None
  if jastrow == JastrowType.SIMPLE_EE:
    jastrow_init, jastrow_apply = make_simple_ee_jastrow(lattice, ndim)
  elif jastrow == JastrowType.MONOLAYER:
    kwargs = dict(jastrow_kwargs) if jastrow_kwargs is not None else {}
    jastrow_init, jastrow_apply = make_monolayer_jastrow(
        lattice, ndim, **kwargs
    )
  elif jastrow != JastrowType.NONE:
    raise ValueError(f'Unknown Jastrow Factor type: {jastrow}')

  return jastrow_init, jastrow_apply
