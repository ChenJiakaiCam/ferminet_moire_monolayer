"""2D-periodic Ewald summation for a single 2D carrier plane with dielectric screening."""

from typing import Callable, Optional, Sequence

from ferminet.moire import geometry
import jax
import jax.numpy as jnp
import numpy as np
from scipy import special as scipy_special

_GWEIGHT_TOL = 1e-12


def _halfspace_gpoint_indices_2d(ewald_gmax: int) -> np.ndarray:
  """Enumerates non-zero 2D integer indices covering half of reciprocal space.

  Returns all integer pairs ``(m_1, m_2)`` in ``[-ewald_gmax, ewald_gmax]^2``
  whose first non-zero entry is positive, avoiding double-counting of ``G`` and
  ``-G``.
  """
  if ewald_gmax < 1:
    raise ValueError(f'ewald_gmax must be >= 1, got {ewald_gmax}.')
  full = np.arange(-ewald_gmax, ewald_gmax + 1, dtype=np.int32)
  positive = np.arange(1, ewald_gmax + 1, dtype=np.int32)
  zero = np.array([0], dtype=np.int32)

  g0_1, g0_2 = np.meshgrid(positive, full, indexing='ij')
  block0 = np.stack([g0_1.ravel(), g0_2.ravel()], axis=-1)

  g1_1, g1_2 = np.meshgrid(zero, positive, indexing='ij')
  block1 = np.stack([g1_1.ravel(), g1_2.ravel()], axis=-1)

  return np.concatenate([block0, block1], axis=0)


class MonolayerEwaldSum:
  r"""2D-periodic Ewald summation for a single 2D plane of carriers.

  Evaluates the 2D-periodic sum of the 3D Coulomb interaction screened by a
  relative dielectric permittivity ``epsilon_r`` with a uniform neutralizing
  background:

  .. math::
      v(r) = \frac{1}{\epsilon_r r}, \qquad
      U_{\mathrm{Coulomb}} = \frac{1}{\epsilon_r}
      \left(U_{\mathrm{real}} + U_{\mathrm{recip}} + U_{\mathrm{self}} + U_{\mathbf{G}=\mathbf{0}}\right).

  Uses the standard 2D Parry/Ewald decomposition so that the total energy is
  independent of the Ewald splitting parameter ``alpha``.
  """

  def __init__(
      self,
      lattice: geometry.Array,
      nspins: Optional[Sequence[int]] = None,
      dielectric_constant: float = 1.0,
      ewald_gmax: int = 100,
      nlatvec: int = 1,
      alpha: Optional[float] = None,
      n_particles: Optional[int] = None,
  ):
    """Initializes precomputed lattice and reciprocal-space Ewald tables.

    Args:
      lattice: Shape ``(2, 2)`` simulation supercell matrix (columns are direct
        lattice vectors).
      nspins: Sequence ``(n_up, n_down)`` of spin-up and spin-down particle
        counts.
      dielectric_constant: Relative dielectric constant ``epsilon_r > 0``.
      ewald_gmax: Reciprocal-space cutoff index in each direction.
      nlatvec: Real-space image shell cutoff around the minimum-image vector.
      alpha: Optional override for Ewald splitting parameter ``alpha``. Defaults
        to ``5.0 / h_min`` where ``h_min`` is the smallest perpendicular height
        of the simulation cell.
      n_particles: Optional direct total particle count when ``nspins`` is
        omitted.
    """
    if n_particles is not None:
      self.n_particles = int(n_particles)
    elif nspins is not None:
      self.n_particles = geometry.total_particle_count(nspins)
    else:
      raise ValueError('Either nspins or n_particles must be provided.')
    if dielectric_constant <= 0.0:
      raise ValueError(
          f'dielectric_constant must be > 0, got {dielectric_constant}.'
      )
    if nlatvec < 0:
      raise ValueError(f'nlatvec must be >= 0, got {nlatvec}.')

    self.latvec_np = geometry.validate_lattice_2d(lattice)
    self.dielectric_constant = float(dielectric_constant)
    self.area = float(abs(np.linalg.det(self.latvec_np)))
    self.dist_fn = geometry.make_minimum_image_distance_fn(self.latvec_np)

    grid = np.meshgrid(
        np.arange(-nlatvec, nlatvec + 1, dtype=np.int32),
        np.arange(-nlatvec, nlatvec + 1, dtype=np.int32),
        indexing='ij',
    )
    indices = np.stack([grid[0].ravel(), grid[1].ravel()], axis=-1)
    self.lattice_displacements = jnp.asarray(indices @ self.latvec_np.T)
    self.center_image_idx = 2 * nlatvec * (nlatvec + 1)

    inv_lat = np.linalg.inv(self.latvec_np)
    smallest_height = float(np.amin(1.0 / np.linalg.norm(inv_lat, axis=1)))
    self.alpha = float(5.0 / smallest_height if alpha is None else alpha)
    if self.alpha <= 0.0:
      raise ValueError(f'Ewald alpha must be > 0, got {self.alpha}.')

    g_indices = _halfspace_gpoint_indices_2d(ewald_gmax)
    rec_rows = 2.0 * np.pi * inv_lat
    gpoints_np = g_indices @ rec_rows
    gnorm_np = np.linalg.norm(gpoints_np, axis=-1)

    w0_np = (
        2.0
        * np.pi
        * scipy_special.erfc(gnorm_np / (2.0 * self.alpha))
        / (self.area * gnorm_np)
    )
    keep = w0_np > _GWEIGHT_TOL
    self.gpoints = jnp.asarray(gpoints_np[keep])
    self.gweight = jnp.asarray(w0_np[keep])

    self.self_const_factor = -self.alpha / np.sqrt(np.pi)
    self.g0_factor = -2.0 * np.sqrt(np.pi) / (self.area * self.alpha)

  def _unscreened_energy(self, coords: jnp.ndarray) -> jnp.ndarray:
    """Computes unscreened 2D-periodic Ewald energy for coordinates ``(N, 2)``."""
    n_parts = coords.shape[0]
    if n_parts == 0:
      return jnp.asarray(0.0, dtype=coords.dtype)

    dtype = coords.dtype
    lat_disp = self.lattice_displacements.astype(dtype)
    gpoints = self.gpoints.astype(dtype)
    gweight = self.gweight.astype(dtype)

    displacements, _ = self.dist_fn(coords, coords)
    rvec = displacements[None, ...] + lat_disp[:, None, None, :]
    r_sq = jnp.sum(rvec**2, axis=-1)

    n_imgs = lat_disp.shape[0]
    mask = jnp.ones((n_imgs, n_parts, n_parts), dtype=dtype)
    mask = mask.at[self.center_image_idx].set(
        1.0 - jnp.eye(n_parts, dtype=dtype)
    )

    r_sq_safe = jnp.where(mask < 0.5, 1.0, jnp.maximum(r_sq, 1e-24))
    r_safe = jnp.sqrt(r_sq_safe)

    pot_term = jax.lax.erfc(self.alpha * r_safe) / r_safe
    v_real = 0.5 * jnp.sum(pot_term * mask)

    phases = gpoints @ coords.T
    sf_real = jnp.sum(jnp.cos(phases), axis=-1)
    sf_imag = jnp.sum(jnp.sin(phases), axis=-1)
    v_recip = jnp.dot(gweight, sf_real**2 + sf_imag**2)

    v_self = jnp.asarray(self.self_const_factor, dtype=dtype) * n_parts
    v_g0 = 0.5 * jnp.asarray(self.g0_factor, dtype=dtype) * (n_parts**2)

    return v_real + v_recip + v_self + v_g0

  def energy(self, positions: jnp.ndarray) -> jnp.ndarray:
    """Evaluates total screened Coulomb energy ``U_ee`` in Hartree."""
    coords = jnp.reshape(positions, (self.n_particles, 2))
    return (1.0 / self.dielectric_constant) * self._unscreened_energy(coords)


def make_monolayer_ewald_potential(
    lattice: geometry.Array,
    nspins: Sequence[int],
    dielectric_constant: float = 1.0,
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    alpha: Optional[float] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Creates a callable evaluating the 2D-periodic screened Coulomb energy."""
  ewald = MonolayerEwaldSum(
      lattice=lattice,
      nspins=nspins,
      dielectric_constant=dielectric_constant,
      ewald_gmax=ewald_gmax,
      nlatvec=nlatvec,
      alpha=alpha,
  )
  return ewald.energy
