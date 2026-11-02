"""2D periodic geometry utilities for monolayer moire superlattice systems.

All lattice matrices follow FermiNet's column-vector convention: a 2D lattice
matrix ``L`` has shape ``(2, 2)`` where ``L[:, 0]`` and ``L[:, 1]`` are the two
direct lattice vectors ``a_1`` and ``a_2``. Consequently, for a row vector
``r = (x, y)`` or array of positions with trailing dimension ``2``:

  - Fractional coordinates: ``s = r @ inv(L).T``
  - Cartesian coordinates: ``r = s @ L.T``
  - Reciprocal lattice vectors (rows): ``B = 2 * pi * inv(L)`` so that
    ``B @ L = 2 * pi * I``.
"""

from typing import Sequence, Tuple, Union

from ferminet import mcmc
import jax.numpy as jnp
import numpy as np

Array = Union[np.ndarray, jnp.ndarray]


def validate_nonnegative_int(value: object, name: str = 'count') -> int:
  """Validates that ``value`` is a non-negative integer."""
  if isinstance(value, (bool, np.bool_)):
    raise ValueError(f'{name} must be a non-negative integer, got {value!r}.')
  if isinstance(value, (int, np.integer)):
    int_val = int(value)
  elif isinstance(value, (float, np.floating)):
    if not np.isfinite(value) or not float(value).is_integer():
      raise ValueError(f'{name} must be an integer, got {value!r}.')
    int_val = int(value)
  else:
    raise ValueError(f'{name} must be a non-negative integer, got {value!r}.')
  if int_val < 0:
    raise ValueError(f'{name} must be >= 0, got {value!r}.')
  return int_val


def spin_particle_counts(nspins: Sequence[int]) -> Tuple[int, int]:
  """Resolves ``(n_up, n_down)`` from a 2-channel ``nspins`` sequence.

  Args:
    nspins: Sequence ``(n_up, n_down)`` (or length-1 ``(n_up,)``) of
      non-negative integer particle counts per spin channel.

  Returns:
    Tuple ``(n_up, n_down)`` of spin-up and spin-down particle counts.

  Raises:
    ValueError: If ``len(nspins)`` is not 1 or 2, or if any count is negative.
  """
  counts = [validate_nonnegative_int(n, 'nspins entry') for n in nspins]
  if len(counts) == 2:
    return counts[0], counts[1]
  if len(counts) == 1:
    return counts[0], 0
  raise ValueError(
      f'Expected nspins to have length 2 (n_up, n_down), got {nspins}.'
  )


def total_particle_count(nspins: Sequence[int]) -> int:
  """Returns the total number of particles ``n_up + n_down``."""
  n_up, n_down = spin_particle_counts(nspins)
  return n_up + n_down


def triangular_lattice(lattice_constant: float = 1.0) -> np.ndarray:
  """Returns a 2x2 triangular primitive lattice matrix (columns are vectors).

  The primitive vectors are ``a_1 = a_M * (sqrt(3)/2, -1/2)`` and
  ``a_2 = a_M * (0, 1)``, matching the canonical 120-degree hexagonal/triangular
  moire basis.

  Args:
    lattice_constant: Moire lattice constant ``a_M`` in Bohr.

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are ``a_1`` and ``a_2``.
  """
  if lattice_constant <= 0.0:
    raise ValueError(
        f'lattice_constant must be positive, got {lattice_constant}.'
    )
  basis = np.array(
      [
          [np.sqrt(3.0) / 2.0, 0.0],
          [-0.5, 1.0],
      ],
      dtype=np.float64,
  )
  return float(lattice_constant) * basis


def validate_lattice_2d(lattice: Array, name: str = 'lattice') -> np.ndarray:
  """Validates that ``lattice`` is a non-singular 2x2 matrix."""
  lat = np.asarray(lattice, dtype=np.float64)
  if lat.shape != (2, 2):
    raise ValueError(f'{name} must have shape (2, 2), got {lat.shape}.')
  if not np.all(np.isfinite(lat)):
    raise ValueError(f'{name} must have finite entries.')
  area = abs(np.linalg.det(lat))
  if area <= 1e-12:
    raise ValueError(f'{name} must be non-singular, got det={area}.')
  return lat


def supercell_lattice(
    primitive_lattice: Array,
    supercell_matrix: Array,
) -> np.ndarray:
  """Constructs a 2D simulation supercell ``L_sim = L_prim @ S``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` matrix whose columns are primitive
      moire lattice vectors.
    supercell_matrix: Shape ``(2, 2)`` integer matrix with non-zero determinant.

  Returns:
    NumPy array of shape ``(2, 2)`` whose columns are the supercell lattice
    vectors.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sc = np.asarray(supercell_matrix, dtype=np.float64)
  if sc.shape != (2, 2):
    raise ValueError(
        f'supercell_matrix must have shape (2, 2), got {sc.shape}.'
    )
  if not np.allclose(sc, np.round(sc), atol=1e-8):
    raise ValueError('supercell_matrix must have integer entries.')
  sc_int = np.round(sc)
  det = abs(round(float(np.linalg.det(sc_int))))
  if det <= 0:
    raise ValueError('supercell_matrix must have non-zero determinant.')
  return prim @ sc_int


def validate_commensurate(
    primitive_lattice: Array,
    simulation_lattice: Array,
    atol: float = 1e-5,
) -> np.ndarray:
  """Checks that ``simulation_lattice`` is an integer supercell of ``primitive_lattice``.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).
    simulation_lattice: Shape ``(2, 2)`` simulation supercell lattice (columns).
    atol: Tolerance for integer check on ``inv(L_prim) @ L_sim``.

  Returns:
    The rounded ``(2, 2)`` integer supercell matrix ``S = inv(L_prim) @ L_sim``.

  Raises:
    ValueError: If ``simulation_lattice`` is not commensurate with
      ``primitive_lattice``.
  """
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  sim = validate_lattice_2d(simulation_lattice, name='simulation_lattice')
  ratio = np.linalg.inv(prim) @ sim
  rounded = np.round(ratio)
  if not np.allclose(ratio, rounded, atol=atol):
    raise ValueError(
        'Simulation supercell is not commensurate with primitive moire '
        f'lattice: inv(L_prim) @ L_sim = {ratio.tolist()}.'
    )
  return rounded.astype(int)


def cell_area(lattice: Array) -> jnp.ndarray:
  """Returns the area ``|det(lattice)|`` of a 2D cell."""
  return jnp.abs(jnp.linalg.det(jnp.asarray(lattice)))


def reciprocal_lattice(lattice: Array) -> jnp.ndarray:
  """Returns the reciprocal lattice matrix whose columns are ``b_1, b_2``."""
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat).T


def reciprocal_vectors(lattice: Array) -> jnp.ndarray:
  """Returns the reciprocal lattice matrix whose rows are ``b_1, b_2``."""
  lat = jnp.asarray(lattice)
  return 2.0 * jnp.pi * jnp.linalg.inv(lat)


def lll_reduced_basis(lattice: Array, delta: float = 0.75) -> np.ndarray:
  """LLL-reduces a column-vector lattice basis so 1-shell MIC search is exact.

  Args:
    lattice: Shape ``(ndim, ndim)`` matrix whose columns are lattice vectors.
    delta: Lovasz reduction parameter in ``(0.25, 1.0)``.

  Returns:
    NumPy array of shape ``(ndim, ndim)`` whose columns span the same lattice.
  """
  basis = np.asarray(lattice, dtype=np.float64).T.copy()
  ndim = basis.shape[0]
  if not np.all(np.isfinite(basis)):
    return np.asarray(lattice, dtype=np.float64)
  norm_scale = np.linalg.norm(basis, axis=1).max()
  if abs(np.linalg.det(basis)) <= np.finfo(np.float64).eps * norm_scale**ndim:
    return np.asarray(lattice, dtype=np.float64)

  def gram_schmidt() -> Tuple[np.ndarray, np.ndarray]:
    _, r = np.linalg.qr(basis.T)
    b_star_norm = np.diag(r)
    mu = np.zeros((ndim, ndim), dtype=np.float64)
    for i in range(ndim):
      mu[i, :i] = r[:i, i] / b_star_norm[:i]
    return b_star_norm, mu

  k = 1
  while k < ndim:
    b_star_norm, mu = gram_schmidt()
    for j in range(k - 1, -1, -1):
      if abs(mu[k, j]) > 0.5:
        basis[k] = basis[k] - np.round(mu[k, j]) * basis[j]
        b_star_norm, mu = gram_schmidt()
    lovasz = (delta - mu[k, k - 1] ** 2) * b_star_norm[k - 1] ** 2
    if b_star_norm[k] ** 2 >= lovasz:
      k += 1
    else:
      basis[[k, k - 1]] = basis[[k - 1, k]]
      k = max(k - 1, 1)
  return basis.T


def _triangular_reciprocal_basis_rows(primitive_lattice: Array) -> np.ndarray:
  """Returns validated (b_1, b_2) reciprocal basis rows for a triangular lattice."""
  prim = validate_lattice_2d(primitive_lattice, name='primitive_lattice')
  rec_rows = 2.0 * np.pi * np.linalg.inv(prim)

  def _is_triangular_pair(rows: np.ndarray) -> bool:
    norms = np.linalg.norm(rows, axis=1)
    if not np.isclose(norms[0], norms[1], rtol=1e-5, atol=1e-8):
      return False
    cos_theta = np.dot(rows[0], rows[1]) / (norms[0] * norms[1])
    return bool(np.isclose(abs(cos_theta), 0.5, rtol=1e-5, atol=1e-6))

  if not _is_triangular_pair(rec_rows):
    rec_rows = lll_reduced_basis(rec_rows.T).T
    if not _is_triangular_pair(rec_rows):
      raise ValueError(
          'primitive_lattice must define a 2D triangular lattice with six '
          f'equal-length first-shell reciprocal vectors, got {prim.tolist()}.'
      )
  return rec_rows


def moire_first_shell_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the six first-shell moire reciprocal vectors ``g_1, ..., g_6``.

  Orders the six equal-length vectors of norm ``G_M = 4*pi / (sqrt(3)*a_M)`` in
  60-degree steps so that ``first_shell[0:5:2]`` always forms a C3-symmetric
  triad summing to zero.

  Args:
    primitive_lattice: Shape ``(2, 2)`` primitive moire lattice (columns).

  Returns:
    Array of shape ``(6, 2)`` containing the six first-shell G-vectors as rows.
  """
  rec_rows = _triangular_reciprocal_basis_rows(primitive_lattice)
  b1 = jnp.asarray(rec_rows[0])
  b2 = jnp.asarray(rec_rows[1])
  if float(np.dot(rec_rows[0], rec_rows[1])) > 0.0:
    return jnp.stack([b1, b2, b2 - b1, -b1, -b2, b1 - b2], axis=0)
  return jnp.stack([b1, b1 + b2, b2, -b1, -(b1 + b2), -b2], axis=0)


def moire_harmonic_gvectors(primitive_lattice: Array) -> jnp.ndarray:
  """Returns the three C3-related first-shell moire reciprocal vectors.

  Returns three equal-length first-shell reciprocal vectors ``(g_1, g_2, g_3)``
  separated by 120 degrees satisfying ``g_1 + g_2 + g_3 = 0``, defining the
  three-harmonic continuum moire potential
  ``V(r) = 2 V_0 sum_{m=1}^3 cos(g_m . (r - r_0) + phi)``.
  """
  first_shell = moire_first_shell_gvectors(primitive_lattice)
  return first_shell[0:5:2]


def moire_harmonic_shells_from_first_shell(
    first_shell_triad: Array,
    num_shells: int = 1,
) -> jnp.ndarray:
  """Constructs ``num_shells`` C3-symmetric reciprocal vector triads from the 1st shell.

  Given the first-shell C3 triad ``(g_1, g_2, g_3)`` separated by 120 degrees
  with ``g_1 + g_2 + g_3 = 0``, every non-zero reciprocal lattice vector belongs
  to a unique ``+-C3`` triad indexed by integers ``(p, q)`` with
  ``p >= 1, q >= 0``:

    ``G_1^{(p,q)} = p g_1 - q g_3``
    ``G_2^{(p,q)} = p g_2 - q g_1``
    ``G_3^{(p,q)} = p g_3 - q g_2``

  ordered by increasing ``(Q(p, q) = p^2 + p q + q^2, q, p)``.
  """
  n_shells = validate_nonnegative_int(num_shells, 'num_shells')
  if n_shells < 1:
    raise ValueError(f'num_shells must be >= 1, got {num_shells}.')
  g_triad = jnp.asarray(first_shell_triad)
  if g_triad.shape != (3, 2):
    raise ValueError(
        f'first_shell_triad must have shape (3, 2), got {g_triad.shape}.'
    )
  g1, g2, g3 = g_triad[0], g_triad[1], g_triad[2]

  max_idx = int(np.ceil(np.sqrt(float(n_shells)))) + 2
  candidates = []
  for p in range(1, max_idx + 1):
    for q in range(0, max_idx + 1):
      q_metric = p * p + p * q + q * q
      candidates.append((q_metric, q, p))
  candidates.sort()
  selected = candidates[:n_shells]

  shells = []
  for _, q, p in selected:
    pf = float(p)
    qf = float(q)
    shells.append(
        jnp.stack(
            [pf * g1 - qf * g3, pf * g2 - qf * g1, pf * g3 - qf * g2],
            axis=0,
        )
    )
  return jnp.stack(shells, axis=0)


def moire_harmonic_shells(
    primitive_lattice: Array,
    num_shells: int = 1,
) -> jnp.ndarray:
  """Returns the first ``num_shells`` C3-symmetric moire reciprocal triads."""
  first_shell_triad = moire_harmonic_gvectors(primitive_lattice)
  return moire_harmonic_shells_from_first_shell(
      first_shell_triad, num_shells=num_shells
  )


def cartesian_to_fractional(
    positions: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts Cartesian coordinates to fractional coordinates in the cell."""
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  inv_lat = jnp.linalg.inv(lat)
  orig_shape = pos.shape
  coords = jnp.reshape(pos, (-1, ndim))
  fractional = coords @ inv_lat.T
  return jnp.reshape(fractional, orig_shape)


def fractional_to_cartesian(
    fractional: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Converts fractional cell coordinates to Cartesian coordinates."""
  frac = jnp.asarray(fractional)
  lat = jnp.asarray(lattice, dtype=frac.dtype)
  orig_shape = frac.shape
  coords = jnp.reshape(frac, (-1, ndim))
  cartesian = coords @ lat.T
  return jnp.reshape(cartesian, orig_shape)


def wrap_positions(
    positions: Array,
    lattice: Array,
    ndim: int = 2,
) -> jnp.ndarray:
  """Wraps Cartesian positions into the primary simulation cell ``[0, 1)^ndim``."""
  pos = jnp.asarray(positions)
  lat = jnp.asarray(lattice, dtype=pos.dtype)
  rec = 2.0 * jnp.pi * jnp.linalg.inv(lat)
  return mcmc.map_to_simulation_cell(pos, lat, rec, ndim)


def make_minimum_image_distance_fn(lattice: Array):
  """Builds an AD-safe minimum-image displacement and distance function in 2D."""
  reduced_np = lll_reduced_basis(lattice)
  ndim = reduced_np.shape[0]
  inv_reduced_np = np.linalg.inv(reduced_np)
  grid = np.meshgrid(*[np.array([-1, 0, 1]) for _ in range(ndim)], indexing='ij')
  point_list = np.stack([g.ravel() for g in grid], axis=-1)
  shifts_np = point_list @ reduced_np.T

  def distance_fn(
      pos_a: jnp.ndarray, pos_b: jnp.ndarray
  ) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dtype = pos_a.dtype
    reduced = jnp.asarray(reduced_np, dtype=dtype)
    inv_reduced = jnp.asarray(inv_reduced_np, dtype=dtype)
    shifts = jnp.asarray(shifts_np, dtype=dtype)

    diff = pos_a[..., :, None, :] - pos_b[..., None, :, :]
    frac_diff = diff @ inv_reduced.T
    wrapped = diff - jnp.round(frac_diff) @ reduced.T
    diff_all = wrapped[..., None, :] + shifts
    dists_sq_all = jnp.sum(diff_all**2, axis=-1)
    min_idx = jnp.argmin(dists_sq_all, axis=-1)
    best_disp = jnp.take_along_axis(
        diff_all, min_idx[..., None, None], axis=-2
    )[..., 0, :]
    best_dist_sq = jnp.sum(best_disp**2, axis=-1)
    zero_mask = best_dist_sq == 0.0
    best_dist = jnp.where(
        zero_mask,
        0.0,
        jnp.sqrt(jnp.where(zero_mask, 1.0, best_dist_sq)),
    )
    return best_disp, best_dist

  return distance_fn
