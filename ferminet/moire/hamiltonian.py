"""Hamiltonian for 2D monolayer moire superlattice systems."""

from typing import Callable, Mapping, Optional, Sequence, Tuple, Union

import chex
from ferminet import hamiltonian
from ferminet import networks
from ferminet.moire import ewald as moire_ewald
from ferminet.moire import geometry
from ferminet.utils import utils
import folx
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

HARTREE_TO_MEV: float = 27211.386245988
ScalarOrSequence = Union[float, Sequence[float], geometry.Array]


def local_kinetic_energy(
    f: networks.FermiNetLike,
    mass: float = 1.0,
    use_scan: bool = False,
    complex_output: bool = False,
    laplacian_method: str = 'default',
) -> hamiltonian.KineticEnergy:
  r"""Creates a function for the effective-mass local kinetic energy.

  Evaluates:
    ``T_L(R) = -(1 / (2 * mass)) * sum_i (nabla_i^2 Psi(R)) / Psi(R)``.

  Args:
    f: Callable evaluating the wavefunction as a ``(phase_or_sign, logabs)``
      tuple.
    mass: Carrier effective mass ``m^* / m_0 > 0``.
    use_scan: Whether to use ``lax.scan`` for the default Laplacian loop.
    complex_output: Whether ``f`` outputs a complex phase in its first slot.
    laplacian_method: Either ``'default'`` or ``'folx'``.

  Returns:
    Callable ``ke(params, data) -> scalar`` evaluating the local kinetic energy.
  """
  if mass <= 0.0:
    raise ValueError(f'Effective mass must be positive, got {mass}.')
  inv_mass = 1.0 / float(mass)
  phase_f = utils.select_output(f, 0)
  logabs_f = utils.select_output(f, 1)

  if laplacian_method == 'default':

    def _lapl_over_f_default(params, data):
      pos = data.positions
      n = pos.shape[0]
      eye = jnp.eye(n, dtype=pos.dtype)
      grad_f = jax.grad(logabs_f, argnums=1)

      def grad_f_closure(x):
        return grad_f(params, x, data.spins, data.atoms, data.charges)

      primal, dgrad_f = jax.linearize(grad_f_closure, pos)
      phase_primal = None

      if complex_output:
        grad_phase = jax.grad(phase_f, argnums=1)

        def grad_phase_closure(x):
          return grad_phase(params, x, data.spins, data.atoms, data.charges)

        phase_primal, dgrad_phase = jax.linearize(grad_phase_closure, pos)
        hessian_diagonal = lambda i: (
            dgrad_f(eye[i])[i] + 1.0j * dgrad_phase(eye[i])[i]
        )
      else:
        hessian_diagonal = lambda i: dgrad_f(eye[i])[i]

      if use_scan:
        _, diagonal = lax.scan(
            lambda i, _: (i + 1, hessian_diagonal(i)), 0, None, length=n
        )
        result = -0.5 * inv_mass * jnp.sum(diagonal)
      else:
        init_val = jnp.asarray(
            0.0 + 0.0j if complex_output else 0.0,
            dtype=(
                jnp.complex128
                if (complex_output and pos.dtype == jnp.float64)
                else (jnp.complex64 if complex_output else pos.dtype)
            ),
        )
        result = (
            -0.5
            * inv_mass
            * lax.fori_loop(
                0, n, lambda i, val: val + hessian_diagonal(i), init_val
            )
        )

      result -= 0.5 * inv_mass * jnp.sum(primal**2)
      if complex_output:
        assert phase_primal is not None
        result += 0.5 * inv_mass * jnp.sum(phase_primal**2)
        result -= 1.0j * inv_mass * jnp.sum(primal * phase_primal)
      return result

    return _lapl_over_f_default

  elif laplacian_method == 'folx':

    def _lapl_over_f_folx(params, data):
      f_closure = lambda x: f(params, x, data.spins, data.atoms, data.charges)
      f_wrapped = folx.forward_laplacian(f_closure, sparsity_threshold=6)
      output = f_wrapped(data.positions)
      result = (
          -0.5
          * inv_mass
          * (
              output[1].laplacian
              + jnp.sum(output[1].jacobian.dense_array ** 2)
          )
      )
      if complex_output:
        result -= 0.5j * inv_mass * output[0].laplacian
        result += 0.5 * inv_mass * jnp.sum(output[0].jacobian.dense_array ** 2)
        result -= (
            1.0j
            * inv_mass
            * jnp.sum(
                output[0].jacobian.dense_array * output[1].jacobian.dense_array
            )
        )
      return result

    return _lapl_over_f_folx

  else:
    raise NotImplementedError(
        f'Laplacian method {laplacian_method} not implemented.'
    )


# Built-in moire potential parameter presets in the carrier potential convention:
# V(r) = 2 * sum_{s=1}^{N_shells} V_s * sum_{j=1}^3 cos(G_j^{(s)} . (r - r_0) + phi_s),
# where amplitudes V_s are in Hartree (1 Ha = 27211.386245988 meV) and
# phases phi_s are in radians, ordered according to
# geometry.moire_harmonic_shells(primitive_lattice, num_shells).
MOIRE_POTENTIAL_PRESETS: Mapping[
    str, Tuple[Tuple[float, ...], Tuple[float, ...]]
] = {
    # 1. Zhang, Yuan, and Fu (2020) [arXiv:1910.14061] & Luo, Dai, and Fu (2024)
    # [arXiv:2406.17645] 1-shell charge-transfer TMD (WSe2/WS2) continuum model:
    # V_0 = -15.0 meV (-5.512e-4 Ha), phi = +45 deg (+pi/4 rad).
    # Produces primary MM (A-sublattice) minimum and secondary MX (B-sublattice)
    # charge-transfer minimum within a single 2D plane.
    'zhang_2020_1shell': (
        (-5.512e-4,),
        (float(np.pi / 4.0),),
    ),
    'luo_2024_wse2_ws2': (
        (-15.0 / HARTREE_TO_MEV,),
        (float(np.pi / 4.0),),
    ),
    'charge_transfer_tmd_1shell': (
        (-15.0 / HARTREE_TO_MEV,),
        (float(np.pi / 4.0),),
    ),
    # 2. Luskin, Geier, Fu, and Zhu (2026) [arXiv:2608.08917] Table S1
    # 3-shell rigid DFT fits (hole convention U_h = -V_v):
    'luskin_2026_3shell_scan': (
        (-1.344658e-4, +1.025306e-5, -2.215984e-5),
        (0.888198, 0.185005, -1.923353),
    ),
    'luskin_2026_3shell_dftd3': (
        (-1.801452e-4, +1.113504e-5, -2.631251e-5),
        (0.941954, -0.002705, -1.717928),
    ),
    # 3. Luskin, Geier, Fu, and Zhu (2026) [arXiv:2608.08917] full untwisted
    # (theta = 0 deg, delta = 0.04) relaxed scalar potential:
    'luskin_2026_relaxed_scan': (
        (
            -2.808078e-4,
            -2.625710e-5,
            -1.665482e-5,
            -3.403920e-6,
            -3.511100e-6,
            -2.563096e-6,
        ),
        (-0.455925, -3.070619, 1.947429, 2.755679, -2.674509, 1.952711),
    ),
    'luskin_2026_relaxed_3shell_scan': (
        (-2.808078e-4, -2.625710e-5, -1.665482e-5),
        (-0.455925, -3.070619, 1.947429),
    ),
    'luskin_2026_relaxed_dftd3': (
        (
            -3.201871e-4,
            -2.646910e-5,
            -1.827854e-5,
            -3.606763e-6,
            -3.604609e-6,
            -2.913338e-6,
        ),
        (-0.554206, 3.140469, 1.646516, 2.616314, -2.617504, 1.737927),
    ),
    'luskin_2026_relaxed_3shell_dftd3': (
        (-3.201871e-4, -2.646910e-5, -1.827854e-5),
        (-0.554206, 3.140469, 1.646516),
    ),
    'luskin_2026_vm_relax_only_scan': (
        (
            -1.405746e-4,
            -4.876900e-6,
            -1.466914e-5,
            -1.807245e-6,
            -1.997600e-6,
            -2.294530e-6,
        ),
        (-0.936193, -2.749852, 1.797346, 2.455897, -2.333573, 1.818268),
    ),
    'luskin_2026_vm_relax_only_dftd3': (
        (
            -1.889297e-4,
            -4.785667e-6,
            -1.709443e-5,
            -2.158672e-6,
            -2.155135e-6,
            -2.720477e-6,
        ),
        (-0.991179, 3.135378, 1.485998, 2.255780, -2.257292, 1.604413),
    ),
    # 4. Wang, Gao, and Tong (2026) [arXiv:2602.09526] 3-shell and 1-shell fits:
    'wang_2026_rtype_3shell': (
        (-3.116975e-4, -8.268598e-7, -4.673433e-5),
        (1.820520, 0.0, 1.800620),
    ),
    'wang_2026_htype_3shell': (
        (-1.845635e-4, -2.097774e-6, -3.614323e-5),
        (-2.056730, 0.0, -0.841550),
    ),
    'wang_2026_rtype_1shell': (
        (-2.671676e-4,),
        (-0.865683,),
    ),
    'wang_2026_htype_1shell': (
        (-1.782342e-4,),
        (1.103048,),
    ),
}


def resolve_moire_potential_params(
    amplitude: ScalarOrSequence = 0.0,
    phase: ScalarOrSequence = 0.0,
    model: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
  """Normalizes scalar, multi-shell sequence, or preset moire potential parameters."""
  if model is not None and str(model).strip().lower() not in ('', 'none'):
    key = str(model).strip().lower()
    if key not in MOIRE_POTENTIAL_PRESETS:
      valid = ', '.join(sorted(MOIRE_POTENTIAL_PRESETS.keys()))
      raise ValueError(
          f'Unknown moire potential preset {model!r}. Available presets: '
          f'{valid}.'
      )
    v_preset, phi_preset = MOIRE_POTENTIAL_PRESETS[key]
    return (
        np.asarray(v_preset, dtype=np.float64),
        np.asarray(phi_preset, dtype=np.float64),
    )

  v_arr = np.atleast_1d(np.asarray(amplitude, dtype=np.float64))
  phi_arr = np.atleast_1d(np.asarray(phase, dtype=np.float64))
  if v_arr.ndim != 1 or v_arr.size < 1:
    raise ValueError(
        f'Moire amplitude must be a scalar or non-empty 1D sequence, got shape '
        f'{v_arr.shape}.'
    )
  if phi_arr.ndim != 1 or phi_arr.size < 1:
    raise ValueError(
        f'Moire phase must be a scalar or non-empty 1D sequence, got shape '
        f'{phi_arr.shape}.'
    )
  if not (np.all(np.isfinite(v_arr)) and np.all(np.isfinite(phi_arr))):
    raise ValueError('Moire amplitude and phase entries must be finite.')

  if phi_arr.size == 1 and v_arr.size > 1:
    phi_arr = np.broadcast_to(phi_arr, v_arr.shape).copy()
  elif v_arr.size != phi_arr.size:
    raise ValueError(
        f'Moire amplitude and phase must have the same number of shells, got '
        f'len(amplitude)={v_arr.size} and len(phase)={phi_arr.size}.'
    )
  return v_arr, phi_arr


def single_species_moire_potential(
    coords: jnp.ndarray,
    harmonic_gvectors: geometry.Array,
    amplitude: ScalarOrSequence,
    phase: ScalarOrSequence = 0.0,
    offset: Sequence[float] | geometry.Array = (0.0, 0.0),
) -> jnp.ndarray:
  r"""Evaluates the single- or multi-shell scalar moire potential in a 2D plane.

  Evaluates the C3-symmetric harmonic expansion:

  .. math::
    V(\mathbf{r}) = 2 \sum_{s=1}^{N_{\mathrm{shells}}} V_s
      \sum_{m=1}^{3} \cos\!\left(
        \mathbf{G}_m^{(s)} \cdot (\mathbf{r} - \mathbf{r}_0) + \phi_s
      \right).
  """
  if coords.shape[0] == 0:
    return jnp.asarray(0.0, dtype=coords.dtype)

  dtype = coords.dtype
  v_arr = jnp.atleast_1d(jnp.asarray(amplitude, dtype=dtype))
  phi_arr = jnp.atleast_1d(jnp.asarray(phase, dtype=dtype))
  if v_arr.ndim != 1 or phi_arr.ndim != 1:
    raise ValueError(
        'amplitude and phase in single_species_moire_potential must be '
        f'scalars or 1D sequences, got shapes {v_arr.shape} and {phi_arr.shape}.'
    )
  num_shells = int(v_arr.shape[0])
  if phi_arr.shape[0] == 1 and num_shells > 1:
    phi_arr = jnp.broadcast_to(phi_arr, (num_shells,))
  elif phi_arr.shape[0] != num_shells:
    raise ValueError(
        f'amplitude and phase must have the same number of shells, got '
        f'{num_shells} and {phi_arr.shape[0]}.'
    )

  if not isinstance(amplitude, jax.core.Tracer):
    if np.all(np.asarray(amplitude, dtype=np.float64) == 0.0):
      return jnp.asarray(0.0, dtype=dtype)

  gvecs = jnp.asarray(harmonic_gvectors, dtype=dtype)
  if gvecs.ndim == 2:
    if gvecs.shape != (3, 2):
      raise ValueError(
          f'2D harmonic_gvectors must have shape (3, 2), got {gvecs.shape}.'
      )
    if num_shells == 1:
      gvecs = gvecs[None, :, :]
    else:
      gvecs = jnp.asarray(
          geometry.moire_harmonic_shells_from_first_shell(
              gvecs, num_shells=num_shells
          ),
          dtype=dtype,
      )
  elif gvecs.ndim == 3:
    if gvecs.shape[1:] != (3, 2):
      raise ValueError(
          '3D harmonic_gvectors must have shape (num_shells, 3, 2), got '
          f'{gvecs.shape}.'
      )
    if gvecs.shape[0] < num_shells:
      raise ValueError(
          f'harmonic_gvectors has {gvecs.shape[0]} shells, fewer than '
          f'len(amplitude)={num_shells}.'
      )
    gvecs = gvecs[:num_shells]
  else:
    raise ValueError(
        'harmonic_gvectors must have shape (3, 2) or (num_shells, 3, 2), got '
        f'{gvecs.shape}.'
    )

  r0 = jnp.asarray(offset, dtype=dtype)
  shifted = coords - r0[None, :]
  projections = jnp.einsum('id,smd->ism', shifted, gvecs)
  shell_cos_sum = jnp.sum(
      jnp.cos(projections + phi_arr[None, :, None]), axis=(0, 2)
  )
  return 2.0 * jnp.sum(v_arr * shell_cos_sum)


def make_moire_potential(
    nspins: Optional[Sequence[int]] = None,
    simulation_lattice: Optional[geometry.Array] = None,
    moire_lattice: Optional[geometry.Array] = None,
    v_0: ScalarOrSequence = 0.0,
    phi: ScalarOrSequence = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
    potential_model: Optional[str] = None,
) -> Callable[[jnp.ndarray], jnp.ndarray]:
  """Builds a callable evaluating the monolayer moire potential energy.

  Validates that the primitive moire lattice is commensurate with
  ``simulation_lattice``. Supports 1-shell scalar parameters, multi-shell
  sequences ``(v_1, ..., v_S)``, and named literature presets in
  ``MOIRE_POTENTIAL_PRESETS`` via ``potential_model``.
  """
  n_total = (
      geometry.total_particle_count(nspins) if nspins is not None else None
  )
  if simulation_lattice is None:
    raise ValueError('simulation_lattice must be provided.')
  sim_lat = geometry.validate_lattice_2d(
      simulation_lattice, name='simulation_lattice'
  )
  prim_lat = sim_lat if moire_lattice is None else moire_lattice

  v_arr, phi_arr = resolve_moire_potential_params(
      v_0, phi, model=potential_model
  )

  if np.any(v_arr != 0.0):
    geometry.validate_commensurate(prim_lat, sim_lat)
    gvecs = np.asarray(
        geometry.moire_harmonic_shells(prim_lat, num_shells=len(v_arr)),
        dtype=np.float64,
    )
  else:
    gvecs = np.zeros((len(v_arr), 3, 2), dtype=np.float64)

  offset_arr = np.asarray(offset, dtype=np.float64)
  if offset_arr.shape != (2,):
    raise ValueError(f'offset must have shape (2,), got {offset_arr.shape}.')

  def moire_potential_fn(positions: jnp.ndarray) -> jnp.ndarray:
    if n_total is not None:
      coords = jnp.reshape(positions, (n_total, 2))
    else:
      coords = jnp.reshape(positions, (-1, 2))
    return single_species_moire_potential(
        coords,
        gvecs,
        amplitude=v_arr,
        phase=phi_arr,
        offset=offset_arr,
    )

  return moire_potential_fn


def local_energy(
    f: networks.FermiNetLike,
    charges: jnp.ndarray,
    nspins: Sequence[int],
    use_scan: bool = False,
    ndim: int = 2,
    complex_output: bool = False,
    laplacian_method: str = 'default',
    states: int = 0,
    state_specific: bool = False,
    pp_type: str = 'ccecp',
    pp_symbols: Sequence[str] | None = None,
    lattice: Optional[geometry.Array] = None,
    moire_lattice: Optional[geometry.Array] = None,
    mass: float = 1.0,
    dielectric_constant: float = 1.0,
    v_0: ScalarOrSequence = 0.0,
    phi: ScalarOrSequence = 0.0,
    offset: Sequence[float] = (0.0, 0.0),
    potential_model: Optional[str] = None,
    ewald_gmax: int = 100,
    nlatvec: int = 1,
    ewald_alpha: Optional[float] = None,
) -> hamiltonian.LocalEnergy:
  """Creates the local energy function for a 2D monolayer moire superlattice.

  Matches the ``MakeLocalEnergy`` calling convention used by ``ferminet.train``.
  """
  if ndim != 2:
    raise ValueError(
        f'Moire monolayer Hamiltonian requires ndim=2, got ndim={ndim}.'
    )
  if states > 0 or state_specific:
    raise NotImplementedError(
        'Excited states are not implemented for moire monolayer Hamiltonian.'
    )
  if pp_symbols:
    raise NotImplementedError(
        'Pseudopotentials are not supported for moire monolayer Hamiltonian.'
    )
  del charges, pp_type

  if lattice is None:
    lattice = np.eye(2)

  ke_fn = local_kinetic_energy(
      f=f,
      mass=mass,
      use_scan=use_scan,
      complex_output=complex_output,
      laplacian_method=laplacian_method,
  )

  moire_pot_fn = make_moire_potential(
      nspins=nspins,
      simulation_lattice=lattice,
      moire_lattice=moire_lattice,
      v_0=v_0,
      phi=phi,
      offset=offset,
      potential_model=potential_model,
  )

  coulomb_pot_fn = moire_ewald.make_monolayer_ewald_potential(
      lattice=lattice,
      nspins=nspins,
      dielectric_constant=dielectric_constant,
      ewald_gmax=ewald_gmax,
      nlatvec=nlatvec,
      alpha=ewald_alpha,
  )

  def _e_l(
      params: networks.ParamTree,
      key: chex.PRNGKey,
      data: networks.FermiNetData,
  ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
    del key
    kinetic = ke_fn(params, data)
    v_moire = moire_pot_fn(data.positions)
    v_coulomb = coulomb_pot_fn(data.positions)
    return kinetic + v_moire + v_coulomb, None

  return _e_l
