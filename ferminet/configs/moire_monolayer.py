"""2D monolayer charge-transfer TMD moire superlattice configuration for FermiNet.

Models spinful carriers in a single 2D plane subject to a periodic scalar moire
potential ``V(r)`` and 2D-periodic screened Coulomb interactions ``1 / (eps * r)``.

By default, parameters are set to the charge-transfer TMD heterobilayer
(WSe2/WS2) continuum model of Zhang, Yuan, & Fu (PRB 2020) and Luo, Dai, & Fu
(arXiv:2406.17645):
  - ``V_0 = -15.0 meV`` (``-5.512e-4 Ha``), ``phi = +45 deg`` (``pi/4 rad``)
  - ``mass = 0.50 m_e``, ``dielectric_constant = 5.0``
  - ``lattice_constant = 8.2 nm = 154.96 Bohr``
In this charge-transfer regime (``phi ~ 45 deg``), doping past half-filling
``nu = 1 + x`` localizes 1 spin-1/2 moment per moire cell in the deep primary
``MM`` (A-sublattice) potential minimum while additional ``x`` carriers populate
the shallower, dispersive ``MX`` (B-sublattice) secondary minimum, realizing a
two-sublattice Periodic Anderson / Kondo lattice within a single 2D plane.
"""

from typing import Tuple, Union
from ferminet import base_config
from ferminet.moire import geometry
from ferminet.moire import hamiltonian as moire_hamiltonian
from ferminet.utils import system
import ml_collections
import numpy as np


def _format_shell_param(arr: np.ndarray) -> Union[float, Tuple[float, ...]]:
  """Formats a 1D shell parameter array as a float (1 shell) or tuple."""
  if arr.size == 1:
    return float(arr[0])
  return tuple(float(x) for x in arr)


def _resolve_moire_nspins(
    mcfg: ml_collections.ConfigDict,
) -> Tuple[int, int]:
  """Resolves the 2-channel ``(n_up, n_down)`` spin tuple."""
  if mcfg.get('nspins') is not None:
    nspins = tuple(
        geometry.validate_nonnegative_int(x, 'moire.nspins entry')
        for x in mcfg.nspins
    )
    return geometry.spin_particle_counts(nspins)

  spin_polarized = bool(mcfg.get('spin_polarized', False))
  n_total_spec = mcfg.get('n_particles')
  if n_total_spec is None:
    n_total_spec = mcfg.get('n_electrons')

  if n_total_spec is not None:
    n_total = geometry.validate_nonnegative_int(
        n_total_spec, 'moire.n_particles'
    )
    if spin_polarized:
      return (n_total, 0)
    n_up = n_total - (n_total // 2)
    n_down = n_total // 2
    return (n_up, n_down)

  n_up = geometry.validate_nonnegative_int(
      mcfg.get('n_up', 3), 'moire.n_up'
  )
  n_down = geometry.validate_nonnegative_int(
      mcfg.get('n_down', 3), 'moire.n_down'
  )
  if spin_polarized:
    return (n_up + n_down, 0)
  return (n_up, n_down)


def _configure_moire_system(
    cfg: ml_collections.ConfigDict,
) -> ml_collections.ConfigDict:
  """Resolves moire geometry, spin counts, and Hamiltonian/network kwargs."""
  mcfg = cfg.system.moire

  nspins = _resolve_moire_nspins(mcfg)
  n_total = geometry.total_particle_count(nspins)
  if n_total <= 0:
    raise ValueError(
        f'Total particle count must be > 0, got nspins={nspins}.'
    )

  with cfg.ignore_type():
    cfg.system.electrons = nspins
  cfg.system.ndim = 2
  cfg.system.molecule = [system.Atom('X', (0.0, 0.0))]

  if mcfg.primitive_lattice is not None:
    prim_lattice = geometry.validate_lattice_2d(
        mcfg.primitive_lattice, name='moire.primitive_lattice'
    )
  else:
    prim_lattice = geometry.triangular_lattice(float(mcfg.lattice_constant))

  if mcfg.supercell_lattice is not None:
    sim_lattice = geometry.validate_lattice_2d(
        mcfg.supercell_lattice, name='moire.supercell_lattice'
    )
  else:
    sim_lattice = geometry.supercell_lattice(
        prim_lattice, np.asarray(mcfg.supercell_matrix)
    )

  moire_lat = (
      prim_lattice
      if mcfg.moire_lattice is None
      else geometry.validate_lattice_2d(
          mcfg.moire_lattice, name='moire.moire_lattice'
      )
  )

  potential_model = mcfg.get('potential_model', None)
  v_arr, phi_arr = moire_hamiltonian.resolve_moire_potential_params(
      mcfg.v_0, mcfg.phi, model=potential_model
  )

  if np.any(v_arr != 0.0):
    geometry.validate_commensurate(moire_lat, sim_lattice)

  with cfg.ignore_type():
    cfg.system.lattice = sim_lattice
    if not cfg.system.make_local_energy_fn:
      cfg.system.make_local_energy_fn = (
          'ferminet.moire.hamiltonian.local_energy'
      )
    local_energy_defaults = {
        'lattice': sim_lattice,
        'moire_lattice': moire_lat,
        'mass': float(mcfg.mass),
        'dielectric_constant': float(mcfg.dielectric_constant),
        'v_0': _format_shell_param(v_arr),
        'phi': _format_shell_param(phi_arr),
        'offset': tuple(float(x) for x in mcfg.offset),
        'potential_model': potential_model,
        'ewald_gmax': int(mcfg.ewald_gmax),
        'nlatvec': int(mcfg.nlatvec),
        'ewald_alpha': mcfg.ewald_alpha,
    }
    local_energy_kwargs = dict(cfg.system.make_local_energy_kwargs)
    local_energy_kwargs.update(local_energy_defaults)
    cfg.system.make_local_energy_kwargs = local_energy_kwargs

    if not cfg.network.make_feature_layer_fn:
      cfg.network.make_feature_layer_fn = (
          'ferminet.pbc.feature_layer.make_pbc_feature_layer'
      )
    feature_layer_kwargs = dict(cfg.network.make_feature_layer_kwargs)
    feature_layer_kwargs.setdefault('include_r_ae', False)
    feature_layer_kwargs.update({
        'lattice': sim_lattice,
        'moire_lattice': (
            moire_lat if bool(mcfg.get('include_moire_features', True)) else None
        ),
    })
    cfg.network.make_feature_layer_kwargs = feature_layer_kwargs

    jastrow_choice = str(cfg.network.get('jastrow', 'default')).lower()
    if jastrow_choice == 'default':
      cfg.network.jastrow = 'monolayer'

    if (
        cfg.network.get('network_type', 'ferminet') == 'psiformer'
        and not cfg.network.full_det
    ):
      raise ValueError(
          "Psiformer only supports full_det=True; set cfg.network.full_det=True "
          "or use cfg.network.network_type='ferminet'."
      )

    jastrow_kwargs = (
        dict(cfg.network.jastrow_kwargs)
        if cfg.network.get('jastrow_kwargs') is not None
        else {}
    )
    jastrow_kwargs.update({
        'mass': float(mcfg.mass),
        'dielectric_constant': float(mcfg.dielectric_constant),
    })
    cfg.network.jastrow_kwargs = jastrow_kwargs

  return cfg


def finalize(cfg: ml_collections.ConfigDict) -> ml_collections.ConfigDict:
  """Finalizes moire monolayer configuration during base_config.resolve(cfg)."""
  return _configure_moire_system(cfg)


def get_config() -> ml_collections.ConfigDict:
  """Returns default ConfigDict for a 2D charge-transfer TMD moire monolayer."""
  cfg = base_config.default()

  cfg.system.ndim = 2
  cfg.pretrain.method = None
  cfg.pretrain.iterations = 0
  cfg.mcmc.scale_by_nuclear_distance = False
  cfg.mcmc.move_width = 2.0
  cfg.network.full_det = True
  cfg.network.jastrow = 'monolayer'
  cfg.network.make_envelope_fn = ''
  cfg.network.make_envelope_kwargs = {}
  cfg.optim.spin_energy = 0.0
  cfg.observables.s2 = False
  cfg.observables.dipole = False
  cfg.observables.density = False

  # Default: WSe2/WS2 single-plane charge-transfer TMD model (Option A)
  # a_M = 8.2 nm = 154.9575 Bohr, m^* = 0.50 m_e, eps = 5.0,
  # V_0 = -15.0 meV = -5.51236e-4 Ha, phi = pi/4 (45 deg).
  # In a 2x2 supercell (4 moire unit cells), n_up=3, n_down=3 corresponds to
  # doped charge-transfer filling nu = 1 + 1/2 = 1.5 (4 localized A-site moments
  # + 2 itinerant B-site carriers).
  cfg.system.moire = ml_collections.ConfigDict(
      {
          'n_up': 3,
          'n_down': 3,
          'nspins': None,
          'n_particles': None,
          'n_electrons': None,
          'spin_polarized': False,
          'mass': 0.50,
          'lattice_constant': 154.9575,
          'primitive_lattice': None,
          'supercell_matrix': ((2, 0), (0, 2)),
          'supercell_lattice': None,
          'moire_lattice': None,
          'v_0': -15.0 / moire_hamiltonian.HARTREE_TO_MEV,
          'phi': float(np.pi / 4.0),
          'offset': (0.0, 0.0),
          'potential_model': 'luo_2024_wse2_ws2',
          'include_moire_features': True,
          'dielectric_constant': 5.0,
          'ewald_gmax': 100,
          'nlatvec': 1,
          'ewald_alpha': None,
      },
      type_safe=False,
  )

  cfg = _configure_moire_system(cfg)
  with cfg.ignore_type():
    cfg.system.set_molecule = finalize
    cfg.config_module = '.moire_monolayer'
  return cfg
