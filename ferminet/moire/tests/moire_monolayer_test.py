# Copyright 2026 DeepMind Technologies Limited.
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

"""Tests for 2D monolayer moire potentials, Ewald summation, and Hamiltonian."""

from absl.testing import absltest
from absl.testing import parameterized
from ferminet import jastrows
from ferminet import networks
from ferminet.configs import moire_monolayer
from ferminet.moire import ewald
from ferminet.moire import geometry
from ferminet.moire import hamiltonian
from ferminet.pbc import feature_layer as pbc_feature_layer
import jax
import jax.numpy as jnp
import numpy as np


class MoireMonolayerTest(parameterized.TestCase):

  def test_moire_harmonic_shells_geometry_and_c3_symmetry(self):
    a_m = 132.28
    prim = geometry.triangular_lattice(a_m)
    g_first = np.asarray(geometry.moire_harmonic_gvectors(prim))
    shells = np.asarray(geometry.moire_harmonic_shells(prim, num_shells=10))
    self.assertEqual(shells.shape, (10, 3, 2))

    # Shell 0 must be identical to moire_harmonic_gvectors.
    np.testing.assert_allclose(shells[0], g_first, rtol=1e-6, atol=1e-6)

    # Expected squared-norm multipliers Q(p, q) = p^2 + p*q + q^2.
    expected_q = [1, 3, 4, 7, 7, 9, 12, 13, 13, 16]
    g_m_norm = 4.0 * np.pi / (np.sqrt(3.0) * a_m)

    # 120-degree counterclockwise rotation matrix.
    c, s = -0.5, np.sqrt(3.0) / 2.0
    r_c3 = np.array([[c, -s], [s, c]], dtype=np.float64)

    seen_vectors = []
    for idx, q_mult in enumerate(expected_q):
      triad = shells[idx]
      np.testing.assert_allclose(
          np.sum(triad, axis=0), np.zeros(2), atol=1e-5
      )
      norms = np.linalg.norm(triad, axis=1)
      np.testing.assert_allclose(
          norms, np.full(3, np.sqrt(q_mult) * g_m_norm), rtol=1e-5
      )
      np.testing.assert_allclose(
          (r_c3 @ triad[0]), triad[1], rtol=1e-5, atol=1e-6
      )
      np.testing.assert_allclose(
          (r_c3 @ triad[1]), triad[2], rtol=1e-5, atol=1e-6
      )
      np.testing.assert_allclose(
          (r_c3 @ triad[2]), triad[0], rtol=1e-5, atol=1e-6
      )

      integer_proj = (triad @ prim) / (2.0 * np.pi)
      np.testing.assert_allclose(
          integer_proj, np.round(integer_proj), atol=1e-5
      )

      for g_vec in triad:
        for prev in seen_vectors:
          self.assertGreater(np.linalg.norm(g_vec - prev), 1e-4)
          self.assertGreater(np.linalg.norm(g_vec + prev), 1e-4)
        seen_vectors.append(g_vec)

  def test_option_a_charge_transfer_tmd_wyckoff_hierarchy(self):
    # Option A (single-plane charge-transfer TMD, phi = pi/4, V_0 = -15 meV):
    # At Wyckoff A = (0, 0), V_M(A) = 2*V_0*(3*cos(pi/4)) = 3*sqrt(2)*V_0 < 0
    # (primary triangular minimum).
    # At Wyckoff B = (1/3, 2/3) and C = (2/3, 1/3),
    # V_M(B) = 2*V_0*(3*cos(pi/4 - 2*pi/3)) < V_M(C) = 2*V_0*(3*cos(pi/4 + 2*pi/3)).
    prim = geometry.triangular_lattice(132.28)
    pot_fn = hamiltonian.make_moire_potential(
        simulation_lattice=prim,
        moire_lattice=prim,
        potential_model='charge_transfer_tmd_1shell',
    )
    r_a = prim @ np.array([0.0, 0.0])
    r_b = prim @ np.array([2.0 / 3.0, 1.0 / 3.0])
    r_c = prim @ np.array([1.0 / 3.0, 2.0 / 3.0])
    v_a = float(pot_fn(jnp.asarray(r_a))) * hamiltonian.HARTREE_TO_MEV
    v_b = float(pot_fn(jnp.asarray(r_b))) * hamiltonian.HARTREE_TO_MEV
    v_c = float(pot_fn(jnp.asarray(r_c))) * hamiltonian.HARTREE_TO_MEV

    expected_a = 6.0 * (-15.0) * np.cos(np.pi / 4.0)
    expected_b = 6.0 * (-15.0) * np.cos(np.pi / 4.0 - 2.0 * np.pi / 3.0)
    expected_c = 6.0 * (-15.0) * np.cos(np.pi / 4.0 + 2.0 * np.pi / 3.0)
    self.assertAlmostEqual(v_a, expected_a, places=4)
    self.assertAlmostEqual(v_b, expected_b, places=4)
    self.assertAlmostEqual(v_c, expected_c, places=4)
    self.assertLess(v_a, v_b)
    self.assertLess(v_b, v_c)

  @parameterized.parameters(
      *list(hamiltonian.MOIRE_POTENTIAL_PRESETS.keys())
  )
  def test_preset_periodicity_c3_symmetry_and_zero_cell_mean(self, preset_name):
    a_m = 132.28
    prim = geometry.triangular_lattice(a_m)
    sim = geometry.supercell_lattice(prim, np.array([[2, 0], [0, 2]]))
    offset = np.array([7.0, -13.0])

    pot_fn = hamiltonian.make_moire_potential(
        simulation_lattice=sim,
        moire_lattice=prim,
        offset=offset,
        potential_model=preset_name,
    )

    rng = np.random.default_rng(42)
    pts = rng.uniform(-100.0, 100.0, size=(6, 2))
    c, s = -0.5, np.sqrt(3.0) / 2.0
    r_c3 = np.array([[c, -s], [s, c]], dtype=np.float64)

    for pt in pts:
      val_0 = float(pot_fn(jnp.asarray(pt)))
      val_a1 = float(pot_fn(jnp.asarray(pt + prim[:, 0])))
      val_a2 = float(pot_fn(jnp.asarray(pt + prim[:, 1])))
      self.assertAlmostEqual(val_0, val_a1, places=6)
      self.assertAlmostEqual(val_0, val_a2, places=6)

      pt_rot = offset + r_c3 @ (pt - offset)
      val_rot = float(pot_fn(jnp.asarray(pt_rot)))
      self.assertAlmostEqual(val_0, val_rot, places=6)

  def test_monolayer_ewald_alpha_invariance_and_periodicity(self):
    lat = geometry.supercell_lattice(
        geometry.triangular_lattice(20.0), [[2, 0], [0, 2]]
    )
    n_particles = 4
    key = jax.random.PRNGKey(17)
    frac = jax.random.uniform(key, shape=(n_particles, 2))
    pos = (frac @ jnp.asarray(lat).T).reshape(-1)

    ew1 = ewald.MonolayerEwaldSum(
        lat,
        n_particles=n_particles,
        dielectric_constant=5.0,
        ewald_gmax=80,
        nlatvec=2,
        alpha=0.18,
    )
    ew2 = ewald.MonolayerEwaldSum(
        lat,
        n_particles=n_particles,
        dielectric_constant=5.0,
        ewald_gmax=80,
        nlatvec=2,
        alpha=0.36,
    )
    np.testing.assert_allclose(
        ew1.energy(pos), ew2.energy(pos), atol=1e-4, rtol=1e-4
    )

    # Shift particle 0 by a supercell lattice vector: energy must be invariant.
    pos_shifted = pos.reshape(n_particles, 2).at[0].add(jnp.asarray(lat[:, 0]))
    np.testing.assert_allclose(
        ew1.energy(pos), ew1.energy(pos_shifted.reshape(-1)), atol=1e-5
    )

  def test_monolayer_jastrow_and_pbc_feature_layer_gradients_finite(self):
    prim = jnp.asarray(geometry.triangular_lattice(50.0))
    sim = jnp.asarray(geometry.supercell_lattice(prim, [[2, 0], [0, 2]]))

    init_j, apply_j = jastrows.make_monolayer_jastrow(
        lattice=sim, ndim=2, mass=0.45, dielectric_constant=5.0
    )
    params_j = init_j()
    pos = jnp.array([[2.0, 3.0], [15.0, -4.0], [-8.0, 12.0]])
    ee_disp = pos[:, None, :] - pos[None, :, :]
    val = apply_j(ee_disp, params_j, nspins=(2, 1))
    self.assertLess(float(val), 0.0)

    grad_pos = jax.grad(
        lambda p: apply_j(p[:, None, :] - p[None, :, :], params_j, (2, 1))
    )(pos)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_pos))))

    fl = pbc_feature_layer.make_pbc_feature_layer(
        natoms=1,
        nspins=(2, 1),
        ndim=2,
        lattice=sim,
        include_r_ae=False,
        moire_lattice=prim,
    )
    dims, fl_params = fl.init()
    self.assertEqual(dims, (8, 5))

    def feat_sum(p):
      ae = p[:, None, :]
      ee = p[:, None, :] - p[None, :, :]
      r_ae = jnp.linalg.norm(ae, axis=-1)
      r_ee = jnp.linalg.norm(ee + jnp.eye(3)[..., None], axis=-1)
      ae_f, ee_f = fl.apply(ae=ae, r_ae=r_ae, ee=ee, r_ee=r_ee, **fl_params)
      return jnp.sum(ae_f) + jnp.sum(ee_f)

    grad_fl = jax.grad(feat_sum)(pos)
    self.assertTrue(bool(jnp.all(jnp.isfinite(grad_fl))))

  def test_moire_monolayer_config_and_local_energy(self):
    cfg = moire_monolayer.get_config()
    cfg.system.moire.n_up = 2
    cfg.system.moire.n_down = 1
    cfg.system.moire.supercell_matrix = ((2, 0), (0, 1))
    cfg = moire_monolayer.finalize(cfg)

    kwargs = cfg.system.make_local_energy_kwargs
    self.assertEqual(cfg.system.electrons, (2, 1))
    self.assertAlmostEqual(
        float(kwargs['v_0']) * hamiltonian.HARTREE_TO_MEV, -15.0, places=4
    )
    self.assertAlmostEqual(float(kwargs['phi']), np.pi / 4.0, places=5)

    def dummy_wf(params, pos, spins, atoms, charges):
      del params, spins, atoms, charges
      return jnp.ones((), dtype=pos.dtype), -0.001 * jnp.sum(pos**2)

    e_l_fn = hamiltonian.local_energy(
        f=dummy_wf,
        charges=jnp.zeros((1,)),
        nspins=cfg.system.electrons,
        use_scan=False,
        complex_output=False,
        **kwargs,
    )
    data = networks.FermiNetData(
        positions=jnp.array(
            [10.0, -5.0, -12.0, 8.0, 3.0, 14.0], dtype=jnp.float32
        ),
        spins=jnp.array([1.0, 1.0, -1.0], dtype=jnp.float32),
        atoms=jnp.zeros((1, 2), dtype=jnp.float32),
        charges=jnp.zeros((1,), dtype=jnp.float32),
    )
    total_e, _ = e_l_fn({}, jax.random.PRNGKey(0), data)
    self.assertTrue(bool(jnp.isfinite(total_e)))


if __name__ == '__main__':
  absltest.main()
