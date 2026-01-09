"""Tests for domain-specific unit modules."""

import pytest
import numpy as np
from dimtensor import DimArray
from dimtensor.core.dimensions import Dimension, DIMENSIONLESS


class TestAstronomyUnits:
    """Test astronomy units."""

    def test_parsec_dimension(self):
        """Parsec should have length dimension."""
        from dimtensor.domains.astronomy import parsec
        assert parsec.dimension == Dimension(length=1)

    def test_parsec_scale(self):
        """Parsec should be approximately 3.086e16 m."""
        from dimtensor.domains.astronomy import parsec
        assert pytest.approx(parsec.scale, rel=1e-3) == 3.0857e16

    def test_au_dimension(self):
        """AU should have length dimension."""
        from dimtensor.domains.astronomy import AU
        assert AU.dimension == Dimension(length=1)

    def test_au_scale(self):
        """AU should be approximately 1.496e11 m."""
        from dimtensor.domains.astronomy import AU
        assert pytest.approx(AU.scale, rel=1e-3) == 1.496e11

    def test_light_year_dimension(self):
        """Light-year should have length dimension."""
        from dimtensor.domains.astronomy import light_year
        assert light_year.dimension == Dimension(length=1)

    def test_light_year_scale(self):
        """Light-year should be approximately 9.461e15 m."""
        from dimtensor.domains.astronomy import light_year
        assert pytest.approx(light_year.scale, rel=1e-3) == 9.461e15

    def test_solar_mass_dimension(self):
        """Solar mass should have mass dimension."""
        from dimtensor.domains.astronomy import solar_mass
        assert solar_mass.dimension == Dimension(mass=1)

    def test_solar_mass_scale(self):
        """Solar mass should be approximately 1.989e30 kg."""
        from dimtensor.domains.astronomy import solar_mass
        assert pytest.approx(solar_mass.scale, rel=1e-3) == 1.989e30

    def test_solar_luminosity_dimension(self):
        """Solar luminosity should have power dimension."""
        from dimtensor.domains.astronomy import solar_luminosity
        assert solar_luminosity.dimension == Dimension(mass=1, length=2, time=-3)

    def test_arcsecond_dimensionless(self):
        """Arcsecond should be dimensionless (angular)."""
        from dimtensor.domains.astronomy import arcsecond
        assert arcsecond.dimension == DIMENSIONLESS

    def test_parsec_to_light_year_conversion(self):
        """Test parsec to light-year conversion."""
        from dimtensor.domains.astronomy import parsec, light_year
        # 1 parsec = 3.26156 light-years
        ratio = parsec.conversion_factor(light_year)
        assert pytest.approx(ratio, rel=1e-3) == 3.262

    def test_distance_calculation(self):
        """Test using astronomy units in calculations."""
        from dimtensor.domains.astronomy import parsec, AU
        # Proxima Centauri is about 1.3 parsec away
        distance_pc = DimArray([1.3], parsec)
        # Convert to AU
        distance_au = distance_pc.to(AU)
        # 1 pc ~ 206265 AU
        expected = 1.3 * 206265
        assert pytest.approx(distance_au.magnitude()[0], rel=1e-2) == expected

    def test_kiloparsec_megaparsec(self):
        """Test kpc and Mpc units."""
        from dimtensor.domains.astronomy import parsec, kiloparsec, megaparsec
        assert pytest.approx(kiloparsec.scale / parsec.scale) == 1000
        assert pytest.approx(megaparsec.scale / parsec.scale) == 1e6

    def test_earth_jupiter_masses(self):
        """Test Earth and Jupiter mass units."""
        from dimtensor.domains.astronomy import earth_mass, jupiter_mass
        # Jupiter is about 318 Earth masses
        ratio = jupiter_mass.scale / earth_mass.scale
        assert pytest.approx(ratio, rel=1e-2) == 318


class TestChemistryUnits:
    """Test chemistry units."""

    def test_dalton_dimension(self):
        """Dalton should have mass dimension."""
        from dimtensor.domains.chemistry import dalton
        assert dalton.dimension == Dimension(mass=1)

    def test_dalton_scale(self):
        """Dalton should be approximately 1.661e-27 kg."""
        from dimtensor.domains.chemistry import dalton
        assert pytest.approx(dalton.scale, rel=1e-3) == 1.661e-27

    def test_molar_dimension(self):
        """Molar should have amount/volume dimension."""
        from dimtensor.domains.chemistry import molar
        assert molar.dimension == Dimension(amount=1, length=-3)

    def test_molar_concentration_hierarchy(self):
        """Test mM, uM, nM scale relationships."""
        from dimtensor.domains.chemistry import molar, millimolar, micromolar, nanomolar
        # Scale factors relative to mol/m^3
        assert pytest.approx(molar.scale / millimolar.scale) == 1000
        assert pytest.approx(millimolar.scale / micromolar.scale) == 1000
        assert pytest.approx(micromolar.scale / nanomolar.scale) == 1000

    def test_ppm_dimensionless(self):
        """ppm should be dimensionless."""
        from dimtensor.domains.chemistry import ppm
        assert ppm.dimension == DIMENSIONLESS

    def test_ppm_scale(self):
        """ppm should have scale 1e-6."""
        from dimtensor.domains.chemistry import ppm
        assert ppm.scale == 1e-6

    def test_ppb_ppt_scales(self):
        """Test ppb and ppt scales."""
        from dimtensor.domains.chemistry import ppb, ppt
        assert ppb.scale == 1e-9
        assert ppt.scale == 1e-12

    def test_angstrom_dimension(self):
        """Angstrom should have length dimension."""
        from dimtensor.domains.chemistry import angstrom
        assert angstrom.dimension == Dimension(length=1)

    def test_angstrom_scale(self):
        """Angstrom should be 1e-10 m."""
        from dimtensor.domains.chemistry import angstrom
        assert angstrom.scale == 1e-10

    def test_molal_dimension(self):
        """Molality should have amount/mass dimension."""
        from dimtensor.domains.chemistry import molal
        assert molal.dimension == Dimension(amount=1, mass=-1)

    def test_hartree_energy_dimension(self):
        """Hartree should have energy dimension."""
        from dimtensor.domains.chemistry import hartree
        assert hartree.dimension == Dimension(mass=1, length=2, time=-2)

    def test_debye_dipole_moment(self):
        """Debye should have dipole moment dimension (charge * length)."""
        from dimtensor.domains.chemistry import debye
        # Dipole moment = charge * length = C * m = A * s * m
        assert debye.dimension == Dimension(current=1, time=1, length=1)

    def test_concentration_calculation(self):
        """Test using chemistry units in calculations."""
        from dimtensor.domains.chemistry import molar, millimolar
        conc = DimArray([0.001], molar)
        conc_mm = conc.to(millimolar)
        assert pytest.approx(conc_mm.magnitude()[0], rel=1e-6) == 1.0


class TestEngineeringUnits:
    """Test engineering units."""

    def test_megapascal_dimension(self):
        """MPa should have pressure dimension."""
        from dimtensor.domains.engineering import MPa
        assert MPa.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_megapascal_scale(self):
        """MPa should be 1e6 Pa."""
        from dimtensor.domains.engineering import MPa
        assert MPa.scale == 1e6

    def test_ksi_dimension(self):
        """ksi should have pressure dimension."""
        from dimtensor.domains.engineering import ksi
        assert ksi.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_ksi_scale(self):
        """ksi should be approximately 6.895e6 Pa."""
        from dimtensor.domains.engineering import ksi
        assert pytest.approx(ksi.scale, rel=1e-3) == 6.895e6

    def test_btu_dimension(self):
        """BTU should have energy dimension."""
        from dimtensor.domains.engineering import BTU
        assert BTU.dimension == Dimension(mass=1, length=2, time=-2)

    def test_btu_scale(self):
        """BTU should be approximately 1055 J."""
        from dimtensor.domains.engineering import BTU
        assert pytest.approx(BTU.scale, rel=1e-3) == 1055

    def test_horsepower_dimension(self):
        """Horsepower should have power dimension."""
        from dimtensor.domains.engineering import hp
        assert hp.dimension == Dimension(mass=1, length=2, time=-3)

    def test_horsepower_scale(self):
        """Horsepower should be approximately 745.7 W."""
        from dimtensor.domains.engineering import hp
        assert pytest.approx(hp.scale, rel=1e-3) == 745.7

    def test_kilowatt_hour_dimension(self):
        """kWh should have energy dimension."""
        from dimtensor.domains.engineering import kWh
        assert kWh.dimension == Dimension(mass=1, length=2, time=-2)

    def test_kilowatt_hour_scale(self):
        """kWh should be 3.6e6 J."""
        from dimtensor.domains.engineering import kWh
        assert kWh.scale == 3.6e6

    def test_rpm_dimension(self):
        """rpm should have frequency dimension."""
        from dimtensor.domains.engineering import rpm
        assert rpm.dimension == Dimension(time=-1)

    def test_gpm_dimension(self):
        """gpm should have volumetric flow dimension."""
        from dimtensor.domains.engineering import gpm
        assert gpm.dimension == Dimension(length=3, time=-1)

    def test_foot_pound_torque_dimension(self):
        """ft·lb should have torque/energy dimension."""
        from dimtensor.domains.engineering import ft_lb
        assert ft_lb.dimension == Dimension(mass=1, length=2, time=-2)

    def test_mil_dimension(self):
        """mil should have length dimension."""
        from dimtensor.domains.engineering import mil
        assert mil.dimension == Dimension(length=1)

    def test_mil_scale(self):
        """mil should be 2.54e-5 m (0.001 inch)."""
        from dimtensor.domains.engineering import mil
        assert pytest.approx(mil.scale, rel=1e-6) == 2.54e-5

    def test_pressure_conversion(self):
        """Test pressure unit conversions."""
        from dimtensor.domains.engineering import MPa, ksi
        from dimtensor.core.units import psi
        # 1 ksi = 1000 psi
        stress_ksi = DimArray([1.0], ksi)
        stress_mpa = stress_ksi.to(MPa)
        # 1 ksi ~ 6.895 MPa
        assert pytest.approx(stress_mpa.magnitude()[0], rel=1e-3) == 6.895

    def test_power_conversion(self):
        """Test power unit conversions."""
        from dimtensor.domains.engineering import hp
        from dimtensor.core.units import W
        power_hp = DimArray([1.0], hp)
        power_w = power_hp.to(W)
        assert pytest.approx(power_w.magnitude()[0], rel=1e-3) == 745.7

    def test_metric_vs_mechanical_horsepower(self):
        """Test metric vs mechanical horsepower."""
        from dimtensor.domains.engineering import hp, PS
        # Mechanical hp is about 1.4% more than metric hp
        ratio = hp.scale / PS.scale
        assert pytest.approx(ratio, rel=1e-3) == 1.014


class TestNuclearUnits:
    """Test nuclear physics units."""

    def test_electronvolt_dimension(self):
        """Electronvolt should have energy dimension."""
        from dimtensor.domains.nuclear import eV
        assert eV.dimension == Dimension(mass=1, length=2, time=-2)

    def test_electronvolt_scale(self):
        """eV should be exactly 1.602176634e-19 J."""
        from dimtensor.domains.nuclear import eV
        assert eV.scale == 1.602176634e-19

    def test_kev_scale(self):
        """keV should be 1000 eV."""
        from dimtensor.domains.nuclear import eV, keV
        assert pytest.approx(keV.scale / eV.scale) == 1000

    def test_mev_scale(self):
        """MeV should be 1e6 eV."""
        from dimtensor.domains.nuclear import eV, MeV
        assert pytest.approx(MeV.scale / eV.scale) == 1e6

    def test_gev_scale(self):
        """GeV should be 1e9 eV."""
        from dimtensor.domains.nuclear import eV, GeV
        assert pytest.approx(GeV.scale / eV.scale) == 1e9

    def test_barn_dimension(self):
        """Barn should have area dimension."""
        from dimtensor.domains.nuclear import barn
        assert barn.dimension == Dimension(length=2)

    def test_barn_scale(self):
        """Barn should be exactly 1e-28 m²."""
        from dimtensor.domains.nuclear import barn
        assert barn.scale == 1e-28

    def test_millibarn_scale(self):
        """Millibarn should be 0.001 barn."""
        from dimtensor.domains.nuclear import barn, millibarn
        assert pytest.approx(millibarn.scale / barn.scale) == 0.001

    def test_microbarn_scale(self):
        """Microbarn should be 1e-6 barn."""
        from dimtensor.domains.nuclear import barn, microbarn
        assert pytest.approx(microbarn.scale / barn.scale) == 1e-6

    def test_becquerel_dimension(self):
        """Becquerel should have frequency dimension."""
        from dimtensor.domains.nuclear import becquerel
        assert becquerel.dimension == Dimension(time=-1)

    def test_becquerel_scale(self):
        """Becquerel should be 1.0 (SI base)."""
        from dimtensor.domains.nuclear import becquerel
        assert becquerel.scale == 1.0

    def test_curie_scale(self):
        """Curie should be exactly 3.7e10 Bq."""
        from dimtensor.domains.nuclear import curie
        assert curie.scale == 3.7e10

    def test_gray_dimension(self):
        """Gray should have absorbed dose dimension (L²·T⁻²)."""
        from dimtensor.domains.nuclear import gray
        assert gray.dimension == Dimension(length=2, time=-2)

    def test_gray_scale(self):
        """Gray should be 1.0 (SI base)."""
        from dimtensor.domains.nuclear import gray
        assert gray.scale == 1.0

    def test_rad_scale(self):
        """Rad should be exactly 0.01 Gy."""
        from dimtensor.domains.nuclear import rad
        assert rad.scale == 0.01

    def test_sievert_dimension(self):
        """Sievert should have dose equivalent dimension (L²·T⁻²)."""
        from dimtensor.domains.nuclear import sievert
        assert sievert.dimension == Dimension(length=2, time=-2)

    def test_sievert_scale(self):
        """Sievert should be 1.0 (SI base)."""
        from dimtensor.domains.nuclear import sievert
        assert sievert.scale == 1.0

    def test_rem_scale(self):
        """Rem should be exactly 0.01 Sv."""
        from dimtensor.domains.nuclear import rem
        assert rem.scale == 0.01

    def test_proton_mass_energy(self):
        """Test proton rest mass energy calculation."""
        from dimtensor.domains.nuclear import MeV
        from dimtensor.core.units import joule
        # Proton rest mass: 938.3 MeV
        energy = DimArray([938.3], MeV)
        energy_j = energy.to(joule)
        assert pytest.approx(energy_j.magnitude()[0], rel=1e-3) == 1.503e-10

    def test_radioactivity_conversion(self):
        """Test radioactivity unit conversion."""
        from dimtensor.domains.nuclear import becquerel, curie
        # 1 Ci = 3.7e10 Bq
        activity = DimArray([1.0], curie)
        activity_bq = activity.to(becquerel)
        assert pytest.approx(activity_bq.magnitude()[0]) == 3.7e10


class TestGeophysicsUnits:
    """Test geophysics units."""

    def test_gal_dimension(self):
        """Gal should have acceleration dimension."""
        from dimtensor.domains.geophysics import gal
        assert gal.dimension == Dimension(length=1, time=-2)

    def test_gal_scale(self):
        """Gal should be 0.01 m/s² (1 cm/s²)."""
        from dimtensor.domains.geophysics import gal
        assert gal.scale == 0.01

    def test_milligal_scale(self):
        """Milligal should be 0.001 Gal."""
        from dimtensor.domains.geophysics import gal, milligal
        assert pytest.approx(milligal.scale / gal.scale) == 0.001

    def test_eotvos_dimension(self):
        """Eotvos should have gravity gradient dimension (T⁻²)."""
        from dimtensor.domains.geophysics import eotvos
        assert eotvos.dimension == Dimension(time=-2)

    def test_eotvos_scale(self):
        """Eotvos should be 1e-9 s⁻²."""
        from dimtensor.domains.geophysics import eotvos
        assert eotvos.scale == 1e-9

    def test_darcy_dimension(self):
        """Darcy should have area dimension (permeability)."""
        from dimtensor.domains.geophysics import darcy
        assert darcy.dimension == Dimension(length=2)

    def test_darcy_scale(self):
        """Darcy should be approximately 9.869e-13 m²."""
        from dimtensor.domains.geophysics import darcy
        assert pytest.approx(darcy.scale, rel=1e-6) == 9.869233e-13

    def test_millidarcy_scale(self):
        """Millidarcy should be 0.001 darcy."""
        from dimtensor.domains.geophysics import darcy, millidarcy
        assert pytest.approx(millidarcy.scale / darcy.scale) == 0.001

    def test_gamma_dimension(self):
        """Gamma should have magnetic flux density dimension."""
        from dimtensor.domains.geophysics import gamma
        assert gamma.dimension == Dimension(mass=1, time=-2, current=-1)

    def test_gamma_scale(self):
        """Gamma should be 1e-9 T (1 nT)."""
        from dimtensor.domains.geophysics import gamma
        assert gamma.scale == 1e-9

    def test_oersted_dimension(self):
        """Oersted should have magnetic field intensity dimension."""
        from dimtensor.domains.geophysics import oersted
        assert oersted.dimension == Dimension(current=1, length=-1)

    def test_oersted_scale(self):
        """Oersted should be approximately 79.577 A/m."""
        from dimtensor.domains.geophysics import oersted
        assert pytest.approx(oersted.scale, rel=1e-3) == 79.5774715

    def test_gravity_anomaly_calculation(self):
        """Test gravity anomaly in milligal."""
        from dimtensor.domains.geophysics import milligal
        from dimtensor.core.units import m, s
        # 50 mGal gravity anomaly
        g_anomaly = DimArray([50.0], milligal)
        # Convert to m/s²
        accel_unit = m / (s**2)
        g_si = g_anomaly.to(accel_unit)
        assert pytest.approx(g_si.magnitude()[0], rel=1e-6) == 5e-4

    def test_earth_magnetic_field_gamma(self):
        """Test Earth's magnetic field in gamma."""
        from dimtensor.domains.geophysics import gamma
        from dimtensor.core.units import tesla
        # Earth's field ~ 50,000 gamma = 50,000 nT = 50 µT
        earth_field = DimArray([50000.0], gamma)
        field_t = earth_field.to(tesla)
        assert pytest.approx(field_t.magnitude()[0], rel=1e-6) == 5e-5

    def test_permeability_conversion(self):
        """Test permeability unit conversion."""
        from dimtensor.domains.geophysics import darcy, millidarcy
        # Typical sandstone: 100 mD
        perm = DimArray([100.0], millidarcy)
        perm_darcy = perm.to(darcy)
        assert pytest.approx(perm_darcy.magnitude()[0]) == 0.1


class TestBiophysicsUnits:
    """Test biophysics units."""

    def test_katal_dimension(self):
        """Katal should have enzyme activity dimension (N·T⁻¹)."""
        from dimtensor.domains.biophysics import katal
        assert katal.dimension == Dimension(amount=1, time=-1)

    def test_katal_scale(self):
        """Katal should be 1.0 (SI base, mol/s)."""
        from dimtensor.domains.biophysics import katal
        assert katal.scale == 1.0

    def test_enzyme_unit_dimension(self):
        """Enzyme unit should have enzyme activity dimension."""
        from dimtensor.domains.biophysics import enzyme_unit
        assert enzyme_unit.dimension == Dimension(amount=1, time=-1)

    def test_enzyme_unit_scale(self):
        """Enzyme unit should be 1e-6 mol / 60 s."""
        from dimtensor.domains.biophysics import enzyme_unit
        expected = 1e-6 / 60.0
        assert pytest.approx(enzyme_unit.scale, rel=1e-6) == expected

    def test_enzyme_unit_to_katal(self):
        """Test enzyme unit to katal conversion."""
        from dimtensor.domains.biophysics import enzyme_unit, katal
        # 1 U = 1.667e-8 kat
        ratio = enzyme_unit.scale / katal.scale
        assert pytest.approx(ratio, rel=1e-3) == 1.667e-8

    def test_cells_per_mL_dimension(self):
        """Cells per mL should have number density dimension (L⁻³)."""
        from dimtensor.domains.biophysics import cells_per_mL
        assert cells_per_mL.dimension == Dimension(length=-3)

    def test_cells_per_mL_scale(self):
        """Cells per mL should be 1e6 m⁻³."""
        from dimtensor.domains.biophysics import cells_per_mL
        assert cells_per_mL.scale == 1e6

    def test_cells_per_uL_scale(self):
        """Cells per μL should be 1e9 m⁻³."""
        from dimtensor.domains.biophysics import cells_per_uL
        assert cells_per_uL.scale == 1e9

    def test_cell_concentration_conversion(self):
        """Test cell concentration conversion."""
        from dimtensor.domains.biophysics import cells_per_mL, cells_per_uL
        # 1000 cells/μL = 1e6 cells/mL
        conc_ul = DimArray([1000.0], cells_per_uL)
        conc_ml = conc_ul.to(cells_per_mL)
        assert pytest.approx(conc_ml.magnitude()[0]) == 1e6

    def test_millivolt_dimension(self):
        """Millivolt should have voltage dimension."""
        from dimtensor.domains.biophysics import millivolt
        assert millivolt.dimension == Dimension(mass=1, length=2, time=-3, current=-1)

    def test_millivolt_scale(self):
        """Millivolt should be 0.001 V."""
        from dimtensor.domains.biophysics import millivolt
        assert millivolt.scale == 0.001

    def test_membrane_potential(self):
        """Test membrane potential calculation."""
        from dimtensor.domains.biophysics import millivolt
        from dimtensor.core.units import volt
        # Resting potential: -70 mV
        v_rest = DimArray([-70.0], millivolt)
        v_si = v_rest.to(volt)
        assert pytest.approx(v_si.magnitude()[0]) == -0.07

    def test_enzyme_activity_calculation(self):
        """Test enzyme activity conversion."""
        from dimtensor.domains.biophysics import enzyme_unit, katal
        # 100 U enzyme activity
        activity_u = DimArray([100.0], enzyme_unit)
        activity_kat = activity_u.to(katal)
        assert pytest.approx(activity_kat.magnitude()[0], rel=1e-3) == 1.667e-6


class TestMaterialsUnits:
    """Test materials science units."""

    def test_strain_dimensionless(self):
        """Strain should be dimensionless."""
        from dimtensor.domains.materials import strain
        from dimtensor.core.dimensions import DIMENSIONLESS
        assert strain.dimension == DIMENSIONLESS

    def test_microstrain_scale(self):
        """Microstrain should be 1e-6."""
        from dimtensor.domains.materials import microstrain
        assert microstrain.scale == 1e-6

    def test_percent_strain_scale(self):
        """Percent strain should be 0.01."""
        from dimtensor.domains.materials import percent_strain
        assert percent_strain.scale == 0.01

    def test_vickers_dimensionless(self):
        """Vickers hardness should be dimensionless."""
        from dimtensor.domains.materials import vickers
        from dimtensor.core.dimensions import DIMENSIONLESS
        assert vickers.dimension == DIMENSIONLESS

    def test_brinell_dimensionless(self):
        """Brinell hardness should be dimensionless."""
        from dimtensor.domains.materials import brinell
        from dimtensor.core.dimensions import DIMENSIONLESS
        assert brinell.dimension == DIMENSIONLESS

    def test_rockwell_C_dimensionless(self):
        """Rockwell C hardness should be dimensionless."""
        from dimtensor.domains.materials import rockwell_C
        from dimtensor.core.dimensions import DIMENSIONLESS
        assert rockwell_C.dimension == DIMENSIONLESS

    def test_mpa_sqrt_m_dimension(self):
        """MPa·√m should have fracture toughness dimension."""
        from dimtensor.domains.materials import MPa_sqrt_m
        from fractions import Fraction
        assert MPa_sqrt_m.dimension == Dimension(mass=1, length=Fraction(-1, 2), time=-2)

    def test_mpa_sqrt_m_scale(self):
        """MPa·√m should have scale 1e6."""
        from dimtensor.domains.materials import MPa_sqrt_m
        assert MPa_sqrt_m.scale == 1e6

    def test_ksi_sqrt_in_dimension(self):
        """ksi·√in should have fracture toughness dimension."""
        from dimtensor.domains.materials import ksi_sqrt_in
        from fractions import Fraction
        assert ksi_sqrt_in.dimension == Dimension(mass=1, length=Fraction(-1, 2), time=-2)

    def test_fracture_toughness_conversion(self):
        """Test fracture toughness conversion."""
        from dimtensor.domains.materials import MPa_sqrt_m, ksi_sqrt_in
        # 1 ksi·√in ≈ 1.099 MPa·√m
        k_ksi = DimArray([1.0], ksi_sqrt_in)
        k_mpa = k_ksi.to(MPa_sqrt_m)
        assert pytest.approx(k_mpa.magnitude()[0], rel=1e-3) == 1.099

    def test_w_per_m_k_dimension(self):
        """W/(m·K) should have thermal conductivity dimension."""
        from dimtensor.domains.materials import W_per_m_K
        assert W_per_m_K.dimension == Dimension(mass=1, length=1, time=-3, temperature=-1)

    def test_w_per_m_k_scale(self):
        """W/(m·K) should have scale 1.0 (SI base)."""
        from dimtensor.domains.materials import W_per_m_K
        assert W_per_m_K.scale == 1.0

    def test_s_per_m_dimension(self):
        """S/m should have electrical conductivity dimension."""
        from dimtensor.domains.materials import S_per_m
        assert S_per_m.dimension == Dimension(mass=-1, length=-3, time=3, current=2)

    def test_s_per_m_scale(self):
        """S/m should have scale 1.0 (SI base)."""
        from dimtensor.domains.materials import S_per_m
        assert S_per_m.scale == 1.0

    def test_strain_conversion(self):
        """Test strain unit conversion."""
        from dimtensor.domains.materials import microstrain, strain
        # 1500 microstrain = 0.0015 strain
        epsilon = DimArray([1500.0], microstrain)
        epsilon_base = epsilon.to(strain)
        assert pytest.approx(epsilon_base.magnitude()[0]) == 0.0015


class TestPhotometryUnits:
    """Test photometry units."""

    def test_lumen_dimension(self):
        """Lumen should have luminous flux dimension."""
        from dimtensor.domains.photometry import lumen
        assert lumen.dimension == Dimension(luminosity=1)

    def test_lumen_scale(self):
        """Lumen should have scale 1.0 (SI derived)."""
        from dimtensor.domains.photometry import lumen
        assert lumen.scale == 1.0

    def test_lux_dimension(self):
        """Lux should have illuminance dimension."""
        from dimtensor.domains.photometry import lux
        assert lux.dimension == Dimension(luminosity=1, length=-2)

    def test_lux_scale(self):
        """Lux should have scale 1.0 (lm/m²)."""
        from dimtensor.domains.photometry import lux
        assert lux.scale == 1.0

    def test_foot_candle_dimension(self):
        """Foot-candle should have illuminance dimension."""
        from dimtensor.domains.photometry import foot_candle
        assert foot_candle.dimension == Dimension(luminosity=1, length=-2)

    def test_foot_candle_scale(self):
        """Foot-candle should be approximately 10.764 lux."""
        from dimtensor.domains.photometry import foot_candle
        # 1 fc = 1 lm/ft² = 10.764 lm/m²
        assert pytest.approx(foot_candle.scale, rel=1e-3) == 10.764

    def test_nit_dimension(self):
        """Nit should have luminance dimension."""
        from dimtensor.domains.photometry import nit
        assert nit.dimension == Dimension(luminosity=1, length=-2)

    def test_nit_scale(self):
        """Nit should have scale 1.0 (cd/m²)."""
        from dimtensor.domains.photometry import nit
        assert nit.scale == 1.0

    def test_stilb_dimension(self):
        """Stilb should have luminance dimension."""
        from dimtensor.domains.photometry import stilb
        assert stilb.dimension == Dimension(luminosity=1, length=-2)

    def test_stilb_scale(self):
        """Stilb should be 10000 cd/m²."""
        from dimtensor.domains.photometry import stilb
        assert stilb.scale == 10000.0

    def test_lambert_dimension(self):
        """Lambert should have luminance dimension."""
        from dimtensor.domains.photometry import lambert
        assert lambert.dimension == Dimension(luminosity=1, length=-2)

    def test_lambert_scale(self):
        """Lambert should be 10000/π cd/m²."""
        from dimtensor.domains.photometry import lambert
        import math
        assert pytest.approx(lambert.scale, rel=1e-6) == 10000.0 / math.pi

    def test_lm_per_w_dimension(self):
        """lm/W should have luminous efficacy dimension."""
        from dimtensor.domains.photometry import lm_per_W
        assert lm_per_W.dimension == Dimension(luminosity=1, mass=-1, length=-2, time=3)

    def test_lm_per_w_scale(self):
        """lm/W should have scale 1.0 (SI derived)."""
        from dimtensor.domains.photometry import lm_per_W
        assert lm_per_W.scale == 1.0

    def test_illuminance_conversion(self):
        """Test illuminance conversion."""
        from dimtensor.domains.photometry import lux, foot_candle
        # 100 lux to foot-candles
        illum = DimArray([100.0], lux)
        illum_fc = illum.to(foot_candle)
        assert pytest.approx(illum_fc.magnitude()[0], rel=1e-2) == 9.29

    def test_luminance_conversion(self):
        """Test luminance conversion."""
        from dimtensor.domains.photometry import nit, stilb
        # 1000 nit = 0.1 stilb
        brightness = DimArray([1000.0], nit)
        brightness_sb = brightness.to(stilb)
        assert pytest.approx(brightness_sb.magnitude()[0]) == 0.1


class TestInformationUnits:
    """Test information theory units."""

    def test_bit_dimensionless(self):
        """Bit should be dimensionless."""
        from dimtensor.domains.information import bit
        from dimtensor.core.dimensions import DIMENSIONLESS
        assert bit.dimension == DIMENSIONLESS

    def test_bit_scale(self):
        """Bit should have scale 1.0."""
        from dimtensor.domains.information import bit
        assert bit.scale == 1.0

    def test_nat_scale(self):
        """Nat should be ln(2) bits."""
        from dimtensor.domains.information import nat
        import math
        assert pytest.approx(nat.scale, rel=1e-6) == math.log(2)

    def test_byte_scale(self):
        """Byte should be 8 bits."""
        from dimtensor.domains.information import byte
        assert byte.scale == 8.0

    def test_kilobyte_scale(self):
        """Kilobyte should be 1024 bytes = 8192 bits."""
        from dimtensor.domains.information import kilobyte
        assert kilobyte.scale == 8192.0

    def test_megabyte_scale(self):
        """Megabyte should be 1024² bytes."""
        from dimtensor.domains.information import megabyte
        assert megabyte.scale == 8388608.0  # 1024^2 * 8

    def test_gigabyte_scale(self):
        """Gigabyte should be 1024³ bytes."""
        from dimtensor.domains.information import gigabyte
        assert gigabyte.scale == 8589934592.0  # 1024^3 * 8

    def test_bit_per_second_dimension(self):
        """Bit per second should have frequency dimension."""
        from dimtensor.domains.information import bit_per_second
        assert bit_per_second.dimension == Dimension(time=-1)

    def test_bit_per_second_scale(self):
        """Bit per second should have scale 1.0."""
        from dimtensor.domains.information import bit_per_second
        assert bit_per_second.scale == 1.0

    def test_kilobit_per_second_scale(self):
        """Kilobit per second should be 1000 bit/s."""
        from dimtensor.domains.information import kilobit_per_second
        assert kilobit_per_second.scale == 1e3

    def test_megabit_per_second_scale(self):
        """Megabit per second should be 1e6 bit/s."""
        from dimtensor.domains.information import megabit_per_second
        assert megabit_per_second.scale == 1e6

    def test_byte_per_second_scale(self):
        """Byte per second should be 8 bit/s."""
        from dimtensor.domains.information import byte_per_second
        assert byte_per_second.scale == 8.0

    def test_data_size_conversion(self):
        """Test data size conversion."""
        from dimtensor.domains.information import byte, bit
        # 1024 bytes to bits
        data = DimArray([1024.0], byte)
        data_bits = data.to(bit)
        assert pytest.approx(data_bits.magnitude()[0]) == 8192.0

    def test_data_rate_conversion(self):
        """Test data rate conversion."""
        from dimtensor.domains.information import megabit_per_second, kilobit_per_second
        # 1 Mbps = 1000 kbps
        rate = DimArray([1.0], megabit_per_second)
        rate_kbps = rate.to(kilobit_per_second)
        assert pytest.approx(rate_kbps.magnitude()[0]) == 1000.0


class TestImperialUnits:
    """Test Imperial and US customary units."""

    def test_inch_dimension(self):
        """Inch should have length dimension."""
        from dimtensor.domains.imperial import inch
        assert inch.dimension == Dimension(length=1)

    def test_inch_scale(self):
        """Inch should be exactly 0.0254 m."""
        from dimtensor.domains.imperial import inch
        assert inch.scale == 0.0254

    def test_foot_scale(self):
        """Foot should be 12 inches = 0.3048 m."""
        from dimtensor.domains.imperial import foot
        assert foot.scale == 0.3048

    def test_yard_scale(self):
        """Yard should be 3 feet = 0.9144 m."""
        from dimtensor.domains.imperial import yard
        assert yard.scale == 0.9144

    def test_mile_scale(self):
        """Mile should be 5280 feet = 1609.344 m."""
        from dimtensor.domains.imperial import mile
        assert mile.scale == 1609.344

    def test_pound_dimension(self):
        """Pound should have mass dimension."""
        from dimtensor.domains.imperial import pound
        assert pound.dimension == Dimension(mass=1)

    def test_pound_scale(self):
        """Pound should be exactly 0.45359237 kg."""
        from dimtensor.domains.imperial import pound
        assert pound.scale == 0.45359237

    def test_ounce_scale(self):
        """Ounce should be 1/16 pound."""
        from dimtensor.domains.imperial import pound, ounce
        assert pytest.approx(ounce.scale * 16) == pound.scale

    def test_gallon_dimension(self):
        """Gallon should have volume dimension."""
        from dimtensor.domains.imperial import gallon
        assert gallon.dimension == Dimension(length=3)

    def test_gallon_scale(self):
        """Gallon should be approximately 3.785 L."""
        from dimtensor.domains.imperial import gallon
        assert pytest.approx(gallon.scale, rel=1e-6) == 3.785411784e-3

    def test_pint_scale(self):
        """Pint should be 1/8 gallon."""
        from dimtensor.domains.imperial import gallon, pint
        assert pytest.approx(pint.scale * 8) == gallon.scale

    def test_btu_dimension(self):
        """BTU should have energy dimension."""
        from dimtensor.domains.imperial import BTU
        assert BTU.dimension == Dimension(mass=1, length=2, time=-2)

    def test_btu_scale(self):
        """BTU should be approximately 1055 J."""
        from dimtensor.domains.imperial import BTU
        assert pytest.approx(BTU.scale, rel=1e-3) == 1055.06

    def test_psi_dimension(self):
        """psi should have pressure dimension."""
        from dimtensor.domains.imperial import psi
        assert psi.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_psi_scale(self):
        """psi should be approximately 6895 Pa."""
        from dimtensor.domains.imperial import psi
        assert pytest.approx(psi.scale, rel=1e-3) == 6894.757

    def test_mph_dimension(self):
        """mph should have velocity dimension."""
        from dimtensor.domains.imperial import mph
        assert mph.dimension == Dimension(length=1, time=-1)

    def test_mph_scale(self):
        """mph should be approximately 0.447 m/s."""
        from dimtensor.domains.imperial import mph
        assert pytest.approx(mph.scale, rel=1e-3) == 0.44704

    def test_length_conversion(self):
        """Test length conversion."""
        from dimtensor.domains.imperial import foot, mile
        from dimtensor.core.units import meter
        # 1 mile in feet
        dist = DimArray([1.0], mile)
        dist_ft = dist.to(foot)
        assert pytest.approx(dist_ft.magnitude()[0]) == 5280.0

    def test_speed_conversion(self):
        """Test speed conversion."""
        from dimtensor.domains.imperial import mph
        from dimtensor.core.units import m, s
        # 60 mph to m/s
        speed = DimArray([60.0], mph)
        speed_si = speed.to(m / s)
        assert pytest.approx(speed_si.magnitude()[0], rel=1e-3) == 26.82


class TestNaturalUnits:
    """Test natural units (c = ℏ = 1)."""

    def test_gev_dimension(self):
        """GeV should have energy dimension."""
        from dimtensor.domains.natural import GeV
        assert GeV.dimension == Dimension(mass=1, length=2, time=-2)

    def test_gev_scale(self):
        """GeV should be 1.602176634e-10 J."""
        from dimtensor.domains.natural import GeV
        assert GeV.scale == 1.602176634e-10

    def test_mev_scale(self):
        """MeV should be 1e-3 GeV."""
        from dimtensor.domains.natural import GeV, MeV
        assert pytest.approx(MeV.scale / GeV.scale) == 1e-3

    def test_gev_mass_dimension(self):
        """GeV/c² should have energy dimension (in natural units)."""
        from dimtensor.domains.natural import GeV_mass
        assert GeV_mass.dimension == Dimension(mass=1, length=2, time=-2)

    def test_to_natural_mass(self):
        """Test converting SI mass to natural units."""
        from dimtensor.domains.natural import to_natural
        from dimtensor.core.units import kg
        # Electron mass: 9.109e-31 kg ~ 0.511 MeV
        m_e = DimArray([9.109e-31], kg)
        m_nat = to_natural(m_e, 1e6)  # Convert to MeV
        assert pytest.approx(m_nat.magnitude()[0], rel=1e-2) == 0.511

    def test_to_natural_length(self):
        """Test converting SI length to natural units."""
        from dimtensor.domains.natural import to_natural
        from dimtensor.core.units import meter
        # 1 fm = 1e-15 m ~ 5.068 GeV^-1
        length = DimArray([1e-15], meter)
        length_nat = to_natural(length, 1e9)  # Convert to GeV^-1
        assert pytest.approx(length_nat.magnitude()[0], rel=1e-2) == 5.07

    def test_from_natural_mass(self):
        """Test converting natural mass to SI."""
        from dimtensor.domains.natural import from_natural
        # Electron mass in natural units: 0.511 MeV
        m_nat = 0.511
        m_si = from_natural(m_nat, "mass", 1e6)
        # Handle scalar case (0-dimensional array)
        value = float(m_si.magnitude()) if m_si.magnitude().ndim == 0 else m_si.magnitude()[0]
        assert pytest.approx(value, rel=1e-2) == 9.109e-31

    def test_from_natural_energy(self):
        """Test converting natural energy to SI."""
        from dimtensor.domains.natural import from_natural
        # 1 GeV to Joules
        e_nat = 1.0
        e_si = from_natural(e_nat, "energy", 1e9)
        # Handle scalar case (0-dimensional array)
        value = float(e_si.magnitude()) if e_si.magnitude().ndim == 0 else e_si.magnitude()[0]
        assert pytest.approx(value, rel=1e-6) == 1.602176634e-10

    def test_from_natural_length(self):
        """Test converting natural length to SI."""
        from dimtensor.domains.natural import from_natural
        # 1 GeV^-1 to meters
        l_nat = 1.0
        l_si = from_natural(l_nat, "length", 1e9)
        # Handle scalar case (0-dimensional array)
        value = float(l_si.magnitude()) if l_si.magnitude().ndim == 0 else l_si.magnitude()[0]
        assert pytest.approx(value, rel=1e-2) == 1.973e-16

    def test_from_natural_velocity(self):
        """Test converting natural velocity to SI."""
        from dimtensor.domains.natural import from_natural
        # 0.1 c to m/s
        v_nat = 0.1
        v_si = from_natural(v_nat, "velocity", 1e9)
        # Handle scalar case (0-dimensional array)
        value = float(v_si.magnitude()) if v_si.magnitude().ndim == 0 else v_si.magnitude()[0]
        # c = 299792458 m/s, so 0.1*c = 29979245.8 m/s
        assert pytest.approx(value, rel=1e-5) == 29979245.8


class TestPlanckUnits:
    """Test Planck units."""

    def test_planck_length_dimension(self):
        """Planck length should have length dimension."""
        from dimtensor.domains.planck import planck_length
        assert planck_length.dimension == Dimension(length=1)

    def test_planck_length_scale(self):
        """Planck length should be approximately 1.616e-35 m."""
        from dimtensor.domains.planck import planck_length
        assert pytest.approx(planck_length.scale, rel=1e-2) == 1.616e-35

    def test_planck_mass_dimension(self):
        """Planck mass should have mass dimension."""
        from dimtensor.domains.planck import planck_mass
        assert planck_mass.dimension == Dimension(mass=1)

    def test_planck_mass_scale(self):
        """Planck mass should be approximately 2.176e-8 kg."""
        from dimtensor.domains.planck import planck_mass
        assert pytest.approx(planck_mass.scale, rel=1e-2) == 2.176e-8

    def test_planck_time_dimension(self):
        """Planck time should have time dimension."""
        from dimtensor.domains.planck import planck_time
        assert planck_time.dimension == Dimension(time=1)

    def test_planck_time_scale(self):
        """Planck time should be approximately 5.391e-44 s."""
        from dimtensor.domains.planck import planck_time
        assert pytest.approx(planck_time.scale, rel=1e-2) == 5.391e-44

    def test_planck_energy_dimension(self):
        """Planck energy should have energy dimension."""
        from dimtensor.domains.planck import planck_energy
        assert planck_energy.dimension == Dimension(mass=1, length=2, time=-2)

    def test_planck_energy_scale(self):
        """Planck energy should be approximately 1.956e9 J."""
        from dimtensor.domains.planck import planck_energy
        assert pytest.approx(planck_energy.scale, rel=1e-2) == 1.956e9

    def test_planck_temperature_dimension(self):
        """Planck temperature should have temperature dimension."""
        from dimtensor.domains.planck import planck_temperature
        assert planck_temperature.dimension == Dimension(temperature=1)

    def test_planck_temperature_scale(self):
        """Planck temperature should be approximately 1.417e32 K."""
        from dimtensor.domains.planck import planck_temperature
        assert pytest.approx(planck_temperature.scale, rel=1e-2) == 1.417e32

    def test_planck_charge_dimension(self):
        """Planck charge should have charge dimension."""
        from dimtensor.domains.planck import planck_charge
        assert planck_charge.dimension == Dimension(current=1, time=1)

    def test_planck_charge_scale(self):
        """Planck charge should be approximately 1.876e-18 C."""
        from dimtensor.domains.planck import planck_charge
        assert pytest.approx(planck_charge.scale, rel=1e-2) == 1.876e-18

    def test_planck_force_dimension(self):
        """Planck force should have force dimension."""
        from dimtensor.domains.planck import planck_force
        assert planck_force.dimension == Dimension(mass=1, length=1, time=-2)

    def test_planck_force_scale(self):
        """Planck force should be approximately 1.210e44 N."""
        from dimtensor.domains.planck import planck_force
        assert pytest.approx(planck_force.scale, rel=1e-2) == 1.210e44

    def test_planck_power_dimension(self):
        """Planck power should have power dimension."""
        from dimtensor.domains.planck import planck_power
        assert planck_power.dimension == Dimension(mass=1, length=2, time=-3)

    def test_planck_area_dimension(self):
        """Planck area should have area dimension."""
        from dimtensor.domains.planck import planck_area
        assert planck_area.dimension == Dimension(length=2)

    def test_planck_volume_dimension(self):
        """Planck volume should have volume dimension."""
        from dimtensor.domains.planck import planck_volume
        assert planck_volume.dimension == Dimension(length=3)

    def test_planck_length_to_meter(self):
        """Test Planck length conversion."""
        from dimtensor.domains.planck import planck_length
        from dimtensor.core.units import meter
        l_p = DimArray([1.0], planck_length)
        l_m = l_p.to(meter)
        assert pytest.approx(l_m.magnitude()[0], rel=1e-2) == 1.616e-35


class TestDomainImports:
    """Test that domain modules are importable."""

    def test_import_astronomy(self):
        """Test importing astronomy module."""
        from dimtensor.domains import astronomy
        assert hasattr(astronomy, "parsec")
        assert hasattr(astronomy, "AU")
        assert hasattr(astronomy, "solar_mass")

    def test_import_chemistry(self):
        """Test importing chemistry module."""
        from dimtensor.domains import chemistry
        assert hasattr(chemistry, "molar")
        assert hasattr(chemistry, "dalton")
        assert hasattr(chemistry, "ppm")

    def test_import_engineering(self):
        """Test importing engineering module."""
        from dimtensor.domains import engineering
        assert hasattr(engineering, "MPa")
        assert hasattr(engineering, "hp")
        assert hasattr(engineering, "BTU")

    def test_import_acoustics(self):
        """Test importing acoustics module."""
        from dimtensor.domains import acoustics
        assert hasattr(acoustics, "rayl")
        assert hasattr(acoustics, "uPa")
        assert hasattr(acoustics, "dB")
        assert hasattr(acoustics, "phon")
        assert hasattr(acoustics, "sone")

    def test_import_from_dimtensor(self):
        """Test importing domains from main package."""
        from dimtensor import domains
        assert hasattr(domains, "astronomy")
        assert hasattr(domains, "chemistry")
        assert hasattr(domains, "engineering")
        assert hasattr(domains, "acoustics")

    def test_import_nuclear(self):
        """Test importing nuclear module."""
        from dimtensor.domains import nuclear
        assert hasattr(nuclear, "MeV")
        assert hasattr(nuclear, "barn")
        assert hasattr(nuclear, "becquerel")

    def test_import_geophysics(self):
        """Test importing geophysics module."""
        from dimtensor.domains import geophysics
        assert hasattr(geophysics, "gal")
        assert hasattr(geophysics, "darcy")
        assert hasattr(geophysics, "gamma")

    def test_import_biophysics(self):
        """Test importing biophysics module."""
        from dimtensor.domains import biophysics
        assert hasattr(biophysics, "katal")
        assert hasattr(biophysics, "enzyme_unit")
        assert hasattr(biophysics, "cells_per_mL")

    def test_import_materials(self):
        """Test importing materials module."""
        from dimtensor.domains import materials
        assert hasattr(materials, "strain")
        assert hasattr(materials, "MPa_sqrt_m")
        assert hasattr(materials, "W_per_m_K")

    def test_import_photometry(self):
        """Test importing photometry module."""
        from dimtensor.domains import photometry
        assert hasattr(photometry, "lumen")
        assert hasattr(photometry, "lux")
        assert hasattr(photometry, "nit")

    def test_import_information(self):
        """Test importing information module."""
        from dimtensor.domains import information
        assert hasattr(information, "bit")
        assert hasattr(information, "byte")
        assert hasattr(information, "kilobyte")

    def test_import_imperial(self):
        """Test importing imperial module."""
        from dimtensor.domains import imperial
        assert hasattr(imperial, "inch")
        assert hasattr(imperial, "pound")
        assert hasattr(imperial, "gallon")

    def test_import_natural(self):
        """Test importing natural module."""
        from dimtensor.domains import natural
        assert hasattr(natural, "GeV")
        assert hasattr(natural, "to_natural")
        assert hasattr(natural, "from_natural")

    def test_import_planck(self):
        """Test importing planck module."""
        from dimtensor.domains import planck
        assert hasattr(planck, "planck_length")
        assert hasattr(planck, "planck_mass")
        assert hasattr(planck, "planck_time")

    def test_direct_unit_imports(self):
        """Test importing units directly."""
        from dimtensor.domains.astronomy import parsec, AU, solar_mass
        from dimtensor.domains.chemistry import molar, dalton, ppm
        from dimtensor.domains.engineering import MPa, hp, BTU
        from dimtensor.domains.acoustics import rayl, uPa, dB, phon, sone
        from dimtensor.domains.cgs import dyne, erg, gauss, poise, stokes
        from dimtensor.domains.nuclear import MeV, barn, becquerel
        from dimtensor.domains.geophysics import gal, darcy
        from dimtensor.domains.biophysics import katal, enzyme_unit
        from dimtensor.domains.materials import strain, MPa_sqrt_m
        from dimtensor.domains.photometry import lumen, lux, nit
        from dimtensor.domains.information import bit, byte
        from dimtensor.domains.imperial import inch, foot, pound, gallon, psi, mph
        from dimtensor.domains.natural import GeV
        from dimtensor.domains.planck import planck_length, planck_mass

        # Just verify they're Unit objects
        assert parsec.symbol == "pc"
        assert molar.symbol == "M"
        assert MPa.symbol == "MPa"
        assert rayl.symbol == "rayl"
        assert uPa.symbol == "μPa"
        assert dB.symbol == "dB"
        assert dyne.symbol == "dyn"
        assert erg.symbol == "erg"
        assert gauss.symbol == "G"
        assert MeV.symbol == "MeV"
        assert barn.symbol == "b"
        assert gal.symbol == "Gal"
        assert katal.symbol == "kat"
        assert lumen.symbol == "lm"
        assert inch.symbol == "in"


class TestAcousticsUnits:
    """Test acoustics units."""

    def test_rayl_dimension(self):
        """Rayl should have acoustic impedance dimension (M L^-2 T^-1)."""
        from dimtensor.domains.acoustics import rayl
        assert rayl.dimension == Dimension(mass=1, length=-2, time=-1)

    def test_rayl_scale(self):
        """Rayl should have scale 1.0 (SI base)."""
        from dimtensor.domains.acoustics import rayl
        assert rayl.scale == 1.0

    def test_micropascal_dimension(self):
        """Micropascal should have pressure dimension."""
        from dimtensor.domains.acoustics import micropascal
        assert micropascal.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_micropascal_scale(self):
        """Micropascal should be 1e-6 Pa."""
        from dimtensor.domains.acoustics import uPa
        assert uPa.scale == 1e-6

    def test_kilohertz_dimension(self):
        """kHz should have frequency dimension (T^-1)."""
        from dimtensor.domains.acoustics import kHz
        assert kHz.dimension == Dimension(time=-1)

    def test_kilohertz_scale(self):
        """kHz should be 1000 Hz."""
        from dimtensor.domains.acoustics import kHz, Hz
        assert kHz.scale == 1000.0
        assert kHz.scale / Hz.scale == 1000

    def test_decibel_dimensionless(self):
        """Decibel should be dimensionless."""
        from dimtensor.domains.acoustics import dB
        assert dB.dimension == DIMENSIONLESS

    def test_phon_dimensionless(self):
        """Phon should be dimensionless."""
        from dimtensor.domains.acoustics import phon
        assert phon.dimension == DIMENSIONLESS

    def test_sone_dimensionless(self):
        """Sone should be dimensionless."""
        from dimtensor.domains.acoustics import sone
        assert sone.dimension == DIMENSIONLESS

    def test_frequency_conversion(self):
        """Test frequency unit conversions."""
        from dimtensor.domains.acoustics import Hz, kHz
        freq_khz = DimArray([1.0], kHz)
        freq_hz = freq_khz.to(Hz)
        assert pytest.approx(freq_hz.magnitude()[0]) == 1000.0

    def test_pressure_conversion(self):
        """Test pressure unit conversions."""
        from dimtensor.domains.acoustics import Pa, uPa
        # Reference pressure for SPL: 20 uPa
        ref_pressure = DimArray([20.0], uPa)
        ref_pressure_pa = ref_pressure.to(Pa)
        assert pytest.approx(ref_pressure_pa.magnitude()[0]) == 20e-6

    def test_acoustic_impedance_of_air(self):
        """Test acoustic impedance calculation for air."""
        from dimtensor.domains.acoustics import rayl, Pa
        from dimtensor.core.units import s, m
        # Air at STP: Z = 415 rayl = 415 Pa·s/m
        impedance = DimArray([415.0], rayl)

        # Verify dimension
        assert impedance.dimension == Dimension(mass=1, length=-2, time=-1)

        # Test that rayl can be constructed from Pa·s/m
        unit_composite = Pa * s / m
        assert unit_composite.dimension == rayl.dimension

    def test_pressure_from_impedance_velocity(self):
        """Test pressure = impedance × velocity relationship."""
        from dimtensor.domains.acoustics import rayl
        from dimtensor.core.units import Pa, m, s

        # Acoustic impedance of air
        Z = DimArray([415.0], rayl)
        # Particle velocity
        v = DimArray([0.001], m / s)

        # Pressure = impedance × velocity
        p = Z * v

        # Result should have pressure dimension
        assert p.dimension == Dimension(mass=1, length=-1, time=-2)

        # Can convert to Pascal
        p_pa = p.to(Pa)
        assert pytest.approx(p_pa.magnitude()[0]) == 0.415

    def test_sound_speed_calculation(self):
        """Test calculating sound speed from bulk modulus and density."""
        from dimtensor.domains.acoustics import rayl
        from dimtensor.core.units import Pa, kg, m, s

        # For air at STP:
        # density ≈ 1.2 kg/m³
        # bulk modulus ≈ 1.42e5 Pa
        # speed of sound c = sqrt(K/ρ) ≈ 343 m/s
        # acoustic impedance Z = ρ·c ≈ 411-415 rayl

        density = DimArray([1.2], kg / (m**3))
        speed = DimArray([343.0], m / s)

        impedance = density * speed

        # Should have rayl dimension
        assert impedance.dimension == rayl.dimension

        # Convert to rayl
        impedance_rayl = impedance.to(rayl)
        assert 400 < impedance_rayl.magnitude()[0] < 420

    def test_micropascal_reference(self):
        """Test that 20 uPa is the standard SPL reference."""
        from dimtensor.domains.acoustics import uPa, Pa
        # Standard reference for sound pressure level in air
        ref = DimArray([20.0], uPa)
        ref_si = ref.to(Pa)
        assert pytest.approx(ref_si.magnitude()[0]) == 2e-5

    def test_acoustic_power_units(self):
        """Test acoustic power units."""
        from dimtensor.domains.acoustics import W
        from dimtensor.core.units import watt
        # W should be an alias for watt
        assert W.dimension == watt.dimension
        assert W.scale == watt.scale


class TestCrossDomainCalculations:
    """Test calculations using units from multiple domains."""

    def test_stellar_density(self):
        """Calculate stellar density using astronomy units."""
        from dimtensor.domains.astronomy import solar_mass, solar_radius
        from dimtensor.core.units import kg, m
        import math

        mass = DimArray([1.0], solar_mass)
        radius = DimArray([1.0], solar_radius)

        # Convert to SI units first
        mass_kg = mass.to(kg)
        radius_m = radius.to(m)

        # Density = M / (4/3 * pi * R^3)
        volume = (4/3) * math.pi * (radius_m ** 3)
        density = mass_kg / volume

        # Sun's density is about 1.4 g/cm^3 = 1400 kg/m^3
        assert density.dimension == Dimension(mass=1, length=-3)
        # Check order of magnitude
        assert 1000 < density.magnitude()[0] < 2000

    def test_molecular_energy_conversion(self):
        """Test converting molecular energies."""
        from dimtensor.domains.chemistry import hartree
        from dimtensor.core.units import eV

        # 1 Hartree ~ 27.2 eV
        energy_ha = DimArray([1.0], hartree)
        energy_ev = energy_ha.to(eV)
        assert pytest.approx(energy_ev.magnitude()[0], rel=1e-2) == 27.2

    def test_engineering_energy_conversion(self):
        """Test energy unit conversions across domains."""
        from dimtensor.domains.engineering import BTU, kWh
        from dimtensor.core.units import J

        # Convert 1 kWh to BTU
        energy_kwh = DimArray([1.0], kWh)
        energy_btu = energy_kwh.to(BTU)
        # 1 kWh ~ 3412 BTU
        assert pytest.approx(energy_btu.magnitude()[0], rel=1e-2) == 3412


class TestCGSUnits:
    """Test CGS (Centimeter-Gram-Second) units."""

    def test_dyne_dimension(self):
        """Dyne should have force dimension."""
        from dimtensor.domains.cgs import dyne
        assert dyne.dimension == Dimension(mass=1, length=1, time=-2)

    def test_dyne_scale(self):
        """1 dyne = 1e-5 N."""
        from dimtensor.domains.cgs import dyne
        assert dyne.scale == 1e-5

    def test_erg_dimension(self):
        """Erg should have energy dimension."""
        from dimtensor.domains.cgs import erg
        assert erg.dimension == Dimension(mass=1, length=2, time=-2)

    def test_erg_scale(self):
        """1 erg = 1e-7 J."""
        from dimtensor.domains.cgs import erg
        assert erg.scale == 1e-7

    def test_barye_dimension(self):
        """Barye should have pressure dimension."""
        from dimtensor.domains.cgs import barye
        assert barye.dimension == Dimension(mass=1, length=-1, time=-2)

    def test_barye_scale(self):
        """1 Ba = 0.1 Pa."""
        from dimtensor.domains.cgs import barye
        assert barye.scale == 0.1

    def test_poise_dimension(self):
        """Poise should have dynamic viscosity dimension."""
        from dimtensor.domains.cgs import poise
        assert poise.dimension == Dimension(mass=1, length=-1, time=-1)

    def test_poise_scale(self):
        """1 P = 0.1 Pa·s."""
        from dimtensor.domains.cgs import poise
        assert poise.scale == 0.1

    def test_centipoise_scale(self):
        """1 cP = 0.001 Pa·s (water viscosity)."""
        from dimtensor.domains.cgs import centipoise
        assert centipoise.scale == 0.001

    def test_stokes_dimension(self):
        """Stokes should have kinematic viscosity dimension."""
        from dimtensor.domains.cgs import stokes
        assert stokes.dimension == Dimension(length=2, time=-1)

    def test_stokes_scale(self):
        """1 St = 1e-4 m²/s."""
        from dimtensor.domains.cgs import stokes
        assert stokes.scale == 1e-4

    def test_gauss_dimension(self):
        """Gauss should have magnetic flux density dimension."""
        from dimtensor.domains.cgs import gauss
        assert gauss.dimension == Dimension(mass=1, time=-2, current=-1)

    def test_gauss_scale(self):
        """1 G = 1e-4 T."""
        from dimtensor.domains.cgs import gauss
        assert gauss.scale == 1e-4

    def test_maxwell_dimension(self):
        """Maxwell should have magnetic flux dimension."""
        from dimtensor.domains.cgs import maxwell
        assert maxwell.dimension == Dimension(mass=1, length=2, time=-2, current=-1)

    def test_maxwell_scale(self):
        """1 Mx = 1e-8 Wb."""
        from dimtensor.domains.cgs import maxwell
        assert maxwell.scale == 1e-8

    def test_oersted_dimension(self):
        """Oersted should have magnetic field strength dimension."""
        from dimtensor.domains.cgs import oersted
        assert oersted.dimension == Dimension(current=1, length=-1)

    def test_oersted_scale(self):
        """1 Oe ≈ 79.577 A/m."""
        from dimtensor.domains.cgs import oersted
        assert pytest.approx(oersted.scale, rel=1e-3) == 79.577

    def test_statcoulomb_dimension(self):
        """Statcoulomb should have charge dimension."""
        from dimtensor.domains.cgs import statcoulomb
        assert statcoulomb.dimension == Dimension(current=1, time=1)

    def test_statcoulomb_scale(self):
        """1 statC ≈ 3.336e-10 C."""
        from dimtensor.domains.cgs import statcoulomb
        assert pytest.approx(statcoulomb.scale, rel=1e-3) == 3.336e-10

    def test_statampere_dimension(self):
        """Statampere should have current dimension."""
        from dimtensor.domains.cgs import statampere
        assert statampere.dimension == Dimension(current=1)

    def test_statvolt_dimension(self):
        """Statvolt should have voltage dimension."""
        from dimtensor.domains.cgs import statvolt
        assert statvolt.dimension == Dimension(mass=1, length=2, time=-3, current=-1)

    def test_statvolt_scale(self):
        """1 statV ≈ 299.79 V."""
        from dimtensor.domains.cgs import statvolt
        assert pytest.approx(statvolt.scale, rel=1e-3) == 299.79

    def test_gal_dimension(self):
        """Gal should have acceleration dimension."""
        from dimtensor.domains.cgs import gal
        assert gal.dimension == Dimension(length=1, time=-2)

    def test_gal_scale(self):
        """1 Gal = 0.01 m/s²."""
        from dimtensor.domains.cgs import gal
        assert gal.scale == 0.01

    def test_dyne_to_newton_conversion(self):
        """Test dyne to newton conversion."""
        from dimtensor.domains.cgs import dyne
        from dimtensor.core.units import newton
        force_cgs = DimArray([1e5], dyne)
        force_si = force_cgs.to(newton)
        assert pytest.approx(force_si.magnitude()[0]) == 1.0

    def test_erg_to_joule_conversion(self):
        """Test erg to joule conversion."""
        from dimtensor.domains.cgs import erg
        from dimtensor.core.units import joule
        energy_cgs = DimArray([1e7], erg)
        energy_si = energy_cgs.to(joule)
        assert pytest.approx(energy_si.magnitude()[0]) == 1.0

    def test_gauss_to_tesla_conversion(self):
        """Test gauss to tesla conversion."""
        from dimtensor.domains.cgs import gauss
        from dimtensor.core.units import tesla
        field_cgs = DimArray([1e4], gauss)
        field_si = field_cgs.to(tesla)
        assert pytest.approx(field_si.magnitude()[0]) == 1.0

    def test_earth_magnetic_field(self):
        """Test Earth's magnetic field in gauss."""
        from dimtensor.domains.cgs import gauss
        from dimtensor.core.units import tesla
        # Earth's field ~ 0.5 G = 50 µT
        earth_field_g = DimArray([0.5], gauss)
        earth_field_t = earth_field_g.to(tesla)
        assert pytest.approx(earth_field_t.magnitude()[0], rel=1e-2) == 5e-5

    def test_water_viscosity_centipoise(self):
        """Test water viscosity in centipoise."""
        from dimtensor.domains.cgs import centipoise
        from dimtensor.core.units import pascal, second
        # Water at 20°C: ~ 1 cP
        viscosity_cp = DimArray([1.0], centipoise)
        # Convert to Pa·s
        pa_s = pascal * second
        viscosity_si = viscosity_cp.to(pa_s)
        assert pytest.approx(viscosity_si.magnitude()[0]) == 0.001
