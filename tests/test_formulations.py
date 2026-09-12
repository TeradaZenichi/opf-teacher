import unittest

from opf import Formulation
from opf.case_source import validate_case_config
from opf.active_model import build_active_power_model as legacy_active_builder
from opf.formulations import parse_formulation
from opf.model import build_model as legacy_socp_builder
from opf.single_phase.active_model import build_active_power_model
from opf.single_phase.distflow_socp import build_model as build_single_phase_socp_model
from teacher import Teacher


class FormulationSelectionTest(unittest.TestCase):
    def test_legacy_imports_point_to_single_phase_implementations(self):
        self.assertIs(legacy_active_builder, build_active_power_model)
        self.assertIs(legacy_socp_builder, build_single_phase_socp_model)

    def test_legacy_teacher_flags_select_single_phase_formulations(self):
        full = Teacher("examples/case5")
        active = Teacher("examples/case5", active_power_only=True)

        self.assertIs(full.formulation, Formulation.SINGLE_PHASE_SOCP)
        self.assertIs(active.formulation, Formulation.SINGLE_PHASE_ACTIVE)
        self.assertFalse(full.active_power_only)
        self.assertTrue(active.active_power_only)

    def test_explicit_formulation_and_aliases(self):
        self.assertIs(
            Teacher("examples/case5", formulation="single_phase_active").formulation,
            Formulation.SINGLE_PHASE_ACTIVE,
        )
        self.assertIs(
            parse_formulation("balanced_socp"),
            Formulation.SINGLE_PHASE_SOCP,
        )
        self.assertIs(
            parse_formulation("unbalanced_ac_ivr"),
            Formulation.THREE_PHASE_IVR,
        )

    def test_configuration_file_is_a_case_source_and_selects_formulation(self):
        teacher = Teacher("examples/case5_unbalanced/config.json")

        self.assertIs(teacher.formulation, Formulation.THREE_PHASE_IVR)
        self.assertEqual(teacher.case.name, "case5_unbalanced")
        self.assertEqual(teacher.case.config_path.name, "config.json")

    def test_conflicting_legacy_flag_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "conflicts"):
            Teacher(
                "examples/case5",
                active_power_only=True,
                formulation="single_phase_socp",
            )

    def test_three_phase_boundary_rejects_legacy_case(self):
        with self.assertRaisesRegex(ValueError, "three-phase demand column"):
            Teacher("examples/case5", formulation="three_phase_ivr")

    def test_unknown_formulation_lists_supported_values(self):
        with self.assertRaisesRegex(ValueError, "Unknown OPF formulation"):
            parse_formulation("unknown")

    def test_unknown_case_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_case_config({"schema_version": 2})


if __name__ == "__main__":
    unittest.main()
