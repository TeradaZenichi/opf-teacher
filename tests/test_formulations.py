import unittest

from opf import Formulation
from opf.common import validate_case_config
from opf.formulations import parse_formulation
from opf.single_phase import ActivePowerModel, DistFlowModel
from teacher import Teacher


class FormulationSelectionTest(unittest.TestCase):
    def test_model_classes_keep_separate_builds_independent(self):
        for model_type in (ActivePowerModel, DistFlowModel):
            with self.subTest(model=model_type.__name__):
                case = Teacher("examples/case5").case
                builder = model_type(case)
                first = builder.build()
                original_limit = first.pch["b1", 0].ub
                case.bess[0].p_charge_max_kw = 10.0
                second = builder.build()
                self.assertIsNot(first, second)
                self.assertEqual(first.pch["b1", 0].ub, original_limit)
                self.assertAlmostEqual(second.pch["b1", 0].ub, 0.01)
                self.assertEqual(first.nconstraints(), second.nconstraints())

    def test_default_and_explicit_single_phase_formulations(self):
        full = Teacher("examples/case5")
        active = Teacher("examples/case5", formulation="single_phase_active")

        self.assertIs(full.formulation, Formulation.SINGLE_PHASE_SOCP)
        self.assertIs(active.formulation, Formulation.SINGLE_PHASE_ACTIVE)

    def test_explicit_formulation_names_and_enum_values(self):
        for formulation in Formulation:
            with self.subTest(formulation=formulation):
                self.assertIs(parse_formulation(formulation.value), formulation)
                self.assertIs(parse_formulation(formulation), formulation)

    def test_configuration_file_is_a_case_source_and_selects_formulation(self):
        teacher = Teacher("examples/case5_unbalanced/config.json")

        self.assertIs(teacher.formulation, Formulation.THREE_PHASE_IVR)
        self.assertEqual(teacher.case.name, "case5_unbalanced")
        self.assertEqual(teacher.case.config_path.name, "config.json")

    def test_only_official_formulation_names_are_accepted(self):
        for name in ("balanced_active", "balanced_socp", "distflow_socp", "unbalanced_ac_ivr"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Unknown OPF formulation"):
                parse_formulation(name)

    def test_three_phase_boundary_rejects_legacy_case(self):
        with self.assertRaisesRegex(ValueError, "three-phase demand column"):
            Teacher("examples/case5", formulation="three_phase_ivr")

    def test_unknown_formulation_lists_supported_values(self):
        with self.assertRaisesRegex(ValueError, "Unknown OPF formulation"):
            parse_formulation("unknown")

    def test_unknown_case_schema_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "schema_version"):
            validate_case_config({"schema_version": 2})

    def test_case_object_must_match_the_selected_formulation(self):
        single_phase = Teacher("examples/case5").case
        three_phase = Teacher("examples/case5_unbalanced").case
        with self.assertRaisesRegex(TypeError, "three_phase.Case"):
            Teacher(single_phase, formulation="three_phase_ivr")
        with self.assertRaisesRegex(TypeError, "Single-phase"):
            Teacher(three_phase, formulation="single_phase_active")


if __name__ == "__main__":
    unittest.main()
