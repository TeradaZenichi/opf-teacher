import math
from importlib.util import find_spec
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from opf_teacher import PairConfig, Teacher, collect_pairs
from opf_teacher.opf.single_phase.distflow_socp import build_model
if find_spec("opendss_environment") is not None:
    from opendss_environment import EnvironmentConfig, MicrogridEnv
else:
    MicrogridEnv = None


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class TariffTest(unittest.TestCase):
    @unittest.skipIf(MicrogridEnv is None, "Install the environment extra to test execution")
    def test_independent_tariff_is_preserved_in_load_horizon_pairs_and_execution(self):
        for name, formulation in (("single_phase_case", "single_phase_socp"),
                                  ("single_phase_case", "single_phase_active"),
                                  ("three_phase_transformer_case", "three_phase_ivr")):
            with self.subTest(formulation=formulation), TemporaryDirectory() as directory:
                case = Path(directory) / "case"
                shutil.copytree(FIXTURES / name, case)
                (case / "devices.json").write_text("{}", encoding="utf-8")
                price_path = case / "price.csv"
                prices = pd.read_csv(price_path).iloc[:2].copy()
                prices["sell_price_per_kwh"] = [0.03, 0.07]
                prices.to_csv(price_path, index=False)
                demand_path = case / "demand.csv"
                demand = pd.read_csv(demand_path).iloc[:2].copy()
                for column in demand:
                    if column.startswith("P"):
                        demand[column] = -demand[column].abs() * 0.1
                    elif column.startswith("Q"):
                        demand[column] = 0.0
                demand.to_csv(demand_path, index=False)
                teacher = Teacher(case, formulation=formulation)
                if not np.allclose(teacher.case.horizon(1).sell_prices, [0.07]):
                    raise ValueError("Horizon must preserve the independent tariff")
                solved = teacher.solve()
                x, y = collect_pairs(case, PairConfig(formulation=formulation, window=None))
                env = MicrogridEnv(EnvironmentConfig(case))
                observation, _ = env.reset(seed=42)
                for step, action in enumerate(y):
                    if not math.isclose(x[step]["grid"]["sell_price_per_kwh"], prices["sell_price_per_kwh"].iloc[step]):
                        raise ValueError("Pair replay must advance the sell tariff with the timestamp")
                    snapshot = Teacher(teacher.case.horizon(step), formulation=formulation)
                    snapshot.observe_dict(observation)
                    observation, _, _, _, info = env.step(action)
                    if not math.isclose(solved.grid.result.cost.iloc[step], info["cost"], abs_tol=1e-5):
                        raise ValueError("Teacher and environment export costs disagree")
                if solved.summary.energy_cost >= 0:
                    raise ValueError("The export test must produce a credit")

    def test_sell_price_above_buy_selects_exclusive_grid_exchange(self):
        teacher = Teacher(FIXTURES / "single_phase_case", formulation="single_phase_socp")
        teacher.case.sell_price = teacher.case.price + 0.1
        model = build_model(teacher.case)
        if model.component("no_roundtrip") is None or model.no_roundtrip[0].level != 1:
            raise ValueError("Export premium requires exclusive grid exchange")

    def test_absent_sell_column_preserves_ratio_fallback(self):
        for name in ("single_phase_case", "three_phase_transformer_case"):
            case = Teacher(FIXTURES / name).case
            case.grid.feed_in_ratio = 0.23
            if not np.allclose(case.sell_prices, case.price * 0.23):
                raise ValueError("Cases without a sell column must retain ratio pricing")
