"""Translate continuous quadratic cones to Clarabel's sparse conic format."""
import math

import clarabel
import numpy as np
import pyomo.environ as pyo
from pyomo.repn import generate_standard_repn
from scipy import sparse


class ConicModel:
    def __init__(self, model):
        if any(model.component_data_objects(pyo.SOSConstraint, active=True)):
            raise ValueError("Clarabel requires a continuous model; use Gurobi for SOS constraints")
        self.variables = list(model.component_data_objects(pyo.Var))
        if any(variable.is_integer() for variable in self.variables):
            raise ValueError("Clarabel does not support integer variables")
        self.indices = {id(variable): index for index, variable in enumerate(self.variables)}
        self.rows, self.offsets, self.cones = [], [], []
        objective = next(model.component_data_objects(pyo.Objective, active=True))
        representation = generate_standard_repn(objective.expr)
        if representation.quadratic_vars or representation.nonlinear_expr is not None or objective.sense != pyo.minimize:
            raise ValueError("The conic OPF requires a linear minimization objective")
        self.cost = np.zeros(len(self.variables))
        for variable, coefficient in zip(representation.linear_vars, representation.linear_coefs):
            self.cost[self.indices[id(variable)]] = float(coefficient)
        for index, variable in enumerate(self.variables):
            bounds = (pyo.value(variable), pyo.value(variable)) if variable.fixed else variable.bounds
            self.linear({index: 1.0}, 0.0, bounds)
        for constraint in model.component_data_objects(pyo.Constraint, active=True):
            representation = generate_standard_repn(constraint.body)
            if representation.nonlinear_expr is not None:
                raise ValueError(f"Unsupported nonlinear constraint: {constraint.name}")
            coefficients = {self.indices[id(variable)]: float(coefficient)
                            for variable, coefficient in zip(representation.linear_vars, representation.linear_coefs)}
            constant = float(representation.constant or 0.0)
            if representation.quadratic_vars:
                if constraint.lower is not None or constraint.upper is None:
                    raise ValueError(f"Unsupported quadratic bound: {constraint.name}")
                self.quadratic(representation, float(pyo.value(constraint.upper)))
            else:
                self.linear(coefficients, constant, (constraint.lb, constraint.ub))

    def linear(self, coefficients, constant, bounds):
        lower, upper = bounds
        if lower is not None and lower == upper:
            self.add([coefficients], [constant - lower], clarabel.ZeroConeT(1))
            return
        if lower is not None:
            self.add([coefficients], [constant - lower], clarabel.NonnegativeConeT(1))
        if upper is not None:
            self.add([{key: -value for key, value in coefficients.items()}], [upper - constant], clarabel.NonnegativeConeT(1))

    def add(self, rows, offsets, cone):
        self.rows.extend(rows)
        self.offsets.extend(float(value) for value in offsets)
        self.cones.append(cone)

    def quadratic(self, representation, upper):
        terms = [(self.indices[id(left)], self.indices[id(right)], float(coefficient))
                 for (left, right), coefficient in zip(representation.quadratic_vars, representation.quadratic_coefs)]
        indices = sorted({index for left, right, _ in terms for index in (left, right)})
        positions = {index: position for position, index in enumerate(indices)}
        matrix = np.zeros((len(indices), len(indices)))
        for left, right, coefficient in terms:
            if left == right:
                matrix[positions[left], positions[right]] += coefficient
            else:
                matrix[positions[left], positions[right]] += coefficient / 2
                matrix[positions[right], positions[left]] += coefficient / 2
        eigenvalues, vectors = np.linalg.eigh(matrix)
        cutoff = 1e-12 * max(1.0, abs(eigenvalues).max())
        linear = {self.indices[id(variable)]: -float(coefficient)
                  for variable, coefficient in zip(representation.linear_vars, representation.linear_coefs)}
        offset = upper - float(representation.constant or 0.0)
        if eigenvalues.min() >= -cutoff:
            rows = [linear]
            for value, vector in zip(eigenvalues, vectors.T):
                if value > cutoff:
                    rows.append({index: 2 * math.sqrt(value) * float(coefficient) for index, coefficient in zip(indices, vector)})
            rows.append(linear.copy())
            self.add(rows, [offset + 1.0] + [0.0] * (len(rows) - 2) + [offset - 1.0], clarabel.SecondOrderConeT(len(rows)))
            return
        cross = [(left, right, -coefficient) for left, right, coefficient in terms if left != right and coefficient < 0]
        diagonal = [(left, coefficient) for left, right, coefficient in terms if left == right and coefficient >= 0]
        if len(cross) != 1 or len(diagonal) + 1 != len(terms) or linear or offset != 0.0:
            raise ValueError("Unsupported nonconvex quadratic constraint")
        left, right, coefficient = cross[0]
        if any(self.variables[index].lb is None or self.variables[index].lb < 0 for index in (left, right)):
            raise ValueError("Rotated cones require nonnegative factors")
        rows = [{left: 1.0, right: 1.0}]
        rows.extend({index: 2 * math.sqrt(value / coefficient)} for index, value in diagonal)
        rows.append({left: 1.0, right: -1.0})
        self.add(rows, [0.0] * len(rows), clarabel.SecondOrderConeT(len(rows)))

    def solve(self, config):
        row_indices, column_indices, values = [], [], []
        for row_index, row in enumerate(self.rows):
            for column_index, value in row.items():
                if value != 0:
                    row_indices.append(row_index)
                    column_indices.append(column_index)
                    values.append(-value)
        size = len(self.variables)
        matrix = sparse.csc_matrix((values, (row_indices, column_indices)), shape=(len(self.rows), size))
        settings = clarabel.DefaultSettings()
        settings.verbose = config.tee
        for name, value in config.solver_options.items():
            if not hasattr(settings, name):
                raise ValueError(f"Unknown Clarabel option: {name}")
            setattr(settings, name, value)
        solver = clarabel.DefaultSolver(sparse.csc_matrix((size, size)), self.cost, matrix, np.array(self.offsets), self.cones, settings)
        result = solver.solve()
        if result.status != clarabel.SolverStatus.Solved:
            raise RuntimeError(f"Clarabel did not solve to optimality: {result.status}")
        for variable, value in zip(self.variables, result.x):
            variable.set_value(float(value), skip_validation=True)
        return {"name": "clarabel", "status": "optimal", "iterations": result.iterations,
                "primal_residual": result.r_prim, "dual_residual": result.r_dual,
                "absolute_gap": abs(result.obj_val - result.obj_val_dual)}
