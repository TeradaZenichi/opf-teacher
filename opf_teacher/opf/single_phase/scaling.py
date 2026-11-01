"""Solver scaling; public variables and case bases retain their original units."""
import pyomo.environ as pyo


def scale_device(model, device, nominals):
    for component, nominal in nominals:
        if component is not None:
            for time in model.T:
                key = device.id, time
                if key in component:
                    model.scaling_factor[component[key]] = 1.0 / nominal


def scale_model(model, case):
    demand = max(sum(abs(bus.p_load_kw.iloc[t]) + abs(bus.q_load_kw.iloc[t]) for bus in case.buses.values()) for t in case.periods)
    solar = max(sum(float(device.avail_kw.iloc[t]) for device in case.pv) for t in case.periods)
    power = case.base.s_base_kva / max(demand, solar, 1.0)
    if model.component("scaling_factor") is None:
        model.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    scale = model.scaling_factor
    scale.clear()
    for variable in (model.P, model.Q, model.qgrid):
        scale[variable] = power
    if hasattr(model, "pgrid"):
        scale[model.pgrid] = power
    else:
        scale[model.pimp] = power
        scale[model.pexp] = power
    scale[model.l] = power ** 2
    scale[model.balance_p] = power
    scale[model.balance_q] = power
    scale[model.socp] = power ** 2
    if hasattr(model, "cost_rate"):
        for time in model.T:
            price = max(abs(float(case.price.iloc[time])), abs(float(case.price.iloc[time]) * case.grid.feed_in_ratio)) or 1.0
            scale[model.cost_rate[time]] = power / price
            scale[model.buy_cost[time]] = power / price
            scale[model.sell_cost[time]] = power / price

    for device in case.bess:
        rating = case.base.pu_power(device.s_max_kva) or 1.0
        capacity = case.base.pu_energy(device.e_cap_kwh)
        loss = case.base.pu_power(device.q_loss_rated_kw) or 1.0
        charge = case.base.pu_power(device.p_charge_max_kw) or 1.0
        discharge = case.base.pu_power(device.p_discharge_max_kw) or 1.0
        scale_device(model, device, (
            (model.pch, charge), (model.pdis, discharge), (model.qbess, rating),
            (model.soc, capacity), (model.soc_balance, capacity),
            (model.bess_capability, rating ** 2), (model.pbess_loss, loss),
            (model.bess_q_loss, rating ** 2 * loss),
        ))
        if device.id in model.soc_terminal:
            scale[model.soc_terminal[device.id]] = 1.0 / capacity

    for device in case.pv:
        rating = case.base.pu_power(device.s_max_kva) or 1.0
        loss = case.base.pu_power(device.q_loss_rated_kw) or 1.0
        generation = case.base.pu_power(device.p_max_kw) or 1.0
        scale_device(model, device, (
            (model.ppv, generation), (model.qpv, rating), (model.ppv_loss, loss),
            (model.pv_capability, rating ** 2), (model.pv_available_power, generation),
            (model.pv_q_loss, rating ** 2 * loss), (model.pv_no_night_q, rating),
            (model.component("pv_fixed_pf"), rating),
        ))
    return pyo.TransformationFactory("core.scale_model").create_using(model)
