"""Pre-action observations and commands for one interval, in physical units."""


class BusState:
    def __init__(self, v_before_pu, p_load_kw, q_load_kvar):
        self.v_before_pu = v_before_pu
        self.p_load_kw, self.q_load_kvar = p_load_kw, q_load_kvar


class BessState:
    def __init__(self, soc_before_frac, previous_p_kw, previous_q_kvar):
        self.soc_before_frac = soc_before_frac
        self.previous_p_kw, self.previous_q_kvar = previous_p_kw, previous_q_kvar


class PvState:
    def __init__(self, available_kw, previous_p_kw, previous_q_kvar):
        self.available_kw = available_kw
        self.previous_p_kw, self.previous_q_kvar = previous_p_kw, previous_q_kvar


class GridState:
    def __init__(self, buy_price_per_kwh, sell_price_per_kwh):
        self.buy_price_per_kwh = buy_price_per_kwh
        self.sell_price_per_kwh = sell_price_per_kwh


class BessAction:
    """Positive P charges the battery; positive Q injects reactive power."""

    def __init__(self, p_net_kw, q_injection_kvar):
        self.p_net_kw, self.q_injection_kvar = p_net_kw, q_injection_kvar


class PvAction:
    """Requested generation and Q injection; the environment handles night losses."""

    def __init__(self, generation_kw, q_injection_kvar):
        self.generation_kw, self.q_injection_kvar = generation_kw, q_injection_kvar


def state_values(state):
    if state is None:
        raise ValueError("Missing pre-action observation; call Teacher.observe first")
    return dict(vars(state))


def local_state_action(device, bus):
    if bus.id != device.bus:
        raise ValueError(f"Device {device.id!r} is not connected to bus {bus.id!r}")
    if device.action is None:
        raise ValueError("No teacher action; solve the observed case first")
    x = {"bus": state_values(bus.state), "device": state_values(device.state)}
    return x, dict(vars(device.action))
