"""Importação da rede OpenDSS para o modelo equilibrado."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import re
from typing import Any, Mapping


@dataclass(frozen=True)
class OpenDSSBusData:
    id: int
    name: str
    phases: tuple[int, ...]
    kv_base_ln: float


@dataclass(frozen=True)
class OpenDSSBranchData:
    name: str
    from_bus: int
    to_bus: int
    phases: tuple[int, ...]
    r_ohm: float
    x_ohm: float
    s_max_kva: float
    norm_amps: float
    length: float
    units: int | str
    r_matrix_ohm: tuple[tuple[float, ...], ...]
    x_matrix_ohm: tuple[tuple[float, ...], ...]
    element_type: str = "line"
    tap_ratio: float = 1.0
    r_pu_on_rating: float | None = None
    x_pu_on_rating: float | None = None
    impedance_base_kva: float | None = None
    connections: tuple[str, ...] = ()


@dataclass(frozen=True)
class OpenDSSNetworkData:
    buses: tuple[OpenDSSBusData, ...]
    branches: tuple[OpenDSSBranchData, ...]
    root_bus: int
    bus_name_to_id: dict[str, int]


def load_opendss_network(
    master_path: str | Path,
    *,
    slack_bus: str | int | None = None,
    bus_ids: Mapping[str, int] | None = None,
    fallback_v_base_kv: float | None = None,
    dss: Any | None = None,
) -> OpenDSSNetworkData:
    """Compila o Master.dss e retorna a topologia radial."""
    master = Path(master_path).expanduser().resolve()
    if not master.is_file():
        raise FileNotFoundError(f"OpenDSS master file not found: {master}")

    dss = dss if dss is not None else _new_dss()
    caller_directory = Path.cwd()
    try:
        dss.text("clear")
        dss.text(f'compile "{master}"')
        _raise_dss_error(dss, master)
        dss_interface = getattr(dss, "dssinterface", None)
        if dss_interface is not None and int(getattr(dss_interface, "num_circuits", 1)) == 0:
            raise RuntimeError(f"OpenDSS did not create a circuit while compiling {master}")
        # Alguns arquivos não executam CalcVoltageBases no final.
        dss.text("calcvoltagebases")
        _raise_dss_error(dss, master)
    finally:
        os.chdir(caller_directory)

    names = [_bus_name(n) for n in dss.circuit.buses_names]
    if not names:
        raise ValueError(f"OpenDSS circuit in {master} contains no buses")
    if len(set(names)) != len(names):
        raise ValueError("OpenDSS bus names are not unique after normalization")

    name_to_id = _assign_bus_ids(names, bus_ids)
    buses = tuple(_read_bus(dss, name, name_to_id[name]) for name in names)
    bus_by_id = {bus.id: bus for bus in buses}

    lines = _read_lines(
        dss,
        name_to_id,
        bus_by_id,
        fallback_v_base_kv=fallback_v_base_kv,
    )
    transformers = _read_transformers(dss, name_to_id)
    raw_branches = lines + transformers
    if not raw_branches:
        raise ValueError(f"OpenDSS circuit in {master} contains no supported branches")

    root = _resolve_root(slack_bus, name_to_id, raw_branches)
    branches = _orient_radial(tuple(bus_by_id), raw_branches, root)
    return OpenDSSNetworkData(
        buses=buses,
        branches=branches,
        root_bus=root,
        bus_name_to_id=name_to_id,
    )


def resolve_bus_id(value: str | int, bus_name_to_id: Mapping[str, int]) -> int:
    """Resolve nome ou identificador numérico de uma barra."""
    if isinstance(value, bool):
        raise ValueError(f"Invalid bus reference: {value!r}")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    name = _bus_name(text.lstrip("_"))
    if name in bus_name_to_id:
        return bus_name_to_id[name]
    try:
        return int(text)
    except ValueError as exc:
        known = ", ".join(sorted(bus_name_to_id))
        raise ValueError(f"Unknown OpenDSS bus {value!r}; known buses: {known}") from exc


def _new_dss():
    try:
        from opendssdirect import dss as opendssdirect
    except ImportError as exc:
        raise RuntimeError(
            "OpenDSS input requires 'opendssdirect.py'. "
            "Install the project dependencies with: pip install -r requirements.txt"
        ) from exc
    # Mantém o circuito ativo isolado de outras instâncias no mesmo processo.
    return _OpenDSSDirectAdapter(opendssdirect.NewContext())


class _OpenDSSDirectAdapter:
    """Interface mínima usada pelo importador."""

    def __init__(self, dss):
        self.text = lambda command: dss.Text.Command(command)
        self.circuit = _ODDCircuit(dss)
        self.bus = _ODDBus(dss)
        self.lines = _ODDLines(dss)
        self.transformers = _ODDTransformers(dss)
        self.errorinterface = _ODDError(dss)
        self.dssinterface = _ODDInterface(dss)


class _ODDCircuit:
    def __init__(self, dss):
        self._dss = dss

    @property
    def buses_names(self):
        return self._dss.Circuit.AllBusNames()

    def set_active_bus(self, name):
        return self._dss.Circuit.SetActiveBus(name)


class _ODDBus:
    def __init__(self, dss):
        self._dss = dss

    @property
    def nodes(self):
        return self._dss.Bus.Nodes()

    @property
    def kv_base(self):
        return self._dss.Bus.kVBase()


class _ODDLines:
    def __init__(self, dss):
        self._dss = dss

    @property
    def names(self):
        return self._dss.Lines.AllNames()

    @property
    def name(self):
        return self._dss.Lines.Name()

    @name.setter
    def name(self, value):
        self._dss.Lines.Name(value)

    @property
    def bus1(self): return self._dss.Lines.Bus1()
    @property
    def bus2(self): return self._dss.Lines.Bus2()
    @property
    def phases(self): return self._dss.Lines.Phases()
    @property
    def length(self): return self._dss.Lines.Length()
    @property
    def units(self): return self._dss.Lines.Units()
    @property
    def r1(self): return self._dss.Lines.R1()
    @property
    def x1(self): return self._dss.Lines.X1()
    @property
    def rmatrix(self): return self._dss.Lines.RMatrix()
    @property
    def xmatrix(self): return self._dss.Lines.XMatrix()
    @property
    def norm_amps(self): return self._dss.Lines.NormAmps()


class _ODDTransformers:
    def __init__(self, dss):
        self._dss = dss

    @property
    def names(self):
        return self._dss.Transformers.AllNames()

    @property
    def name(self):
        return self._dss.Transformers.Name()

    @name.setter
    def name(self, value):
        self._dss.Transformers.Name(value)

    @property
    def buses(self): return self._dss.CktElement.BusNames()
    @property
    def phases(self): return self._dss.CktElement.NumPhases()
    @property
    def num_windings(self): return self._dss.Transformers.NumWindings()
    @property
    def winding(self): return self._dss.Transformers.Wdg()
    @winding.setter
    def winding(self, value): self._dss.Transformers.Wdg(value)
    @property
    def kv(self): return self._dss.Transformers.kV()
    @property
    def kva(self): return self._dss.Transformers.kVA()
    @property
    def resistance_percent(self): return self._dss.Transformers.R()
    @property
    def tap(self): return self._dss.Transformers.Tap()
    @property
    def is_delta(self): return self._dss.Transformers.IsDelta()
    @property
    def xhl_percent(self): return self._dss.Transformers.Xhl()


class _ODDError:
    def __init__(self, dss):
        self._dss = dss

    @property
    def error_code(self):
        return self._dss.Error.Number()

    @property
    def error_desc(self):
        return self._dss.Error.Description()


class _ODDInterface:
    def __init__(self, dss):
        self._dss = dss

    @property
    def num_circuits(self):
        return self._dss.Basic.NumCircuits()


def _raise_dss_error(dss, master: Path) -> None:
    error_interface = getattr(dss, "errorinterface", None)
    if error_interface is not None:
        number = int(getattr(error_interface, "error_code", 0) or 0)
        if number:
            description = getattr(error_interface, "error_desc", "unknown OpenDSS error")
            raise RuntimeError(f"OpenDSS failed to compile {master}: {description}")
    error = getattr(dss, "error", None)
    number = getattr(error, "number", 0) if error is not None else 0
    if not number:
        return
    description = getattr(error, "description", "unknown OpenDSS error")
    raise RuntimeError(f"OpenDSS failed to compile {master}: {description}")


def _bus_name(value: str) -> str:
    """Normaliza o nome e remove os nós do terminal."""
    return str(value).strip().split(".", 1)[0].casefold()


def _assign_bus_ids(
    names: list[str], explicit: Mapping[str, int] | None
) -> dict[str, int]:
    result: dict[str, int] = {}
    used: set[int] = set()
    for raw_name, raw_id in (explicit or {}).items():
        name = _bus_name(raw_name)
        if name not in names:
            raise ValueError(f"Configured OpenDSS bus id refers to unknown bus {raw_name!r}")
        bus_id = int(raw_id)
        if bus_id in used:
            raise ValueError(f"Duplicate configured OpenDSS bus id: {bus_id}")
        result[name] = bus_id
        used.add(bus_id)

    for name in names:
        if name in result:
            continue
        match = re.search(r"(\d+)$", name)
        candidate = int(match.group(1)) if match else None
        if candidate is not None and candidate not in used:
            result[name] = candidate
            used.add(candidate)

    next_id = 1
    for name in names:
        if name in result:
            continue
        while next_id in used:
            next_id += 1
        result[name] = next_id
        used.add(next_id)
    return result


def _read_bus(dss, name: str, bus_id: int) -> OpenDSSBusData:
    dss.circuit.set_active_bus(name)
    phases = tuple(int(node) for node in dss.bus.nodes if int(node) > 0)
    kv_base = float(getattr(dss.bus, "kv_base", 0.0) or 0.0)
    return OpenDSSBusData(bus_id, name, phases, kv_base)


def _read_lines(
    dss,
    name_to_id: Mapping[str, int],
    bus_by_id: Mapping[int, OpenDSSBusData],
    *,
    fallback_v_base_kv: float | None,
) -> tuple[OpenDSSBranchData, ...]:
    rows: list[OpenDSSBranchData] = []
    for line_name in tuple(dss.lines.names or ()):
        dss.lines.name = line_name
        name = str(dss.lines.name)
        bus1_raw, bus2_raw = str(dss.lines.bus1), str(dss.lines.bus2)
        bus1_name, bus2_name = _bus_name(bus1_raw), _bus_name(bus2_raw)
        try:
            bus1, bus2 = name_to_id[bus1_name], name_to_id[bus2_name]
        except KeyError as exc:
            raise ValueError(f"Line {name!r} references an unknown bus") from exc

        length = float(dss.lines.length)
        units = dss.lines.units
        n_phases = int(getattr(dss.lines, "phases", 3) or 3)
        if n_phases != 3:
            raise ValueError(
                f"Line {name!r} has {n_phases} phase(s); the current balanced "
                "positive-sequence adapter supports only three-phase lines"
            )
        phases = _terminal_phases(bus1_raw, n_phases)
        r_ohm = float(dss.lines.r1) * length
        x_ohm = float(dss.lines.x1) * length
        norm_amps = float(getattr(dss.lines, "norm_amps", 0.0) or 0.0)
        if norm_amps <= 0.0:
            raise ValueError(
                f"Line {name!r} has no positive NormAmps rating; define NormAmps "
                "in its Line or LineCode"
            )

        kv_ln = bus_by_id[bus1].kv_base_ln
        if kv_ln > 0.0:
            s_max_kva = n_phases * kv_ln * norm_amps
        elif fallback_v_base_kv is not None:
            factor = math.sqrt(3.0) if n_phases >= 3 else 1.0
            s_max_kva = factor * float(fallback_v_base_kv) * norm_amps
        else:
            raise ValueError(
                f"Cannot derive thermal rating for line {name!r}: Bus.kVBase is zero "
                "and no fallback voltage was configured"
            )

        r_matrix = _matrix(getattr(dss.lines, "rmatrix", ()), n_phases, length)
        x_matrix = _matrix(getattr(dss.lines, "xmatrix", ()), n_phases, length)
        rows.append(OpenDSSBranchData(
            name=name,
            from_bus=bus1,
            to_bus=bus2,
            phases=phases,
            r_ohm=r_ohm,
            x_ohm=x_ohm,
            s_max_kva=s_max_kva,
            norm_amps=norm_amps,
            length=length,
            units=units,
            r_matrix_ohm=r_matrix,
            x_matrix_ohm=x_matrix,
            element_type="line",
        ))
    return tuple(rows)


def _read_transformers(
    dss,
    name_to_id: Mapping[str, int],
) -> tuple[OpenDSSBranchData, ...]:
    interface = getattr(dss, "transformers", None)
    names = tuple(getattr(interface, "names", ()) or ()) if interface else ()
    rows: list[OpenDSSBranchData] = []

    for transformer_name in names:
        interface.name = transformer_name
        name = str(interface.name)
        n_phases = int(interface.phases)
        n_windings = int(interface.num_windings)
        if n_phases != 3 or n_windings != 2:
            raise ValueError(
                f"Transformer {name!r} has {n_phases} phase(s) and "
                f"{n_windings} winding(s); only three-phase two-winding "
                "transformers are supported by the balanced adapter"
            )

        terminals = tuple(str(bus) for bus in interface.buses)
        if len(terminals) != 2:
            raise ValueError(f"Transformer {name!r} does not expose two terminals")
        bus_names = tuple(_bus_name(bus) for bus in terminals)
        try:
            bus1, bus2 = (name_to_id[bus_names[0]], name_to_id[bus_names[1]])
        except KeyError as exc:
            raise ValueError(f"Transformer {name!r} references an unknown bus") from exc

        windings = []
        for winding in (1, 2):
            interface.winding = winding
            windings.append({
                "kv": float(interface.kv),
                "kva": float(interface.kva),
                "r_percent": float(interface.resistance_percent),
                "tap": float(interface.tap),
                "connection": "delta" if bool(interface.is_delta) else "wye",
            })
        w1, w2 = windings
        if min(w1["kv"], w2["kv"], w1["kva"], w2["kva"], w1["tap"], w2["tap"]) <= 0.0:
            raise ValueError(f"Transformer {name!r} has non-positive kV, kVA or tap data")

        # %R usa a base de cada enrolamento; XHL usa a base do enrolamento 1.
        r_pu = (
            w1["r_percent"] / 100.0
            + w2["r_percent"] / 100.0 * w1["kva"] / w2["kva"]
        )
        x_pu = float(interface.xhl_percent) / 100.0
        z_base_w1 = w1["kv"] ** 2 * 1e3 / w1["kva"]
        s_max_kva = min(w1["kva"], w2["kva"])
        rows.append(OpenDSSBranchData(
            name=name,
            from_bus=bus1,
            to_bus=bus2,
            phases=_terminal_phases(terminals[0], n_phases),
            r_ohm=r_pu * z_base_w1,
            x_ohm=x_pu * z_base_w1,
            s_max_kva=s_max_kva,
            norm_amps=s_max_kva / (math.sqrt(3.0) * w1["kv"]),
            length=1.0,
            units="transformer",
            r_matrix_ohm=(),
            x_matrix_ohm=(),
            element_type="transformer",
            tap_ratio=w2["tap"] / w1["tap"],
            r_pu_on_rating=r_pu,
            x_pu_on_rating=x_pu,
            impedance_base_kva=w1["kva"],
            connections=(w1["connection"], w2["connection"]),
        ))
    return tuple(rows)


def _terminal_phases(terminal: str, count: int) -> tuple[int, ...]:
    parts = str(terminal).strip().split(".")[1:]
    nodes = tuple(int(part) for part in parts if part.isdigit() and int(part) > 0)
    return nodes[:count] if nodes else tuple(range(1, count + 1))


def _matrix(values, order: int, scale: float) -> tuple[tuple[float, ...], ...]:
    flat = [float(value) * scale for value in (values or ())]
    if not flat:
        return ()
    if len(flat) == order * order:
        return tuple(tuple(flat[i * order:(i + 1) * order]) for i in range(order))
    if len(flat) == order * (order + 1) // 2:
        matrix = [[0.0] * order for _ in range(order)]
        k = 0
        for i in range(order):
            for j in range(i + 1):
                matrix[i][j] = matrix[j][i] = flat[k]
                k += 1
        return tuple(tuple(row) for row in matrix)
    raise ValueError(
        f"OpenDSS impedance matrix has {len(flat)} values; expected "
        f"{order * order} or {order * (order + 1) // 2}"
    )


def _resolve_root(
    slack_bus: str | int | None,
    name_to_id: Mapping[str, int],
    branches: tuple[OpenDSSBranchData, ...],
) -> int:
    if slack_bus is not None:
        root = resolve_bus_id(slack_bus, name_to_id)
        if root not in name_to_id.values():
            raise ValueError(f"Configured slack bus id {root} is not in the OpenDSS circuit")
        return root

    from_buses = {branch.from_bus for branch in branches}
    to_buses = {branch.to_bus for branch in branches}
    candidates = from_buses - to_buses
    if len(candidates) != 1:
        raise ValueError(
            "Could not infer one OpenDSS slack bus from line orientation; set "
            "network.slack_bus in config.json"
        )
    return next(iter(candidates))


def _orient_radial(
    bus_ids: tuple[int, ...],
    branches: tuple[OpenDSSBranchData, ...],
    root: int,
) -> tuple[OpenDSSBranchData, ...]:
    if len(branches) != len(bus_ids) - 1:
        raise ValueError(
            "The current OPF OpenDSS adapter supports radial networks composed "
            f"of lines and two-winding transformers: found {len(bus_ids)} buses "
            f"and {len(branches)} supported branches"
        )

    adjacent: dict[int, list[tuple[int, OpenDSSBranchData]]] = {b: [] for b in bus_ids}
    for branch in branches:
        adjacent[branch.from_bus].append((branch.to_bus, branch))
        adjacent[branch.to_bus].append((branch.from_bus, branch))

    visited = {root}
    queue = deque([root])
    oriented: list[OpenDSSBranchData] = []
    while queue:
        parent = queue.popleft()
        for child, branch in adjacent[parent]:
            if child in visited:
                continue
            visited.add(child)
            queue.append(child)
            tap_ratio = (
                1.0 / branch.tap_ratio if branch.from_bus != parent
                else branch.tap_ratio
            )
            oriented.append(replace(
                branch,
                from_bus=parent,
                to_bus=child,
                tap_ratio=tap_ratio,
            ))

    missing = set(bus_ids) - visited
    if missing:
        raise ValueError(f"OpenDSS line network is disconnected; unreachable bus ids: {sorted(missing)}")
    return tuple(oriented)
