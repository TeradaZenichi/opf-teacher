# opf-teacher

Operação ótima de baterias (BESS) e DERs em **redes de distribuição** para
redução de custo, via **OPF** (fluxo de potência ótimo).

O modelo usa o **branch flow model (DistFlow, Baran-Wu)** para redes radiais com
**relaxação cônica (SOCP)** e aproximação de tensão fixa `v ≈ v_nom`:

```
P_j² + Q_j²  ≤  ℓ_j          (cone, com v ≈ 1 p.u.)
v_j = v_i − 2(r_j P_j + x_j Q_j) + (r_j² + x_j²) ℓ_j     (queda de tensão)
soc_t = soc_{t−1} + (η_c·pch − pdis/η_d)·Δt              (dinâmica da bateria)
min  Σ_t  preço_t · (import_t − r_feedin·export_t) · Δt   (custo de energia)
```

A formulação matemática completa (índices, parâmetros, variáveis, objetivo e
todas as restrições) está em [docs/formulation.md](docs/formulation.md).

## Estrutura

```
opf/
├── components.py  classes de dominio (Base, Bus, Branch, Grid, Bess, Pv, Case)
│                  com parametros fisicos + um .result preenchido apos o solve
├── data.py        load_case(dir) -> Case   (lê JSON+CSV)
├── model.py       build_model(case) -> modelo Pyomo (DistFlow SOCP, converte p/ p.u.)
└── results.py     attach_results(model, case) -> popula .result e .summary
teacher.py         BessOpt: facade  ->  BessOpt(dir).build().solve()
run.py             resolve + gera painel de graficos em figures/
examples/
└── case5/         exemplo: feeder radial de 5 barras, 1 BESS, 1 PV
```

Cada componente carrega seus **parâmetros** (em unidades físicas) e, após o
solve, seu **resultado** em `.result` (séries indexadas por timestamp). A
conversão para p.u. é centralizada em `Base` e acontece dentro do modelo.

### Anatomia de um exemplo (`examples/case5/`)

| Arquivo | Papel |
|---|---|
| `config.json`  | sistema/rede: base p.u. (kVA/kV), grid/subestação, limites de tensão, objetivo |
| `devices.json` | DERs controláveis: `bess`, `pv`, `evse` (existência + parâmetros estáticos) |
| `bus.csv`      | barras: `bus_id, type, name` |
| `branch.csv`   | ramos radiais: `from_bus, to_bus, r_ohm, x_ohm, s_max_kva` |
| `demand.csv`   | carga (wide): `timestamp, P2, Q2, …, Pn, Qn` em kW / kVAr |
| `pv.csv`       | disponibilidade PV: `timestamp, PV<bus>` em kW |
| `price.csv`    | tarifa: `timestamp, price_per_kwh` |

Convenção de unidades nos arquivos: **kW / kVAr / kWh / Ω / kV**. O loader
converte tudo para por-unidade na base do sistema (`config.base`); o eixo de
tempo (períodos e `Δt`) é derivado dos `timestamp` das séries.

## Uso

```python
from teacher import BessOpt

case = BessOpt("examples/case5").build().solve(solver="gurobi_direct")

print(case.summary)                    # custo, energia, perdas, tensão min/max
print(case.bess[0].result.soc_kwh)     # SoC (kWh) da bateria por timestamp
print(case.bess[0].result.p_net_kw)    # carga (+) / descarga (-) (kW)
print(case.grid.result.import_kw)      # importação da rede (kW)
print(case.buses[4].result.v_pu)       # tensão (p.u.) na barra 4
```

`solve()` retorna o próprio `Case` com cada componente populado: `case.grid`,
`case.bess[i]`, `case.pv[i]`, `case.buses[b]`, `case.branches[i]` — todos com
`.result` em kW/kVAr/kV/kWh.

### Visualização

`run.py` resolve um exemplo e gera um painel (tarifa×import, bateria×SoC,
PV×carga, tensões) salvo em `figures/`:

```bash
python run.py                       # examples/case5
python run.py examples/case5 --show # abre janela do matplotlib
```

## Solver

O problema é um **MISOCP**: objetivo linear, restrição cônica (SOCP) e uma
restrição **SOS1** por período (não-simultaneidade import/export). O solver
precisa suportar cone quadrático **e** SOS/branch-and-bound — LP/MILP puro
(ex.: HiGHS) e NLP contínuo puro (ex.: IPOPT) **não** servem. Opções:

- **Gurobi** (`gurobi_direct`) — usado por padrão. Resolve SOCP + SOS1 nativamente.
- **SCIP** (`scip` / PySCIPOpt) — open-source, MINLP/MISOCP.

(Sem a restrição SOS1 — usando a variante com regularização ou `Pgrid` único — o
modelo volta a ser um SOCP convexo e o IPOPT também resolve.)

### Licença Gurobi (acadêmica, gratuita)

O `gurobipy` instalado via `pip` vem com uma **licença restrita** que só resolve
modelos pequenos (o `case5` completo de 24h excede o limite por causa das
restrições cônicas). Para liberar:

1. Crie/entre na conta em <https://portal.gurobi.com> com o e-mail acadêmico.
2. Solicite uma licença **Academic WLS** (Web License Service) — é a que funciona
   com o `gurobipy` do pip, sem precisar instalar o Gurobi completo.
3. Baixe o `gurobi.lic` (contém `WLSACCESSID`, `WLSSECRET`, `LICENSEID`) e coloque
   em `C:\Users\<você>\gurobi.lic`, ou exporte as variáveis de ambiente
   correspondentes.

Verifique com:

```python
import gurobipy as gp
gp.Env().start()   # sem erro => licença ativa
```
