# OPF formulation

[Versão em português](formulation.pt-BR.md)

Optimal operation of BESS and distributed energy resources in a radial
distribution network using the DistFlow model with an SOCP relaxation. All
electrical quantities are represented in per unit.

## Notation

- Buses $i,j \in B$ and branches $(i,j) \in L$, directed from parent $i$ to
  child $j$. Every non-root bus has one incoming branch.
- Time periods $t \in T$, duration $\Delta t$ in hours, and root bus $r$.
- Children of bus $j$: $A(j)=\{k:(j,k)\in L\}$.
- BESS units $s\in S$ and PV systems $g\in G$. Sets $S_j$ and $G_j$ contain
  the devices connected to bus $j$.

## Variables

$$
\begin{aligned}
v_{j,t} &= |V_{j,t}|^2 &&\text{(squared voltage magnitude)}\\
\ell_{ij,t} &= |I_{ij,t}|^2 &&\text{(squared branch current)}\\
P_{ij,t},\ Q_{ij,t} && &\text{(sending-end active and reactive power)}\\
P^{\text{grid,imp}}_{r,t},\ P^{\text{grid,exp}}_{r,t},\ Q^{\text{grid}}_{r,t}
&& &\text{(substation exchange)}\\
P^{\text{BESS,ch}}_{s,t},\ P^{\text{BESS,dis}}_{s,t},\ Q^{\text{BESS}}_{s,t},\ E^{\text{BESS}}_{s,t}
&& &\text{(BESS charge, discharge, reactive power, and energy)}\\
P^{\text{loss,Q}}_{s,t} && &\text{(incremental BESS inverter loss)}\\
P^{\text{PV}}_{g,t},\ Q^{\text{PV}}_{g,t},\ P^{\text{loss,Q}}_{g,t}
&& &\text{(PV active power, reactive power, and incremental loss)}
\end{aligned}
$$

## Parameters

$$
\begin{aligned}
r_{ij},\ x_{ij} &:\quad\text{branch resistance and reactance}\\
\tau_{ij} &:\quad\text{tap ratio in pu; }\tau_{ij}=1\text{ for lines}\\
\overline{\ell}_{ij}=\big(s^{\max}_{ij}\big)^2
&:\quad\text{squared-current limit}\\
P^{\text d}_{j,t},\ Q^{\text d}_{j,t} &:\quad\text{active and reactive demand}\\
\underline v_j,\ \overline v_j,\ v^{\text ref}
&:\quad\text{voltage limits and substation reference}\\
\overline E_s,\ \eta^{\text ch}_s,\ \eta^{\text dis}_s,
\ \text{SoC}^{\text ini}_s,\ \text{SoC}^{\min}_s,\ \text{SoC}^{\max}_s
&:\quad\text{BESS parameters}\\
\overline P^{\text ch}_s,\ \overline P^{\text dis}_s,\ \overline S^{\text{BESS}}_s
&:\quad\text{BESS power limits and inverter rating}\\
\overline P^{\text{loss,Q}}_s
&:\quad\text{BESS loss at }|Q|=\overline S^{\text{BESS}}_s\\
P^{\text{PV,av}}_{g,t},\ \overline S^{\text{PV}}_g,\ \tan\varphi_g
&:\quad\text{PV availability, rating, and fixed-PF ratio}\\
\overline P^{\text{loss,Q}}_g
&:\quad\text{PV loss at }|Q|=\overline S^{\text{PV}}_g\\
\overline P^{\text imp},\ \overline P^{\text exp},\ \overline Q^{\text grid},\ \rho
&:\quad\text{grid limits and feed-in ratio}\\
\pi_t,\ S_{\text base} &:\quad\text{energy price and system power base}
\end{aligned}
$$

## Objective

The objective is the net energy cost:

$$
\min \sum_{t\in T}\pi_t
\left(P^{\text{grid,imp}}_{r,t}-\rho P^{\text{grid,exp}}_{r,t}\right)
S_{\text base}\Delta t.
$$

## Constraints

### Net bus injection

$$
p^{\text inj}_{j,t}=
\left(P^{\text{grid,imp}}_{j,t}-P^{\text{grid,exp}}_{j,t}\right)\big|_{j=r}
+\sum_{s\in S_j}\left(P^{\text{BESS,dis}}_{s,t}-P^{\text{BESS,ch}}_{s,t}\right)
+\sum_{g\in G_j}P^{\text{PV}}_{g,t}-P^{\text d}_{j,t}
$$

$$
q^{\text inj}_{j,t}=Q^{\text grid}_{j,t}\big|_{j=r}
+\sum_{s\in S_j}Q^{\text{BESS}}_{s,t}
+\sum_{g\in G_j}Q^{\text{PV}}_{g,t}-Q^{\text d}_{j,t}.
$$

The symbol $r$ denotes the root or substation bus. The notation
$X\big|_{j=r}$ does not set every bus index to $r$; it includes $X$ only in the
root-bus equation:

$$
X\big|_{j=r}=\begin{cases}
X, & j=r,\\
0, & j\neq r.
\end{cases}
$$

Grid import, export, and reactive power therefore appear only at the
substation. At every other bus, net injection consists of local devices minus
local demand. The root-bus balance also has no incoming flow from a parent
branch.

### DistFlow balance

For every $j\in B$ and $t\in T$:

$$
\sum_{k\in A(j)}P_{jk,t}=
\left(P_{ij,t}-r_{ij}\ell_{ij,t}\right)\big|_{j\neq r}+p^{\text inj}_{j,t}
$$

$$
\sum_{k\in A(j)}Q_{jk,t}=
\left(Q_{ij,t}-x_{ij}\ell_{ij,t}\right)\big|_{j\neq r}+q^{\text inj}_{j,t}.
$$

### Voltage drop and transformer tap

For every $(i,j)\in L$ and $t\in T$:

$$
v_{j,t}=\tau_{ij}^2\left[
v_{i,t}-2\left(r_{ij}P_{ij,t}+x_{ij}Q_{ij,t}\right)
+\left(r_{ij}^2+x_{ij}^2\right)\ell_{ij,t}
\right].
$$

For transformers, leakage impedance is referred to the sending side and
followed by the ideal transformer. Transformer impedances are converted from
their rated kVA base to the OPF power base.

### SOCP relaxation

$$
P_{ij,t}^2+Q_{ij,t}^2\leq v_{i,t}\ell_{ij,t}.
$$

After the solve, tightness is checked with

$$
g_{ij,t}=v_{i,t}\ell_{ij,t}-P_{ij,t}^2-Q_{ij,t}^2.
$$

The default scaled-gap tolerance is $10^{-5}$. The report also includes gap
relative to branch flow, equivalent current error in amperes, and equivalent
loss error in watts. Tightness is classified as `tight`, `acceptable`, or
`not_tight`; it is a numerical and physical diagnostic, not a statistical
probability.

### Network limits

$$
0\leq\ell_{ij,t}\leq\overline\ell_{ij},\qquad
\underline v_j^{,2}\leq v_{j,t}\leq\overline v_j^{,2},\qquad
v_{r,t}=\big(v^{\text ref}\big)^2.
$$

### BESS

$$
E^{\text{BESS}}_{s,t}=
\begin{cases}
\text{SoC}^{\text ini}_s\overline E_s+
\left(\eta^{\text ch}_sP^{\text{BESS,ch}}_{s,t}
-P^{\text{BESS,dis}}_{s,t}/\eta^{\text dis}_s-P^{\text{loss,Q}}_{s,t}\right)\Delta t,&t=t_0,\\[4pt]
E^{\text{BESS}}_{s,t-1}+
\left(\eta^{\text ch}_sP^{\text{BESS,ch}}_{s,t}
-P^{\text{BESS,dis}}_{s,t}/\eta^{\text dis}_s-P^{\text{loss,Q}}_{s,t}\right)\Delta t,&t>t_0.
\end{cases}
$$

$$
\text{SoC}^{\min}_s\overline E_s\leq E^{\text{BESS}}_{s,t}
\leq\text{SoC}^{\max}_s\overline E_s
$$

$$
0\leq P^{\text{BESS,ch}}_{s,t}\leq\overline P^{\text ch}_s,\qquad
0\leq P^{\text{BESS,dis}}_{s,t}\leq\overline P^{\text dis}_s.
$$

The BESS inverter rating is enforced by

$$
\left(P^{\text{BESS,dis}}_{s,t}-P^{\text{BESS,ch}}_{s,t}\right)^2+
\left(Q^{\text{BESS}}_{s,t}\right)^2\leq
\left(\overline S^{\text{BESS}}_s\right)^2.
$$

The incremental reactive-power loss satisfies

$$
\overline P^{\text{loss,Q}}_s\left(Q^{\text{BESS}}_{s,t}\right)^2
\leq
\left(\overline S^{\text{BESS}}_s\right)^2P^{\text{loss,Q}}_{s,t}.
$$

At $|Q|=\overline S^{\text{BESS}}_s$, the minimum loss is
$\overline P^{\text{loss,Q}}_s$. This loss drains stored energy.

With `reactive_control: true`, the OPF selects $Q^{\text{BESS}}_{s,t}$. When
the option is omitted or false, $Q^{\text{BESS}}_{s,t}=0$.

If cyclic operation is enabled:

$$
E^{\text{BESS}}_{s,t_{\text end}}=\text{SoC}^{\text ini}_s\overline E_s.
$$

The reported net BESS power follows the convention
$P^{\text{BESS,ch}}-P^{\text{BESS,dis}}$: positive for charging and negative
for discharging.

### PV inverter

$$
P^{\text{PV}}_{g,t}+P^{\text{loss,Q}}_{g,t}
\leq P^{\text{PV,av}}_{g,t}
$$

Equality is used for a non-curtailable PV inverter.

$$
\left(P^{\text{PV}}_{g,t}\right)^2+
\left(Q^{\text{PV}}_{g,t}\right)^2\leq
\left(\overline S^{\text{PV}}_g\right)^2.
$$

$$
\overline P^{\text{loss,Q}}_g\left(Q^{\text{PV}}_{g,t}\right)^2
\leq
\left(\overline S^{\text{PV}}_g\right)^2P^{\text{loss,Q}}_{g,t}.
$$

For PV, reactive-power loss consumes part of the available solar power. The
model does not include grid-powered night-VAR operation, so a positive rated
loss prevents reactive support when $P^{\text{PV,av}}_{g,t}=0$.

Both loss relations are convex epigraphs. With positive energy prices, cost
minimization also minimizes $P^{\text{loss,Q}}$, so the inequalities are tight
at the optimum.

For `fixed_pf`, $Q^{\text{PV}}_{g,t}=\tan\varphi_gP^{\text{PV}}_{g,t}$.
The other supported modes are `optimal`, `volt-var`, `volt-watt`, and
`volt-var-watt`.

### Substation

$$
0\leq P^{\text{grid,imp}}_{r,t}\leq\overline P^{\text imp},\qquad
0\leq P^{\text{grid,exp}}_{r,t}\leq\overline P^{\text exp},\qquad
\left|Q^{\text grid}_{r,t}\right|\leq\overline Q^{\text grid}.
$$

An SOS1 constraint prevents simultaneous import and export:

$$
\text{SOS1}\left(\left\{P^{\text{grid,imp}}_{r,t},
P^{\text{grid,exp}}_{r,t}\right\}\right),\qquad \forall t\in T.
$$
