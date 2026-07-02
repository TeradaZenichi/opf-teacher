# Formulação do problema

Operação ótima de BESS/DERs em rede **radial** de distribuição, via **branch
flow model (DistFlow, Baran–Wu)** com **relaxação cônica (SOCP)** e aproximação
de tensão fixa $v \approx 1\,\text{p.u.}$. Tudo em por-unidade na base do sistema.

## Convenção

- Barras $i, j \in B$; ramo $(i,j) \in L$ orientado do pai $i$ para o filho $j$
  (rede radial: cada barra $\neq r$ tem exatamente um ramo de entrada).
- Tempo $t \in T$; passo $\Delta t$ (h).
- Barra da subestação (slack): $r$. Filhos de $j$: $A(j) = \{k : (j,k)\in L\}$.
- Conjuntos de barras com dispositivo: $B^{\text{BESS}}$, $B^{\text{PV}}$.
- **Superíndice** = tipo (Grid/BESS/PV); **subíndice** = barra(s) e tempo.

## Variáveis

$$
\begin{aligned}
v_{j,t} &= |V_{j,t}|^2 &&\text{(tensão ao quadrado)}\\
\ell_{ij,t} &= |I_{ij,t}|^2 &&\text{(corrente ao quadrado no ramo } i\!\to\! j)\\
P_{ij,t},\ Q_{ij,t} && &\text{(fluxos ativo/reativo no extremo emissor)}\\
P^{\text{grid,imp}}_{j,t},\ P^{\text{grid,exp}}_{j,t},\ Q^{\text{grid}}_{j,t} && &\text{(importação/exportação/reativo da rede, em } j=r)\\
P^{\text{BESS,ch}}_{j,t},\ P^{\text{BESS,dis}}_{j,t},\ E^{\text{BESS}}_{j,t} && &\text{(carga/descarga/energia armazenada)}\\
P^{\text{PV}}_{j,t} && &\text{(geração fotovoltaica despachada)}
\end{aligned}
$$

## Parâmetros

$$
\begin{aligned}
&r_{ij},\ x_{ij} &&\text{resistência/reatância do ramo}\\
&\overline{\ell}_{ij} = \big(s^{\max}_{ij}\big)^2 &&\text{limite térmico (corrente}^2)\\
&P^{\text{d}}_{j,t},\ Q^{\text{d}}_{j,t} &&\text{demanda ativa/reativa}\\
&\underline{v}_j,\ \overline{v}_j,\ v^{\text{ref}} &&\text{limites de tensão e referência da subestação}\\
&\overline{E}_j,\ \eta^{\text{ch}}_j,\ \eta^{\text{dis}}_j,\ \text{SoC}^{\text{ini}}_j,\ \text{SoC}^{\min}_j,\ \text{SoC}^{\max}_j &&\text{parâmetros do BESS}\\
&\overline{P}^{\text{ch}}_j,\ \overline{P}^{\text{dis}}_j &&\text{potências máximas de carga/descarga}\\
&P^{\text{PV,av}}_{j,t},\ \tan\varphi_j &&\text{disponibilidade PV e razão Q/P}\\
&\overline{P}^{\text{imp}},\ \overline{P}^{\text{exp}},\ \overline{Q}^{\text{grid}},\ \rho &&\text{limites da rede e tarifa de feed-in}\\
&\pi_t,\ S_{\text{base}},\ \varepsilon &&\text{tarifa (R\$/kWh), base e regularização}
\end{aligned}
$$

## Função objetivo

Minimizar o custo de energia (importação menos receita de exportação), com uma
pequena regularização $\varepsilon$ que evita import/export simultâneos:

$$
\min \quad \sum_{t \in T} \pi_t \left( P^{\text{grid,imp}}_{r,t} - \rho\, P^{\text{grid,exp}}_{r,t} \right) S_{\text{base}}\, \Delta t
\;+\; \varepsilon \sum_{t \in T} \left( P^{\text{grid,imp}}_{r,t} + P^{\text{grid,exp}}_{r,t} \right)
$$

## Restrições

### Injeção líquida na barra (geração − carga)

$$
p^{\text{inj}}_{j,t} =
\big(P^{\text{grid,imp}}_{j,t} - P^{\text{grid,exp}}_{j,t}\big)\big|_{j=r}
+ \big(P^{\text{BESS,dis}}_{j,t} - P^{\text{BESS,ch}}_{j,t}\big)\big|_{j \in B^{\text{BESS}}}
+ P^{\text{PV}}_{j,t}\big|_{j \in B^{\text{PV}}}
- P^{\text{d}}_{j,t}
$$

$$
q^{\text{inj}}_{j,t} =
Q^{\text{grid}}_{j,t}\big|_{j=r}
+ \tan\varphi_j\, P^{\text{PV}}_{j,t}\big|_{j \in B^{\text{PV}}}
- Q^{\text{d}}_{j,t}
$$

### Balanço de potência (DistFlow), $\forall j \in B,\ t \in T$

$$
\sum_{k \in A(j)} P_{jk,t} = \big(P_{ij,t} - r_{ij}\,\ell_{ij,t}\big)\big|_{j \neq r} + p^{\text{inj}}_{j,t}
$$

$$
\sum_{k \in A(j)} Q_{jk,t} = \big(Q_{ij,t} - x_{ij}\,\ell_{ij,t}\big)\big|_{j \neq r} + q^{\text{inj}}_{j,t}
$$

### Queda de tensão, $\forall (i,j) \in L,\ t \in T$

$$
v_{j,t} = v_{i,t} - 2\big(r_{ij} P_{ij,t} + x_{ij} Q_{ij,t}\big) + \big(r_{ij}^2 + x_{ij}^2\big)\,\ell_{ij,t}
$$

### Relaxação cônica (SOCP), $\forall (i,j) \in L,\ t \in T$

$$
P_{ij,t}^2 + Q_{ij,t}^2 \le \ell_{ij,t}
\qquad\big(\text{exato: } P_{ij,t}^2 + Q_{ij,t}^2 \le v_{i,t}\,\ell_{ij,t}\big)
$$

### Limites de rede (térmico e tensão)

$$
0 \le \ell_{ij,t} \le \overline{\ell}_{ij}, \qquad
\underline{v}_j^{\,2} \le v_{j,t} \le \overline{v}_j^{\,2}, \qquad
v_{r,t} = \big(v^{\text{ref}}\big)^2
$$

### BESS, $\forall j \in B^{\text{BESS}},\ t \in T$

$$
E^{\text{BESS}}_{j,t} =
\begin{cases}
\text{SoC}^{\text{ini}}_j\, \overline{E}_j + \big(\eta^{\text{ch}}_j P^{\text{BESS,ch}}_{j,t} - P^{\text{BESS,dis}}_{j,t}/\eta^{\text{dis}}_j\big)\Delta t, & t = t_0\\[4pt]
E^{\text{BESS}}_{j,t-1} + \big(\eta^{\text{ch}}_j P^{\text{BESS,ch}}_{j,t} - P^{\text{BESS,dis}}_{j,t}/\eta^{\text{dis}}_j\big)\Delta t, & t > t_0
\end{cases}
$$

$$
\text{SoC}^{\min}_j\, \overline{E}_j \le E^{\text{BESS}}_{j,t} \le \text{SoC}^{\max}_j\, \overline{E}_j, \qquad
E^{\text{BESS}}_{j,\,t_{\text{end}}} = \text{SoC}^{\text{ini}}_j\, \overline{E}_j \ \text{(SoC cíclico)}
$$

$$
0 \le P^{\text{BESS,ch}}_{j,t} \le \overline{P}^{\text{ch}}_j, \qquad
0 \le P^{\text{BESS,dis}}_{j,t} \le \overline{P}^{\text{dis}}_j
$$

### PV, $\forall j \in B^{\text{PV}},\ t \in T$

$$
0 \le P^{\text{PV}}_{j,t} \le P^{\text{PV,av}}_{j,t}
\qquad(\text{ou } P^{\text{PV}}_{j,t} = P^{\text{PV,av}}_{j,t} \text{ se não-curtailável})
$$

### Rede / subestação, $\forall t \in T$

$$
0 \le P^{\text{grid,imp}}_{r,t} \le \overline{P}^{\text{imp}}, \qquad
0 \le P^{\text{grid,exp}}_{r,t} \le \overline{P}^{\text{exp}}, \qquad
\big|Q^{\text{grid}}_{r,t}\big| \le \overline{Q}^{\text{grid}}
$$
