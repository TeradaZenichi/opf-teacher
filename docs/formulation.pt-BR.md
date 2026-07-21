# Formulação do problema

[English version](formulation.md)

Operação ótima de BESS/DERs em rede radial de distribuição usando o modelo
DistFlow com relaxação cônica (SOCP). As grandezas elétricas são representadas
em pu.

## Convenção

- Barras $i, j \in B$; ramo $(i,j) \in L$ orientado do pai $i$ para o filho $j$
  (rede radial: cada barra $\neq r$ tem exatamente um ramo de entrada).
- Tempo $t \in T$; passo $\Delta t$ (h). Barra da subestação (slack): $r$.
- Filhos de $j$: $A(j) = \{k : (j,k)\in L\}$.
- Dispositivos: BESS $s \in S$, PV $g \in G$. Conectados à barra $j$:
  $S_j \subseteq S$ e $G_j \subseteq G$ (podem ter mais de um por barra).
- **Superíndice** = tipo (Grid/BESS/PV); **subíndice** = barra(s) $i,j$, dispositivo
  $s,g$ e tempo $t$.

## Variáveis

$$
\begin{aligned}
v_{j,t} &= |V_{j,t}|^2 &&\text{(tensão ao quadrado)}\\
\ell_{ij,t} &= |I_{ij,t}|^2 &&\text{(corrente ao quadrado no ramo } i\!\to\! j)\\
P_{ij,t},\ Q_{ij,t} && &\text{(fluxos ativo/reativo no extremo emissor)}\\
P^{\text{grid,imp}}_{r,t},\ P^{\text{grid,exp}}_{r,t},\ Q^{\text{grid}}_{r,t} && &\text{(importação/exportação/reativo da rede na subestação)}\\
P^{\text{BESS,ch}}_{s,t},\ P^{\text{BESS,dis}}_{s,t},\ Q^{\text{BESS}}_{s,t},\ E^{\text{BESS}}_{s,t}
&& &\text{(carga, descarga, reativo e energia do BESS } s)\\
P^{\text{loss,Q}}_{s,t} && &\text{(perda incremental do inversor BESS)}\\
P^{\text{PV}}_{g,t},\ Q^{\text{PV}}_{g,t},\ P^{\text{loss,Q}}_{g,t},\ P^{\text{PV,grid}}_{g,t}
&& &\text{(potências do PV, perda incremental e consumo da rede)}
\end{aligned}
$$

## Parâmetros

$$
\begin{aligned}
r_{ij},\ x_{ij} &:\quad \text{resistência e reatância do ramo}\\
\tau_{ij} &:\quad \text{relação de tap em pu; igual a 1 para linhas}\\
\overline{\ell}_{ij}=\big(s^{\max}_{ij}\big)^2
&:\quad \text{limite de corrente ao quadrado}\\
P^{\text d}_{j,t},\ Q^{\text d}_{j,t}
&:\quad \text{demandas ativa e reativa}\\
\underline v_j,\ \overline v_j,\ v^{\text ref}
&:\quad \text{limites de tensão e referência da subestação}\\
\overline E_s,\ \eta^{\text ch}_s,\ \eta^{\text dis}_s,
\ \text{SoC}^{\text ini}_s,\ \text{SoC}^{\min}_s,\ \text{SoC}^{\max}_s
&:\quad \text{parâmetros do BESS}\\
\overline P^{\text ch}_s,\ \overline P^{\text dis}_s,\ \overline S^{\text{BESS}}_s
&:\quad \text{limites de potência e capacidade do inversor BESS}\\
\overline P^{\text{loss,Q}}_s
&:\quad \text{perda BESS em }|Q|=\overline S^{\text{BESS}}_s\\
P^{\text{PV,av}}_{g,t},\ \overline S^{\text{PV}}_g,\ \tan\varphi_g
&:\quad \text{disponibilidade, potência aparente e razão Q/P do PV}\\
\overline P^{\text{loss,Q}}_g
&:\quad \text{perda PV em }|Q|=\overline S^{\text{PV}}_g\\
\overline P^{\text imp},\ \overline P^{\text exp},\ \overline Q^{\text grid},\ \rho
&:\quad \text{limites da rede e razão de remuneração da exportação}\\
\pi_t,\ S_{\text base} &:\quad \text{tarifa e base de potência do sistema}
\end{aligned}
$$

## Função objetivo

Minimizar o custo de energia (importação menos receita de exportação):

$$
\min \quad \sum_{t \in T} \pi_t \left( P^{\text{grid,imp}}_{r,t} - \rho\, P^{\text{grid,exp}}_{r,t} \right) S_{\text{base}}\, \Delta t
$$

## Restrições

### Injeção líquida na barra (geração − carga)

$$
p^{\text{inj}}_{j,t} =
\big(P^{\text{grid,imp}}_{j,t} - P^{\text{grid,exp}}_{j,t}\big)\big|_{j=r}
+ \sum_{s \in S_j}\big(P^{\text{BESS,dis}}_{s,t} - P^{\text{BESS,ch}}_{s,t}\big)
+ \sum_{g \in G_j}\left(P^{\text{PV}}_{g,t}-P^{\text{PV,grid}}_{g,t}\right)
- P^{\text{d}}_{j,t}
$$

$$
q^{\text{inj}}_{j,t} =
Q^{\text{grid}}_{j,t}\big|_{j=r}
+ \sum_{s \in S_j} Q^{\text{BESS}}_{s,t}
+ \sum_{g \in G_j} Q^{\text{PV}}_{g,t}
- Q^{\text{d}}_{j,t}
$$

A letra $r$ identifica a barra raiz, isto é, a subestação. A notação
$X\big|_{j=r}$ não impõe $j=r$ para todas as barras; ela inclui $X$ somente
quando a equação está sendo escrita para a barra raiz:

$$
X\big|_{j=r}=\begin{cases}
X, & j=r,\\
0, & j\neq r.
\end{cases}
$$

Assim, importação, exportação e potência reativa da rede aparecem apenas na
injeção da subestação. Nas demais barras, a injeção contém somente os
dispositivos locais e a demanda. Pelo mesmo motivo, o balanço da barra raiz não
possui fluxo de entrada vindo de um ramo pai.

### Balanço de potência (DistFlow), $\forall j \in B,\ t \in T$

$$
\sum_{k \in A(j)} P_{jk,t} = \big(P_{ij,t} - r_{ij}\,\ell_{ij,t}\big)\big|_{j \neq r} + p^{\text{inj}}_{j,t}
$$

$$
\sum_{k \in A(j)} Q_{jk,t} = \big(Q_{ij,t} - x_{ij}\,\ell_{ij,t}\big)\big|_{j \neq r} + q^{\text{inj}}_{j,t}
$$

### Queda de tensão, $\forall (i,j) \in L,\ t \in T$

$$
v_{j,t} = \tau_{ij}^2\left[v_{i,t} - 2\big(r_{ij} P_{ij,t} + x_{ij} Q_{ij,t}\big) + \big(r_{ij}^2 + x_{ij}^2\big)\,\ell_{ij,t}\right]
$$

Para linhas, $\tau_{ij}=1$. Para transformadores, a impedância de dispersão é
representada no lado emissor e seguida pelo transformador ideal; as impedâncias
são convertidas para a base de potência do OPF usando a base nominal do trafo.

### Relaxação cônica (SOCP), $\forall (i,j) \in L,\ t \in T$

$$
P_{ij,t}^2 + Q_{ij,t}^2 \le v_{i,t}\,\ell_{ij,t}
$$

A qualidade da relaxação é verificada após a solução pelo resíduo

$$
g_{ij,t}=v_{i,t}\ell_{ij,t}-P_{ij,t}^2-Q_{ij,t}^2.
$$

O código reporta o maior valor absoluto e uma versão normalizada. O critério é
atendido quando o maior resíduo normalizado não supera a tolerância configurada.
A tolerância padrão é $10^{-5}$. Também são reportados o gap relativo ao fluxo,
o erro equivalente de corrente em ampères e o erro equivalente de perdas em
watts. A classificação é `tight`, `acceptable` ou `not_tight`; trata-se de um
diagnóstico numérico e físico, não de uma probabilidade estatística.

### Limites de rede (térmico e tensão)

$$
0 \le \ell_{ij,t} \le \overline{\ell}_{ij}, \qquad
\underline{v}_j^{\,2} \le v_{j,t} \le \overline{v}_j^{\,2}, \qquad
v_{r,t} = \big(v^{\text{ref}}\big)^2
$$

### BESS, $\forall s \in S,\ t \in T$

$$
E^{\text{BESS}}_{s,t} =
\begin{cases}
\text{SoC}^{\text{ini}}_s\, \overline{E}_s + \big(\eta^{\text{ch}}_s P^{\text{BESS,ch}}_{s,t} - P^{\text{BESS,dis}}_{s,t}/\eta^{\text{dis}}_s-P^{\text{loss,Q}}_{s,t}\big)\Delta t, & t = t_0\\[4pt]
E^{\text{BESS}}_{s,t-1} + \big(\eta^{\text{ch}}_s P^{\text{BESS,ch}}_{s,t} - P^{\text{BESS,dis}}_{s,t}/\eta^{\text{dis}}_s-P^{\text{loss,Q}}_{s,t}\big)\Delta t, & t > t_0
\end{cases}
$$

$$
\text{SoC}^{\min}_s\, \overline{E}_s \le E^{\text{BESS}}_{s,t} \le \text{SoC}^{\max}_s\, \overline{E}_s, \qquad
E^{\text{BESS}}_{s,\,t_{\text{end}}} = \text{SoC}^{\text{ini}}_s\, \overline{E}_s \ \text{(SoC cíclico)}
$$

$$
0 \le P^{\text{BESS,ch}}_{s,t} \le \overline{P}^{\text{ch}}_s, \qquad
0 \le P^{\text{BESS,dis}}_{s,t} \le \overline{P}^{\text{dis}}_s
$$

O inversor do BESS respeita:

$$
\left(P^{\text{BESS,dis}}_{s,t}-P^{\text{BESS,ch}}_{s,t}\right)^2+
\left(Q^{\text{BESS}}_{s,t}\right)^2\le
\left(\overline S^{\text{BESS}}_s\right)^2.
$$

A perda incremental causada por $Q$ é limitada por:

$$
\overline P^{\text{loss,Q}}_s\left(Q^{\text{BESS}}_{s,t}\right)^2
\leq
\left(\overline S^{\text{BESS}}_s\right)^2P^{\text{loss,Q}}_{s,t}.
$$

Logo, em $|Q|=\overline S^{\text{BESS}}_s$, a perda mínima é
$\overline P^{\text{loss,Q}}_s$. Essa perda drena a energia armazenada.

Com `reactive_control: true`, $Q^{\text{BESS}}_{s,t}$ é escolhido pelo OPF.
Quando a opção é omitida ou falsa, $Q^{\text{BESS}}_{s,t}=0$.

### PV, $\forall g \in G,\ t \in T$

$$
P^{\text{PV}}_{g,t}+P^{\text{loss,Q}}_{g,t}-P^{\text{PV,grid}}_{g,t}
\leq P^{\text{PV,av}}_{g,t}
\qquad(\text{igualdade se não-curtailável})
$$

$$
\left(P^{\text{PV}}_{g,t}\right)^2+
\left(Q^{\text{PV}}_{g,t}\right)^2\le
\left(\overline S^{\text{PV}}_g\right)^2.
$$

$$
\overline P^{\text{loss,Q}}_g\left(Q^{\text{PV}}_{g,t}\right)^2
\leq
\left(\overline S^{\text{PV}}_g\right)^2P^{\text{loss,Q}}_{g,t}.
$$

Durante o dia, a perda reativa consome parte da potência solar disponível. Por
padrão, $Q^{\text{PV}}_{g,t}=0$ quando
$P^{\text{PV,av}}_{g,t}=0$. Com `night_var: true`, o PV pode fornecer reativo
sem sol e:

$$
P^{\text{PV}}_{g,t}=0,\qquad
P^{\text{PV,grid}}_{g,t}=P^{\text{loss,Q}}_{g,t}.
$$

Assim, a perda noturna entra como consumo ativo na barra. O parâmetro
`q_loss_rated_kw` deve ser positivo quando `night_var` está ativo. A pequena
potência de perdas não é descontada do limite nominal em kVA para evitar má
condição numérica; o limite principal continua aplicado a $P^{\text{PV}}$ e
$Q^{\text{PV}}$.

As duas relações de perda são epígrafes convexas. Com tarifas de energia
positivas, minimizar o custo também minimiza $P^{\text{loss,Q}}$, fazendo as
desigualdades atuarem como igualdades na solução ótima.

No modo `fixed_pf`, $Q^{\text{PV}}_{g,t}=\tan\varphi_gP^{\text{PV}}_{g,t}$.
Os demais modos são `optimal`, `volt-var`, `volt-watt` e `volt-var-watt`.

### Rede / subestação, $\forall t \in T$

$$
0 \le P^{\text{grid,imp}}_{r,t} \le \overline{P}^{\text{imp}}, \qquad
0 \le P^{\text{grid,exp}}_{r,t} \le \overline{P}^{\text{exp}}, \qquad
\big|Q^{\text{grid}}_{r,t}\big| \le \overline{Q}^{\text{grid}}
$$

Não-simultaneidade import/export, sem binárias, via **SOS1** (no máximo um dos
dois é não-nulo em cada período):

$$
\text{SOS1}\big(\{\,P^{\text{grid,imp}}_{r,t},\ P^{\text{grid,exp}}_{r,t}\,\}\big), \qquad \forall t \in T
$$
