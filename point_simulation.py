"""
Carnot Battery - Fixed-Point Simulation  (matplotlib table display)
=====================================================================
Physical picture
  - A Heat Pump charges a HOT water tank at T_hot = 150 C
  - The ORC is driven by the hot tank (thermal battery at T_hot)
  - The COLD tank at T_cold = 60 C is the ORC condensation side (NOT a
    preheater limiting evaporation pressure)
  - An intermediate water loop links the tanks to the cycles

Operating conditions
  T_source = 60 C   (waste-heat source for HP evaporator)
  T_amb    = 24 C   (ambient, used only as reference here)
  T_hot    = 100 C  (hot storage tank -- ORC heat source)
  T_cold   = 60 C   (cold storage tank -- ORC heat sink)

Mass flows
  m_dot_water_storage = 1 kg/s  (hot-tank water -> ORC evaporator HX)
  m_dot_butane_ORC    = 1 kg/s  (ORC n-Butane, user-specified)
  m_dot_butane_HP     = derived (hot-tank steady-state balance)
  m_dot_water_source  = derived (shown for reference, 10 K drop)

Physical constraints  (identical to Config in carnot_battery.py)
  Working fluid : n-Butane (HEOS)
  dT_pinch      = 3 K
  Superheat     = 5 K
  Subcooling    = 5 K
  eta_comp      = 0.65
  eta_exp       = 0.75
  eta_pump      = 0.60

HP note: T_cond_sat = T_hot + dT_pinch = 153 C > n-Butane Tc (~152 C)
  --> HP runs TRANSCRITICAL.  High-side pressure = 1.1 x Pc.
      Gas cooler cools supercritical butane to T_hot - subcooling.
"""

import CoolProp.CoolProp as CP
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Fixed inputs
# ─────────────────────────────────────────────────────────────────────────────
FLUID = "HEOS::n-Butane"
WATER = "Water"

T_source = 60.0    # degC
T_amb    = 24.0    # degC
T_hot    = 100.0   # degC  (hot tank)
T_cold   = 60.0    # degC  (cold tank -- ORC condenser)

m_dot_water_storage = 1.0   # kg/s  (hot-tank water -> ORC evap HX)
m_dot_butane_ORC    = 1.0   # kg/s  (ORC butane)

dT_pp    = 3.0    # K  pinch
SH       = 5.0    # K  superheat
SC       = 5.0    # K  sub-cooling
eta_comp = 0.65
eta_exp  = 0.75
eta_pump = 0.60

# Fluid critical properties
T_crit = CP.PropsSI("Tcrit", FLUID) - 273.15   # degC  (~151.98)
P_crit = CP.PropsSI("Pcrit", FLUID)             # Pa    (~37.96 bar)

K   = lambda T: T + 273.15
bar = lambda P: P / 1e5


# =============================================================================
# ORC CYCLE  (discharging -- hot tank -> electricity)
#   Evaporation driven by T_hot; cold tank sets condensation temperature
# =============================================================================
T_evap_orc = T_hot  - dT_pp   # 147 C  (limited by hot tank + pinch)
T_cond_orc = T_cold + dT_pp   # 63 C   (limited by cold tank + pinch)

# State 1 -- condenser outlet (sub-cooled liquid at low pressure)
P_lo_orc = CP.PropsSI('P', 'T', K(T_cond_orc), 'Q', 1, FLUID)
T1_orc   = T_cond_orc - SC
h1_orc   = CP.PropsSI('H', 'T', K(T1_orc), 'P', P_lo_orc, FLUID)
s1_orc   = CP.PropsSI('S', 'T', K(T1_orc), 'P', P_lo_orc, FLUID)

# State 2 -- pump outlet (high pressure)
P_hi_orc = CP.PropsSI('P', 'T', K(T_evap_orc), 'Q', 1, FLUID)
h2s_orc  = CP.PropsSI('H', 'P', P_hi_orc, 'S', s1_orc, FLUID)
h2_orc   = h1_orc + (h2s_orc - h1_orc) / eta_pump
T2_orc   = CP.PropsSI('T', 'H', h2_orc, 'P', P_hi_orc, FLUID) - 273.15

# State 3 -- evaporator outlet (superheated vapor at T_evap+SH)
T3_orc   = T_evap_orc + SH
h3_orc   = CP.PropsSI('H', 'T', K(T3_orc), 'P', P_hi_orc, FLUID)
s3_orc   = CP.PropsSI('S', 'T', K(T3_orc), 'P', P_hi_orc, FLUID)

# State 4 -- expander outlet
h4s_orc  = CP.PropsSI('H', 'P', P_lo_orc, 'S', s3_orc, FLUID)
h4_orc   = h3_orc - eta_exp * (h3_orc - h4s_orc)
T4_orc   = CP.PropsSI('T', 'H', h4_orc, 'P', P_lo_orc, FLUID) - 273.15

# Energy balance (per kg butane)
w_pump_orc = h2_orc - h1_orc
q_evap_orc = h3_orc - h2_orc
w_exp_orc  = h3_orc - h4_orc
q_cond_orc = h4_orc - h1_orc
w_net_orc  = w_exp_orc - w_pump_orc
eta_ORC    = w_net_orc / q_evap_orc

# Scale to mass flow
Q_evap_orc = m_dot_butane_ORC * q_evap_orc    # W
W_exp_orc  = m_dot_butane_ORC * w_exp_orc     # W
W_pump_orc = m_dot_butane_ORC * w_pump_orc    # W
W_net_orc  = m_dot_butane_ORC * w_net_orc     # W
Q_cond_orc = m_dot_butane_ORC * q_cond_orc    # W

# Hot-tank water temperature drop
cp_w_hot    = CP.PropsSI('C', 'T', K(T_hot), 'P', 4e5, WATER)
dT_w_hot    = Q_evap_orc / (m_dot_water_storage * cp_w_hot)
T_w_hot_out = T_hot - dT_w_hot


# =============================================================================
# HP CYCLE  (charging -- source -> hot tank)
#   T_cond_sat = 153 C > Tc  --> TRANSCRITICAL
#   P_hi_hp = 1.1 x Pc  (supercritical gas-cooler pressure)
#   Butane cooled to T_hot - SC in the gas cooler
# =============================================================================
T_evap_hp = T_source - dT_pp   # 57 C  evaporation sat. temp
P_lo_hp   = CP.PropsSI('P', 'T', K(T_evap_hp), 'Q', 1, FLUID)
P_hi_hp   = 1.10 * P_crit       # supercritical high-side pressure

# State A1 -- evaporator outlet (superheated vapor)
T_A1  = T_evap_hp + SH
h_A1  = CP.PropsSI('H', 'T', K(T_A1), 'P', P_lo_hp, FLUID)
s_A1  = CP.PropsSI('S', 'T', K(T_A1), 'P', P_lo_hp, FLUID)

# State A2 -- compressor outlet (supercritical)
h_A2s = CP.PropsSI('H', 'P', P_hi_hp, 'S', s_A1, FLUID)
h_A2  = h_A1 + (h_A2s - h_A1) / eta_comp
T_A2  = CP.PropsSI('T', 'H', h_A2, 'P', P_hi_hp, FLUID) - 273.15

# State A3 -- gas-cooler outlet  (supercritical, cooled to T_hot - SC)
T_A3  = T_hot - SC    # 145 C  (supercritical butane exit of gas cooler)
h_A3  = CP.PropsSI('H', 'T', K(T_A3), 'P', P_hi_hp, FLUID)

# State A4 -- expansion valve outlet (isenthalpic)
h_A4  = h_A3
T_A4  = CP.PropsSI('T', 'H', h_A4, 'P', P_lo_hp, FLUID) - 273.15

# Energy balance per kg butane (HP)
w_comp_hp = h_A2 - h_A1
q_gc_hp   = h_A2 - h_A3    # J/kg heat released in gas cooler (to hot tank)
q_evap_hp = h_A1 - h_A4    # J/kg heat absorbed from source
COP_HP    = q_gc_hp / w_comp_hp

# Derive HP butane flow from hot-tank balance
m_dot_HP  = Q_evap_orc / q_gc_hp     # kg/s

W_comp_HP = m_dot_HP * w_comp_hp
Q_gc_HP   = m_dot_HP * q_gc_hp       # W  == Q_evap_orc
Q_evap_HP = m_dot_HP * q_evap_hp

# Source water info
cp_w_src            = CP.PropsSI('C', 'T', K(T_source), 'P', 1e5, WATER)
m_dot_src_for_10K   = Q_evap_HP / (10.0 * cp_w_src)
dT_src_at_1kgs      = Q_evap_HP / (1.0  * cp_w_src)


# =============================================================================
# Global
# =============================================================================
RTE = COP_HP * eta_ORC


# =============================================================================
# MATPLOTLIB DISPLAY
# =============================================================================
fig = plt.figure(figsize=(16, 11))
fig.patch.set_facecolor('#F0F4F8')

# ── colour palette ──
C_orc   = '#1A6FA8'   # blue
C_hp    = '#C0392B'   # red
C_sum   = '#1E8449'   # green
C_head  = '#2C3E50'   # dark header text
C_lblbg = '#D6EAF8'   # ORC header bg
C_lbl2  = '#FADBD8'   # HP header bg
C_lbl3  = '#D5F5E3'   # summary header bg
WHITE   = 'white'

title_fs = 10
cell_fs  = 9.5

# ── layout: 3 axes rows ──
ax_orc  = fig.add_axes([0.03, 0.62, 0.45, 0.30])   # ORC states
ax_hp   = fig.add_axes([0.52, 0.62, 0.45, 0.30])   # HP  states
ax_orc_en = fig.add_axes([0.03, 0.30, 0.45, 0.25]) # ORC energy breakdown
ax_hp_en  = fig.add_axes([0.52, 0.30, 0.45, 0.25]) # HP  energy breakdown
ax_sum  = fig.add_axes([0.03, 0.04, 0.94, 0.21])   # Global summary

for ax in [ax_orc, ax_hp, ax_orc_en, ax_hp_en, ax_sum]:
    ax.set_axis_off()


# ─── helper to draw a table with a coloured header ──────────────────────────
def draw_table(ax, col_labels, row_data, row_colors,
               title, title_color, header_bg, col_widths=None):
    ax.set_title(title, fontsize=title_fs, fontweight='bold',
                 color=title_color, pad=6)
    n_cols = len(col_labels)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols

    tbl = ax.table(
        cellText=row_data,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        colWidths=col_widths,
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(cell_fs)
    tbl.scale(1, 1.55)

    # Style header row
    for j in range(n_cols):
        cell = tbl[0, j]
        cell.set_facecolor(header_bg)
        cell.set_text_props(fontweight='bold', color=C_head)
        cell.set_edgecolor('#AAAAAA')

    # Style data rows
    for i, rc in enumerate(row_colors):
        for j in range(n_cols):
            cell = tbl[i + 1, j]
            cell.set_facecolor(rc)
            cell.set_edgecolor('#CCCCCC')

    return tbl


# ─── ORC state-point table ───────────────────────────────────────────────────
orc_cols  = ['State', 'Description', 'T [°C]', 'P [bar]', 'h [kJ/kg]', 's [kJ/kg·K]']
orc_rows  = [
    ['1', 'Cond outlet (liq)',   f'{T1_orc:.2f}',  f'{bar(P_lo_orc):.3f}', f'{h1_orc/1e3:.2f}', f'{s1_orc/1e3:.4f}'],
    ['2', 'Pump outlet (liq)',   f'{T2_orc:.2f}',  f'{bar(P_hi_orc):.3f}', f'{h2_orc/1e3:.2f}', '—'],
    ['3', 'Evap outlet (vap)',   f'{T3_orc:.2f}',  f'{bar(P_hi_orc):.3f}', f'{h3_orc/1e3:.2f}', f'{s3_orc/1e3:.4f}'],
    ['4', 'Exp outlet (wet/vap)',f'{T4_orc:.2f}',  f'{bar(P_lo_orc):.3f}', f'{h4_orc/1e3:.2f}', '—'],
]
row_colors_orc = ['#EBF5FB', '#D6EAF8', '#EBF5FB', '#D6EAF8']
draw_table(ax_orc, orc_cols, orc_rows, row_colors_orc,
           f'ORC — State Points  (T_evap_sat = {T_evap_orc:.0f} °C  |  T_cond_sat = {T_cond_orc:.0f} °C  |  PR = {P_hi_orc/P_lo_orc:.3f})',
           C_orc, C_lblbg,
           col_widths=[0.06, 0.28, 0.14, 0.14, 0.18, 0.20])

# ─── HP state-point table ────────────────────────────────────────────────────
hp_cols  = ['State', 'Description', 'T [°C]', 'P [bar]', 'h [kJ/kg]', 's [kJ/kg·K]']
hp_rows  = [
    ['A1', 'Evap outlet (SH vap)', f'{T_A1:.2f}',  f'{bar(P_lo_hp):.3f}', f'{h_A1/1e3:.2f}', f'{s_A1/1e3:.4f}'],
    ['A2', 'Comp outlet (SC vap)', f'{T_A2:.2f}',  f'{bar(P_hi_hp):.3f}', f'{h_A2/1e3:.2f}', '—'],
    ['A3', 'Gas-cooler out (SC)',   f'{T_A3:.2f}',  f'{bar(P_hi_hp):.3f}', f'{h_A3/1e3:.2f}', '—'],
    ['A4', 'Valve outlet (2ph)',    f'{T_A4:.2f}',  f'{bar(P_lo_hp):.3f}', f'{h_A4/1e3:.2f}', '—'],
]
row_colors_hp = ['#FDEDEC', '#FADBD8', '#FDEDEC', '#FADBD8']
draw_table(ax_hp, hp_cols, hp_rows, row_colors_hp,
           f'HP — State Points  *** TRANSCRITICAL ***  (P_hi = 1.1×Pc = {bar(P_hi_hp):.2f} bar  |  PR = {P_hi_hp/P_lo_hp:.3f})',
           C_hp, C_lbl2,
           col_widths=[0.07, 0.30, 0.14, 0.14, 0.18, 0.17])

# ─── ORC energy breakdown ────────────────────────────────────────────────────
orc_en_cols = ['Quantity', 'Per kg butane [kJ/kg]', f'Scaled  m={m_dot_butane_ORC} kg/s  [kW]']
orc_en_rows = [
    ['q_evap  (from hot tank)',  f'{q_evap_orc/1e3:+.3f}', f'{Q_evap_orc/1e3:+.3f}'],
    ['w_pump  (pump in)',        f'{w_pump_orc/1e3:+.3f}',  f'{W_pump_orc/1e3:+.3f}'],
    ['w_exp   (expander out)',   f'{w_exp_orc/1e3:+.3f}',   f'{W_exp_orc/1e3:+.3f}'],
    ['w_net   (net output)',     f'{w_net_orc/1e3:+.3f}',   f'{W_net_orc/1e3:+.3f}'],
    ['q_cond  (to cold tank)',   f'{q_cond_orc/1e3:+.3f}',  f'{Q_cond_orc/1e3:+.3f}'],
    [f'\u03b7_ORC',             f'{eta_ORC*100:.2f} %',    '—'],
]
rc_orc_en = ['#EBF5FB']*6
rc_orc_en[-1] = '#AED6F1'
draw_table(ax_orc_en, orc_en_cols, orc_en_rows, rc_orc_en,
           f'ORC — Energy Balance    Hot-tank water: {T_hot:.0f} °C → {T_w_hot_out:.1f} °C  (ΔT = {dT_w_hot:.1f} K)',
           C_orc, C_lblbg)

# ─── HP energy breakdown ─────────────────────────────────────────────────────
hp_en_cols = ['Quantity', 'Per kg butane [kJ/kg]', f'Scaled  m={m_dot_HP:.4f} kg/s  [kW]']
hp_en_rows = [
    ['q_evap  (from source)',    f'{q_evap_hp/1e3:+.3f}',  f'{Q_evap_HP/1e3:+.3f}'],
    ['w_comp  (compressor in)',  f'{w_comp_hp/1e3:+.3f}',  f'{W_comp_HP/1e3:+.3f}'],
    ['q_gc    (to hot tank)',    f'{q_gc_hp/1e3:+.3f}',    f'{Q_gc_HP/1e3:+.3f}'],
    ['COP_HP  (= q_gc/w_comp)', f'{COP_HP:.4f}',           '—'],
    ['m_dot HP derived',        '—',                        f'{m_dot_HP:.4f} kg/s'],
    ['Source water  dT@1 kg/s', '—',                        f'{dT_src_at_1kgs:.2f} K'],
]
rc_hp_en    = ['#FDEDEC']*6
rc_hp_en[3] = '#F1948A'
draw_table(ax_hp_en, hp_en_cols, hp_en_rows, rc_hp_en,
           f'HP — Energy Balance  (gas-cooler exit = {T_A3:.0f} °C  |  n-Butane Tc = {T_crit:.1f} °C)',
           C_hp, C_lbl2)

# ─── Global summary ──────────────────────────────────────────────────────────
sum_cols = [
    'COP_HP', 'η_ORC [%]', 'RTE [%]',
    'W_net ORC [kW]', 'W_comp HP [kW]',
    'Q_evap HP [kW]', 'Q_cond ORC [kW]',
    'PR_HP', 'PR_ORC',
]
sum_row = [[
    f'{COP_HP:.4f}',
    f'{eta_ORC*100:.2f}',
    f'{RTE*100:.2f}',
    f'{W_net_orc/1e3:.3f}',
    f'{W_comp_HP/1e3:.3f}',
    f'{Q_evap_HP/1e3:.3f}',
    f'{Q_cond_orc/1e3:.3f}',
    f'{P_hi_hp/P_lo_hp:.3f}',
    f'{P_hi_orc/P_lo_orc:.3f}',
]]
draw_table(ax_sum, sum_cols, sum_row, ['#D5F5E3'],
           'GLOBAL PERFORMANCE SUMMARY',
           C_sum, C_lbl3)

# ─── operating conditions banner ─────────────────────────────────────────────
cond_text = (
    f"Operating conditions:  "
    f"T_hot = {T_hot:.0f} °C  |  T_cold = {T_cold:.0f} °C  |  "
    f"T_source = {T_source:.0f} °C  |  T_amb = {T_amb:.0f} °C  ||  "
    f"m_dot_water_storage = {m_dot_water_storage} kg/s  |  "
    f"m_dot_butane_ORC = {m_dot_butane_ORC} kg/s  |  "
    f"ΔT_pinch = {dT_pp} K  |  SH = {SH} K  |  SC = {SC} K  |  "
    f"η_comp = {eta_comp}  |  η_exp = {eta_exp}  |  η_pump = {eta_pump}"
)
fig.text(0.5, 0.975, 'Carnot Battery — Fixed-Point Simulation',
         ha='center', va='top', fontsize=14, fontweight='bold', color=C_head)
fig.text(0.5, 0.950, cond_text,
         ha='center', va='top', fontsize=8, color='#555555',
         style='italic', wrap=True)

# HP transcritical note
note = (
    "⚠  HP is TRANSCRITICAL  (T_cond_sat = 153 °C > n-Butane Tc ≈ 151.98 °C).\n"
    f"   P_hi_HP = 1.1 × Pc = {bar(P_hi_hp):.2f} bar.  "
    f"Gas-cooler cools butane from {T_A2:.1f} °C down to {T_A3:.0f} °C  "
    f"(T_hot − subcooling = {T_hot:.0f} − {SC:.0f} K)."
)
fig.text(0.52, 0.595, note, ha='left', va='top', fontsize=8,
         color=C_hp, style='italic',
         bbox=dict(boxstyle='round,pad=0.4', fc='#FDEDEC', ec=C_hp, alpha=0.85))

plt.savefig('point_simulation_results.png', dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
print("Figure saved as point_simulation_results.png")
plt.show()
