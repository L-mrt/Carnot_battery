"""
Polynomial Regression Analysis - Carnot Battery
================================================
- COP_HP(T_hot)   : T_source = 60°C, T_hot in [90, 140] °C
- eta_ORC(T_hot)  : T_amb   = 25°C, T_hot in [90, 140] °C
- RTE(T_hot)      = COP_HP(T_hot) × eta_ORC(T_hot)
- d(RTE)/d(T_hot) via symbolic derivation of polynomial product
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from carnot_battery import CarnotBattery, Config

# ==============================================================================
# 1. Generate data — fine T_hot grid
# ==============================================================================

config = Config()

T_hot_range = np.arange(90, 141, 1)   # 90 → 140 °C inclusive, step 1°C
T_source_fixed = 60.0                  # °C
T_amb_fixed    = 25.0                  # °C

# COP depends only on T_source and T_hot → pick any T_amb (won't affect result)
# eta_ORC depends only on T_hot and T_amb → pick any T_source (won't affect result)

COP_data     = []
eta_ORC_data = []

print("Computing COP and eta_ORC over T_hot = 90 … 140 °C …")
for T_hot in T_hot_range:
    # COP — T_source=60, T_amb arbitrary
    b_cop = CarnotBattery(T_source=T_source_fixed, T_amb=T_amb_fixed,
                          T_hot=float(T_hot), config=config)
    cop, _ = b_cop.calc_hp_cycle()

    # eta_ORC — T_amb=25, T_source arbitrary
    b_orc = CarnotBattery(T_source=T_source_fixed, T_amb=T_amb_fixed,
                          T_hot=float(T_hot), config=config)
    eta, _ = b_orc.calc_orc_cycle()

    COP_data.append(cop)
    eta_ORC_data.append(eta)

COP_data     = np.array(COP_data)
eta_ORC_data = np.array(eta_ORC_data)
RTE_data      = COP_data * eta_ORC_data

print(f"  COP range      : {COP_data.min():.3f} – {COP_data.max():.3f}")
print(f"  eta_ORC range  : {eta_ORC_data.min():.4f} – {eta_ORC_data.max():.4f}")
print(f"  RTE range      : {RTE_data.min()*100:.2f}% – {RTE_data.max()*100:.2f}%")

# ==============================================================================
# 2. Polynomial regression
# ==============================================================================

# Centre T_hot for numerical stability
T_center = (T_hot_range.min() + T_hot_range.max()) / 2   # 115 °C
T_norm   = T_hot_range - T_center

# Degree selection: try 2 and 3, keep 3 (better fit)
DEG_COP = 3
DEG_ORC = 3

coeffs_cop = np.polyfit(T_norm, COP_data,     DEG_COP)
coeffs_orc = np.polyfit(T_norm, eta_ORC_data, DEG_ORC)

poly_cop = np.poly1d(coeffs_cop)
poly_orc = np.poly1d(coeffs_orc)

# Goodness of fit (R²)
def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - y_true.mean())**2)
    return 1 - ss_res / ss_tot

R2_cop = r2(COP_data,     poly_cop(T_norm))
R2_orc = r2(eta_ORC_data, poly_orc(T_norm))
print(f"\nPolynomial fit degree {DEG_COP}:")
print(f"  R² COP    = {R2_cop:.6f}")
print(f"  R² ORC    = {R2_orc:.6f}")

# ==============================================================================
# 3. Derivative of RTE = COP × eta_ORC
# ==============================================================================
# RTE(x)  = p_cop(x) × p_orc(x)   with x = T_hot - T_center
# d(RTE)/dx = p_cop'(x)×p_orc(x) + p_cop(x)×p_orc'(x)
# d(RTE)/d(T_hot) = d(RTE)/dx  (since dx/dT_hot = 1)

poly_rte        = poly_cop * poly_orc           # polynomial product
dpoly_cop       = poly_cop.deriv()              # p_cop'(x)
dpoly_orc       = poly_orc.deriv()              # p_orc'(x)
dpoly_rte       = dpoly_cop * poly_orc + poly_cop * dpoly_orc   # product rule

# Evaluate on fine grid
T_fine_norm  = np.linspace(T_norm.min(), T_norm.max(), 500)
T_fine       = T_fine_norm + T_center

COP_fit      = poly_cop(T_fine_norm)
eta_ORC_fit  = poly_orc(T_fine_norm)
RTE_fit      = COP_fit * eta_ORC_fit
dRTE_dThot   = dpoly_rte(T_fine_norm)

# Print symbolic expression in T_hot (substituted back)
print("\n" + "="*60)
print("Polynomial expressions  (x = T_hot − {:.0f}°C)".format(T_center))
print("="*60)
print(f"\nCOP_HP(x)  = {poly_cop}")
print(f"\neta_ORC(x) = {poly_orc}")
print(f"\nRTE(x)     = COP × eta_ORC =")
print(f"  {poly_rte}")
print(f"\nd(RTE)/d(T_hot) = d(RTE)/dx =")
print(f"  {dpoly_rte}")

# Sign check
print("\n" + "="*60)
print("Sign of d(RTE)/d(T_hot) over [90, 140] °C :")
print("="*60)
all_negative = np.all(dRTE_dThot < 0)
all_positive = np.all(dRTE_dThot > 0)
print(f"  Min : {dRTE_dThot.min():.4f}")
print(f"  Max : {dRTE_dThot.max():.4f}")
if all_negative:
    print("  → d(RTE)/d(T_hot) < 0 sur tout l'intervalle ✓")
elif all_positive:
    print("  → d(RTE)/d(T_hot) > 0 sur tout l'intervalle")
else:
    zero_crossings = T_fine[np.where(np.diff(np.sign(dRTE_dThot)))[0]]
    print(f"  → Changement de signe aux T_hot ≈ {zero_crossings} °C")

# ==============================================================================
# 4. Figure
# ==============================================================================

fig = plt.figure(figsize=(14, 10), dpi=120)
gs  = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.35)

# ── Panel 1 : COP ──────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.scatter(T_hot_range, COP_data, s=18, color='steelblue',
            zorder=5, label='Simulation')
ax1.plot(T_fine, COP_fit, color='steelblue', lw=2,
         label=f'Poly deg {DEG_COP}  (R²={R2_cop:.5f})')
ax1.set_xlabel('T_hot [°C]', fontsize=11)
ax1.set_ylabel('COP_HP [−]', fontsize=11)
ax1.set_title(f'COP — T_source = {T_source_fixed:.0f}°C', fontsize=12, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.35)

# ── Panel 2 : eta_ORC ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2.scatter(T_hot_range, eta_ORC_data * 100, s=18, color='darkorange',
            zorder=5, label='Simulation')
ax2.plot(T_fine, eta_ORC_fit * 100, color='darkorange', lw=2,
         label=f'Poly deg {DEG_ORC}  (R²={R2_orc:.5f})')
ax2.set_xlabel('T_hot [°C]', fontsize=11)
ax2.set_ylabel('η_ORC [%]', fontsize=11)
ax2.set_title(f'η_ORC — T_amb = {T_amb_fixed:.0f}°C', fontsize=12, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.35)

# ── Panel 3 : RTE ──────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(T_hot_range, RTE_data * 100, s=18, color='seagreen',
            zorder=5, label='Simulation')
ax3.plot(T_fine, RTE_fit * 100, color='seagreen', lw=2,
         label=f'Poly deg {DEG_COP+DEG_ORC}  (produit)')
ax3.set_xlabel('T_hot [°C]', fontsize=11)
ax3.set_ylabel('RTE [%]', fontsize=11)
ax3.set_title('RTE = COP × η_ORC', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(True, alpha=0.35)

# ── Panel 4 : d(RTE)/d(T_hot) ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(T_fine, dRTE_dThot, color='crimson', lw=2,
         label='d(RTE)/d(T_hot)  [1/°C]')
ax4.axhline(0, color='k', lw=1, ls='--')
ax4.fill_between(T_fine, dRTE_dThot, 0,
                 where=(dRTE_dThot < 0), alpha=0.15, color='crimson',
                 label='Zone négative')
ax4.fill_between(T_fine, dRTE_dThot, 0,
                 where=(dRTE_dThot >= 0), alpha=0.15, color='limegreen',
                 label='Zone positive')
ax4.set_xlabel('T_hot [°C]', fontsize=11)
ax4.set_ylabel('d(RTE)/d(T_hot)  [1/°C]', fontsize=11)
ax4.set_title('Dérivée de la RTE par rapport à T_hot', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.35)

plt.suptitle(
    f'Analyse polynomiale — T_source={T_source_fixed:.0f}°C  |  T_amb={T_amb_fixed:.0f}°C  |  T_hot ∈ [90, 140]°C',
    fontsize=13, fontweight='bold', y=1.01
)

plt.savefig('polynomial_analysis.png', dpi=120, bbox_inches='tight')
print("\n✓ Figure saved: polynomial_analysis.png")
plt.show()
