"""
Carnot Battery Simulation (PTES) - Full Factorial Sensitivity Analysis
Steady-state model with heat pump, thermal storage, and ORC cycle
Based on Laterre et al., 2024

System description:
- Heat Pump (charging)  : absorbs heat from T_source, rejects it at T_hot
- ORC (discharging)     : takes heat from T_hot, rejects it at T_amb
- Full 3D factorial sweep (T_hot, T_amb, T_source)
- Pressure ratio tracking
"""

import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, Tuple, Optional
import itertools
import warnings
warnings.filterwarnings('ignore')

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")


# ==============================================================================
# Configuration
# ==============================================================================

class Config:
    """Configuration for Carnot Battery simulation"""
    
    # Working fluids
    WORKING_FLUID = "HEOS::n-Butane"
    STORAGE_FLUID = "Water"
    
    # Nominal temperatures [°C] (for single-point tests)
    T_SOURCE_IN = 24.0
    T_SINK_IN = 15.0
    T_HOT_TANK = 150.0
    
    # Storage pressure [Pa]
    P_STORAGE = 7.5e5
    
    # Thermodynamic parameters [K]
    DELTA_T_PINCH = 3.0
    SUPERHEAT = 5.0
    SUBCOOLING = 5.0
    
    # Isentropic efficiencies [-]
    ETA_COMPRESSOR = 0.65
    ETA_EXPANDER = 0.75
    ETA_PUMP = 0.60
    
    # Full factorial sweep ranges (for main analysis)
    T_HOT_SWEEP = np.arange(90, 151, 10)        # [90, 100, ..., 150]
    # Sweep ambient from -10 to 40 °C in 5 °C steps
    T_AMB_SWEEP = np.arange(-10, 41, 5)
    T_SOURCE_SWEEP = np.array([24, 60])
    
    # Output files
    EXCEL_FILENAME = "sensitivity_results.xlsx"
    PLOT_FILENAME = "3D_sensitivity.png"
    FIGURE_DPI = 120
    # Control automatic saving of generated figures (set False to disable)
    SAVE_PLOTS = True


# ==============================================================================
# Carnot Battery Model
# ==============================================================================

class CarnotBattery:
    """
    Complete thermodynamic model of a Carnot Battery (PTES)

    Heat Pump (charging) : T_source → T_hot
    ORC (discharging)    : T_hot → T_amb
    """
    
    def __init__(self, T_source: float, T_amb: float, T_hot: float,
                 config: Config = None):
        """
        Initialize Carnot Battery with operating temperatures
        
        Args:
            T_source: Waste heat source temperature for HP evaporator [°C]
            T_amb: Ambient sink temperature for ORC condenser [°C]
            T_hot: Hot tank temperature [°C]
            config: Configuration object
        """
        if config is None:
            config = Config()
        
        self.working_fluid = config.WORKING_FLUID
        self.T_source = T_source
        self.T_amb = T_amb
        self.T_hot = T_hot
        
        self.delta_T_pp = config.DELTA_T_PINCH
        self.superheat = config.SUPERHEAT
        self.subcooling = config.SUBCOOLING
        
        self.eta_comp = config.ETA_COMPRESSOR
        self.eta_exp = config.ETA_EXPANDER
        self.eta_pump = config.ETA_PUMP
        
        # Store pressure ratios for output
        self.PR_HP = np.nan
        self.PR_ORC = np.nan
        # Per-kg energy terms (J/kg)
        self.q_cond_per_kg = np.nan
        self.w_comp_per_kg = np.nan
        self.q_evap_per_kg = np.nan
        self.w_pump_per_kg = np.nan
        self.w_exp_per_kg = np.nan
        self.w_net_per_kg = np.nan
    
    def calc_hp_cycle(self) -> Tuple[float, float]:
        """
        Calculate heat pump cycle (charging)
        
        Returns:
            (COP, pressure_ratio)
        """
        # Saturation temperatures
        T_evap_sat = self.T_source - self.delta_T_pp
        T_cond_sat = self.T_hot + self.delta_T_pp
        
        # Point 1: Evaporator outlet (superheated vapor)
        T1_K = (T_evap_sat + self.superheat) + 273.15
        P1 = CP.PropsSI('P', 'T', T_evap_sat + 273.15, 'Q', 1, self.working_fluid)
        h1 = CP.PropsSI('H', 'T', T1_K, 'P', P1, self.working_fluid)
        s1 = CP.PropsSI('S', 'T', T1_K, 'P', P1, self.working_fluid)
        
        # Point 2: Compressor outlet
        P2 = CP.PropsSI('P', 'T', T_cond_sat + 273.15, 'Q', 1, self.working_fluid)
        h2s = CP.PropsSI('H', 'P', P2, 'S', s1, self.working_fluid)
        h2 = h1 + (h2s - h1) / self.eta_comp
        
        # Point 3: Condenser outlet (subcooled liquid)
        T3_K = (T_cond_sat - self.subcooling) + 273.15
        h3 = CP.PropsSI('H', 'T', T3_K, 'P', P2, self.working_fluid)
        
        # Point 4: After expansion (isenthalpic)
        h4 = h3
        
        # Energy balance
        w_comp = h2 - h1
        q_cond = h2 - h3
        COP = q_cond / w_comp
        
        # Store pressure ratio
        self.PR_HP = P2 / P1

        # Store per-kg energy terms (J/kg)
        self.q_cond_per_kg = q_cond
        self.w_comp_per_kg = w_comp
        
        return COP, self.PR_HP
    
    def calc_orc_cycle(self) -> Tuple[float, float]:
        """
        Calculate ORC cycle (discharging)

        Heat source : hot tank at T_hot
        Heat sink   : ambient at T_amb
        
        Returns:
            (eta_ORC, pressure_ratio)
        """
        # Evaporator saturation temperature limited by hot tank pinch
        T_evap_sat = self.T_hot - self.delta_T_pp
        T_cond_sat = self.T_amb + self.delta_T_pp
        
        # Point 1: Condenser outlet (subcooled liquid)
        T1_K = (T_cond_sat - self.subcooling) + 273.15
        P1 = CP.PropsSI('P', 'T', T_cond_sat + 273.15, 'Q', 1, self.working_fluid)
        h1 = CP.PropsSI('H', 'T', T1_K, 'P', P1, self.working_fluid)
        s1 = CP.PropsSI('S', 'T', T1_K, 'P', P1, self.working_fluid)
        
        # Point 2: Pump outlet (high pressure side)
        P2 = CP.PropsSI('P', 'T', T_evap_sat + 273.15, 'Q', 1, self.working_fluid)
        h2s = CP.PropsSI('H', 'P', P2, 'S', s1, self.working_fluid)
        h2 = h1 + (h2s - h1) / self.eta_pump
        
        # Point 3: Evaporator outlet (superheated vapor)
        T3_K = (T_evap_sat + self.superheat) + 273.15
        h3 = CP.PropsSI('H', 'T', T3_K, 'P', P2, self.working_fluid)
        s3 = CP.PropsSI('S', 'T', T3_K, 'P', P2, self.working_fluid)
        
        # Point 4: Expander outlet
        h4s = CP.PropsSI('H', 'P', P1, 'S', s3, self.working_fluid)
        h4 = h3 - self.eta_exp * (h3 - h4s)
        
        # Energy balance
        w_pump = h2 - h1
        q_evap = h3 - h2
        w_exp = h3 - h4
        w_net = w_exp - w_pump
        eta_ORC = w_net / q_evap
        
        # Store pressure ratio
        self.PR_ORC = P2 / P1

        # Store per-kg energy terms (J/kg)
        self.q_evap_per_kg = q_evap
        self.w_pump_per_kg = w_pump
        self.w_exp_per_kg = w_exp
        self.w_net_per_kg = w_net
        
        return eta_ORC, self.PR_ORC

    def compute_mass_flow(self, Q_dot_cond: Optional[float] = None, W_dot_net: Optional[float] = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Estimate mass flow rates from requested power values.

        Args:
            Q_dot_cond: thermal power at the condenser side [W] (optional)
            W_dot_net: net mechanical/electrical power from ORC [W] (optional)

        Returns:
            (m_dot_cond, m_dot_orc) in kg/s (None if not computable or not requested)

        Notes:
            - This method relies on per-kg energy terms computed by
              `calc_hp_cycle()` and `calc_orc_cycle()` (or by calling
              `compute_rte()` beforehand). Enthalpy differences are in J/kg,
              so dividing a power (W = J/s) by J/kg yields kg/s.
        """
        # Ensure per-kg terms are computed
        try:
            if np.isnan(self.q_cond_per_kg) or np.isnan(self.w_net_per_kg):
                # Attempt to compute cycles to populate per-kg values
                self.calc_hp_cycle()
                self.calc_orc_cycle()
        except Exception:
            pass

        m_dot_cond = None
        m_dot_orc = None

        if Q_dot_cond is not None:
            if np.isfinite(self.q_cond_per_kg) and abs(self.q_cond_per_kg) > 1e-12:
                m_dot_cond = Q_dot_cond / self.q_cond_per_kg
            else:
                m_dot_cond = None

        if W_dot_net is not None:
            if np.isfinite(self.w_net_per_kg) and abs(self.w_net_per_kg) > 1e-12:
                m_dot_orc = W_dot_net / self.w_net_per_kg
            else:
                m_dot_orc = None

        return m_dot_cond, m_dot_orc
    
    def compute_rte(self) -> Tuple[float, float, float, float, float]:
        """
        Compute round trip efficiency
        
        Returns:
            (COP, eta_ORC, RTE, PR_HP, PR_ORC)
        """
        try:
            COP, PR_HP = self.calc_hp_cycle()
            eta_ORC, PR_ORC = self.calc_orc_cycle()
            RTE = COP * eta_ORC
            return COP, eta_ORC, RTE, PR_HP, PR_ORC
        except Exception as e:
            # Return NaN if computation fails (invalid operating point)
            return np.nan, np.nan, np.nan, np.nan, np.nan


# ==============================================================================
# Full Factorial Sweep
# ==============================================================================

def run_full_factorial_sweep(config: Config = None) -> pd.DataFrame:
    """
    Run full 3D factorial sweep  (T_hot × T_amb × T_source)
    
    Returns:
        DataFrame with all combinations and results
    """
    if config is None:
        config = Config()
    
    # Generate all combinations
    combinations = list(itertools.product(
        config.T_HOT_SWEEP,
        config.T_AMB_SWEEP,
        config.T_SOURCE_SWEEP
    ))
    
    n_total = len(combinations)
    print(f"\n{'='*70}")
    print(f"FULL FACTORIAL SENSITIVITY ANALYSIS")
    print(f"{'='*70}")
    print(f"Total combinations: {n_total}")
    print(f"  T_hot:    {len(config.T_HOT_SWEEP)} values")
    print(f"  T_amb:    {len(config.T_AMB_SWEEP)} values")
    print(f"  T_source: {len(config.T_SOURCE_SWEEP)} values")
    print(f"{'='*70}\n")
    
    results = []
    
    # Use tqdm if available
    iterator = tqdm(combinations, desc="Computing") if HAS_TQDM else combinations
    
    for T_hot, T_amb, T_source in iterator:
        battery = CarnotBattery(T_source, T_amb, T_hot, config)
        COP, eta_ORC, RTE, PR_HP, PR_ORC = battery.compute_rte()

        results.append({
            'T_hot': T_hot,
            'T_amb': T_amb,
            'T_source': T_source,
            'COP_HP': COP,
            'Eta_ORC': eta_ORC,
            'RTE': RTE,
            'Pressure_Ratio_HP': PR_HP,
            'Pressure_Ratio_ORC': PR_ORC
        })
    
    df = pd.DataFrame(results)

    # Keep NaNs to preserve identical grid count per T_source. Report valid points.
    n_total = len(df)
    n_valid = df['RTE'].dropna().shape[0]
    n_invalid = n_total - n_valid
    if n_invalid > 0:
        print(f"\nNote: {n_invalid} points failed or are physically invalid (NaNs)")

    print(f"\n✓ Computed {n_valid} valid operating points out of {n_total} attempted\n")

    return df


def export_results(df: pd.DataFrame, config: Config):
    """Export results to Excel"""
    print(f"{'='*70}")
    print(f"EXCEL EXPORT")
    print(f"{'='*70}")
    
    try:
        with pd.ExcelWriter(config.EXCEL_FILENAME, engine='openpyxl') as writer:
            # Main data sheet
            df.to_excel(writer, sheet_name='Full_Factorial_Results', index=False)
            
            # Summary statistics
            summary = pd.DataFrame({
                'Metric': ['RTE', 'COP_HP', 'Eta_ORC'],
                'Min': [df['RTE'].min(), df['COP_HP'].min(), df['Eta_ORC'].min()],
                'Max': [df['RTE'].max(), df['COP_HP'].max(), df['Eta_ORC'].max()],
                'Mean': [df['RTE'].mean(), df['COP_HP'].mean(), df['Eta_ORC'].mean()],
                'Std': [df['RTE'].std(), df['COP_HP'].std(), df['Eta_ORC'].std()]
            })
            summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
        
        print(f"✓ Results saved to: {config.EXCEL_FILENAME}")
        print(f"  - Sheet 1: Full_Factorial_Results ({len(df)} rows)")
        print(f"  - Sheet 2: Summary_Statistics")
        print(f"{'='*70}\n")
    
    except Exception as e:
        print(f"✗ Excel export error: {e}\n")


def plot_3d_results(df: pd.DataFrame, config: Config):
    """
    Generate 3D plots (one per T_source)
    X=T_hot, Y=T_amb, Z=RTE, color=RTE
    """
    print(f"{'='*70}")
    print(f"3D VISUALIZATION")
    print(f"{'='*70}")
    
    T_source_values = sorted(df['T_source'].unique())
    n_plots = len(T_source_values)
    
    fig = plt.figure(figsize=(7*n_plots, 6), dpi=config.FIGURE_DPI)
    
    for idx, T_src in enumerate(T_source_values):
        ax = fig.add_subplot(1, n_plots, idx+1, projection='3d')
        
        # All attempted points for this T_source
        data_all = df[df['T_source'] == T_src]
        data_valid = data_all.dropna(subset=['RTE'])
        # Exclude negative / zero RTE values
        n_before = len(data_valid)
        data_valid = data_valid[data_valid['RTE'] > 0]
        n_omitted = n_before - len(data_valid)
        if n_omitted > 0:
            print(f"Note: {n_omitted} non-positive RTE points omitted from 3D plot for T_source={T_src}°C")

        # Plot valid points (colored by RTE value)
        sc = ax.scatter(
            data_valid['T_hot'],
            data_valid['T_amb'],
            data_valid['RTE'] * 100,  # Convert to percentage
            c=data_valid['RTE'] * 100,
            cmap='viridis',
            s=35,
            alpha=0.85,
            edgecolors='k',
            linewidths=0.3
        )
        
        # Labels
        ax.set_xlabel('T_hot [°C]', fontsize=10, labelpad=8)
        ax.set_ylabel('T_amb [°C]', fontsize=10, labelpad=8)
        ax.set_zlabel('RTE [%]', fontsize=10, labelpad=8)
        ax.set_title(f'T_source = {T_src:.0f}°C', fontsize=12, fontweight='bold', pad=15)
        
        # Colorbar
        cbar = plt.colorbar(sc, ax=ax, pad=0.1, shrink=0.8)
        cbar.set_label('RTE [%]', fontsize=9)
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # View angle
        ax.view_init(elev=20, azim=45)
    
    plt.suptitle('Round Trip Efficiency - Sensitivity Analysis (T_hot × T_amb × T_source)', 
                 fontsize=13, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    if config.SAVE_PLOTS:
        plt.savefig(config.PLOT_FILENAME, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ 3D plots saved to: {config.PLOT_FILENAME}")
    else:
        print("(Saving of 3D results plot disabled)")
    print(f"{'='*70}\n")
    
    plt.show()


def plot_2d_rte_vs_thot(df: pd.DataFrame, config: Config,
                        T_source_fixed: float = 60.0,
                        T_amb_fixed: float = 25.0):
    """
    2D plot: RTE [%] vs T_hot
    Fixed conditions: T_source = T_source_fixed, T_amb = T_amb_fixed
    """
    print(f"{'='*70}")
    print(f"2D VISUALIZATION  (T_source={T_source_fixed}°C, T_amb={T_amb_fixed}°C)")
    print(f"{'='*70}")

    # Filter the DataFrame
    mask = (
        np.isclose(df['T_source'], T_source_fixed) &
        np.isclose(df['T_amb'], T_amb_fixed)
    )
    data = df[mask].dropna(subset=['RTE'])
    data = data[data['RTE'] > 0].sort_values('T_hot')

    if data.empty:
        print(f"✗ No valid (RTE > 0) data found for T_source={T_source_fixed}°C, T_amb={T_amb_fixed}°C")
        print(f"  Available T_source values: {sorted(df['T_source'].unique())}")
        print(f"  Available T_amb values:    {sorted(df['T_amb'].unique())}")
        return

    fig, ax = plt.subplots(figsize=(8, 5), dpi=config.FIGURE_DPI)

    ax.plot(
        data['T_hot'],
        data['RTE'] * 100,
        marker='o',
        markersize=6,
        linewidth=2,
        color='steelblue'
    )

    ax.set_xlabel('T_hot [°C]', fontsize=12)
    ax.set_ylabel('RTE [%]', fontsize=12)
    ax.set_title(
        f'Round Trip Efficiency vs Hot Tank Temperature\n'
        f'T_source = {T_source_fixed:.0f}°C  |  T_amb = {T_amb_fixed:.0f}°C',
        fontsize=13, fontweight='bold'
    )
    ax.grid(True, alpha=0.4)
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    if config.SAVE_PLOTS:
        fname = f'2D_RTE_vs_Thot_Tsrc{T_source_fixed:.0f}_Tamb{T_amb_fixed:.0f}.png'
        plt.savefig(fname, dpi=config.FIGURE_DPI, bbox_inches='tight')
        print(f"✓ 2D plot saved to: {fname}")
    else:
        print("(Saving of 2D results plot disabled)")
    print(f"{'='*70}\n")

    plt.show()


# ==============================================================================
# Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    """
    Main workflow:
    1. Run full factorial sweep
    2. Export to Excel
    3. Generate 3D plots
    """
    
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print(f"#{'CARNOT BATTERY - FULL FACTORIAL ANALYSIS':^68}#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    # Configuration
    config = Config()
    
    # Run sweep
    df_results = run_full_factorial_sweep(config)
    
    # Export
    export_results(df_results, config)
    
    # Visualize
    plot_3d_results(df_results, config)
    plot_2d_rte_vs_thot(df_results, config, T_source_fixed=60.0, T_amb_fixed=25.0)

    # Final summary
    print("\n" + "#"*70)
    print(f"#{'ANALYSIS COMPLETE':^68}#")
    print("#"*70)
    valid = df_results[df_results['RTE'] > 0]
    print(f"\nKey findings:")
    print(f"  Total computed points : {len(df_results)}")
    print(f"  Valid (RTE > 0) points: {len(valid)}")
    print(f"  RTE range             : {valid['RTE'].min()*100:.2f}% to {valid['RTE'].max()*100:.2f}%")
    print(f"  Mean RTE (valid)      : {valid['RTE'].mean()*100:.2f}%")
    print(f"  Best operating point  :")
    best_idx = valid['RTE'].idxmax()
    best_row = valid.loc[best_idx]
    print(f"    T_hot={best_row['T_hot']:.0f}°C, T_amb={best_row['T_amb']:.0f}°C, "
          f"T_source={best_row['T_source']:.0f}°C")
    print(f"    RTE = {best_row['RTE']*100:.2f}%")
    print(f"\n" + "#"*70 + "\n")
