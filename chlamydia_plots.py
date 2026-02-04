"""
Chlamydia plots
"""

import numpy as np
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt
from chlamydia_models import Chlamydia_BL


def plot_bacterial_load_dynamics(disease_bl, n_trajectories=50):
    """
    Plot bacterial load dynamics and test sensitivity.
    
    Parameters:
    -----------
    disease_bl : Chlamydia_BL
        Bacterial load disease model (for accessing parameters)
    n_trajectories : int
        Number of infection trajectories to simulate
    """
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    scale = 1

    # =======================================================================
    # Bacterial load trajectories (theoretical)
    # =======================================================================
    ax = axes[0]
    
    p = disease_bl.pars
    
    # Sample parameter values from distributions
    for i in range(n_trajectories):
        # Sample half-life
        half_life = p.half_life.rvs(1)[0]
        
        # Calculate rates
        time_to_peak = p.time_to_peak.rvs(1)[0]
        growth_rate = np.log(p.peak_load / p.init_load) / time_to_peak
        decay_rate = np.log(2) / ss.years(half_life*disease_bl.t.dt_year).months        

        # Calculate clearance time
        decay_time = -np.log(p.init_load / p.peak_load) / decay_rate
        ti_clear = time_to_peak + decay_time
        
        # Create trajectory
        infection_times = np.linspace(0, ti_clear, 100)
        loads = []
        
        for t in infection_times:
            if t < time_to_peak:
                # Growth phase
                load = p.init_load * np.exp(growth_rate * t)
            else:
                # Decay phase
                load = p.peak_load * np.exp(-decay_rate * (t - time_to_peak))
            loads.append(load)
        
        # Convert to months
        time = infection_times * scale
        ax.plot(time, loads, alpha=0.3, color='steelblue', lw=0.8)
    
    # Add reference lines
    ax.axhline(p.init_load, color='red', ls='--', lw=2, 
              label=f'Initial load ({p.init_load:.0e})')
    ax.axhline(p.peak_load, color='green', ls='--', lw=2,
              label=f'Peak load ({p.peak_load:.0e})')
    ax.axvline(time_to_peak , color='orange', ls=':', lw=2, alpha=0.7,
              label=f'Time to peak (~{time_to_peak:.0f} days)')
    
    ax.set_xlabel('Weeks since infection', fontsize=12)
    ax.set_ylabel('Bacterial load (copies/mL)', fontsize=12)
    ax.set_yscale('log')
    ax.set_title('Bacterial Load Trajectories\n(sampled parameters)', 
                 weight='bold', fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([1e2, 1e8])
    
    # =======================================================================
    # Test sensitivity vs bacterial load
    # =======================================================================
    ax = axes[1]
    
    loads = np.logspace(1, 8, 200)
    
    # Basic test (flat sensitivity)
    basic_sens = np.full_like(loads, 0.85)
    
    # Load-sensitive test (Michaelis-Menten)
    max_sens = 0.95
    load_50 = 1e4
    min_sens = 0.15
    sensitive_sens = max_sens * loads / (load_50 + loads)
    sensitive_sens = np.maximum(sensitive_sens, min_sens)
    
    ax.plot(loads, basic_sens, 'b-', lw=3, label='Basic test (culture)', alpha=0.8)
    ax.plot(loads, sensitive_sens, 'g-', lw=3, label='Sensitive test (NAAT/PCR)')
    
    # Mark key points
    ax.axvline(load_50, color='gray', ls=':', lw=2, alpha=0.6, 
              label=f'K (50% max) = {load_50:.0e}')
    ax.axhline(0.5 * max_sens, color='gray', ls=':', lw=1.5, alpha=0.4)
    
    # Add annotations for key load points
    for load, label in [(1e3, '10³'), (1e4, '10⁴'), (1e5, '10⁵'), (1e6, '10⁶')]:
        sens = max_sens * load / (load_50 + load)
        sens = max(sens, min_sens)
        ax.plot(load, sens, 'go', ms=6, alpha=0.6)
    
    ax.set_xlabel('Bacterial load (copies/mL)', fontsize=12)
    ax.set_ylabel('Test sensitivity', fontsize=12)
    ax.set_xscale('log')
    ax.set_title('Sensitivity vs Bacterial Load\n(dose-response curve)', 
                 weight='bold', fontsize=12)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.3, which='both')
    ax.set_ylim([0, 1])
    ax.set_xlim([1e1, 1e8])
    
    # =======================================================================
    # P(positive) over infection time
    # =======================================================================
    ax = axes[2]
    
    # Use mean parameter values for typical trajectory
    time_to_peak = p.time_to_peak.pars['mean'].value
    half_life_mean = p.half_life.pars['mean'].value
    growth_rate = np.log(p.peak_load / p.init_load) / time_to_peak
    decay_rate = np.log(2) / ss.years(half_life_mean*0.25).months
    
    # Calculate clearance time
    decay_time = ss.months(-np.log(p.init_load / p.peak_load) / decay_rate)
    dur_inf = time_to_peak + decay_time*0.25 
    ti_clear = time_to_peak + dur_inf
    
    # Create time points
    infection_times = np.linspace(0, 50, 200)
    
    # Calculate bacterial loads
    loads = []
    for t in infection_times:
        if t < time_to_peak:
            load = p.init_load * np.exp(growth_rate * t)
        else:
            load = p.peak_load * np.exp(-decay_rate * (t - time_to_peak))
        loads.append(load)
    loads = np.array(loads)
    
    # Calculate p_positive for both test types
    p_pos_basic = np.full_like(loads, 0.85)
    p_pos_sensitive = max_sens * loads / (load_50 + loads)
    p_pos_sensitive = np.maximum(p_pos_sensitive, min_sens)
    
    # Convert to time units
    time = infection_times * scale
    
    # Plot
    ax.plot(time, p_pos_basic, 'b-', lw=3, label='Basic test (culture)', alpha=0.8)
    ax.plot(time, p_pos_sensitive, 'g-', lw=3, label='Sensitive test (NAAT/PCR)')
    
    # Mark peak
    ax.axvline(time_to_peak , color='orange', ls=':', lw=2, alpha=0.7,
              label=f'Peak load (~{time_to_peak:.0f} days)')
        
    # Mark key time points on sensitive test curve
    peak_idx = np.argmax(loads)
    ax.plot(time[peak_idx], p_pos_sensitive[peak_idx], 'go', ms=10, 
            label=f'Peak sensitivity ({p_pos_sensitive[peak_idx]:.2f})')
    
    ax.set_xlabel('Weeks since infection', fontsize=12)
    ax.set_ylabel('P(positive | infected)', fontsize=12)
    ax.set_title('Detection Probability Over Infection\n(test sensitivity dynamics)', 
                 weight='bold', fontsize=12)
    ax.legend(frameon=False, fontsize=9, loc='lower center')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    ax.set_xlim([0, time[-1]])
    
    plt.tight_layout()
    plt.savefig('bacterial_load_test_dynamics.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: bacterial_load_test_dynamics.png")
    plt.show()
    
    return fig


# ============================================================================
# Running experiments
# ============================================================================
if __name__ == '__main__':


    sc.heading("BACTERIAL LOAD & TEST SENSITIVITY DYNAMICS")
    
    # Create a disease instance just to access parameters
    disease_bl = Chlamydia_BL(
        beta=ss.peryear(0.5),
        init_prev=ss.bernoulli(p=0.05)
    )
    
    # Note: We need to initialize the disease to access parameter sampling
    # For plotting purposes, create a minimal sim
    sim_temp = ss.Sim(
        n_agents=100,
        diseases=disease_bl,
        networks='random'
    )
    sim_temp.init()
    
    # Now plot using the initialized disease
    fig = plot_bacterial_load_dynamics(sim_temp.diseases[0], n_trajectories=50)
    