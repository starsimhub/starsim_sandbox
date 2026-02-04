"""
Chlamydia models with diagnostics.
Design experiments to work with situations where there may be multiple instances of a 
disease model that get used with a single intervention.
"""


import numpy as np
import starsim as ss
import sciris as sc
import matplotlib.pyplot as plt


# ============================================================================
# Testing products
# ============================================================================
class STIDx(ss.Product):
    """
    Base STI diagnostic product with binary outcomes.
    
    This product owns the p_positive state (probability of testing positive)
    and calculates it based on disease states it queries. Different subclasses
    implement different sensitivity curves.
    
    The product is instantiated with a disease reference and then
    queries that disease's states to determine test outcomes.
    """
    
    def __init__(self, disease, pars=None, **kwargs):
        super().__init__()
        self.disease = disease
        self.define_pars(
            base_sensitivity = 0.85,
            base_specificity = 0.99,
        )
        self.update_pars(pars, **kwargs)
        self.define_states(
            ss.FloatArr('p_positive')
        )
        self.result_list = ['positive', 'negative']
        self._p_positive = ss.bernoulli(p=0)  # Set below
        
        return
    
    def calculate_p_positive(self, uids):
        """
        Calculate probability of testing positive for each UID.
        
        Base implementation: simple binary (infected vs susceptible).
        Subclasses override to implement more complex logic.
        """
        disease = self.sim.diseases[self.disease]
        p_pos = np.zeros(len(uids))
        
        # Infected: sensitivity
        infected = disease.infected[uids]
        p_pos[infected] = self.pars.base_sensitivity
        
        # Susceptible: false positive rate
        p_pos[~infected] = 1 - self.pars.base_specificity
        
        return p_pos
    
    def administer(self, sim, uids):
        """
        Administer test and return outcomes.
        
        Product calculates probability of positive test based on disease states,
        then samples outcomes.
        """
        p_pos = self.calculate_p_positive(uids)
        self._p_positive.set(p_pos)
        pos, neg = self._p_positive.split(uids)
        
        # Package outcomes
        outcomes = {r: ss.uids() for r in self.result_list}
        outcomes['positive'] = pos
        outcomes['negative'] = neg
        
        return outcomes


class STIDx_BacterialLoad(STIDx):
    """
    STI diagnostic with bacterial-load-dependent sensitivity.
    
    This product has enhanced sensitivity that depends on bacterial load.
    It queries the disease's bacterial_load state and uses a dose-response
    curve to determine test sensitivity.
    
    This represents a more sensitive molecular test (e.g., PCR/NAAT) that
    can detect lower bacterial loads than traditional culture methods.
    
    Dose-Response Curve (Michaelis-Menten):
    - Load < 100: ~15% sensitivity
    - Load ~1,000: ~35% sensitivity  
    - Load ~10,000: ~68% sensitivity (50% of max at load_50)
    - Load ~100,000: ~92% sensitivity
    - Load > 1,000,000: ~95% sensitivity
    
    Parameters:
    -----------
    max_sensitivity : float
        Maximum achievable sensitivity (at very high loads)
    load_50 : float
        Bacterial load at which sensitivity = 50% of maximum
    min_sensitivity : float
        Minimum sensitivity (even at very low loads)
    """
    
    def __init__(self, disease, pars=None, **kwargs):
        super().__init__(disease=disease)
        self.define_pars(
            max_sensitivity = 0.95,
            load_50 = 1e4,
            min_sensitivity = 0.15,
        )
        return
    
    def calculate_p_positive(self, uids):
        """
        Calculate load-dependent probability of testing positive.
        
        Queries bacterial_load from disease and applies dose-response curve.
        Falls back to simple binary if bacterial_load not available.
        """
        disease = self.sim.diseases[self.disease]
        p_pos = np.zeros(len(uids))
        
        # Check if disease tracks bacterial load
        if hasattr(disease, 'bacterial_load'):
            # Get infection status and loads
            infected = disease.infected[uids]
            
            # Susceptible: false positive rate
            p_pos[~infected] = 1 - self.pars.base_specificity
            
            # Infected: load-dependent sensitivity
            if infected.any():
                loads = disease.bacterial_load[uids[infected]]
                
                # Michaelis-Menten: sensitivity = max * load / (K + load)
                sensitivity = self.pars.max_sensitivity * loads / (self.pars.load_50 + loads)
                
                # Enforce minimum sensitivity
                sensitivity = np.maximum(sensitivity, self.pars.min_sensitivity)
                
                p_pos[infected] = sensitivity
        else:
            # Fallback: simple binary sensitivity
            infected = disease.infected[uids]
            p_pos[infected] = self.pars.max_sensitivity
            p_pos[~infected] = 1 - self.pars.base_specificity
        
        return p_pos


# ============================================================================
# Testing delivery
# ============================================================================
class STITest(ss.Intervention):
    """
    STI testing intervention.
    
    Uses a product to administer tests. Product handles all sensitivity logic.
    """
    
    def __init__(self, name=None, product=None, pars=None, start_year=None, eligibility=None, **kwargs):
        super().__init__(name=name)
        self.define_pars(
            test_prob = ss.bernoulli(0.2),
        )
        self.start_year = start_year
        self.eligibility = eligibility
        self.product = product
        
        # Initialize states
        self.define_states(
            ss.BoolState('ever_tested'),
            ss.BoolState('diagnosed'),
            ss.FloatArr('ti_tested'),
            ss.FloatArr('ti_positive'),
            ss.FloatArr('ti_negative'),
            ss.FloatArr('n_tests', default=0),
        )
        
        # Track outcomes
        if self.product is not None:
            self.outcomes = {outcome: ss.uids() for outcome in self.product.result_list}

        return
    
    def init_pre(self, sim):
        super().init_pre(sim)
        if self.start_year is None:
            self.start_year = sim.pars.start
        return
    
    def init_results(self):
        super().init_results()
        self.define_results(
            ss.Result('new_tests', dtype=int, label='New tests'),
            ss.Result('new_positives', dtype=int, label='New positive tests'),
            ss.Result('new_diagnoses', dtype=int, label='New diagnoses'),
        )
        return
    
    def step(self):
        """Apply testing intervention"""
        sim = self.sim
        if sim.now < self.start_year:
            return
        
        # Get eligible people
        if callable(self.eligibility):
            eligible = self.eligibility(sim)
        elif self.eligibility is not None:
            eligible = self.eligibility
        else:
            eligible = sim.people.alive.uids
        
        if len(eligible) == 0:
            return
        
        # Sample who gets tested
        test_uids = self.pars.test_prob.filter(eligible)
        
        if len(test_uids) == 0:
            return
        
        # Administer tests (product handles all sensitivity logic)
        outcomes = self.product.administer(sim, test_uids)
        
        # Record outcomes
        self.ever_tested[test_uids] = True
        self.ti_tested[test_uids] = self.ti
        self.n_tests[test_uids] += 1
        self.last_outcomes = outcomes        

        # Record positive/negative
        if len(outcomes['positive']):
            self.ti_positive[outcomes['positive']] = self.ti
            self.diagnosed[outcomes['positive']] = True
        if len(outcomes['negative']):
            self.ti_negative[outcomes['negative']] = self.ti
        
        return
    
    def update_results(self):
        """Record results"""
        super().update_results()
        ti = self.ti
        self.results.new_tests[ti] = np.count_nonzero(self.ti_tested == ti)
        self.results.new_positives[ti] = np.count_nonzero(self.ti_positive == ti)
        new_dx = (self.ti_positive == ti) & (self.n_tests == 1)
        self.results.new_diagnoses[ti] = np.count_nonzero(new_dx)
        return


# ============================================================================
# Disease models
# ============================================================================
class Chlamydia_Simple(ss.Infection):
    """
    Simple chlamydia: S->I->S with fixed duration.
    """
    
    def __init__(self, name='ct', pars=None, **kwargs):
        super().__init__(name=name)
        
        self.define_pars(
            beta = ss.peryear(0.3),
            init_prev = ss.bernoulli(p=0.05),
            dur_inf = ss.lognorm_ex(mean=ss.months(14)),  # ~14 months
        )
        self.update_pars(pars, **kwargs)
        
        self.define_states(
            ss.FloatArr('ti_clearance', label='Time of clearance'),
        )
        return
    
    def step_state(self):
        """Clear infections"""
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
        return
    
    def set_prognoses(self, uids, sources=None):
        """Set infection outcomes"""
        super().set_prognoses(uids, sources)
        
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        
        dur_inf = self.pars.dur_inf.rvs(uids)
        self.ti_clearance[uids] = self.ti + dur_inf
        
        return
    
    def step_die(self, uids):
        """Clear states for dead agents"""
        self.susceptible[uids] = False
        self.infected[uids] = False
        return


class Chlamydia_BL(ss.Infection):
    """
    Chlamydia with bacterial load dynamics.
    
    Tracks bacterial_load which products can query for enhanced sensitivity.
    """
    
    def __init__(self, name='ct', pars=None, **kwargs):
        super().__init__(name=name)
        
        self.define_pars(
            beta = ss.peryear(0.3),
            init_prev = ss.bernoulli(p=0.05),
            # Bacterial load dynamics
            init_load = 1e3,
            peak_load = 1e7,
            time_to_peak = ss.lognorm_ex(ss.weeks(8), ss.weeks(4)),
            half_life = ss.lognorm_ex(ss.months(1), ss.months(0.5)),
            ct_beta = 0.5,
        )
        self.update_pars(pars, **kwargs)
        
        self.define_states(
            ss.FloatArr('bacterial_load', default=0.0, label='Bacterial load'),
            ss.FloatArr('ti_peak_load', label='Time of peak load'),
            ss.FloatArr('growth_rate', label='Load growth rate'),
            ss.FloatArr('decay_rate', label='Load decay rate'),
            ss.FloatArr('ti_clearance', label='Time of clearance'),
        )
        return
    
    def step_state(self):
        """Update bacterial load and clear infections"""
        if self.infected.any():
            self.update_bacterial_load(self.infected.uids)
            self.update_rel_trans()
        
        clearing = (self.infected & (self.ti_clearance <= self.ti)).uids
        if len(clearing):
            self.infected[clearing] = False
            self.susceptible[clearing] = True
            self.bacterial_load[clearing] = 0.0
            self.rel_trans[clearing] = 1.0
        
        return
    
    def update_bacterial_load(self, uids):
        """Update bacterial load over time"""
        growing = (self.ti_peak_load[uids] > self.ti)
        decaying = ~growing
        
        grow_uids = uids[growing]
        decay_uids = uids[decaying]
        
        if len(grow_uids):
            self.bacterial_load[grow_uids] *= np.exp(
                self.growth_rate[grow_uids] * self.sim.pars.dt
            )
        
        if len(decay_uids):
            self.bacterial_load[decay_uids] *= np.exp(
                -self.decay_rate[decay_uids] * self.sim.pars.dt
            )
        
        return
    
    def update_rel_trans(self):
        """Map bacterial load to transmissibility"""
        infected_uids = self.infected.uids
        if len(infected_uids) == 0:
            return
        
        loads = np.maximum(self.bacterial_load[infected_uids], 1.0)
        log_loads = np.log10(loads)
        self.rel_trans[infected_uids] = (
            2.0 / (1.0 + np.exp(-self.pars.ct_beta * log_loads)) - 1.0
        )
        return
    
    def set_prognoses(self, uids, sources=None):
        """Set infection outcomes and bacterial load"""
        super().set_prognoses(uids, sources)
        
        p = self.pars
        
        self.susceptible[uids] = False
        self.infected[uids] = True
        self.ti_infected[uids] = self.ti
        
        # Initialize bacterial load
        time_to_peak = p.time_to_peak.rvs(uids)
        self.bacterial_load[uids] = p.init_load
        self.ti_peak_load[uids] = self.ti + time_to_peak
        
        self.growth_rate[uids] = np.log(p.peak_load / p.init_load) / time_to_peak
        
        half_life = p.half_life.rvs(uids)
        self.decay_rate[uids] = np.log(2) / ss.years(half_life*self.t.dt_year).months
        
        decay_time = ss.months(-np.log(p.init_load / p.peak_load) / self.decay_rate[uids])
        dur_inf = time_to_peak + decay_time/self.t.dt 
        self.ti_clearance[uids] = self.ti + dur_inf
        
        return
    
    def step_die(self, uids):
        """Clear states for dead agents"""
        self.susceptible[uids] = False
        self.infected[uids] = False
        self.bacterial_load[uids] = 0.0
        return


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
    
    # ========================================================================
    # Scenario 1: Simple model with basic test
    # ========================================================================
    sc.heading("Scenario 1: simple model + basic test")
    
    chlamydia_simple = Chlamydia_Simple(
        beta=ss.peryear(0.5),
        init_prev=ss.bernoulli(p=0.05)
    )
    
    basic_test = STIDx(
        disease=0,
        base_sensitivity=0.85,
        base_specificity=0.99
    )
    
    test_basic = STITest(
        product=basic_test,
        annual_test_prob=0.3,
        start_year=2022
    )
    
    sim1 = ss.Sim(
        n_agents=10000,
        start=2020,
        dur=10,
        dt=0.25,
        networks='random',
        diseases=chlamydia_simple,
        interventions=test_basic,
        label='Simple + Basic Test'
    )
    sim1.run()
    print(f"✓ {sim1.label}")
    
    # ========================================================================
    # Scenario 2: BL model with basic test (ignores load)
    # ========================================================================
    sc.heading("Scenario 2: BL Model + Basic Test (load-agnostic)")
    
    chlamydia_bl2 = Chlamydia_BL(
        beta=ss.peryear(0.5),
        init_prev=ss.bernoulli(p=0.05)
    )
    
    basic_test2 = STIDx(
        disease=0,
        base_sensitivity=0.85,
        base_specificity=0.99
    )
    
    test_basic2 = STITest(
        product=basic_test2,
        annual_test_prob=0.3,
        start_year=2022
    )
    
    sim2 = ss.Sim(
        n_agents=10000,
        start=2020,
        dur=10,
        dt=0.25,
        networks='random',
        diseases=chlamydia_bl2,
        interventions=test_basic2,
        label='BL + Basic Test'
    )
    sim2.run()
    print(f"✓ {sim2.label}")
    
    # ========================================================================
    # Scenario 3: BL model with load-sensitive test
    # ========================================================================
    sc.heading("Scenario 3: BL Model + Load-Sensitive Test")
    
    chlamydia_bl3 = Chlamydia_BL(
        beta=ss.peryear(0.5),
        init_prev=ss.bernoulli(p=0.05)
    )
    
    sensitive_test = STIDx_BacterialLoad(
        disease=0,
        max_sensitivity=0.95,
        load_50=1e4,
        specificity=0.99
    )
    
    test_sensitive = STITest(
        product=sensitive_test,
        annual_test_prob=0.3,
        start_year=2022
    )
    
    sim3 = ss.Sim(
        n_agents=10000,
        start=2020,
        dur=10,
        dt=0.25,
        networks='random',
        diseases=chlamydia_bl3,
        interventions=test_sensitive,
        label='BL + Sensitive Test'
    )
    sim3.run()
    print(f"✓ {sim3.label}")
    
    # ========================================================================
    # Scenario 4: Multiple tests on same disease!
    # ========================================================================
    sc.heading("Scenario 4: BL Model + 2 Tests")
    
    chlamydia_bl4 = Chlamydia_BL(
        beta=ss.peryear(0.5),
        init_prev=ss.bernoulli(p=0.05)
    )
    
    # Cheap basic test (low sensitivity)
    cheap_test = STIDx(
        disease=0,
        base_sensitivity=0.70,
        base_specificity=0.98
    )
    
    test_cheap = STITest(
        name='poc',
        product=cheap_test,
        annual_test_prob=0.4,  # More frequent (cheaper)
        start_year=2022,
        label='POC test'
    )
    
    # Expensive sensitive test (load-dependent)
    expensive_test = STIDx_BacterialLoad(
        disease=0,
        max_sensitivity=0.98,
        load_50=5e3,  # More sensitive
        base_specificity=0.995
    )
    
    test_expensive = STITest(
        name='lab',
        product=expensive_test,
        annual_test_prob=0.15,  # Less frequent (expensive)
        start_year=2022,
        label='Lab test'
    )
    
    sim4 = ss.Sim(
        n_agents=10000,
        start=2020,
        dur=10,
        dt=0.25,
        networks='random',
        diseases=chlamydia_bl4,
        interventions=[test_cheap, test_expensive],
        label='BL + Two Tests'
    )
    sim4.run()
    print(f"✓ {sim4.label}")
    
