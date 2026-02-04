# Chlamydia Testing Demo: Disease-Agnostic Interventions

This demo illustrates how to create interventions that work with multiple disease models without modification. The key innovation is using **disease-specific products** that handle model differences, allowing the intervention logic to remain completely agnostic to which disease model is being used.

## Architecture Overview

### The Challenge

In real-world modeling, you often need the same intervention (e.g., testing, treatment) to work with different disease models that may have varying levels of complexity. For example:

- **Simple model**: Basic S→I→S dynamics with fixed clearance time
- **Complex model**: S→I→S with bacterial load dynamics affecting transmission and detectability

Traditional approaches would require either:
1. Separate intervention code for each model (code duplication)
2. Complex branching logic in the intervention (hard to maintain)
3. Modifying disease models to expose a common interface (inflexible)

### The Solution: Disease-Specific Products

We solve this by separating concerns:

```
┌─────────────────────────────────────────────────┐
│  Intervention (STITest)                         │
│  • Deployment logic (who, when, how often)      │
│  • Model-agnostic                               │
│  • Uses products to administer tests            │
└────────────┬────────────────────────────────────┘
             │
             │ uses
             ↓
┌─────────────────────────────────────────────────┐
│  Product (STIDx / STIDx_BacterialLoad)          │
│  • Test characteristics (sensitivity, spec)     │
│  • Disease-specific logic                       │
│  • Queries disease states                       │
│  • Calculates p_positive                        │
└────────────┬────────────────────────────────────┘
             │
             │ queries
             ↓
┌─────────────────────────────────────────────────┐
│  Disease (Chlamydia_Simple / Chlamydia_BL)      │
│  • Disease dynamics                             │
│  • States: infected, bacterial_load, etc.       │
└─────────────────────────────────────────────────┘
```

## Components

### 1. Disease Models

**`Chlamydia_Simple`**: Basic S→I→S model
- Fixed ~14 month duration
- Constant transmissibility while infected
- Minimal states: `infected`, `ti_clearance`

**`Chlamydia_BL`**: S→I→S with bacterial load
- Dynamic bacterial load (growth → peak → decay)
- Load modulates `rel_trans` (transmission probability)
- Additional states: `bacterial_load`, growth/decay rates, peak timing

Both inherit from `ss.Infection` and use built-in transmission logic.

### 2. Products (Disease-Specific)

**`STIDx`**: Basic diagnostic test
- Fixed sensitivity (85%) for infected individuals
- Works with any disease that has `infected` state
- Represents traditional culture-based tests

**`STIDx_BacterialLoad`**: Load-sensitive diagnostic
- Sensitivity depends on bacterial load (Michaelis-Menten curve)
- Queries `bacterial_load` state if available
- Falls back to fixed sensitivity if not
- Represents modern molecular tests (NAAT/PCR)

**Key innovation**: Products own the `p_positive` state (probability of testing positive) and calculate it by querying disease states. This keeps test characteristics with the test, where they belong.

### 3. Intervention (Disease-Agnostic)

**`STITest`**: Testing intervention
- Handles deployment logic: who gets tested, when, how often
- Completely agnostic to disease model details
- Simply calls `product.administer()` and records outcomes
- Works identically with both simple and complex models

## Usage Examples

### Same intervention, different models:

```python
# Scenario 1: Simple model with basic test
disease = Chlamydia_Simple()
product = STIDx(disease=0, base_sensitivity=0.85)
intervention = STITest(product=product, annual_test_prob=0.3)

sim = ss.Sim(diseases=disease, interventions=intervention)
```

```python
# Scenario 2: Complex model with load-sensitive test
disease = Chlamydia_BL()
product = STIDx_BacterialLoad(disease=0, max_sensitivity=0.95, load_50=1e4)
intervention = STITest(product=product, annual_test_prob=0.3)

sim = ss.Sim(diseases=disease, interventions=intervention)
```

**Note**: The intervention code (`STITest`) is **identical** in both cases. Only the product changes.

### Multiple tests on same disease:

```python
# Run two different tests simultaneously
disease = Chlamydia_BL()

# Cheap, less sensitive test (high volume)
poc_product = STIDx(disease=0, base_sensitivity=0.70)
poc_test = STITest(name='poc', product=poc_product, annual_test_prob=0.4)

# Expensive, more sensitive test (low volume)
lab_product = STIDx_BacterialLoad(disease=0, max_sensitivity=0.98, load_50=5e3)
lab_test = STITest(name='lab', product=lab_product, annual_test_prob=0.15)

sim = ss.Sim(diseases=disease, interventions=[poc_test, lab_test])
```

This mirrors real-world testing strategies where routine screening (POC) and confirmatory testing (lab) occur simultaneously.

## Key Design Principles

### 1. Separation of Concerns
- **Intervention**: When/who/how often to test
- **Product**: What the test detects and how well
- **Disease**: Biological dynamics

### 2. Products Query, Don't Modify
Products read disease states but don't modify them (except through the intervention's actions). This maintains clear data flow.

### 3. No Connectors Needed
Unlike earlier designs, we don't need connectors to translate between disease states and test probabilities. The product handles this translation internally, keeping test-specific logic encapsulated.

### 4. Extensibility
Adding new test types is trivial:
- Subclass `STIDx`
- Override `calculate_p_positive()` with your sensitivity logic
- Product automatically works with the intervention

## Bacterial Load Dynamics

The `Chlamydia_BL` model implements realistic bacterial load kinetics:

**Growth Phase** (0-8 weeks):
- Exponential growth: `load(t) = init_load × exp(k_growth × t)`
- Initial load: 10³ copies/mL
- Peak load: 10⁷ copies/mL

**Decay Phase** (8 weeks - clearance):
- Exponential decay: `load(t) = peak_load × exp(-k_decay × (t - t_peak))`
- Half-life: ~2.5 weeks (sampled from lognormal distribution)
- Clears when load returns to initial level (~14 months total)

**Impact on Transmission**:
- `rel_trans = 2/(1 + exp(-β × log₁₀(load))) - 1`
- Higher load → higher transmission probability
- Creates realistic time-varying infectiousness

**Impact on Detection**:
- Sensitivity = `max_sens × load / (K + load)` (Michaelis-Menten)
- Low early in infection (~20% sensitivity)
- High at peak (~90% sensitivity)  
- Declining in late infection
- Explains why some infections are missed despite testing

## Test Performance Characteristics

### Basic Test (STIDx)
- **Sensitivity**: 85% (fixed)
- **Specificity**: 99%
- Represents culture-based methods
- Misses ~15% of infections regardless of load

### Sensitive Test (STIDx_BacterialLoad)
- **Maximum sensitivity**: 95%
- **Minimum sensitivity**: 15%
- **K value (load_50)**: 10⁴ copies/mL
- Represents NAAT/PCR methods
- Performance depends on infection stage:
  - Early infection (low load): ~20-40% sensitivity
  - Peak infection (high load): ~90-95% sensitivity
  - Late infection (declining load): ~60-80% sensitivity

## Visualizations

Run the plotting script to generate comprehensive figures:

```python
from chlamydia_plots import plot_bacterial_load_dynamics

# Create disease instance
disease_bl = Chlamydia_BL()
sim = ss.Sim(n_agents=100, diseases=disease_bl, networks='random')
sim.initialize()

# Generate plots
plot_bacterial_load_dynamics(sim.diseases[0], n_trajectories=50)
```

This produces three panels:
- **Panel D**: Bacterial load trajectories over complete infection course
- **Panel E**: Test sensitivity vs bacterial load (dose-response curve)
- **Panel F**: Probability of positive test over infection time

## Files

- `chlamydia_models.py`: Disease models and testing infrastructure
- `chlamydia_plots.py`: Visualization functions
- `README.md`: This file

## Key Takeaway

This demo shows how to build **composable, reusable intervention infrastructure** in Starsim. By using disease-specific products that encapsulate model differences, you can write intervention logic once and apply it to any compatible disease model. This pattern scales to treatment interventions, vaccination campaigns, contact tracing, and any other intervention that needs to work across multiple disease implementations.

The same `STITest` intervention code works unchanged with:
- Simple vs complex disease models
- Different test products with different sensitivities
- Multiple simultaneous tests on the same disease
- Future disease models you haven't written yet

