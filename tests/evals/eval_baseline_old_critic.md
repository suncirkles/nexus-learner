# Detailed Evaluation Results

## Item 1: Active_recall
**Bad Data:** False
**Topic:** Chemical Kinetics > Rate Law and Rate Expression

### Source Text
```text
The rate of a chemical reaction at a given temperature may depend on the concentration of one or more reactants and products. The representation of rate of reaction in terms of concentration of the reactants is known as rate law. It is also called as rate equation or rate expression. Consider a general reaction: aA + bB → cC + dD. The rate expression is: Rate = k[A]^x [B]^y, where k is the rate constant and x, y are the orders with respect to A and B respectively. The sum of powers x + y gives the overall order of the reaction. The rate constant k is a proportionality constant specific to a particular reaction at a given temperature. Rate law for any reaction cannot be predicted by merely looking at the balanced chemical equation but must be determined experimentally.
```

**Question:** What is the rate law (rate expression) for a general reaction aA + bB → cC + dD, and what does each symbol represent?

**Answer:** The rate law is: Rate = k[A]^x [B]^y, where k is the rate constant (a proportionality constant specific to a particular reaction at a given temperature), [A] and [B] are the concentrations of reactants A and B, and x and y are the orders of the reaction with respect to A and B respectively. The overall order of the reaction is x + y.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 2: Active_recall
**Bad Data:** False
**Topic:** Chemical Kinetics > Arrhenius Equation and Temperature Dependence

### Source Text
```text
It has been found that for a chemical reaction with rise in temperature by 10 degrees, the rate constant is nearly doubled. The temperature dependence of the rate of a chemical reaction can be accurately explained by Arrhenius equation: k = A * e^(-Ea/RT), where A is the Arrhenius factor (also called the frequency factor or pre-exponential factor), R is the gas constant, and Ea is the activation energy measured in joules per mole. Taking natural logarithm of both sides: ln k = -Ea/RT + ln A. A plot of ln k versus 1/T gives a straight line with slope = -Ea/R and intercept = ln A. Increasing the temperature or decreasing the activation energy will result in an increase in the rate of the reaction.
```

**Question:** State the Arrhenius equation and identify what each term represents.

**Answer:** The Arrhenius equation is: k = A · e^(−Ea/RT), where:
- **k** = rate constant
- **A** = Arrhenius factor (frequency factor / pre-exponential factor)
- **Ea** = activation energy (measured in joules per mole)
- **R** = gas constant
- **T** = absolute temperature

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 3: Active_recall
**Bad Data:** False
**Topic:** PySpark > Spark Architecture

### Source Text
```text
Driver Program: This is your main program. It is the brain that coordinates everything. When you write spark.read.csv(...), the Driver creates a plan for how to execute it. Cluster Manager: Think of this as the HR department. It manages resources (CPU, memory) across the cluster. Common options: YARN (Hadoop), Kubernetes, Mesos, or Standalone. Executors: These are the workers. Each executor runs on a separate machine and processes a chunk of your data. They do the actual computation. Tasks: The smallest unit of work. Each executor runs multiple tasks in parallel. If you have 100 partitions of data and 10 executors, each executor handles approximately 10 tasks. Flow: You write your code → Driver creates execution plan (DAG) → Cluster Manager allocates executors → Data is split into partitions → Each executor processes its partitions → Results are collected back to the Driver.
```

**Question:** In Spark Architecture, what are the four core components and what is the primary role of each?

**Answer:** 1. **Driver Program** – The brain/main program that coordinates everything and creates the execution plan (DAG) for how to process data.
2. **Cluster Manager** – Manages resources (CPU, memory) across the cluster; common options include YARN, Kubernetes, Mesos, and Standalone.
3. **Executors** – Workers that run on separate machines and process chunks of data, performing the actual computation.
4. **Tasks** – The smallest unit of work; each executor runs multiple tasks in parallel.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 4: Active_recall
**Bad Data:** False
**Topic:** Chemical Reactions and Equations > Displacement and Double Displacement Reactions

### Source Text
```text
In a displacement reaction, a more reactive element replaces a less reactive element from a compound. When an iron nail is placed in a copper sulphate solution, iron displaces copper from copper sulphate solution forming iron sulphate, which is green in colour. The reaction is: Fe + CuSO4 → FeSO4 + Cu. Therefore, the blue colour of copper sulphate solution fades and green colour appears. In a double displacement reaction, two atoms or groups of atoms switch places to form two new compounds. For example: Na2CO3 + CaCl2 → CaCO3 + 2NaCl. Here, sodium and calcium exchange their anions to form two new compounds. Such reactions in which ions exchange between two reactants are called double displacement or metathesis reactions.
```

**Question:** An iron nail is placed in a copper sulphate (CuSO₄) solution. What type of reaction occurs, what is the chemical equation, and what observable colour change takes place?

**Answer:** A displacement reaction occurs. Iron, being more reactive, displaces copper from copper sulphate solution:

Fe + CuSO₄ → FeSO₄ + Cu

The blue colour of the copper sulphate solution fades as copper is displaced, and a green colour appears due to the formation of iron sulphate (FeSO₄).

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 5: Fill_blank
**Bad Data:** False
**Topic:** Chemical Kinetics > Half-Life of Reactions

### Source Text
```text
The half-life of a reaction is the time in which the concentration of a reactant is reduced to one half of its initial concentration. It is represented as t1/2. For the first order reaction, the integrated rate equation is: ln[R] = -kt + ln[R]0. At t1/2, [R] = [R]0/2. Substituting into the integrated equation: ln([R]0/2) = -k * t1/2 + ln[R]0, which simplifies to: t1/2 = 0.693/k. It is notably independent of initial concentration — only the rate constant k determines it. For a zero order reaction by contrast, t1/2 = [R]0 / (2k), meaning half-life is directly proportional to the initial concentration. The half-life concept is widely used in radioactive decay calculations.
```

**Question:** For a first order reaction, the half-life is given by the formula t1/2 = ___.

**Answer:** 0.693/k

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 6: Fill_blank
**Bad Data:** False
**Topic:** Chemical Kinetics > Molecularity of Reactions

### Source Text
```text
The number of reacting species (atoms, ions or molecules) taking part in an elementary reaction and bringing about the chemical change is called molecularity of a reaction. Unimolecular reactions involve decomposition or rearrangement of a single reacting species; for example: NH4NO2 → N2 + 2H2O. Bimolecular reactions involve simultaneous collision between two species; for example: 2HI → H2 + I2. Trimolecular or termolecular reactions involve simultaneous collision between three reacting species; for example: 2NO + O2 → 2NO2. The probability that more than three molecules can collide and react simultaneously is very small. Hence, reactions with molecularity greater than three are extremely rare. Molecularity is applicable only for elementary reactions; for complex reactions it has no meaning.
```

**Question:** The number of reacting species (atoms, ions or molecules) taking part in an elementary reaction and bringing about the chemical change is called ___ of a reaction.

**Answer:** molecularity

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 7: Fill_blank
**Bad Data:** False
**Topic:** PySpark > Resilient Distributed Datasets (RDDs)

### Source Text
```text
RDD stands for Resilient Distributed Dataset. It is the fundamental data structure in Spark. Think of it as an immutable, distributed collection of objects that can be processed in parallel. Resilient: If a partition of data is lost because a machine fails, Spark can recompute it from the original source using the lineage graph. Distributed: Data is split across multiple machines in the cluster. Dataset: A collection of records or rows of data. RDD operations fall into two categories: Transformations (lazy operations that define what to do but do NOT execute immediately — examples: map(), filter(), flatMap(), reduceByKey()) and Actions (operations that force Spark to execute the plan and return results — examples: collect(), count(), first(), saveAsTextFile()). In modern PySpark, DataFrames are preferred over RDDs because they benefit from the Catalyst optimizer.
```

**Question:** In PySpark, RDD stands for Resilient ___ Dataset.

**Answer:** Distributed

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 8: Fill_blank
**Bad Data:** False
**Topic:** Chemical Reactions and Equations > Balancing Chemical Equations

### Source Text
```text
A reaction which has an equal number of atoms of all the elements on both sides of the chemical equation is called a balanced chemical equation. The law of conservation of mass states that mass can neither be created nor destroyed in a chemical reaction. Hence, the total mass of reactants must equal the total mass of the products, meaning the total number of atoms of each element must be equal on both sides. For example, hydrogen gas combines with nitrogen to form ammonia: N2 + 3H2 → 2NH3. On the left side: 2 nitrogen atoms and 6 hydrogen atoms. On the right side: 2 nitrogen atoms and 6 hydrogen atoms. This equation is balanced. Another example: 2H2 + O2 → 2H2O — each side has 4 hydrogen atoms and 2 oxygen atoms.
```

**Question:** A reaction which has an equal number of atoms of all the elements on both sides of the chemical equation is called a ___ chemical equation.

**Answer:** balanced

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 9: Numerical
**Bad Data:** False
**Topic:** Chemical Kinetics > Average Rate of Reaction

### Source Text
```text
The decomposition of N2O5 in CCl4 at 318 K has been studied by monitoring the concentration of N2O5 in the solution. Initially the concentration of N2O5 is 2.33 mol/L and after 184 minutes, it is reduced to 2.08 mol/L. The reaction takes place according to: 2N2O5(g) → 4NO2(g) + O2(g). The average rate is: Average Rate = -(1/2) * delta[N2O5] / delta_t = -(1/2) * (2.08 - 2.33) / 184 min = -(1/2) * (-0.25/184) = 6.79 * 10^-4 mol/L/min. Converting: 6.79 * 10^-4 mol/L/min * 60 min/h = 4.07 * 10^-2 mol/L/h. The rate of production of NO2 is 4 times the rate of disappearance of N2O5 (from stoichiometry), so rate of NO2 production = 4 * 6.79 * 10^-4 = 2.72 * 10^-3 mol/L/min.
```

**Question:** The decomposition of H2O2 in aqueous solution at 300 K is monitored by tracking its concentration over time. The reaction is:

**2H₂O₂(aq) → 2H₂O(l) + O₂(g)**

The following concentration data is recorded:

| Time (min) | [H₂O₂] (mol/L) |
|---|---|
| 0 | 3.60 |
| 240 | 3.12 |

(a) Calculate the average rate of decomposition of H₂O₂ in mol/L/min.
(b) Calculate the average rate of production of O₂ in mol/L/min.
(c) Express the average rate of decomposition of H₂O₂ in mol/L/h.

**Answer:** **Given:**
- Δ[H₂O₂] = 3.12 − 3.60 = −0.48 mol/L
- Δt = 240 min
- Stoichiometry: 2H₂O₂ → 2H₂O + O₂

**(a) Average rate of decomposition of H₂O₂:**

Average Rate = −(1/2) × Δ[H₂O₂] / Δt
= −(1/2) × (−0.48) / 240
= −(1/2) × (−0.002)
= **1.00 × 10⁻³ mol/L/min**

**(b) Rate of production of O₂:**

From stoichiometry, 2 mol H₂O₂ decompose for every 1 mol O₂ produced.
The rate of O₂ production equals the average rate (as defined above):

Rate of O₂ production = (1/2) × |Δ[H₂O₂]| / Δt = 1.00 × 10⁻³ mol/L/min

Alternatively, since the stoichiometric coefficient of O₂ is 1 and of H₂O₂ is 2:
Rate of O₂ production = (1/2) × 2.00 × 10⁻³ = **1.00 × 10⁻³ mol/L/min**

Note: The rate of disappearance of H₂O₂ (without the 1/2 factor) = 0.48/240 = 2.00 × 10⁻³ mol/L/min, and O₂ is produced at half that rate = **1.00 × 10⁻³ mol/L/min**.

**(c) Conversion to mol/L/h:**

1.00 × 10⁻³ mol/L/min × 60 min/h = **6.00 × 10⁻² mol/L/h**

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 10: Numerical
**Bad Data:** False
**Topic:** Chemical Kinetics > First Order Rate Constant

### Source Text
```text
The initial concentration of N2O5 in the first order reaction N2O5(g) → 2NO2(g) + 1/2 O2(g) was 1.24 * 10^-2 mol/L at 318 K. The concentration of N2O5 after 60 minutes was 0.20 * 10^-2 mol/L. For a first order reaction, the rate constant k is calculated using: k = (2.303 / t) * log([R]0 / [R]). Substituting: k = (2.303 / 60 min) * log(1.24 * 10^-2 / 0.20 * 10^-2) = (2.303 / 60) * log(6.2) = (2.303 / 60) * 0.792 = 0.0304 min^-1. The units of k for a first order reaction are time^-1 (e.g., s^-1 or min^-1), confirming this is a first order reaction.
```

**Question:** A first order decomposition reaction of H2O2 is carried out at 300 K. The initial concentration of H2O2 is 3.60 × 10⁻² mol/L. After 80 minutes, the concentration drops to 0.45 × 10⁻² mol/L.

Using the first order rate constant formula:

**k = (2.303 / t) × log([R]₀ / [R])**

Calculate the rate constant k (in min⁻¹) for this reaction. Show all steps.

**Answer:** Given:
- [R]₀ = 3.60 × 10⁻² mol/L
- [R] = 0.45 × 10⁻² mol/L
- t = 80 min

Step 1 — Calculate the concentration ratio:
[R]₀ / [R] = (3.60 × 10⁻²) / (0.45 × 10⁻²) = 8.0

Step 2 — Take the log:
log(8.0) = 0.9031

Step 3 — Substitute into the formula:
k = (2.303 / 80) × 0.9031
k = 0.028788 × 0.9031
k ≈ 0.0260 min⁻¹

The rate constant k ≈ 0.0260 min⁻¹. The unit time⁻¹ confirms this is a first order reaction.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 11: Numerical
**Bad Data:** False
**Topic:** Chemical Kinetics > Half-Life Calculations

### Source Text
```text
A first order reaction is found to have a rate constant k = 5.5 * 10^-14 s^-1. For a first order reaction, half-life is given by: t1/2 = 0.693 / k. Substituting: t1/2 = 0.693 / (5.5 * 10^-14 s^-1) = 1.26 * 10^13 s. It can be shown that for a first order reaction, the time required for completion of 99.9% of the reaction is 10 times the half-life. When 99.9% is complete: [R] = [R]0 - 0.999[R]0 = 0.001[R]0. So k = (2.303/t) * log([R]0/0.001[R]0) = (2.303/t) * log(1000) = 6.909/t, giving t = 6.909/k. Since t1/2 = 0.693/k, we have t/t1/2 = 6.909/0.693 = 10.
```

**Question:** A radioactive isotope undergoes first-order decay with a rate constant k = 3.2 × 10⁻⁶ s⁻¹.

**(a)** Calculate the half-life (t₁/₂) of the isotope.

**(b)** Calculate the time required for 99.9% of the isotope to decay.

**(c)** Verify numerically that the time for 99.9% completion is exactly 10 times the half-life.

Use the first-order relationships:
- t₁/₂ = 0.693 / k
- For 99.9% completion: t = 6.909 / k
- Ratio: t / t₁/₂ = 6.909 / 0.693

**Answer:** **(a) Half-life calculation:**

t₁/₂ = 0.693 / k = 0.693 / (3.2 × 10⁻⁶ s⁻¹)

t₁/₂ = **2.166 × 10⁵ s** (≈ 2.17 × 10⁵ s)

---

**(b) Time for 99.9% completion:**

When 99.9% has decayed: [R] = 0.001 [R]₀

Using the first-order integrated rate law:

k = (2.303 / t) × log([R]₀ / [R])

k = (2.303 / t) × log(1 / 0.001) = (2.303 / t) × log(1000) = (2.303 / t) × 3 = 6.909 / t

Solving for t:

t = 6.909 / k = 6.909 / (3.2 × 10⁻⁶ s⁻¹)

t = **2.159 × 10⁶ s** (≈ 2.16 × 10⁶ s)

---

**(c) Verification of the 10× relationship:**

t / t₁/₂ = (6.909 / k) / (0.693 / k) = 6.909 / 0.693 = **10.0**

Numerically: 2.159 × 10⁶ s / 2.166 × 10⁵ s ≈ 10.0 ✓

This confirms that for any first-order reaction, the time for 99.9% completion is always exactly 10 times the half-life.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 12: Numerical
**Bad Data:** False
**Topic:** Chemical Kinetics > Arrhenius Two-Temperature Calculation

### Source Text
```text
The rate constants of a reaction at 500 K and 700 K are 0.02 s^-1 and 0.07 s^-1 respectively. Using the Arrhenius equation at two temperatures: log(k2/k1) = (Ea / 2.303 R) * (T2 - T1) / (T1 * T2). Substituting: log(0.07/0.02) = Ea * (700 - 500) / (2.303 * 8.314 * 500 * 700). This gives: log(3.5) = 0.544 = Ea * 200 / (2.303 * 8.314 * 350000) = Ea * 5.714 * 10^-4 / 19.15. Solving: Ea = 0.544 * 19.15 / (5.714 * 10^-4) = 18230.8 J/mol approximately 18.2 kJ/mol. The pre-exponential factor A is then found from k = A * e^(-Ea/RT): 0.02 = A * e^(-18230.8 / (8.314 * 500)), giving A approximately 1.61.
```

**Question:** The rate constants of a reaction measured at two different temperatures are given below:

| Temperature (K) | Rate Constant (s⁻¹) |
|-----------------|---------------------|
| 600             | 0.045               |
| 800             | 0.180               |

Using the two-temperature Arrhenius equation:

$$\log\left(\frac{k_2}{k_1}\right) = \frac{E_a}{2.303 \, R} \cdot \frac{T_2 - T_1}{T_1 \cdot T_2}$$

where R = 8.314 J mol⁻¹ K⁻¹, calculate the activation energy **Eₐ** (in kJ/mol). Then, using k = A·e^(−Eₐ/RT) at T = 600 K, determine the pre-exponential factor **A**.

**Answer:** Step 1 — Compute the ratio and its logarithm:
k₂/k₁ = 0.180 / 0.045 = 4.0
log(4.0) = 0.6021

Step 2 — Compute the temperature factor:
T₂ − T₁ = 800 − 600 = 200 K
T₁ × T₂ = 600 × 800 = 480,000 K²
(T₂ − T₁)/(T₁ × T₂) = 200 / 480,000 = 4.1667 × 10⁻⁴ K⁻¹

Step 3 — Compute the denominator constant:
2.303 × R = 2.303 × 8.314 = 19.147 J mol⁻¹ K⁻¹

Step 4 — Solve for Eₐ:
0.6021 = Eₐ × (4.1667 × 10⁻⁴) / 19.147
Eₐ = 0.6021 × 19.147 / (4.1667 × 10⁻⁴)
Eₐ = 11.531 / (4.1667 × 10⁻⁴)
Eₐ ≈ 27,674 J/mol ≈ 27.67 kJ/mol

Step 5 — Find the pre-exponential factor A using k₁ = 0.045 s⁻¹ at T = 600 K:
Eₐ/RT = 27,674 / (8.314 × 600) = 27,674 / 4,988.4 = 5.548
e^(−5.548) = 0.003876
A = k₁ / e^(−Eₐ/RT) = 0.045 / 0.003876 ≈ 11.61 s⁻¹

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 13: Scenario
**Bad Data:** False
**Topic:** Chemical Kinetics > Effect of Catalyst on Reaction Rate

### Source Text
```text
A catalyst is a substance which increases the rate of a reaction without itself undergoing any permanent chemical change. For example, MnO2 catalyses the decomposition of KClO3: 2KClO3 → (MnO2 catalyst) 2KCl + 3O2. According to intermediate complex theory, a catalyst participates by forming temporary bonds with the reactants, creating an intermediate complex that decomposes to yield products and regenerates the catalyst. A catalyst provides an alternate pathway by reducing the activation energy between reactants and products. A catalyst does not alter the Gibbs energy (delta G) of a reaction — it catalyses only spontaneous reactions. It does not change the equilibrium constant but helps in attaining equilibrium faster by catalysing both the forward and backward reactions equally.
```

**Question:** A student adds MnO2 to a reaction mixture of KClO3, hoping that the catalyst will shift the equilibrium position toward products and increase the yield of O2. What actually happens, and why is the student's expectation incorrect?

**Answer:** The student's expectation is incorrect. MnO2 acts as a catalyst and does not alter the Gibbs energy (ΔG) or the equilibrium constant of the reaction. It only provides an alternate pathway with lower activation energy, allowing the system to reach equilibrium faster by catalysing both the forward and backward reactions equally. The final equilibrium position — and therefore the yield of O2 — remains unchanged; the reaction simply attains that equilibrium more quickly.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 14: Scenario
**Bad Data:** False
**Topic:** PySpark > Lazy Evaluation and DAG Execution

### Source Text
```text
Lazy evaluation is the single most important concept in PySpark. Unlike Pandas, PySpark does NOT execute your code line by line. Instead, it builds up a plan called a DAG (Directed Acyclic Graph) and only executes when you explicitly ask for a result. Transformations are lazy operations that define what to do but do NOT execute: filter(), select(), groupBy(), join(), withColumn(). These just add steps to the plan. Actions trigger execution: show(), count(), collect(), write(). Example: df = spark.read.csv('huge_file.csv') then filtered = df.filter(col('age') > 25) then grouped = filtered.groupBy('department').count() — none of these execute yet. Only grouped.show() triggers execution of the entire chain. This design allows Spark to optimize the full query plan (e.g., predicate pushdown, column pruning) before any data is actually read.
```

**Question:** Consider the following PySpark code:

```python
df = spark.read.csv('huge_file.csv')
filtered = df.filter(col('age') > 25)
grouped = filtered.groupBy('department').count()
```

What happens if you add `grouped.show()` at the end? When exactly does Spark read the CSV and execute the filter and groupBy operations?

**Answer:** Nothing executes until `grouped.show()` is called. `show()` is an **action**, which triggers execution of the entire DAG chain. Only at this point does Spark:
1. Read `huge_file.csv`
2. Apply `filter(col('age') > 25)`
3. Apply `groupBy('department').count()`

Before `show()`, all three lines merely build a logical plan (DAG). Spark can also optimize this full plan — for example, via predicate pushdown or column pruning — before any data is actually read.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 15: Scenario
**Bad Data:** False
**Topic:** Chemical Reactions and Equations > Exothermic Reactions and Corrosion

### Source Text
```text
Chemical reactions that release energy in the form of heat, light, or sound are called exothermic reactions. Reactions that absorb energy or require energy in order to proceed are called endothermic reactions. Magnesium is very reactive. When stored, it reacts with oxygen to form a layer of magnesium oxide (MgO) on its surface. This oxide layer is stable and prevents further reaction — acting as a protective barrier (passivation). Rusting of iron is a slow exothermic oxidation reaction: iron reacts with oxygen and water forming hydrated iron(III) oxide (Fe2O3.xH2O). Corrosion can be prevented by painting, galvanisation (coating with zinc), or using stainless steel alloys. The process of coating iron/steel with zinc is called galvanisation, and it protects iron even when the coating is scratched, because zinc is more reactive and acts as a sacrificial anode.
```

**Question:** A steel bridge is coated with zinc (galvanised) to prevent rusting. Over time, a small section of the zinc coating gets scratched, exposing the iron underneath. What happens to the iron at the scratched spot, and why does galvanisation still protect it even after the scratch?

**Answer:** Even after the zinc coating is scratched and iron is exposed, the iron does NOT rust immediately. This is because zinc is more reactive than iron, so zinc preferentially reacts with oxygen and moisture instead of iron. Zinc acts as a **sacrificial anode** — it corrodes in place of the iron, continuing to protect the underlying steel. This is different from a paint coating, which would allow rusting to begin as soon as the iron is exposed.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 16: Scenario
**Bad Data:** False
**Topic:** PySpark > Partitioning and Data Distribution

### Source Text
```text
Data in Spark is split across multiple machines through partitions. When you create an RDD using sc.parallelize(data, numSlices=4), data is divided into 4 partitions distributed across the cluster. If you have 100 partitions of data and 10 executors, each executor handles approximately 10 tasks in parallel. Writing partitioned data: df.write.mode('overwrite').partitionBy('year', 'month').parquet('output/partitioned/') creates a directory structure organised by year and month. Partitioned writes dramatically improve query performance because Spark can skip entire folders based on partition values — this is called partition pruning. Warning: Never use coalesce(1) on large datasets as it forces all data onto one partition (one machine), defeating the purpose of distributed computing and likely causing an OutOfMemoryError.
```

**Question:** A data engineer runs the following PySpark command on a 500 GB dataset:

```python
df.coalesce(1).write.mode('overwrite').parquet('output/final/')
```

What happens when this code is executed, and why is it considered dangerous?

**Answer:** Using `coalesce(1)` forces all 500 GB of data onto a single partition on one machine. This defeats the purpose of distributed computing because the entire dataset must be processed by a single executor instead of being spread across the cluster. The likely outcome is an **OutOfMemoryError**, as one machine cannot hold the full dataset in memory. The correct approach is to avoid `coalesce(1)` on large datasets and instead allow Spark to distribute the data across multiple partitions.

Evaluation scores attached below.

### Scores
- Code Graders: {'has_flashcards': True, 'valid_count': True, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': True, 'is_self_contained': True}
- Critic Agent: {'aggregate': 4, 'accuracy': 4, 'logic': 4, 'grounding': 4, 'clarity': 4, 'should_reject': False}
---

## Item 17: Active_recall
**Bad Data:** True
**Topic:** Unknown Document > Unclassified Data

### Source Text
```text
x % $ & * ( ) ) 
  table 1 1 2 3 

```

> **Result:** Agent refused to generate flashcards.

### Scores
- Code Graders: {'has_flashcards': False, 'valid_count': False, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
---

## Item 18: Active_recall
**Bad Data:** True
**Topic:** Unknown Document > Unclassified Data

### Source Text
```text
The
```

> **Result:** Agent refused to generate flashcards.

### Scores
- Code Graders: {'has_flashcards': False, 'valid_count': False, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
---

## Item 19: Active_recall
**Bad Data:** True
**Topic:** Unknown Document > Unclassified Data

### Source Text
```text
This page intentionally left blank.
```

> **Result:** Agent refused to generate flashcards.

### Scores
- Code Graders: {'has_flashcards': False, 'valid_count': False, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
---

## Item 20: Active_recall
**Bad Data:** True
**Topic:** Unknown Document > Unclassified Data

### Source Text
```text
abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234abcd1234
```

> **Result:** Agent refused to generate flashcards.

### Scores
- Code Graders: {'has_flashcards': False, 'valid_count': False, 'correct_type': True, 'valid_rubric': True}
- Basic LLM Judge: {'is_grounded': False, 'is_self_contained': False}
---

