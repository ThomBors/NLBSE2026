
# Problem Formulation

We have a set of labels \(l \in L\) with current counts:

\[
P_l = \text{number of positive instances for label } l
\]  
\[
N_l = \text{number of negative instances for label } l
\]

We want to achieve a target positive-to-negative ratio:

\[
x = \frac{P_l^\text{new}}{N_l^\text{new}}
\]

**Constraint:** Adding a positive instance to one label increases negatives for all other labels. Let \(a_l\) be the number of positives we add to label \(l\). Then:

\[
P_l^\text{new} = P_l + a_l
\]  
\[
N_l^\text{new} = N_l + \sum_{j \neq l} a_j
\]

The ratio equation becomes:

\[
\frac{P_l + a_l}{N_l + \sum_{j \neq l} a_j} = x, \quad \forall l \in L
\]

---

# Iterative Algorithm

1. Initialize \(a_l = 0\) for all labels \(l \in L\).  
2. Repeat until all ratios \(\ge x\):

\[
\text{for each label } l: \quad
\text{if } \frac{P_l + a_l}{N_l + \sum_{j \neq l} a_j} < x, \quad a_l \leftarrow a_l + 1
\]

3. Compute final counts:

\[
P_l^\text{final} = P_l + a_l
\]  
\[
N_l^\text{final} = N_l + \sum_{j \neq l} a_j
\]
