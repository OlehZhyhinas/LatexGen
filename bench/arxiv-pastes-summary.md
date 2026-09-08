# arXiv paste benchmark summary

- items: 120
- mean chars: 445.7
- custom-macro math items: 0
- api queries: 0
- papers considered: 64
- papers with items: 50
- downloaded pdf: 0
- downloaded source: 0

## by category
- astro-ph.GA: 14
- cond-mat.stat-mech: 9
- cs.IT: 8
- cs.LG: 2
- econ.TH: 23
- hep-th: 1
- math.AP: 9
- math.CO: 1
- math.NA: 4
- math.PR: 22
- physics.class-ph: 4
- q-bio.PE: 5
- quant-ph: 3
- stat.ML: 15

## by n_display
- 0: 102
- 1: 14
- 2: 4

## by extractor
- pdftotext: 90
- pdftotext -layout: 30

## skipped papers
- no_aligned_paragraph: 3
- no_candidate_paragraph: 7
- not_latex_source: 3
- paper_error: 1

## examples

### arxiv-2609.03750-p3
source: https://arxiv.org/abs/2609.03750

input:

```
The estimates in Lemmas 3.4, 3.5, and 3.6 remain valid for ϵ = 0 after deleting the
stochastic terms, with the same constants. Hence K = {K (τ ) : τ ∈ R} is a common closed D-pullback
absorbing family for all ϵ ∈ [0, ϵ0 ). This uniformity is used in Section 4.
```

reference:

```
The estimates in Lemmas (ref), (ref), and (ref) remain valid for \( \epsilon=0 \) after deleting the stochastic terms, with the same constants. Hence \( \mathscr K=\{\mathscr K(\tau):\tau\in\mathbb R\} \) is a common closed \( \mathfrak D \) -pullback absorbing family for all \( \epsilon\in[0,\epsilon_0) \) . This uniformity is used in Section (ref).
```

### arxiv-2609.03367-p2
source: https://arxiv.org/abs/2609.03367

input:

```
Molecular hydrogen is included as a dynamically important species because it is the dominant coolant at
low temperatures T ≲ 104 K. The H2 molecule has a low excitation temperature and can efficiently radiate away
thermal energy through rotational and vibrational transitions, making it crucial for the thermal evolution of the
cloud [32, 34]. The H2 fraction evolves according to formation on dust grains, which scales with metallicity,
and destruction by UV radiation from Pop III stars [44
```

reference:

```
Molecular hydrogen is included as a dynamically important species because it is the dominant coolant at low temperatures \( T \lesssim 10^4 \) K. The H \( _2 \) molecule has a low excitation temperature and can efficiently radiate away thermal energy through rotational and vibrational transitions, making it crucial for the thermal evolution of the cloud [ref]. The H \( _2 \) fraction evolves according to formation on dust grains, which scales with metallicity, and destruction by UV radiation from Pop III stars [ref].
```

### arxiv-2609.01544-p1
source: https://arxiv.org/abs/2609.01544

input:

```
Problem (3) is also related to sparse quantile linear regression. Indeed, it is equivalent to

(
min

ω,c: ∥ω∥0 ≤d

)
n

(b + h) X
1
ρτ yi − (ω ⊤ xi + c) + ∥ω∥22 ,
n
2γ
i=1


b
where τ = b+h
and ρτ (u) = u τ − 1{u<0} . In other words, the goal of variable selection for
the feature-based newsvendor is equivalent to estimating sparse coefficients that minimize
the ℓ2 -regularized empirical quantile regression
```

reference:

```
Problem (ref) is also related to sparse quantile linear regression. Indeed, it is equivalent to \[ \min_{\omega,c:\ \|\omega\|_0\le d}
\left\{
\frac{(b+h)}{n}\sum_{i=1}^n
\rho_{\tau}\big(y_i-(\omega^\top x_i+c)\big)
+
\frac{1}{2\gamma}\|\omega\|_2^2
\right\}, \] where \( \tau=\frac{b}{b+h} \) and \( \rho_{\tau}(u)=u\big(\tau-\mathbf{1}_{\{u<0\}}\big) \) .
In other words, the goal of variable selection for the feature-based newsvendor is equivalent to estimating sparse coefficients that minimize the \( \ell_2 \) -regularized empirical quantile regression loss.
```
