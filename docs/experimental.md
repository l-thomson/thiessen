# Experimental items

The stable surface is the method as published: the models and components
of Stone and Gosling (2025) and of CRAN AddiVortes. Everything else the
crate adds is compiled only with the Cargo feature `experimental`, is
tested to the same standard, and is outside the semver promise: its
configuration surface and sampled values may change in any release, with
a changelog line. Enabling the feature does not change the draws of a
configuration that uses no experimental option.

The Python and R packages build the core without the feature by default,
and every released artefact ships that way. The opt-in is a build-time
one, since the feature is a `#[cfg]` in the core and an experimental
item's sampled values may change in a patch release:

```sh
THIESSEN_EXPERIMENTAL=1 R CMD INSTALL r
pip install ./python --config-settings build-args="--features experimental"
```

A binding reports its own setting, `core_experimental()` in R and
`thiessen._native.EXPERIMENTAL` in Python. A configuration or a saved fit
naming a gated item in a build without the feature is rejected by
`Config::validate` with `Error::RequiresFeature`, a saved fit through
`Fitted::load`, which keeps the error's type where the `Deserialize` impl
has only the format's text; it reaches R as the condition class
`thiessen_requires_feature` and Python as `RequiresFeatureError`. The
outcome families have a constructor in both
bindings whatever the build carries, so a script naming one is portable,
and the core's entry points for the gated models (`fit_aft`,
`Sampler::aft`, `Fitted::predict_category_probabilities` and their
siblings) exist in every build and return the same error without the
feature, so a binding wraps one surface.
A fit saved from a build with the feature does not load in one without
it.

The stabilisation rule is stated once, in the crate-root documentation
(`crates/thiessen/src/lib.rs`, Stability): graduation is a pull request
against that rule, not a ticket. This file is the status table for every
gated item; the pull-request column is the public record of each item's
history. In R, each item's help page carries the experimental badge, and
[`experimental_outcomes`](https://l-thomson.github.io/thiessen/r/reference/experimental_outcomes.html)
states the policy.

## Table

| Item | Kind | Configuration | R entry point | Python entry point | Feature since | Calibration | Status | Pull request |
|---|---|---|---|---|---|---|---|---|
| Minkowski distance (Manhattan as p = 1) | distance | `geometry.metric` entries `{"minkowski": {"p": ...}}`, `{"manhattan": {}}` | `geometry_params(metric = list(list(minkowski = list(p = 3))))`, `"manhattan"` | `GeometryParams(metric=[{"minkowski": {"p": 3.0}}])`, `"manhattan"` | 0.3.0 | conformance, small SBC at p = 1 | experimental | [#61](https://github.com/l-thomson/thiessen/pull/61) |
| Cosine distance | distance | `geometry.metric` entry `{"cosine": {}}` | `geometry_params(metric = list("cosine"))` | `GeometryParams(metric=["cosine"])` | 0.3.0 | conformance (no triangle inequality), small SBC | experimental | [#62](https://github.com/l-thomson/thiessen/pull/62) |
| Gower distance | distance | `geometry.metric` entries `{"gower": {"kind": "numeric" or "categorical"}}` | `geometry_params(metric = list(list(gower = list(kind = "numeric"))))` | `GeometryParams(metric=[{"gower": {"kind": "numeric"}}])` | 0.3.0 | conformance, small SBC | experimental | [#63](https://github.com/l-thomson/thiessen/pull/63) |
| Mahalanobis distance | distance | `geometry.metric` entry `"mahalanobis"` with `geometry.precision` | `geometry_params(metric = list("mahalanobis"), precision = P)` | `GeometryParams(metric=["mahalanobis"], precision=P)` | 0.3.0 | conformance, small SBC | experimental | [#64](https://github.com/l-thomson/thiessen/pull/64) |
| Per-column composite | distance | `group` label on the minkowski, manhattan, cosine and gower entries | `list(cosine = list(group = 1))` entries of `metric` | `{"cosine": {"group": 1}}` entries of `metric` | 0.3.0 | conformance per member metric, small SBC | experimental | [#65](https://github.com/l-thomson/thiessen/pull/65) |
| Weighted inclusion | inclusion prior | `structure.inclusion` entry `{"weighted": {"weights": [...]}}` | `structure_params(inclusion = weighted_inclusion(w))` | `StructureParams(inclusion=weighted_inclusion(w))` | 0.3.0 | conformance, small SBC | experimental | [#66](https://github.com/l-thomson/thiessen/pull/66) |
| DART inclusion | inclusion prior (model-grade validation) | `structure.inclusion` entry `{"dart": {"a": ..., "b": ..., "rho": ...}}` | `structure_params(inclusion = dart_inclusion())` | `StructureParams(inclusion=dart_inclusion())` | 0.3.0 | SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#67](https://github.com/l-thomson/thiessen/pull/67) |
| Linear cell basis | cell basis (model-grade validation) | `mean_params.cell.basis` entry `"linear"` | `term_params(cell = cell_params(basis = "linear"))` | `TermParams(cell=CellParams(basis="linear"))` | 0.3.0 | known answer; SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#68](https://github.com/l-thomson/thiessen/pull/68) |
| Soft membership | membership rule (model-grade validation) | `mean_params.geometry.membership` entry `{"soft": {"rate": ...}}` | `geometry_params(membership = soft_membership())` | `GeometryParams(membership=soft_membership())` | 0.3.0 | known answer; SBC and Geweke, both sizes; broken-sampler fixture | experimental | [#78](https://github.com/l-thomson/thiessen/pull/78) |
| Tobit outcome | outcome model (model-grade validation) | `outcome` entry `{"tobit": {"lower": ..., "upper": ...}}` | `thiessen_control(outcome = tobit_outcome(lower = 0))` | `Model(outcome=tobit(lower=0.0))` | 0.3.0 | known answer (censored-likelihood quadrature); SBC and Geweke, both sizes | experimental | [#81](https://github.com/l-thomson/thiessen/pull/81) |
| AFT outcome | outcome model (model-grade validation) | `outcome` entry `{"aft": {}}` with `fit_aft(x, times, events)` | `thiessen(x, Surv(time, event))`, `aft_outcome()` | `Model().fit(X, Surv.from_arrays(event, time))`, `aft()` | 0.3.0 | known answer (censored-likelihood quadrature); SBC and Geweke, both sizes; informational `abart` comparison | experimental | [#82](https://github.com/l-thomson/thiessen/pull/82) |
| Interval-censored outcome | outcome model (model-grade validation) | `outcome` entry `{"interval_censored": {}}` with `fit_interval_censored(x, lower, upper)` | `thiessen(x, Surv(lower, upper, type = "interval2"))`, `interval_censored_outcome()` | `Model().fit(X, np.column_stack([lower, upper]))`, `interval_censored()` | 0.3.0 | known answer (interval-likelihood quadrature); SBC and Geweke, both sizes | experimental | [#83](https://github.com/l-thomson/thiessen/pull/83) |
| Ordinal outcome | outcome model (model-grade validation) | `outcome` entry `{"ordinal": {"categories": ...}}` | `thiessen(x, ordered)`, `ordinal_outcome()`; `predict(type = "probs")` | `Model().fit(X, pd.Categorical(..., ordered=True))`, `ordinal()`; `predict_proba(X)` | 0.3.0 | known answer (cutpoint and cell-mean quadrature); SBC and Geweke, both sizes, cutpoints covered; broken-sampler fixture; full-size cutpoint ESS check | experimental | [#85](https://github.com/l-thomson/thiessen/pull/85) |
| Student-t outcome | outcome model (model-grade validation) | `outcome` entry `{"student_t": {"df": 4.0}}` or `{"student_t": {"df": [3.0, 6.0, 12.0]}}` | `thiessen_control(outcome = student_t_outcome(df = 4))` | `Model(outcome=student_t(df=4.0))` | 0.3.0 | known answer (marginal t-likelihood quadrature, fixed and grid df); SBC and Geweke, both sizes (fixed df); no acceptance-ratio term, so no broken-sampler fixture | experimental | [#98](https://github.com/l-thomson/thiessen/pull/98) |
| Laplace outcome | outcome model (model-grade validation) | `outcome` entry `{"laplace": {}}` | `thiessen_control(outcome = laplace_outcome())` | `Model(outcome=laplace())` | 0.3.0 | known answer (marginal Laplace-likelihood quadrature); SBC and Geweke, both sizes; no acceptance-ratio term, so no broken-sampler fixture | experimental | [#99](https://github.com/l-thomson/thiessen/pull/99) |

Columns: the configuration field or variant; the R and Python entry
points that reach it; the first core version carrying the item behind the
feature; calibration status (SBC and Geweke at the two sizes, or the
component conformance tests); experimental or stabilised, with the core
version of stabilisation; the pull request that added the item, its
public record.

## References

- Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
  Voronoi tessellations. Journal of Computational and Graphical
  Statistics 34(3), 859-871.
