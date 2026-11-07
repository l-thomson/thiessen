# Binding parity

The core's configuration format against its place on each binding's
surface. Rendered by `tools/parity_table.py` from the core's serialised
defaults. The Python suite renders it again and fails on any difference,
and each binding's parity test proves every listed option constructible,
so this file is regenerated, never edited.

The rows are the published surface: an extension built with the core's
`experimental` feature carries the further outcome families and component
options of `docs/experimental.md`, whose table names the entry point of
each; this table does not list them. R's `outcome` defaults to `NULL`
and Python's to `None`, the family then taken from the response.

The serialised names, and so the stored configuration, are the same on
every surface. Three constructors depart from them:

- The outcome constructors carry an `_outcome` suffix in R: `gaussian()`
  there would mask the exported `stats::gaussian()`, and every family
  takes the suffix rather than only the one that clashes.
- The Rust builder sets the two ensemble sizes with `Config::with_m` and
  `Config::with_m_var`, the paper's m and m'. Every other option has a
  `with_` method of the field's name, which sets a field the two
  ensembles share on both.
- The one R shortcut, `thiessen_control(tessellations =)`, sets
  `mean_params.tessellations`.

Python groups run-length settings flat on the estimator and `Model`; R
groups them in `general_params()`.

| Core option | Python | R |
| --- | --- | --- |
| `outcome.gaussian.nu` | `gaussian(nu=)` in `outcome=` | `gaussian_outcome(nu = )` in `outcome = ` |
| `outcome.gaussian.q` | `gaussian(q=)` in `outcome=` | `gaussian_outcome(q = )` in `outcome = ` |
| `outcome.probit.offset` | `probit(offset=)` in `outcome=` | `probit_outcome(offset = )` in `outcome = ` |
| `mean_params.tessellations` | `TermParams(tessellations=)` in `mean_params=` | `term_params(tessellations = )` in `mean_params = ` |
| `mean_params.k` | `TermParams(k=)` in `mean_params=` | `term_params(k = )` in `mean_params = ` |
| `mean_params.lambda_c` | `TermParams(lambda_c=)` in `mean_params=` | `term_params(lambda_c = )` in `mean_params = ` |
| `mean_params.geometry.metric` | `GeometryParams(metric=)` in `mean_params=` | `geometry_params(metric = )` in `mean_params = ` |
| `mean_params.geometry.sigma_c` | `GeometryParams(sigma_c=)` in `mean_params=` | `geometry_params(sigma_c = )` in `mean_params = ` |
| `mean_params.structure.omega` | `StructureParams(omega=)` in `mean_params=` | `structure_params(omega = )` in `mean_params = ` |
| `variance_params.tessellations` | `TermParams(tessellations=)` in `variance_params=` | `term_params(tessellations = )` in `variance_params = ` |
| `variance_params.k` | `TermParams(k=)` in `variance_params=` | `term_params(k = )` in `variance_params = ` |
| `variance_params.lambda_c` | `TermParams(lambda_c=)` in `variance_params=` | `term_params(lambda_c = )` in `variance_params = ` |
| `variance_params.geometry.metric` | `GeometryParams(metric=)` in `variance_params=` | `geometry_params(metric = )` in `variance_params = ` |
| `variance_params.geometry.sigma_c` | `GeometryParams(sigma_c=)` in `variance_params=` | `geometry_params(sigma_c = )` in `variance_params = ` |
| `variance_params.structure.omega` | `StructureParams(omega=)` in `variance_params=` | `structure_params(omega = )` in `variance_params = ` |
| `general_params.burn_in` | `burn_in=` on `Model` and the estimators | `general_params(burn_in = )` |
| `general_params.draws` | `draws=` on `Model` and the estimators | `general_params(draws = )` |
| `general_params.thinning` | `thinning=` on `Model` and the estimators | `general_params(thinning = )` |
| `general_params.prior_only` | `prior_only=` on `Model` and the estimators | `general_params(prior_only = )` |

Groups without a row: `mean_params.cell` and `variance_params.cell` carry
no field on the stable surface; the within-cell basis is experimental,
`cell_params(basis = )` in R and `CellParams(basis=)` in Python
(`docs/experimental.md`).

The seed and the thread count are not part of the configuration. The seed
is `random_state` in Python and `seed` in R, resolved by each language's
rule and passed to the core unchanged. The thread count is `n_threads` on
Python's `Model`, `n_jobs` on its estimators (the scikit-learn name) and
`threads` in R.
