# thiessen: Bayesian Additive Voronoi Tessellations

Bayesian regression on a sum of Voronoi tessellations (AddiVortes; Stone
and Gosling, 2025,
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)
), a variant of BART (Chipman, George and McCulloch, 2010,
[doi:10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285) ) in which
a cell is a region of the covariate space rather than a box. Provides
the Gaussian, binary probit and heteroscedastic models, with methods for
prediction, posterior intervals and summaries. Further outcome families
are experimental and reach a fit only in a source build made with the
environment variable 'THIESSEN_EXPERIMENTAL' set to 1. The sampler is
the 'thiessen' Rust crate, built and linked from vendored sources.

## References

Chipman, H. A., George, E. I. and McCulloch, R. E. (2010). BART:
Bayesian additive regression trees. *The Annals of Applied Statistics*
4(1), 266-298.
[doi:10.1214/09-AOAS285](https://doi.org/10.1214/09-AOAS285)

Stone, A. and Gosling, J. P. (2025). AddiVortes: (Bayesian) additive
Voronoi tessellations. *Journal of Computational and Graphical
Statistics* 34(3), 859-871.
[doi:10.1080/10618600.2024.2414104](https://doi.org/10.1080/10618600.2024.2414104)

## See also

Useful links:

- <https://l-thomson.github.io/thiessen/r/>

- <https://github.com/l-thomson/thiessen>

- Report bugs at <https://github.com/l-thomson/thiessen/issues>

## Author

**Maintainer**: Leo Thomson <hello@leothomson.dev>

Authors:

- Leo Thomson <hello@leothomson.dev>

Other contributors:

- The authors of the vendored Rust crates (listed in inst/AUTHORS)
  \[copyright holder\]
