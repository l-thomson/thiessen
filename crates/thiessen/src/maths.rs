//! Transcendental functions through `libm`, so the reference target does not
//! depend on the system libc, and the special functions the prior
//! calibration and the predictive distribution use.

pub(crate) fn ln(x: f64) -> f64 {
    libm::log(x)
}

#[cfg(feature = "experimental")]
pub(crate) fn ln_1p(x: f64) -> f64 {
    libm::log1p(x)
}

pub(crate) fn exp(x: f64) -> f64 {
    libm::exp(x)
}

pub(crate) fn erfc(x: f64) -> f64 {
    libm::erfc(x)
}

pub(crate) fn lgamma(x: f64) -> f64 {
    libm::lgamma(x)
}

pub(crate) fn sin(x: f64) -> f64 {
    libm::sin(x)
}

pub(crate) fn cos(x: f64) -> f64 {
    libm::cos(x)
}

pub(crate) fn acos(x: f64) -> f64 {
    libm::acos(x)
}

#[cfg(feature = "experimental")]
pub(crate) fn powf(x: f64, y: f64) -> f64 {
    libm::pow(x, y)
}

/// Standard normal CDF, Phi(z) = erfc(-z / sqrt 2) / 2.
pub(crate) fn normal_cdf(z: f64) -> f64 {
    0.5 * erfc(-z * std::f64::consts::FRAC_1_SQRT_2)
}

/// Standard normal quantile Phi^-1(p), p in (0, 1), by bisection on
/// [`normal_cdf`] over [-40, 40] (where Phi is 0 or 1 to double precision).
pub(crate) fn normal_quantile(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    let (mut lo, mut hi) = (-40.0_f64, 40.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if normal_cdf(mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Regularised lower incomplete gamma function P(a, x), a > 0, x >= 0:
/// series for x < a + 1, continued fraction otherwise (Press et al. 2007,
/// s. 6.2).
pub(crate) fn gamma_p(a: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && x >= 0.0);
    if x == 0.0 {
        return 0.0;
    }
    let log_prefactor = a * ln(x) - x - lgamma(a);
    if x < a + 1.0 {
        let mut term = 1.0 / a;
        let mut sum = term;
        let mut denominator = a;
        for _ in 0..500 {
            denominator += 1.0;
            term *= x / denominator;
            sum += term;
            if term.abs() < sum.abs() * 1e-16 {
                break;
            }
        }
        sum * exp(log_prefactor)
    } else {
        const TINY: f64 = 1e-300;
        let mut b = x + 1.0 - a;
        let mut c = 1.0 / TINY;
        let mut d = 1.0 / b;
        let mut h = d;
        for i in 1..500 {
            let an = -(i as f64) * (i as f64 - a);
            b += 2.0;
            d = an * d + b;
            if d.abs() < TINY {
                d = TINY;
            }
            c = b + an / c;
            if c.abs() < TINY {
                c = TINY;
            }
            d = 1.0 / d;
            let delta = d * c;
            h *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        1.0 - exp(log_prefactor) * h
    }
}

/// Quantile of chi^2_nu at probability p in (0, 1), by bisection on
/// [`gamma_p`].
pub(crate) fn chi2_quantile(p: f64, nu: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0 && nu > 0.0);
    let a = 0.5 * nu;
    let mut hi = nu.max(1.0);
    while gamma_p(a, 0.5 * hi) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if gamma_p(a, 0.5 * mid) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Continued fraction of the regularised incomplete beta function
/// (Press et al. 2007, s. 6.4, modified Lentz).
#[cfg(feature = "experimental")]
fn beta_cf(a: f64, b: f64, x: f64) -> f64 {
    const TINY: f64 = 1e-300;
    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;
    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;
    for m in 1..=300 {
        let m = f64::from(m);
        let m2 = 2.0 * m;
        let numerator = m * (b - m) * x / ((qam + m2) * (a + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;
        let numerator = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2));
        d = 1.0 + numerator * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + numerator / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let delta = d * c;
        h *= delta;
        if (delta - 1.0).abs() < 1e-16 {
            break;
        }
    }
    h
}

/// Regularised incomplete beta function I_x(a, b), a > 0, b > 0, x in
/// [0, 1]: the continued fraction on whichever of x and 1 - x converges
/// fast (Press et al. 2007, s. 6.4).
#[cfg(feature = "experimental")]
pub(crate) fn beta_i(a: f64, b: f64, x: f64) -> f64 {
    debug_assert!(a > 0.0 && b > 0.0 && (0.0..=1.0).contains(&x));
    if x == 0.0 {
        return 0.0;
    }
    if x == 1.0 {
        return 1.0;
    }
    let ln_front = lgamma(a + b) - lgamma(a) - lgamma(b) + a * ln(x) + b * ln(1.0 - x);
    if x < (a + 1.0) / (a + b + 2.0) {
        exp(ln_front) * beta_cf(a, b, x) / a
    } else {
        1.0 - exp(ln_front) * beta_cf(b, a, 1.0 - x) / b
    }
}

/// Standard Student-t CDF at `t` with `nu` degrees of freedom, through
/// I_x(nu / 2, 1 / 2) at x = nu / (nu + t^2).
#[cfg(feature = "experimental")]
pub(crate) fn student_t_cdf(t: f64, nu: f64) -> f64 {
    debug_assert!(nu > 0.0);
    let tail = 0.5 * beta_i(0.5 * nu, 0.5, nu / (nu + t * t));
    if t >= 0.0 {
        1.0 - tail
    } else {
        tail
    }
}

/// Standard Student-t quantile at probability p in (0, 1) with `nu`
/// degrees of freedom, by bisection on [`student_t_cdf`] over a bracket
/// doubled until it covers p.
#[cfg(feature = "experimental")]
pub(crate) fn student_t_quantile(p: f64, nu: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0 && nu > 0.0);
    if p < 0.5 {
        return -student_t_quantile(1.0 - p, nu);
    }
    let mut hi = 1.0_f64;
    while student_t_cdf(hi, nu) < p {
        hi *= 2.0;
    }
    let mut lo = 0.0_f64;
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if student_t_cdf(mid, nu) < p {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

/// Standard Laplace CDF at `z`: exp(z) / 2 below 0, 1 - exp(-z) / 2
/// above.
#[cfg(feature = "experimental")]
pub(crate) fn laplace_cdf(z: f64) -> f64 {
    if z < 0.0 {
        0.5 * exp(z)
    } else {
        1.0 - 0.5 * exp(-z)
    }
}

/// Standard Laplace quantile at probability p in (0, 1): ln(2p) below
/// the median, -ln(2(1 - p)) above.
#[cfg(feature = "experimental")]
pub(crate) fn laplace_quantile(p: f64) -> f64 {
    debug_assert!(p > 0.0 && p < 1.0);
    if p < 0.5 {
        ln(2.0 * p)
    } else {
        -ln(2.0 * (1.0 - p))
    }
}

/// Type 7 quantile of an ascending-sorted non-empty slice: h = p (n - 1),
/// linear interpolation between neighbours.
pub(crate) fn quantile_sorted(sorted: &[f64], p: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let h = p * (n - 1) as f64;
    let lo = h.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    sorted[lo] + (sorted[hi] - sorted[lo]) * (h - lo as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() <= tol, "{a} vs {b}");
    }

    #[test]
    fn normal_cdf_reference_values() {
        close(normal_cdf(0.0), 0.5, 1e-15);
        close(normal_cdf(1.959_963_984_540_054), 0.975, 1e-12);
        close(normal_cdf(-3.0), 0.001_349_898_031_630_09, 1e-12);
    }

    #[test]
    fn normal_quantile_inverts_the_cdf() {
        close(normal_quantile(0.5), 0.0, 1e-12);
        close(normal_quantile(0.975), 1.959_963_984_540_054, 1e-9);
        close(normal_quantile(0.001_349_898_031_630_09), -3.0, 1e-9);
        close(normal_cdf(normal_quantile(0.3)), 0.3, 1e-12);
    }

    #[test]
    fn gamma_p_identities() {
        close(gamma_p(1.0, 2.0), 1.0 - exp(-2.0), 1e-14);
        close(gamma_p(0.5, 0.5), 0.682_689_492_137_086, 1e-12);
        close(gamma_p(5.0, 20.0), 0.999_983_055_256, 1e-10);
    }

    #[test]
    fn chi2_quantile_reference_values() {
        close(chi2_quantile(0.95, 1.0), 3.841_458_820_694_12, 1e-9);
        close(chi2_quantile(0.15, 6.0), 2.661_273_176_1, 1e-8);
        close(chi2_quantile(0.5, 10.0), 9.341_817_765_6, 1e-8);
    }

    /// pt and qt reference values from R 4.4.0; nu = 1 is the Cauchy
    /// with its closed forms.
    #[cfg(feature = "experimental")]
    #[test]
    fn student_t_cdf_and_quantile_reference_values() {
        close(student_t_cdf(0.0, 5.0), 0.5, 1e-15);
        close(student_t_cdf(1.0, 1.0), 0.75, 1e-12);
        close(student_t_cdf(-1.0, 1.0), 0.25, 1e-12);
        close(student_t_cdf(2.0, 10.0), 0.963_306, 1e-6);
        close(student_t_quantile(0.975, 1.0), 12.706_204_736, 1e-6);
        close(student_t_quantile(0.975, 2.0), 4.302_652_730, 1e-7);
        close(student_t_quantile(0.975, 5.0), 2.570_581_836, 1e-7);
        close(student_t_quantile(0.025, 5.0), -2.570_581_836, 1e-7);
        for p in [0.01, 0.2, 0.5, 0.9, 0.999] {
            for nu in [1.0, 3.0, 30.0] {
                close(student_t_cdf(student_t_quantile(p, nu), nu), p, 1e-12);
            }
        }
        // Large nu reproduces the Normal quantile.
        close(student_t_quantile(0.975, 1e6), normal_quantile(0.975), 1e-4);
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn laplace_cdf_and_quantile_invert() {
        close(laplace_cdf(0.0), 0.5, 1e-15);
        close(laplace_cdf(-1.0), 0.5 * (-1.0_f64).exp(), 1e-15);
        close(laplace_quantile(0.975), -(0.05_f64).ln(), 1e-14);
        for p in [0.01, 0.3, 0.5, 0.8, 0.999] {
            close(laplace_cdf(laplace_quantile(p)), p, 1e-14);
        }
    }

    #[cfg(feature = "experimental")]
    #[test]
    fn beta_i_identities() {
        // I_x(1, 1) = x; I_x(a, b) = 1 - I_{1-x}(b, a).
        close(beta_i(1.0, 1.0, 0.3), 0.3, 1e-14);
        close(beta_i(2.5, 4.0, 0.4), 1.0 - beta_i(4.0, 2.5, 0.6), 1e-14);
        close(beta_i(2.0, 3.0, 0.0), 0.0, 0.0);
        close(beta_i(2.0, 3.0, 1.0), 1.0, 0.0);
    }

    #[test]
    fn quantile_sorted_interpolates() {
        let v = [1.0, 2.0, 3.0, 4.0];
        close(quantile_sorted(&v, 0.5), 2.5, 0.0);
        close(quantile_sorted(&v, 0.0), 1.0, 0.0);
        close(quantile_sorted(&v, 1.0), 4.0, 0.0);
        close(quantile_sorted(&[7.0], 0.3), 7.0, 0.0);
    }
}
