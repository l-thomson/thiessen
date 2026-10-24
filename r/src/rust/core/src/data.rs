//! The design matrix (`Data`), fit-time warnings and boundary validation.

use std::sync::OnceLock;

use crate::error::{invalid, Error, Result};

/// A row-major n by p matrix of finite or non-finite `f64` values. Callers
/// pass raw, unscaled data; scaling is the sampler's.
#[derive(Debug, Clone)]
pub struct Data {
    values: Vec<f64>,
    n_rows: usize,
    n_cols: usize,
    /// Column-major mirror of `values`, built on first use.
    columns: OnceLock<Vec<f64>>,
}

impl PartialEq for Data {
    fn eq(&self, other: &Self) -> bool {
        self.values == other.values && self.n_rows == other.n_rows && self.n_cols == other.n_cols
    }
}

impl Data {
    /// A matrix from a row-major buffer of `n_rows * n_cols` values.
    ///
    /// # Errors
    ///
    /// `InvalidDataShape` when the buffer length does not match the shape.
    pub fn new(values: Vec<f64>, n_rows: usize, n_cols: usize) -> Result<Self> {
        if values.len() != n_rows * n_cols {
            return Err(Error::InvalidDataShape {
                reason: format!(
                    "{} values cannot form a {n_rows} by {n_cols} matrix",
                    values.len()
                ),
            });
        }
        Ok(Self {
            values,
            n_rows,
            n_cols,
            columns: OnceLock::new(),
        })
    }

    /// A matrix from rows of equal length.
    ///
    /// # Errors
    ///
    /// `InvalidDataShape` for ragged rows.
    pub fn from_rows<R: AsRef<[f64]>>(rows: &[R]) -> Result<Self> {
        let n_cols = rows.first().map_or(0, |r| r.as_ref().len());
        let mut values = Vec::with_capacity(rows.len() * n_cols);
        for (i, row) in rows.iter().enumerate() {
            let row = row.as_ref();
            if row.len() != n_cols {
                return Err(Error::InvalidDataShape {
                    reason: format!("row {i} has {} values but row 0 has {n_cols}", row.len()),
                });
            }
            values.extend_from_slice(row);
        }
        Ok(Self {
            values,
            n_rows: rows.len(),
            n_cols,
            columns: OnceLock::new(),
        })
    }

    /// Number of rows n.
    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    /// Number of columns p.
    pub fn n_cols(&self) -> usize {
        self.n_cols
    }

    /// Row `i`.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.values[i * self.n_cols..(i + 1) * self.n_cols]
    }

    /// The row-major buffer.
    pub fn values(&self) -> &[f64] {
        &self.values
    }

    /// The values column-major, `n_rows` per column, built on first use
    /// and kept for the life of the matrix.
    pub(crate) fn columns(&self) -> &[f64] {
        self.columns.get_or_init(|| {
            let mut columns = vec![0.0; self.values.len()];
            if self.n_cols > 0 {
                for (i, row) in self.values.chunks_exact(self.n_cols).enumerate() {
                    for (j, &v) in row.iter().enumerate() {
                        columns[j * self.n_rows + i] = v;
                    }
                }
            }
            columns
        })
    }

    /// Rows `range` as a matrix of their own.
    pub(crate) fn rows(&self, range: std::ops::Range<usize>) -> Self {
        Self {
            values: self.values[range.start * self.n_cols..range.end * self.n_cols].to_vec(),
            n_rows: range.len(),
            n_cols: self.n_cols,
            columns: OnceLock::new(),
        }
    }

    /// The buffer and shape.
    pub fn into_parts(self) -> (Vec<f64>, usize, usize) {
        (self.values, self.n_rows, self.n_cols)
    }
}

/// A non-fatal condition noticed at fit time, returned through
/// [`Fitted::warnings`](crate::Fitted::warnings); never printed.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Warning {
    /// More covariates than observations.
    MoreFeaturesThanObservations {
        /// Number of covariates.
        p: usize,
        /// Number of observations.
        n: usize,
    },
    /// A warm start asked for more chains than the previous fit holds
    /// draws up to the starting draw, so every chain starts from that
    /// draw.
    WarmStartDrawsReused {
        /// Number of chains.
        chains: usize,
        /// Draws available, counting back from the starting draw.
        draws: usize,
    },
}

impl std::fmt::Display for Warning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Warning::MoreFeaturesThanObservations { p, n } => {
                write!(f, "more features ({p}) than observations ({n})")
            }
            Warning::WarmStartDrawsReused { chains, draws } => write!(
                f,
                "{chains} chains but only {draws} draws to start them from; \
                 every chain starts from the same draw"
            ),
        }
    }
}

/// Fit-boundary validation, in this order: `RowCountMismatch`,
/// `InsufficientObservations` (n >= 2), `NoFeatures` (p >= 1),
/// `NonFiniteResponse`, `NonFiniteFeature` (row-major, first offence),
/// `DegenerateResponse`, `DegenerateFeature`, `InvalidHyperparameter` for
/// omega outside (0, p] when p >= 2.
pub(crate) fn validate_fit(x: &Data, y: &[f64], omega: f64) -> Result<()> {
    if y.len() != x.n_rows {
        return Err(Error::RowCountMismatch {
            y_len: y.len(),
            x_rows: x.n_rows,
        });
    }
    if x.n_rows < 2 {
        return Err(Error::InsufficientObservations {
            found: x.n_rows,
            required: 2,
        });
    }
    if x.n_cols == 0 {
        return Err(Error::NoFeatures);
    }
    if let Some(row) = y.iter().position(|v| !v.is_finite()) {
        return Err(Error::NonFiniteResponse { row });
    }
    scan_finite(x)?;
    if y.iter().all(|&v| v == y[0]) {
        return Err(Error::DegenerateResponse);
    }
    for col in 0..x.n_cols {
        let first = x.values[col];
        if (1..x.n_rows).all(|r| x.values[r * x.n_cols + col] == first) {
            return Err(Error::DegenerateFeature { col });
        }
    }
    // omega / p is the inclusion probability theta of the Binomial(p - 1,
    // theta) dimension-count prior; omega > p gives theta > 1. At p = 1 the
    // dimension count is fixed and omega is not read.
    let p = x.n_cols;
    if p >= 2 && (omega.is_nan() || omega > p as f64) {
        return Err(invalid(
            "omega",
            format!("must not exceed the number of features p = {p}, got {omega}"),
        ));
    }
    Ok(())
}

/// Predict-boundary validation: `FeatureCountMismatch`, then
/// `NonFiniteFeature`. An empty matrix is valid at predict.
pub(crate) fn validate_predict(x: &Data, expected_cols: usize) -> Result<()> {
    if x.n_cols != expected_cols {
        return Err(Error::FeatureCountMismatch {
            expected: expected_cols,
            found: x.n_cols,
        });
    }
    scan_finite(x)
}

pub(crate) fn fit_warnings(x: &Data) -> Vec<Warning> {
    let mut warnings = Vec::new();
    if x.n_cols > x.n_rows {
        warnings.push(Warning::MoreFeaturesThanObservations {
            p: x.n_cols,
            n: x.n_rows,
        });
    }
    warnings
}

fn scan_finite(x: &Data) -> Result<()> {
    if let Some(offset) = x.values.iter().position(|v| !v.is_finite()) {
        return Err(Error::NonFiniteFeature {
            row: offset / x.n_cols,
            col: offset % x.n_cols,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Data, Vec<f64>) {
        let x = Data::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0], 3, 2).unwrap();
        (x, vec![0.5, 1.5, 2.5])
    }

    #[test]
    fn construction() {
        assert!(Data::new(vec![0.0; 6], 2, 4).is_err());
        assert!(Data::from_rows(&[vec![1.0, 2.0], vec![3.0]]).is_err());
        let x = Data::from_rows(&[[1.0, 2.0], [3.0, 4.0]]).unwrap();
        assert_eq!((x.n_rows(), x.n_cols()), (2, 2));
        assert_eq!(x.row(1), &[3.0, 4.0]);
        let empty = Data::new(vec![], 0, 3).unwrap();
        assert_eq!(empty.n_rows(), 0);
    }

    #[test]
    fn fit_checks_in_order() {
        let (x, y) = fixture();
        assert!(validate_fit(&x, &y, 1.0).is_ok());
        assert_eq!(
            validate_fit(&x, &[1.0], 1.0).unwrap_err(),
            Error::RowCountMismatch {
                y_len: 1,
                x_rows: 3
            }
        );
        let one = Data::new(vec![1.0, 2.0], 1, 2).unwrap();
        assert!(matches!(
            validate_fit(&one, &[1.0], 1.0).unwrap_err(),
            Error::InsufficientObservations { found: 1, .. }
        ));
        let no_cols = Data::new(vec![], 3, 0).unwrap();
        assert_eq!(
            validate_fit(&no_cols, &y, 1.0).unwrap_err(),
            Error::NoFeatures
        );
        assert_eq!(
            validate_fit(&x, &[0.5, f64::NAN, 2.5], 1.0).unwrap_err(),
            Error::NonFiniteResponse { row: 1 }
        );
        let bad = Data::new(vec![1.0, 10.0, 2.0, f64::NAN, f64::NAN, 30.0], 3, 2).unwrap();
        assert_eq!(
            validate_fit(&bad, &y, 1.0).unwrap_err(),
            Error::NonFiniteFeature { row: 1, col: 1 }
        );
        assert_eq!(
            validate_fit(&x, &[7.0, 7.0, 7.0], 1.0).unwrap_err(),
            Error::DegenerateResponse
        );
        let constant = Data::new(vec![1.0, 10.0, 1.0, 20.0, 1.0, 30.0], 3, 2).unwrap();
        assert_eq!(
            validate_fit(&constant, &y, 1.0).unwrap_err(),
            Error::DegenerateFeature { col: 0 }
        );
    }

    #[test]
    fn omega_must_not_exceed_p() {
        let (x, y) = fixture();
        assert!(matches!(
            validate_fit(&x, &y, 2.5).unwrap_err(),
            Error::InvalidHyperparameter { ref name, .. } if name == "omega"
        ));
        assert!(validate_fit(&x, &y, 2.0).is_ok());
        assert!(validate_fit(&x, &y, f64::NAN).is_err());
        let one_col = Data::new(vec![1.0, 2.0, 3.0], 3, 1).unwrap();
        assert!(validate_fit(&one_col, &y, 3.0).is_ok());
    }

    #[test]
    fn predict_checks() {
        let (x, _) = fixture();
        assert_eq!(
            validate_predict(&x, 5).unwrap_err(),
            Error::FeatureCountMismatch {
                expected: 5,
                found: 2
            }
        );
        let constant = Data::new(vec![1.0, 1.0], 1, 2).unwrap();
        assert!(validate_predict(&constant, 2).is_ok());
    }

    #[test]
    fn warnings() {
        let x = Data::new(vec![0.0; 6], 2, 3).unwrap();
        assert_eq!(
            fit_warnings(&x),
            vec![Warning::MoreFeaturesThanObservations { p: 3, n: 2 }]
        );
        assert_eq!(
            fit_warnings(&x)[0].to_string(),
            "more features (3) than observations (2)"
        );
        assert!(fit_warnings(&Data::new(vec![0.0; 6], 3, 2).unwrap()).is_empty());
    }

    mod props {
        use proptest::prelude::*;

        use super::*;

        proptest! {
            #[test]
            fn constructors_and_validators_never_panic(
                values in prop::collection::vec(prop::num::f64::ANY, 0..64),
                n_cols in 1usize..8,
                y in prop::collection::vec(prop::num::f64::ANY, 0..12),
                omega in prop::num::f64::ANY,
                expected_cols in 0usize..8,
            ) {
                let n_rows = values.len() / n_cols;
                let x = Data::new(values[..n_rows * n_cols].to_vec(), n_rows, n_cols).unwrap();
                let _ = validate_fit(&x, &y, omega);
                let _ = validate_predict(&x, expected_cols);
                let _ = fit_warnings(&x);
            }
        }
    }
}
