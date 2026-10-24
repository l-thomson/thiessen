//! The crate's error type.

/// Every error the crate returns. Input validation is an error, never a
/// panic.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Error {
    /// A hyperparameter has an invalid value.
    #[error("invalid hyperparameter `{name}`: {reason}")]
    InvalidHyperparameter {
        /// Field name as it appears on [`Config`](crate::Config).
        name: String,
        /// What was expected and what was found.
        reason: String,
    },
    /// A [`Data`](crate::Data) constructor was given values that cannot form
    /// the requested matrix.
    #[error("invalid data shape: {reason}")]
    InvalidDataShape {
        /// Buffer length against shape, or ragged rows.
        reason: String,
    },
    /// The response and the design matrix disagree on the number of rows.
    #[error("response has {y_len} rows but the design matrix has {x_rows}")]
    RowCountMismatch {
        /// Length of the response.
        y_len: usize,
        /// Rows of the design matrix.
        x_rows: usize,
    },
    /// Fewer observations than the sampler needs.
    #[error("found {found} observations but at least {required} are required")]
    InsufficientObservations {
        /// Rows supplied.
        found: usize,
        /// Rows required.
        required: usize,
    },
    /// A response value is NaN or infinite.
    #[error("response value in row {row} is not finite")]
    NonFiniteResponse {
        /// Zero-based row.
        row: usize,
    },
    /// A design-matrix value is NaN or infinite.
    #[error("design value at row {row}, column {col} is not finite")]
    NonFiniteFeature {
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        col: usize,
    },
    /// The response is constant.
    #[error("response is constant (zero variance)")]
    DegenerateResponse,
    /// A design column is constant.
    #[error("feature in column {col} is constant (zero range)")]
    DegenerateFeature {
        /// Zero-based column.
        col: usize,
    },
    /// Prediction input has a different number of columns from the fitted
    /// model.
    #[error("model expected {expected} features but got {found}")]
    FeatureCountMismatch {
        /// Columns the model was fitted on.
        expected: usize,
        /// Columns supplied.
        found: usize,
    },
    /// A quantile probability or interval level is outside (0, 1).
    #[error("probability {value} is not in the open interval (0, 1)")]
    InvalidProbability {
        /// The offending value.
        value: f64,
    },
    /// A saved model failed validation on load.
    #[error("invalid saved model: {reason}")]
    InvalidSavedModel {
        /// The invariant that failed.
        reason: String,
    },
    /// The design matrix has no columns.
    #[error("design matrix has no columns")]
    NoFeatures,
    /// A probit response value is not 0 or 1.
    #[error("response value in row {row} is not 0 or 1")]
    InvalidLabel {
        /// Zero-based row.
        row: usize,
    },
    /// A method has no meaning under the fitted model.
    #[error("`{method}` is not defined under the {model} model")]
    NotApplicable {
        /// The method called.
        method: String,
        /// The fitted model's name.
        model: String,
    },
    /// An option is compiled only with a Cargo feature this build lacks.
    #[error("{item} requires the `{feature}` feature")]
    RequiresFeature {
        /// The option named as the caller supplied it.
        item: String,
        /// The Cargo feature.
        feature: &'static str,
    },
    /// A survival time is not finite and positive.
    #[cfg(feature = "experimental")]
    #[error("survival time at row {row} must be finite and positive")]
    InvalidSurvivalTime {
        /// Zero-based row.
        row: usize,
    },
    /// The event indicator and the times differ in length.
    #[cfg(feature = "experimental")]
    #[error("event indicator has {events} rows but the times have {times}")]
    EventCountMismatch {
        /// Length of the event indicator.
        events: usize,
        /// Length of the times.
        times: usize,
    },
    /// A response value lies beyond a declared censoring limit.
    #[cfg(feature = "experimental")]
    #[error("response value at row {row} lies beyond the censoring limit")]
    ResponseBeyondLimit {
        /// Zero-based row.
        row: usize,
    },
    /// The lower and the upper bound vectors differ in length.
    #[cfg(feature = "experimental")]
    #[error("lower bounds have {lower} rows but the upper bounds have {upper}")]
    BoundCountMismatch {
        /// Length of the lower bounds.
        lower: usize,
        /// Length of the upper bounds.
        upper: usize,
    },
    /// A pair of censoring bounds does not form a valid interval.
    #[cfg(feature = "experimental")]
    #[error("censoring interval at row {row} is not valid")]
    InvalidInterval {
        /// Zero-based row.
        row: usize,
    },
    /// An ordinal response value is not one of the category codes.
    #[cfg(feature = "experimental")]
    #[error("response value in row {row} is not an integer category code below {categories}")]
    InvalidOrdinalLabel {
        /// Zero-based row.
        row: usize,
        /// The category count K of the configuration.
        categories: usize,
    },
    /// A value in a categorical column is not an integer level code.
    #[error("design value at row {row}, column {col} is not an integer level code")]
    InvalidCategoryCode {
        /// Zero-based row.
        row: usize,
        /// Zero-based column.
        col: usize,
    },
    /// Chains given to [`Fitted::pool`](crate::Fitted::pool) were not
    /// fitted the same way.
    #[error("chains cannot be pooled: {reason}")]
    MismatchedChains {
        /// The invariant that failed.
        reason: String,
    },
    /// A chain state given to [`Sampler::restore`](crate::Sampler::restore)
    /// or [`Sampler::start_from`](crate::Sampler::start_from) does not
    /// belong to the sampler's model and training data.
    #[error("the state does not fit this sampler: {reason}")]
    IncompatibleState {
        /// The mismatch.
        reason: String,
    },
}

/// `Result<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

pub(crate) fn invalid(name: &str, reason: impl Into<String>) -> Error {
    Error::InvalidHyperparameter {
        name: name.into(),
        reason: reason.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_messages() {
        let cases = [
            (
                Error::InvalidHyperparameter {
                    name: "m".into(),
                    reason: "must be at least 1".into(),
                },
                "invalid hyperparameter `m`: must be at least 1",
            ),
            (
                Error::InvalidDataShape {
                    reason: "6 values cannot form a 2 by 4 matrix".into(),
                },
                "invalid data shape: 6 values cannot form a 2 by 4 matrix",
            ),
            (
                Error::RowCountMismatch {
                    y_len: 10,
                    x_rows: 12,
                },
                "response has 10 rows but the design matrix has 12",
            ),
            (
                Error::InsufficientObservations {
                    found: 1,
                    required: 2,
                },
                "found 1 observations but at least 2 are required",
            ),
            (Error::NoFeatures, "design matrix has no columns"),
            (
                Error::NonFiniteResponse { row: 3 },
                "response value in row 3 is not finite",
            ),
            (
                Error::NonFiniteFeature { row: 3, col: 7 },
                "design value at row 3, column 7 is not finite",
            ),
            (
                Error::DegenerateResponse,
                "response is constant (zero variance)",
            ),
            (
                Error::DegenerateFeature { col: 2 },
                "feature in column 2 is constant (zero range)",
            ),
            (
                Error::FeatureCountMismatch {
                    expected: 5,
                    found: 4,
                },
                "model expected 5 features but got 4",
            ),
            (
                Error::InvalidProbability { value: 1.5 },
                "probability 1.5 is not in the open interval (0, 1)",
            ),
            (
                Error::InvalidSavedModel {
                    reason: "no draws".into(),
                },
                "invalid saved model: no draws",
            ),
            (
                Error::InvalidLabel { row: 4 },
                "response value in row 4 is not 0 or 1",
            ),
            (
                Error::InvalidCategoryCode { row: 4, col: 1 },
                "design value at row 4, column 1 is not an integer level code",
            ),
            (
                Error::RequiresFeature {
                    item: "model `robust`".into(),
                    feature: "experimental",
                },
                "model `robust` requires the `experimental` feature",
            ),
            (
                Error::NotApplicable {
                    method: "prediction_interval".into(),
                    model: "probit".into(),
                },
                "`prediction_interval` is not defined under the probit model",
            ),
            (
                Error::MismatchedChains {
                    reason: "configurations differ".into(),
                },
                "chains cannot be pooled: configurations differ",
            ),
        ];
        for (error, message) in cases {
            assert_eq!(error.to_string(), message);
        }
    }
}
