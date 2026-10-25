//! `Sampler::xaddivortes` (`xaddivortes.rs`): the gate, the settings, the
//! start contract, determinism, a fixed-seed regression fixture and a
//! statistical check on Friedman's first function.

mod common;

use common::{fixture, TestRng, SEED};
use thiessen::{Data, Sampler, XAddiVortesParams};

/// Friedman (1991) function 1 on `p` uniform covariates (the last p - 5
/// irrelevant): the noiseless truth and a response with N(0, noise^2)
/// errors.
#[allow(dead_code)]
fn friedman1(n: usize, p: usize, noise: f64, seed: u64) -> (Data, Vec<f64>, Vec<f64>) {
    let mut rng = TestRng(seed);
    let rows: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..p).map(|_| rng.uniform()).collect())
        .collect();
    let truth: Vec<f64> = rows
        .iter()
        .map(|r| {
            10.0 * (std::f64::consts::PI * r[0] * r[1]).sin()
                + 20.0 * (r[2] - 0.5) * (r[2] - 0.5)
                + 10.0 * r[3]
                + 5.0 * r[4]
        })
        .collect();
    let y = truth.iter().map(|t| t + noise * rng.normal()).collect();
    (Data::from_rows(&rows).unwrap(), truth, y)
}

#[cfg(not(feature = "experimental"))]
#[test]
fn xaddivortes_requires_the_feature() {
    let (config, x, y) = fixture();
    let mut s = Sampler::new(&config, &x, &y, SEED).unwrap();
    assert!(matches!(
        s.xaddivortes(&XAddiVortesParams::default()),
        Err(thiessen::Error::RequiresFeature { .. })
    ));
}

/// The settings exist, validate and round-trip in every build, so a
/// binding wraps one surface.
#[test]
fn settings_validate_and_round_trip() {
    let d = XAddiVortesParams::default();
    d.validate().unwrap();
    let json = serde_json::to_string(&d).unwrap();
    assert_eq!(serde_json::from_str::<XAddiVortesParams>(&json).unwrap(), d);
    let partial: XAddiVortesParams = serde_json::from_str(r#"{"rounds": 7}"#).unwrap();
    assert_eq!(partial, XAddiVortesParams::default().with_rounds(7));
    assert!(serde_json::from_str::<XAddiVortesParams>(r#"{"budget": 50}"#).is_err());
    let bad = [
        d.clone().with_value_scale(0.0),
        d.clone().with_value_scale(1.5),
        d.clone().with_value_scale(f64::NAN),
        d.clone().with_replace_share(-0.1),
        d.clone().with_replace_share(1.1),
        d.clone().with_max_columns(0),
        d.clone().with_centres(0),
        d.clone().with_centres(70_000),
    ];
    for b in bad {
        let e = b.validate().unwrap_err();
        assert!(
            matches!(&e, thiessen::Error::InvalidHyperparameter { name, .. } if name.starts_with("xaddivortes.")),
            "{e}"
        );
    }
}

#[cfg(feature = "experimental")]
mod fit {
    use super::*;
    use proptest::prelude::*;
    use thiessen::Config;

    fn fit(
        config: &Config,
        x: &Data,
        y: &[f64],
        seed: u64,
        p: &XAddiVortesParams,
    ) -> (Sampler, thiessen::FastFit) {
        let mut s = Sampler::new(config, x, y, seed).unwrap();
        let f = s.xaddivortes(p).unwrap();
        (s, f)
    }

    fn rmse(a: &[f64], b: &[f64]) -> f64 {
        (a.iter().zip(b).map(|(u, v)| (u - v) * (u - v)).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// The fit is a valid chain state: m tessellations, finite values, the
    /// configured rounds; the same for the same seed and different for
    /// another.
    #[test]
    fn fit_is_valid_and_reproducible() {
        let (config, x, y) = fixture();
        let p = XAddiVortesParams::default();
        let (a, fa) = fit(&config, &x, &y, SEED, &p);
        let (b, fb) = fit(&config, &x, &y, SEED, &p);
        let (c, _) = fit(&config, &x, &y, SEED + 1, &p);
        assert_eq!(a.tessellations(), b.tessellations());
        assert_eq!(fa, fb);
        assert_ne!(a.tessellations(), c.tessellations());
        assert_eq!(a.tessellations().len(), 15);
        assert_eq!(fa.train_score.len(), p.rounds + 2);
        assert!(fa.sigma.is_finite() && fa.sigma > 0.0);
        assert_eq!(fa.column_scores.len(), 2);
        assert!(a.fitted_values().iter().all(|v| v.is_finite()));
    }

    /// The training score has one entry after the build, one per round and
    /// one after the final refit, and ends at the training RMSE of the kept
    /// fit's prediction; sigma, corrected for the fit's degrees of freedom,
    /// is at least that.
    #[test]
    fn train_score_and_sigma() {
        let (config, x, y) = fixture();
        let (mut s, f) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
        assert_eq!(f.train_score.len(), XAddiVortesParams::default().rounds + 2);
        s.keep();
        let pred = s.finish().unwrap().predict(&x).unwrap();
        let train = rmse(&pred, &y);
        let last = *f.train_score.last().unwrap();
        assert!(
            (last - train).abs() < 1e-9 * (1.0 + train),
            "{last} vs {train}"
        );
        assert!(
            f.sigma >= train,
            "sigma {} below the training RMSE {train}",
            f.sigma
        );
    }

    /// The exact sampler runs on from the fit (the warm start), and the
    /// chain is reproducible.
    #[test]
    fn warm_start_runs_on() {
        let (config, x, y) = fixture();
        let run = || {
            let params = XAddiVortesParams::default().with_rounds(5);
            let (mut s, f) = fit(&config, &x, &y, SEED, &params);
            assert_eq!(f.train_score.len(), 5 + 2);
            for _ in 0..10 {
                s.step();
                s.keep();
            }
            (s.tessellations().to_vec(), s.sigma_sq())
        };
        assert_eq!(run(), run());
    }

    /// The one-call fast fit equals the sampler path on the chain-0 seed.
    #[test]
    fn fit_xaddivortes_equals_the_sampler_path() {
        let (config, x, y) = fixture();
        let params = XAddiVortesParams::default();
        let (fitted, fast) = thiessen::fit_xaddivortes(&config, &x, &y, SEED, &params).unwrap();
        let (mut s, f) = fit(&config, &x, &y, thiessen::chain_seed(SEED, 0), &params);
        s.keep();
        assert_eq!(fast, f);
        assert_eq!(
            fitted.predict(&x).unwrap(),
            s.finish().unwrap().predict(&x).unwrap()
        );
        assert_eq!(fitted.n_draws(), 1);
    }

    /// One column fits, and a tied, two-valued response is screened
    /// without error.
    #[test]
    fn one_column_and_a_two_valued_response_fit() {
        let rows: Vec<[f64; 1]> = (0..120).map(|i| [i as f64 / 120.0]).collect();
        let x = Data::from_rows(&rows).unwrap();
        let y: Vec<f64> = rows.iter().map(|r| f64::from(r[0] > 0.4)).collect();
        let config = Config::new().with_m(20);
        let (s, f) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
        assert!(s.tessellations().iter().all(|t| t.dims() == [0]));
        assert_eq!(f.column_scores.len(), 1);
        assert!(f.sigma.is_finite());
    }

    /// The cross-binding fixture: `fit_xaddivortes` on the shared fixture
    /// with seed 7 and default settings, its predictions at the training
    /// rows. The Python package tests against the same file, so the two
    /// surfaces agree bit for bit. `XADDI_WRITE_FIXTURE=1` rewrites it.
    #[test]
    fn cross_binding_fixture() {
        let (config, x, y) = fixture();
        let (fitted, fast) =
            thiessen::fit_xaddivortes(&config, &x, &y, SEED, &XAddiVortesParams::default())
                .unwrap();
        let mut text = format!("sigma {:?}\n", fast.sigma);
        for v in fitted.predict(&x).unwrap() {
            text.push_str(&format!("{v:?}\n"));
        }
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/xaddivortes/cross_binding.txt"
        );
        if std::env::var("XADDI_WRITE_FIXTURE").is_ok() {
            std::fs::write(path, &text).unwrap();
        }
        assert_eq!(std::fs::read_to_string(path).unwrap(), text);
    }

    /// Chains from fast starts: reproducible, the configured number of
    /// draws, independent of the thread count, and different from cold
    /// chains.
    #[test]
    fn chains_from_fast_starts() {
        let (config, x, y) = fixture();
        let params = XAddiVortesParams::default().with_rounds(5);
        let (a, ma) = thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 2, 1, &params).unwrap();
        assert!(a.warnings().is_empty(), "{:?}", a.warnings());
        let (b, mb) = thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 2, 2, &params).unwrap();
        assert_eq!(ma, mb);
        assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
        assert_eq!(a.n_draws(), 2 * 60);
        let (_, cold) = thiessen::fit_chains_with_threads(&config, &x, &y, SEED, 2, 1).unwrap();
        assert_ne!(ma, cold);
        assert!(thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 0, 1, &params).is_err());
    }

    /// Every setting's edge runs and stays a valid state: one column per
    /// tessellation, one centre, no replacement, no rounds.
    #[test]
    fn edge_settings_run() {
        let (config, x, y) = fixture();
        let d = XAddiVortesParams::default();
        let variants = [
            d.clone().with_max_columns(1),
            d.clone().with_max_columns(5),
            d.clone().with_centres(1),
            d.clone().with_centres(300),
            d.clone().with_replace_share(0.0),
            d.clone().with_replace_share(1.0),
            d.clone().with_value_scale(1.0),
            d.clone().with_rounds(0),
        ];
        for v in variants {
            let (s, f) = fit(&config, &x, &y, SEED, &v);
            assert_eq!(s.tessellations().len(), 15, "{v:?}");
            assert!(f.sigma.is_finite(), "{v:?}");
            assert_eq!(f.train_score.len(), v.rounds + 2, "{v:?}");
        }
    }

    /// More than 256 centres store cells as `u16`: the kept fit's
    /// prediction at the training rows reproduces the final training score.
    #[test]
    fn wide_cells_predict_what_the_fit_scored() {
        let (x, _, y) = friedman1(2000, 5, 1.0, 11);
        let params = XAddiVortesParams::default()
            .with_centres(1000)
            .with_rounds(2);
        let (mut s, f) = fit(&Config::new().with_m(20), &x, &y, SEED, &params);
        s.keep();
        let pred = s.finish().unwrap().predict(&x).unwrap();
        let train = rmse(&pred, &y);
        let last = *f.train_score.last().unwrap();
        assert!(
            (last - train).abs() < 1e-9 * (1.0 + train),
            "{last} vs {train}"
        );
    }

    /// With m below 4 a full replacement share still replaces: the rounds
    /// change the tessellations.
    #[test]
    fn small_ensembles_replace() {
        let (_, x, y) = fixture();
        for m in 1..4 {
            let config = Config::new().with_m(m);
            let d = XAddiVortesParams::default().with_replace_share(1.0);
            let (a, _) = fit(&config, &x, &y, SEED, &d.clone().with_rounds(0));
            let (b, _) = fit(&config, &x, &y, SEED, &d.with_rounds(3));
            assert_ne!(a.tessellations(), b.tessellations(), "m = {m}");
        }
    }

    /// Fewer than four tessellations build and refit.
    #[test]
    fn small_ensembles_work() {
        let (_, x, y) = fixture();
        for m in 1..4 {
            let config = Config::new().with_m(m);
            let (s, f) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
            assert_eq!(s.tessellations().len(), m);
            assert!(s
                .tessellations()
                .iter()
                .all(|t| t.mus().iter().any(|&v| v != 0.0)));
            assert!(f.sigma.is_finite(), "m = {m}");
        }
    }

    /// Fixed-seed regression fixture: the default fit on the shared
    /// fixture. A change here is a change to the method's output.
    #[test]
    fn regression_fixture() {
        let (config, x, y) = fixture();
        let (mut s, f) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
        s.keep();
        let pred = s.finish().unwrap().predict(&x).unwrap();
        let digest: f64 = pred
            .iter()
            .enumerate()
            .map(|(i, v)| (i + 1) as f64 * v)
            .sum();
        let got = format!("sigma={:.12} digest={:.12}", f.sigma, digest);
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/fixtures/xaddivortes/regression.txt"
        );
        if std::env::var("XADDI_WRITE_FIXTURE").is_ok() {
            std::fs::write(path, format!("{got}\n")).unwrap();
        }
        assert_eq!(got, std::fs::read_to_string(path).unwrap().trim());
    }

    /// Friedman 1 at n = 2,000, p = 10, noise 1, four replicates (data
    /// seeds 101 to 104, test seeds 202 to 205, 1,000 test rows): the
    /// mean relative RMSE of the fast fit at the defaults against the
    /// noiseless truth is at most that of the evaluated method on the same
    /// replicates and fit seed, with its locked settings at a budget of 20
    /// sweeps (`fixtures/xaddivortes/friedman_reference.txt`).
    #[test]
    fn friedman1_accuracy() {
        let replicates = 4;
        let mut total = 0.0;
        for r in 0..replicates {
            let (x, _, y) = friedman1(2000, 10, 1.0, 101 + r);
            let (xt, truth, _) = friedman1(1000, 10, 1.0, 202 + r);
            let (mut s, _) = fit(
                &Config::default(),
                &x,
                &y,
                thiessen::chain_seed(3, 0),
                &XAddiVortesParams::default(),
            );
            s.keep();
            let pred = s.finish().unwrap().predict(&xt).unwrap();
            let mean = truth.iter().sum::<f64>() / truth.len() as f64;
            let sd = (truth.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>()
                / truth.len() as f64)
                .sqrt();
            total += rmse(&pred, &truth) / sd;
        }
        let rel = total / replicates as f64;
        let reference: f64 = include_str!("fixtures/xaddivortes/friedman_reference.txt")
            .trim()
            .parse()
            .unwrap();
        assert!(
            rel <= reference,
            "mean relative RMSE {rel} against the reference's {reference}"
        );
    }

    /// The fast fit replaces the cold start; a chain that has swept is
    /// refused.
    #[test]
    fn a_chain_that_has_swept_is_refused() {
        let (config, x, y) = fixture();
        let mut s = Sampler::new(&config, &x, &y, SEED).unwrap();
        s.step();
        let err = s.xaddivortes(&XAddiVortesParams::default()).unwrap_err();
        assert!(matches!(err, thiessen::Error::InvalidHyperparameter { .. }));
    }

    /// A restored chain, and a prior that excludes a column, are refused.
    #[test]
    fn a_restored_chain_or_a_zero_inclusion_weight_is_refused() {
        let (config, x, y) = fixture();
        let previous = thiessen::fit(&config, &x, &y, SEED).unwrap();
        let mut s = Sampler::new(&config, &x, &y, SEED).unwrap();
        s.start_from(&previous, 0).unwrap();
        assert!(s.xaddivortes(&XAddiVortesParams::default()).is_err());
        let excluding = config.with_inclusion(thiessen::Inclusion::Weighted {
            weights: vec![1.0, 0.0],
        });
        let mut s = Sampler::new(&excluding, &x, &y, SEED).unwrap();
        assert!(s.xaddivortes(&XAddiVortesParams::default()).is_err());
    }

    /// The fitted model's noise draw is the fit's sigma, so every interval
    /// method uses one value.
    #[test]
    fn the_fitted_model_carries_the_fit_sigma() {
        let (config, x, y) = fixture();
        for rounds in [0, 5] {
            let params = XAddiVortesParams::default().with_rounds(rounds);
            let (fitted, fast) = thiessen::fit_xaddivortes(&config, &x, &y, SEED, &params).unwrap();
            let sigma = fitted.sigma();
            assert_eq!(sigma.len(), 1);
            assert!(
                (sigma[0] - fast.sigma).abs() <= 1e-14 * fast.sigma,
                "rounds {rounds}: {} against {}",
                sigma[0],
                fast.sigma
            );
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        /// Any valid input gives a valid state.
        #[test]
        fn any_input_gives_a_valid_state(
            n in 8usize..60,
            p in 1usize..5,
            m in 1usize..12,
            rounds in 0usize..6,
            seed in 0u64..1000,
            centres in 1usize..20,
            max_columns in 1usize..4,
            replace_share in 0.0f64..=1.0,
        ) {
            let mut rng = TestRng(seed);
            let rows: Vec<Vec<f64>> = (0..n).map(|_| (0..p).map(|_| rng.uniform()).collect()).collect();
            let y: Vec<f64> = rows.iter().map(|r| r[0] + 0.1 * rng.normal()).collect();
            let x = Data::from_rows(&rows).unwrap();
            let params = XAddiVortesParams::default()
                .with_rounds(rounds)
                .with_centres(centres)
                .with_max_columns(max_columns)
                .with_replace_share(replace_share);
            let mut s = Sampler::new(&Config::new().with_m(m), &x, &y, seed).unwrap();
            let f = s.xaddivortes(&params).unwrap();
            prop_assert_eq!(s.tessellations().len(), m);
            prop_assert!(f.sigma.is_finite() && f.sigma > 0.0);
            prop_assert_eq!(f.train_score.len(), rounds + 2);
            prop_assert!(s.fitted_values().iter().all(|v| v.is_finite()));
        }
    }
}
