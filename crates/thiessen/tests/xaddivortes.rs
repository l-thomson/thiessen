//! `Sampler::xaddivortes` (`xaddivortes.rs`): the gate, the settings, the
//! start contract, the budget, determinism, a fixed-seed regression
//! fixture and a statistical check on Friedman's first function.

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
    let partial: XAddiVortesParams = serde_json::from_str(r#"{"budget": 50}"#).unwrap();
    assert_eq!(partial, XAddiVortesParams::default().budget(50.0));
    assert!(serde_json::from_str::<XAddiVortesParams>(r#"{"budjet": 50}"#).is_err());
    let bad = [
        d.clone().budget(0.0),
        d.clone().budget(f64::NAN),
        d.clone().value_scale(0.0),
        d.clone().value_scale(1.5),
        d.clone().main_effect_share(-0.1),
        d.clone().replace_share(1.1),
        d.clone().score_floor(-1.0),
        d.clone().max_columns(0),
        d.clone().centres(0),
        d.clone().pair_screen(1),
        d.clone().column_bins(1),
        d.clone().column_bins(256),
        d.clone().response_bins(1),
        d.clone().tries(0),
        d.clone().refit_passes(0),
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

    /// The fit is a valid chain state: m tessellations, nonempty cells,
    /// finite values, within the budget; the same for the same seed and
    /// different for another.
    #[test]
    fn fit_is_valid_reproducible_and_within_budget() {
        let (config, x, y) = fixture();
        let p = XAddiVortesParams::default();
        let (a, fa) = fit(&config, &x, &y, SEED, &p);
        let (b, fb) = fit(&config, &x, &y, SEED, &p);
        let (c, _) = fit(&config, &x, &y, SEED + 1, &p);
        assert_eq!(a.tessellations(), b.tessellations());
        assert_eq!(fa, fb);
        assert_ne!(a.tessellations(), c.tessellations());
        assert_eq!(a.tessellations().len(), 15);
        assert!(fa.planned_sweeps <= p.budget * thiessen::PLANNING_MARGIN + 0.15 / 15.0);
        assert!(fa.sigma.is_finite() && fa.sigma > 0.0);
        assert_eq!(fa.column_scores.len(), 2);
        assert!(a.fitted_values().iter().all(|v| v.is_finite()));
    }

    /// The prediction of a kept fast fit is the sum of its cell values, and
    /// its training RMSE is the reported sigma.
    #[test]
    fn sigma_is_the_training_residual_rms() {
        let (config, x, y) = fixture();
        let (mut s, f) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
        s.keep();
        let fitted = s.finish().unwrap();
        let pred = fitted.predict(&x).unwrap();
        let rms = (pred
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / y.len() as f64)
            .sqrt();
        assert!(
            (rms - f.sigma).abs() < 1e-9 * (1.0 + rms),
            "{rms} vs {}",
            f.sigma
        );
    }

    /// The exact sampler runs on from the fit (the warm start), and the
    /// chain is reproducible.
    #[test]
    fn warm_start_runs_on() {
        let (config, x, y) = fixture();
        let run = || {
            let (mut s, _) = fit(
                &config,
                &x,
                &y,
                SEED,
                &XAddiVortesParams::default().budget(5.0),
            );
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

    /// The cross-binding fixture: `fit_xaddivortes` on the shared fixture
    /// with seed 7 and default settings, its predictions at the training
    /// rows. The R and Python packages test against the same file, so the
    /// three surfaces agree bit for bit. `XADDI_WRITE_FIXTURE=1` rewrites
    /// it.
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
        let params = XAddiVortesParams::default().budget(5.0);
        let (a, ma) = thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 2, 1, &params).unwrap();
        let (b, mb) = thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 2, 2, &params).unwrap();
        assert_eq!(ma, mb);
        assert_eq!(a.predict(&x).unwrap(), b.predict(&x).unwrap());
        assert_eq!(a.n_draws(), 2 * 60);
        let (_, cold) = thiessen::fit_chains_with_threads(&config, &x, &y, SEED, 2, 1).unwrap();
        assert_ne!(ma, cold);
        assert!(thiessen::fit_chains_from_fast(&config, &x, &y, SEED, 0, 1, &params).is_err());
    }

    /// A tiny budget leaves tessellations at zero and reports them.
    #[test]
    fn a_tiny_budget_zeroes_what_it_cannot_build() {
        let (config, x, y) = fixture();
        let (s, f) = fit(
            &config,
            &x,
            &y,
            SEED,
            &XAddiVortesParams::default().budget(0.05),
        );
        assert!(f.zeroed > 0);
        assert_eq!(f.rounds, 0);
        assert_eq!(s.tessellations().len(), 15);
    }

    /// Every option runs and stays a valid state; the defaults equal the
    /// options set to their default values.
    #[test]
    fn options_run() {
        let (config, x, y) = fixture();
        let d = XAddiVortesParams::default();
        let variants = [
            d.clone().max_columns(1),
            d.clone().max_columns(3),
            d.clone().tries(1),
            d.clone().rescore(true),
            d.clone().refit_passes(1),
            d.clone().update_sigma(false),
            d.clone().worth(thiessen::Worth::SquaredValues),
            d.clone().replace_share(0.0),
            d.clone().main_effect_share(1.0),
            d.clone().accept_if_better(true),
        ];
        for v in variants {
            let (s, f) = fit(&config, &x, &y, SEED, &v);
            assert_eq!(s.tessellations().len(), 15, "{v:?}");
            assert!(f.sigma.is_finite(), "{v:?}");
            assert!(
                f.planned_sweeps <= v.budget * thiessen::PLANNING_MARGIN + 0.15 / 15.0,
                "{v:?}"
            );
        }
        let (a, _) = fit(&config, &x, &y, SEED, &d);
        let (b, _) = fit(&config, &x, &y, SEED, &d.clone().tries(4).refit_passes(2));
        assert_eq!(a.tessellations(), b.tessellations());
    }

    /// The training score has one entry after the build and one per
    /// round, ends at the fit's sigma, and recording it changes nothing.
    #[test]
    fn train_score_path() {
        let (config, x, y) = fixture();
        let d = XAddiVortesParams::default();
        let (a, f) = fit(&config, &x, &y, SEED, &d);
        assert_eq!(f.train_score.len(), f.rounds + 1);
        let last = *f.train_score.last().unwrap();
        assert!(
            (last - f.sigma).abs() <= 1e-12 * f.sigma.max(1.0),
            "{last} {}",
            f.sigma
        );
        let (b, _) = fit(&config, &x, &y, SEED, &d);
        assert_eq!(a.tessellations(), b.tessellations());
    }

    /// With `accept_if_better` the training error never rises from round
    /// to round.
    #[test]
    fn guarded_rounds_never_raise_the_training_error() {
        let (config, x, y) = fixture();
        for seed in 0..8 {
            let params = XAddiVortesParams::default()
                .accept_if_better(true)
                .budget(30.0);
            let (_, f) = fit(&config, &x, &y, seed, &params);
            for w in f.train_score.windows(2) {
                assert!(
                    w[1] <= w[0] * (1.0 + 1e-12),
                    "seed {seed}: {:?}",
                    f.train_score
                );
            }
        }
    }

    /// Fewer than four tessellations work (the locked search code indexed
    /// past the ensemble there).
    #[test]
    fn small_ensembles_work() {
        let (_, x, y) = fixture();
        for m in 1..4 {
            let config = Config::new().with_m(m);
            let (s, _) = fit(&config, &x, &y, SEED, &XAddiVortesParams::default());
            assert_eq!(s.tessellations().len(), m);
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
        let got = format!(
            "rounds={} planned={:.12} sigma={:.12} digest={:.12}",
            f.rounds, f.planned_sweeps, f.sigma, digest
        );
        let want = include_str!("fixtures/xaddivortes/regression.txt").trim();
        assert_eq!(got, want);
    }

    /// Friedman 1 at n = 2,000, p = 10, noise 1, B = 20: with the locked
    /// search method's settings (the stage-one defaults), the fast fit's
    /// relative RMSE against the noiseless truth on 1,000 test rows stays
    /// within 10% of the locked method's on the same data and seed
    /// (`experiments/stage1/friedman_locked.txt`). Every setting is pinned,
    /// because the package defaults differ and are covered by the
    /// regression fixture instead.
    #[test]
    fn friedman1_accuracy() {
        let (x, _, y) = friedman1(2000, 10, 1.0, 101);
        let (xt, truth, _) = friedman1(1000, 10, 1.0, 202);
        let (mut s, _) = fit(
            &Config::default(),
            &x,
            &y,
            thiessen::chain_seed(3, 0),
            &XAddiVortesParams::default()
                .budget(20.0)
                .max_columns(2)
                .centres(16)
                .value_scale(0.5)
                .main_effect_share(0.5)
                .replace_share(0.25)
                .pair_screen(12)
                .tries(1)
                .rescore(false)
                .refit_passes(1)
                .update_sigma(false)
                .worth(thiessen::Worth::SquaredValues),
        );
        s.keep();
        let pred = s.finish().unwrap().predict(&xt).unwrap();
        let mean = truth.iter().sum::<f64>() / truth.len() as f64;
        let sd = (truth.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / truth.len() as f64)
            .sqrt();
        let rmse = (pred
            .iter()
            .zip(&truth)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            / truth.len() as f64)
            .sqrt();
        let rel = rmse / sd;
        let locked: f64 = include_str!("fixtures/xaddivortes/friedman_locked.txt")
            .trim()
            .parse()
            .unwrap();
        assert!(
            rel <= 1.1 * locked,
            "relative RMSE {rel} against the locked method's {locked}"
        );
    }

    /// Writes the Friedman data above as TSV (header, response last) for
    /// the locked driver: `XADDI_EXPORT=<dir> cargo test ... -- --ignored export`.
    #[test]
    #[ignore = "writes files; run by hand"]
    fn export_friedman_for_the_locked_driver() {
        use std::io::Write;
        let dir = std::env::var("XADDI_EXPORT").expect("XADDI_EXPORT");
        for (name, n, seed) in [("train", 2000, 101), ("test", 1000, 202)] {
            let (x, truth, y) = friedman1(n, 10, 1.0, seed);
            let mut f = std::fs::File::create(format!("{dir}/friedman_{name}.tsv")).unwrap();
            writeln!(
                f,
                "{}\ty",
                (0..10)
                    .map(|q| format!("x{q}"))
                    .collect::<Vec<_>>()
                    .join("\t")
            )
            .unwrap();
            for (i, yi) in y.iter().enumerate() {
                let row: Vec<String> = x.row(i).iter().map(|v| format!("{v:?}")).collect();
                writeln!(f, "{}\t{yi:?}", row.join("\t")).unwrap();
            }
            let mut t = std::fs::File::create(format!("{dir}/friedman_{name}_truth.txt")).unwrap();
            for v in truth {
                writeln!(t, "{v:?}").unwrap();
            }
        }
    }

    /// The guard's checks are planned for: a guarded fit stays within
    /// the same plan as an unguarded one, at every budget.
    #[test]
    fn guarded_fits_stay_within_the_plan() {
        let (config, x, y) = fixture();
        let m = 15.0;
        for step in 4..240 {
            let budget = step as f64 * 0.25;
            let params = XAddiVortesParams::default()
                .budget(budget)
                .accept_if_better(true);
            let (_, f) = fit(&config, &x, &y, SEED, &params);
            let limit = budget * thiessen::PLANNING_MARGIN + 0.15 / m;
            let scoring = ((48f64).ln() + 4.0 + 3.0 * 253.0) * 0.15 / m;
            assert!(
                f.planned_sweeps <= limit.max(scoring) + 1e-9,
                "budget {budget}: planned {}",
                f.planned_sweeps
            );
        }
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

    /// The fitted model's noise draw is the fit's sigma, with or without
    /// the noise update, so every interval method uses one value.
    #[test]
    fn the_fitted_model_carries_the_fit_sigma() {
        let (config, x, y) = fixture();
        for update in [true, false] {
            for budget in [0.5, 50.0] {
                let params = XAddiVortesParams::default()
                    .budget(budget)
                    .update_sigma(update);
                let (fitted, fast) =
                    thiessen::fit_xaddivortes(&config, &x, &y, SEED, &params).unwrap();
                let sigma = fitted.sigma();
                assert_eq!(sigma.len(), 1);
                assert!(
                    (sigma[0] - fast.sigma).abs() <= 1e-14 * fast.sigma,
                    "update_sigma {update}, budget {budget}: {} against {}",
                    sigma[0],
                    fast.sigma
                );
            }
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(24))]
        /// Any valid input gives a valid state within the budget.
        #[test]
        fn any_input_gives_a_valid_state(
            n in 8usize..60,
            p in 1usize..5,
            m in 1usize..12,
            budget in 0.05f64..30.0,
            seed in 0u64..1000,
            centres in 1usize..20,
            max_columns in 1usize..4,
        ) {
            let mut rng = TestRng(seed);
            let rows: Vec<Vec<f64>> = (0..n).map(|_| (0..p).map(|_| rng.uniform()).collect()).collect();
            let y: Vec<f64> = rows.iter().map(|r| r[0] + 0.1 * rng.normal()).collect();
            let x = Data::from_rows(&rows).unwrap();
            let params = XAddiVortesParams::default().budget(budget).centres(centres).max_columns(max_columns);
            let mut s = Sampler::new(&Config::new().with_m(m), &x, &y, seed).unwrap();
            let f = s.xaddivortes(&params).unwrap();
            prop_assert_eq!(s.tessellations().len(), m);
            // The plan: 0.8 B plus at most one worth pass (1.5 n of 10 n m),
            // or the fixed cost of scoring the columns when that is larger.
            let s_pairs = (params.pair_screen * (params.pair_screen - 1) / 2) as f64;
            let scoring = ((n as f64).ln() + 2.0 * p as f64 + 3.0 * s_pairs) * 0.15 / m as f64;
            let limit = (budget * thiessen::PLANNING_MARGIN + 0.15 / m as f64).max(scoring);
            prop_assert!(f.planned_sweeps <= limit + 1e-9, "planned {} limit {}", f.planned_sweeps, limit);
            prop_assert!(f.sigma.is_finite());
            prop_assert_eq!(f.train_score.len(), f.rounds + 1);
            let guarded = params.clone().accept_if_better(true);
            let mut g = Sampler::new(&Config::new().with_m(m), &x, &y, seed).unwrap();
            let fg = g.xaddivortes(&guarded).unwrap();
            prop_assert!(fg.planned_sweeps <= limit + 1e-9, "guarded planned {} limit {}", fg.planned_sweeps, limit);
            for w in fg.train_score.windows(2) {
                prop_assert!(w[1] <= w[0] * (1.0 + 1e-12), "{:?}", fg.train_score);
            }
            prop_assert!(s.fitted_values().iter().all(|v| v.is_finite()));
        }
    }
}
