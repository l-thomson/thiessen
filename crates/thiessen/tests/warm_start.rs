//! Warm starts from a previous fit (`state.rs`, `Sampler::restore`,
//! `Sampler::start_from`, `Init`): the gate, state round trips on every
//! model, reproducibility, the chain rule and the compatibility checks.

mod common;

use common::SEED;
use thiessen::Sampler;

#[cfg(not(feature = "experimental"))]
#[test]
fn warm_starts_require_the_feature() {
    let (config, x, y) = common::fixture();
    let previous = thiessen::fit(&config, &x, &y, SEED).unwrap();
    let mut s = Sampler::new(&config, &x, &y, SEED).unwrap();
    let gated = |r: thiessen::Result<()>| matches!(r, Err(thiessen::Error::RequiresFeature { .. }));
    assert!(gated(s.state().map(drop)));
    assert!(gated(previous.state(0).map(drop)));
    assert!(gated(s.start_from(&previous, 0)));
    let init = thiessen::Init::From {
        fitted: &previous,
        draw: None,
    };
    let r = thiessen::fit_chains_with_init(&config, &x, &y, SEED, 2, 1, init);
    assert!(matches!(r, Err(thiessen::Error::RequiresFeature { .. })));
}

/// The cold start through the init-aware entry point is the existing fit.
#[test]
fn the_cold_init_is_the_cold_fit() {
    let (config, x, y) = common::fixture();
    let (a, fa) = thiessen::fit_chains_with_threads(&config, &x, &y, SEED, 2, 1).unwrap();
    let (b, fb) =
        thiessen::fit_chains_with_init(&config, &x, &y, SEED, 2, 1, thiessen::Init::Cold).unwrap();
    assert_eq!(a, b);
    assert_eq!(fa, fb);
}

#[cfg(feature = "experimental")]
mod experimental {
    use super::*;
    use common::TestRng;
    use thiessen::{Basis, Config, Data, EnsembleState, Inclusion, Init, Membership, Outcome};

    const N: usize = 80;

    fn design(seed: u64) -> (Data, Vec<f64>) {
        let mut rng = TestRng(seed);
        let rows: Vec<Vec<f64>> = (0..N)
            .map(|_| (0..3).map(|_| rng.uniform()).collect())
            .collect();
        let y = rows
            .iter()
            .map(|r| (6.0 * r[0]).sin() + r[1] + 0.2 * rng.normal())
            .collect();
        (Data::from_rows(&rows).unwrap(), y)
    }

    /// One model's data and how to construct its sampler.
    enum Response {
        Plain(Vec<f64>),
        Aft(Vec<f64>, Vec<bool>),
        Bounds(Vec<f64>, Vec<f64>),
    }

    struct Case {
        name: &'static str,
        config: Config,
        x: Data,
        response: Response,
    }

    impl Case {
        fn sampler(&self, seed: u64) -> Sampler {
            match &self.response {
                Response::Plain(y) => Sampler::new(&self.config, &self.x, y, seed),
                Response::Aft(t, e) => Sampler::aft(&self.config, &self.x, t, e, seed),
                Response::Bounds(l, u) => {
                    Sampler::interval_censored(&self.config, &self.x, l, u, seed)
                }
            }
            .unwrap_or_else(|e| panic!("{}: {e}", self.name))
        }

        fn fit(&self, seed: u64, chains: usize, init: Init<'_>) -> thiessen::Fitted {
            let r = match &self.response {
                Response::Plain(y) => {
                    thiessen::fit_chains_with_init(&self.config, &self.x, y, seed, chains, 1, init)
                }
                Response::Aft(t, e) => thiessen::fit_aft_chains_with_init(
                    &self.config,
                    &self.x,
                    t,
                    e,
                    seed,
                    chains,
                    1,
                    init,
                ),
                Response::Bounds(l, u) => thiessen::fit_interval_censored_chains_with_init(
                    &self.config,
                    &self.x,
                    l,
                    u,
                    seed,
                    chains,
                    1,
                    init,
                ),
            };
            r.unwrap_or_else(|e| panic!("{}: {e}", self.name)).0
        }
    }

    fn cases() -> Vec<Case> {
        let (x, y) = design(11);
        let base = Config::new().with_m(10).with_burn_in(20).with_draws(10);
        let plain = |name, config: Config, y: Vec<f64>| Case {
            name,
            config,
            x: x.clone(),
            response: Response::Plain(y),
        };
        let labels: Vec<f64> = y.iter().map(|&v| f64::from(v > 0.5)).collect();
        let codes: Vec<f64> = y
            .iter()
            .map(|&v| {
                if v < 0.2 {
                    0.0
                } else if v < 0.9 {
                    1.0
                } else {
                    2.0
                }
            })
            .collect();
        let floor = y.iter().copied().fold(f64::INFINITY, f64::min) + 0.3;
        let censored: Vec<f64> = y.iter().map(|&v| v.max(floor)).collect();
        let times: Vec<f64> = y.iter().map(|&v| (v + 1.0).exp()).collect();
        let events: Vec<bool> = (0..N).map(|i| i % 4 != 0).collect();
        let (lower, upper): (Vec<f64>, Vec<f64>) = y
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                if i % 3 == 0 {
                    (v - 0.2, v + 0.2)
                } else {
                    (v, v)
                }
            })
            .unzip();
        let dart = Inclusion::Dart {
            a: 0.5,
            b: 1.0,
            rho: None,
        };
        vec![
            plain("gaussian", base.clone(), y.clone()),
            plain("heteroscedastic", base.clone().with_m_var(4), y.clone()),
            plain(
                "probit",
                base.clone().with_outcome(Outcome::probit()),
                labels,
            ),
            plain(
                "ordinal",
                base.clone().with_outcome(Outcome::ordinal(3)),
                codes,
            ),
            plain(
                "tobit",
                base.clone().with_outcome(Outcome::tobit(Some(floor), None)),
                censored,
            ),
            plain(
                "student_t",
                base.clone().with_outcome(Outcome::student_t(4.0)),
                y.clone(),
            ),
            plain(
                "student_t_grid",
                base.clone()
                    .with_outcome(Outcome::student_t_grid(vec![3.0, 6.0, 12.0])),
                y.clone(),
            ),
            plain(
                "laplace",
                base.clone().with_outcome(Outcome::laplace()),
                y.clone(),
            ),
            plain("dart", base.clone().with_inclusion(dart), y.clone()),
            plain(
                "soft",
                base.clone().with_membership(Membership::soft()),
                y.clone(),
            ),
            plain("linear", base.clone().with_basis(Basis::Linear), y.clone()),
            Case {
                name: "aft",
                config: base.clone().with_outcome(Outcome::aft()),
                x: x.clone(),
                response: Response::Aft(times, events),
            },
            Case {
                name: "interval_censored",
                config: base.with_outcome(Outcome::interval_censored()),
                x,
                response: Response::Bounds(lower, upper),
            },
        ]
    }

    fn stepped(case: &Case, seed: u64, sweeps: usize) -> Sampler {
        let mut s = case.sampler(seed);
        for _ in 0..sweeps {
            s.step();
        }
        s
    }

    /// A state taken mid-run and restored into a fresh sampler reads back
    /// unchanged, rebuilds the running fit to rounding, and runs on.
    #[test]
    fn a_restored_state_is_that_state() {
        for case in cases() {
            let source = stepped(&case, SEED, 15);
            let state = source.state().unwrap();
            let mut s = case.sampler(SEED + 1);
            s.restore(&state)
                .unwrap_or_else(|e| panic!("{}: {e}", case.name));
            assert_eq!(s.state().unwrap(), state, "{}", case.name);
            let (a, b) = (source.fitted_values(), s.fitted_values());
            let scale = a.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let gap = a
                .iter()
                .zip(&b)
                .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()));
            assert!(
                gap <= 1e-12 * scale,
                "{}: fitted values differ by {gap}",
                case.name
            );
            for _ in 0..5 {
                s.step();
            }
            assert!(
                s.fitted_values().iter().all(|v| v.is_finite()),
                "{}",
                case.name
            );
        }
    }

    /// A clone given its own state back runs on as the original does, so
    /// the restore rebuilds every cache a sweep reads (precisions,
    /// variance totals, inclusion weights) and not only the fit.
    #[test]
    fn a_chain_given_its_own_state_runs_on_unchanged() {
        for case in cases() {
            let mut a = stepped(&case, SEED, 30);
            let mut b = a.clone();
            b.restore(&a.state().unwrap()).unwrap();
            for _ in 0..10 {
                a.step();
                b.step();
            }
            let (fa, fb) = (a.fitted_values(), b.fitted_values());
            let scale = fa.iter().fold(1.0_f64, |m, v| m.max(v.abs()));
            let gap = fa
                .iter()
                .zip(&fb)
                .fold(0.0_f64, |m, (u, v)| m.max((u - v).abs()));
            assert!(
                gap <= 1e-9 * scale,
                "{}: fits differ by {gap} after 10 sweeps",
                case.name
            );
            assert_eq!(
                a.tessellations().len(),
                b.tessellations().len(),
                "{}",
                case.name
            );
        }
    }

    #[test]
    fn states_round_trip_through_json() {
        for case in cases() {
            let state = stepped(&case, SEED, 10).state().unwrap();
            let json = serde_json::to_string(&state).unwrap();
            let back: EnsembleState = serde_json::from_str(&json).unwrap();
            assert_eq!(back, state, "{}", case.name);
        }
    }

    /// The state of a kept draw is the one the chain kept.
    #[test]
    fn a_fit_returns_the_state_of_each_kept_draw() {
        for case in cases() {
            let mut s = case.sampler(SEED);
            let mut states = Vec::new();
            for _ in 0..3 {
                s.step();
                states.push(s.state().unwrap());
                s.keep();
            }
            let fitted = s.finish().unwrap();
            for (d, state) in states.iter().enumerate() {
                assert_eq!(&fitted.state(d).unwrap(), state, "{}", case.name);
            }
            assert!(fitted.state(3).is_err(), "{}", case.name);
        }
    }

    #[test]
    fn warm_started_fits_reproduce() {
        for case in cases() {
            let previous = case.fit(SEED, 1, Init::Cold);
            let init = Init::From {
                fitted: &previous,
                draw: None,
            };
            let a = case.fit(SEED + 5, 3, init);
            let b = case.fit(SEED + 5, 3, init);
            assert_eq!(a, b, "{}", case.name);
            assert_ne!(a, case.fit(SEED + 5, 3, Init::Cold), "{}", case.name);
        }
    }

    /// With as many chains as the previous fit, each continues its own.
    #[test]
    fn each_chain_continues_its_own() {
        for case in cases() {
            let previous = case.fit(SEED, 2, Init::Cold);
            let per_chain = previous.n_draws() / 2;
            let draws = [per_chain - 1, 2 * per_chain - 1];
            let continued = case.fit(
                SEED + 1,
                2,
                Init::From {
                    fitted: &previous,
                    draw: None,
                },
            );
            let named = case.fit(
                SEED + 1,
                2,
                Init::Draws {
                    fitted: &previous,
                    draws: &draws,
                },
            );
            assert_eq!(continued, named, "{}", case.name);
        }
    }

    /// A warm-started fit does not depend on the thread count.
    #[test]
    fn warm_started_fits_ignore_the_thread_count() {
        let case = &cases()[0];
        let Response::Plain(y) = &case.response else {
            unreachable!()
        };
        let previous = case.fit(SEED, 3, Init::Cold);
        let init = Init::From {
            fitted: &previous,
            draw: None,
        };
        let one =
            thiessen::fit_chains_with_init(&case.config, &case.x, y, SEED + 2, 3, 1, init).unwrap();
        let three =
            thiessen::fit_chains_with_init(&case.config, &case.x, y, SEED + 2, 3, 3, init).unwrap();
        assert_eq!(one, three);
    }

    /// Chain k starts from draw `draw - k` (stochtree's rule).
    #[test]
    fn chains_count_back_from_the_starting_draw() {
        let case = &cases()[0];
        let previous = case.fit(SEED, 1, Init::Cold);
        for (k, d) in [9, 8, 7].into_iter().enumerate() {
            let mut s = case.sampler(thiessen::chain_seed(SEED, k));
            s.start_from(&previous, d).unwrap();
            assert_eq!(s.state().unwrap(), previous.state(d).unwrap());
        }
        let quiet = case.fit(
            SEED,
            3,
            Init::From {
                fitted: &previous,
                draw: Some(4),
            },
        );
        assert!(quiet.warnings().is_empty());
        let shared = case.fit(
            SEED,
            3,
            Init::From {
                fitted: &previous,
                draw: Some(1),
            },
        );
        assert_eq!(
            shared.warnings(),
            &[thiessen::Warning::WarmStartDrawsReused {
                chains: 3,
                draws: 2
            }]
        );
    }

    /// `Init::Draws` names each chain's draw.
    #[test]
    fn named_draws_start_each_chain() {
        let case = &cases()[0];
        let previous = case.fit(SEED, 2, Init::Cold);
        let draws = [9, 19];
        let a = case.fit(
            SEED + 1,
            2,
            Init::Draws {
                fitted: &previous,
                draws: &draws,
            },
        );
        let mut chains: Vec<Sampler> = (0..2)
            .map(|k| {
                let mut s = case.sampler(thiessen::chain_seed(SEED + 1, k));
                s.start_from(&previous, draws[k]).unwrap();
                s
            })
            .collect();
        let Response::Plain(y) = &case.response else {
            unreachable!()
        };
        let mut refs: Vec<&mut Sampler> = chains.iter_mut().collect();
        Sampler::advance_all(&mut refs, 20, 10, 1, 1);
        let (b, _) = thiessen::Fitted::pool_samplers(chains, &case.x, y).unwrap();
        assert_eq!(a, b);
        let wrong = thiessen::fit_chains_with_init(
            &case.config,
            &case.x,
            y,
            SEED,
            3,
            1,
            Init::Draws {
                fitted: &previous,
                draws: &draws,
            },
        );
        assert!(matches!(
            wrong,
            Err(thiessen::Error::InvalidHyperparameter { .. })
        ));
    }

    #[test]
    fn a_fit_of_other_data_or_another_model_is_refused() {
        let cases = cases();
        let gaussian = &cases[0];
        let previous = gaussian.fit(SEED, 1, Init::Cold);
        let incompatible = |s: &mut Sampler| {
            matches!(
                s.start_from(&previous, 0),
                Err(thiessen::Error::IncompatibleState { .. })
            )
        };
        let (x, y) = design(12);
        let mut other_data = Sampler::new(&gaussian.config, &x, &y, SEED).unwrap();
        assert!(incompatible(&mut other_data));
        let Response::Plain(y) = &gaussian.response else {
            unreachable!()
        };
        let mut other_m =
            Sampler::new(&gaussian.config.clone().with_m(12), &gaussian.x, y, SEED).unwrap();
        assert!(incompatible(&mut other_m));
        assert!(incompatible(&mut cases[1].sampler(SEED)));
        let mut longer = Sampler::new(
            &gaussian.config.clone().with_draws(50),
            &gaussian.x,
            y,
            SEED,
        )
        .unwrap();
        longer.start_from(&previous, 0).unwrap();
        let mut kept = gaussian.sampler(SEED);
        kept.step();
        kept.keep();
        assert!(incompatible(&mut kept));
        let mut s = gaussian.sampler(SEED);
        assert!(matches!(
            s.start_from(&previous, previous.n_draws()),
            Err(thiessen::Error::InvalidHyperparameter { .. })
        ));
    }

    /// A malformed state is refused and leaves the sampler as it was.
    #[test]
    fn a_malformed_state_is_refused_without_side_effects() {
        let case = &cases()[0];
        let good = stepped(case, SEED, 10).state().unwrap();
        let json = serde_json::to_value(&good).unwrap();
        let edit = |f: &dyn Fn(&mut serde_json::Value)| -> EnsembleState {
            let mut v = json.clone();
            f(&mut v);
            serde_json::from_value(v).unwrap()
        };
        let bad = [
            edit(&|v| {
                v["tessellations"].as_array_mut().unwrap().pop();
            }),
            edit(&|v| v["tessellations"][0]["dims"][0] = serde_json::json!(7)),
            edit(&|v| v["sigma_sq"] = serde_json::Value::Null),
            edit(&|v| v["sigma_sq"] = serde_json::json!(-1.0)),
            edit(&|v| v["tessellations"][0]["tau"] = serde_json::json!(0.5)),
            edit(&|v| {
                let t = &mut v["tessellations"][0];
                let d = t["dims"].as_array().unwrap().len();
                let first: Vec<serde_json::Value> = t["centres"].as_array().unwrap()[..d].to_vec();
                t["centres"].as_array_mut().unwrap().extend(first);
                t["mus"]
                    .as_array_mut()
                    .unwrap()
                    .push(serde_json::json!(0.0));
            }),
        ];
        let mut s = case.sampler(SEED + 2);
        let before = s.state().unwrap();
        for state in &bad {
            assert!(matches!(
                s.restore(state),
                Err(thiessen::Error::IncompatibleState { .. })
            ));
            assert_eq!(s.state().unwrap(), before);
        }
    }
}
