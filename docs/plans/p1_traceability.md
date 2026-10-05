# P-I traceability: `prereg/p1.md` to code, tests and pre-flight output

One row per rule and per reported quantity in `prereg/p1.md` (author's item P3).

- **Files:**
  - `run.py`, `p1_data.py`, `p1_hover.py`, `p1_rules.py`, `launch.py`: in
    `experiments/p1_quadrotor/`;
  - `analysis.py`, `train.py`, `models.py`, `geometry.py`: in `lens/`;
  - `quadrotor.py`: `plants/quadrotor.py`.
- **Tests:**
  - `pipe::`: `tests/test_p1_pipeline.py`;
  - `models::`: `tests/test_p1_models.py`;
  - `quad::`: `tests/test_quadrotor.py`;
  - `lens::`: `tests/test_lens.py`.
- **Pre-flight output:** under `results/p1_preflight/`, from `preflight.py all`. The
  stopping rules are under `preflight.py stops`, in `preflight_stops.json`.

## Plant and data

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Plant | 12 states, 4 thrusts, approved parameters, g = 9.81, RK4 dt = 0.01 | `quadrotor.f`, `rk4_step`, `QuadrotorParams`; `p1_data.quadrotor_plant` | `quad::test_params_are_the_approved_values`, `test_rk4_is_fourth_order`, `test_hover_is_an_equilibrium` | Not exercised: the pre-flight's plant is synthetic, with the same dimensions, trim and box (`preflight.synthetic_plant`) |
| Plant | No drag | `QuadrotorParams(drag=False)`; config `plant.drag: false` | `quad::test_linear_drag_flag` | Same |
| Plant | Yaw not wrapped | `quadrotor.f`, `rk4_step` (no modulo) | `pipe::test_yaw_is_not_wrapped` | Same |
| Data | Box ranges, symmetric about hover; thrust u0(1 ± 0.5), never clipped | `p1_data.quadrotor_half_widths`, `sample` | `pipe::test_sampling_order_seed_and_box`, `test_quadrotor_sampling_box_only` | `data_manifest.csv`, `data/*.npz` |
| Data | Sizes 20,000 / 5,000 / 5,000; train, val, test in order from `default_rng(0)` | `p1_data.make_dataset`; config `sampling` | `pipe::test_sampling_order_seed_and_box` | `data/train.npz` etc. (real sizes) |
| Data | Target y = (x_{t+1} − x_t)/dt; z-scored with training statistics; J_std | `p1_data.target_fn`, `standardiser`, `to_std_jacobian` | `pipe::test_dataset_shapes_standardisation_and_jacobians`, `test_physical_jacobian_inverts_the_standardisation` | `data/*.npz` |
| Data | SHA-256 manifest verified before use | `p1_data.save_dataset`, `load_dataset` | `pipe::test_manifest_round_trip_and_tamper_refusal` | `data_manifest.csv`; `preflight_stops.json` `manifest_mismatch` |

## Surrogates and training

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Surrogates | The two architectures | `models.trace`, `_run`, `forward` | `models::test_forward_matches_a_hand_written_pass`, `test_shapes_and_first_layer` | All 40 runs, `train/runs/*.json` |
| Surrogates | Initialisations; shared weights within a seed | `models.init_params`, `_linear` | `models::test_initialisations` | Same |
| Surrogates | Shared minibatch stream | `train.train` (`fold_in(PRNGKey(seed), step)`) | `models::test_minibatches_are_the_same_for_both_initialisations_of_a_seed` | Same |
| Surrogates | First layer shared across cells | `models.init_params` (E, b drawn first) | `pipe::test_first_layer_is_shared_across_architectures_and_depths` | Same |
| Surrogates | The 40-model grid | `run.grid` | `pipe::test_grid_is_the_planned_40` | `train/train_summary.csv` (40 rows) |
| Training | Minibatch Adam, batch 2,048, lr 3e-3, evaluation every 500 steps on the full validation set | `train.train`; `run.train_cfg`; config `training` | `models::test_training_fits_a_synthetic_target`, `test_minibatch_size_is_checked_and_patience_can_be_disabled`; `pipe::test_config_training_defaults_and_long_budget_subset` | `train/history/*.csv` (`val_mse` every 500 steps) |
| Training | Early stopping: patience 10%, tolerance 1%; the best held-out parameters returned | `train.train` | `models::test_early_stopping_patience_and_tolerance`, `test_training_fits_a_synthetic_target` | `train/train_summary.csv` (`best_step`, `stopped_step`, `stopped_early`) |
| Training | Long-budget subset: 10 runs, fixed budget, snapshots | `run.stage_train(long=True)`; `grid(cfg, "long_budget")` | `pipe::test_long_run_snapshots_history_and_manifest`, `test_config_training_defaults_and_long_budget_subset` | `train_long/`, `checkpoints/long/<name>/step*.npz` |
| Training | Lens log: z*, κ, ‖c⊥‖, r* and r_eff widths; along u_min and d₁: r*, r_eff, D, r_eff/D, S | `train.lens_record`; `run.history_arrays` | `models::test_lens_logging`, `test_lens_record_along_u_min` | `train*/history/*.csv`, `*.npz` (`widths`, `widths_eff`, `u_min`, `z_star`) |
| Surrogates | Execution: launcher, pinned environment, 3 processes | `launch.ENV`, `env`, `run_pool`, `N_PROC` | `pipe::test_launcher_pins_the_flags` | `train*/logs/*.log` (every run started by the launcher) |

## Definitions

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Definitions | μ; hover_std; ‖z* − μ‖, ‖z* − hover_std‖ | `run.analyse_one` | `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `z_star_to_mean`, `z_star_to_hover` |
| Definitions | d₁ | `analysis.data_d1` | `pipe::test_lens_distances_sharpness_and_directions` | `analyse.csv` `d1_eigen_ratio`, `d1_*` |
| Definitions | u_min | `analysis.u_min` | `models::test_lens_record_along_u_min`; `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `u_min_*`; `analyse_lens.npz` |
| Definitions | D(d) | `analysis.half_width` | `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `u_min_D`, `d1_D` |
| Definitions | κ; degenerate; r_eff | `geometry.lens`, `line` | `lens::` (geometry tests); `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `kappa`, `degenerate`, `norm_c_perp`, `u_min_r_eff` |
| Definitions | Lens distance ρ_eff, ordering identical to ρ | `analysis.lens_distances` | `pipe::test_lens_distances_sharpness_and_directions`, `test_analyse_one_quantities_as_defined` | `analyse.csv` `dist_kind` = rho_eff |
| Definitions | Jacobian error (relative Frobenius, full 12 × 16) | `analysis.jacobians`, `jacobian_error` | `pipe::test_jacobians_against_finite_differences`, `test_surrogate_equal_to_the_truth_gives_zero_error`, `test_analyse_one_quantities_as_defined` | `analyse.csv` `err_median`, `err_q*` |
| Definitions | R (ceil-sized 10% and 50% sets, medians) | `analysis.near_far` | `pipe::test_near_far_split`, `test_analyse_one_quantities_as_defined` | `analyse.csv` `nf_ratio`, `nf_near`, `nf_far`, `nf_n_near`, `nf_n_far` |
| Definitions | rel-MSE (validation, test) | `run.analyse_one` | `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `rel_mse_val`, `rel_mse_test` |
| Definitions | Cells; "holds in a cell" (4 of 5) | `p1_rules._cells`; config `predictions.min_seeds` | `pipe::test_prediction_rules_on_constructed_rows` | `predictions.json` |

## Predictions, H1 and G2

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Prediction 1 | Rule (a) ‖z* − μ‖ ≤ 0.1 and (b) R ≥ 3; per cell; overall in every cell | `p1_rules.p1`, `p1_overall` | `pipe::test_prediction_rules_on_constructed_rows`, `test_divergence_is_recorded_and_fails_every_rule` | `predictions.json` `p1`, `p1_overall` |
| Prediction 1 | Each clause separately; ‖z* − hover_std‖ | `p1_rules.p1` (`n_a`, `n_b`); `run.analyse_one` | Same; `pipe::test_analyse_one_quantities_as_defined` | `predictions.json` `p1[].n_a`, `n_b`; `analyse.csv` `z_star_to_hover` |
| Prediction 2 | (a) initial median r* (64 directions, rng 0) in [0.4675, 1.87] | `run.initial_r_star_median`; `p1_rules.p2` | `pipe::test_p3_spearman_and_initial_r_star`, `test_prediction_rules_on_constructed_rows` | `analyse.csv` `r_star_init_median`; `predictions.json` `p2.a_*` |
| Prediction 2 | (b) R_torch < R_zero paired by seed, 4 of 5 per cell; overall | `p1_rules.p2` | Same; divergence test | `predictions.json` `p2.b_cells`, `p2.holds` |
| Prediction 2 | Both arms' held-out rel-MSE | `run.analyse_one` | `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `rel_mse_val`, `rel_mse_test` |
| Prediction 3 | Score D/r_eff along u_min; affected fraction (error > 3 × far median) | `analysis.direction_sharpness`; `run.analyse_one` | `pipe::test_analyse_one_quantities_as_defined` | `analyse.csv` `u_min_sharpness`, `affected_frac` |
| Prediction 3 | Positive Spearman ≥ 0.7 over 40; average-rank ties; fails if any model diverged | `p1_rules.p3` | `pipe::test_prediction_rules_on_constructed_rows` (negative ρ fails), `test_divergence_is_recorded_and_fails_every_rule` | `predictions.json` `p3` |
| Prediction 3 | Coverage along u_min; both along d₁ (reported) | `analysis.direction_sharpness`; `run.p3_spearman` | `pipe::test_lens_distances_sharpness_and_directions`, `test_p3_spearman_and_initial_r_star` | `analyse.csv` `u_min_coverage`, `d1_*` |
| Prediction 3 | Within-initialisation Spearman (n = 20 each; reported) | `p1_rules.p3` (`within_init`) | `pipe::test_prediction_rules_on_constructed_rows` | `predictions.json` `p3.within_init` |
| Prediction 4 | Primary ratio P_out/P_1 along u_min (2,001 points on [−D, D], s(0) at t = 0; h₁ without the head) | `analysis.attenuation_P`; `models.hidden`; config `analysis.p4_grid` | `pipe::test_attenuation_P_is_unit_free_and_exact_at_the_centre`, `test_analyse_one_quantities_as_defined` | `analyse.csv` `p4_P_ratio_u_min`, `p4_P_out_u_min`, `p4_P_1_u_min` |
| Prediction 4 | Rule on the two NormedLinear 3-block cells; point prediction (median of five zero-bias models in [0.1, 0.4]) | `p1_rules.p4`; config `predictions.p4` | `pipe::test_prediction_rules_on_constructed_rows` | `predictions.json` `p4` |
| Prediction 4 | Reported: pre-norm 3-block cells; head-on-block-1 ratio | `p1_rules.p4`; `analysis.attenuation_S`; `models.trace(upto=1)` | `pipe::test_truncated_trace_and_attenuation_S`, `test_prediction_rules_on_constructed_rows` | `predictions.json` `p4.reported_*`; `analyse.csv` `p4_S_*` |
| Prediction 5 | Per-run rule (fold ≥ 10, ratio ≥ 0.1, κ ≤ 0.1 at the end) | `run.p5_run` | `pipe::test_p5_rule` | `p5_train.csv`, `p5_train_long.csv` |
| Prediction 5 | Primary test on the long-budget subset (end = last step; per architecture, both) | `run.stage_p5`, `p5_cells`; `p1_rules.p5` | `pipe::test_stage_p5_reads_the_lens_logs`, `test_prediction_rules_on_constructed_rows` | `p5_train_long_cells.csv`; `predictions.json` `p5` |
| Prediction 5 | Early-stopping version (end = best step; per zero-bias cell); d₁ secondary | `run.stage_p5`; history `r_eff_over_D_d1` | `pipe::test_stage_p5_reads_the_lens_logs` | `p5_train_cells.csv`; `train/history/*.csv` |
| Prediction 5 | The race: 2 × 2 per cell with diverged counted apart; R and r_eff/D at each snapshot | `p1_rules.race`; `run.stage_analyse_long`, `race_point` | `pipe::test_prediction_rules_on_constructed_rows`, `test_analyse_long_traces_prediction_1_over_snapshots` | `predictions.json` `race`; `analyse_long.csv` |
| Hover | A := ∂y/∂x, B := ∂y/∂u in physical units, truth and surrogate alike | `p1_hover.true_y_jacobians`, `physical_jacobian` | `pipe::test_quadrotor_true_hover_jacobians_match_the_rk4_map`, `test_physical_jacobian_inverts_the_standardisation` | `hover.csv` |
| Hover | Relative Frobenius errors of A and B | `p1_hover.hover_check` | `pipe::test_surrogate_equal_to_the_truth_gives_zero_error`, `test_hover_check_signs_and_instability` | `hover.csv` `rel_err_A`, `rel_err_B` |
| Hover | Sign mask from ∂f (\|∂f\| > 1e-3 max, per matrix); the masked comparison; the truth-vs-field precondition | `p1_hover.true_f_jacobians`, `sign_mask`, `masked_sign_agreement`, `mask_sign_check` | `pipe::test_sign_mask_from_the_vector_field` | `hover.csv` `sign_agree_A/B`, `n_sign_A/B` |
| Hover | Unmasked disagreement count (reported) | `p1_hover.unmasked_disagreements` | `pipe::test_sign_mask_from_the_vector_field` | `hover.csv` `n_sign_disagree_unmasked_A/B` |
| Hover | LQR gain with Bryson Q, R; closed-loop eigenvalues, stable, abscissa | `p1_hover.bryson`, `hover_check`; `control/lqr.py` | `pipe::test_lqr_known_cases`, `test_surrogate_equal_to_the_truth_gives_zero_error`, `test_analyse_one_quantities_as_defined` (Bryson) | `hover.csv` `rel_err_K`, `spectral_abscissa`, `stable`, `true_closed_loop_abscissa` |
| Hover | "No stabilising gain" | `p1_hover.hover_check` | `pipe::test_no_stabilising_gain_counts_for_h1` | `hover.csv` `no_stabilising_gain`, `lqr_ok` |
| H1 | Per model: masked sign, no gain or unstable; per cell 4 of 5; overall every zero-bias cell | `p1_hover.h1_fail`; `p1_rules.h1` | `pipe::test_no_stabilising_gain_counts_for_h1`, `test_prediction_rules_on_constructed_rows`, `test_divergence_is_recorded_and_fails_every_rule` | `hover.csv` `h1_fail`; `predictions.json` `h1` |
| Hover | Trims: committed file, SHA-256 verified, inside the box, trims of RK4; truth depends only on yaw | `run.stage_trims`, `load_trims`; `p1_hover.sample_trims`, `check_trims`, `trim_truth` | `pipe::test_trims_stage_regeneration_identity`, `test_quadrotor_trims_are_trims_inside_the_box` | `hover_trims.csv` (200 trims per model) |
| Hover | Per trim: errors, sign agreement on the trim's mask, ρ_eff, variation error | `p1_hover.trim_check` | `pipe::test_surrogate_equal_to_the_truth_gives_zero_error`, `test_sign_mask_from_the_vector_field` | `hover_trims.csv` |
| Hover | Per model: medians and Spearman correlations with lens distance | `run.hover_one` | `pipe::test_surrogate_equal_to_the_truth_gives_zero_error` | `hover.csv` `trims_*` |
| Gate G2 | At least 2 of 4 zero-bias cells; P-III cells | `p1_rules.g2` | `pipe::test_prediction_rules_on_constructed_rows` | `predictions.json` `g2` |

## Stopping rules

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Stopping rules | Corollary 1 deviation > 1e-10 stops the analysis | `run.check_corollary1`, `stage_analyse`; `analysis.corollary1_line` | `pipe::test_corollary1_deviation_stops_the_analysis`, `test_stage_analyse_stops_on_a_forced_corollary1_deviation`, `test_corollary1_line_on_a_random_layer` | `analyse.csv` `corollary1_max_rel_dev`; `preflight_stops.json` `corollary1_deviation` |
| Stopping rules | Data-manifest mismatch stops | `p1_data.load_dataset`; `provenance.verify_sha256` | `pipe::test_manifest_round_trip_and_tamper_refusal` | `preflight_stops.json` `manifest_mismatch` |
| Stopping rules | Merging: missing run, SHA-256 mismatch, several commits, dirty tree | `run.stage_gather` | `pipe::test_gather_verifies_every_run`, `test_gather_refuses_a_dirty_worktree` | `train*/outputs_manifest.csv`; `preflight_stops.json` `missing_run`, `dirty_worktree_at_gather` |
| Stopping rules | Divergence recorded; fails every rule; no seed replaced | `train.train` (`diverged`); `run.stage_train`; `p1_rules` | `pipe::test_divergence_is_recorded_and_fails_every_rule` | `preflight_stops.json` `injected_nan`; `train_summary.csv` `diverged` |
| Stopping rules | Interruption restarts from step 0; no resume | `run.stage_train` (removes the stale manifest, then trains from step 0) | `pipe::test_completed_runs_are_skipped_and_others_restart_from_step_0` | (By construction) |
| Stopping rules | Atomic outputs, manifest last; completed runs skipped | `run.atomic_write`, `save_npz`, `write_manifest`, `run_complete` | `pipe::test_save_npz_is_deterministic_and_atomic`, `test_completed_runs_are_skipped_and_others_restart_from_step_0` | `train*/runs/*_manifest.csv` |
| Gating | Quadrotor stages refuse without the tag and on a dirty tree | `run.main`; `provenance.require_prereg` | `pipe::test_gated_stages_refuse_without_the_prereg_tag` | (Not run: the pre-flight calls stage functions on synthetic data) |

## Reported regardless

| p1.md section | Rule or quantity | Implemented by | Test | Pre-flight output |
| --- | --- | --- | --- | --- |
| Reported regardless | Training history, lens log, divergences | `run.stage_train`, `history_arrays` | `pipe::test_long_run_snapshots_history_and_manifest` | `train*/history/`, `train_summary.csv` |
| Reported regardless | Final lens arrays (z*, widths, u_min) | `run.stage_analyse` | `pipe::test_stage_analyse_stops_on_a_forced_corollary1_deviation` (arrays present) | `analyse_lens.npz` |
| Reported regardless | Config, commit, versions, data manifest, per-run manifests | `run.meta`, `write_manifest`, `stage_gather` | `pipe::test_long_run_snapshots_history_and_manifest`, `test_gather_verifies_every_run` | `train*/runs/*.json`, `predictions.json` (meta) |
