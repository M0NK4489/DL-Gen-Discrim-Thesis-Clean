# Sorted Thesis Model Notebooks

This folder contains the cleaned notebook set for the thesis experiments. It is intended to be the supervisor/future-student version of the project: each model has one notebook that shows the current method clearly, without the older exploratory clutter from the main working folders.

The thesis question being tested here is:

Can conditional generative time-series models forecast household energy profiles competitively against established discriminative forecasting baselines, and how much does past kWh context matter?

## Method Summary

The data contains household electricity readings in kWh, sampled every 30 minutes over roughly one year, with static household/location covariates and time-varying covariates. Each experiment resamples the data to a selected frequency, builds fixed-length windows, trains a model, then evaluates predicted/generated future kWh windows against held-out validation households.

The key controls are:

- `FREQ`: temporal resolution of the data, such as `30min`, `1H`, `2H`, or `3H`.
- `PRIMARY_HORIZON`: forecast target duration, such as `24h`, `7d`, or `28d`.
- `SEQ_LEN`: number of target timesteps implied by `FREQ` and `PRIMARY_HORIZON`.
- `CONTEXT_LEN`: number of past timesteps supplied before the target window in context-forecasting runs.

For example, a `28d` horizon at `30min` frequency gives `1344` target steps, while the same `28d` horizon at `1H` gives `672` target steps. The calendar duration is the same; only the resolution changes.

## Experiment Settings

There are two main experimental settings.

1. Context forecasting: models receive a past kWh/context window plus known future covariates and static covariates, then predict the target horizon.
2. No-context generative forecasting: generative models receive known future covariates and static covariates, but no past kWh context, then generate the target horizon.

The context forecasting notebooks use the current fixed-time context rule:

| Horizon | Context Duration |
|---|---:|
| `24h` | `2d` |
| `7d` | `7d` |
| `28d` | `7d` |

This means context duration is comparable across frequencies. A `7d` context is still seven calendar days whether the data is sampled at `30min`, `1H`, `2H`, or `3H`.

## Shared Preprocessing

Across the notebooks, the intended workflow is:

1. Load the household energy dataset.
2. Resample each customer to the selected `FREQ`.
3. Add cyclical calendar covariates for hour, weekday, and month using sin/cos encoding.
4. Encode static categorical fields such as station and trial region as categorical IDs.
5. Scale numeric covariates and kWh values.
6. Split customers into train and validation groups so validation households are unseen during training.
7. Build sliding windows containing context, future target, future time covariates, static numeric covariates, and static categorical covariates.
8. Train the model.
9. Generate validation forecasts/samples.
10. Export metrics and diagnostic plots.

## Generative Models

Each generative model directory contains:

- `*_context_forecasting.ipynb`: context-aware probabilistic forecasting version.
- `*_no_context_forecasting.ipynb`: no-past-context version generated from the current no-context server job template.

Included generative models:

- `TimeVAE`
- `Transformer_TimeVAE`
- `Diffusion`
- `Transformer_Diffusion`
- `Autoregressive_Transformer`
- `DoppelGANger`

For context forecasting, these models are given the same kind of past-context information as the discriminative baselines. For no-context forecasting, only the generative models are included because they can sample plausible outputs directly from covariates without requiring an observed past kWh input window.

## Discriminative Baselines

Each baseline directory contains one context forecasting notebook:

- `TFT`
- `N-Hits`
- `TSMixer`
- `iTransform`

These models require an input/context window for both training and inference, so they belong only to the context forecasting comparison.

## Evaluation Workbooks

The combined Excel files should be placed in this folder for presentation/report analysis.

Expected workbooks:

- `forecast_validation_model_metrics_combined.xlsx`
- `validation_model_metrics_combined.xlsx`

The exact filenames can differ slightly, but the intended distinction is:

- Forecast validation workbook: compares the true forecasting experiments where models receive a context window and predict a future target window. This is the fairest comparison between generative forecasters and discriminative baselines.
- Validation metrics workbook: compares the earlier/no-context conditional generation style results, where generative models produce a target window from covariates without using past kWh context. This is useful for testing whether a model can generate plausible household profiles when a future user has no historical consumption record.

## How To Read The Results

Lower is better for most error and distribution metrics. Coverage is different: it is best when it is close to the target interval level, usually around `0.90` for a 90% prediction interval.

Important columns usually include:

- `model_type` or `model_family`: which model produced the result.
- `freq`: the temporal resolution.
- `horizon`: the forecast duration.
- `seq_len`: number of target timesteps.
- `context_or_input_len`: number of past input/context timesteps, if applicable.
- `metric`: the metric name.
- `value`: the metric value.

Metric interpretation:

- `MAE_median`: average absolute error of the median forecast. Lower means the typical predicted profile is closer to the real profile.
- `RMSE_mean`: squared-error-sensitive point metric. Lower means better, and large peak errors are penalised more heavily than in MAE.
- `PeakMAE`: error on peak usage values. Lower means the model is better at matching high-consumption events.
- `CRPS`: probabilistic forecast quality based on the predictive distribution/quantiles. Lower means the uncertainty distribution is better calibrated and sharper.
- `QuantileLoss`: pinball loss across forecast quantiles. Lower means better quantile predictions.
- `Winkler`: interval score. Lower means prediction intervals are narrower while still covering the true values.
- `Coverage`: proportion of real values inside the prediction interval. For a 90% interval, values near `0.90` are ideal; very low means under-coverage, very high can mean overly wide intervals.
- `KL_hist`: histogram KL divergence between real and generated/forecast kWh distributions. Lower means the marginal kWh distribution is closer.
- `DTW_mean`: Dynamic Time Warping distance between real and predicted profiles. Lower means the sequence shapes align better, even if peaks are slightly shifted.
- `FTSD`: Fréchet-style distance over summary time-series features. Lower means the generated/forecast windows look closer to real windows in aggregate.
- `ACF_error`: difference in autocorrelation structure. Lower means the model better preserves temporal dependence and repeating usage patterns.
- `corr_real` and `corr_fake`: compare relationships between covariates, such as temperature, and kWh. The gap between them matters more than either value alone.
- `CompositeScore`: project-specific summary score used for quick tuning scans. Lower is better, but it should not replace inspecting the individual metrics.

## Recommended Presentation Use

For presentation slides, avoid showing every metric at once. A useful breakdown is:

1. Point forecast accuracy: `MAE_median`, `RMSE_mean`, `PeakMAE`.
2. Probabilistic quality: `CRPS`, `QuantileLoss`, `Winkler`, `Coverage`.
3. Distribution/profile realism: `KL_hist`, `DTW_mean`, `FTSD`, `ACF_error`.

The strongest result is not necessarily the model with the best value on one metric. A good model should have low point error, reasonable coverage, and realistic distribution/profile metrics.

When comparing frequency and horizon:

- Compare models within the same horizon first, because a `24h` task and a `28d` task have different difficulty.
- Compare frequencies within the same horizon to see whether finer temporal resolution helps or makes the sequence harder to model.
- Use the forecast validation workbook for direct comparison against baselines.
- Use the no-context/generative validation workbook to discuss practical deployment scenarios where a household has covariates but no historical kWh record.

## Support

`_support/plot_customer_window_method_visualisation.ipynb` creates the context/target window visualisation used for reports and presentations.
