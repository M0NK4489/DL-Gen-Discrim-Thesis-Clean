# Sorted Thesis Model Notebooks

This folder contains the current working notebooks for the thesis experiments, copied out of the cluttered development tree into one readable structure.

The experiment has two main settings:

1. Context forecasting: models receive a past kWh/context window plus known future covariates and static covariates, then predict the target horizon.
2. No-context generative forecasting: generative models receive covariates but no past kWh context, then generate the target horizon.

The context forecasting notebooks use the current fixed-time context rule:

| Horizon | Context Duration |
|---|---:|
| `24h` | `2d` |
| `7d` | `7d` |
| `28d` | `7d` |

The target sequence length is still derived from the selected frequency and horizon. For example, at `30min`, a `28d` target is `1344` steps; at `1H`, the same `28d` target is `672` steps.

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

`TimeGAN` is not included in this sorted experiment set because the current comparable context-forecasting workflow was not maintained for it, and it was unstable in prior training runs.

## Discriminative Baselines

Each baseline directory contains one context forecasting notebook:

- `TFT`
- `N-Hits`
- `TSMixer`
- `iTransform`

These models require an input/context window for both training and inference, so they belong only to the context forecasting comparison.

## Support

`_support/plot_customer_window_method_visualisation.ipynb` creates the context/target window visualisation used for reports and presentations.

## Notes

These notebooks are copied or converted from the current working model folders and server-job templates. Outputs have been cleared in this `sorted/` copy so the notebooks open cleanly and rerun from scratch.
