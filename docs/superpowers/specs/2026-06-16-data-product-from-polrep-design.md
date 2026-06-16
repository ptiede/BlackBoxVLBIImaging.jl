# Data product follows the sky polarization representation

**Date:** 2026-06-16

## Problem

The data loader (`src/data/dataloader.jl`) always extracts `Coherencies()` — full
4-element polarization coherency matrices. This is wrong when the sky model is
`TotalIntensity` (Stokes I only): in that case we should fit **complex Stokes I
visibilities** (`Comrade.Visibilities`), not coherencies.

The polarization choice lives in the image config (`[model] polrep`) and is parsed in
`src/config/sky_config.jl`. The instrument side already adapts (the gain scheme selects
`SingleStokesGain` vs `JonesG`), but the data product does not. This design closes that
gap by deriving the data product from `polrep`.

## Decisions

- **Derive the data product from `polrep`** (not an explicit data-config key). The user
  sets only `polrep`; the data product follows automatically. `TotalIntensity` →
  `Visibilities`, `PolExp`/`Poincare` → `Coherencies` (unchanged).
- **`TotalIntensity` + dlist errors** with a clear message. dlist (`read_dlist`) builds
  coherency matrices by construction; we do not convert. Polarized polreps with dlist are
  unchanged.

## Changes

1. **`src/config/sky_config.jl` — single source of truth for polrep.**
   Extract `sky_polrep(cfg) = _parse_polrep(String(get(get(cfg, "model", ...), "polrep",
   "PolExp")))` and call it at the existing line 169 so `build_sky_config` and the data
   path parse `polrep` identically.

2. **`src/data/dataloader.jl` — product selection.**
   Add `data_product(::TotalIntensity) = Visibilities` and `data_product(::PolRep) =
   Coherencies`. `build_data_uvfits` gains a `polrep::PolRep` keyword and extracts
   `data_product(polrep)(; time_average, frequency_average)` in place of the hardcoded
   `Coherencies`. `add_fractional_noise` and `reset_mounts!` already accept visibility
   tables, so the rest is unchanged. Rename the local `dcoh → dvis` for accuracy.

3. **`src/data/dataloader.jl` — dlist guard.**
   `build_data_dlist` gains the same `polrep::PolRep` keyword and errors when
   `polrep isa TotalIntensity`: "dlist files are polarized coherencies; use a polarized
   polrep or convert to uvfits."

4. **`src/config/data_config.jl` — thread it through.**
   `build_data_config` gains `polrep::PolRep = PolExp()` (default preserves today's
   `Coherencies` behavior for existing direct callers/tests) and passes it to the uvfits
   and dlist builders.

5. **`src/pipeline/run.jl` — wire it.**
   Parse `polrep = sky_polrep(skycfg)` before loading data and pass it:
   `build_data_config(datacfg; base_dir=..., polrep)`.

## Data flow

```
image.toml [model] polrep
   -> sky_polrep(skycfg)
   -> build_data_config(datacfg; polrep)
   -> data_product(polrep)            # Visibilities | Coherencies
   -> extract_table(uvd, product(; time_average, frequency_average))
```

The existing "build data before sky (so beamsize feeds the sky prior)" ordering is
untouched — only the cheap polrep-string parse moves earlier.

## Backward compatibility

`build_data_config` / `build_data_uvfits` / `build_data_dlist` default to `polrep =
PolExp()`, yielding `Coherencies` exactly as today. The two existing callers that don't
pass `polrep` (`test/runtests.jl`, `drivers/reactant_consistency_scan.jl`) are unaffected.
No config schema change.

## Testing

- Unit: `data_product(TotalIntensity()) === Visibilities` and
  `data_product(PolExp()) === Coherencies`.
- Unit: `build_data_dlist(...; polrep=TotalIntensity())` is `@test_throws ErrorException`
  (errors before/independent of touching a real file is acceptable).
- The uvfits extraction path is exercised by the existing data-dependent integration test
  (guarded on workshop data being present).
