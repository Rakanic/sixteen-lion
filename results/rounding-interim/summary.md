# Rendering-aware rounding: INTERIM, 18 of 48 models (aerial_rocks_02, metal_plate, rough_wood); final in results/rounding/

Decoder frozen, block scales unchanged, same storage. FLIP change vs round-to-nearest (negative = better).

| format   | trained with   |   models | texel-rounding, test lights   | shading-rounding, test lights   | texel-rounding, unseen lights   | shading-rounding, unseen lights   | shading beats texel (unseen)   |
|:---------|:---------------|---------:|:------------------------------|:--------------------------------|:--------------------------------|:----------------------------------|:-------------------------------|
| mxfp4    | shading        |        4 | -0.9 %                        | -5.0 %                          | -1.0 %                          | -5.0 %                            | 4/4                            |
| mxfp4    | texel          |        6 | -2.3 %                        | -7.5 %                          | -2.3 %                          | -7.1 %                            | 6/6                            |
| nvfp4    | shading        |        4 | -0.7 %                        | -4.0 %                          | -0.8 %                          | -4.1 %                            | 4/4                            |
| nvfp4    | texel          |        4 | -0.7 %                        | -4.5 %                          | -0.8 %                          | -4.1 %                            | 4/4                            |

## Win counts over all models

- test lights: shading-rounding beats nearest in 18/18, beats texel-rounding in 18/18; texel-rounding beats nearest in 18/18
- unseen lights: shading-rounding beats nearest in 18/18, beats texel-rounding in 18/18; texel-rounding beats nearest in 18/18

## Per material (mean over formats, training losses, seeds)

| material        |   texel-rounding, test |   shading-rounding, test |   texel-rounding, unseen |   shading-rounding, unseen |
|:----------------|-----------------------:|-------------------------:|-------------------------:|---------------------------:|
| aerial_rocks_02 |                   -5.1 |                    -11.9 |                     -4.9 |                      -11.5 |
| metal_plate     |                   -0.6 |                     -2.6 |                     -0.7 |                       -2.7 |
| rough_wood      |                   -1   |                     -6.7 |                     -1   |                       -6.3 |

## Seed spread (shading-rounding FLIP change, test lights)

- largest gap between the two seeds: 0.6 percentage points

## Equivalent memory (texel-trained models, from the milestone curves)

- mxfp4: FLIP -7.5 % equals about 22 % more latent memory at plain rounding
- nvfp4: FLIP -4.5 % equals about 11 % more latent memory at plain rounding
