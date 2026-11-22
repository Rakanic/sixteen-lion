# Rendering-aware rounding: 48 trained models

Decoder frozen, block scales unchanged, same storage. FLIP change vs round-to-nearest (negative = better).

| format   | trained with   |   models | texel-rounding, test lights   | shading-rounding, test lights   | texel-rounding, unseen lights   | shading-rounding, unseen lights   | shading beats texel (unseen)   |
|:---------|:---------------|---------:|:------------------------------|:--------------------------------|:--------------------------------|:----------------------------------|:-------------------------------|
| mxfp4    | shading        |       12 | -3.1 %                        | -7.8 %                          | -3.2 %                          | -7.9 %                            | 12/12                          |
| mxfp4    | texel          |       12 | -3.0 %                        | -8.1 %                          | -3.2 %                          | -8.2 %                            | 12/12                          |
| nvfp4    | shading        |       12 | -2.6 %                        | -7.0 %                          | -2.7 %                          | -7.1 %                            | 12/12                          |
| nvfp4    | texel          |       12 | -2.7 %                        | -7.4 %                          | -2.9 %                          | -7.5 %                            | 12/12                          |

## Win counts over all models

- test lights: shading-rounding beats nearest in 48/48, beats texel-rounding in 48/48; texel-rounding beats nearest in 48/48
- unseen lights: shading-rounding beats nearest in 48/48, beats texel-rounding in 48/48; texel-rounding beats nearest in 48/48

## Per material (mean over formats, training losses, seeds)

| material        |   texel-rounding, test |   shading-rounding, test |   texel-rounding, unseen |   shading-rounding, unseen |
|:----------------|-----------------------:|-------------------------:|-------------------------:|---------------------------:|
| aerial_rocks_02 |                   -4.7 |                    -10.6 |                     -4.4 |                      -10.2 |
| castle_brick_07 |                   -3.8 |                    -10.7 |                     -3.8 |                      -10.1 |
| denim_fabric    |                   -6.6 |                    -12.5 |                     -7.5 |                      -14   |
| metal_plate     |                   -0.6 |                     -2.6 |                     -0.7 |                       -2.7 |
| rough_wood      |                   -1   |                     -6.7 |                     -1   |                       -6.3 |
| rusty_metal_02  |                   -0.3 |                     -2.3 |                     -0.4 |                       -2.7 |

## Seed spread (shading-rounding FLIP change, test lights)

- largest gap between the two seeds: 4.8 percentage points

## Equivalent memory (texel-trained models, from the milestone curves)

- mxfp4: FLIP -8.1 % equals about 24 % more latent memory at plain rounding
- nvfp4: FLIP -7.4 % equals about 20 % more latent memory at plain rounding
