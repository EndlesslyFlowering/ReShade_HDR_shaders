# Democratisation of HDR analysis and other HDR things

## [Download](https://github.com/EndlesslyFlowering/ReShade_HDR_shaders/archive/refs/heads/master.zip)

## The HDR analysis shader and other specific HDR shaders will only work in DirectX 11, 12 or the Vulkan API!

## Important features of the HDR analysis shader
Type in the possible values and hit ENTER to apply them. Clearing the input field and hitting ENTER again brings back the default value.
#### CSP_OVERRIDE:
Overrides the colour space the shader uses for its calculations.\
Possible values of interest are `CSP_HDR10` for HDR10 and `CSP_SCRGB` for scRGB.\
Sanity check the values the shader outputs as this is a manual override.

#### CIE_DIAGRAM:
Choose between the CIE 1931 xy diagram or the CIE 1976 UCS u'v' diagram.\
Possible values are `CIE_1931` and `CIE_1976`.

#### Disabling parts of the shader:
You can disable parts of the shader by setting one of the following macros to `NO`:\
`ENABLE_CIE_FEATURES`\
`ENABLE_CLL_FEATURES`\
`ENABLE_CSP_FEATURES`
