#include "ReShade.fxh"
#include "lilium__colour_space.fxh"

uniform uint INPUT_TRC
<
  ui_label  = "input TRC";
  ui_type   = "combo";
  ui_items  = "sRGB\0"
              "gamma 2.2\0"
              "gamma 2.4\0"
              "PQ\0";
> = 0;

#define TRC_SRGB     0
#define TRC_GAMMA_22 1
#define TRC_GAMMA_24 2
#define TRC_PQ       3

uniform float SDR_WHITEPOINT_NITS
<
  ui_label = "SDR whitepoint (in nits)";
   ui_type = "drag";
    ui_min = 1.f;
    ui_max = 300.f;
   ui_step = 1.f;
> = 80.f;

uniform bool DO_GAMMA_ADJUST
<
  ui_label = "gamma adjust";
> = false;

uniform float GAMMA_ADJUST
<
  ui_label = "gamma adjust";
  ui_type  = "drag";
  ui_min   = -1.f;
  ui_max   =  1.f;
  ui_step  =  0.001f;
> = 0.f;

uniform bool CLAMP
<
  ui_category = "clamping";
  ui_label    = "clamp values";
> = false;

uniform float CLAMP_NEGATIVE_TO
<
  ui_category = "clamping";
  ui_label    = "clamp negative values to";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      =  0.f;
  ui_step     =  0.1f;
> = -125.f;

uniform float CLAMP_POSITIVE_TO
<
  ui_category = "clamping";
  ui_label    = "clamp positive values to";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 125.f;
  ui_step     = 0.1f;
> = 125.f;


void scRGB_TRC_Fix(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 fixedGamma = input;

  if (INPUT_TRC == TRC_SRGB) {
    fixedGamma = Extended_sRGB_TRC(fixedGamma);
  }
  else if (INPUT_TRC == TRC_GAMMA_22) {
    fixedGamma = Extended_22_TRC(fixedGamma);
  }
  else if (INPUT_TRC == TRC_GAMMA_24) {
    fixedGamma = Extended_24_TRC(fixedGamma);
  }
  else if (INPUT_TRC == TRC_PQ) {
    fixedGamma = mul(BT2020_To_BT709, PQ_EOTF(fixedGamma)) * 125.f;
  }

  if (CLAMP)
    fixedGamma = clamp(fixedGamma, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);

#if (CSP_OVERRIDE == CSP_PS5)
    fixedGamma = mul(BT709_To_BT2020, fixedGamma);
#endif

  if (DO_GAMMA_ADJUST)
    fixedGamma = ExtendedGammaAdjust(fixedGamma, 1.f + GAMMA_ADJUST);

//  if (dot(BT709_To_XYZ[1].rgb, fixedGamma) < 0.f)
//    fixedGamma = float3(0.f, 0.f, 0.f);

#if (CSP_OVERRIDE == CSP_PS5)
    fixedGamma *= (SDR_WHITEPOINT_NITS / 100.f);
#else
    fixedGamma *= (SDR_WHITEPOINT_NITS / 80.f);
#endif

  fixedGamma = fixNAN(fixedGamma);

  //fixedGamma = clamp(fixedGamma, -65504.f, 125.f);

  output = float4(fixedGamma, 1.f);
}


technique lilium__scRGB_TRC_Fix
{
  pass scRGB_TRC_Fix
  {
    VertexShader = PostProcessVS;
     PixelShader = scRGB_TRC_Fix;
  }
}
