#include "ReShade.fxh"
#include "colorspace.fxh"

uniform uint INPUT_GAMMA
<
  ui_label  = "input Gamma";
  ui_type   = "combo";
  ui_items  = "sRGB\0"
              "gamma 2.2\0"
              "gamma 2.4\0";
> = 0;

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


void scRGB_gamma_fix(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 fixedGamma = input;

  if (CLAMP)
    fixedGamma = clamp(fixedGamma, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);

  if (INPUT_GAMMA == 0)
    fixedGamma = X_sRGB_EOTF(fixedGamma);
  else if (INPUT_GAMMA == 1)
    fixedGamma = X_power_EOTF(fixedGamma, 2.2f);
  else
    fixedGamma = X_power_EOTF(fixedGamma, 2.4f);

  if (DO_GAMMA_ADJUST)
    fixedGamma = X_gamma_adjust(fixedGamma, 1.f + GAMMA_ADJUST);

//  if (dot(BT709_to_XYZ[1].rgb, fixedGamma) < 0.f)
//    fixedGamma = float3(0.f, 0.f, 0.f);

  fixedGamma *= (SDR_WHITEPOINT_NITS / 80.f);

  fixedGamma = fixNAN(fixedGamma);

  //fixedGamma = clamp(fixedGamma, -65504.f, 125.f);

  output = float4(fixedGamma, 1.f);
}


technique scRGB_gamma_fix
{
  pass scRGB_gamma_fix
  {
    VertexShader = PostProcessVS;
     PixelShader = scRGB_gamma_fix;
  }
}
