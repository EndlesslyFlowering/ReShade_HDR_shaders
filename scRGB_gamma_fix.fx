#include "ReShade.fxh"
#include "colorspace.fxh"

uniform float SDR_WHITEPOINT_NITS
<
  ui_label = "SDR whitepoint (in nits)";
   ui_type = "drag";
    ui_min = 1.f;
    ui_max = 300.f;
   ui_step = 1.f;
> = 80.f;


void scRGB_gamma_fix(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 fixedGamma;

  fixedGamma = clamp(input, -65504.f, 65504.f);

  fixedGamma = XsRGB_inverse_EOTF(fixedGamma);

  fixedGamma *= (SDR_WHITEPOINT_NITS / 80.f);

  fixedGamma = fixNAN(fixedGamma);

  fixedGamma = clamp(fixedGamma, -65504.f, 125.f);

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