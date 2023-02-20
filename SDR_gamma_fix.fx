#include "ReShade.fxh"
#include "colorspace.fxh"

//#define _DEBUG

uniform float TARGET_INVERSE_GAMMA
<
  ui_label = "target power Gamma";
  ui_type  = "drag";
  ui_min   = 1.f;
  ui_max   = 3.f;
  ui_step  = 0.01f;
> = 2.2f;

uniform bool USE_BT1886
<
  ui_category = "BT.1886";
  ui_label    = "use BT.1886 gamma for blackpoint correction";
> = false;

uniform float TARGET_WHITEPOINT
<
  ui_category = "BT.1886";
  ui_label    = "target whitepoint (in nits)";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1000.f;
  ui_step     = 0.5f;
> = 100.f;

uniform float TARGET_BLACKPOINT
<
  ui_category = "BT.1886";
  ui_label    = "target blackpoint (in nits)";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.001f;
> = 0.f;


float3 BT1886_gamma(
  const float3 V,
  const float  targetWhitepoint,
  const float  targetBlackpoint,
  const float  targetInverseGamma,
  const float  targetGamma)
{
  const float Lw = targetWhitepoint;
  const float Lb = targetBlackpoint;

  const float powLw = pow(Lw, targetGamma);
  const float powLb = pow(Lb, targetGamma);

  const float powLw_minus_powLb = powLw - powLb;

  const float a = pow(powLw_minus_powLb, targetInverseGamma);
  const float b = powLb /
                  (powLw_minus_powLb);

  const float3 L = a * pow(max(V + b.xxx, 0.f.xxx), targetInverseGamma);

  return pow(L / targetWhitepoint, targetGamma);
}


void SDR_gamma_fix(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 fixedGamma = sRGB_inverse_EOTF(input);

  const float targetGamma = 1.f / TARGET_INVERSE_GAMMA;

  fixedGamma = pow(fixedGamma, targetGamma);

  if (USE_BT1886)
  {
    fixedGamma = BT1886_gamma(fixedGamma, TARGET_WHITEPOINT, TARGET_BLACKPOINT, TARGET_INVERSE_GAMMA, targetGamma);
  }

  output = float4(fixedGamma, 1.f);
}


technique SDR_gamma_fix
{
  pass SDR_gamma_fix
  {
    VertexShader = PostProcessVS;
     PixelShader = SDR_gamma_fix;
  }
}