#include "lilium__include/colour_space.fxh"

//#define _DEBUG

// TODO:
// - implement vertex shader for optimisation ?
// - add namespace for UI

uniform uint INPUT_TRC
<
  ui_label = "input Gamma";
  ui_type  = "combo";
  ui_items = "sRGB\0"
             "power Gamma\0";
> = 0;

uniform uint TARGET_TRC
<
  ui_label = "target Gamma";
  ui_type  = "combo";
  ui_items = "power Gamma\0"
             "sRGB\0";
> = 0;

uniform float INPUT_POWER_GAMMA
<
  ui_label = "input power Gamma";
  ui_type  = "drag";
  ui_min   = 1.f;
  ui_max   = 3.f;
  ui_step  = 0.01f;
> = 2.2f;

uniform float TARGET_POWER_GAMMA
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
  ui_label    = "use BT.1886 for blackpoint correction";
> = false;

uniform float BT1886_TARGET_WHITEPOINT
<
  ui_category = "BT.1886";
  ui_label    = "target whitepoint";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1000.f;
  ui_step     = 0.5f;
> = 100.f;

uniform float BT1886_TARGET_BLACKPOINT
<
  ui_category = "BT.1886";
  ui_label    = "target blackpoint";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.0001f;
> = 0.f;


float3 Bt1886(
  const float3 V,
  const float  TargetInverseGamma)
{
  static const float powLw = pow(BT1886_TARGET_WHITEPOINT, TargetInverseGamma);
  static const float powLb = pow(BT1886_TARGET_BLACKPOINT, TargetInverseGamma);

  static const float powLw_minus_powLb = powLw - powLb;

  static const float a = pow(powLw_minus_powLb, TARGET_POWER_GAMMA);
  static const float b = powLb
                       / powLw_minus_powLb;

  float3 L = a * pow(max(V + b, 0.f), TARGET_POWER_GAMMA);

  return pow(L / BT1886_TARGET_WHITEPOINT, TargetInverseGamma);
}


float3 Bt1886Srgb(
  const float3 V)
{
  static const float powLw = Csp::Trc::LinearTo::Srgb(BT1886_TARGET_WHITEPOINT);
  static const float powLb = Csp::Trc::LinearTo::Srgb(BT1886_TARGET_BLACKPOINT);

  static const float powLw_minus_powLb = powLw - powLb;

  static const float a = Csp::Trc::SrgbTo::Linear(powLw_minus_powLb);
  static const float b = powLb
                       / powLw_minus_powLb;

  float3 L = a * Csp::Trc::SrgbTo::Linear(max(V + b, 0.f));

  return Csp::Trc::LinearTo::Srgb(L / BT1886_TARGET_WHITEPOINT);
}


void PS_SdrTrcFix(
  in  float4 VPos   : SV_Position,
  out float4 Output : SV_Target0)
{
  const float4 input = tex2Dfetch(ReShade::BackBuffer, int2(VPos.xy));

  float3 fixedGamma = input.rgb;

  if (INPUT_TRC == 0)
  {
    fixedGamma = Csp::Trc::SrgbTo::Linear(fixedGamma);
  }
  else
  {
    fixedGamma = pow(fixedGamma, INPUT_POWER_GAMMA);
  }

  if (TARGET_TRC == 0)
  {
    float targetInverseGamma = 1.f / TARGET_POWER_GAMMA;

    fixedGamma = pow(fixedGamma, targetInverseGamma);

    if (USE_BT1886)
    {
      fixedGamma = Bt1886(fixedGamma, targetInverseGamma);
    }
  }
  else
  {
    fixedGamma = Csp::Trc::LinearTo::Srgb(fixedGamma);

    if (USE_BT1886)
    {
      fixedGamma = Bt1886Srgb(fixedGamma);
    }
  }

  Output = float4(fixedGamma, input.a);
}


technique lilium__sdr_trc_fix
<
  ui_label = "Lilium's SDR TRC fix";
>
{
  pass PS_SdrTrcFix
  {
    VertexShader = VS_PostProcessWithoutTexCoord;
     PixelShader = PS_SdrTrcFix;
  }
}
