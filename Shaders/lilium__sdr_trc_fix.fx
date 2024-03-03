#include "lilium__include/colour_space.fxh"

//#define _DEBUG

// TODO:
// - implement vertex shader for optimisation ?
// - add namespace for UI

uniform uint INPUT_TRC
<
  ui_label = "input Gamma";
  ui_type  = "combo";
  ui_items = "power Gamma\0"
             "sRGB\0";
> = 0;

uniform uint TARGET_TRC
<
  ui_label = "target Gamma";
  ui_type  = "combo";
  ui_items = "sRGB\0"
             "power Gamma\0";
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
  const float3 LinearColour)
{
  float3 V = pow(LinearColour, 1.f / 2.4f);

  static const float powLw = pow(BT1886_TARGET_WHITEPOINT, 1.f / 2.4f);
  static const float powLb = pow(BT1886_TARGET_BLACKPOINT, 1.f / 2.4f);

  static const float powLw_minus_powLb = powLw - powLb;

  static const float a = pow(powLw_minus_powLb, 2.4f);
  static const float b = powLb
                       / powLw_minus_powLb;

  float3 L = a * pow(max(V + b, 0.f), 2.4f);

  //normalise so that 0 is the black point and 1 the white point of the display
  return (L - BT1886_TARGET_BLACKPOINT) / (BT1886_TARGET_WHITEPOINT - BT1886_TARGET_BLACKPOINT);
}


void PS_SdrTrcFix(
  in  float4 Position : SV_Position,
  out float4 Output   : SV_Target0)
{
  const float4 input = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  float3 fixedGamma = input.rgb;

  if (INPUT_TRC == 0)
  {
    fixedGamma = pow(fixedGamma, INPUT_POWER_GAMMA);
  }
  else
  {
    fixedGamma = Csp::Trc::SrgbTo::Linear(fixedGamma);
  }

  if (USE_BT1886)
  {
    fixedGamma = Bt1886(fixedGamma);
  }

  if (TARGET_TRC == 0)
  {
    fixedGamma = Csp::Trc::LinearTo::Srgb(fixedGamma);
  }
  else
  {
    fixedGamma = pow(fixedGamma, 1.f / TARGET_POWER_GAMMA);
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
