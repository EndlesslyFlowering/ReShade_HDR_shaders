#include "lilium__include/colour_space.fxh"

//#define _DEBUG

// TODO:
// - implement vertex shader for optimisation ?
// - add namespace for UI

uniform uint INPUT_TRC
<
  ui_label = "input TRC";
  ui_type  = "combo";
  ui_items = "sRGB\0"
             "power Gamma\0";
> = 0;

uniform uint TARGET_TRC
<
  ui_label = "target TRC";
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
  ui_label    = "use BT.1886 gamma for blackpoint correction";
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
  ui_step     = 0.001f;
> = 0.f;


float3 Bt1886(
  float3 V,
  float  TargetInverseGamma)
{
  float powLw = pow(BT1886_TARGET_WHITEPOINT, TargetInverseGamma);
  float powLb = pow(BT1886_TARGET_BLACKPOINT, TargetInverseGamma);

  float powLw_minus_powLb = powLw - powLb;

  float a = pow(powLw_minus_powLb, TARGET_POWER_GAMMA);
  float b = powLb /
            (powLw_minus_powLb);

  float3 L = a * pow(max(V + b.xxx, 0.f.xxx), TARGET_POWER_GAMMA);

  return pow(L / BT1886_TARGET_WHITEPOINT, TargetInverseGamma);
}


void PS_SdrTrcFix(
  in  float4 VPos     : SV_Position,
  in  float2 TexCoord : TEXCOORD0,
  out float4 Output   : SV_Target0)
{
  float3 fixedGamma = tex2D(ReShade::BackBuffer, TexCoord).rgb;

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
    Csp::Trc::LinearTo::Srgb(fixedGamma);
  }

  Output = float4(fixedGamma, 1.f);
}


technique lilium__sdr_trc_fix
<
  ui_label = "Lilium's SDR TRC fix";
>
{
  pass PS_SdrTrcFix
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_SdrTrcFix;
  }
}
