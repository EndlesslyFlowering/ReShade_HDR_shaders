//todo: add source

#include "ReShade.fxh"
#include "lilium__colour_space.fxh"

uniform float INTENSITY
<
  ui_label = "grain intensity";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 1.f;
  ui_step  = 0.001f;
> = 0.025f;

uniform int RANDOM
<
  source = "random";
  min    = 0;
  max    = 100000;
>;

float permute(float X)
{
    X = (34.f * X + 1.f) * X;
    return frac(X * 1.f / 289.f) * 289.f;
}

float rand(inout float State)
{
    State = permute(State);
    return frac(State * 1.f / 41.f);
}

void filmgrain(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_TARGET)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 m    = float3(texcoord, RANDOM / 100000.f) + 1.f.xxx;
  float state = permute(permute(m.x) + m.y) + m.z;

  const float a0 =  0.151015505647689;
  const float a1 = -0.5303572634357367;
  const float a2 =  1.365020122861334;
  const float b0 =  0.132089632343748;
  const float b1 = -0.7607324991323768;

  float p = 0.95f * rand(state) + 0.025f;
  float q = p - 0.5f;
  float r = q * q;

  float grain = q * (a2 + (a1 * r + a0) / (r*r + b1*r + b0));
  grain *= 0.255121822830526; // normalize to (-1, 1)

  float min_rgb = min(min(input.r, input.g), input.b);
  min_rgb = min_rgb < 0.f
          ? min_rgb * 1.25f
          : min_rgb * 0.75f;
  float max_rgb = max(max(input.r, input.g), input.b) * 1.25f;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  float3 ycbcr = RGB_AP0_D65_To_YCbCr(PQ_Inverse_EOTF(mul(BT709_To_AP0_D65,  input / 125.f)));
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
  float3 ycbcr = RGB_BT2020_To_YCbCr(input);
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
  float3 ycbcr = RGB_AP0_D65_To_YCbCr(PQ_Inverse_EOTF(mul(BT2020_To_AP0_D65, input / 100.f)));
#else //ACTUAL_COLOUR_SPACE == CSP_SRGB
  float3 ycbcr = RGB_BT709_To_YCbCr(input);
#endif

  ycbcr.x += (INTENSITY * grain);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  float3 rgb = mul(AP0_D65_To_BT709,  PQ_EOTF(YCbCr_AP0_D65_To_RGB(ycbcr))) * 125.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
  float3 rgb = YCbCr_BT2020_To_RGB(ycbcr);
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
  float3 rgb = mul(AP0_D65_To_BT2020, PQ_EOTF(YCbCr_AP0_D65_To_RGB(ycbcr))) * 100.f;
#else //ACTUAL_COLOUR_SPACE == CSP_SRGB
  float3 rgb = YCbCr_BT709_To_RGB(ycbcr);
#endif

  //output = float4(rgb, 1.f);
  output = float4(clamp(rgb, min_rgb.xxx, max_rgb.xxx), 1.f);
}

technique lilium__filmgrain
{
  pass filmgrain
  {
    VertexShader = PostProcessVS;
     PixelShader = filmgrain;
  }
}
