// gaussian film grain by haasn (https://github.com/haasn/gentoo-conf/blob/xor/home/nand/.mpv/shaders/filmgrain.glsl)

#include "lilium__include/colour_space.fxh"

// TODO:
// - implement vertex shader for optimisation
// - add namespace for UI

uniform float INTENSITY
<
  ui_label = "grain INTENSITY";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 0.1f;
  ui_step  = 0.0001f;
> = 0.025f;

uniform int RANDOM
<
  source = "random";
  min    = 0;
  max    = 100000;
>;

float Permute(float X)
{
    X = (34.f * X + 1.f) * X;
    return frac(X * 1.f / 289.f) * 289.f;
}

float Rand(inout float State)
{
    State = Permute(State);
    return frac(State * 1.f / 41.f);
}

static const float a0 =  0.151015505647689;
static const float a1 = -0.5303572634357367;
static const float a2 =  1.365020122861334;
static const float b0 =  0.132089632343748;
static const float b1 = -0.7607324991323768;

void PS_Filmgrain(
      float4 Vpos     : SV_Position,
      float2 Texcoord : TEXCOORD,
  out float4 Output   : SV_TARGET)
{
  float3 input = tex2D(ReShade::BackBuffer, Texcoord).rgb;

  float3 m     = float3(Texcoord, RANDOM / 100000.f) + 1.f;
  float  state = Permute(Permute(m.x) + m.y) + m.z;

  float p = 0.95f * Rand(state) + 0.025f;
  float q = p - 0.5f;
  float r = q * q;

  float Grain = q * (a2 + (a1 * r + a0) / (r*r + b1*r + b0));
  Grain *= 0.255121822830526; // normalize to (-1, 1)

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  float3 Ycbcr = Csp::Ycbcr::FromRgb::Ap0D65(Csp::Trc::ToPq(Csp::Mat::Bt709To::Ap0D65(input / 125.f)));

#elif defined(IS_HDR10_LIKE_CSP)

  float3 Ycbcr = Csp::Ycbcr::FromRgb::Bt2020(input);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  float3 Ycbcr = Csp::Ycbcr::FromRgb::Ap0D65(Csp::Trc::ToPq(Csp::Mat::Bt2020To::Ap0D65(input / 100.f)));

#else //ACTUAL_COLOUR_SPACE == CSP_SRGB

  float3 Ycbcr = Csp::Ycbcr::FromRgb::Bt709(input);

#endif //ACTUAL_COLOUR_SPACE ==

//  float maxMul = min(1.25f + INTENSITY, 2.f);
//  float minMul = 2.01f - maxMul;
//
//  float maxY = Ycbcr.x * maxMul;
//  float minY = Ycbcr.x * minMul;

  Ycbcr.x += (INTENSITY * Grain);

  //Ycbcr.x = clamp(Ycbcr.x, minY, maxY);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  float3 Rgb = Csp::Mat::Ap0D65To::Bt709(Csp::Trc::FromPq(Csp::Ycbcr::ToRgb::Ap0D65(Ycbcr))) * 125.f;

#elif defined(IS_HDR10_LIKE_CSP)

  float3 Rgb = Csp::Ycbcr::ToRgb::Bt2020(Ycbcr);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  float3 Rgb = Csp::Mat::Ap0D65To::Bt2020(Csp::Trc::FromPq(Csp::Ycbcr::ToRgb::Ap0D65(Ycbcr))) * 100.f;

#else //ACTUAL_COLOUR_SPACE == CSP_SRGB

  float3 Rgb = Csp::Ycbcr::ToRgb::Bt709(Ycbcr);

#endif //ACTUAL_COLOUR_SPACE ==

  Output = float4(Rgb, 1.f);
}

technique lilium__filmgrain
<
  ui_label = "Lilium's filmgrain";
>
{
  pass PS_Filmgrain
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_Filmgrain;
  }
}
