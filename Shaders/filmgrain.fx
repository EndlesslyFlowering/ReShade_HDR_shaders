//todo: add source

#include "ReShade.fxh"
#include "colorspace.fxh"

uniform float INTENSITY
<
  ui_label    = "grain intensity";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.001f;
> = 0.025f;

uniform int random
<
  source = "random";
  min    = 0;
  max    = 100000;
>;

float permute(float x)
{
    x = (34.f * x + 1.f) * x;
    return frac(x * 1.f / 289.f) * 289.f;
}

float rand(inout float state)
{
    state = permute(state);
    return frac(state * 1.f / 41.f);
}

void filmgrain(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_TARGET)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 m = float3(texcoord, float(random) / 100000.f) + 1.f.xxx;
  float state = permute(permute(m.x) + m.y) + m.z;

  const float a0 = 0.151015505647689;
  const float a1 = -0.5303572634357367;
  const float a2 = 1.365020122861334;
  const float b0 = 0.132089632343748;
  const float b1 = -0.7607324991323768;

  float p = 0.95f * rand(state) + 0.025f;
  float q = p - 0.5f;
  float r = q * q;

  float grain = q * (a2 + (a1 * r + a0) / (r*r + b1*r + b0));
  grain *= 0.255121822830526; // normalize to [-1,1)

  float y;   //luma
  float c_b; //chroma b
  float c_r; //chroma r
  if (BUFFER_COLOR_SPACE == CSP_PQ)
  {
    y   = dot(input, K_BT2020);
    c_b = (input.b - y) / BT2020_KB_helper;
    c_r = (input.r - y) / BT2020_KR_helper;
  }
  else /*if (BUFFER_COLOR_SPACE == CSP_SCRGB)*/
  {
    y   = dot(input, K_BT709);
    c_b = (input.b - y) / BT709_KB_helper;
    c_r = (input.r - y) / BT709_KR_helper;
  }
  y += (INTENSITY * grain);

  float3 rgb;

  if (BUFFER_COLOR_SPACE == CSP_PQ)
  {
    rgb = float3(y + BT2020_KR_helper * c_r,
                 y - BT2020_KG_helper[0] * c_b - BT2020_KG_helper[1] * c_r,
                 y + BT2020_KB_helper * c_b);
  }
  else /*if (BUFFER_COLOR_SPACE == CSP_SCRGB)*/
  {
    rgb = float3(y + BT709_KR_helper * c_r,
                 y - BT709_KG_helper[0] * c_b - BT709_KG_helper[1] * c_r,
                 y + BT709_KB_helper * c_b);
  }

  output = float4(rgb, 1.f);
}

technique filmgrain
{
  pass filmgrain
  {
    VertexShader = PostProcessVS;
    PixelShader  = filmgrain;
  }
}