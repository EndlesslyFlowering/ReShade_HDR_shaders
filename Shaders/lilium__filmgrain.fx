//todo: add source

#include "ReShade.fxh"
#include "lilium__colorspace.fxh"

uniform float INTENSITY
<
  ui_label = "grain intensity";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 1.f;
  ui_step  = 0.001f;
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

  float3 m = float3(texcoord, random / 100000.f) + 1.f.xxx;
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
  grain *= 0.255121822830526; // normalize to [-1,1)

  float min_rgb = min(min(input.r, input.g), input.b);
  min_rgb = min_rgb < 0.f
          ? min_rgb * 1.25f
          : min_rgb * 0.75f;
  float max_rgb = max(max(input.r, input.g), input.b) * 1.25f;

  float y;   //luma
  float c_b; //chroma b
  float c_r; //chroma r
  if (BUFFER_COLOR_SPACE == CSP_PQ)
  {
    y   = dot(input, K_BT2020);
    c_b = (input.b - y) / KB_BT2020_helper;
    c_r = (input.r - y) / KR_BT2020_helper;
  }
  else /*if (BUFFER_COLOR_SPACE == CSP_SCRGB)*/
  {
    y   = dot(input, K_BT709);
    c_b = (input.b - y) / KB_BT709_helper;
    c_r = (input.r - y) / KR_BT709_helper;
  }
  y += (INTENSITY * grain);

  float3 rgb;

  if (BUFFER_COLOR_SPACE == CSP_PQ)
  {
    rgb = float3(y + KR_BT2020_helper * c_r,
                 y - KG_BT2020_helper[0] * c_b - KG_BT2020_helper[1] * c_r,
                 y + KB_BT2020_helper * c_b);
  }
  else /*if (BUFFER_COLOR_SPACE == CSP_SCRGB)*/
  {
    rgb = float3(y + KR_BT709_helper * c_r,
                 y - KG_BT709_helper[0] * c_b - KG_BT709_helper[1] * c_r,
                 y + KB_BT709_helper * c_b);
  }

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
