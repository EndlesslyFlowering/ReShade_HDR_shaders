#include "ReShade.fxh"
#include "inverse_tone_mappers.fxh"


uniform uint INVERSE_TONEMAPPING_METHOD
<
  ui_label = "inverse tone mapping method";
  ui_type  = "combo";
  ui_items = "BT.2446 Method A\0"
             "map SDR into HDR\0"
             "BT.2446 Methoc C (itm)\0";
> = 0;

uniform float TARGET_PEAK_NITS
<
  ui_label = "target peak luminance (nits)";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 10000.f;
  ui_step  = 10.f;
> = 1000.f;

uniform float REF_WHITE_NITS
<
  ui_label   = "reference white luminance (nits)";
  ui_tooltip = "can't be higher than \"target peak luminance\"";
  ui_type    = "drag";
  ui_min     = 0.f;
  ui_max     = 1200.f;
  ui_step    = 0.1f;
> = 100.f;

uniform bool AUTO_REF_WHITE
<
  ui_label   = "automatically calculate \"reference white luminance\"";
  ui_tooltip = "only for inverse tone mapping!";
> = false;

uniform bool USE_TARGET_PEAK_FOR_SDR_HDR_MAPPING
<
  ui_label   = "use \"target peak luminance\" as white point for SDR->HDR mapping";
  ui_tooltip = "100% white will be at the nits of \"target peak luminance\"";
> = false;

uniform uint EXPAND_GAMUT
<
  ui_label   = "Vivid HDR";
  ui_type    = "combo";
  ui_items   = "no\0my expanded colourspace\0expand colourspace\0brighter highlights\0";
  ui_tooltip = "interesting gamut expansion things from Microsoft\n"
               "and me ;)\n"
               "makes things look more colourful";
> = 0;

//uniform uint CONTENT_GAMMA
//<
//	ui_label = "content gamma";
//	ui_type  = "combo";
//	ui_items = " 2.2\0 2.4\0 sRGB\0";
//> = 1;

uniform float GAMMA_IN
<
  ui_category = "BT.2446 Method A";
  ui_label    = "gamma in";
  ui_type     = "drag";
  ui_min      = -0.4f;
  ui_max      =  0.6f;
  ui_step     =  0.005f;
> = 0.f;

uniform float GAMMA_OUT
<
  ui_category = "BT.2446 Method A";
     ui_label = "gamma out";
      ui_type = "drag";
       ui_min = -1.f;
       ui_max =  1.f;
      ui_step =  0.005f;
> = 0.f;

uniform float ALPHA
<
  ui_category = "BT.2446 Method C";
     ui_label = "alpha";
      ui_type = "drag";
       ui_min = 0.f;
       ui_max = 1.5f;
      ui_step = 0.001f;
> = 0.f;

//uniform float K1
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "k1";
//      ui_type = "drag";
//       ui_min = 0.001f;
//       ui_max = 1.f;
//      ui_step = 0.001f;
//> = 0.83802f;
//
//uniform float INFLECTION_POINT
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "inflection point";
//      ui_type = "drag";
//       ui_min = 0.001f;
//       ui_max = 100.f;
//      ui_step = 0.001f;
//> = 58.535046646;


void BT2446_itm(
      float4     vpos : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4   output : SV_Target)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 hdr;

  //hdr = sRGB_inverse_EOTF(input);
  hdr = gamut(input, EXPAND_GAMUT);

  switch (INVERSE_TONEMAPPING_METHOD)
  {
    case 0:
    {
      hdr = BT2446A_inverseToneMapping(
        hdr,
        TARGET_PEAK_NITS,
        !AUTO_REF_WHITE ? REF_WHITE_NITS : 1.19691 * pow(TARGET_PEAK_NITS, 0.745534) - 3.10051,
        GAMMA_IN,
        GAMMA_OUT);
    }
    break;
    case 1:
    {
      hdr = mapSDRintoHDR(
        hdr,
        TARGET_PEAK_NITS,
        USE_TARGET_PEAK_FOR_SDR_HDR_MAPPING);
    }
    break;
    case 2:
    {
      hdr = BT2446C_inverseToneMapping(
        hdr,
        REF_WHITE_NITS,
        0.33f - ALPHA);
    }
    break;
  }

  if(CSP_PQ)
    hdr = PQ_inverse_EOTF(hdr);
  else if(CSP_SCRGB)
  {
    hdr = mul(bt2020_to_bt709_matrix, hdr);
    hdr = hdr * 10000.f / 80.f;
  }
  else
    hdr = float3(0.f, 0.f, 0.f);

  output = float4(hdr, 1.f);
}

technique inverse_tone_mapping
{
  pass inverse_tone_mapping
  {
    VertexShader = PostProcessVS;
     PixelShader = BT2446_itm;
  }
}