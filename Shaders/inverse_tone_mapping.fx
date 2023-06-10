#include "ReShade.fxh"
#include "inverse_tone_mappers.fxh"
#include "DrawText_fix.fxh"


uniform uint INVERSE_TONE_MAPPING_METHOD
<
  ui_category = "global";
  ui_label    = "inverse tone mapping method";
  ui_type     = "combo";
  ui_items    = "BT.2446 Method A\0"
                "BT.2446 Methoc C\0"
                "Dice inverse\0"
                "map SDR into HDR\0";
> = 0;

#define ITM_METHOD_BT2446A          0
#define ITM_METHOD_BT2446C          1
#define ITM_METHOD_DICE_INVERSE     2
#define ITM_METHOD_MAP_SDR_INTO_HDR 3

//uniform uint EXPAND_GAMUT
//<
//  ui_label   = "Vivid HDR";
//  ui_type    = "combo";
//  ui_items   = "no\0"
//               "my expanded colourspace\0"
//               "expand colourspace\0"
//               "brighter highlights\0";
//  ui_tooltip = "interesting gamut expansion things from Microsoft\n"
//               "and me ;)\n"
//               "makes things look more colourful";
//> = 0;

uniform uint CONTENT_GAMMA
<
  ui_category = "global";
  ui_label    = "content gamma";
  ui_type     = "combo";
  ui_items    = "sRGB\0"
                "2.2\0"
                "2.4\0"
                "linear (scRGB)\0";
> = 1;

#define CONTENT_GAMMA_SRGB   0
#define CONTENT_GAMMA_22     1
#define CONTENT_GAMMA_24     2
#define CONTENT_GAMMA_LINEAR 3

uniform float TARGET_PEAK_NITS
<
  ui_category = "global";
  ui_label    = "target peak luminance (nits)";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 1000.f;

uniform float REF_WHITE_NITS_BT2446A
<
  ui_category = "BT.2446 Method A";
  ui_label    = "reference white luminance (nits)";
  ui_tooltip  = "can't be higher than \"target peak luminance\"";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.1f;
> = 100.f;

uniform float MAX_INPUT_NITS_BT2446A
<
  ui_category = "BT.2446 Method A";
  ui_label    = "max input luminance (nits)";
  ui_tooltip  = "can't be lower than \"reference white luminance\"\n"
                "can't be higher than \"target peak luminance\"";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.1f;
> = 100.f;

//uniform bool AUTO_REF_WHITE
//<
//  ui_category = "BT.2446 Method A";
//  ui_label    = "automatically calculate \"reference white luminance\"";
//> = false;

uniform float GAMUT_EXPANSION_BT2446A
<
  ui_category = "BT.2446 Method A";
  ui_label    = "gamut expansion";
  ui_tooltip  = "1.10 is the default of the spec\n"
                "1.05 about matches the input color space\n"
                "1.00 slightly reduces the color space";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1.2f;
  ui_step     = 0.005f;
> = 1.1f;

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

uniform float REF_WHITE_NITS_BT2446C
<
  ui_category = "BT.2446 Method C";
  ui_label    = "reference white luminance (nits)";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.01f;
> = 100.f;

uniform float ALPHA
<
  ui_category = "BT.2446 Method C";
     ui_label = "alpha";
      ui_type = "drag";
       ui_min = 0.f;
       ui_max = 0.33f;
      ui_step = 0.001f;
> = 0.33f;

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

//uniform bool USE_ACHROMATIC_CORRECTION
//<
//  ui_category = "BT.2446 Method C";
//  ui_label    = "use achromatic correction for really bright elements";
//> = false;
//
//uniform float SIGMA
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "correction factor";
//      ui_type = "drag";
//       ui_min = 0.f;
//       ui_max = 10.f;
//      ui_step = 0.001f;
//> = 0.5f;

uniform float TARGET_PEAK_NITS_MAP_SDR_INTO_HDR
<
  ui_category = "map SDR into HDR";
  ui_label    = "target peak luminance (nits)";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1000.f;
  ui_step     = 0.1f;
> = 203.f;

uniform float DICE_REFERENCE_WHITE
<
  ui_category = "Dice";
  ui_label    = "reference white luminance (nits)";
  ui_tooltip  = "can't be higher than \"target peak luminance\"";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 400.f;
  ui_step     = 0.1f;
> = 100.f;

uniform float DICE_SHOULDER_START
<
  ui_category = "Dice";
  ui_label    = "shoulder start (in %)";
  ui_tooltip  = "set this to where the luminance expansion starts";
  ui_type     = "drag";
  ui_min      = 0.1f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 50.f;


void BT2446_itm(
      float4     vpos : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4   output : SV_Target)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3 hdr;

  switch (CONTENT_GAMMA)
  {
    default:
    {
      hdr = input;
    }
    break;
    case CONTENT_GAMMA_SRGB:
    {
      hdr = extended_sRGB_EOTF(input);
    }
    break;
    case CONTENT_GAMMA_22:
    {
      hdr = extended_22_EOTF(input);
    }
    break;
    case CONTENT_GAMMA_24:
    {
      hdr = extended_24_EOTF(input);
    }
    break;
  }

  //hdr = gamut(hdr, EXPAND_GAMUT);

  const float dice_reference_white = (DICE_REFERENCE_WHITE / 80.f);

  if (INVERSE_TONE_MAPPING_METHOD != ITM_METHOD_DICE_INVERSE)
  {
    hdr = mul(BT709_to_BT2020, hdr);
  }
  else
  {
    hdr = clamp(mul(BT709_to_AP0_D65, hdr * dice_reference_white / 125.f), 0.f, 65504.f);
  }

  switch (INVERSE_TONE_MAPPING_METHOD)
  {
    case ITM_METHOD_BT2446A:
    {
      const float input_nits_factor = MAX_INPUT_NITS_BT2446A > REF_WHITE_NITS_BT2446A
                                    ? MAX_INPUT_NITS_BT2446A / REF_WHITE_NITS_BT2446A
                                    : 1.f;

      float reference_white_nits = REF_WHITE_NITS_BT2446A * input_nits_factor;
            reference_white_nits = reference_white_nits < TARGET_PEAK_NITS
                                 ? reference_white_nits
                                 : TARGET_PEAK_NITS;

      hdr = BT2446A_inverse_tone_mapping(hdr,
                                         TARGET_PEAK_NITS,
                                         reference_white_nits,
                                         input_nits_factor,
                                         GAMUT_EXPANSION_BT2446A,
                                         GAMMA_IN,
                                         GAMMA_OUT);
    }
    break;
    case ITM_METHOD_BT2446C:
    {
      hdr = BT2446C_inverse_tone_mapping(hdr,
                                         REF_WHITE_NITS_BT2446C > 153.9f
                                       ? 1.539f
                                       : REF_WHITE_NITS_BT2446C / 100.f,
                                         0.33f - ALPHA);
                                         //USE_ACHROMATIC_CORRECTION,
                                         //SIGMA);
    }
    break;
    case ITM_METHOD_DICE_INVERSE:
    {
      const float target_CLL_normalised = TARGET_PEAK_NITS / 10000.f;
      hdr = dice_inverse_tone_mapper(hdr,
                                     nits_to_I(DICE_REFERENCE_WHITE),
                                     nits_to_I(DICE_SHOULDER_START / 100.f * DICE_REFERENCE_WHITE));
    }
    break;
    case ITM_METHOD_MAP_SDR_INTO_HDR:
    {
      hdr = mapSDRintoHDR(hdr,
                          TARGET_PEAK_NITS_MAP_SDR_INTO_HDR);
    }
    break;
  }

  if(BUFFER_COLOR_SPACE == CSP_PQ)
  {
    if (INVERSE_TONE_MAPPING_METHOD == ITM_METHOD_DICE_INVERSE)
    {
      hdr = mul(AP0_D65_to_BT2020, hdr);
    }
    hdr = PQ_inverse_EOTF(hdr);
  }
  else if(BUFFER_COLOR_SPACE == CSP_SCRGB)
  {
    if (INVERSE_TONE_MAPPING_METHOD != ITM_METHOD_DICE_INVERSE)
    {
      hdr = mul(BT2020_to_BT709, hdr);
    }
    else
    {
      hdr = mul(AP0_D65_to_BT709, hdr);
    }
    hdr = hdr * 125.f; // 125 = 10000 / 80
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
