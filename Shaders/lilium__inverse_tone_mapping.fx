#include "lilium__include/inverse_tone_mappers.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))

// TODO:
// - implement vertex shader for optimisation
// - add namespace for UI

#include "lilium__include/draw_text_fix.fxh"

//#define ENABLE_DICE

uniform uint ITM_METHOD
<
  ui_category = "global";
  ui_label    = "inverse tone mapping method";
  ui_tooltip  = "BT.2446 Method A:"
           "\n" "  Expands the whole range of the image according to the input and target brightness."
           "\n" "  It is designed for tone mapping 100 nits to 1000 nits originally"
           "\n" "  and therefore can't handle expanding the brightness too much."
           "\n" "BT.2446 Methoc C:"
           "\n" "  Maps the brightness level directly rather than using a curve to do the expansion."
           "\n" "  The input brightness alone dictactes the target brightness."
           "\n" "Dice inverse:"
           "\n" "  Not yet finished..."
           "\n" "map SDR into HDR:"
           "\n" "  Straight up map the SDR image into the HDR container according to the target brightness set.";
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

uniform uint CONTENT_TRC
<
  ui_category = "global";
  ui_label    = "content TRC";
  ui_tooltip  = "TRC = tone reproduction curve"
           "\n" "also wrongly known as \"gamma\"";
  ui_type     = "combo";
  ui_items    = "sRGB\0"
                "2.2\0"
                "2.4\0"
                "linear (scRGB)\0";
> = 1;

#define CONTENT_TRC_SRGB     0
#define CONTENT_TRC_GAMMA_22 1
#define CONTENT_TRC_GAMMA_24 2
#define CONTENT_TRC_LINEAR   3

uniform float TARGET_BRIGHTNESS
<
  ui_category = "global";
  ui_label    = "target brightness";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 1.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 1000.f;

uniform float BT2446A_INPUT_BRIGHTNESS
<
  ui_category = "BT.2446 Method A";
  ui_label    = "input brightness";
  ui_tooltip  = "Sets the brightness to this value for the inverse tone mapping process."
           "\n" "Controls the average brightness."
           "\n" "Can't be higher than \"target brightness\"!";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.1f;
> = 100.f;

uniform float BT2446A_MAX_INPUT_BRIGHTNESS
<
  ui_category = "BT.2446 Method A";
  ui_label    = "max input brightness";
  ui_tooltip  = "Controls how much of the \"overbright\" brightness will be processed."
           "\n" "Analyse a good value with the HDR analysis shader."
           "\n" "Can't be lower than \"input brightness\"."
           "\n" "Can't be higher than \"target brightness\"!";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.1f;
> = 100.f;

//uniform bool BT2446A_AUTO_REF_WHITE
//<
//  ui_category = "BT.2446 Method A";
//  ui_label    = "automatically calculate \"reference white luminance\"";
//> = false;

uniform float BT2446A_GAMUT_EXPANSION
<
  ui_category = "BT.2446 Method A";
  ui_label    = "gamut expansion";
  ui_tooltip  = "1.10 is the default of the spec"
           "\n" "1.05 about matches the input colour space"
           "\n" "1.00 slightly reduces the colour space";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 1.2f;
  ui_step     = 0.005f;
> = 1.1f;

uniform float BT2446A_GAMMA_IN
<
  ui_category = "BT.2446 Method A";
  ui_label    = "gamma adjustment before inverse tone mapping";
  ui_type     = "drag";
  ui_min      = -0.4f;
  ui_max      =  0.6f;
  ui_step     =  0.005f;
> = 0.f;

uniform float BT2446A_GAMMA_OUT
<
  ui_category = "BT.2446 Method A";
     ui_label = "gamma adjustment after inverse tone mapping";
      ui_type = "drag";
       ui_min = -1.f;
       ui_max =  1.f;
      ui_step =  0.005f;
> = 0.f;

uniform float BT2446C_INPUT_BRIGHTNESS
<
  ui_category = "BT.2446 Method C";
  ui_label    = "input brightness";
  ui_tooltip  = "Also controls the output brightness:"
           "\n" "103.2 nits ->  400 nits"
           "\n" "107.1 nits ->  500 nits"
           "\n" "110.1 nits ->  600 nits"
           "\n" "112.6 nits ->  700 nits"
           "\n" "114.8 nits ->  800 nits"
           "\n" "116.7 nits ->  900 nits"
           "\n" "118.4 nits -> 1000 nits";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 1.f;
  ui_max      = 1200.f;
  ui_step     = 0.01f;
> = 100.f;

uniform float BT2446C_ALPHA
<
  ui_category = "BT.2446 Method C";
  ui_label    = "alpha";
  ui_tooltip  = "Better preserves achromatic (= without colour) brightness levels.";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 0.33f;
  ui_step     = 0.001f;
> = 0.33f;

//uniform float BT2446C_K1
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "k1";
//      ui_type = "drag";
//       ui_min = 0.001f;
//       ui_max = 1.f;
//      ui_step = 0.001f;
//> = 0.83802f;
//
//uniform float BT2446C_INFLECTION_POINT
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "inflection point";
//      ui_type = "drag";
//       ui_min = 0.001f;
//       ui_max = 100.f;
//      ui_step = 0.001f;
//> = 58.535046646;

//uniform bool BT2446C_USE_ACHROMATIC_CORRECTION
//<
//  ui_category = "BT.2446 Method C";
//  ui_label    = "use achromatic correction for really bright elements";
//> = false;
//
//uniform float BT2446C_SIGMA
//<
//  ui_category = "BT.2446 Method C";
//     ui_label = "correction factor";
//      ui_type = "drag";
//       ui_min = 0.f;
//       ui_max = 10.f;
//      ui_step = 0.001f;
//> = 0.5f;

#ifdef ENABLE_DICE
uniform float DICE_INPUT_BRIGHTNESS
<
  ui_category = "Dice";
  ui_label    = "input brightness";
  ui_tooltip  = "Can't be higher than \"target brightness\"";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 400.f;
  ui_step     = 0.1f;
> = 100.f;

uniform float DICE_SHOULDER_START
<
  ui_category = "Dice";
  ui_label    = "shoulder start";
  ui_tooltip  = "Set this to where the brightness expansion starts";
  ui_type     = "drag";
  ui_units    = "%%";
  ui_min      = 0.1f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 50.f;
#endif //ENABLE_DICE

uniform float MAP_SDR_INTO_HDR_TARGET_BRIGHTNESS
<
  ui_category = "map SDR into HDR";
  ui_label    = "target brightness";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 1.f;
  ui_max      = 1000.f;
  ui_step     = 0.1f;
> = 203.f;


void PS_InverseToneMapping(
      float4     VPos : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4   Output : SV_Target)
{
  const float3 input = tex2D(ReShade::BackBuffer, TexCoord).rgb;

  float3 hdr;

  switch (CONTENT_TRC)
  {
    default:
    {
      hdr = input;
    }
    break;
    case CONTENT_TRC_SRGB:
    {
      hdr = Csp::Trc::FromExtendedSrgb(input);
    }
    break;
    case CONTENT_TRC_GAMMA_22:
    {
      hdr = Csp::Trc::FromExtendedGamma22(input);
    }
    break;
    case CONTENT_TRC_GAMMA_24:
    {
      hdr = Csp::Trc::FromExtendedGamma24(input);
    }
    break;
  }

  //hdr = gamut(hdr, EXPAND_GAMUT);

#ifdef ENABLE_DICE
  const float diceReferenceWhite = (DICE_INPUT_BRIGHTNESS / 80.f);
#endif //ENABLE_DICE

  if (ITM_METHOD != ITM_METHOD_DICE_INVERSE)
  {
    hdr = Csp::Mat::Bt709To::Bt2020(hdr);
  }
#ifdef ENABLE_DICE
  else
  {
    hdr = clamp(Csp::Mat::Bt709To::Ap0D65(hdr * diceReferenceWhite / 125.f), 0.f, 65504.f);
  }
#endif //ENABLE_DICE

  switch (ITM_METHOD)
  {
    case ITM_METHOD_BT2446A:
    {
      const float inputNitsFactor = BT2446A_MAX_INPUT_BRIGHTNESS > BT2446A_INPUT_BRIGHTNESS
                                  ? BT2446A_MAX_INPUT_BRIGHTNESS / BT2446A_INPUT_BRIGHTNESS
                                  : 1.f;

      float referenceWhiteNits = BT2446A_INPUT_BRIGHTNESS * inputNitsFactor;
            referenceWhiteNits = referenceWhiteNits < TARGET_BRIGHTNESS
                               ? referenceWhiteNits
                               : TARGET_BRIGHTNESS;

      hdr = InverseToneMapping::Bt2446a(hdr,
                                        TARGET_BRIGHTNESS,
                                        referenceWhiteNits,
                                        inputNitsFactor,
                                        BT2446A_GAMUT_EXPANSION,
                                        BT2446A_GAMMA_IN,
                                        BT2446A_GAMMA_OUT);
    }
    break;
    case ITM_METHOD_BT2446C:
    {
      hdr = InverseToneMapping::Bt2446c(hdr,
                                        BT2446C_INPUT_BRIGHTNESS > 153.9f
                                      ? 1.539f
                                      : BT2446C_INPUT_BRIGHTNESS / 100.f,
                                        0.33f - BT2446C_ALPHA);
                                        //BT2446C_USE_ACHROMATIC_CORRECTION,
                                        //BT2446C_SIGMA);
    }
    break;

#ifdef ENABLE_DICE

    case ITM_METHOD_DICE_INVERSE:
    {
      const float target_CLL_normalised = TARGET_BRIGHTNESS / 10000.f;
      hdr = InverseToneMapping::Dice::InverseToneMapper(
              hdr,
              Csp::Trc::ToPqFromNits(DICE_INPUT_BRIGHTNESS),
              Csp::Trc::ToPqFromNits(DICE_SHOULDER_START / 100.f * DICE_INPUT_BRIGHTNESS));
    }
    break;

#endif //ENABLE_DICE

    case ITM_METHOD_MAP_SDR_INTO_HDR:
    {
      hdr = InverseToneMapping::MapSdrIntoHdr(hdr,
                                              MAP_SDR_INTO_HDR_TARGET_BRIGHTNESS);
    }
    break;
  }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

#ifdef ENABLE_DICE

  if (ITM_METHOD == ITM_METHOD_DICE_INVERSE)
  {
    hdr = Csp::Mat::Ap0D65To::Bt2020(hdr);
  }

#endif //ENABLE_DICE

  hdr = Csp::Trc::ToPq(hdr);

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  if (ITM_METHOD != ITM_METHOD_DICE_INVERSE)
  {
    hdr = Csp::Mat::Bt2020To::Bt709(hdr);
  }

#ifdef ENABLE_DICE

  else
  {
    hdr = Csp::Mat::Ap0D65To::Bt709(hdr);
  }

#endif //ENABLE_DICE

  hdr *= 125.f; // 125 = 10000 / 80

#else //ACTUAL_COLOUR_SPACE ==

  hdr = float3(0.f, 0.f, 0.f);

#endif //ACTUAL_COLOUR_SPACE ==

  Output = float4(hdr, 1.f);
}

technique lilium__inverse_tone_mapping
<
  ui_label = "Lilium's inverse tone mapping";
>
{
  pass PS_InverseToneMapping
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_InverseToneMapping;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__inverse_tone_mapping
<
  ui_label = "Lilium's inverse tone mapping (ERROR)";
>
CS_ERROR

#endif //is hdr API and hdr colour space
