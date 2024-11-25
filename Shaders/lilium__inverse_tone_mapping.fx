#include "lilium__include/inverse_tone_mappers.fxh"


#if (defined(IS_ANALYSIS_CAPABLE_API) \
  && defined(IS_HDR_CSP))


//#include "lilium__include/draw_text_fix.fxh"

//#define ENABLE_DICE

namespace Ui
{
  namespace Itm
  {
    namespace Global
    {
      uniform uint ItmMethod
      <
        ui_category = "global";
        ui_label    = "inverse tone mapping method";
        ui_tooltip  = "BT.2446 Method A:"
                 "\n" "  Expands the whole range of the image according to the input and target brightness."
                 "\n" "  It is designed for inverse tone mapping 100 nits to 1000 nits originally"
                 "\n" "  and therefore can't handle expanding the brightness too much."
                 "\n" "BT.2446 Methoc C:"
                 "\n" "  Maps the brightness level directly rather than using a curve to do the expansion."
                 "\n" "  The input brightness alone dictactes the target brightness."
#ifdef ENABLE_DICE
                 "\n" "Dice inverse:"
                 "\n" "  Not yet finished..."
#endif
                 ;
        ui_type     = "combo";
        ui_items    = "BT.2446 Method A\0"
                      "BT.2446 Methoc C\0"
#ifdef ENABLE_DICE
                      "Dice inverse\0"
#endif
                      ;
      > = 0;

#define ITM_METHOD_BT2446A          0
#define ITM_METHOD_BT2446C          1
#define ITM_METHOD_DICE_INVERSE     2

      uniform uint InputTrc
      <
        ui_category = "global";
        ui_label    = "input gamma";
        ui_tooltip  = "\"linear with SDR black floor emulation (scRGB)\" fixes the sRGB<->gamma 2.2 mismatch";
        ui_type     = "combo";
        ui_items    = "2.2\0"
                      "2.4\0"
                      "linear (scRGB)\0"
                      "linear with SDR black floor emulation (scRGB)\0"
                      "sRGB\0";
      > = 0;

#define CONTENT_TRC_GAMMA_22                    0
#define CONTENT_TRC_GAMMA_24                    1
#define CONTENT_TRC_LINEAR                      2
#define CONTENT_TRC_LINEAR_WITH_BLACK_FLOOR_EMU 3
#define CONTENT_TRC_SRGB                        4

      uniform uint OverbrightHandling
      <
        ui_label    = "overbright bits handling";
        ui_tooltip  = "- filmic roll off uses the inverse of the gamma function to create a smooth roll off"
                 "\n" "- linear takes the input value as is without applying any modifications"
                 "\n" "- apply gamma applies the gamma normally, which leads to an exponential increase of the brightness that may be undesireable"
                 "\n" "- clamp clamps the overbright bits away (mostly for testing)";
        ui_type     = "combo";
        ui_items    = "filmic roll off (S-curve)\0"
                      "linear\0"
                      "apply gamma\0"
                      "clamp\0";
      > = 0;

#define OVERBRIGHT_HANDLING_S_CURVE     0
#define OVERBRIGHT_HANDLING_LINEAR      1
#define OVERBRIGHT_HANDLING_APPLY_GAMMA 2
#define OVERBRIGHT_HANDLING_CLAMP       3

      uniform float TargetBrightness
      <
        ui_category = "global";
        ui_label    = "target brightness";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 1.f;
        ui_max      = 10000.f;
        ui_step     = 10.f;
      > = 600.f;
    }

    namespace Bt2446A
    {
      uniform uint Bt2446AProcessingMode
      <
        ui_category = "BT.2446 Method A";
        ui_label    = "processing mode";
        ui_tooltip  = "YCbCr-like: imitates the look of the original YCbCr processing perfectly"
                 "\n" "luminance:  scale RGB according to brightness";
        ui_type     = "combo";
        ui_items    = "luminance (looks more natural)\0"
                      "YCbCr-like (looks like the original)\0";
      > = 0;

      uniform float Bt2446AInputBrightness
      <
        ui_category = "BT.2446 Method A";
        ui_label    = "input white point";
        ui_tooltip  = "Sets the brightness to this value for the inverse tone mapping process."
                 "\n" "Controls the average brightness."
                 "\n"
                 "\n" "If you want to change just the average brightness,"
                 "\n" "adjust both \"input white point\" and \"max input brightness\""
                 "\n" "to the same value."
                 "\n"
                 "\n" "If higher than \"max input brightness\" then this is the \"max input brightness\"!"
                 "\n" "Can't be higher than \"target brightness\"!";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 1.f;
        ui_max      = 1200.f;
        ui_step     = 0.1f;
      > = 100.f;

      uniform float Bt2446AMaxInputBrightness
      <
        ui_category = "BT.2446 Method A";
        ui_label    = "max input brightness";
        ui_tooltip  = "Controls how much of the \"overbright\" brightness will be processed"
                 "\n" "in the valid range of the inverse tone mapping process."
                 "\n" "If overbright values are above this value they will grow exponentially fast!"
                 "\n" "Analyse a good value with the \"map SDR into HDR\" and \"HDR analysis\" shader"
                 "\n" "before applying the inverse tone mapping shader."
                 "\n"
                 "\n" "If you want to change just the average brightness,"
                 "\n" "adjust both \"input white point\" and \"max input brightness\""
                 "\n" "to the same value."
                 "\n"
                 "\n" "If lower than \"input white point\" then \"input white point\" is the \"max input brightness\"!"
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

      uniform float GammaIn
      <
        ui_category = "BT.2446 Method A";
        ui_label    = "gamma adjustment before inverse tone mapping";
        ui_type     = "drag";
        ui_min      = -0.4f;
        ui_max      =  0.6f;
        ui_step     =  0.005f;
      > = 0.f;

      uniform float GammaOut
      <
        ui_category = "BT.2446 Method A";
           ui_label = "gamma adjustment after inverse tone mapping";
            ui_type = "drag";
             ui_min = -1.f;
             ui_max =  1.f;
            ui_step =  0.005f;
      > = 0.f;
    }

    namespace Bt2446C
    {
      uniform float Bt2446CInputBrightness
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

      uniform float Alpha
      <
        ui_category = "BT.2446 Method C";
        ui_label    = "alpha";
        ui_tooltip  = "Better preserves achromatic (= without colour) brightness levels.";
        ui_type     = "drag";
        ui_min      = 0.f;
        ui_max      = 0.33f;
        ui_step     = 0.001f;
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
      //uniform float InflectionPoint
      //<
      //  ui_category = "BT.2446 Method C";
      //     ui_label = "inflection point";
      //      ui_type = "drag";
      //       ui_min = 0.001f;
      //       ui_max = 100.f;
      //      ui_step = 0.001f;
      //> = 58.535046646;

      //uniform bool AchromaticCorrection
      //<
      //  ui_category = "BT.2446 Method C";
      //  ui_label    = "use achromatic correction for really bright elements";
      //> = false;
      //
      //uniform float Sigma
      //<
      //  ui_category = "BT.2446 Method C";
      //     ui_label = "correction factor";
      //      ui_type = "drag";
      //       ui_min = 0.f;
      //       ui_max = 10.f;
      //      ui_step = 0.001f;
      //> = 0.5f;
    }

#ifdef ENABLE_DICE
    namespace Dice
    {
      uniform float DiceInputBrightness
      <
        ui_category = "Dice";
        ui_label    = "input brightness";
        ui_tooltip  = "Can't be higher than \"target brightness\"";
        ui_type     = "drag";
        ui_min      = 1.f;
        ui_max      = 400.f;
        ui_step     = 0.1f;
      > = 100.f;

      uniform float ShoulderStart
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
    }
#endif //ENABLE_DICE
  }
}


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


void PS_InverseToneMapping(
      float4 Position : SV_Position,
  out float4 Output   : SV_Target0)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  float3 colour = inputColour.rgb;

  switch (Ui::Itm::Global::InputTrc)
  {
    case CONTENT_TRC_GAMMA_22:
    {
      BRANCH()
      if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma22SCurveTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedGamma22LinearTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_APPLY_GAMMA)
      {
        colour = sign(colour) * pow(abs(colour), 2.2f);
      }
      else
      {
        colour = saturate(colour);
        colour = pow(colour, 2.2f);
      }
    }
    break;
    case CONTENT_TRC_GAMMA_24:
    {
      BRANCH()
      if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma24SCurveTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedGamma24LinearTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_APPLY_GAMMA)
      {
        colour = sign(colour) * pow(abs(colour), 2.4f);
      }
      else
      {
        colour = saturate(colour);
        colour = pow(colour, 2.4f);
      }
    }
    break;
    case CONTENT_TRC_LINEAR:
    {
      BRANCH()
      if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_CLAMP)
      {
        colour = saturate(colour);
      }
    }
    break;
    case CONTENT_TRC_LINEAR_WITH_BLACK_FLOOR_EMU:
    {
      BRANCH()
      if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma22SCurveTo::Linear(Csp::Trc::LinearTo::Srgb(colour));
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_LINEAR)
      {
        float3 absColour  = abs(colour);
        float3 signColour = sign(colour);
        [branch]
        if (absColour.r < 1.f)
        {
          colour.r = signColour.r * pow(Csp::Trc::LinearTo::Srgb(absColour.r), 2.2f);
        }
        [branch]
        if (absColour.g < 1.f)
        {
          colour.g = signColour.g * pow(Csp::Trc::LinearTo::Srgb(absColour.g), 2.2f);
        }
        [branch]
        if (absColour.b < 1.f)
        {
          colour.b = signColour.b * pow(Csp::Trc::LinearTo::Srgb(absColour.b), 2.2f);
        }
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_APPLY_GAMMA)
      {
        colour = sign(colour) * pow(Csp::Trc::LinearTo::Srgb(abs(colour)), 2.2f);
      }
      else
      {
        colour = saturate(colour);
        colour = pow(Csp::Trc::LinearTo::Srgb(colour), 2.2f);
      }
    }
    break;
    case CONTENT_TRC_SRGB:
    {
      BRANCH()
      if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedSrgbSCurveTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedSrgbLinearTo::Linear(colour);
      }
      else if (Ui::Itm::Global::OverbrightHandling == OVERBRIGHT_HANDLING_APPLY_GAMMA)
      {
        colour = sign(colour) * Csp::Trc::SrgbTo::Linear(abs(colour));
      }
      else
      {
        colour = saturate(colour);
        colour = Csp::Trc::SrgbTo::Linear(colour);
      }
    }
    break;
    default:
      break;
  }

  //colour = gamut(colour, EXPAND_GAMUT);

  switch (Ui::Itm::Global::ItmMethod)
  {
    case ITM_METHOD_BT2446A:
    {
      float inputNitsFactor = Ui::Itm::Bt2446A::Bt2446AMaxInputBrightness > Ui::Itm::Bt2446A::Bt2446AInputBrightness
                            ? Ui::Itm::Bt2446A::Bt2446AMaxInputBrightness / Ui::Itm::Bt2446A::Bt2446AInputBrightness
                            : 1.f;

      float referenceWhiteNits = Ui::Itm::Bt2446A::Bt2446AInputBrightness * inputNitsFactor;
            referenceWhiteNits = referenceWhiteNits < Ui::Itm::Global::TargetBrightness
                               ? referenceWhiteNits
                               : Ui::Itm::Global::TargetBrightness;

      colour = Itmos::Bt2446A(colour,
                              Ui::Itm::Bt2446A::Bt2446AProcessingMode,
                              Ui::Itm::Global::TargetBrightness,
                              referenceWhiteNits,
                              inputNitsFactor,
                              Ui::Itm::Bt2446A::GammaIn,
                              Ui::Itm::Bt2446A::GammaOut);
    } break;

    case ITM_METHOD_BT2446C:
    {
      colour = Itmos::Bt2446C(colour,
                              Ui::Itm::Bt2446C::Bt2446CInputBrightness > 153.9f
                            ? 1.539f
                            : Ui::Itm::Bt2446C::Bt2446CInputBrightness / 100.f,
                              0.33f - Ui::Itm::Bt2446C::Alpha);
                              //BT2446C_USE_ACHROMATIC_CORRECTION,
                              //BT2446C_SIGMA);
    } break;

#ifdef ENABLE_DICE

    case ITM_METHOD_DICE_INVERSE:
    {
      float targetNitsNormalised = Ui::Itm::Global::TargetBrightness / 10000.f;
      colour = Itmos::Dice::InverseToneMapper(
                 colour,
                 Csp::Trc::NitsTo::Pq(Ui::Itm::Dice::DiceInputBrightness),
                 Csp::Trc::NitsTo::Pq(Ui::Itm::Dice::ShoulderStart / 100.f * Ui::Itm::Dice::DiceInputBrightness));
    } break;

#endif //ENABLE_DICE
  }

  Output = float4(colour, inputColour.a);
}


technique lilium__inverse_tone_mapping
<
  ui_label = "Lilium's inverse tone mapping";
>
{
  pass PS_InverseToneMapping
  {
    VertexShader = VS_PostProcessWithoutTexCoord;
     PixelShader = PS_InverseToneMapping;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__inverse_tone_mapping
<
  ui_label = "Lilium's inverse tone mapping (ERROR)";
>
VS_ERROR

#endif //is hdr API and hdr colour space
