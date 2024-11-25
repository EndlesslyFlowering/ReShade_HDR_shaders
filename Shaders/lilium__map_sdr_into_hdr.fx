#include "lilium__include/colour_space.fxh"


#if (defined(IS_ANALYSIS_CAPABLE_API) \
  && defined(IS_HDR_CSP))


uniform uint INPUT_TRC
<
  ui_label    = "input gamma";
  ui_tooltip  = "\"linear with SDR black floor emulation (scRGB)\" fixes the sRGB<->gamma 2.2 mismatch";
  ui_type     = "combo";
  ui_items    = "2.2\0"
                "2.4\0"
                "linear (scRGB)\0"
                "linear with SDR black floor emulation (scRGB)\0"
                "sRGB\0"
                "PQ\0";
> = 0;

#undef TRC_PQ

#define TRC_GAMMA_22                    0
#define TRC_GAMMA_24                    1
#define TRC_LINEAR                      2
#define TRC_LINEAR_WITH_BLACK_FLOOR_EMU 3
#define TRC_SRGB                        4
#define TRC_PQ                          5

uniform uint OVERBRIGHT_HANDLING
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

uniform float SDR_WHITEPOINT_NITS
<
  ui_label   = "SDR whitepoint";
  ui_tooltip = "Only works when input gamma is not PQ!";
  ui_type    = "drag";
  ui_units   = " nits";
  ui_min     = 1.f;
  ui_max     = 300.f;
  ui_step    = 1.f;
> = 80.f;

uniform bool ENABLE_GAMMA_ADJUST
<
  ui_label   = "enable gamma adjust";
  ui_tooltip = "Only works when input gamma is not PQ!";
> = false;

uniform float GAMMA_ADJUST
<
  ui_label   = "gamma adjust";
  ui_tooltip = "Only works when input gamma is not PQ!";
  ui_type    = "drag";
  ui_min     = -1.f;
  ui_max     =  1.f;
  ui_step    =  0.001f;
> = 0.f;

uniform bool ENABLE_CLAMPING
<
  ui_category = "clamping";
  ui_label    = "enable clamping";
> = false;

uniform float CLAMP_NEGATIVE_TO
<
  ui_category = "clamping";
  ui_label    = "clamp negative values to";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      = 0.f;
  ui_step     = 0.1f;
> = -125.f;

uniform float CLAMP_POSITIVE_TO
<
  ui_category = "clamping";
  ui_label    = "clamp positive values to";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 125.f;
  ui_step     = 0.1f;
> = 125.f;


void PS_MapSdrIntoHdr
(
      float4 Position : SV_Position,
  out float4 Output   : SV_Target0
)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  float3 colour = inputColour.rgb;

  static const bool inputTrcIsPq = INPUT_TRC == TRC_PQ;

  switch(INPUT_TRC)
  {
    case TRC_GAMMA_22:
    {
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma22SCurveTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedGamma22LinearTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_APPLY_GAMMA)
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
    case TRC_GAMMA_24:
    {
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma24SCurveTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedGamma24LinearTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_APPLY_GAMMA)
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
    case TRC_LINEAR:
    {
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_CLAMP)
      {
        colour = saturate(colour);
      }
    }
    break;
    case TRC_LINEAR_WITH_BLACK_FLOOR_EMU:
    {
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedGamma22SCurveTo::Linear(Csp::Trc::LinearTo::Srgb(colour));
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_LINEAR)
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
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_APPLY_GAMMA)
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
    case TRC_SRGB:
    {
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_S_CURVE)
      {
        colour = Csp::Trc::ExtendedSrgbSCurveTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_LINEAR)
      {
        colour = Csp::Trc::ExtendedSrgbLinearTo::Linear(colour);
      }
      else
      BRANCH()
      if (OVERBRIGHT_HANDLING == OVERBRIGHT_HANDLING_APPLY_GAMMA)
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
    case TRC_PQ:
    {
      //scRGB
      colour = ConditionallyLineariseHdr10(colour);
      colour = ConditionallyConvertBt2020ToBt709(colour);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      colour *= 125.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
      colour *= 100.f;
#endif
    }
    break;
    default:
      break;
  }

  BRANCH()
  if (ENABLE_CLAMPING)
  {
    colour = clamp(colour, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);
  }

  BRANCH()
  if (ENABLE_GAMMA_ADJUST
   && !inputTrcIsPq)
  {
    colour = Csp::Trc::ExtendedGammaAdjust(colour, 1.f + GAMMA_ADJUST);
  }

//  if (dot(Bt709ToXYZ[1].rgb, colour) < 0.f)
//    colour = float3(0.f, 0.f, 0.f);

  BRANCH()
  if (!inputTrcIsPq)
  {

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    colour *= (SDR_WHITEPOINT_NITS / 80.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    colour *= (SDR_WHITEPOINT_NITS / 10000.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    colour *= (SDR_WHITEPOINT_NITS / 100.f);

#endif

    //HDR10
    colour = ConditionallyConvertBt709ToBt2020(colour);
    colour = ConditionallyConvertNormalisedBt2020ToHdr10(colour);
  }

  //colour = fixNAN(colour);

  Output = float4(colour, inputColour.a);
}


technique lilium__map_SDR_into_HDR
<
  ui_label = "Lilium's map SDR into HDR";
>
{
  pass PS_MapSdrIntoHdr
  {
    VertexShader = VS_PostProcessWithoutTexCoord;
     PixelShader = PS_MapSdrIntoHdr;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__map_SDR_into_HDR
<
  ui_label = "Lilium's map SDR into HDR (ERROR)";
>
VS_ERROR

#endif //is hdr API and hdr colour space
