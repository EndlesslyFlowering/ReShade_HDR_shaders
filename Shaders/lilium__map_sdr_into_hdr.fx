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

#define TRC_GAMMA_22                    0
#define TRC_GAMMA_24                    1
#define TRC_LINEAR                      2
#define TRC_LINEAR_WITH_BLACK_FLOOR_EMU 3
#define TRC_SRGB                        4
#define TRC_PQ                          5

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


// convert BT.709 to BT.2020
float3 ConditionallyConvertBt709ToBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
#endif
  return Colour;
}

// convert HDR10 to linear BT.2020
float3 ConditionallyLineariseHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)
  Colour = Csp::Trc::PqTo::Linear(Colour);
#endif
  return Colour;
}

// convert BT.2020 to BT.709
float3 ConditionallyConvertBt2020To709(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
#endif
  return Colour;
}


void PS_MapSdrIntoHdr(
      float4 Position : SV_Position,
  out float4 Output   : SV_Target0)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  float3 colour = inputColour.rgb;

  static const bool inputTrcIsPq = INPUT_TRC == TRC_PQ;

  if (INPUT_TRC == TRC_GAMMA_22)
  {
    colour = Csp::Trc::ExtendedGamma22To::Linear(colour);
    colour = ConditionallyConvertBt709ToBt2020(colour);
  }
  else if (INPUT_TRC == TRC_GAMMA_24)
  {
    colour = Csp::Trc::ExtendedGamma24To::Linear(colour);
    colour = ConditionallyConvertBt709ToBt2020(colour);
  }
  else if (INPUT_TRC == TRC_LINEAR)
  {
    colour = ConditionallyConvertBt709ToBt2020(colour);
  }
  else if (INPUT_TRC == TRC_LINEAR_WITH_BLACK_FLOOR_EMU)
  {
    colour = Csp::Trc::ExtendedGamma22To::Linear(Csp::Trc::LinearTo::Srgb(colour));
    colour = ConditionallyConvertBt709ToBt2020(colour);
  }
  else if (INPUT_TRC == TRC_SRGB)
  {
    colour = Csp::Trc::ExtendedSrgbTo::Linear(colour);
    colour = ConditionallyConvertBt709ToBt2020(colour);
  }
  else //if (inputTrcIsPq)
  {
    colour = ConditionallyLineariseHdr10(colour);
    colour = ConditionallyConvertBt2020To709(colour);
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    colour *= 125.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
    colour *= 100.f;
#endif
  }

  if (ENABLE_CLAMPING)
  {
    colour = clamp(colour, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);
  }

  if (ENABLE_GAMMA_ADJUST
   && !inputTrcIsPq)
  {
    colour = Csp::Trc::ExtendedGammaAdjust(colour, 1.f + GAMMA_ADJUST);
  }

//  if (dot(Bt709ToXYZ[1].rgb, colour) < 0.f)
//    colour = float3(0.f, 0.f, 0.f);

  if (!inputTrcIsPq)
  {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    colour *= (SDR_WHITEPOINT_NITS / 80.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    colour *= (SDR_WHITEPOINT_NITS / 100.f);

#endif
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
