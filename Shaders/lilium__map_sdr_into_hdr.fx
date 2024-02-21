#include "lilium__include/colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_HDR_CSP))


uniform uint INPUT_TRC
<
  ui_label    = "input gamma";
  ui_type     = "combo";
  ui_items    = "gamma 2.2\0"
                "gamma 2.4\0"
                "sRGB\0"
                "PQ\0";
> = 0;

#define TRC_GAMMA_22 0
#define TRC_GAMMA_24 1
#define TRC_SRGB     2
#define TRC_PQ       3

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


void PS_MapSdrIntoHdr(
      float4 VPos     : SV_Position,
  out float4 Output   : SV_Target0)
{
  float4 inputColour = tex2Dfetch(ReShade::BackBuffer, int2(VPos.xy));

  float3 fixedGamma = inputColour.rgb;

  static const bool inputTrcIsPq = INPUT_TRC == TRC_PQ;

  if (INPUT_TRC == TRC_GAMMA_22)
  {
    fixedGamma = Csp::Trc::ExtendedGamma22To::Linear(fixedGamma);
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)
    fixedGamma = Csp::Mat::Bt709To::Bt2020(fixedGamma);
#endif
  }
  else if (INPUT_TRC == TRC_GAMMA_24)
  {
    fixedGamma = Csp::Trc::ExtendedGamma24To::Linear(fixedGamma);
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)
    fixedGamma = Csp::Mat::Bt709To::Bt2020(fixedGamma);
#endif
  }
  else if (INPUT_TRC == TRC_SRGB)
  {
      fixedGamma = Csp::Trc::ExtendedSrgbTo::Linear(fixedGamma);
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)
    fixedGamma = Csp::Mat::Bt709To::Bt2020(fixedGamma);
#endif
  }
  else //if (inputTrcIsPq)
  {
#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)
    fixedGamma = Csp::Trc::PqTo::Linear(fixedGamma);
#endif
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    fixedGamma = Csp::Mat::Bt2020To::Bt709(fixedGamma) * 125.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
    fixedGamma = fixedGamma * 100.f;
#endif
  }

  if (ENABLE_CLAMPING)
  {
    fixedGamma = clamp(fixedGamma, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);
  }

  if (ENABLE_GAMMA_ADJUST
   && !inputTrcIsPq)
  {
    fixedGamma = Csp::Trc::ExtendedGammaAdjust(fixedGamma, 1.f + GAMMA_ADJUST);
  }

//  if (dot(Bt709ToXYZ[1].rgb, fixedGamma) < 0.f)
//    fixedGamma = float3(0.f, 0.f, 0.f);

  if (!inputTrcIsPq)
  {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    fixedGamma *= (SDR_WHITEPOINT_NITS / 80.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    fixedGamma *= (SDR_WHITEPOINT_NITS / 100.f);

#endif
  }

  //fixedGamma = fixNAN(fixedGamma);

  Output = float4(fixedGamma, inputColour.a);
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
CS_ERROR

#endif //is hdr API and hdr colour space
