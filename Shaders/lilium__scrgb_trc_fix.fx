#include "lilium__include/colour_space.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


uniform uint INPUT_TRC
<
  ui_label    = "input TRC";
  ui_type     = "combo";
  ui_tooltip  = "TRC = tone reproduction curve"
           "\n" "also wrongly known as \"gamma\"";
  ui_items    = "sRGB\0"
                "gamma 2.2\0"
                "gamma 2.4\0"
                "PQ\0";
> = 0;

#define TRC_SRGB     0
#define TRC_GAMMA_22 1
#define TRC_GAMMA_24 2
#define TRC_PQ       3

uniform float SDR_WHITEPOINT_NITS
<
  ui_label = "SDR whitepoint (in nits)";
  ui_type  = "drag";
  ui_units = " nits";
  ui_min   = 1.f;
  ui_max   = 300.f;
  ui_step  = 1.f;
> = 80.f;

uniform bool ENABLE_GAMMA_ADJUST
<
  ui_label = "enable gamma adjust";
> = false;

uniform float GAMMA_ADJUST
<
  ui_label = "gamma adjust";
  ui_type  = "drag";
  ui_min   = -1.f;
  ui_max   =  1.f;
  ui_step  =  0.001f;
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


void ScrgbTrcFix(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 Output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, TexCoord).rgb;

  float3 fixedGamma = input;

  if (INPUT_TRC == TRC_SRGB)
  {
    fixedGamma = Csp::Trc::FromExtendedSrgb(fixedGamma);
  }
  else if (INPUT_TRC == TRC_GAMMA_22)
  {
    fixedGamma = Csp::Trc::FromExtendedGamma22(fixedGamma);
  }
  else if (INPUT_TRC == TRC_GAMMA_24)
  {
    fixedGamma = Csp::Trc::FromExtendedGamma24(fixedGamma);
  }
#if (CSP_OVERRIDE != CSP_PS5)
  else if (INPUT_TRC == TRC_PQ)
  {
    fixedGamma = Csp::Mat::Bt2020To::Bt709(Csp::Trc::FromPq(fixedGamma)) * 125.f;
  }
#endif

  if (ENABLE_CLAMPING)
  {
    fixedGamma = clamp(fixedGamma, CLAMP_NEGATIVE_TO, CLAMP_POSITIVE_TO);
  }

#if (CSP_OVERRIDE == CSP_PS5)
  if (INPUT_TRC != TRC_PQ)
  {
    fixedGamma = Csp::Mat::Bt709To::Bt2020(fixedGamma);
  }
#endif


  if (ENABLE_GAMMA_ADJUST)
  {
    fixedGamma = Csp::Trc::ExtendedGammaAdjust(fixedGamma, 1.f + GAMMA_ADJUST);
  }

//  if (dot(Bt709ToXYZ[1].rgb, fixedGamma) < 0.f)
//    fixedGamma = float3(0.f, 0.f, 0.f);

#if (CSP_OVERRIDE == CSP_PS5)

    fixedGamma *= (SDR_WHITEPOINT_NITS / 100.f);

#else

    fixedGamma *= (SDR_WHITEPOINT_NITS / 80.f);

#endif

  fixedGamma = fixNAN(fixedGamma);

  //fixedGamma = clamp(fixedGamma, -65504.f, 125.f);

  Output = float4(fixedGamma, 1.f);
}


technique lilium__scRGB_trc_fix
<
  ui_label = "Lilium's scRGB TRC fix";
>
{
  pass scRGB_TRC_fix
  {
    VertexShader = PostProcessVS;
     PixelShader = ScrgbTrcFix;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__scRGB_trc_fix
<
  ui_label = "Lilium's scRGB TRC fix (ERROR)";
>
{
  pass CS_Error
  {
    ComputeShader = CS_Error<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

#endif //is hdr API and hdr colour space
