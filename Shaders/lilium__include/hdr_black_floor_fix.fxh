#pragma once

#include "colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_HDR_CSP))


// TODO:
// add as post adjustment in tone mapping and inverse tone mapping


namespace Ui
{
  namespace HdrBlackFloorFix
  {
    namespace Gamma22Emu
    {
      uniform bool EnableGamma22Emu
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "enable SDR black floor emulation";
        ui_tooltip  = "This emulates how the black floor looks on an SDR display using gamma 2.2.";
      > = false;

      uniform uint ProcessingColourSpace
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "processing colour space";
        ui_tooltip  = "Using BT.709 will not push affected colours outside of BT.709."
                 "\n" "Using DCI-P3 can push affected colours into DCI-P3."
                 "\n" "Using BT.2020 can push affected colours into DCI-P3 and BT.2020.";
        ui_type     = "combo";
        ui_items    = "BT.709\0"
                      "DCI-P3\0"
                      "BT.2020\0";
      > = 0;

#define HDR_BF_FIX_CSP_BT709  0
#define HDR_BF_FIX_CSP_DCI_P3 1
#define HDR_BF_FIX_CSP_BT2020 2

      uniform float WhitePoint
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "processing cut off";
        ui_tooltip  = "How much of the lower range range should be processed.";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 40.f;
        ui_max      = 300.f;
        ui_step     = 0.5f;
      > = 80.f;

      uniform bool OnlyLowerBlackLevels
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "only lower black levels";
        ui_tooltip  = "The gamma 2.2 emulation lowers black levels and slightly raises hightlights."
                 "\n" "This option only enables the lowering of black levels.";
      > = false;
    }

    namespace Lowering
    {
      uniform bool EnableLowering
      <
        ui_category = "black floor lowering";
        ui_label    = "enable black floor lowering";
      > = false;

      uniform uint ProcessingMode
      <
        ui_category = "black floor lowering";
        ui_label    = "black floor lowering processing mode";
        ui_type     = "combo";
        ui_tooltip  = "ICtCp:     process in ICtCp space (best quality)"
                 "\n" "YCbCr:     process in YCbCr space"
                 "\n" "YRGB:      process RGB according to brightness"
                 "\n" "RGB in PQ: process RGB encoded in PQ according to brightness"
                 "\n" "RGB:       process RGB according to brightness (different approach)";
        ui_items    = "ICtCp\0"
                      "YCbCr\0"
                      "YRGB\0"
                      "RGB in PQ\0"
                      "RGB\0";
      > = 0;

#define PRO_MODE_ICTCP     0
#define PRO_MODE_YCBCR     1
#define PRO_MODE_YRGB      2
#define PRO_MODE_RGB_IN_PQ 3
#define PRO_MODE_RGB       4

      uniform float OldBlackPoint
      <
        ui_category  = "black floor lowering";
        ui_label     = "old black point";
        ui_type      = "slider";
        ui_units     = " nits";
        ui_min       = 0.f;
        ui_max       = 0.5f;
        ui_step      = 0.0000001f;
      > = 0.f;

      uniform float RollOffStoppingPoint
      <
        ui_category  = "black floor lowering";
        ui_label     = "roll off stopping point";
        ui_tooltip   = "How much of the lower image range is used"
                  "\n" "to roll off from the new blackpoint.";
        ui_type      = "drag";
        ui_units     = " nits";
        ui_min       = 1.f;
        ui_max       = 20.f;
        ui_step      = 0.01f;
      > = 10.f;

      uniform float NewBlackPoint
      <
        ui_category  = "black floor lowering";
        ui_label     = "new black point";
        ui_tooltip   = "Can be negative for the sake of having a true 0 black point."
                  "\n" "Because some processing modes can't reach true 0 in every case.";
        ui_type      = "drag";
        ui_units     = " nits";
        ui_min       = -0.1f;
        ui_max       = 0.1f;
        ui_step      = 0.0000001f;
      > = 0.f;
    }
  }
}


void Gamma22Emulation(
  inout float3 Colour,
        float  WhitePointNormalised)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
  {
    Colour = Csp::Mat::Bt709To::DciP3(Colour);
  }
  else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
  {
    Colour = Csp::Mat::Bt709To::Bt2020(Colour);
  }

#elif defined(IS_HDR10_LIKE_CSP)

  if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
  {
    Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  }
  else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
  {
    Colour = Csp::Mat::Bt2020To::DciP3(Colour);
  }

#endif //IS_XXX_LIKE_CSP

  const bool3 isInProcessingRange = Colour < WhitePointNormalised;
  const bool3 isAbove0            = Colour >  0.f;

  const bool3 needsProcessing = isInProcessingRange && isAbove0;

  float3 processedColour = Colour;

  if (needsProcessing.r)
  {
    processedColour.r = pow(Csp::Trc::LinearTo::Srgb(processedColour.r / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.r > Colour.r)
    {
      processedColour.r = Colour.r;
    }
  }
  if (needsProcessing.g)
  {
    processedColour.g = pow(Csp::Trc::LinearTo::Srgb(processedColour.g / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.g > Colour.g)
    {
      processedColour.g = Colour.g;
    }
  }
  if (needsProcessing.b)
  {
    processedColour.b = pow(Csp::Trc::LinearTo::Srgb(processedColour.b / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.b > Colour.b)
    {
      processedColour.b = Colour.b;
    }
  }

  Colour = processedColour;

  return;
}

#define BLACK_POINT_ADAPTION(T)                            \
  T BlackPointAdaption(                                    \
    T C1,                                                  \
    float OldBlackPoint,                                   \
    float RollOffMinusOldBlackPoint,                       \
    float MinLum)                                          \
  {                                                        \
    T C2;                                                  \
                                                           \
    /*E1*/                                                 \
    C2 = (C1 - OldBlackPoint) / RollOffMinusOldBlackPoint; \
                                                           \
    /*E3*/                                                 \
    C2 += MinLum * pow((1.f - C2), 4.f);                   \
                                                           \
    /*E4*/                                                 \
    return C2 * RollOffMinusOldBlackPoint + OldBlackPoint; \
  }

BLACK_POINT_ADAPTION(float)
BLACK_POINT_ADAPTION(float3)


float GetNits(float3 Colour)
{
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
    {
      return dot(Colour, Csp::Mat::Bt709ToXYZ[1].rgb);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      return dot(Colour, Csp::Mat::DciP3ToXYZ[1].rgb);
    }
    else //if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
    {
      return dot(Colour, Csp::Mat::Bt2020ToXYZ[1].rgb);
    }
  }
  else
  {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    return dot(Colour, Csp::Mat::Bt709ToXYZ[1].rgb);

#elif defined(IS_HDR10_LIKE_CSP)

    return dot(Colour, Csp::Mat::Bt2020ToXYZ[1].rgb);

#endif
  }
}


void ConvertToWorkingCsp(inout float3 Colour)
{
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
    {
      Colour = Csp::Mat::Bt709To::Bt2020(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      Colour = Csp::Mat::DciP3To::Bt2020(Colour);
    }
    return;
  }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  else
  {
    Colour = Csp::Mat::Bt709To::Bt2020(Colour);
    return;
  }
#endif
}


void ConvertToOutputCspAfterProcessing(inout float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  Colour *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Colour = Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  Colour = Csp::Trc::LinearTo::Hlg(Colour);

#endif //ACTUAL_COLOUR_SPACE ==

  return;
}


void ConvertToOutputCspWithoutProcessing(inout float3 Colour)
{
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      Colour = Csp::Mat::DciP3To::Bt709(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
    {
      Colour = Csp::Mat::Bt2020To::Bt709(Colour);
    }

    Colour *= 125.f;
    return;

#elif defined(IS_HDR10_LIKE_CSP)

    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
    {
      Colour = Csp::Mat::Bt709To::Bt2020(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      Colour = Csp::Mat::DciP3To::Bt2020(Colour);
    }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    Colour = Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    Colour = Csp::Trc::LinearTo::Hlg(Colour);

#endif
    return;

#endif

  }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour *= 125.f;
  return;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Colour = Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  Colour = Csp::Trc::LinearTo::Hlg(Colour);

#endif
}


void LowerBlackFloor(
  inout float3 Colour,
        float  RollOffStoppingPoint,
        float  OldBlackPoint,
        float  RollOffMinusOldBlackPoint,
        float  MinLum)
{
  // ICtCp mode
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_ICTCP)
  {
    //to L'M'S'
    float3 PqLms;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
    {
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
      {
        PqLms = Csp::Ictcp::Bt709To::PqLms(Colour);
      }
      else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
      {
        PqLms = Csp::Ictcp::DciP3To::PqLms(Colour);
      }
      else //if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
      {
        PqLms = Csp::Ictcp::Bt2020To::PqLms(Colour);
      }
    }
    else
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      PqLms = Csp::Ictcp::Bt709To::PqLms(Colour);

#elif defined(IS_HDR10_LIKE_CSP)

      PqLms = Csp::Ictcp::Bt2020To::PqLms(Colour);

#endif
    }

    //Intensity
    float i1 = 0.5f * PqLms.x + 0.5f * PqLms.y;

    if (i1 <= RollOffStoppingPoint)
    {
      float i2 = BlackPointAdaption(i1,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      //to RGB
      Colour = Csp::Ictcp::IctcpTo::Bt2020(float3(i2,
                                                  dot(PqLms, PqLmsToIctcp[1]),
                                                  dot(PqLms, PqLmsToIctcp[2])));

      Colour = max(Colour, 0.f);

      ConvertToOutputCspAfterProcessing(Colour);
      return;
    }
    else
    {
      ConvertToOutputCspWithoutProcessing(Colour);
      return;
    }
  }
  // YCbCr mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YCBCR)
  {
    float3 inputInPq = Colour;
    ConvertToWorkingCsp(inputInPq);
    inputInPq = Csp::Trc::LinearTo::Pq(inputInPq);

    float y1 = dot(inputInPq, KBt2020);

    if (y1 <= RollOffStoppingPoint)
    {
      float y2 = BlackPointAdaption(y1,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      Colour = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(y2,
                                                     (inputInPq.b - y1) / KbBt2020,
                                                     (inputInPq.r - y1) / KrBt2020));

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)

      Colour = Csp::Trc::PqTo::Linear(Colour);
      ConvertToOutputCspAfterProcessing(Colour);

#else

      Colour = max(Colour, 0.f);

#endif
      return;
    }
    else
    {
      ConvertToOutputCspWithoutProcessing(Colour);
      return;
    }
  }
  // YRGB mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YRGB)
  {
    float y1 = GetNits(Colour);

    float y1InPq = Csp::Trc::LinearTo::Pq(y1);

    if (y1InPq <= RollOffStoppingPoint)
    {
      ConvertToWorkingCsp(Colour);

      float y2 = BlackPointAdaption(y1InPq,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      y2 = Csp::Trc::PqTo::Linear(y2);

      Colour = max(y2 / y1 * Colour, 0.f);

      ConvertToOutputCspAfterProcessing(Colour);
      return;
    }
    else
    {
      ConvertToOutputCspWithoutProcessing(Colour);
      return;
    }
  }
  // RGB in PQ mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB_IN_PQ)
  {
    float nits = GetNits(Colour);

    if (nits <= RollOffStoppingPoint)
    {
      float3 rgbInPq = Colour;
      ConvertToWorkingCsp(rgbInPq);
      rgbInPq = Csp::Trc::LinearTo::Pq(rgbInPq);

      rgbInPq = BlackPointAdaption(rgbInPq,
                                   OldBlackPoint,
                                   RollOffMinusOldBlackPoint,
                                   MinLum);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)

      Colour = Csp::Trc::PqTo::Linear(rgbInPq);
      ConvertToOutputCspAfterProcessing(Colour);

#else

      Colour = max(rgbInPq, 0.f);

#endif
      return;
    }
    else
    {
      ConvertToOutputCspWithoutProcessing(Colour);
      return;
    }
  }
  // RBG mode
  else // if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB)
  {
    if (GetNits(Colour) <= RollOffStoppingPoint)
    {
      float3 rgb = Colour;
      ConvertToWorkingCsp(rgb);

      //E1
      rgb = BlackPointAdaption(rgb,
                               OldBlackPoint,
                               RollOffMinusOldBlackPoint,
                               MinLum);

      Colour = max(rgb, 0.f);

      ConvertToOutputCspAfterProcessing(Colour);
      return;
    }
    else
    {
      ConvertToOutputCspWithoutProcessing(Colour);
      return;
    }
  }

}

#endif //is hdr API and hdr colour space
