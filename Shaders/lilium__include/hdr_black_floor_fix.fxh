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


float3 Gamma22Emulation(
  const float3 Colour,
  const float  WhitePointNormalised)
{
  float3 correctCspColour;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
  {
    correctCspColour = Csp::Mat::Bt709To::DciP3(Colour);
  }
  else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
  {
    correctCspColour = Csp::Mat::Bt709To::Bt2020(Colour);
  }

#elif defined(IS_HDR10_LIKE_CSP)

  if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
  {
    correctCspColour = Csp::Mat::Bt2020To::Bt709(Colour);
  }
  else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
  {
    correctCspColour = Csp::Mat::Bt2020To::DciP3(Colour);
  }

#endif //IS_XXX_LIKE_CSP

  else
  {
    correctCspColour = Colour;
  }

  const bool3 isInProcessingRange = correctCspColour < WhitePointNormalised;
  const bool3 isAbove0            = correctCspColour >  0.f;

  const bool3 needsProcessing = isInProcessingRange && isAbove0;

  float3 processedColour = correctCspColour;

  if (needsProcessing.r)
  {
    processedColour.r = pow(Csp::Trc::LinearTo::Srgb(correctCspColour.r / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.r > correctCspColour.r)
    {
      processedColour.r = correctCspColour.r;
    }
  }
  if (needsProcessing.g)
  {
    processedColour.g = pow(Csp::Trc::LinearTo::Srgb(correctCspColour.g / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.g > correctCspColour.g)
    {
      processedColour.g = correctCspColour.g;
    }
  }
  if (needsProcessing.b)
  {
    processedColour.b = pow(Csp::Trc::LinearTo::Srgb(correctCspColour.b / WhitePointNormalised), 2.2f) * WhitePointNormalised;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::OnlyLowerBlackLevels
     && processedColour.b > correctCspColour.b)
    {
      processedColour.b = correctCspColour.b;
    }
  }

  return processedColour;
}

#define BLACK_POINT_ADAPTION(T)                            \
  T BlackPointAdaption(                                    \
    const T     C1,                                        \
    const float OldBlackPoint,                             \
    const float RollOffMinusOldBlackPoint,                 \
    const float MinLum)                                    \
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


float GetNits(const float3 Colour)
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

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return dot(Colour, Csp::Mat::Bt709ToXYZ[1].rgb);

#elif defined(IS_HDR10_LIKE_CSP)

  return dot(Colour, Csp::Mat::Bt2020ToXYZ[1].rgb);

#endif
}


float3 ConvertToWorkingCsp(const float3 Colour)
{
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
    {
      return Csp::Mat::Bt709To::Bt2020(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      return Csp::Mat::DciP3To::Bt2020(Colour);
    }
  }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Mat::Bt709To::Bt2020(Colour);

#endif
}


float3 ConvertToOutputCspAfterProcessing(const float3 Colour)
{
  float3 outputColour;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  outputColour  = Csp::Mat::Bt2020To::Bt709(Colour);
  outputColour *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  outputColour = Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  outputColour = Csp::Trc::LinearTo::Hlg(Colour);

#endif //ACTUAL_COLOUR_SPACE ==

  return outputColour;
}


float3 ConvertToOutputCspWithoutProcessing(const float3 Colour)
{

  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    float3 outputColour;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      outputColour = Csp::Mat::DciP3To::Bt709(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
    {
      outputColour = Csp::Mat::Bt2020To::Bt709(Colour);
    }

    outputColour *= 125.f;

#elif defined(IS_HDR10_LIKE_CSP)

    if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
    {
      outputColour = Csp::Mat::Bt709To::Bt2020(Colour);
    }
    else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
    {
      outputColour = Csp::Mat::DciP3To::Bt2020(Colour);
    }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    outputColour = Csp::Trc::LinearTo::Pq(outputColour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    outputColour = Csp::Trc::LinearTo::Hlg(outputColour);

#endif
    return outputColour;

#endif

  }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Colour * 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  return Csp::Trc::LinearTo::Hlg(Colour);

#endif
}


float3 LowerBlackFloor(
  const float3 Rgb,
  const float  RollOffStoppingPoint,
  const float  OldBlackPoint,
  const float  RollOffMinusOldBlackPoint,
  const float  MinLum)
{
  // ICtCp mode
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_ICTCP)
  {
    //to L'M'S'
    float3 pqLms;

    if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
    {
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
      {
        pqLms = Csp::Ictcp::Bt709To::PqLms(Rgb);
      }
      else if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
      {
        pqLms = Csp::Ictcp::DciP3To::PqLms(Rgb);
      }
      else //if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
      {
        pqLms = Csp::Ictcp::Bt2020To::PqLms(Rgb);
      }
    }
    else
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      pqLms = Csp::Ictcp::Bt709To::PqLms(Rgb);

#elif defined(IS_HDR10_LIKE_CSP)

      pqLms = Csp::Ictcp::Bt2020To::PqLms(Rgb);

#endif
    }

    //Intensity
    float i1 = 0.5f * pqLms.x + 0.5f * pqLms.y;

    if (i1 <= RollOffStoppingPoint)
    {
      float i2 = BlackPointAdaption(i1,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      //to RGB
      float3 outputRgb = Csp::Ictcp::IctcpTo::Bt2020(float3(i2,
                                                            dot(pqLms, PqLmsToIctcp[1]),
                                                            dot(pqLms, PqLmsToIctcp[2])));

      outputRgb = max(outputRgb, 0.f);

      return ConvertToOutputCspAfterProcessing(outputRgb);
    }
    else
    {
      return ConvertToOutputCspWithoutProcessing(Rgb);
    }
  }
  // YCbCr mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YCBCR)
  {
    float3 inputInPq = Csp::Trc::LinearTo::Pq(ConvertToWorkingCsp(Rgb));

    float y1 = dot(inputInPq, KBt2020);

    if (y1 <= RollOffStoppingPoint)
    {
      float y2 = BlackPointAdaption(y1,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      //to RGB
      float3 outputRgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(y2,
                                                               (inputInPq.b - y1) / KbBt2020,
                                                               (inputInPq.r - y1) / KrBt2020));

      outputRgb = max(outputRgb, 0.f);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)

      outputRgb = Csp::Trc::PqTo::Linear(outputRgb);
      outputRgb = ConvertToOutputCspAfterProcessing(outputRgb);

#endif
      return outputRgb;
    }
    else
    {
      return ConvertToOutputCspWithoutProcessing(Rgb);
    }
  }
  // YRGB mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YRGB)
  {
    float y1 = GetNits(Rgb);

    float y1InPq = Csp::Trc::LinearTo::Pq(y1);

    if (y1InPq <= RollOffStoppingPoint)
    {
      float y2 = BlackPointAdaption(y1InPq,
                                    OldBlackPoint,
                                    RollOffMinusOldBlackPoint,
                                    MinLum);

      y2 = Csp::Trc::PqTo::Linear(y2);

      float3 outputRgb = y2 / y1 * Rgb;

      return ConvertToOutputCspWithoutProcessing(outputRgb);
    }
    else
    {
      return ConvertToOutputCspWithoutProcessing(Rgb);
    }
  }
  // RGB in PQ mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB_IN_PQ)
  {
    float nits = GetNits(Rgb);

    if (nits <= RollOffStoppingPoint)
    {
      float3 rgbInPq1 = Csp::Trc::LinearTo::Pq(ConvertToWorkingCsp(Rgb));

      float3 rgbInPq2 = BlackPointAdaption(rgbInPq1,
                                           OldBlackPoint,
                                           RollOffMinusOldBlackPoint,
                                           MinLum);

      float3 outputRgb = max(rgbInPq2, 0.f);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10)

      outputRgb = Csp::Trc::PqTo::Linear(outputRgb);
      outputRgb = ConvertToOutputCspAfterProcessing(outputRgb);

#endif
      return outputRgb;
    }
    else
    {
      return ConvertToOutputCspWithoutProcessing(Rgb);
    }
  }
  // RBG mode
  else // if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB)
  {
    if (GetNits(Rgb) <= RollOffStoppingPoint)
    {
      float3 rgb1 = ConvertToWorkingCsp(Rgb);

      float3 rgb2 = BlackPointAdaption(rgb1,
                                       OldBlackPoint,
                                       RollOffMinusOldBlackPoint,
                                       MinLum);

      float3 outputRgb = max(rgb2, 0.f);

      return ConvertToOutputCspAfterProcessing(outputRgb);
    }
    else
    {
      return ConvertToOutputCspWithoutProcessing(Rgb);
    }
  }

}

#endif //is hdr API and hdr colour space
