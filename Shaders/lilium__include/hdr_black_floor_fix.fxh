#pragma once

#define ENABLE_COLOUR_OBJECT

#include "colour_space.fxh"


#if (defined(IS_ANALYSIS_CAPABLE_API) \
  && (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
   || ACTUAL_COLOUR_SPACE == CSP_HDR10))


// TODO:
// add as post adjustment in tone mapping and inverse tone mapping


namespace Ui
{
  namespace HdrBlackFloorFix
  {
    namespace Gamma22Emu
    {
      uniform bool G22EmuEnable
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "enable SDR black floor emulation";
        ui_tooltip  = "This emulates how the black floor looks on an SDR display using gamma 2.2.";
      > = false;

      uniform uint G22EmuProcessingColourSpace
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

#define G22_EMU_CSP_BT709  0
#define G22_EMU_CSP_DCI_P3 1
#define G22_EMU_CSP_BT2020 2

      uniform float G22EmuWhitePoint
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "processing cut off";
        ui_tooltip  = "How much of the lower range range should be processed.";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 10.f;
        ui_max      = 400.f;
        ui_step     = 0.5f;
      > = 80.f;

      uniform bool G22EmuOnlyLowerBlackLevels
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "only lower black levels";
        ui_tooltip  = "The gamma 2.2 emulation lowers black levels and slightly raises hightlights."
                 "\n" "This option only enables the lowering of black levels.";
      > = false;
    }

    namespace GammaAdjustment
    {
      uniform bool GAEnable
      <
        ui_category = "gamma adjustment";
        ui_label    = "enable gamma adjustment";
      > = false;

      uniform uint GAProcessingColourSpace
      <
        ui_category = "gamma adjustment";
        ui_label    = "processing colour space";
        ui_tooltip  = "Using BT.709 will not push affected colours outside of BT.709."
                 "\n" "Using DCI-P3 can push affected colours into DCI-P3."
                 "\n" "Using BT.2020 can push affected colours into DCI-P3 and BT.2020.";
        ui_type     = "combo";
        ui_items    = "BT.709\0"
                      "DCI-P3\0"
                      "BT.2020\0";
      > = 0;

#define GA_CSP_BT709  0
#define GA_CSP_DCI_P3 1
#define GA_CSP_BT2020 2

      uniform float GAGamma
      <
        ui_category = "gamma adjustment";
        ui_label    = "gamma";
        ui_type     = "drag";
        ui_min      = 0.5f;
        ui_max      = 1.5f;
        ui_step     = 0.001f;
      > = 1.f;

      uniform float GAWhitePoint
      <
        ui_category = "gamma adjustment";
        ui_label    = "processing cut off";
        ui_tooltip  = "How much of the lower range range should be processed.";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 10.f;
        ui_max      = 400.f;
        ui_step     = 0.5f;
      > = 80.f;
    }

    namespace Lowering
    {
      uniform bool LoweringEnable
      <
        ui_category = "black floor lowering";
        ui_label    = "enable black floor lowering";
      > = false;

      uniform uint ProcessingMode
      <
        ui_category = "black floor lowering";
        ui_label    = "black floor lowering processing mode";
        ui_type     = "combo";
        ui_tooltip  = "YRGB:      process RGB according to brightness"
                 "\n" "RGB in PQ: process RGB encoded in PQ according to brightness"
                 "\n" "RGB:       process RGB according to brightness (different approach)";
        ui_items    = "YRGB\0"
                      "RGB in PQ\0"
                      "RGB\0";
      > = 0;

#define PRO_MODE_YRGB      0
#define PRO_MODE_RGB_IN_PQ 1
#define PRO_MODE_RGB       2

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
        ui_max       = 50.f;
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


CO::ColourObject ConvertColourForGamma22Emulation
(
  CO::ColourObject CO
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  CO = CO::ConvertTrcTo::LinearNormalised(CO);

  BRANCH()
  if (Ui::HdrBlackFloorFix::Gamma22Emu::G22EmuProcessingColourSpace == G22_EMU_CSP_BT709)
  {
    CO = CO::ConvertPrimariesTo::Bt709(CO);
  }
  else
#endif
  BRANCH()
  if (Ui::HdrBlackFloorFix::Gamma22Emu::G22EmuProcessingColourSpace == G22_EMU_CSP_DCI_P3)
  {
    CO = CO::ConvertPrimariesTo::DciP3(CO);
  }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  else
  BRANCH()
  if (Ui::HdrBlackFloorFix::Gamma22Emu::G22EmuProcessingColourSpace == G22_EMU_CSP_BT2020)
  {
    CO = CO::ConvertPrimariesTo::Bt2020(CO);
  }
#endif

  return CO;
}


void Gamma22Emulation
(
  inout CO::ColourObject CO,
  const float            WhitePointNormalised,
  inout bool             ProcessingDone
)
{
  CO = ConvertColourForGamma22Emulation(CO);

  const float maxRange =
    Ui::HdrBlackFloorFix::Gamma22Emu::G22EmuOnlyLowerBlackLevels ? 0.3892221149351f * WhitePointNormalised
                                                                 : WhitePointNormalised;

  const bool3 isInProcessingRange = CO.RGB < maxRange;
  const bool3 isAbove0            = CO.RGB > 0.f;

  const bool3 needsProcessing = isInProcessingRange && isAbove0;

  [branch]
  if (any(needsProcessing))
  {
    float3 processedColour = pow(Csp::Trc::LinearTo::Srgb(CO.RGB / WhitePointNormalised), 2.2f)
                           * WhitePointNormalised;

    CO.RGB = needsProcessing ? processedColour
                             : CO.RGB;

    ProcessingDone = true;
  }

  return;
}


void GammaAdjustment
(
  inout CO::ColourObject CO,
  const float            WhitePointNormalised,
  inout bool             ProcessingDone
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  CO = CO::ConvertTrcTo::LinearNormalised(CO);
#endif

  BRANCH()
  if (Ui::HdrBlackFloorFix::GammaAdjustment::GAProcessingColourSpace == GA_CSP_BT709)
  {
    CO = CO::ConvertPrimariesTo::Bt709(CO);
  }
  else
  BRANCH()
  if (Ui::HdrBlackFloorFix::GammaAdjustment::GAProcessingColourSpace == GA_CSP_DCI_P3)
  {
    CO = CO::ConvertPrimariesTo::DciP3(CO);
  }
  else
  BRANCH()
  if (Ui::HdrBlackFloorFix::GammaAdjustment::GAProcessingColourSpace == GA_CSP_BT2020)
  {
    CO = CO::ConvertPrimariesTo::Bt2020(CO);
  }


  const bool3 isInProcessingRange = CO.RGB < WhitePointNormalised;
  const bool3 isAbove0            = CO.RGB > 0.f;

  const bool3 needsProcessing = isInProcessingRange && isAbove0;

  [branch]
  if (any(needsProcessing))
  {

    float3 processedColour = pow(CO.RGB / WhitePointNormalised, Ui::HdrBlackFloorFix::GammaAdjustment::GAGamma)
                           * WhitePointNormalised;

    CO.RGB = needsProcessing ? processedColour
                             : CO.RGB;

    ProcessingDone = true;
  }

  return;
}


void Gamma22EmulationAndGammaAdjustment
(
  inout CO::ColourObject CO,
  const float            WhitePointNormalised,
  inout bool             ProcessingDone
)
{
  CO = ConvertColourForGamma22Emulation(CO);

  const bool3 isInProcessingRange = CO.RGB < WhitePointNormalised;
  const bool3 isAbove0            = CO.RGB > 0.f;

  const bool3 needsProcessing = isInProcessingRange && isAbove0;

  [branch]
  if (any(needsProcessing))
  {
    const float currentPow = 2.2f *
                             Ui::HdrBlackFloorFix::GammaAdjustment::GAGamma;

    float3 processedColour = pow(Csp::Trc::LinearTo::Srgb(CO.RGB / WhitePointNormalised), currentPow)
                           * WhitePointNormalised;

    CO.RGB = needsProcessing ? processedColour
                             : CO.RGB;

    ProcessingDone = true;
  }

  return;
}


#define BLACK_POINT_ADAPTION(T)            \
  T BlackPointAdaption                     \
  (                                        \
    const T     C1,                        \
    const float OldBlackPoint,             \
    const float RollOffMinusOldBlackPoint, \
    const float MinLum                     \
  )                                        \
  {                                        \
    T C2;                                  \
                                           \
    /*E1*/                                 \
    C2 = (C1 - OldBlackPoint)              \
       / RollOffMinusOldBlackPoint;        \
                                           \
    /*E3*/                                 \
    T e3;                                  \
    e3 = 1.f - C2;                         \
    e3 = e3*e3;                            \
    e3 = e3*e3;                            \
    C2 = MinLum * e3 + C2;                 \
                                           \
    /*E4*/                                 \
    return C2                              \
         * RollOffMinusOldBlackPoint       \
         + OldBlackPoint;                  \
  }

BLACK_POINT_ADAPTION(float)
BLACK_POINT_ADAPTION(float3)


void LowerBlackFloor
(
  inout CO::ColourObject CO,
  const float            RollOffStoppingPoint,
  const float            OldBlackPoint,
  const float            RollOffMinusOldBlackPoint,
  const float            MinLum,
  const bool             ProcessingDone
)
{
  switch(Ui::HdrBlackFloorFix::Lowering::ProcessingMode)
  {
    // YRGB mode
    case PRO_MODE_YRGB:
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      CO = CO::ConvertCspTo::ScRgb(CO);

#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)

      CO::ColourObject CO_org;

      CO_org = CO;

      BRANCH()
      if (CO.trc == TRC_PQ)
      {
        CO = CO::ConvertTrcTo::LinearNormalised(CO);
      }
      else
      {
        CO = CO::ConvertPrimariesTo::Bt2020(CO);
      }

#endif

      float y1 = CO::GetLuminance::LinearNormalised(CO);

      float y1InPq = Csp::Trc::LinearTo::Pq(y1);

      [branch]
      if (y1InPq <= RollOffStoppingPoint)
      {
        float y2 = BlackPointAdaption(y1InPq,
                                      OldBlackPoint,
                                      RollOffMinusOldBlackPoint,
                                      MinLum);

        y2 = Csp::Trc::PqTo::Linear(y2);

        CO.RGB = y2 / y1 * CO.RGB;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertCspTo::ScRgb(CO);

        return;

#endif
      }
      else
      [branch]
      if (!ProcessingDone)
      {
        discard;
      }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      else
      {
        CO = CO::ConvertCspTo::ScRgb(CO);

        return;
      }
#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)

      CO = CO::ConvertTrcTo::Pq(CO);

      return;

#endif
    }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    break;
#endif
    // RGB in PQ mode
    case PRO_MODE_RGB_IN_PQ:
    {
      float nits = CO::GetLuminance::LinearNormalised(CO);

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      BRANCH()
      if (CO.trc != TRC_PQ)
      {
        CO = CO::ConvertCspTo::Hdr10(CO);
      }

#endif

      float nitsInPq = Csp::Trc::LinearTo::Pq(nits);

      [branch]
      if (nitsInPq <= RollOffStoppingPoint)
      {
        float3 rgbInPq1;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertCspTo::Hdr10(CO);

#endif

        rgbInPq1 = CO.RGB;

        float3 rgbInPq2 = BlackPointAdaption(rgbInPq1,
                                             OldBlackPoint,
                                             RollOffMinusOldBlackPoint,
                                             MinLum);

        CO.RGB = max(rgbInPq2, 0.f);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertCspTo::ScRgb(CO);

#endif
        return;
      }
      else
      [branch]
      if (!ProcessingDone)
      {
        discard;
      }
      else
      {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertCspTo::ScRgb(CO);

#endif

        return;
      }
    }
    break;
    // RGB mode
    case PRO_MODE_RGB:
    {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      BRANCH()
      if (CO.trc == TRC_PQ)
      {
        CO = CO::ConvertTrcTo::LinearNormalised(CO);
      }
      else
      {
        CO = CO::ConvertPrimariesTo::Bt2020(CO);
      }

#endif

      float nits = CO::GetLuminance::LinearNormalised(CO);

      [branch]
      if (nits <= RollOffStoppingPoint)
      {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertTrcTo::LinearNormalised(CO);
        CO = CO::ConvertPrimariesTo::Bt2020(CO);

#endif

        float3 rgb2 = BlackPointAdaption(CO.RGB,
                                         OldBlackPoint,
                                         RollOffMinusOldBlackPoint,
                                         MinLum);

        CO.RGB = max(rgb2, 0.f);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        CO = CO::ConvertCspTo::ScRgb(CO);

        return;

#endif
      }
      else
      [branch]
      if (!ProcessingDone)
      {
        discard;
      }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      else
      {
        CO = CO::ConvertCspTo::ScRgb(CO);

        return;
      }
#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)

      CO = CO::ConvertTrcTo::Pq(CO);

      return;

#endif
    }
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    break;
#endif
    default:
    {
      CO.RGB = 0.f;
      return;
    }
  }
}

#endif //is hdr API and hdr colour space
