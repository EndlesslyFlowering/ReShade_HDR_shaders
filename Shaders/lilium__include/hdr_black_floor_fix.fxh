#pragma once

#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "colour_space.fxh"


namespace Ui
{
  namespace HdrBlackFloorFix
  {
    namespace Gamma22Emu
    {
      uniform bool EnableGamma22Emu
      <
        ui_category = "black floor gamma 2.2 emulation";
        ui_label    = "enable gamma 2.2 black floor emulation";
      > = false;

      uniform float WhitePoint
      <
        ui_category = "black floor gamma 2.2 emulation";
        ui_label     = "white point";
        ui_type      = "drag";
        ui_drag_desc = " nits";
        ui_min       = 10.f;
        ui_max       = 250.f;
        ui_step      = 0.5f;
      > = 80.f;
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

      uniform float SourceBlackPoint
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
        ui_type      = "drag";
        ui_units     = " nits";
        ui_min       = 1.f;
        ui_max       = 20.f;
        ui_step      = 0.01f;
      > = 10.f;

      uniform float TargetBlackPoint
      <
        ui_category  = "black floor lowering";
        ui_label     = "new black point";
        ui_type      = "drag";
        ui_units     = " nits";
        ui_min       = -0.1f;
        ui_max       = 0.1f;
        ui_step      = 0.0001f;
      > = 0.f;
    }
  }
}


void Gamma22Emulation(
  inout float3 Rgb,
  const float  WhitePointNormalised)
{
  Rgb = float3(pow(Csp::Trc::ToSrgb(Rgb.r / WhitePointNormalised), 2.2f) * WhitePointNormalised,
               pow(Csp::Trc::ToSrgb(Rgb.g / WhitePointNormalised), 2.2f) * WhitePointNormalised,
               pow(Csp::Trc::ToSrgb(Rgb.b / WhitePointNormalised), 2.2f) * WhitePointNormalised);
}


float3 LowerBlackFloor(
  const float3 Input,
  const float  RollOffStoppingPoint,
  const float  SourceBlackPoint,
  const float  RollOffMinusOldBlackPoint,
  const float  MinLum)
{
  // ICtCp mode
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_ICTCP)
  {
    //to L'M'S'
    float3 Lms = Csp::Trc::ToPq(Csp::Ictcp::Mat::Bt2020To::Lms(Input));

    const float i1 = 0.5f * Lms.x + 0.5f * Lms.y;

    if (i1 <= RollOffStoppingPoint)
    {
      //E1
      float i2 = (i1 - SourceBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      i2 += MinLum * pow((1.f - i2), 4.f);

      //E4
      i2 = i2 * RollOffMinusOldBlackPoint + SourceBlackPoint;

      //to L'M'S'
      Lms = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(i2,
                                                   dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[1]),
                                                   dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[2])));
      //to LMS
      Lms = Csp::Trc::FromPq(Lms);
      //to RGB
      return clamp(Csp::Ictcp::Mat::LmsTo::Bt2020(Lms), 0.f, 65504.f);
    }
    else
    {
      return Input;
    }
  }
  // YCbCr mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YCBCR)
  {
    const float3 InputInPq = Csp::Trc::ToPq(Input);

    const float y1 = dot(InputInPq, Csp::KHelpers::Bt2020::K);

    if (y1 <= RollOffStoppingPoint)
    {  
      //E1
      float y2 = (y1 - SourceBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      y2 += MinLum * pow((1.f - y2), 4.f);
  
      //E4
      y2 = y2 * RollOffMinusOldBlackPoint + SourceBlackPoint;
  
      return Csp::Trc::FromPq(
        clamp(
          Csp::Ycbcr::ToRgb::Bt2020(
            float3(y2,
                   (InputInPq.b - y1) / Csp::KHelpers::Bt2020::Kb,
                   (InputInPq.r - y1) / Csp::KHelpers::Bt2020::Kr))
        , 0.f, 65504.f));
    }
    else
    {
      return Input;
    }
  }
  // YRGB mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YRGB)
  {
    const float y1     = dot(Input, Csp::Mat::Bt2020ToXYZ[1].rgb);
    const float y1InPq = Csp::Trc::ToPq(y1);

    if (y1InPq <= RollOffStoppingPoint)
    {
      //E1
      float y2 = (y1InPq - SourceBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      y2 += MinLum * pow((1.f - y2), 4.f);

      //E4
      y2 = y2 * RollOffMinusOldBlackPoint + SourceBlackPoint;

      y2 = Csp::Trc::FromPq(y2);

      return clamp(y2 / y1 * Input, 0.f, 65504.f);
    }
    else
    {
      return Input;
    }
  }
  // RGB in PQ mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB_IN_PQ)
  {
    if (Csp::Trc::ToPq(dot(Input, Csp::Mat::Bt2020ToXYZ[1])) <= RollOffStoppingPoint)
    {
      const float3 InputInPq = Csp::Trc::ToPq(Input);

      //E1
      float3 rgb = (InputInPq - SourceBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      rgb += MinLum * pow((1.f - rgb), 4.f);

      //E4
      rgb = rgb * RollOffMinusOldBlackPoint + SourceBlackPoint;

      return Csp::Trc::FromPq(clamp(rgb, 0.f, 65504.f));
    }
    else
    {
      return Input;
    }
  }
  // RBG mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB)
  {
    if (dot(Input, Csp::Mat::Bt2020ToXYZ[1]) <= RollOffStoppingPoint)
    {
      //E1
      float3 rgb = (Input - SourceBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      rgb += MinLum * pow((1.f - rgb), 4.f);

      //E4
      rgb = rgb * RollOffMinusOldBlackPoint + SourceBlackPoint;

      return clamp(rgb, 0.f, 65504.f);
    }
    else
    {
      return Input;
    }
  }
  else
  {
    return float3(0.f, 0.f, 0.f);
  }

}

#endif
