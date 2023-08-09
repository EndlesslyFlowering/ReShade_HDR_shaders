#pragma once

#include "colour_space.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


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

      uniform float WhitePoint
      <
        ui_category = "SDR black floor emulation";
        ui_label    = "processing cut off";
        ui_tooltip  = "How much of the lower range range should be processed.";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 40.f;
        ui_max      = 250.f;
        ui_step     = 0.5f;
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
        ui_step      = 0.0001f;
      > = 0.f;
    }
  }
}


void Gamma22Emulation(
  inout float3 Rgb,
        float  WhitePointNormalised)
{
  if (dot(Csp::Mat::Bt2020ToXYZ[1], Rgb) <= WhitePointNormalised)
  {
    Rgb = pow(Csp::Trc::ToSrgb(Rgb / WhitePointNormalised), 2.2f) * WhitePointNormalised;
    return;
  }

//  float3 RgbNormalised = Rgb / WhitePointNormalised;
//
//  if (Rgb.r <= WhitePointNormalised)
//  {
//    Rgb.r = pow(Csp::Trc::ToSrgb(RgbNormalised.r), 2.2f) * WhitePointNormalised;
//  }
//  if (Rgb.g <= WhitePointNormalised)
//  {
//    Rgb.g = pow(Csp::Trc::ToSrgb(RgbNormalised.g), 2.2f) * WhitePointNormalised;
//  }
//  if (Rgb.b <= WhitePointNormalised)
//  {
//    Rgb.b = pow(Csp::Trc::ToSrgb(RgbNormalised.b), 2.2f) * WhitePointNormalised;
//  }
}


float3 LowerBlackFloor(
  float3 Input,
  float  RollOffStoppingPoint,
  float  OldBlackPoint,
  float  RollOffMinusOldBlackPoint,
  float  MinLum)
{
  // ICtCp mode
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_ICTCP)
  {
    //to L'M'S'
    float3 Lms = Csp::Trc::ToPq(Csp::Ictcp::Mat::Bt2020To::Lms(Input));

    float i1 = 0.5f * Lms.x + 0.5f * Lms.y;

    if (i1 <= RollOffStoppingPoint)
    {
      //E1
      float i2 = (i1 - OldBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      i2 += MinLum * pow((1.f - i2), 4.f);

      //E4
      i2 = i2 * RollOffMinusOldBlackPoint + OldBlackPoint;

      //to L'M'S'
      Lms = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(i2,
                                                   dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[1]),
                                                   dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[2])));
      //to LMS
      Lms = Csp::Trc::FromPq(Lms);
      //to RGB
      return max(Csp::Ictcp::Mat::LmsTo::Bt2020(Lms), 0.f);
    }
    discard;
  }
  // YCbCr mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YCBCR)
  {
    float3 inputInPq = Csp::Trc::ToPq(Input);

    float y1 = dot(inputInPq, Csp::KHelpers::Bt2020::K);

    if (y1 <= RollOffStoppingPoint)
    {
      //E1
      float y2 = (y1 - OldBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      y2 += MinLum * pow((1.f - y2), 4.f);

      //E4
      y2 = y2 * RollOffMinusOldBlackPoint + OldBlackPoint;

      return Csp::Trc::FromPq(
        max(
          Csp::Ycbcr::ToRgb::Bt2020(
            float3(y2,
                   (inputInPq.b - y1) / Csp::KHelpers::Bt2020::Kb,
                   (inputInPq.r - y1) / Csp::KHelpers::Bt2020::Kr))
        , 0.f));
    }
    discard;
  }
  // YRGB mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_YRGB)
  {
    float y1     = dot(Input, Csp::Mat::Bt2020ToXYZ[1].rgb);
    float y1InPq = Csp::Trc::ToPq(y1);

    if (y1InPq <= RollOffStoppingPoint)
    {
      //E1
      float y2 = (y1InPq - OldBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      y2 += MinLum * pow((1.f - y2), 4.f);

      //E4
      y2 = y2 * RollOffMinusOldBlackPoint + OldBlackPoint;

      y2 = Csp::Trc::FromPq(y2);

      return max(y2 / y1 * Input, 0.f);
    }
    discard;
  }
  // RGB in PQ mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB_IN_PQ)
  {
    if (Csp::Trc::ToPq(dot(Input, Csp::Mat::Bt2020ToXYZ[1])) <= RollOffStoppingPoint)
    {
      float3 inputInPq = Csp::Trc::ToPq(Input);

      //E1
      float3 rgb = (inputInPq - OldBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      rgb += MinLum * pow((1.f - rgb), 4.f);

      //E4
      rgb = rgb * RollOffMinusOldBlackPoint + OldBlackPoint;

      return Csp::Trc::FromPq(max(rgb, 0.f));
    }
    discard;
  }
  // RBG mode
  else if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode == PRO_MODE_RGB)
  {
    if (dot(Input, Csp::Mat::Bt2020ToXYZ[1]) <= RollOffStoppingPoint)
    {
      //E1
      float3 rgb = (Input - OldBlackPoint) / RollOffMinusOldBlackPoint;

      //E3
      rgb += MinLum * pow((1.f - rgb), 4.f);

      //E4
      rgb = rgb * RollOffMinusOldBlackPoint + OldBlackPoint;

      return max(rgb, 0.f);
    }
    discard;
  }
  else
  {
    return float3(0.f, 0.f, 0.f);
  }

}

#endif //is hdr API and hdr colour space
