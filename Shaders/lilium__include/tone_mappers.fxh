#pragma once

#include "hdr_analysis.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_POSSIBLE_HDR_CSP))


namespace Tmos
{
  // gamma
  static const float removeGamma = 2.4f;
  static const float applyGamma  = 1.f / removeGamma;

  // Rep. ITU-R BT.2446-1 Table 2 & 3
  void Bt2446A(
    inout float3 Colour,
          float  MaxNits,
          float  TargetNits,
          float  GamutCompression)
  {
    // adjust the max of 1 according to maxCLL
    Colour *= (10000.f / MaxNits);

    // non-linear transfer function RGB->R'G'B'
    Colour = pow(Colour, applyGamma);

#define ycbcr Colour
    //to Y'C'bC'r
    ycbcr = Csp::Ycbcr::RgbTo::YcbcrBt2020(Colour);

    // tone mapping step 1
    //pHDR
    float pHdr = 1.f + 32.f * pow(MaxNits /
                                  10000.f
                              , applyGamma);

    //Y'p
    float yP = (log(1.f + (pHdr - 1.f) * ycbcr.x)) /
               log(pHdr);

    // tone mapping step 2
    //Y'c
    float yC = yP <= 0.7399f                ? 1.0770f * yP
             : yP > 0.7399f && yP < 0.9909f ? (-1.1510f * (yP * yP)) + (2.7811f * yP) - 0.6302f
                                            : (0.5000f * yP) + 0.5000f;

    // tone mapping step 3
    //pSDR
    float pSdr = 1.f + 32.f * pow(
                                  TargetNits /
                                  10000.f
                              , applyGamma);

    //Y'SDR
    float ySdr = (pow(pSdr, yC) - 1.f) /
                 (pSdr - 1.f);

    //f(Y'SDR)
    float colourScaling = ySdr /
                          (GamutCompression * ycbcr.x);

    //C'b,tmo
    float cbTmo = colourScaling * ycbcr.y;

    //C'r,tmo
    float crTmo = colourScaling * ycbcr.z;

    //Y'tmo
    float yTmo = ySdr - max(0.1f * crTmo, 0.f);

    Colour = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yTmo,
                                                   cbTmo,
                                                   crTmo));

    // avoid invalid colours
    Colour = max(Colour, 0.f);

    // gamma decompression and adjust to TargetNits
    Colour = pow(Colour, removeGamma) * (TargetNits / 10000.f);
  }

  void Bt2446A_MOD1(
         inout float3 Colour,
               float  MaxNits,
               float  TargetNits,
               float  GamutCompression,
               float  TestH,
               float  TestS)
  {
    // adjust the max of 1 according to maxCLL
    Colour *= (10000.f / MaxNits);

    // non-linear transfer function RGB->R'G'B'
    Colour = pow(Colour, applyGamma);

    //to Y'C'bC'r
    ycbcr = Csp::Ycbcr::RgbTo::YcbcrBt2020(Colour);

    // tone mapping step 1
    //pHDR
    float pHdr = 1.f + 32.f * pow(
                                  TestH /
                                  10000.f
                              , applyGamma);

    //Y'p
    float y = (log(1.f + (pHdr - 1.f) * ycbcr.x)) /
              log(pHdr);

    // tone mapping step 2
    //Y'c
    y = y <= 0.7399f
      ? 1.0770f * y
      : y > 0.7399f && y < 0.9909f
      ? (-1.1510f * pow(y , 2)) + (2.7811f * y) - 0.6302f
      : (0.5000f * y) + 0.5000f;

    // tone mapping step 3
    //pSDR
    float pSdr = 1.f + 32.f * pow(
                                  TestS /
                                  10000.f
                              , applyGamma);

    //Y'SDR
    y = (pow(pSdr, y) - 1.f) /
        (pSdr - 1.f);

    //f(Y'SDR)
    float colourScaling = y /
                          (GamutCompression * ycbcr.x);

    //C'b,tmo
    float cbTmo = colourScaling * ycbcr.y;

    //C'r,tmo
    float crTmo = colourScaling * ycbcr.z;

    //Y'tmo
    float yTmo = y - max(0.1f * crTmo, 0.f);

    Colour = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yTmo,
                                              cbTmo,
                                              crTmo));

    // avoid invalid colours
    Colour = max(Colour, 0.f);

    // gamma decompression and adjust to TargetNits
    Colour = pow(Colour, removeGamma) * (TargetNits / 10000.f);
  }

  namespace Bt2390
  {

    void HermiteSpline(
           inout float E1,
                 float KneeStart,
                 float MaxLum)
    {
      float oneMinusKneeStart = 1.f - KneeStart;
      float t = (E1 - KneeStart) / oneMinusKneeStart;
      float tPow2 = pow(t, 2.f);
      float tPow3 = pow(t, 3.f);
      //float tPow2 = t >= 0.f ?  pow( t, 2.f)
      //                       : -pow(-t, 2.f);
      //float tPow3 = t >= 0.f ?  pow( t, 3.f)
      //                       : -pow(-t, 3.f);

      E1 = ( 2.f * tPow3 - 3.f * tPow2 + 1.f) * KneeStart
         + (       tPow3 - 2.f * tPow2 + t)   * oneMinusKneeStart
         + (-2.f * tPow3 + 3.f * tPow2)       * MaxLum;
    }

#define BT2390_PRO_MODE_ICTCP 0
#define BT2390_PRO_MODE_YCBCR 1
#define BT2390_PRO_MODE_YRGB  2
#define BT2390_PRO_MODE_RGB   3

    // works in PQ
    void Eetf(
           inout float3 Colour,
                 uint   ProcessingMode,
                 float  SrcMinPq, // Lb in PQ
                 float  SrcMaxPq, // Lw in PQ
                 float  SrcMaxPqMinusSrcMinPq, // (Lw in PQ) minus (Lb in PQ)
                 float  MinLum,   // minLum
                 float  MaxLum,   // maxLum
                 float  KneeStart // KS
    )
    {
      if (ProcessingMode == BT2390_PRO_MODE_ICTCP)
      {
        //to L'M'S'
        Colour = Csp::Trc::LinearTo::Pq(Csp::Ictcp::Mat::Bt2020To::Lms(Colour));

        float i1 = 0.5f * Colour.x + 0.5f * Colour.y;
        //E1
        float i2 = (i1 - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float i2 = i1 / SrcMaxPq;

        //E2
        if (i2 >= KneeStart)
        {
          HermiteSpline(i2, KneeStart, MaxLum);
        }
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
        else if (MinLum == 0.f)
        {
          discard;
        }
#endif

        //E3
        i2 += MinLum * pow((1.f - i2), 4.f);

        //E4
        i2 = i2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //i2 *= SrcMaxPq;

        float minI = max(min((i1 / i2), (i2 / i1)), 0.f); // max to avoid invalid colours

        //to L'M'S'
        Colour = Csp::Ictcp::Mat::IctcpTo::PqLms(
                   float3(i2,
                          dot(Colour, PqLmsToIctcp[1]) * minI,
                          dot(Colour, PqLmsToIctcp[2]) * minI));

        //to LMS
        Colour = Csp::Trc::PqTo::Linear(Colour);
        //to RGB
        Colour = max(Csp::Ictcp::Mat::LmsTo::Bt2020(Colour), 0.f);

      }
      else if (ProcessingMode == BT2390_PRO_MODE_YCBCR)
      {
        float y1 = dot(Colour, KBt2020);
        //E1
        float y2 = (y1 - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float y2 = y1 / SrcMaxPq;

        //E2
        if (y2 >= KneeStart)
        {
          HermiteSpline(y2, KneeStart, MaxLum);
        }
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
        else if (MinLum == 0.f)
        {
          discard;
        }
#endif

        //E3
        y2 += MinLum * pow((1.f - y2), 4.f);

        //E4
        y2 = max(y2 * SrcMaxPqMinusSrcMinPq + SrcMinPq, 0.f); // max to avoid invalid colours
        //y2 *= SrcMaxPq;

        float minY = min((y1 / y2), (y2 / y1));

        Colour = max(
                   Csp::Ycbcr::YcbcrTo::RgbBt2020(
                     float3(y2,
                            (Colour.b - y1) / KbBt2020 * minY,
                            (Colour.r - y1) / KrBt2020 * minY))
                 , 0.f);

      }
      else if (ProcessingMode == BT2390_PRO_MODE_YRGB)
      {
        float y1 = dot(Colour, Csp::Mat::Bt2020ToXYZ[1].rgb);
        //E1
        float y2 = (Csp::Trc::LinearTo::Pq(y1) - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float y2 = Csp::Trc::LinearTo::Pq(y1) / SrcMaxPq;

        //E2
        if (y2 >= KneeStart)
        {
          HermiteSpline(y2, KneeStart, MaxLum);
        }
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
        else if (MinLum == 0.f)
        {
          discard;
        }
#endif

        //E3
        y2 += MinLum * pow((1.f - y2), 4.f);

        //E4
        y2 = y2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //y2 *= SrcMaxPq;

        y2 = Csp::Trc::PqTo::Linear(y2);

        Colour = max(y2 / y1 * Colour, 0.f);

      }
      else // if (ProcessingMode == BT2390_PRO_MODE_RGB)
      {
        //E1
        Colour = (Colour - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //Colour /= SrcMaxPq;

        //E2
        if (Colour.r >= KneeStart)
        {
          HermiteSpline(Colour.r, KneeStart, MaxLum);
        }
        if (Colour.g >= KneeStart)
        {
          HermiteSpline(Colour.g, KneeStart, MaxLum);
        }
        if (Colour.b >= KneeStart)
        {
          HermiteSpline(Colour.b, KneeStart, MaxLum);
        }

        //E3
        Colour += MinLum * pow((1.f - Colour), 4.f);

        //E4
        Colour = max(Colour * SrcMaxPqMinusSrcMinPq + SrcMinPq, 0.f);
        //Colour *= SrcMaxPq;
      }
    }
  }


  namespace Dice
  {
#define DICE_PRO_MODE_ICTCP 0
#define DICE_PRO_MODE_YCBCR 1

#define DICE_WORKING_COLOUR_SPACE_BT2020  0
#define DICE_WORKING_COLOUR_SPACE_AP0_D65 1

    // Applies exponential "Photographic" luminance compression
    float RangeCompress(float x)
    {
      return 1.f - exp(-x);
    }

    float LuminanceCompress(
            float Channel,
            float TargetCllInPq,
            float ShoulderStartInPq)
    {
      return (TargetCllInPq - ShoulderStartInPq)
           * RangeCompress((Channel       - ShoulderStartInPq) /
                           (TargetCllInPq - ShoulderStartInPq))
           + ShoulderStartInPq;

//      return Channel < ShoulderStartInPq
//           ? Channel
//           : (TargetNits - ShoulderStart)
//           * RangeCompress((Channel       - ShoulderStartInPq) /
//                           (TargetCllInPq - ShoulderStartInPq))
//           + ShoulderStartInPq;
    }

    // remap from infinite
    // ShoulderStart denotes the point where we change from linear to shoulder
    void ToneMapper(
           inout float3 Colour,
                 uint   ProcessingMode,
                 float  TargetCllInPq,
                 float  ShoulderStartInPq)
    {

    // why does this not work?!
    //  float3x3 RgbToLms = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                    ? RGB_AP0_D65_To_LMS
    //                    : RGB_BT2020_To_LMS;
    //
    //  float3x3 LmsToRgb = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                    ? LmsToRgb_AP0_D65
    //                    : LmsToRgb_BT2020;
    //
    //  float3 KFactors = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                  ? K_AP0_D65
    //                  : K_BT2020;
    //
    //  float  KbHelper = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                  ? KB_AP0_D65_HELPER
    //                  : KB_BT2020_HELPER;
    //  float  KrHelper = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                  ? KR_AP0_D65_HELPER
    //                  : KR_BT2020_HELPER;
    //  float2 KgHelper = WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
    //                  ? KG_AP0_D65_HELPER
    //                  : KG_BT2020_HELPER;

//      float3x3 RgbToLms = Csp::Ictcp::Mat::Bt2020ToLms;
//      float3x3 LmsToRgb = Csp::Ictcp::Mat::LmsToBt2020;
//      float3   KFactors = Csp::KHelpers::Bt2020::K;
//      float    KbHelper = Csp::KHelpers::Bt2020::Kb;
//      float    KrHelper = Csp::KHelpers::Bt2020::Kr;
//      float2   KgHelper = Csp::KHelpers::Bt2020::Kg;
//
//      if (WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
//      {
//        RgbToLms = Csp::Ictcp::Mat::Ap0D65ToLms;
//        LmsToRgb = Csp::Ictcp::Mat::LmsToAp0D65;
//        KFactors = Csp::KHelpers::Ap0D65::K;
//        KbHelper = Csp::KHelpers::Ap0D65::Kb;
//        KrHelper = Csp::KHelpers::Ap0D65::Kr;
//        KgHelper = Csp::KHelpers::Ap0D65::Kg;
//      }

      // ICtCp and YCbCr method copied from BT.2390
      if (ProcessingMode == DICE_PRO_MODE_ICTCP)
      {
        //to L'M'S'
        Colour = Csp::Trc::LinearTo::Pq(Csp::Ictcp::Mat::Bt2020To::Lms(Colour));

//        //to L'M'S'
//        Colour = Csp::Trc::LinearTo::Pq(mul(RgbToLms, Colour));

        //Intensity
        float i1 = 0.5f * Colour.x + 0.5f * Colour.y;

        if (i1 < ShoulderStartInPq)
        {
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
          discard;
#endif
        }
        else
        {
          float i2 = LuminanceCompress(i1, TargetCllInPq, ShoulderStartInPq);

          float minI = min(min((i1 / i2), (i2 / i1)) * 1.1f, 1.f);

          //to L'M'S'
          Colour = Csp::Ictcp::Mat::IctcpTo::PqLms(
                    float3(i2,
                           dot(Colour, PqLmsToIctcp[1]) * minI,
                           dot(Colour, PqLmsToIctcp[2]) * minI));

          //to LMS
          Colour = Csp::Trc::PqTo::Linear(Colour);
          //to RGB
          Colour = max(Csp::Ictcp::Mat::LmsTo::Bt2020(Colour), 0.f);

//          //to RGB
//          Colour = max(mul(LmsToRgb, Colour), 0.f);
        }
      }
      else
      {
        float y1 = dot(Colour, KBt2020);

        //float y1 = dot(Colour, KFactors);

        if (y1 < ShoulderStartInPq)
        {
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
          discard;
#endif
        }
        else
        {
          float y2 = LuminanceCompress(y1, TargetCllInPq, ShoulderStartInPq);

          float minY = min(min((y1 / y2), (y2 / y1)) * 1.1f, 1.f);

          //to RGB
          Colour = max(
                     Csp::Ycbcr::YcbcrTo::RgbBt2020(
                       float3(y2,
                              (Colour.b - y1) / KbBt2020 * minY,
                              (Colour.r - y1) / KrBt2020 * minY))
                   , 0.f);

//          float cb2 = (Colour.b - y1) / KbHelper * minY;
//          float cr2 = (Colour.r - y1) / KrHelper * minY;
//
//          return max(float3(y2 + KrHelper    * cr2,
//                            y2 - KgHelper[0] * cb2 - KgHelper[1] * cr2,
//                            y2 + KbHelper    * cb2), 0.f);
        }
      }

//      return float3(LuminanceCompress(Colour.r, TargetCllInPq, ShoulderStartInPq),
//                    LuminanceCompress(Colour.g, TargetCllInPq, ShoulderStartInPq),
//                    LuminanceCompress(Colour.b, TargetCllInPq, ShoulderStartInPq));
    }
  }

}

#endif //is hdr API and hdr colour space
