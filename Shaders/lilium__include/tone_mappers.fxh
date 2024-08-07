#pragma once

#include "colour_space.fxh"


#if (defined(IS_COMPUTE_CAPABLE_API) \
  && defined(IS_HDR_CSP))



namespace Tmos
{
  // Rep. ITU-R BT.2446-1 Table 2 & 3
  void Bt2446A(
    inout       float3 Colour,
          const float  MaxNits,
          const float  TargetNits)
  {
    //pSDR and pHDR
    static const float2 pSdrHdr = 1.f + 32.f * pow(float2(TargetNits, MaxNits) / 10000.f, 1.f / 2.4f);
#define pSdr pSdrHdr[0]
#define pHdr pSdrHdr[1]

    //scRGB
    Colour = ConditionallyNormaliseScRgb(Colour);
    //HDR10
    Colour = ConditionallyLineariseHdr10(Colour);

    // adjust the max of 1 according to max nits
    Colour *= (10000.f / MaxNits);

    // get luminance
    float yHdr = GetLuminance(Colour);
    //clamp to avoid invalid numbers
    yHdr = max(yHdr, 1e-20);
    //Y'HDR
    yHdr = pow(yHdr, 1.f / 2.4f);

    //Y'p
    const float yP = log(1.f + (pHdr - 1.f) * yHdr)
                   / log(pHdr);

    // tone mapping step 2
    //Y'c
    const float yC = yP <= 0.7399f ? 1.0770f * yP
                   : yP >= 0.9909f ? (0.5000f * yP) + 0.5000f
                                   : (-1.1510f * (yP * yP)) + (2.7811f * yP) - 0.6302f;

    //Y'SDR
    const float ySdr = (pow(pSdr, yC) - 1.f)
                     / (pSdr - 1.f);

    const float2 ySdrYHdr = pow(float2(ySdr, yHdr), 2.4f);

    Colour *= (ySdrYHdr[0] / ySdrYHdr[1]);

    // adjust to TargetNits
    Colour *= (TargetNits / 10000.f);

    //scRGB
    Colour = ConditionallyConvertNormalisedBt709ToScRgb(Colour);
    //HDR10
    Colour = ConditionallyConvertNormalisedBt2020ToHdr10(Colour);

    return;
#undef pSdr
#undef pHdr
  }

  float3 Bt2446A_MOD1(
    inout       float3 Colour,
          const float  MaxNits,
          const float  TargetNits,
          const float  LumaPostAdjust,
          const float  GamutCompression,
          const float  TestH,
          const float  TestS)
  {
    float3 Rgb = Colour;

    //scRGB
    Rgb = ConditionallyNormaliseScRgb(Rgb);
    //HDR10
    Rgb = ConditionallyLineariseHdr10(Rgb);

    // adjust the max of 1 according to maxCLL
    Rgb *= (10000.f / MaxNits);

    // non-linear transfer function RGB->R'G'B'
    Rgb = pow(Rgb, 1.f / 2.4f);

    //to Y'C'bC'r
    float3 ycbcr = Csp::Ycbcr::RgbTo::YcbcrBt2020(Rgb);

    // tone mapping step 1
    //pHDR
    float pHdr = 1.f + 32.f * pow(
                                  TestH /
                                  10000.f
                              , 1.f / 2.4f);

    //Y'p
    float yP = (log(1.f + (pHdr - 1.f) * ycbcr.x)) /
                log(pHdr);

    // tone mapping step 2
    //Y'c
    float yC = yP <= 0.7399f ? 1.0770f * yP
             : yP >= 0.9909f ? (0.5000f * yP) + 0.5000f
                             : (-1.1510f * (yP * yP)) + (2.7811f * yP) - 0.6302f;

    // tone mapping step 3
    //pSDR
    float pSdr = 1.f + 32.f * pow(
                                  TestS /
                                  10000.f
                              , 1.f / 2.4f);

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
    float yTmo = ySdr - max(LumaPostAdjust * crTmo, 0.f);

    Rgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yTmo,
                                                cbTmo,
                                                crTmo));

    // avoid invalid colours
    Rgb = max(Rgb, 0.f);

    // gamma decompression and adjust to TargetNits
    Rgb = pow(Rgb, 2.4f) * (TargetNits / 10000.f);

    //scRGB
    Rgb = ConditionallyConvertNormalisedBt2020ToScRgb(Rgb);
    //HDR10
    Rgb = ConditionallyConvertNormalisedBt2020ToHdr10(Rgb);

    Colour = Rgb;
  }

  namespace Bt2390
  {

    float HermiteSpline(
      const float E1,
      const float KneeStart,
      const float MaxLum)
    {
      float oneMinusKneeStart = 1.f - KneeStart;
      float t = (E1 - KneeStart) / oneMinusKneeStart;
      float tPow2 = t * t;
      float tPow3 = tPow2 * t;
      //float tPow2 = t >= 0.f ?  pow( t, 2.f)
      //                       : -pow(-t, 2.f);
      //float tPow3 = t >= 0.f ?  pow( t, 3.f)
      //                       : -pow(-t, 3.f);

      return ( 2.f * tPow3 - 3.f * tPow2 + 1.f) * KneeStart
           + (       tPow3 - 2.f * tPow2 + t)   * oneMinusKneeStart
           + (-2.f * tPow3 + 3.f * tPow2)       * MaxLum;
    }

#define BT2390_PRO_MODE_ICTCP 0
#define BT2390_PRO_MODE_YRGB  1
#define BT2390_PRO_MODE_RGB   2

    // works in PQ
    void Eetf(
      inout       float3 Colour,
            const uint   ProcessingMode,
            const float  SrcMinPq,  // Lb in PQ
            const float  SrcMaxPq,  // Lw in PQ
            const float  SrcMaxPqMinusSrcMinPq, // (Lw in PQ) minus (Lb in PQ)
            const float  MinLum,    // minLum
            const float  MaxLum,    // maxLum
            const float  KneeStart) // KS
    {
      BRANCH(x)
      if (ProcessingMode == BT2390_PRO_MODE_ICTCP)
      {
        float3 Rgb = ConditionallyLineariseHdr10(Colour);

        //to L'M'S'
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float3 pqLms = Csp::Ictcp::ScRgbTo::PqLms(Rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float3 pqLms = Csp::Ictcp::Bt2020To::PqLms(Rgb);
#endif

        float i1 = 0.5f * pqLms.x + 0.5f * pqLms.y;
        //E1
        float i2 = (i1 - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float i2 = i1 / SrcMaxPq;

        //E2
        [branch]
        if (i2 >= KneeStart)
        {
          i2 = HermiteSpline(i2, KneeStart, MaxLum);
        }
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
        else if (MinLum == 0.f)
        {
          discard;
        }
#endif

        //E3
        float e3;
        e3 = 1.f - i2;
        e3 = e3*e3*e3*e3;
        i2 += MinLum * e3;

        //E4
        i2 = i2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //i2 *= SrcMaxPq;

        float3 ictcp = float3(i2,
                              dot(pqLms, Csp::Ictcp::PqLmsToIctcp[1]),
                              dot(pqLms, Csp::Ictcp::PqLmsToIctcp[2]));

        //to RGB
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        Rgb = Csp::Ictcp::IctcpTo::ScRgb(ictcp);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        Rgb = Csp::Ictcp::IctcpTo::Bt2020(ictcp);
#endif

        //HDR10
        Rgb = ConditionallyConvertNormalisedBt2020ToHdr10(Rgb);

        Colour = Rgb;

      }
      else if (ProcessingMode == BT2390_PRO_MODE_YRGB)
      {
        //HDR10
        float3 Rgb = ConditionallyLineariseHdr10(Colour);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float y1 = dot(Rgb, Csp::Mat::ScRgbToXYZ[1]);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float y1 = dot(Rgb, Csp::Mat::Bt2020ToXYZ[1]);
#endif
        //E1
        float y2 = (Csp::Trc::LinearTo::Pq(y1) - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float y2 = Csp::Trc::LinearTo::Pq(y1) / SrcMaxPq;

        //E2
        [branch]
        if (y2 >= KneeStart)
        {
          y2 = HermiteSpline(y2, KneeStart, MaxLum);
        }
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
        else if (MinLum == 0.f)
        {
          discard;
        }
#endif

        //E3
        float e3;
        e3 = 1.f - y2;
        e3 = e3*e3*e3*e3;
        y2 += MinLum * e3;

        //E4
        y2 = y2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //y2 *= SrcMaxPq;

        y2 = Csp::Trc::PqTo::Linear(y2);

        Rgb = y2 / y1 * Rgb;

        //HDR10
        Rgb = ConditionallyConvertNormalisedBt2020ToHdr10(Rgb);

        Colour = Rgb;

      }
      else // if (ProcessingMode == BT2390_PRO_MODE_RGB)
      {
        float3 Rgb = ConditionallyConvertScRgbToHdr10(Colour);

        //E1
        Rgb = (Rgb - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //Rgb /= SrcMaxPq;

        //E2
        [branch]
        if (Rgb.r >= KneeStart)
        {
          Rgb.r = HermiteSpline(Rgb.r, KneeStart, MaxLum);
        }
        [branch]
        if (Rgb.g >= KneeStart)
        {
          Rgb.g = HermiteSpline(Rgb.g, KneeStart, MaxLum);
        }
        [branch]
        if (Rgb.b >= KneeStart)
        {
          Rgb.b = HermiteSpline(Rgb.b, KneeStart, MaxLum);
        }

        //E3
        Rgb += MinLum * pow((1.f - Rgb), 4.f);

        //E4
        Rgb = Rgb * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //Rgb *= SrcMaxPq;

        Rgb = ConditionallyConvertHdr10ToScRgb(Rgb);

        Colour = Rgb;
      }
    }
  }


  namespace Dice
  {
#define DICE_PRO_MODE_ICTCP 0
#define DICE_PRO_MODE_YRGB  1

#define DICE_WORKING_COLOUR_SPACE_BT2020  0
#define DICE_WORKING_COLOUR_SPACE_AP0_D65 1

    // Applies exponential "Photographic" luminance compression
    float RangeCompress(float x)
    {
      return 1.f - exp(-x);
    }

    float LuminanceCompress(
      const float Channel,
      const float ShoulderStartInPq,
      const float TargetLuminanceInPqMinusShoulderStartInPq)
    {
      return RangeCompress((Channel - ShoulderStartInPq)
                         / TargetLuminanceInPqMinusShoulderStartInPq)
           * TargetLuminanceInPqMinusShoulderStartInPq
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
      inout       float3 Colour,
            const uint   ProcessingMode,
            const float  ShoulderStartInPq,
            const float  TargetLuminanceInPqMinusShoulderStartInPq)
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

//      float3x3 RgbToLms = Csp::Ictcp::Bt2020ToLms;
//      float3x3 LmsToRgb = Csp::Ictcp::LmsToBt2020;
//      float3   KFactors = Csp::KHelpers::Bt2020::K;
//      float    KbHelper = Csp::KHelpers::Bt2020::Kb;
//      float    KrHelper = Csp::KHelpers::Bt2020::Kr;
//      float2   KgHelper = Csp::KHelpers::Bt2020::Kg;
//
//      if (WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
//      {
//        RgbToLms = Csp::Ictcp::Ap0D65ToLms;
//        LmsToRgb = Csp::Ictcp::LmsToAp0D65;
//        KFactors = Csp::KHelpers::Ap0D65::K;
//        KbHelper = Csp::KHelpers::Ap0D65::Kb;
//        KrHelper = Csp::KHelpers::Ap0D65::Kr;
//        KgHelper = Csp::KHelpers::Ap0D65::Kg;
//      }

      // ICtCp and YRGB method copied from BT.2390
      BRANCH(x)
      if (ProcessingMode == DICE_PRO_MODE_ICTCP)
      {
        //HDR10
        float3 Rgb = ConditionallyLineariseHdr10(Colour);

        //to L'M'S'
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float3 pqLms = Csp::Ictcp::ScRgbTo::PqLms(Rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float3 pqLms = Csp::Ictcp::Bt2020To::PqLms(Rgb);
#endif

        //Intensity
        float i1 = 0.5f * pqLms.x + 0.5f * pqLms.y;

        if (i1 < ShoulderStartInPq)
        {
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
          discard;
#endif
        }
        else
        {
          float i2 = LuminanceCompress(i1, ShoulderStartInPq, TargetLuminanceInPqMinusShoulderStartInPq);

          float3 ictcp = float3(i2,
                                dot(pqLms, Csp::Ictcp::PqLmsToIctcp[1]),
                                dot(pqLms, Csp::Ictcp::PqLmsToIctcp[2]));

          //to RGB
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
          Rgb = Csp::Ictcp::IctcpTo::ScRgb(ictcp);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
          Rgb = Csp::Ictcp::IctcpTo::Bt2020(ictcp);
#endif

          //HDR10
          Rgb = ConditionallyConvertNormalisedBt2020ToHdr10(Rgb);

          Colour = Rgb;
        }
      }
      else // if (ProcessingMode == DICE_PRO_MODE_YRGB)
      {
        //HDR10
        float3 Rgb = ConditionallyLineariseHdr10(Colour);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float y1 = dot(Rgb, Csp::Mat::ScRgbToXYZ[1]);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float y1 = dot(Rgb, Csp::Mat::Bt2020ToXYZ[1]);
#endif

        float y2 = Csp::Trc::LinearTo::Pq(y1);

        if (y2 < ShoulderStartInPq)
        {
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
          discard;
#endif
        }
        else
        {
          y2 = LuminanceCompress(y2,
                                 ShoulderStartInPq,
                                 TargetLuminanceInPqMinusShoulderStartInPq);

          y2 = Csp::Trc::PqTo::Linear(y2);

          Rgb = y2 / y1 * Rgb;

          Rgb = ConditionallyConvertNormalisedBt2020ToHdr10(Rgb);

          Colour = Rgb;
        }
      }

//      return float3(LuminanceCompress(Colour.r, TargetCllInPq, ShoulderStartInPq),
//                    LuminanceCompress(Colour.g, TargetCllInPq, ShoulderStartInPq),
//                    LuminanceCompress(Colour.b, TargetCllInPq, ShoulderStartInPq));
    }
  }

}

#endif //is hdr API and hdr colour space
