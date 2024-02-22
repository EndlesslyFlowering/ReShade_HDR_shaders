#pragma once

#include "hdr_analysis.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_HDR_CSP))

// normalise so that 10000 = 1
float3 ConditionallyNormaliseScrgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour /= 125.f;
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
float3 ConditionallyConvertScrgbToNormalisedBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = ConditionallyNormaliseScrgb(Colour);
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
// - convert linear to PQ
float3 ConditionallyConvertScrgbToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = ConditionallyConvertScrgbToNormalisedBt2020(Colour);
  Colour = Csp::Trc::LinearTo::Pq(Colour);
#endif
  return Colour;
}

// convert HDR10 to linear BT.2020
float3 ConditionallyLineariseHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::PqTo::Linear(Colour);
#endif
  return Colour;
}

// convert linear BT.2020 to HDR10
float3 ConditionallyConvertLinearBt2020ToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::LinearTo::Pq(Colour);
#endif
  return Colour;
}

// convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt709ToScrgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour *= 125.f;
#endif
  return Colour;
}

// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt2020ToScrgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  Colour = ConditionallyConvertNormalisedBt709ToScrgb(Colour);
#endif
  return Colour;
}

// - convert HDR10 to linear BT.2020
// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertHdr10ToScrgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Trc::PqTo::Linear(Colour);
  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  Colour = ConditionallyConvertNormalisedBt709ToScrgb(Colour);
#endif
  return Colour;
}


namespace Tmos
{
  // gamma
  static const float removeGamma = 2.4f;
  static const float applyGamma  = 1.f / removeGamma;

  // Rep. ITU-R BT.2446-1 Table 2 & 3
  void Bt2446A(
    inout       float3 Colour,
          const float  MaxNits,
          const float  TargetNits,
          const float  GamutCompression)
  {
    float3 Rgb = Colour;

    Rgb = ConditionallyConvertScrgbToNormalisedBt2020(Rgb);
    Rgb = ConditionallyLineariseHdr10(Rgb);

    // adjust the max of 1 according to maxCLL
    Rgb *= (10000.f / MaxNits);

    // non-linear transfer function RGB->R'G'B'
    Rgb = pow(Rgb, applyGamma);

    //to Y'C'bC'r
    float3 ycbcr = Csp::Ycbcr::RgbTo::YcbcrBt2020(Rgb);

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

    Rgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yTmo,
                                                cbTmo,
                                                crTmo));

    // avoid invalid colours
    Rgb = max(Rgb, 0.f);

    // gamma decompression and adjust to TargetNits
    Rgb = pow(Rgb, removeGamma) * (TargetNits / 10000.f);

    Rgb = ConditionallyConvertNormalisedBt2020ToScrgb(Rgb);
    Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);

    Colour = Rgb;
  }

  float3 Bt2446A_MOD1(
    inout       float3 Colour,
          const float  MaxNits,
          const float  TargetNits,
          const float  GamutCompression,
          const float  TestH,
          const float  TestS)
  {
    float3 Rgb = Colour;

    Rgb = ConditionallyLineariseHdr10(Rgb);
    Rgb = ConditionallyConvertScrgbToNormalisedBt2020(Rgb);

    // adjust the max of 1 according to maxCLL
    Rgb *= (10000.f / MaxNits);

    // non-linear transfer function RGB->R'G'B'
    Rgb = pow(Rgb, applyGamma);

    //to Y'C'bC'r
    float3 ycbcr = Csp::Ycbcr::RgbTo::YcbcrBt2020(Rgb);

    // tone mapping step 1
    //pHDR
    float pHdr = 1.f + 32.f * pow(
                                  TestH /
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
                                  TestS /
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

    Rgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yTmo,
                                                cbTmo,
                                                crTmo));

    // avoid invalid colours
    Rgb = max(Rgb, 0.f);

    // gamma decompression and adjust to TargetNits
    Rgb = pow(Rgb, removeGamma) * (TargetNits / 10000.f);

    Rgb = ConditionallyConvertNormalisedBt2020ToScrgb(Rgb);
    Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);

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
#define BT2390_PRO_MODE_YCBCR 1
#define BT2390_PRO_MODE_YRGB  2
#define BT2390_PRO_MODE_RGB   3

    // works in PQ
    void Eetf(
      inout       float3 Colour,
            const uint   ProcessingMode,
            const float  SrcMinPq,  // Lb in PQ
            const float  SrcMaxPq,  // Lw in PQ
            const float  SrcMaxPqMinusSrcMinPq, // (Lw in PQ) minus (Lb in PQ)
            const float  MinLum,    // minLum
            const float  MaxLum,    // maxLum
            const float  KneeStart, // KS
            const bool   EnableBlowingOutHighlights)
    {
      if (ProcessingMode == BT2390_PRO_MODE_ICTCP)
      {
        float3 Rgb = Colour;

        Rgb = ConditionallyNormaliseScrgb(Rgb);
        Rgb = ConditionallyLineariseHdr10(Rgb);

        //to L'M'S'
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float3 pqLms = Csp::Ictcp::Bt709To::PqLms(Rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float3 pqLms = Csp::Ictcp::Bt2020To::PqLms(Rgb);
#endif

        float i1 = 0.5f * pqLms.x + 0.5f * pqLms.y;
        //E1
        float i2 = (i1 - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float i2 = i1 / SrcMaxPq;

        //E2
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
        i2 += MinLum * pow((1.f - i2), 4.f);

        //E4
        i2 = i2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //i2 *= SrcMaxPq;

        float3 ictcp = float3(i2,
                              dot(pqLms, PqLmsToIctcp[1]),
                              dot(pqLms, PqLmsToIctcp[2]));

        if (EnableBlowingOutHighlights)
        {
          float minI = max(min((i1 / i2), (i2 / i1)), 0.f); // max to avoid invalid colours

          ictcp = float3(ictcp.x,
                         ictcp.yz * minI);
        }

        //to RGB
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        Rgb = Csp::Ictcp::IctcpTo::Bt709(ictcp);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        Rgb = Csp::Ictcp::IctcpTo::Bt2020(ictcp);
#endif

        Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);
        Rgb = ConditionallyConvertNormalisedBt709ToScrgb(Rgb);

        Colour = Rgb;

      }
      else if (ProcessingMode == BT2390_PRO_MODE_YCBCR)
      {
        float3 Rgb = ConditionallyConvertScrgbToHdr10(Colour);

        float y1 = dot(Rgb, KBt2020);
        //E1
        float y2 = (y1 - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float y2 = y1 / SrcMaxPq;

        //E2
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
        y2 += MinLum * pow((1.f - y2), 4.f);

        //E4
        y2 = max(y2 * SrcMaxPqMinusSrcMinPq + SrcMinPq, 0.f); // max to avoid invalid colours
        //y2 *= SrcMaxPq;

        float3 ycbcr = float3(y2,
                              (Rgb.b - y1) / KbBt2020,
                              (Rgb.r - y1) / KrBt2020);

        if (EnableBlowingOutHighlights)
        {
          float minY = min((y1 / y2), (y2 / y1));

          ycbcr = float3(ycbcr.x,
                         ycbcr.yz * minY);
        }

        Rgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(ycbcr);

        Rgb = ConditionallyConvertHdr10ToScrgb(Rgb);

        Colour = Rgb;

      }
      else if (ProcessingMode == BT2390_PRO_MODE_YRGB)
      {
        float3 Rgb = ConditionallyLineariseHdr10(Colour);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float y1 = dot(Rgb, Csp::Mat::Bt709ToXYZ[1].rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float y1 = dot(Rgb, Csp::Mat::Bt2020ToXYZ[1].rgb);
#endif
        //E1
        float y2 = (Csp::Trc::LinearTo::Pq(y1) - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //float y2 = Csp::Trc::LinearTo::Pq(y1) / SrcMaxPq;

        //E2
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
        y2 += MinLum * pow((1.f - y2), 4.f);

        //E4
        y2 = y2 * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //y2 *= SrcMaxPq;

        y2 = Csp::Trc::PqTo::Linear(y2);

        Rgb = y2 / y1 * Rgb;

        Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);

        Colour = Rgb;

      }
      else // if (ProcessingMode == BT2390_PRO_MODE_RGB)
      {
        float3 Rgb = ConditionallyConvertScrgbToHdr10(Colour);

        //E1
        Rgb = (Rgb - SrcMinPq) / SrcMaxPqMinusSrcMinPq;
        //Rgb /= SrcMaxPq;

        //E2
        if (Rgb.r >= KneeStart)
        {
          Rgb.r = HermiteSpline(Rgb.r, KneeStart, MaxLum);
        }
        if (Rgb.g >= KneeStart)
        {
          Rgb.g = HermiteSpline(Rgb.g, KneeStart, MaxLum);
        }
        if (Rgb.b >= KneeStart)
        {
          Rgb.b = HermiteSpline(Rgb.b, KneeStart, MaxLum);
        }

        //E3
        Rgb += MinLum * pow((1.f - Rgb), 4.f);

        //E4
        Rgb = Rgb * SrcMaxPqMinusSrcMinPq + SrcMinPq;
        //Rgb *= SrcMaxPq;

        Rgb = ConditionallyConvertHdr10ToScrgb(Rgb);

        Colour = Rgb;
      }
    }
  }


  namespace Dice
  {
#define DICE_PRO_MODE_ICTCP 0
#define DICE_PRO_MODE_YCBCR 1
#define DICE_PRO_MODE_YRGB  2

#define DICE_WORKING_COLOUR_SPACE_BT2020  0
#define DICE_WORKING_COLOUR_SPACE_AP0_D65 1

    // Applies exponential "Photographic" luminance compression
    float RangeCompress(float x)
    {
      return 1.f - exp(-x);
    }

    float LuminanceCompress(
      const float Channel,
      const float TargetCllInPq,
      const float ShoulderStartInPq)
    {
      return (TargetCllInPq - ShoulderStartInPq)
           * RangeCompress((Channel       - ShoulderStartInPq)
                         / (TargetCllInPq - ShoulderStartInPq))
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
            const float  TargetCllInPq,
            const float  ShoulderStartInPq,
            const bool   EnableBlowingOutHighlights)
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

      // ICtCp, YCbCr and YRGB method copied from BT.2390
      if (ProcessingMode == DICE_PRO_MODE_ICTCP)
      {
        float3 Rgb = Colour;

        Rgb = ConditionallyNormaliseScrgb(Rgb);
        Rgb = ConditionallyLineariseHdr10(Rgb);

        //to L'M'S'
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float3 pqLms = Csp::Ictcp::Bt709To::PqLms(Rgb);
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
          float i2 = LuminanceCompress(i1, TargetCllInPq, ShoulderStartInPq);

          float3 ictcp = float3(i2,
                                dot(pqLms, PqLmsToIctcp[1]),
                                dot(pqLms, PqLmsToIctcp[2]));

          if (EnableBlowingOutHighlights)
          {
            float minI = clamp(min((i1 / i2), (i2 / i1)) * 1.1f, 0.f, 1.f); // max to avoid invalid colours

            ictcp = float3(ictcp.x,
                           ictcp.yz * minI);
          }

          //to RGB
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
          Rgb = Csp::Ictcp::IctcpTo::Bt709(ictcp);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
          Rgb = Csp::Ictcp::IctcpTo::Bt2020(ictcp);
#endif

          Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);
          Rgb = ConditionallyConvertNormalisedBt709ToScrgb(Rgb);

          Colour = Rgb;
        }
      }
      else if (ProcessingMode == DICE_PRO_MODE_YCBCR)
      {
        float3 Rgb = ConditionallyConvertScrgbToHdr10(Colour);

        float y1 = dot(Rgb, KBt2020);

        if (y1 < ShoulderStartInPq)
        {
#if (SHOW_ADAPTIVE_MAX_NITS == NO)
          discard;
#endif
        }
        else
        {
          float y2 = LuminanceCompress(y1, TargetCllInPq, ShoulderStartInPq);

          float3 ycbcr = float3(y2,
                                (Colour.b - y1) / KbBt2020,
                                (Colour.r - y1) / KrBt2020);

          if (EnableBlowingOutHighlights)
          {
            float minY = clamp(min((y1 / y2), (y2 / y1)) * 1.1f, 0.f, 1.f); // max to avoid invalid colours

            ycbcr = float3(ycbcr.x,
                           ycbcr.yz * minY);
          }

          //to RGB
          Rgb = Csp::Ycbcr::YcbcrTo::RgbBt2020(ycbcr);

          Rgb = ConditionallyConvertHdr10ToScrgb(Rgb);

          Colour = Rgb;
        }
      }
      else // if (ProcessingMode == DICE_PRO_MODE_YRGB)
      {
        float3 Rgb = Colour;

        Rgb = ConditionallyLineariseHdr10(Rgb);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        float y1 = dot(Rgb, Csp::Mat::Bt709ToXYZ[1].rgb);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        float y1 = dot(Rgb, Csp::Mat::Bt2020ToXYZ[1].rgb);
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
          y2 = LuminanceCompress(y2, TargetCllInPq, ShoulderStartInPq);

          y2 = Csp::Trc::PqTo::Linear(y2);

          Rgb = y2 / y1 * Rgb;

          Rgb = ConditionallyConvertLinearBt2020ToHdr10(Rgb);

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
