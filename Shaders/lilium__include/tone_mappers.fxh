#pragma once

#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "hdr_analysis.fxh"


namespace ToneMapping
{

  // gamma
  static const float inverseGamma = 2.4f;
  static const float gamma        = 1.f / inverseGamma;

  // Rep. ITU-R BT.2446-1 Table 2 & 3
  float3 Bt2446a(
    const float3 Input,
    const float  TargetCll,
    const float  MaxCllIn,
    const float  GamutCompression)
  {
    float3 hdrIn = Input;

    const float maxCll = MaxCllIn > TargetCll
                       ? MaxCllIn
                       : TargetCll;

    // adjust the max of 1 according to maxCll
    hdrIn *= (10000.f / maxCll);

    // non-linear transfer function RGB->R'G'B'
    hdrIn = pow(hdrIn, gamma);

    //Y'C'bC'r
    const float3 ycbcr = Csp::Ycbcr::FromRgb::Bt2020(hdrIn);

    // tone mapping step 1
    const float pHdr = 1.f + 32.f * pow(
                                        maxCll /
                                        10000.f
                                    , gamma);

    //Y'p
    const float yP = (log(1.f + (pHdr - 1.f) * ycbcr.x)) /
                      log(pHdr);

    // tone mapping step 2
    //Y'c
    const float yC = yP <= 0.7399f
                   ? 1.0770f * yP
                   : yP > 0.7399f && yP < 0.9909f
                   ? (-1.1510f * pow(yP , 2)) + (2.7811f * yP) - 0.6302f
                   : (0.5000f * yP) + 0.5000f;

    // tone mapping step 3
    const float pSdr = 1.f + 32.f * pow(
                                        TargetCll /
                                        10000.f
                                    , gamma);

    //Y'sdr
    const float ySdr = (pow(pSdr, yC) - 1.f) /
                       (pSdr - 1.f);

    //f(Y'sdr)
    const float colourScaling = ySdr /
                                (GamutCompression * ycbcr.x);

    //C'b,tmo
    const float cbTmo = colourScaling * ycbcr.y;

    //C'r,tmo
    const float crTmo = colourScaling * ycbcr.z;

    //Y'tmo
    const float yTmo = ySdr
                     - max(0.1f * crTmo, 0.f);

    float3 hdrOut;

    hdrOut = Csp::Ycbcr::ToRgb::Bt2020(float3(yTmo, cbTmo, crTmo));

    hdrOut = saturate(hdrOut);

    // gamma decompression and adjust to TargetCll
    hdrOut = pow(hdrOut, inverseGamma) * (TargetCll / 10000.f);

    return hdrOut;
  }

  float3 Bt2446a_MOD1(
    const float3 Input,
    const float  TargetCll,
    const float  MaxCllIn,
    const float  GamutCompression,
    const float  TestH,
    const float  TestS)
  {
    float3 hdrIn = Input;

    const float maxCll = MaxCllIn > TargetCll
                       ? MaxCllIn
                       : TargetCll;

    // adjust the max of 1 according to maxCll
    hdrIn *= (10000.f / maxCll);

    // non-linear transfer function RGB->R'G'B'
    hdrIn = pow(hdrIn, gamma);

    //Y'C'bC'r
    const float3 ycbcr = Csp::Ycbcr::FromRgb::Bt2020(hdrIn);

    // tone mapping step 1
    const float pHdr = 1.f + 32.f * pow(
                                        TestH /
                                        10000.f
                                    , gamma);

    //Y'p
    const float yP = (log(1.f + (pHdr - 1.f) * ycbcr.x)) /
                      log(pHdr);

    // tone mapping step 2
    //Y'c
    const float yC = yP <= 0.7399f
                   ? 1.0770f * yP
                   : yP > 0.7399f && yP < 0.9909f
                   ? (-1.1510f * pow(yP, 2)) + (2.7811f * yP) - 0.6302f
                   : (0.5000f * yP) + 0.5000f;

    // tone mapping step 3
    const float pSdr = 1.f + 32.f * pow(
                                        TestS /
                                        10000.f
                                    , gamma);

    //Y'sdr
    const float ySdr = (pow(pSdr, yC) - 1.f) /
                       (pSdr - 1.f);

    //f(Y'sdr)
    const float colourScaling = ySdr /
                                (GamutCompression * ycbcr.x);

    //C'b,tmo
    const float cbTmo = colourScaling * ycbcr.y;

    //C'r,tmo
    const float crTmo = colourScaling * ycbcr.z;

    //Y'tmo
    const float yTmo = ySdr
                     - max(0.1f * crTmo, 0.f);

    float3 hdrOut;

    hdrOut = Csp::Ycbcr::ToRgb::Bt2020(float3(yTmo, cbTmo, crTmo));

    hdrOut = saturate(hdrOut);

    // gamma decompression and adjust to TargetCll
    hdrOut = pow(hdrOut, inverseGamma) * (TargetCll / 10000.f);

    return hdrOut;
  }

  namespace Bt2390
  {

    float HermiteSpline(
      const float E1,
      const float KneeStart,
      const float MaxLum)
    {
      const float oneMinusKneeStart = 1.f - KneeStart;
    	const float t = (E1 - KneeStart) / oneMinusKneeStart;
      const float tPow2 = pow(t, 2.f);
      const float tPow3 = pow(t, 3.f);

    	return
        ( 2.f * tPow3 - 3.f * tPow2 + 1.f) * KneeStart
      + (       tPow3 - 2.f * tPow2 + t)   * oneMinusKneeStart
      + (-2.f * tPow3 + 3.f * tPow2)       * MaxLum;
    }

  //const float tgt_max_PQ, // Lmax in PQ
  //const float tgt_min_PQ, // Lmin in PQ
  //const float SrcMaxPq,   // Lw in PQ

#define BT2390_PRO_MODE_ICTCP 0
#define BT2390_PRO_MODE_YCBCR 1
#define BT2390_PRO_MODE_YRGB  2
#define BT2390_PRO_MODE_RGB   3

    // works in PQ
    float3 Eetf(
      const float3 E_,
      const uint   ProcessingMode,
      const float  SrcMinPq, // Lb in PQ
      const float  SrcMaxPq, // Lw in PQ
      const float  MinLum,   // minLum
      const float  MaxLum,   // maxLum
      const float  KneeStart // KS
    )
    {
      if (ProcessingMode == BT2390_PRO_MODE_ICTCP)
      {
        float3 Lms = Csp::Ictcp::Mat::Bt2020To::Lms(E_);
        Lms = Csp::Trc::ToPq(Lms);

        const float i1 = 0.5f * Lms.x + 0.5f * Lms.y;
        //E1
        float i2 = (i1 - SrcMinPq) / (SrcMaxPq - SrcMinPq);
        //float i2 = i1 / SrcMaxPq;

        //E2
        if (i2 >= KneeStart)
        {
          const float ct1 = dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[1]);
          const float cp1 = dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[2]);

          i2 = HermiteSpline(i2, KneeStart, MaxLum);

          //E3
          i2 = i2 + MinLum * pow((1.f - i2), 4.f);

          //E4
          i2 = i2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
          //i2 = i2 * SrcMaxPq;

          const float minI = min((i1 / i2), (i2 / i1));

          //const float3 ct2cp2 = cross(float3(ct1, cp1, 0.f), minI);

          //to L'M'S'
          //Lms = mul(ICtCp_To_Lms, float3(i2, ct2cp2.x, ct2cp2.y));
          Lms = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(i2, minI * ct1, minI * cp1));
          //to LMS
          Lms = Csp::Trc::FromPq(Lms);
          //to RGB
          return clamp(Csp::Ictcp::Mat::LmsTo::Bt2020(Lms), 0.f, 65504.f);
        }
        else
        {
          return E_;
        }
      }
      else if (ProcessingMode == BT2390_PRO_MODE_YCBCR)
      {
        const float y1 = dot(E_, Csp::KHelpers::Bt2020::K);
        //E1
        float y2 = (y1 - SrcMinPq) / (SrcMaxPq - SrcMinPq);
        //float y2 = y1 / SrcMaxPq;

        //E2
        if (y2 >= KneeStart)
        {
          const float cb1 = (E_.b - y1) /
                            Csp::KHelpers::Bt2020::Kb;
          const float cr1 = (E_.r - y1) /
                            Csp::KHelpers::Bt2020::Kr;

          y2 = HermiteSpline(y2, KneeStart, MaxLum);

          //E3
          y2 = y2 + MinLum * pow((1.f - y2), 4.f);

          //E4
          y2 = y2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
          //y2 = y2 * SrcMaxPq;

          const float minY = min((y1 / y2), (y2 / y1));

          //const float3 cb2cr2 = cross(float3(cb1, cr1, 0.f), minY);
          //const float  cb2 = cb2cr2.x;
          //const float  cr2 = cb2cr2.y;

          const float cb2 = minY * cb1;
          const float cr2 = minY * cr1;

          return clamp(Csp::Ycbcr::ToRgb::Bt2020(float3(y2, cb2, cr2)), 0.f, 65504.f);
        }
        else
        {
          return E_;
        }
      }
      else if (ProcessingMode == BT2390_PRO_MODE_YRGB)
      {
        const float y1 = dot(E_, Csp::Mat::Bt2020ToXYZ[1].rgb);
        //E1 relative luminance
        float y2 = (Csp::Trc::ToPq(y1) - SrcMinPq) / (SrcMaxPq - SrcMinPq);
        //float y2 = Csp::Trc::ToPq(y1) / SrcMaxPq;

        //E2
        if (y2 >= KneeStart)
        {
          y2 = HermiteSpline(y2, KneeStart, MaxLum);

          //E3
          y2 = y2 + MinLum * pow((1.f - y2), 4.f);

          //E4
          y2 = y2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
          //y2 = y2 * SrcMaxPq;

          y2 = Csp::Trc::FromPq(y2);

          return clamp(y2 / y1 * E_, 0.f, 65504.f);
        }
        else
        {
          return E_;
        }
      }
      else if (ProcessingMode == BT2390_PRO_MODE_RGB)
      {
        //E1
        float3 Rgb = (E_ - SrcMinPq) / (SrcMaxPq - SrcMinPq);
        //float3 Rgb = E_ / SrcMaxPq;

        //E2
        if (Rgb.r >= KneeStart) {
          Rgb.r = HermiteSpline(Rgb.r, KneeStart, MaxLum);
        }
        if (Rgb.g >= KneeStart) {
          Rgb.g = HermiteSpline(Rgb.g, KneeStart, MaxLum);
        }
        if (Rgb.b >= KneeStart) {
          Rgb.b = HermiteSpline(Rgb.b, KneeStart, MaxLum);
        }

        //E3
        Rgb = Rgb + MinLum * pow((1.f - Rgb), 4.f);

        //E4
        return Rgb * (SrcMaxPq - SrcMinPq) + SrcMinPq;
        //return Rgb * SrcMaxPq;
      }
      else
        return float3(0.f, 0.f, 0.f);
    }
  }


  namespace Dice
  {
#define DICE_PRO_MODE_ICTCP 0
#define DICE_PRO_MODE_YCBCR 1

#define DICE_USE_BT2020  0
#define DICE_USE_AP0_D65 1

    // Applies exponential "Photographic" luminance compression
    float RangeCompress(float x)
    {
      return 1.f - exp(-x);
    }

    float LuminanceCompress(
      const float Colour,
      const float MaxNits,
      const float ShoulderStart)
    {
#if 1
      return ShoulderStart
           + (MaxNits - ShoulderStart)
           * RangeCompress((Colour  - ShoulderStart) /
                           (MaxNits - ShoulderStart));
#else
      return Colour < ShoulderStart
           ? Colour
           : ShoulderStart
           + (MaxNits - ShoulderStart)
           * RangeCompress((Colour  - ShoulderStart) /
                           (MaxNits - ShoulderStart));
#endif
    }

    // remap from infinite
    // ShoulderStart denotes the point where we change from linear to shoulder
    float3 ToneMapper(
            float3 Input,
      const float  MaxNits,
      const float  ShoulderStart,
      const uint   ProcessingMode,
      const uint   WorkingColourSpace)
    {
    
    // why does this not work?!
    //  float3x3 RgbToLms = WorkingColourSpace == DICE_USE_AP0_D65
    //                    ? RGB_AP0_D65_To_LMS
    //                    : RGB_BT2020_To_LMS;
    //
    //  float3x3 LmsToRgb = WorkingColourSpace == DICE_USE_AP0_D65
    //                    ? LmsToRgb_AP0_D65
    //                    : LmsToRgb_BT2020;
    //
    //  float3 KFactors = WorkingColourSpace == DICE_USE_AP0_D65
    //                  ? K_AP0_D65
    //                  : K_BT2020;
    //
    //  float  KbHelper = WorkingColourSpace == DICE_USE_AP0_D65
    //                  ? KB_AP0_D65_HELPER
    //                  : KB_BT2020_HELPER;
    //  float  KrHelper = WorkingColourSpace == DICE_USE_AP0_D65
    //                  ? KR_AP0_D65_HELPER
    //                  : KR_BT2020_HELPER;
    //  float2 KgHelper = WorkingColourSpace == DICE_USE_AP0_D65
    //                  ? KG_AP0_D65_HELPER
    //                  : KG_BT2020_HELPER;

      float3x3 RgbToLms = Csp::Ictcp::Mat::Ap0D65ToLms;
      float3x3 LmsToRgb = Csp::Ictcp::Mat::LmsToAp0D65;
      float3   KFactors = Csp::KHelpers::Ap0D65::K;
      float    KrHelper = Csp::KHelpers::Ap0D65::Kr;
      float    KbHelper = Csp::KHelpers::Ap0D65::Kb;
      float2   KgHelper = Csp::KHelpers::Ap0D65::Kg;

      if (WorkingColourSpace == DICE_USE_BT2020)
      {
        RgbToLms = Csp::Ictcp::Mat::Bt2020ToLms;
        LmsToRgb = Csp::Ictcp::Mat::LmsToBt2020;
        KFactors = Csp::KHelpers::Bt2020::K;
        KrHelper = Csp::KHelpers::Bt2020::Kr;
        KbHelper = Csp::KHelpers::Bt2020::Kb;
        KgHelper = Csp::KHelpers::Bt2020::Kg;
      }

      //YCbCr method copied from BT.2390
      if (ProcessingMode == DICE_PRO_MODE_ICTCP)
      {
        float3 Lms = mul(RgbToLms, Input);

        Lms = Csp::Trc::ToPq(Lms);

        const float i1 = 0.5f * Lms.x + 0.5f * Lms.y;

        if (i1 < ShoulderStart)
          return Input;
        else
        {
          const float ct1 = dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[1]);
          const float cp1 = dot(Lms, Csp::Ictcp::Mat::PqLmsToIctcp[2]);

          const float i2 = LuminanceCompress(i1, MaxNits, ShoulderStart);

          const float minI = min(min((i1 / i2), (i2 / i1)) * 1.1f, 1.f);

          //to L'M'S'
          Lms = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(i2, minI * ct1, minI * cp1));
          //to LMS
          Lms = Csp::Trc::FromPq(Lms);
          //to RGB
          return clamp(mul(LmsToRgb, Lms), 0.f, 65504.f);
        }
      }
      else
      {
        float3 Colour = Csp::Trc::ToPq(Input);

        const float y1 = dot(Colour, KFactors);

        if (y1 < ShoulderStart)
          return Input;
        else
        {
          const float y2 = LuminanceCompress(y1, MaxNits, ShoulderStart);

          const float minY = min(min((y1 / y2), (y2 / y1)) * 1.1f, 1.f);

          const float cb2 = minY * (Colour.b - y1) /
                                     KbHelper;
          const float cr2 = minY * (Colour.r - y1) /
                                     KrHelper;

          //return saturate(float3(y2 + KrHelper * cr2,
          //                       y2 - KgHelper[0] * cb2 - KgHelper[1] * cr2,
          //                       y2 + KbHelper * cb2));

          return clamp(Csp::Trc::FromPq(float3(y2 + KrHelper    * cr2,
                                               y2 - KgHelper[0] * cb2 - KgHelper[1] * cr2,
                                               y2 + KbHelper    * cb2)), 0, 65504.f);
        }
      }

#if 0
      return float3(LuminanceCompress(Colour.r, MaxNits, ShoulderStart),
                    LuminanceCompress(Colour.g, MaxNits, ShoulderStart),
                    LuminanceCompress(Colour.b, MaxNits, ShoulderStart));
#endif
    }
  }

}

#endif
