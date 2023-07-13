#pragma once

#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "hdr_analysis.fxh"

// Rep. ITU-R BT.2446-1 Table 2 & 3

// gamma
static const float inverseGamma = 2.4f;
static const float gamma        = 1.f / inverseGamma;

float3 BT2446A_ToneMapping(
  const float3 Input,
  const float  TargetCLL,
  const float  Max_CLL_In,
  const float  GamutCompression)
{
  float3 hdrIn = Input;

  const float maxCLL = Max_CLL_In > TargetCLL
                     ? Max_CLL_In
                     : TargetCLL;

  // adjust the max of 1 according to maxCLL
  hdrIn *= (10000.f / maxCLL);

  // non-linear transfer function RGB->R'G'B'
  hdrIn = pow(hdrIn, gamma);

  //Y'C'bC'r
  const float3 Ycbcr = Csp::Ycbcr::FromRgb::Bt2020(hdrIn);

  // tone mapping step 1
  const float pHDR = 1.f + 32.f * pow(
                                      maxCLL /
                                      10000.f
                                  , gamma);

  //Y'p
  const float Y_p = (log(1.f + (pHDR - 1.f) * Ycbcr.x)) /
                    log(pHDR);

  // tone mapping step 2
  //Y'c
  const float Y_c = Y_p <= 0.7399f
                  ? 1.0770f * Y_p
                  : Y_p > 0.7399f && Y_p < 0.9909f
                  ? (-1.1510f * pow(Y_p , 2)) + (2.7811f * Y_p) - 0.6302f
                  : (0.5000f * Y_p) + 0.5000f;

  // tone mapping step 3
  const float pSDR = 1.f + 32.f * pow(
                                      TargetCLL /
                                      10000.f
                                  , gamma);

  //Y'sdr
  const float Y_sdr = (pow(pSDR, Y_c) - 1.f) /
                      (pSDR - 1.f);

  //f(Y'sdr)
  const float colourScaling = Y_sdr /
                              (GamutCompression * Ycbcr.x);

  //C'b,tmo
  const float C_b_tmo = colourScaling * Ycbcr.y;

  //C'r,tmo
  const float C_r_tmo = colourScaling * Ycbcr.z;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut = Csp::Ycbcr::ToRgb::Bt2020(float3(Y_tmo, C_b_tmo, C_r_tmo));

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to TargetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (TargetCLL / 10000.f);

  return hdrOut;
}

float3 BT2446A_ToneMapping_Mod1(
  const float3 Input,
  const float  TargetCLL,
  const float  Max_CLL_In,
  const float  GamutCompression,
  const float  TestH,
  const float  TestS)
{
  float3 hdrIn = Input;

  const float maxCLL = Max_CLL_In > TargetCLL
                     ? Max_CLL_In
                     : TargetCLL;

  // adjust the max of 1 according to maxCLL
  hdrIn *= (10000.f / maxCLL);

  // non-linear transfer function RGB->R'G'B'
  hdrIn = pow(hdrIn, gamma);

  //Y'C'bC'r
  const float3 Ycbcr = Csp::Ycbcr::FromRgb::Bt2020(hdrIn);

  // tone mapping step 1
  const float pHDR = 1.f + 32.f * pow(
                                      TestH /
                                      10000.f
                                  , gamma);

  //Y'p
  const float Y_p = (log(1.f + (pHDR - 1.f) * Ycbcr.x)) /
                    log(pHDR);

  // tone mapping step 2
  //Y'c
  const float Y_c = Y_p <= 0.7399f
                  ? 1.0770f * Y_p
                  : Y_p > 0.7399f && Y_p < 0.9909f
                  ? (-1.1510f * pow(Y_p, 2)) + (2.7811f * Y_p) - 0.6302f
                  : (0.5000f * Y_p) + 0.5000f;

  // tone mapping step 3
  const float pSDR = 1.f + 32.f * pow(
                                      TestS /
                                      10000.f
                                  , gamma);

  //Y'sdr
  const float Y_sdr =
    (pow(pSDR, Y_c) - 1.f) /
    (pSDR - 1.f);

  //f(Y'sdr)
  const float colourScaling = Y_sdr /
                              (GamutCompression * Ycbcr.x);

  //C'b,tmo
  const float C_b_tmo = colourScaling * Ycbcr.y;

  //C'r,tmo
  const float C_r_tmo = colourScaling * Ycbcr.z;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut = Csp::Ycbcr::ToRgb::Bt2020(float3(Y_tmo, C_b_tmo, C_r_tmo));

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to TargetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (TargetCLL / 10000.f);

  return hdrOut;
}

float HermiteSpline(
  const float E1,
  const float KneeStart,
  const float MaxLum)
{
	const float T = (E1 - KneeStart) / (1.f - KneeStart);
  const float T_pow_2 = pow(T, 2.f);
  const float T_pow_3 = pow(T, 3.f);

	return
    ( 2.f * T_pow_3 - 3.f * T_pow_2 + 1.f) * KneeStart
  + (       T_pow_3 - 2.f * T_pow_2 + T)   * (1.f - KneeStart)
  + (-2.f * T_pow_3 + 3.f * T_pow_2)       * MaxLum;
}

//const float tgt_max_PQ, // Lmax in PQ
//const float tgt_min_PQ, // Lmin in PQ
//const float SrcMaxPq, // Lw in PQ

#define BT2390_PRO_MODE_RGB   0
#define BT2390_PRO_MODE_YRGB  1
#define BT2390_PRO_MODE_YCBCR 2
#define BT2390_PRO_MODE_ICTCP 3

// works in PQ
float3 BT2390_ToneMapping(
  const float3 E_,
  const uint   ProcessingMode,
//  const float  SrcMinPq, // Lb in PQ
  const float  SrcMaxPq,     // Lw in PQ
  const float  MinLum,       // minLum
  const float  MaxLum,       // maxLum
  const float  KneeStart     // KS
)
{
  if (ProcessingMode == BT2390_PRO_MODE_RGB)
  {
    //E1
    //float3 col = (E_ - SrcMinPq) / (SrcMaxPq - SrcMinPq);
    float3 col = E_ / SrcMaxPq;

    //E2
    if (col.r >= KneeStart) {
      col.r = HermiteSpline(col.r, KneeStart, MaxLum);
    }
    if (col.g >= KneeStart) {
      col.g = HermiteSpline(col.g, KneeStart, MaxLum);
    }
    if (col.b >= KneeStart) {
      col.b = HermiteSpline(col.b, KneeStart, MaxLum);
    }

    //E3
    col = col + MinLum * pow((1.f.xxx - col), 4.f);

    //E4
    //return col * (SrcMaxPq - SrcMinPq) + SrcMinPq;
    return col * SrcMaxPq;
  }
  else if (ProcessingMode == BT2390_PRO_MODE_YRGB)
  {
    const float Y1 = dot(E_, Csp::Mat::Bt2020ToXYZ[1].rgb);
    //E1 relative luminance
    //float Y2 = (PQ_Inverse_EOTF(Y1) - SrcMinPq) / (SrcMaxPq - SrcMinPq);
    float Y2 = Csp::Trc::ToPq(Y1) / SrcMaxPq;

    //E2
    if (Y2 >= KneeStart)
    {
      Y2 = HermiteSpline(Y2, KneeStart, MaxLum);

      //E3
      Y2 = Y2 + MinLum * pow((1.f - Y2), 4.f);

      //E4
      //Y2 = Y2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
      Y2 = Y2 * SrcMaxPq;

      Y2 = Csp::Trc::FromPq(Y2);

      return clamp(Y2 / Y1 * E_, 0.f, 65504.f);
    }
    else
    {
      return E_;
    }
  }
  else if (ProcessingMode == BT2390_PRO_MODE_YCBCR)
  {
    const float Y_1  = dot(E_, Csp::KHelpers::Bt2020::K);
    //E1
    //float Y_2 = (Y_1 - SrcMinPq) / (SrcMaxPq - SrcMinPq);
    float Y_2 = Y_1 / SrcMaxPq;

    //E2
    if (Y_2 >= KneeStart)
    {
      const float C_b1 = (E_.b - Y_1) /
                         Csp::KHelpers::Bt2020::Kb;
      const float C_r1 = (E_.r - Y_1) /
                         Csp::KHelpers::Bt2020::Kr;

      Y_2 = HermiteSpline(Y_2, KneeStart, MaxLum);

      //E3
      Y_2 = Y_2 + MinLum * pow((1.f - Y_2), 4.f);

      //E4
      //Y_2 = Y_2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
      Y_2 = Y_2 * SrcMaxPq;

      const float min_Y = min((Y_1 / Y_2), (Y_2 / Y_1));

      //const float3 C_b2_C_r2 = cross(float3(C_b1, C_r1, 0.f), min_Y);
      //const float  C_b2 = C_b2_C_r2.x;
      //const float  C_r2 = C_b2_C_r2.y;

      const float C_b2 = min_Y * C_b1;
      const float C_r2 = min_Y * C_r1;

      return clamp(Csp::Ycbcr::ToRgb::Bt2020(float3(Y_2, C_b2, C_r2)), 0.f, 65504.f);
    }
    else
    {
      return E_;
    }
  }
  else if (ProcessingMode == BT2390_PRO_MODE_ICTCP)
  {
    float3 LMS = Csp::Ictcp::Mat::Bt2020To::Lms(E_);
    LMS = Csp::Trc::ToPq(LMS);

    const float I1 = 0.5f * LMS.x + 0.5f * LMS.y;
    //E1
    //float I2 = (I1 - SrcMinPq) / (SrcMaxPq - SrcMinPq);
    float I2 = I1 / SrcMaxPq;

    //E2
    if (I2 >= KneeStart)
    {
      const float Ct1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[1]);
      const float Cp1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[2]);

      I2 = HermiteSpline(I2, KneeStart, MaxLum);

      //E3
      I2 = I2 + MinLum * pow((1.f - I2), 4.f);

      //E4
      //I2 = I2 * (SrcMaxPq - SrcMinPq) + SrcMinPq;
      I2 = I2 * SrcMaxPq;

      const float min_I = min((I1 / I2), (I2 / I1));

      //const float3 Ct2_Cp2 = cross(float3(Ct1, Cp1, 0.f), min_I);

      //to L'M'S'
      //LMS = mul(ICtCp_To_LMS, float3(I2, Ct2_Cp2.x, Ct2_Cp2.y));
      LMS = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(I2, min_I * Ct1, min_I * Cp1));
      //to LMS
      LMS = Csp::Trc::FromPq(LMS);
      //to RGB
      return clamp(Csp::Ictcp::Mat::LmsTo::Bt2020(LMS), 0.f, 65504.f);
    }
    else
    {
      return E_;
    }
  }
  else
    return float3(0.f, 0.f, 0.f);
}

#define DICE_PRO_MODE_ICTCP 0
#define DICE_PRO_MODE_YCBCR 1

#define DICE_USE_AP0_D65 0
#define DICE_USE_BT2020  1

// Applies exponential ("Photographic") luma compression
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
float3 dice(
        float3 Input,
  const float  MaxNits,
  const float  ShoulderStart,
  const uint   ProcessingMode,
  const uint   WorkingColourSpace)
{

// why does this not work?!
//  float3x3 RgbToLms = WorkingColourSpace == DICE_USE_AP0_D65
//                      ? RGB_AP0_D65_To_LMS
//                      : RGB_BT2020_To_LMS;
//
//  float3x3 LmsToRgb = WorkingColourSpace == DICE_USE_AP0_D65
//                      ? LmsToRgb_AP0_D65
//                      : LmsToRgb_BT2020;
//
//  float3 K_Factors = WorkingColourSpace == DICE_USE_AP0_D65
//                   ? K_AP0_D65
//                   : K_BT2020;
//
//  float  KB_Helper = WorkingColourSpace == DICE_USE_AP0_D65
//                   ? KB_AP0_D65_HELPER
//                   : KB_BT2020_HELPER;
//  float  KR_Helper = WorkingColourSpace == DICE_USE_AP0_D65
//                   ? KR_AP0_D65_HELPER
//                   : KR_BT2020_HELPER;
//  float2 KG_Helper = WorkingColourSpace == DICE_USE_AP0_D65
//                   ? KG_AP0_D65_HELPER
//                   : KG_BT2020_HELPER;

  float3x3 RgbToLms = Csp::Ictcp::Mat::Ap0D65ToLms;
  float3x3 LmsToRgb = Csp::Ictcp::Mat::LmsToAp0D65;
  float3   K_Factors  = Csp::KHelpers::Ap0D65::K;
  float    KR_Helper  = Csp::KHelpers::Ap0D65::Kr;
  float    KB_Helper  = Csp::KHelpers::Ap0D65::Kb;
  float2   KG_Helper  = Csp::KHelpers::Ap0D65::Kg;

  if (WorkingColourSpace == DICE_USE_BT2020)
  {
    RgbToLms = Csp::Ictcp::Mat::Bt2020ToLms;
    LmsToRgb = Csp::Ictcp::Mat::LmsToBt2020;
    K_Factors  = Csp::KHelpers::Bt2020::K;
    KR_Helper  = Csp::KHelpers::Bt2020::Kr;
    KB_Helper  = Csp::KHelpers::Bt2020::Kb;
    KG_Helper  = Csp::KHelpers::Bt2020::Kg;
  }

  //YCbCr method copied from BT.2390
  if (ProcessingMode == DICE_PRO_MODE_ICTCP)
  {
    float3 LMS = mul(RgbToLms, Input);

    LMS = Csp::Trc::ToPq(LMS);

    const float I1 = 0.5f * LMS.x + 0.5f * LMS.y;

    if (I1 < ShoulderStart)
      return Input;
    else
    {
      const float Ct1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[1]);
      const float Cp1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[2]);

      const float I2 = LuminanceCompress(I1, MaxNits, ShoulderStart);

      const float min_I = min(min((I1 / I2), (I2 / I1)) * 1.1f, 1.f);

      //to L'M'S'
      LMS = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(I2, min_I * Ct1, min_I * Cp1));
      //to LMS
      LMS = Csp::Trc::FromPq(LMS);
      //to RGB
      return clamp(mul(LmsToRgb, LMS), 0.f, 65504.f);
    }
  }
  else
  {
    float3 Colour = Csp::Trc::ToPq(Input);

    const float Y_1 = dot(Colour, K_Factors);

    if (Y_1 < ShoulderStart)
      return Input;
    else
    {
      const float Y_2 = LuminanceCompress(Y_1, MaxNits, ShoulderStart);

      const float min_Y = min(min((Y_1 / Y_2), (Y_2 / Y_1)) * 1.1f, 1.f);

      const float C_b2 = min_Y * (Colour.b - Y_1) /
                                 KB_Helper;
      const float C_r2 = min_Y * (Colour.r - Y_1) /
                                 KR_Helper;

      //return saturate(float3(Y_2 + KR_Helper * C_r2,
      //                       Y_2 - KG_Helper[0] * C_b2 - KG_Helper[1] * C_r2,
      //                       Y_2 + KB_Helper * C_b2));

      return clamp(Csp::Trc::FromPq(float3(Y_2 + KR_Helper    * C_r2,
                                           Y_2 - KG_Helper[0] * C_b2 - KG_Helper[1] * C_r2,
                                           Y_2 + KB_Helper    * C_b2)), 0, 65504.f);
    }
  }

#if 0
  return float3(LuminanceCompress(Colour.r, MaxNits, ShoulderStart),
                LuminanceCompress(Colour.g, MaxNits, ShoulderStart),
                LuminanceCompress(Colour.b, MaxNits, ShoulderStart));
#endif
}

#endif
