#include "HDR_analysis.fxh"

// Rep. ITU-R BT.2446-1 Table 2 & 3
float3 BT2446A_tone_mapping(
  const float3 input,
  const float  targetCLL,
  const float  maxCLLin,
  const float  gamut_compression)
{
  float3 hdrIn = input;

  // gamma
  const float inverseGamma = 2.4f;
  const float gamma        = 1.f / inverseGamma;

  const float maxCLL = maxCLLin > targetCLL
                     ? maxCLLin
                     : targetCLL;

  // adjust the max of 1 according to maxCLL
  hdrIn *= (10000.f / maxCLL);

  // non-linear transfer function RGB->R'G'B'
  hdrIn = pow(hdrIn, gamma);

  //Y'
  const float Y_ = dot(hdrIn, K_BT2020);

  // tone mapping step 1
  const float pHDR = 1.f + 32.f * pow(
                                      maxCLL /
                                      10000.f
                                  , gamma);

  //Y'p
  const float Y_p = (log(1.f + (pHDR - 1.f) * Y_)) /
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
                                      targetCLL /
                                      10000.f
                                  , gamma);

  //Y'sdr
  const float Y_sdr = (pow(pSDR, Y_c) - 1.f) /
                      (pSDR - 1.f);

  //f(Y'sdr)
  const float colorScaling = Y_sdr /
                             (gamut_compression * Y_);

  //C'b,tmo
  const float C_b_tmo = colorScaling * (hdrIn.b - Y_) /
                                       KB_BT2020_helper;

  //C'r,tmo
  const float C_r_tmo = colorScaling * (hdrIn.r - Y_) /
                                       KR_BT2020_helper;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut = ycbcr_bt2020_to_rgb(float3(Y_tmo, C_b_tmo, C_r_tmo));

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to targetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (targetCLL / 10000.f);

  return hdrOut;
}

float3 BT2446A_tone_mapping_mod1(
  const float3 input,
  const float  targetCLL,
  const float  maxCLLin,
  const float  gamut_compression,
  const float  testH,
  const float  testS)
{
  float3 hdrIn = input;

  // gamma
  const float inverseGamma = 2.4f;
  const float gamma        = 1.f / inverseGamma;

  const float maxCLL = maxCLLin > targetCLL
                     ? maxCLLin
                     : targetCLL;

  // adjust the max of 1 according to maxCLL
  hdrIn *= (10000.f / maxCLL);

  // non-linear transfer function RGB->R'G'B'
  hdrIn = pow(hdrIn, gamma);

  //Y'
  const float Y_ = dot(hdrIn, K_BT2020);

  // tone mapping step 1
  const float pHDR = 1.f + 32.f * pow(
                                      testH /
                                      10000.f
                                  , gamma);

  //Y'p
  const float Y_p = (log(1.f + (pHDR - 1.f) * Y_)) /
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
                                      testS /
                                      10000.f
                                  , gamma);

  //Y'sdr
  const float Y_sdr =
    (pow(pSDR, Y_c) - 1.f) /
    (pSDR - 1.f);

  //f(Y'sdr)
  const float colorScaling = Y_sdr /
                             (gamut_compression * Y_);

  //C'b,tmo
  const float C_b_tmo = colorScaling * (hdrIn.b - Y_) /
                                       KB_BT2020_helper;

  //C'r,tmo
  const float C_r_tmo = colorScaling * (hdrIn.r - Y_) /
                                       KR_BT2020_helper;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut = ycbcr_bt2020_to_rgb(float3(Y_tmo, C_b_tmo, C_r_tmo));

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to targetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (targetCLL / 10000.f);

  return hdrOut;
}

float Hermite_spline(
  const float E1,
  const float knee_start,
  const float max_lum)
{
	const float T = (E1 - knee_start) / (1.f - knee_start);
  const float T_pow_2 = pow(T, 2.f);
  const float T_pow_3 = pow(T, 3.f);

	return
    ( 2.f * T_pow_3 - 3.f * T_pow_2 + 1.f) * knee_start
  + (       T_pow_3 - 2.f * T_pow_2 + T)   * (1.f - knee_start)
  + (-2.f * T_pow_3 + 3.f * T_pow_2)       * max_lum;
}

//const float tgt_max_PQ, // Lmax in PQ
//const float tgt_min_PQ, // Lmin in PQ
//const float src_max_PQ, // Lw in PQ

#define PRO_MODE_RGB   0
#define PRO_MODE_YRGB  1
#define PRO_MODE_YCBCR 2
#define PRO_MODE_ICTCP 3

// works in PQ
float3 BT2390_tone_mapping(
  const float3 E_,
  const uint   processing_mode,
//  const float  src_min_PQ, // Lb in PQ
  const float  src_max_PQ, // Lw in PQ
  const float  min_lum,    // minLum
  const float  max_lum,    // maxLum
  const float  knee_start  // KS
)
{
  if (processing_mode == PRO_MODE_RGB)
  {
    //E1
    //float3 col = (E_ - src_min_PQ) / (src_max_PQ - src_min_PQ);
    float3 col = E_ / src_max_PQ;

    //E2
    if (col.r >= knee_start)
      col.r = Hermite_spline(col.r, knee_start, max_lum);
    if (col.g >= knee_start)
      col.g = Hermite_spline(col.g, knee_start, max_lum);
    if (col.b >= knee_start)
      col.b = Hermite_spline(col.b, knee_start, max_lum);

    //E3
    col = col + min_lum * pow((1.f.xxx - col), 4.f);

    //E4
    //return col * (src_max_PQ - src_min_PQ) + src_min_PQ;
    return col * src_max_PQ;
  }
  else if (processing_mode == PRO_MODE_YRGB)
  {
    const float Y1 = dot(E_, BT2020_to_XYZ[1].rgb);
    //E1 relative luminance
    //float Y2 = (PQ_inverse_EOTF(Y1) - src_min_PQ) / (src_max_PQ - src_min_PQ);
    float Y2 = PQ_inverse_EOTF(Y1) / src_max_PQ;

    //E2
    if (Y2 >= knee_start)
    {
      Y2 = Hermite_spline(Y2, knee_start, max_lum);

      //E3
      Y2 = Y2 + min_lum * pow((1.f - Y2), 4.f);

      //E4
      //Y2 = Y2 * (src_max_PQ - src_min_PQ) + src_min_PQ;
      Y2 = Y2 * src_max_PQ;

      Y2 = PQ_EOTF(Y2);

      return clamp(Y2 / Y1 * E_, 0.f, 65504.f);
    }
    else
    {
      return E_;
    }
  }
  else if (processing_mode == PRO_MODE_YCBCR)
  {
    const float Y_1  = dot(E_, K_BT2020);
    //E1
    //float Y_2 = (Y_1 - src_min_PQ) / (src_max_PQ - src_min_PQ);
    float Y_2 = Y_1 / src_max_PQ;

    //E2
    if (Y_2 >= knee_start)
    {
      const float C_b1 = (E_.b - Y_1) /
                         KB_BT2020_helper;
      const float C_r1 = (E_.r - Y_1) /
                         KR_BT2020_helper;

      Y_2 = Hermite_spline(Y_2, knee_start, max_lum);

      //E3
      Y_2 = Y_2 + min_lum * pow((1.f - Y_2), 4.f);

      //E4
      //Y_2 = Y_2 * (src_max_PQ - src_min_PQ) + src_min_PQ;
      Y_2 = Y_2 * src_max_PQ;

      const float min_Y = min((Y_1 / Y_2), (Y_2 / Y_1));

      //const float3 C_b2_C_r2 = cross(float3(C_b1, C_r1, 0.f), min_Y);
      //const float  C_b2 = C_b2_C_r2.x;
      //const float  C_r2 = C_b2_C_r2.y;

      const float C_b2 = min_Y * C_b1;
      const float C_r2 = min_Y * C_r1;

      return clamp(ycbcr_bt2020_to_rgb(float3(Y_2, C_b2, C_r2)), 0.f, 65504.f);
    }
    else
    {
      return E_;
    }
  }
  else if (processing_mode == PRO_MODE_ICTCP)
  {
    float3 LMS = mul(RGB_BT2020_to_LMS, E_);
    LMS = PQ_inverse_EOTF(LMS);

    const float I1  = 0.5f * LMS.x + 0.5f * LMS.y;
    //E1
    //float I2 = (I1 - src_min_PQ) / (src_max_PQ - src_min_PQ);
    float I2 = I1 / src_max_PQ;

    //E2
    if (I2 >= knee_start)
    {
      const float Ct1 = dot(LMS, LMS_to_ICtCp[1]);
      const float Cp1 = dot(LMS, LMS_to_ICtCp[2]);

      I2 = Hermite_spline(I2, knee_start, max_lum);

      //E3
      I2 = I2 + min_lum * pow((1.f - I2), 4.f);

      //E4
      //I2 = I2 * (src_max_PQ - src_min_PQ) + src_min_PQ;
      I2 = I2 * src_max_PQ;

      const float min_I = min((I1 / I2), (I2 / I1));

      //const float3 Ct2_Cp2 = cross(float3(Ct1, Cp1, 0.f), min_I);

      //to L'M'S'
      //LMS = mul(ICtCp_to_LMS, float3(I2, Ct2_Cp2.x, Ct2_Cp2.y));
      LMS = mul(ICtCp_to_LMS, float3(I2, min_I * Ct1, min_I * Cp1));
      //to LMS
      LMS = PQ_EOTF(LMS);
      //to RGB
      return clamp(mul(LMS_to_RGB_BT2020, LMS), 0.f, 65504.f);
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
float range_compress(float x)
{
  return 1.f - exp(-x);
}

float luminance_compress(float colour, float max_nits, float shoulder_start)
{
#if 1
  return shoulder_start
       + (max_nits - shoulder_start)
       * range_compress((colour   - shoulder_start) /
                        (max_nits - shoulder_start));
#else
  return colour < shoulder_start
       ? colour
       : shoulder_start
       + (max_nits - shoulder_start)
       * range_compress((colour   - shoulder_start) /
                        (max_nits - shoulder_start));
#endif
}

// remap from infinite
// shoulder_start denotes the point where we change from linear to shoulder
float3 dice(
        float3 input,
        float  max_nits,
        float  shoulder_start,
  const uint   processing_mode,
  const uint   working_color_space)
{

// why does this not work?!
//  float3x3 RGB_to_LMS = working_color_space == DICE_USE_AP0_D65
//                            ? RGB_AP0_D65_to_LMS
//                            : RGB_BT2020_to_LMS;
//
//  float3x3 LMS_to_RGB = working_color_space == DICE_USE_AP0_D65
//                            ? LMS_to_RGB_AP0_D65
//                            : LMS_to_RGB_BT2020;
//
//  float3 K_factors = working_color_space == DICE_USE_AP0_D65
//                         ? K_AP0_D65
//                         : K_BT2020;
//
//  float  KB_helper = working_color_space == DICE_USE_AP0_D65
//                         ? KB_AP0_D65_helper
//                         : KB_BT2020_helper;
//  float  KR_helper = working_color_space == DICE_USE_AP0_D65
//                         ? KR_AP0_D65_helper
//                         : KR_BT2020_helper;
//  float2 KG_helper = working_color_space == DICE_USE_AP0_D65
//                         ? KG_AP0_D65_helper
//                         : KG_BT2020_helper;

  float3x3 RGB_to_LMS = RGB_AP0_D65_to_LMS;
  float3x3 LMS_to_RGB = LMS_to_RGB_AP0_D65;
  float3   K_factors  = K_AP0_D65;
  float    KR_helper  = KR_AP0_D65_helper;
  float    KB_helper  = KB_AP0_D65_helper;
  float2   KG_helper  = KG_AP0_D65_helper;

  if (working_color_space == DICE_USE_BT2020)
  {
    RGB_to_LMS = RGB_BT2020_to_LMS;
    LMS_to_RGB = LMS_to_RGB_BT2020;
    K_factors  = K_BT2020;
    KR_helper  = KR_BT2020_helper;
    KB_helper  = KB_BT2020_helper;
    KG_helper  = KG_BT2020_helper;
  }

  //YCbCr method copied from BT.2390
  if (processing_mode == DICE_PRO_MODE_ICTCP)
  {
    float3 LMS = mul(RGB_to_LMS, input);

    LMS = PQ_inverse_EOTF(LMS);

    const float I1 = 0.5f * LMS.x + 0.5f * LMS.y;

    if (I1 < shoulder_start)
      return input;
    else
    {
      const float Ct1 = dot(LMS, LMS_to_ICtCp[1]);
      const float Cp1 = dot(LMS, LMS_to_ICtCp[2]);

      const float I2 = luminance_compress(I1, max_nits, shoulder_start);

      const float min_I = min(min((I1 / I2), (I2 / I1)) * 1.1f, 1.f);

      //to L'M'S'
      LMS = mul(ICtCp_to_LMS, float3(I2, min_I * Ct1, min_I * Cp1));
      //to LMS
      LMS = PQ_EOTF(LMS);
      //to RGB
      return clamp(mul(LMS_to_RGB, LMS), 0.f, 65504.f);
    }
  }
  else
  {
    float3 colour = PQ_inverse_EOTF(input);

    const float Y_1 = dot(colour, K_factors);

    if (Y_1 < shoulder_start)
      return input;
    else
    {
      const float Y_2 = luminance_compress(Y_1, max_nits, shoulder_start);

      const float min_Y = min(min((Y_1 / Y_2), (Y_2 / Y_1)) * 1.1f, 1.f);

      const float C_b2 = min_Y * (colour.b - Y_1) /
                                 KB_helper;
      const float C_r2 = min_Y * (colour.r - Y_1) /
                                 KR_helper;

      //return saturate(float3(Y_2 + KR_helper * C_r2,
      //                       Y_2 - KG_helper[0] * C_b2 - KG_helper[1] * C_r2,
      //                       Y_2 + KB_helper * C_b2));

      return clamp(PQ_EOTF(float3(Y_2 + KR_helper    * C_r2,
                                  Y_2 - KG_helper[0] * C_b2 - KG_helper[1] * C_r2,
                                  Y_2 + KB_helper    * C_b2)), 0, 65504.f);
    }
  }

#if 0
  return float3(luminance_compress(colour.r, max_nits, shoulder_start),
                luminance_compress(colour.g, max_nits, shoulder_start),
                luminance_compress(colour.b, max_nits, shoulder_start));
#endif
}
