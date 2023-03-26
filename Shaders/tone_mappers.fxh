#include "HDR_analysis.fxh"

// Rep. ITU-R BT.2446-1 Table 2 & 3
float3 BT2446A_toneMapping(
  const float3 input,
  const float  targetCLL,
  const float  maxCLLin)
{
  float3 hdrIn = input;

  // gamma
  const float inverseGamma = 2.4f;
  const float gamma        = 1.f / inverseGamma;

  const float maxCLL = maxCLLin > targetCLL
                     ? maxCLLin
                     : targetCLL;

  // adjust the max of 1 according to maxCLL
  hdrIn *= maxCLL;

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
                             (1.1f * Y_);

  //C'b,tmo
  const float C_b_tmo = colorScaling * (hdrIn.b - Y_) /
                                       1.8814f;

  //C'r,tmo
  const float C_r_tmo = colorScaling * (hdrIn.r - Y_) /
                                       1.4746f;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut.r = Y_tmo + 1.4746f  * C_r_tmo;
  hdrOut.g = Y_tmo - 0.164553126843658 * C_b_tmo - 0.571353126843658 * C_r_tmo;
  hdrOut.b = Y_tmo + 1.8814f  * C_b_tmo;

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to targetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (targetCLL / 10000.f);

  return hdrOut;
}

float3 BT2446A_toneMapping_mod1(
  const float3 input,
  const float  targetCLL,
  const float  maxCLLin,
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
  hdrIn *= maxCLL;

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
  float colorScaling = 0.f;
  if (Y_ > 0.f) // avoid division by zero
    colorScaling = Y_sdr /
                   (1.1f * Y_);

  //C'b,tmo
  const float C_b_tmo = colorScaling * (hdrIn.b - Y_) /
                                       1.8814f;

  //C'r,tmo
  const float C_r_tmo = colorScaling * (hdrIn.r - Y_) /
                                       1.4746f;

  //Y'tmo
  const float Y_tmo = Y_sdr
                    - max(0.1f * C_r_tmo, 0.f);

  float3 hdrOut;

  hdrOut.r = Y_tmo + 1.4746f * C_r_tmo;
  hdrOut.g = Y_tmo - 0.164553126843658 * C_b_tmo - 0.571353126843658 * C_r_tmo;
  hdrOut.b = Y_tmo + 1.8814f * C_b_tmo;

  hdrOut = saturate(hdrOut);

  // gamma decompression and adjust to targetCLL
  hdrOut = pow(hdrOut, inverseGamma) * (targetCLL / 10000.f);

  return hdrOut;
}
