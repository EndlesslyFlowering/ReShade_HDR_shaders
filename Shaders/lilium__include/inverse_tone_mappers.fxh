#pragma once

#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "colour_space.fxh"

//float3 gamut(
//  const float3 Input,
//  const uint   gamutExpansionType)
//{
//  float3 sdr = Input;
//
//  //BT.709->BT.2020 colourspace conversion
//  switch (gamutExpansionType)
//  {
//    case 0:
//      sdr = mul(BT709_To_BT2020, sdr);
//      break;
//    case 1:
//      sdr = mul(myExp_BT709_To_BT2020, sdr);
//      break;
//    case 2:
//      sdr = mul(expanded_BT709_To_BT2020_matrix, sdr);
//      break;
//    case 3:
//      sdr = ExpandColourGamutBT2020(sdr, 1.f, 5.f);
//      break;
//  }
//
//  return sdr;
//}


//gamma
static const float gamma        = 2.4f;
static const float inverseGamma = 1.f / gamma;

// outputs normalised values
float3 BT2446A_InverseToneMapping(
  const float3 Input,
  const float  Lhdr,
  const float  Lsdr,
  const float  InputNitsFactor,
  const float  GamutExpansion,
  const float  GammaIn,
  const float  GammaOut)
{
  float3 sdr = saturate(Input / InputNitsFactor);

  //RGB->R'G'B' gamma compression
  sdr = pow(sdr, 1.f / (gamma + GammaIn));

  // Rec. ITU-R BT.2020-2 Table 4
  //Y'C'bC'r,tmo
  const float3 YCbCr_tmo = CSP::YCbCr::FromRGB::BT2020(sdr);

  // adjusted luma component (inverse)
  // get Y'sdr
  const float Y_sdr = YCbCr_tmo.x + max(0.1f * YCbCr_tmo.z, 0.f);

  // Tone mapping step 3 (inverse)
  // get Y'c
  const float pSDR = 1.f + 32.f * pow(
                                      Lsdr /
                                      10000.f
                                  , gamma);

  //Y'c
  //if pSDR == 1 there is a division by zero
  //this happens when Lsdr == 0
  const float Y_c = log((Y_sdr * (pSDR - 1)) + 1) /
                    log(pSDR); //log = ln

  // Tone mapping step 2 (inverse)
  // get Y'p
  float Y_p = 0.f;

  const float Y_p_0 = Y_c / 1.0770f;
  const float Y_p_2 = (Y_c - 0.5000f) /
                      0.5000f;
  //(4.83307641 - 4.604f * Y_c) == (pow(2.7811f, 2) - 4 * (-1.151f) * (-0.6302f - Y_c))
  const float Y_p_1_1 = (-2.7811f + sqrt(4.83307641 - 4.604f * Y_c)) /
                        -2.302f;
  //Y_p_1_2 is never reached
  //const float  Y_p_1_2 = abs(
  //              (_1_first - _1_sqrt) /
  //              _1_div
  //            );

  if (Y_p_0 <= 0.7399f)
    Y_p = Y_p_0;
  else if (Y_p_1_1 > 0.7399f
        && Y_p_1_1 < 0.9909f)
    Y_p = Y_p_1_1;
  //else if (Y_p_1_2 > 0.7399f && Y_p_1_2 < 0.9909f)
  //  Y_p = Y_p_1_2;
  else if (Y_p_2 >= 0.9909f)
    Y_p = Y_p_2;
  else //Y_p_1_1 sometimes (0.12% out of the full RGB range) is less than 0.7399f or more than 0.9909f
  {
    //error is small enough (less than 0.001) for this to be OK
    //ideally you would choose between Y_p_0 and Y_p_1_1 if Y_p_1_1 < 0.7399f depending on which is closer to 0.7399f
    //or between Y_p_1_1 and Y_p_2 if Y_p_1_1 > 0.9909f depending on which is closer to 0.9909f
    Y_p = Y_p_1_1;

    //this clamps it to one float step above 0.7399f or one float step below 0.9909f
    //if (Y_p_1_1 < 0.7399f)
    //  Y_p = 0.73990005f;
    //else
    //  Y_p = 0.9908999f;
  }

  // Tone mapping step 1 (inverse)
  // get Y'
  const float pHDR = 1.f + 32.f * pow(
                                      Lhdr /
                                      10000.f
                                  , gamma);
  //Y'hdr
  //if pHDR == 1 there is a division by zero
  //this happens when Lhdr == 0
  const float Y_hdr = (pow(pHDR, Y_p) - 1.f) /
                      (pHDR - 1.f);

  // Colour scaling function
  // TODO: analyse behaviour of colourScale being 1 or 0.00000001
  //float colourScale = 0.0000001f;
  //if (Y_hdr > 0.f && Y_sdr > 0.f) // avoid division by zero
  // this is actually fine to be infinite because it will create 0 as a result in the next step
  const float colourScale = Y_sdr /
                            (GamutExpansion * Y_hdr);

  // Colour difference signals (inverse) and Luma (inverse)
  // get R'G'B'

//  hdr.b = ((C_b_tmo * 1.8814f) /
//           colourScale) + Y_hdr;
//  hdr.r = ((C_r_tmo * 1.4746f) /
//           colourScale) + Y_hdr;
//  hdr.g = (Y_hdr - (K_BT2020.r * hdr.r + K_BT2020.b * hdr.b)) /
//          K_BT2020.g;

//  produces the same results
  const float C_b_hdr = YCbCr_tmo.y / colourScale;
  const float C_r_hdr = YCbCr_tmo.z / colourScale;

  float3 hdr = CSP::YCbCr::ToRGB::BT2020(float3(Y_hdr, C_b_hdr, C_r_hdr));

  hdr = saturate(hdr); //on edge cases the YCbCr->RGB conversion isn't accurate enough

  // Non-linear transfer function (inverse)
  // get RGB
  hdr = pow(hdr, gamma + GammaIn + GammaOut);

  //expand to target luminance
  hdr *= (Lhdr / 10000.f);

  return hdr;
}


// outputs normalised values
float3 Map_SDR_Into_HDR(
  const float3 Input,
  const float  Brightness)
{
  //map SDR into HDR
  return Input * (Brightness / 10000.f);
}


// HDR reference white in XYZ
#define HDR_REF_WHITE_XYZ float3(192.93f, 203.f, 221.05f)
#define DELTA             6.f / 29.f
#define POW_DELTA_3       pow(DELTA, 3.f)
#define _3_X_POW_DELTA_2  3.f * pow(DELTA, 2.f)

//// outputs normalised values
//float3 BT2446C_InverseToneMapping(
//  const float3 Input,
//  const float  sdr_brightness,
//  const float  Alpha,
////        float  k1,
////        float  inf_point,
//  const bool   use_achromatic_correction,
//  const float  sigma)
//{
//
//  //103.2 =  400 nits
//  //107.1 =  500 nits
//  //110.1 =  600 nits
//  //112.6 =  700 nits
//  //114.8 =  800 nits
//  //116.7 =  900 nits
//  //118.4 = 1000 nits
//  //153.7 is just under 10000 nits for Alpha=0 and above it starts clipping
//  float3 sdr = Input * (sdr_brightness > 153.7f
//                      ? 153.7f
//                      : sdr_brightness);
//
//  //6.1.6 (inverse)
//  //crosstalk matrix from 6.1.2
//  //const float Alpha   = 0.f; //hardcode for now as it gives the best results imo
//  const float xlpha = 1.f - 2.f * Alpha;
//  const float3x3 crosstalkMatrix = float3x3(xlpha, Alpha, Alpha,
//                                             Alpha, xlpha, Alpha,
//                                             Alpha, Alpha, xlpha);
//
//  sdr = mul(crosstalkMatrix, sdr);
//
//  //6.1.5 (inverse)
//  //conversion to XYZ and then Yxy -> x and y is at the end of the achromatic correction or the else case
//  sdr = mul(BT2020_To_XYZ, sdr);
//  const float Y_sdr = sdr.y;
//  const float xyz   = sdr.x + sdr.y + sdr.z;
//  const float x_sdr = sdr.x /
//                      xyz;
//  const float y_sdr = sdr.y /
//                      xyz;
//
//  //6.1.4 (inverse)
//  //inverse tone mapping
//  const float k1 = 0.83802f;
//  const float k2 = 15.09968f;
//  const float k3 = 0.74204f;
//  const float k4 = 78.99439f;
//  const float Y_hdr_ip = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
//  //k1 = 0.83802f;
//  //k2 = 15.09968f;
//  //k3 = 0.74204f;
//  //k4 = 78.99439f;
//  //const float Y_hlg_ref = 203.f;
//  //const float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
//  //const float Y_hdr_ip = inf_point / k1; // 58.5 = 0.80^2.4
//  //const float k2 = inf_point * (1.f - k3);
//  //const float k4 = inf_point - k2 * log(1.f - k3);
//  //const float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / Y_hdr_ip);
//
//  float Y_hdr = Y_sdr / k1;
//
//  if (Y_hdr >= Y_hdr_ip)
//    Y_hdr = (exp((Y_sdr - k4) / k2) + k3) * Y_hdr_ip;
//
//  //6.1.3 (inverse) part 1
//  //convert to XYZ
//  const float  X_hdr_uncor = (x_sdr / y_sdr) * Y_hdr;
//  const float  Z_hdr_uncor = ((1.f - x_sdr - y_sdr) / y_sdr) * Y_hdr;
//        float3 hdr         = float3(X_hdr_uncor, Y_hdr, Z_hdr_uncor);
//
//  bool  useful_correction = false;
//  float x_cor = 0.f;
//  float y_cor = 0.f;
//  // optional chroma correction above HDR Reference White (inverse)
//  if (use_achromatic_correction && sigma > 0.f)
//  {
//    // (3) inverse
//    // XYZ->L*a*b* from (1) which is actually wrong
//    // using correct conversion here
//    //==========================================================================
//    // it seems the ITU was trying to make a faster version for t <= POW_DELTA_3
//    // corrected version is here:
//    // L* = (116 * (t / POW_DELTA_3) - 16) / 10.f
//    // it's missing the division by 10 in the ITU doc
//
//    // get L*
//    const float t_Y = Y_hdr / HDR_REF_WHITE_XYZ.y;
//          float f_Y = 0.f;
//
//    if (t_Y > POW_DELTA_3)
//      f_Y = (pow(t_Y, 1.f / 3.f));
//    else
//      f_Y = (t_Y / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//
//    const float L_star = 116.f * f_Y - 16.f;
//
//    // get a*
//    const float t_X = sdr.x / HDR_REF_WHITE_XYZ.x;
//          float f_X = 0.f;
//
//    if (t_X > POW_DELTA_3)
//      f_X = (pow(t_Y, 1.f / 3.f));
//    else
//      f_X = (t_X / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//    f_X -= f_Y;
//
//    const float a_star = 116.f * f_X - 16.f;
//
//    // get b*
//    const float t_Z = sdr.z / HDR_REF_WHITE_XYZ.z;
//          float f_Z = 0.f;
//
//    if (t_Z > POW_DELTA_3)
//      f_Z = (pow(t_Y, 1.f / 3.f));
//    else
//      f_Z = (t_Z / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//    f_Z = f_Y - f_Z;
//
//    const float b_star = 116.f * f_Z - 16.f;
//
//    // (2) chroma correction above Reference White
//    const float L_star_ref = 100.f;
//    const float L_star_max = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
//
//    // convert to CIELCh
//    const float C_star_ab = sqrt(pow(a_star, 2.f) + pow(b_star, 2.f));
//    const float h_ab      = atan(b_star / a_star);
//
//    float f_cor = 1.f;
//    if (L_star > L_star_ref)
//    {
//      f_cor = 1.f - sigma * (L_star     - L_star_ref) /
//                            (L_star_max - L_star_ref);
//      if (f_cor <= 0.f) // avoid division by zero
//        f_cor = 0.0000001f;
//
//      useful_correction = true;
//
//      // amazing function inversion!!
//      const float C_star_ab_cor = C_star_ab / f_cor;
//
//      // convert back to CIELAB
//      const float a_star_cor = C_star_ab_cor * cos(h_ab);
//      const float b_star_cor = C_star_ab_cor * sin(h_ab);
//
//      // (1) inverse
//      // conversion from L*a*b* to XYZ from (3) and then Yxy
//      float3 XYZ_cor;
//      const float f_Y_cor = (L_star + 16.f) /
//                            116.f;
//      const float f_X_cor = f_Y_cor + a_star_cor /
//                                      500.f;
//      const float f_Z_cor = f_Y_cor - b_star_cor /
//                                      200.f;
//
//      //X
//      if (f_X_cor > DELTA)
//        XYZ_cor.x = HDR_REF_WHITE_XYZ.x * pow(f_X_cor, 3);
//      else
//        XYZ_cor.x = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.x;
//
//      // can you just take the XYZ Y from the input here since it's unchanged?
//      // probably yes
//      //Y
//      if (f_Y_cor > DELTA)
//        XYZ_cor.y = HDR_REF_WHITE_XYZ.y * pow(f_Y_cor, 3.f);
//      else
//        XYZ_cor.y = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.y;
//
//      //Z
//      if (f_Z_cor > DELTA)
//        XYZ_cor.z = HDR_REF_WHITE_XYZ.z * pow(f_Z_cor, 3.f);
//      else
//        XYZ_cor.z = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.z;
//
//      //convert to Yxy without the Y as it is unneeded
//      const float xyz = XYZ_cor.x + XYZ_cor.y + XYZ_cor.z;
//
//      x_cor = XYZ_cor.x /
//              xyz;
//      y_cor = XYZ_cor.y /
//              xyz;
//    }
//  }
//
//  //6.1.3 (inverse) part 2
//  //convert to XYZ if achromaitc correction is used
//  //and then to RGB
//  if (use_achromatic_correction && useful_correction)
//  {
//    const float X_hdr_cor = (x_cor / y_cor) * Y_hdr;
//    const float Z_hdr_cor = ((1.f - x_cor - y_cor) / y_cor) * Y_hdr;
//
//    hdr = float3(X_hdr_cor, Y_hdr, Z_hdr_cor);
//  }
//  hdr = mul(XYZ_To_BT2020, hdr);
//
//  //6.1.2 (inverse)
//  //inverse crosstalk matrix from 6.1.6
//  const float mlpha = 1.f - Alpha;
//  const float3x3 inverseCrosstalkMatrix =
//    mul(1.f / (1.f - 3.f * Alpha), float3x3( mlpha, -Alpha, -Alpha,
//                                            -Alpha,  mlpha, -Alpha,
//                                            -Alpha, -Alpha,  mlpha));
//  hdr = mul(inverseCrosstalkMatrix, hdr);
//
//  hdr = clamp(hdr / 10000.f, 0.f, 1.f);
//
//  return hdr;
//}

// outputs normalised values
float3 BT2446C_InverseToneMapping(
  const float3 Input,
  const float  SdrRelativeBrightness,
  const float  Alpha)
//        float  k1,
//        float  inf_point,
//  const bool   use_achromatic_correction,
//  const float  sigma)
{

  //103.2 =  400 nits
  //107.1 =  500 nits
  //110.1 =  600 nits
  //112.6 =  700 nits
  //114.8 =  800 nits
  //116.7 =  900 nits
  //118.4 = 1000 nits
  //153.7 is just under 10000 nits for Alpha == 0 and above it starts clipping
  float3 sdr = Input * SdrRelativeBrightness;

  //6.1.6 (inverse)
  //crosstalk matrix from 6.1.2
  //const float Alpha   = 0.f; //hardcode for now as it gives the best results imo
  const float xlpha = 1.f - 2.f * Alpha;
  const float3x3 crosstalkMatrix = float3x3(xlpha, Alpha, Alpha,
                                                    Alpha, xlpha, Alpha,
                                                    Alpha, Alpha, xlpha);

  sdr = mul(crosstalkMatrix, sdr);

  //6.1.5 (inverse)
  //conversion to XYZ and then Yxy -> x and y is at the end of the achromatic correction or the else case
  sdr = CSP::Mat::BT2020To::XYZ(sdr);
  const float Y_sdr = sdr.y;
  const float xyz   = sdr.x + sdr.y + sdr.z;
  const float x_sdr = sdr.x /
                      xyz;
  const float y_sdr = sdr.y /
                      xyz;

  //6.1.4 (inverse)
  //inverse tone mapping
  const float k1 = 83.802f; // multiplied by 100
  //const float k1 = 0.83802f;
  const float k2 = 15.09968f;
  const float k3 = 0.74204f;
  const float k4 = 78.99439f;
  //const float Y_hdr_ip = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
  const float Y_hdr_ip = 0.006984922394059541817564; // divided by 10000
  //k1 = 0.83802f;
  //k2 = 15.09968f;
  //k3 = 0.74204f;
  //k4 = 78.99439f;
  //const float Y_hlg_ref = 203.f;
  //const float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
  //const float Y_hdr_ip = inf_point / k1; // 58.5 = 0.80^2.4
  //const float k2 = inf_point * (1.f - k3);
  //const float k4 = inf_point - k2 * log(1.f - k3);
  //const float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / Y_hdr_ip);

  float Y_hdr = Y_sdr / k1;

  if (Y_hdr >= Y_hdr_ip)
    Y_hdr = (exp((Y_sdr * 100.f - k4) / k2) + k3) * Y_hdr_ip;

  //6.1.3 (inverse) part 1
  //convert to XYZ
  const float  X_hdr_uncor = (x_sdr / y_sdr) * Y_hdr;
  const float  Z_hdr_uncor = ((1.f - x_sdr - y_sdr) / y_sdr) * Y_hdr;
        float3 hdr         = float3(X_hdr_uncor, Y_hdr, Z_hdr_uncor);

//  bool  useful_correction = false;
//  float x_cor = 0.f;
//  float y_cor = 0.f;
//  // optional chroma correction above HDR Reference White (inverse)
//  if (use_achromatic_correction && sigma > 0.f)
//  {
//    // (3) inverse
//    // XYZ->L*a*b* from (1) which is actually wrong
//    // using correct conversion here
//    //==========================================================================
//    // it seems the ITU was trying to make a faster version for t <= POW_DELTA_3
//    // corrected version is here:
//    // L* = (116 * (t / POW_DELTA_3) - 16) / 10.f
//    // it's missing the division by 10 in the ITU doc
//
//    // get L*
//    const float t_Y = Y_hdr / HDR_REF_WHITE_XYZ.y;
//          float f_Y = 0.f;
//
//    if (t_Y > POW_DELTA_3)
//      f_Y = (pow(t_Y, 1.f / 3.f));
//    else
//      f_Y = (t_Y / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//
//    const float L_star = 116.f * f_Y - 16.f;
//
//    // get a*
//    const float t_X = sdr.x / HDR_REF_WHITE_XYZ.x;
//          float f_X = 0.f;
//
//    if (t_X > POW_DELTA_3)
//      f_X = (pow(t_Y, 1.f / 3.f));
//    else
//      f_X = (t_X / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//    f_X -= f_Y;
//
//    const float a_star = 116.f * f_X - 16.f;
//
//    // get b*
//    const float t_Z = sdr.z / HDR_REF_WHITE_XYZ.z;
//          float f_Z = 0.f;
//
//    if (t_Z > POW_DELTA_3)
//      f_Z = (pow(t_Y, 1.f / 3.f));
//    else
//      f_Z = (t_Z / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
//    f_Z = f_Y - f_Z;
//
//    const float b_star = 116.f * f_Z - 16.f;
//
//    // (2) chroma correction above Reference White
//    const float L_star_ref = 100.f;
//    const float L_star_max = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
//
//    // convert to CIELCh
//    const float C_star_ab = sqrt(pow(a_star, 2.f) + pow(b_star, 2.f));
//    const float h_ab      = atan(b_star / a_star);
//
//    float f_cor = 1.f;
//    if (L_star > L_star_ref)
//    {
//      f_cor = 1.f - sigma * (L_star     - L_star_ref) /
//                            (L_star_max - L_star_ref);
//      if (f_cor <= 0.f) // avoid division by zero
//        f_cor = 0.0000001f;
//
//      useful_correction = true;
//
//      // amazing function inversion!!
//      const float C_star_ab_cor = C_star_ab / f_cor;
//
//      // convert back to CIELAB
//      const float a_star_cor = C_star_ab_cor * cos(h_ab);
//      const float b_star_cor = C_star_ab_cor * sin(h_ab);
//
//      // (1) inverse
//      // conversion from L*a*b* to XYZ from (3) and then Yxy
//      float3 XYZ_cor;
//      const float f_Y_cor = (L_star + 16.f) /
//                            116.f;
//      const float f_X_cor = f_Y_cor + a_star_cor /
//                                      500.f;
//      const float f_Z_cor = f_Y_cor - b_star_cor /
//                                      200.f;
//
//      //X
//      if (f_X_cor > DELTA)
//        XYZ_cor.x = HDR_REF_WHITE_XYZ.x * pow(f_X_cor, 3);
//      else
//        XYZ_cor.x = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.x;
//
//      // can you just take the XYZ Y from the input here since it's unchanged?
//      // probably yes
//      //Y
//      if (f_Y_cor > DELTA)
//        XYZ_cor.y = HDR_REF_WHITE_XYZ.y * pow(f_Y_cor, 3.f);
//      else
//        XYZ_cor.y = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.y;
//
//      //Z
//      if (f_Z_cor > DELTA)
//        XYZ_cor.z = HDR_REF_WHITE_XYZ.z * pow(f_Z_cor, 3.f);
//      else
//        XYZ_cor.z = L_star * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.z;
//
//      //convert to Yxy without the Y as it is unneeded
//      const float xyz = XYZ_cor.x + XYZ_cor.y + XYZ_cor.z;
//
//      x_cor = XYZ_cor.x /
//              xyz;
//      y_cor = XYZ_cor.y /
//              xyz;
//    }
//  }

  //6.1.3 (inverse) part 2
  //convert to XYZ if achromaitc correction is used
  //and then to RGB
//  if (use_achromatic_correction && useful_correction)
//  {
//    const float X_hdr_cor = (x_cor / y_cor) * Y_hdr;
//    const float Z_hdr_cor = ((1.f - x_cor - y_cor) / y_cor) * Y_hdr;
//
//    hdr = float3(X_hdr_cor, Y_hdr, Z_hdr_cor);
//  }
  hdr = CSP::Mat::XYZTo::BT2020(hdr);

  //6.1.2 (inverse)
  //inverse crosstalk matrix from 6.1.6
  const float mlpha = 1.f - Alpha;
  const float3x3 inverseCrosstalkMatrix =
    mul(1.f / (1.f - 3.f * Alpha), float3x3(mlpha, -Alpha, -Alpha,
                                           -Alpha,  mlpha, -Alpha,
                                           -Alpha, -Alpha,  mlpha));
  hdr = mul(inverseCrosstalkMatrix, hdr);

  hdr = saturate(hdr);

  return hdr;
}



float LuminanceExpand(float Colour, float MaxNits, float ShoulderStart)
{
  float u = log(-((-MaxNits + Colour) / (MaxNits - ShoulderStart)));

  return -MaxNits * u + ShoulderStart * u + ShoulderStart;
}

float3 DiceInverseToneMapper(
  const float3 Input,
        float  MaxNits,
        float  ShoulderStart)
{

  float3x3 RGB_To_LMS = CSP::ICtCp::Mat::AP0_D65_To_LMS;
  float3x3 LMS_To_RGB = CSP::ICtCp::Mat::LMS_To_AP0_D65;
  float3   K_factors  = CSP::K_Helpers::AP0_D65::K;
  float    KR_helper  = CSP::K_Helpers::AP0_D65::KR;
  float    KB_helper  = CSP::K_Helpers::AP0_D65::KB;
  float2   KG_helper  = CSP::K_Helpers::AP0_D65::KG;


  float3 LMS = mul(RGB_To_LMS, Input);

  LMS = CSP::TRC::ToPq(LMS);

  const float I1 = 0.5f * LMS.x + 0.5f * LMS.y;

  if (I1 < ShoulderStart)
  {
    return Input;
  }
  else
  {
    const float Ct1 = dot(LMS, CSP::ICtCp::Mat::PQ_LMS_To_ICtCp[1]);
    const float Cp1 = dot(LMS, CSP::ICtCp::Mat::PQ_LMS_To_ICtCp[2]);

    const float I2 = LuminanceExpand(I1, MaxNits, ShoulderStart);

    const float min_I = min(min((I1 / I2), (I2 / I1)) * 1.1f, 1.f);

    //to L'M'S'
    LMS = CSP::ICtCp::Mat::ICtCpTo::PqLms(float3(I2, min_I * Ct1, min_I * Cp1));
    //to LMS
    LMS = CSP::TRC::FromPq(LMS);
    //to RGB
    return clamp(mul(LMS_To_RGB, LMS), 0.f, 65504.f);
  }

}

#endif
