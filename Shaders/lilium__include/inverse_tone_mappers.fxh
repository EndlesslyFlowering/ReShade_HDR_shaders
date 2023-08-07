#pragma once

#include "colour_space.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


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
//      sdr = mul(Bt709ToBt2020, sdr);
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


namespace InverseToneMapping
{

  //gamma
  static const float gamma        = 2.4f;
  static const float inverseGamma = 1.f / gamma;

  // outputs normalised values
  float3 Bt2446a(
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
    const float3 ycbcrTmo = Csp::Ycbcr::FromRgb::Bt2020(sdr);

    // adjusted luma component (inverse)
    // get Y'sdr
    const float ySdr = ycbcrTmo.x + max(0.1f * ycbcrTmo.z, 0.f);

    // Tone mapping step 3 (inverse)
    // get Y'c
    const float pSdr = 1.f + 32.f * pow(
                                        Lsdr /
                                        10000.f
                                    , gamma);

    //Y'c
    //if pSdr == 1 there is a division by zero
    //this happens when Lsdr == 0
    const float yC = log((ySdr * (pSdr - 1)) + 1) /
                      log(pSdr); //log = ln

    // Tone mapping step 2 (inverse)
    // get Y'p
    float yP = 0.f;

    const float yP0 = yC / 1.0770f;
    const float yP2 = (yC - 0.5000f) /
                        0.5000f;
    //(4.83307641 - 4.604f * yC) == (pow(2.7811f, 2) - 4 * (-1.151f) * (-0.6302f - yC))
    const float yP11 = (-2.7811f + sqrt(4.83307641 - 4.604f * yC)) /
                          -2.302f;
    //yP12 is never reached
    //const float  yP12 = abs(
    //              (_1_first - _1_sqrt) /
    //              _1_div
    //            );

    if (yP0 <= 0.7399f)
      yP = yP0;
    else if (yP11 > 0.7399f
          && yP11 < 0.9909f)
      yP = yP11;
    //else if (yP12 > 0.7399f && yP12 < 0.9909f)
    //  yP = yP12;
    else if (yP2 >= 0.9909f)
      yP = yP2;
    else //yP11 sometimes (0.12% out of the full RGB range) is less than 0.7399f or more than 0.9909f
    {
      //error is small enough (less than 0.001) for this to be OK
      //ideally you would choose between yP0 and yP11 if yP11 < 0.7399f depending on which is closer to 0.7399f
      //or between yP11 and yP2 if yP11 > 0.9909f depending on which is closer to 0.9909f
      yP = yP11;

      //this clamps it to one float step above 0.7399f or one float step below 0.9909f
      //if (yP11 < 0.7399f)
      //  yP = 0.73990005f;
      //else
      //  yP = 0.9908999f;
    }

    // Tone mapping step 1 (inverse)
    // get Y'
    const float pHdr = 1.f + 32.f * pow(
                                        Lhdr /
                                        10000.f
                                    , gamma);
    //Y'hdr
    //if pHdr == 1 there is a division by zero
    //this happens when Lhdr == 0
    const float yHdr = (pow(pHdr, yP) - 1.f) /
                        (pHdr - 1.f);

    // Colour scaling function
    // TODO: analyse behaviour of colourScale being 1 or 0.00000001
    //float colourScale = 0.0000001f;
    //if (yHdr > 0.f && ySdr > 0.f) // avoid division by zero
    // this is actually fine to be infinite because it will create 0 as a result in the next step
    const float colourScale = ySdr /
                              (GamutExpansion * yHdr);

    // Colour difference signals (inverse) and Luma (inverse)
    // get R'G'B'

  //  hdr.b = ((C_b_tmo * 1.8814f) /
  //           colourScale) + yHdr;
  //  hdr.r = ((C_r_tmo * 1.4746f) /
  //           colourScale) + yHdr;
  //  hdr.g = (yHdr - (K_BT2020.r * hdr.r + K_BT2020.b * hdr.b)) /
  //          K_BT2020.g;

  //  produces the same results
    const float cbHdr = ycbcrTmo.y / colourScale;
    const float crHdr = ycbcrTmo.z / colourScale;

    float3 hdr = Csp::Ycbcr::ToRgb::Bt2020(float3(yHdr, cbHdr, crHdr));

    hdr = saturate(hdr); //on edge cases the YCbCr->RGB conversion isn't accurate enough

    // Non-linear transfer function (inverse)
    // get RGB
    hdr = pow(hdr, gamma + GammaIn + GammaOut);

    //expand to target luminance
    hdr *= (Lhdr / 10000.f);

    return hdr;
  } //Bt2446a


  // outputs normalised values
  float3 MapSdrIntoHdr(
    const float3 Input,
    const float  Brightness)
  {
    //map SDR into HDR
    return Input * (Brightness / 10000.f);
  } //MapSdrIntoHdr


  // HDR reference white in XYZ
  #define HDR_REF_WHITE_XYZ float3(192.93f, 203.f, 221.05f)
  #define DELTA             6.f / 29.f
  #define POW_DELTA_3       pow(DELTA, 3.f)
  #define _3_X_POW_DELTA_2  3.f * pow(DELTA, 2.f)

  //// outputs normalised values
  //float3 Bt2446c(
  //  const float3 Input,
  //  const float  SdrBrightness,
  //  const float  Alpha,
  ////        float  k1,
  ////        float  inf_point,
  //  const bool   UseAchromaticCorrection,
  //  const float  Sigma)
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
  //  float3 sdr = Input * (SdrBrightness > 153.7f
  //                      ? 153.7f
  //                      : SdrBrightness);
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
  //  sdr = mul(Bt2020ToXYZ, sdr);
  //  const float YSdr = sdr.y;
  //  const float xyz  = sdr.x + sdr.y + sdr.z;
  //  const float xSdr = sdr.x /
  //                     xyz;
  //  const float ySdr = sdr.y /
  //                     xyz;
  //
  //  //6.1.4 (inverse)
  //  //inverse tone mapping
  //  const float k1 = 0.83802f;
  //  const float k2 = 15.09968f;
  //  const float k3 = 0.74204f;
  //  const float k4 = 78.99439f;
  //  const float YHdrIp = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
  //  //k1 = 0.83802f;
  //  //k2 = 15.09968f;
  //  //k3 = 0.74204f;
  //  //k4 = 78.99439f;
  //  //const float Y_hlg_ref = 203.f;
  //  //const float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
  //  //const float YHdrIp = inf_point / k1; // 58.5 = 0.80^2.4
  //  //const float k2 = inf_point * (1.f - k3);
  //  //const float k4 = inf_point - k2 * log(1.f - k3);
  //  //const float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / YHdrIp);
  //
  //  float YHdr = YSdr / k1;
  //
  //  if (YHdr >= YHdrIp)
  //    YHdr = (exp((YSdr - k4) / k2) + k3) * YHdrIp;
  //
  //  //6.1.3 (inverse) part 1
  //  //convert to XYZ
  //  const float  XHdrUncorrected = (xSdr / ySdr) * YHdr;
  //  const float  ZHdrUncorrected = ((1.f - xSdr - ySdr) / ySdr) * YHdr;
  //        float3 hdr             = float3(XHdrUncorrected, YHdr, ZHdrUncorrected);
  //
  //  bool  usefulCorrection = false;
  //  float xCorrected = 0.f;
  //  float yCorrected = 0.f;
  //  // optional chroma correction above HDR Reference White (inverse)
  //  if (UseAchromaticCorrection && Sigma > 0.f)
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
  //    const float tY = YHdr / HDR_REF_WHITE_XYZ.y;
  //          float fY = 0.f;
  //
  //    if (tY > POW_DELTA_3) {
  //      fY = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fY = (tY / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //
  //    const float lStar = 116.f * fY - 16.f;
  //
  //    // get a*
  //    const float tX = sdr.x / HDR_REF_WHITE_XYZ.x;
  //          float fX = 0.f;
  //
  //    if (tX > POW_DELTA_3) {
  //      fX = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fX = (tX / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fX -= fY;
  //
  //    const float a_star = 116.f * fX - 16.f;
  //
  //    // get b*
  //    const float tZ = sdr.z / HDR_REF_WHITE_XYZ.z;
  //          float fZ = 0.f;
  //
  //    if (tZ > POW_DELTA_3) {
  //      fZ = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fZ = (tZ / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fZ = fY - fZ;
  //
  //    const float bStar = 116.f * fZ - 16.f;
  //
  //    // (2) chroma correction above Reference White
  //    const float lStarRef = 100.f;
  //    const float lStarMax = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
  //
  //    // convert to CIELCh
  //    const float cStarAb = sqrt(pow(a_star, 2.f) + pow(bStar, 2.f));
  //    const float hAb     = atan(bStar / a_star);
  //
  //    float fCorrected = 1.f;
  //    if (lStar > lStarRef)
  //    {
  //      fCorrected = 1.f - Sigma * (lStar    - lStarRef) /
  //                            (lStarMax - lStarRef);
  //      if (fCorrected <= 0.f) { // avoid division by zero
  //        fCorrected = 0.0000001f;
  //      }
  //
  //      usefulCorrection = true;
  //
  //      // amazing function inversion!!
  //      const float cStarAbCorrected = cStarAb / fCorrected;
  //
  //      // convert back to CIELAB
  //      const float aStarCorrected = cStarAbCorrected * cos(hAb);
  //      const float bStarCorrected = cStarAbCorrected * sin(hAb);
  //
  //      // (1) inverse
  //      // conversion from L*a*b* to XYZ from (3) and then Yxy
  //      float3 XYZCorrected;
  //      const float fYCorrected = (lStar + 16.f) /
  //                                116.f;
  //      const float fXCorrected = fYCorrected + aStarCorrected /
  //                                              500.f;
  //      const float fZCorrected = fYCorrected - bStarCorrected /
  //                                              200.f;
  //
  //      //X
  //      if (fXCorrected > DELTA) {
  //        XYZCorrected.x = HDR_REF_WHITE_XYZ.x * pow(fXCorrected, 3);
  //      }
  //      else {
  //        XYZCorrected.x = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.x;
  //      }
  //
  //      // can you just take the XYZ Y from the input here since it's unchanged?
  //      // probably yes
  //      //Y
  //      if (fYCorrected > DELTA) {
  //        XYZCorrected.y = HDR_REF_WHITE_XYZ.y * pow(fYCorrected, 3.f);
  //      }
  //      else {
  //        XYZCorrected.y = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.y;
  //      }
  //
  //      //Z
  //      if (fZCorrected > DELTA) {
  //        XYZCorrected.z = HDR_REF_WHITE_XYZ.z * pow(fZCorrected, 3.f);
  //      }
  //      else {
  //        XYZCorrected.z = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.z;
  //      }
  //
  //      //convert to Yxy without the Y as it is unneeded
  //      const float xyz = XYZCorrected.x + XYZCorrected.y + XYZCorrected.z;
  //
  //      xCorrected = XYZCorrected.x /
  //                   xyz;
  //      yCorrected = XYZCorrected.y /
  //                   xyz;
  //    }
  //  }
  //
  //  //6.1.3 (inverse) part 2
  //  //convert to XYZ if achromaitc correction is used
  //  //and then to RGB
  //  if (UseAchromaticCorrection && usefulCorrection)
  //  {
  //    const float XHdrCorrected = (xCorrected / yCorrected) * YHdr;
  //    const float ZHdrCorrected = ((1.f - xCorrected - yCorrected) / yCorrected) * YHdr;
  //
  //    hdr = float3(XHdrCorrected, YHdr, ZHdrCorrected);
  //  }
  //  hdr = mul(XYZToBt2020, hdr);
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
  float3 Bt2446c(
    const float3 Input,
    const float  SdrRelativeBrightness,
    const float  Alpha)
  //        float  k1,
  //        float  inf_point,
  //  const bool   UseAchromaticCorrection,
  //  const float  Sigma)
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
    sdr = Csp::Mat::Bt2020To::XYZ(sdr);
    const float YSdr = sdr.y;
    const float xyz  = sdr.x + sdr.y + sdr.z;
    const float xSdr = sdr.x /
                       xyz;
    const float ySdr = sdr.y /
                       xyz;

    //6.1.4 (inverse)
    //inverse tone mapping
    const float k1 = 83.802f; // multiplied by 100
    //const float k1 = 0.83802f;
    const float k2 = 15.09968f;
    const float k3 = 0.74204f;
    const float k4 = 78.99439f;
    //const float YHdrIp = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
    const float YHdrIp = 0.006984922394059541817564; // divided by 10000
    //k1 = 0.83802f;
    //k2 = 15.09968f;
    //k3 = 0.74204f;
    //k4 = 78.99439f;
    //const float Y_hlg_ref = 203.f;
    //const float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
    //const float YHdrIp = inf_point / k1; // 58.5 = 0.80^2.4
    //const float k2 = inf_point * (1.f - k3);
    //const float k4 = inf_point - k2 * log(1.f - k3);
    //const float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / YHdrIp);

    float YHdr = YSdr / k1;

    if (YHdr >= YHdrIp) {
      YHdr = (exp((YSdr * 100.f - k4) / k2) + k3) * YHdrIp;
    }

    //6.1.3 (inverse) part 1
    //convert to XYZ
    const float  XHdrUncorrected = (xSdr / ySdr) * YHdr;
    const float  ZHdrUncorrected = ((1.f - xSdr - ySdr) / ySdr) * YHdr;
          float3 hdr         = float3(XHdrUncorrected, YHdr, ZHdrUncorrected);

  //  bool  usefulCorrection = false;
  //  float xCorrected = 0.f;
  //  float yCorrected = 0.f;
  //  // optional chroma correction above HDR Reference White (inverse)
  //  if (UseAchromaticCorrection && Sigma > 0.f)
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
  //    const float tY = YHdr / HDR_REF_WHITE_XYZ.y;
  //          float fY = 0.f;
  //
  //    if (tY > POW_DELTA_3) {
  //      fY = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fY = (tY / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //
  //    const float lStar = 116.f * fY - 16.f;
  //
  //    // get a*
  //    const float tX = sdr.x / HDR_REF_WHITE_XYZ.x;
  //          float fX = 0.f;
  //
  //    if (tX > POW_DELTA_3) {
  //      fX = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fX = (tX / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fX -= fY;
  //
  //    const float a_star = 116.f * fX - 16.f;
  //
  //    // get b*
  //    const float tZ = sdr.z / HDR_REF_WHITE_XYZ.z;
  //          float fZ = 0.f;
  //
  //    if (tZ > POW_DELTA_3) {
  //      fZ = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fZ = (tZ / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fZ = fY - fZ;
  //
  //    const float bStar = 116.f * fZ - 16.f;
  //
  //    // (2) chroma correction above Reference White
  //    const float lStarRef = 100.f;
  //    const float lStarMax = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
  //
  //    // convert to CIELCh
  //    const float cStarAb = sqrt(pow(a_star, 2.f) + pow(bStar, 2.f));
  //    const float hAb      = atan(bStar / a_star);
  //
  //    float fCorrected = 1.f;
  //    if (lStar > lStarRef)
  //    {
  //      fCorrected = 1.f - Sigma * (lStar     - lStarRef) /
  //                                 (lStarMax - lStarRef);
  //      if (fCorrected <= 0.f) // avoid division by zero
  //        fCorrected = 0.0000001f;
  //
  //      usefulCorrection = true;
  //
  //      // amazing function inversion!!
  //      const float cStarAbCorrected = cStarAb / fCorrected;
  //
  //      // convert back to CIELAB
  //      const float aStarCorrected = cStarAbCorrected * cos(hAb);
  //      const float bStarCorrected = cStarAbCorrected * sin(hAb);
  //
  //      // (1) inverse
  //      // conversion from L*a*b* to XYZ from (3) and then Yxy
  //      float3 XYZCorrected;
  //      const float fYCorrected = (lStar + 16.f) /
  //                                116.f;
  //      const float fXCorrected = fYCorrected + aStarCorrected /
  //                                              500.f;
  //      const float fZCorrected = fYCorrected - bStarCorrected /
  //                                              200.f;
  //
  //      //X
  //      if (fXCorrected > DELTA) {
  //        XYZCorrected.x = HDR_REF_WHITE_XYZ.x * pow(fXCorrected, 3);
  //      }
  //      else {
  //        XYZCorrected.x = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.x;
  //      }
  //
  //      // can you just take the XYZ Y from the input here since it's unchanged?
  //      // probably yes
  //      //Y
  //      if (fYCorrected > DELTA) {
  //        XYZCorrected.y = HDR_REF_WHITE_XYZ.y * pow(fYCorrected, 3.f);
  //      }
  //      else {
  //        XYZCorrected.y = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.y;
  //      }
  //
  //      //Z
  //      if (fZCorrected > DELTA) {
  //        XYZCorrected.z = HDR_REF_WHITE_XYZ.z * pow(fZCorrected, 3.f);
  //      }
  //      else {
  //        XYZCorrected.z = lStar * _3_X_POW_DELTA_2 * HDR_REF_WHITE_XYZ.z;
  //      }
  //
  //      //convert to Yxy without the Y as it is unneeded
  //      const float xyz = XYZCorrected.x + XYZCorrected.y + XYZCorrected.z;
  //
  //      xCorrected = XYZCorrected.x /
  //                   xyz;
  //      yCorrected = XYZCorrected.y /
  //                   xyz;
  //    }
  //  }

    //6.1.3 (inverse) part 2
    //convert to XYZ if achromaitc correction is used
    //and then to RGB
  //  if (UseAchromaticCorrection && usefulCorrection)
  //  {
  //    const float XHdrCorrected = (xCorrected / yCorrected) * YHdr;
  //    const float ZHdrCorrected = ((1.f - xCorrected - yCorrected) / yCorrected) * YHdr;
  //
  //    hdr = float3(XHdrCorrected, YHdr, ZHdrCorrected);
  //  }

    hdr = Csp::Mat::XYZTo::Bt2020(hdr);

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
  } //Bt2446c


  namespace Dice
  {

    float LuminanceExpand(float Colour, float MaxNits, float ShoulderStart)
    {
      float u = log(-((-MaxNits + Colour) / (MaxNits - ShoulderStart)));

      return -MaxNits * u + ShoulderStart * u + ShoulderStart;
    } //LuminanceExpand

    float3 InverseToneMapper(
      const float3 Input,
            float  MaxNits,
            float  ShoulderStart)
    {

      float3x3 RgbToLms = Csp::Ictcp::Mat::Ap0D65ToLms;
      float3x3 LmsToRgb = Csp::Ictcp::Mat::LmsToAp0D65;
      float3   K_factors  = Csp::KHelpers::Ap0D65::K;
      float    KR_helper  = Csp::KHelpers::Ap0D65::Kr;
      float    KB_helper  = Csp::KHelpers::Ap0D65::Kb;
      float2   KG_helper  = Csp::KHelpers::Ap0D65::Kg;


      float3 LMS = mul(RgbToLms, Input);

      LMS = Csp::Trc::ToPq(LMS);

      const float I1 = 0.5f * LMS.x + 0.5f * LMS.y;

      if (I1 < ShoulderStart)
      {
        return Input;
      }
      else
      {
        const float Ct1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[1]);
        const float Cp1 = dot(LMS, Csp::Ictcp::Mat::PqLmsToIctcp[2]);

        const float I2 = LuminanceExpand(I1, MaxNits, ShoulderStart);

        const float min_I = min(min((I1 / I2), (I2 / I1)) * 1.1f, 1.f);

        //to L'M'S'
        LMS = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(I2, min_I * Ct1, min_I * Cp1));
        //to LMS
        LMS = Csp::Trc::FromPq(LMS);
        //to RGB
        return clamp(mul(LmsToRgb, LMS), 0.f, 65504.f);
      }
    } //InverseToneMapper

  }

}

#endif //is hdr API and hdr colour space
