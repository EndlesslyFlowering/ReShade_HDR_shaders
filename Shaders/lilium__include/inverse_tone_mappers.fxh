#pragma once

#include "colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_POSSIBLE_HDR_CSP))


//float3 gamut(
//  float3 Input,
//  uint   gamutExpansionType)
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


namespace Itmos
{
  // outputs normalised values
  float3 Bt2446A(
    float3 Input,
    float  Lhdr,
    float  Lsdr,
    float  InputNitsFactor,
    float  GamutExpansion,
    float  GammaIn,
    float  GammaOut)
  {
    float3 sdr = saturate(Input / InputNitsFactor);

    //RGB->R'G'B' gamma compression
    sdr = pow(sdr, 1.f / (Csp::Trc::RemoveGamma24 + GammaIn));

    // Rec. ITU-R BT.2020-2 Table 4
    //Y'C'bC'r,tmo
    float3 ycbcrTmo = Csp::Ycbcr::RgbTo::YcbcrBt2020(sdr);

    // adjusted luma component (inverse)
    // get Y'sdr
    float ySdr = ycbcrTmo.x + max(0.1f * ycbcrTmo.z, 0.f);

    // Tone mapping step 3 (inverse)
    // get Y'c
    float pSdr = 1.f + 32.f * pow(
                                  Lsdr /
                                  10000.f
                              , Csp::Trc::ApplyGamma24);

    //Y'c
    //if pSdr == 1 there is a division by zero
    //this happens when Lsdr == 0
    float yC = log((ySdr * (pSdr - 1)) + 1) /
               log(pSdr); //log = ln

    // Tone mapping step 2 (inverse)
    // get Y'p
    float yP = 0.f;

    float yP0 = yC / 1.0770f;
    float yP2 = (yC - 0.5000f) /
                0.5000f;
    //(4.83307641 - 4.604f * yC) == (pow(2.7811f, 2) - 4 * (-1.151f) * (-0.6302f - yC))
    float yP11 = (-2.7811f + sqrt(pow(2.7811f, 2) - 4 * (-1.151f) * (-0.6302f - yC))) /
                 -2.302f;
    //yP12 is never reached
    //float yP12 = abs((_1_first - _1_sqrt) /
    //                 _1_div);

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
    float pHdr = 1.f + 32.f * pow(
                                  Lhdr /
                                  10000.f
                              , Csp::Trc::ApplyGamma24);
    //Y'hdr
    //if pHdr == 1 there is a division by zero
    //this happens when Lhdr == 0
    float yHdr = (pow(pHdr, yP) - 1.f) /
                 (pHdr - 1.f);

    // Colour scaling function
    // TODO: analyse behaviour of colourScale being 1 or 0.00000001
    //float colourScale = 0.0000001f;
    //if (yHdr > 0.f && ySdr > 0.f) // avoid division by zero
    // this is actually fine to be infinite because it will create 0 as a result in the next step
    float colourScale = ySdr /
                        (GamutExpansion * yHdr);

    // Colour difference signals (inverse) and Luma (inverse)
    // get R'G'B'

  //  hdr.b = ((ycbcrTmo.y * Csp::KHelpers::Bt2020::Kb) /
  //           colourScale) + yHdr;
  //  hdr.r = ((ycbcrTmo.z * Csp::KHelpers::Bt2020::Kr) /
  //           colourScale) + yHdr;
  //  hdr.g = (yHdr - (Csp::KHelpers::Bt2020::K.r * hdr.r + Csp::KHelpers::Bt2020::K.b * hdr.b)) /
  //          Csp::KHelpers::Bt2020::K.g;

    // produces the same results
    float cbHdr = ycbcrTmo.y / colourScale;
    float crHdr = ycbcrTmo.z / colourScale;

    float3 hdr = Csp::Ycbcr::YcbcrTo::RgbBt2020(float3(yHdr, cbHdr, crHdr));

    hdr = max(hdr, 0.f); //on edge cases the YCbCr->RGB conversion isn't accurate enough

    // Non-linear transfer function (inverse)
    // get RGB
    hdr = pow(hdr, Csp::Trc::RemoveGamma24 + GammaIn + GammaOut);

    //expand to target luminance
    hdr *= (Lhdr / 10000.f);

    return hdr;
  } //Bt2446a


  // outputs normalised values
  float3 MapSdrIntoHdr(
    float3 Input,
    float  Brightness)
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
  //  float3 Input,
  //  float  SdrBrightness,
  //  float  Alpha,
  ////  float  k1,
  ////  float  inf_point,
  //  bool   UseAchromaticCorrection,
  //  float  Sigma)
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
  //  //float Alpha = 0.f; //hardcode for now as it gives the best results imo
  //  float xlpha = 1.f - 2.f * Alpha;
  //  float3x3 crosstalkMatrix = float3x3(xlpha, Alpha, Alpha,
  //                                      Alpha, xlpha, Alpha,
  //                                      Alpha, Alpha, xlpha);
  //
  //  sdr = mul(crosstalkMatrix, sdr);
  //
  //  //6.1.5 (inverse)
  //  //conversion to XYZ and then Yxy -> x and y is at the end of the achromatic correction or the else case
  //  sdr = mul(Bt2020ToXYZ, sdr);
  //  float YSdr = sdr.y;
  //  float xyz  = sdr.x + sdr.y + sdr.z;
  //  float xSdr = sdr.x /
  //               xyz;
  //  float ySdr = sdr.y /
  //               xyz;
  //
  //  //6.1.4 (inverse)
  //  //inverse tone mapping
  //  float k1 = 0.83802f;
  //  float k2 = 15.09968f;
  //  float k3 = 0.74204f;
  //  float k4 = 78.99439f;
  //  float YHdrIp = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
  //  //k1 = 0.83802f;
  //  //k2 = 15.09968f;
  //  //k3 = 0.74204f;
  //  //k4 = 78.99439f;
  //  //float Y_hlg_ref = 203.f;
  //  //float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
  //  //float YHdrIp = inf_point / k1; // 58.5 = 0.80^2.4
  //  //float k2 = inf_point * (1.f - k3);
  //  //float k4 = inf_point - k2 * log(1.f - k3);
  //  //float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / YHdrIp);
  //
  //  float YHdr = YSdr / k1;
  //
  //  if (YHdr >= YHdrIp)
  //    YHdr = (exp((YSdr - k4) / k2) + k3) * YHdrIp;
  //
  //  //6.1.3 (inverse) part 1
  //  //convert to XYZ
  //  float  XHdrUncorrected = (xSdr / ySdr) * YHdr;
  //  float  ZHdrUncorrected = ((1.f - xSdr - ySdr) / ySdr) * YHdr;
  //  float3 hdr             = float3(XHdrUncorrected, YHdr, ZHdrUncorrected);
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
  //    float tY = YHdr / HDR_REF_WHITE_XYZ.y;
  //    float fY = 0.f;
  //
  //    if (tY > POW_DELTA_3) {
  //      fY = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fY = (tY / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //
  //    float lStar = 116.f * fY - 16.f;
  //
  //    // get a*
  //    float tX = sdr.x / HDR_REF_WHITE_XYZ.x;
  //    float fX = 0.f;
  //
  //    if (tX > POW_DELTA_3) {
  //      fX = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fX = (tX / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fX -= fY;
  //
  //    float a_star = 116.f * fX - 16.f;
  //
  //    // get b*
  //    float tZ = sdr.z / HDR_REF_WHITE_XYZ.z;
  //    float fZ = 0.f;
  //
  //    if (tZ > POW_DELTA_3) {
  //      fZ = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fZ = (tZ / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fZ = fY - fZ;
  //
  //    float bStar = 116.f * fZ - 16.f;
  //
  //    // (2) chroma correction above Reference White
  //    float lStarRef = 100.f;
  //    float lStarMax = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
  //
  //    // convert to CIELCh
  //    float cStarAb = sqrt(pow(a_star, 2.f) + pow(bStar, 2.f));
  //    float hAb     = atan(bStar / a_star);
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
  //      float cStarAbCorrected = cStarAb / fCorrected;
  //
  //      // convert back to CIELAB
  //      float aStarCorrected = cStarAbCorrected * cos(hAb);
  //      float bStarCorrected = cStarAbCorrected * sin(hAb);
  //
  //      // (1) inverse
  //      // conversion from L*a*b* to XYZ from (3) and then Yxy
  //      float3 XYZCorrected;
  //      float fYCorrected = (lStar + 16.f) /
  //                          116.f;
  //      float fXCorrected = fYCorrected + aStarCorrected /
  //                                        500.f;
  //      float fZCorrected = fYCorrected - bStarCorrected /
  //                                        200.f;
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
  //      float xyz = XYZCorrected.x + XYZCorrected.y + XYZCorrected.z;
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
  //    float XHdrCorrected = (xCorrected / yCorrected) * YHdr;
  //    float ZHdrCorrected = ((1.f - xCorrected - yCorrected) / yCorrected) * YHdr;
  //
  //    hdr = float3(XHdrCorrected, YHdr, ZHdrCorrected);
  //  }
  //  hdr = mul(XYZToBt2020, hdr);
  //
  //  //6.1.2 (inverse)
  //  //inverse crosstalk matrix from 6.1.6
  //  float mlpha = 1.f - Alpha;
  //  float3x3 inverseCrosstalkMatrix =
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
    float3 Input,
    float  SdrRelativeBrightness,
    float  Alpha)
  //  float  k1,
  //  float  inf_point,
  //  bool   UseAchromaticCorrection,
  //  float  Sigma)
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
    //float Alpha = 0.f; //hardcode for now as it gives the best results imo
    float xlpha = 1.f - 2.f * Alpha;
    float3x3 crosstalkMatrix = float3x3(xlpha, Alpha, Alpha,
                                        Alpha, xlpha, Alpha,
                                        Alpha, Alpha, xlpha);

    sdr = mul(crosstalkMatrix, sdr);

    //6.1.5 (inverse)
    //conversion to XYZ and then Yxy -> x and y is at the end of the achromatic correction or the else case
    sdr = Csp::Mat::Bt2020To::XYZ(sdr);
    float YSdr = sdr.y;
    float xyz  = sdr.x + sdr.y + sdr.z;
    float xSdr = sdr.x /
                 xyz;
    float ySdr = sdr.y /
                 xyz;

    //6.1.4 (inverse)
    //inverse tone mapping
    float k1 = 83.802f; // multiplied by 100
    //float k1 = 0.83802f;
    float k2 = 15.09968f;
    float k3 = 0.74204f;
    float k4 = 78.99439f;
    //float YHdrIp = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
    float YHdrIp = 0.006984922394059541817564; // divided by 10000
    //k1 = 0.83802f;
    //k2 = 15.09968f;
    //k3 = 0.74204f;
    //k4 = 78.99439f;
    //float Y_hlg_ref = 203.f;
    //float Y_sdr_wp  = pow(0.96, 2.4f) * 100.f
    //float YHdrIp = inf_point / k1; // 58.5 = 0.80^2.4
    //float k2 = inf_point * (1.f - k3);
    //float k4 = inf_point - k2 * log(1.f - k3);
    //float k3 = -exp((Y_sdr_wp - k4) / k2) + (203.f / YHdrIp);

    float YHdr = YSdr / k1;

    if (YHdr >= YHdrIp) {
      YHdr = (exp((YSdr * 100.f - k4) / k2) + k3) * YHdrIp;
    }

    //6.1.3 (inverse) part 1
    //convert to XYZ
    float  XHdrUncorrected = (xSdr / ySdr) * YHdr;
    float  ZHdrUncorrected = ((1.f - xSdr - ySdr) / ySdr) * YHdr;

    float3 hdr = float3(XHdrUncorrected, YHdr, ZHdrUncorrected);

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
  //    float tY = YHdr / HDR_REF_WHITE_XYZ.y;
  //    float fY = 0.f;
  //
  //    if (tY > POW_DELTA_3) {
  //      fY = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fY = (tY / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //
  //    float lStar = 116.f * fY - 16.f;
  //
  //    // get a*
  //    float tX = sdr.x / HDR_REF_WHITE_XYZ.x;
  //    float fX = 0.f;
  //
  //    if (tX > POW_DELTA_3) {
  //      fX = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fX = (tX / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fX -= fY;
  //
  //    float a_star = 116.f * fX - 16.f;
  //
  //    // get b*
  //    float tZ = sdr.z / HDR_REF_WHITE_XYZ.z;
  //    float fZ = 0.f;
  //
  //    if (tZ > POW_DELTA_3) {
  //      fZ = (pow(tY, 1.f / 3.f));
  //    }
  //    else {
  //      fZ = (tZ / (3.f * pow(DELTA, 2.f)) + (16.f / 116.f));
  //    }
  //    fZ = fY - fZ;
  //
  //    float bStar = 116.f * fZ - 16.f;
  //
  //    // (2) chroma correction above Reference White
  //    float lStarRef = 100.f;
  //    float lStarMax = 116.f * pow(10000.f / 203.f, 1.f / 3.f) - 16.f; // hardcode to PQ max for now
  //
  //    // convert to CIELCh
  //    float cStarAb = sqrt(pow(a_star, 2.f) + pow(bStar, 2.f));
  //    float hAb      = atan(bStar / a_star);
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
  //      float cStarAbCorrected = cStarAb / fCorrected;
  //
  //      // convert back to CIELAB
  //      float aStarCorrected = cStarAbCorrected * cos(hAb);
  //      float bStarCorrected = cStarAbCorrected * sin(hAb);
  //
  //      // (1) inverse
  //      // conversion from L*a*b* to XYZ from (3) and then Yxy
  //      float3 XYZCorrected;
  //      float fYCorrected = (lStar + 16.f) /
  //                          116.f;
  //      float fXCorrected = fYCorrected + aStarCorrected /
  //                                        500.f;
  //      float fZCorrected = fYCorrected - bStarCorrected /
  //                                        200.f;
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
  //      float xyz = XYZCorrected.x + XYZCorrected.y + XYZCorrected.z;
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
  //    float XHdrCorrected = (xCorrected / yCorrected) * YHdr;
  //    float ZHdrCorrected = ((1.f - xCorrected - yCorrected) / yCorrected) * YHdr;
  //
  //    hdr = float3(XHdrCorrected, YHdr, ZHdrCorrected);
  //  }

    hdr = Csp::Mat::XYZTo::Bt2020(hdr);

    //6.1.2 (inverse)
    //inverse crosstalk matrix from 6.1.6
    float mlpha = 1.f - Alpha;
    float3x3 inverseCrosstalkMatrix =
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
      float3 Input,
      float  MaxNits,
      float  ShoulderStart)
    {

      float3x3 RgbToLms  = Ap0D65ToLms;
      float3x3 LmsToRgb  = LmsToAp0D65;
      float3   K_factors = KAp0D65;
      float    KR_helper = KrAp0D65;
      float    KB_helper = KbAp0D65;
      float2   KG_helper = KgAp0D65;


      float3 LMS = mul(RgbToLms, Input);

      LMS = Csp::Trc::LinearTo::Pq(LMS);

      float I1 = 0.5f * LMS.x + 0.5f * LMS.y;

      if (I1 < ShoulderStart)
      {
        return Input;
      }
      else
      {
        float Ct1 = dot(LMS, PqLmsToIctcp[1]);
        float Cp1 = dot(LMS, PqLmsToIctcp[2]);

        float I2 = LuminanceExpand(I1, MaxNits, ShoulderStart);

        float min_I = min(min((I1 / I2), (I2 / I1)) * 1.1f, 1.f);

        //to L'M'S'
        LMS = Csp::Ictcp::Mat::IctcpTo::PqLms(float3(I2, min_I * Ct1, min_I * Cp1));
        //to LMS
        LMS = Csp::Trc::PqTo::Linear(LMS);
        //to RGB
        return max(mul(LmsToRgb, LMS), 0.f);
      }
    } //InverseToneMapper

  }

}

#endif //is hdr API and hdr colour space
