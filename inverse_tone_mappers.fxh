#include "colorspace.fxh"

float3 gamut(
  const float3 input,
  const uint   gamutExpansionType)
{
  float3 sdr = input;

  //BT.709->BT.2020 colorspace conversion
  switch (gamutExpansionType)
  {
    case 0:
      sdr = mul(bt709_to_bt2020_matrix, sdr);
      break;
    case 1:
      sdr = mul(myExp_bt709_to_bt2020, sdr);
      break;
    case 2:
      sdr = mul(expanded_bt709_to_bt2020_matrix, sdr);
      break;
    case 3:
      sdr = ExpandColorGamutBT2020(sdr, 1.0f, 5.0f);
      break;
  }

  return sdr;
}

float3 BT2446A_inverseToneMapping(
  const float3 input,
  const float  peakNits,
  const float  paperWhiteNits,
  const float  gammaIn,
  const float  gammaOut)
{
  float3 sdr = input;

  float Lsdr;
  if (paperWhiteNits > peakNits)
    Lsdr = peakNits;
  else
    Lsdr = paperWhiteNits;

  const float Lhdr = peakNits;

  //gamma
  const float inverseGamma = 2.4f;
  const float gamma = 1.f / inverseGamma;

  //RGB->R'G'B' gamma compression
  sdr = pow(sdr, 1.f / (inverseGamma + gammaIn));

  // Rec. ITU-R BT.2020-2 Table 4
  //Y'tmo
  const float Y_tmo = dot(sdr, K_BT2020);

  //C'b,tmo
  const float C_b_tmo = (sdr.b - Y_tmo) /
                        1.8814f;
  //C'r,tmo
  const float C_r_tmo = (sdr.r - Y_tmo) /
                        1.4746f;

  // adjusted luma component (inverse)
  // get Y'sdr
  const float Y_sdr = Y_tmo + max(0.1f * C_r_tmo, 0.f);

  // Tone mapping step 3 (inverse)
  // get Y'c
  const float pSDR = 1 + 32 * pow(
                                  Lsdr /
                                  10000.f
                              , gamma);

  //Y'c
  const float Y_c = log((Y_sdr * (pSDR - 1)) + 1) /
                    log(pSDR); //log = ln

  // Tone mapping step 2 (inverse)
  // get Y'p
  float Y_p = 0.f;

  const float Y_p_0 = Y_c / 1.0770f;
  const float Y_p_2 = (Y_c - 0.5000f) /
                      0.5000f;

  const float _1_first = -2.7811f;
  const float _1_sqrt  = sqrt(pow(2.7811, 2) - 4 * (-1.151f) * (-0.6302f - Y_c));
  const float _1_div   = -2.302f;
  const float Y_p_1_1  = (_1_first + _1_sqrt) /
                         _1_div;
  //Y_p_1_2 is never reached
  //const float  Y_p_1_2 = abs(
  //						  (_1_first - _1_sqrt) /
  //						  _1_div
  //					  );

  if (Y_p_0 <= 0.7399f)
    Y_p = Y_p_0;
  else if (Y_p_1_1 > 0.7399f
        && Y_p_1_1 < 0.9909f)
    Y_p = Y_p_1_1;
  //else if (Y_p_1_2 > 0.7399f && Y_p_1_2 < 0.9909f)
  //	Y_p = Y_p_1_2;
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
    //	Y_p = 0.73990005f;
    //else
    //	Y_p = 0.9908999f;
  }

  // Tone mapping step 1 (inverse)
  // get Y'
  const float pHDR = 1 + 32 * pow(
                                  Lhdr /
                                  10000.f
                              , gamma);
  //Y'
  const float Y_ = (pow(pHDR, Y_p) - 1) /
                   (pHDR - 1);

  // Colour scaling function
  float colScale = 0.f;
  if (Y_ > 0.f) // avoid division by zero
    colScale = Y_sdr /
               (1.1f * Y_);

  // Colour difference signals (inverse) and Luma (inverse)
  // get R'G'B'
  float3 hdr;
  hdr.b = ((C_b_tmo * 1.8814f) /
           colScale) + Y_;
  hdr.r = ((C_r_tmo * 1.4746f) /
           colScale) + Y_;
  hdr.g = (Y_ - (K_BT2020.r * hdr.r + K_BT2020.b * hdr.b)) /
          K_BT2020.g;

  //hdr = saturate(hdr);

  // Non-linear transfer function (inverse)
  // get RGB
  hdr = pow(hdr, inverseGamma + gammaIn + gammaOut);

  //expand target luminance to 10000nits
  hdr = hdr * Lhdr /
        10000.f;

  return hdr;
}

float3 mapSDRintoHDR(
  const float3 input,
  const float  paperWhiteNits,
  const bool   usePaperWhite)
{
  float3 sdr = input;

  float3 hdr;
  //map SDR into HDR
  if (usePaperWhite == false)
    hdr = sdr * 100.f / 10000.f;
  else
    hdr = sdr * paperWhiteNits / 10000.f;

  return hdr;
}


// HDR reference white in XYZ
static const float3 XnYnZn           = {192.93f, 203.f, 221.05f};
static const float  delta            = 6.f / 29.f;
static const float  pow_delta_3      = pow(delta, 3);
static const float  _3_x_pow_delta_2 = 3 * pow(delta, 2);

float3 BT2446C_inverseToneMapping(
  const float3 input,
  const float  sdrBrightness,
  const float  alpha,
//        float  k1,
//        float  infPoint,
  const bool   useAchromaticCorrection,
  const float  sigma)
{

  //103.2 =  400 nits
  //107.1 =  500 nits
  //110.1 =  600 nits
  //112.6 =  700 nits
  //114.8 =  800 nits
  //116.7 =  900 nits
  //118.4 = 1000 nits
  //153.7 is just under 10000 nits for alpha=0 and above it starts clipping
  float3 sdr = input * (sdrBrightness > 153.7f
                      ? 153.7f
                      : sdrBrightness);

  //6.1.6 (inverse)
  //crosstalk matrix from 6.1.2
  //const float alpha   = 0.f; //hardcode for now as it gives the best results imo
  const float xlpha = 1.f - 2.f * alpha;
  const float3x3 crosstalkMatrix = float3x3(
    xlpha, alpha, alpha,
    alpha, xlpha, alpha,
    alpha, alpha, xlpha);

  sdr = mul(crosstalkMatrix, sdr);

  //6.1.5 (inverse)
  //conversion to XYZ and then Yxy -> x and y is at the end of the achromatic correction or the else case
  sdr = mul(bt2020_to_XYZ, sdr);
  const float Ysdr  = sdr.y;
        float x_sdr = 0.f;
        float y_sdr = 0.f;

  // optional chroma correction above HDR Reference White
  if (useAchromaticCorrection)
  {
    // (3) inverse
    // XYZ->L*a*b* from (1) which is actually wrong
    // using correct conversion here
    //==========================================================================
    // it seems the ITU was trying to make a faster version for t <= pow_delta_3
    // corrected version is here:
    // L* = (116 * (t / pow_delta_3) - 16) / 10.f
    // it's missing the division by 10 in the ITU doc

    // get L*
    const float t_Y = sdr.y / XnYnZn.y;
          float f_Y = 0.f;

    if (t_Y > pow_delta_3)
      f_Y = (pow(t_Y, 1.f / 3.f));
    else
      f_Y = (t_Y / (3 * pow(delta, 2)) + (16.f / 116.f));

    const float L_star = 116 * f_Y - 16;

    // get a*
    const float t_X = sdr.x / XnYnZn.x;
          float f_X = 0.f;

    if (t_X > pow_delta_3)
      f_X = (pow(t_Y, 1.f / 3.f));
    else
      f_X = (t_X / (3 * pow(delta, 2)) + (16.f / 116.f));
    f_X -= f_Y;

    const float a_star = 116 * f_X - 16;

    // get b*
    const float t_Z = sdr.z / XnYnZn.z;
          float f_Z = 0.f;

    if (t_Z > pow_delta_3)
      f_Z = (pow(t_Y, 1.f / 3.f));
    else
      f_Z = (t_Z / (3 * pow(delta, 2)) + (16.f / 116.f));
    f_Z = f_Y - f_Z;

    const float b_star = 116 * f_Z - 16;

    // (2) chroma correction above Reference White
    const float L_star_ref = 100.f;
    const float L_star_max = 116 * pow(10000.f / 203.f, 1.f / 3.f) - 16; // hardcode to PQ max for now

    const float C_star_ab = sqrt(pow(a_star, 2) + pow(b_star, 2));
    const float h_ab      = atan(b_star / a_star);

    float f_cor = 1.f;
    if (L_star > L_star_ref)
    {
      f_cor = 1 - sigma * (L_star     - L_star_ref) /
                          (L_star_max - L_star_ref);
      if (f_cor < 0.f)
        f_cor = 0.f;
    }

    const float C_star_ab_cor = f_cor * C_star_ab;
    const float a_star_cor    = C_star_ab_cor * cos(h_ab);
    const float b_star_cor    = C_star_ab_cor * sin(h_ab);

    // (1) inverse
    // conversion from L*a*b* to XZY from (3) and then Yxy
    float3 XYZ_cor;
    const float f_Y_cor = (L_star + 16) /
                          116.f;
    const float f_X_cor = f_Y_cor + a_star_cor /
                                    500.f;
    const float f_Z_cor = f_Y_cor - b_star_cor /
                                    200.f;

    //X
    if (f_X_cor > delta)
      XYZ_cor.x = XnYnZn.x * pow(f_X_cor, 3);
    else
      XYZ_cor.x = L_star * _3_x_pow_delta_2 * XnYnZn.x;

    // can you just take the XYZ Y from the input here since it's unchanged?
    // probably yes
    //Y
    if (f_Y_cor > delta)
      XYZ_cor.y = XnYnZn.y * pow(f_Y_cor, 3);
    else
      XYZ_cor.y = L_star * _3_x_pow_delta_2 * XnYnZn.y;

    //Z
    if (f_Z_cor > delta)
      XYZ_cor.z = XnYnZn.z * pow(f_Z_cor, 3);
    else
      XYZ_cor.z= L_star * _3_x_pow_delta_2 * XnYnZn.z;

    //convert to Yxy without the Y as it is unneeded
    const float xyz = XYZ_cor.x + XYZ_cor.y + XYZ_cor.z;

    x_sdr = XYZ_cor.x /
            xyz;
    y_sdr = XYZ_cor.y /
            xyz;
  }
  else
  {
    //x and y from 6.1.5
    const float xyz = sdr.x + sdr.y + sdr.z;

    x_sdr = sdr.x /
            xyz;
    y_sdr = sdr.y /
            xyz;
  }

  //6.1.4 (inverse)
  //inverse tone mapping
  const float k1 = 0.83802f;
  const float k2 = 15.09968f;
  const float k3 = 0.74204f;
  const float k4 = 78.99439f;
  const float Yhdr_ip = 69.84922394059541817564; // 58.535046646 / k1; // 58.5 = 0.80^2.4
  //k1 = 0.83802f;
  //k2 = 15.09968f;
  //k3 = 0.74204f;
  //k4 = 78.99439f;
  //const float Yhlg_ref = 203.f;
  //const float Ysdr_wp  = pow(0.96, 2.4f) * 100.f
  //const float Yhdr_ip = infPoint / k1; // 58.5 = 0.80^2.4
  //const float k2 = infPoint * (1.f - k3);
  //const float k4 = infPoint - k2 * log(1.f - k3);
  //const float k3 = -exp((Ysdr_wp - k4) / k2) + (203.f / Yhdr_ip);

  float Yhdr = 0.f;
  const float Yhdr_0 = Ysdr / k1;
  const float Yhdr_1 = (exp((Ysdr - k4) / k2) + k3) * Yhdr_ip;

  if (Yhdr_0 < Yhdr_ip)
    Yhdr = Yhdr_0;
  else
    Yhdr = Yhdr_1;

  //6.1.3 (inverse)
  //convert to XYZ and then to RGB
  const float Xhdr = (x_sdr / y_sdr) * Yhdr;
  const float Zhdr = ((1.f - x_sdr - y_sdr) / y_sdr) * Yhdr;
  float3 hdr = float3(Xhdr, Yhdr, Zhdr);
  hdr = mul(XYZ_to_bt2020, hdr);

  //6.1.2 (inverse)
  //inverse crosstalk matrix from 6.1.6
  const float mlpha = 1.f - alpha;
  const float3x3 inverseCrosstalkMatrix = float3x3(
     mlpha, -alpha, -alpha,
    -alpha,  mlpha, -alpha,
    -alpha, -alpha,  mlpha);
  hdr = mul(mul(1.f / 1.f - 3.f * alpha, inverseCrosstalkMatrix), hdr);

  //map into 10000 nits
  hdr = hdr / 10000.f;

  //safety
  //hdr = saturate(hdr);

  return hdr;
}