#pragma once

#define CSP_UNKNOWN 0
#define CSP_SRGB    1
#define CSP_SCRGB   2
#define CSP_PQ      3
#define CSP_HLG     4
#define CSP_PS5     5

#ifndef CSP_OVERRIDE
  #define CSP_OVERRIDE CSP_UNKNOWN
#endif

#if ((BUFFER_COLOR_SPACE == CSP_SRGB && CSP_OVERRIDE == CSP_UNKNOWN) \
  || (BUFFER_COLOR_SPACE == CSP_SRGB && CSP_OVERRIDE == CSP_SRGB)    \
  || (BUFFER_COLOR_SPACE != CSP_SRGB && CSP_OVERRIDE == CSP_SRGB))
  #define ACTUAL_COLOUR_SPACE CSP_SRGB
  #define FONT_BRIGHTNESS 1

#elif ((BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_UNKNOWN) \
    || (BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_SCRGB)   \
    || (BUFFER_COLOR_SPACE != CSP_SCRGB && CSP_OVERRIDE == CSP_SCRGB))
  #define ACTUAL_COLOUR_SPACE CSP_SCRGB
  #define FONT_BRIGHTNESS 2.5375f // 203.f / 80.f

#elif ((BUFFER_COLOR_SPACE == CSP_PQ && CSP_OVERRIDE == CSP_UNKNOWN) \
    || (BUFFER_COLOR_SPACE == CSP_PQ && CSP_OVERRIDE == CSP_PQ)      \
    || (BUFFER_COLOR_SPACE != CSP_PQ && CSP_OVERRIDE == CSP_PQ))
  #define ACTUAL_COLOUR_SPACE CSP_PQ
  #define FONT_BRIGHTNESS 0.58068888104160783796

#elif ((BUFFER_COLOR_SPACE == CSP_HLG && CSP_OVERRIDE == CSP_UNKNOWN) \
    || (BUFFER_COLOR_SPACE == CSP_HLG && CSP_OVERRIDE == CSP_HLG)     \
    || (BUFFER_COLOR_SPACE != CSP_HLG && CSP_OVERRIDE == CSP_HLG))
  #define ACTUAL_COLOUR_SPACE CSP_HLG
  #define FONT_BRIGHTNESS 0.69691214644230630735

#elif ((BUFFER_COLOR_SPACE == CSP_UNKNOWN && CSP_OVERRIDE == CSP_PS5)  \
    || (BUFFER_COLOR_SPACE != CSP_UNKNOWN && CSP_OVERRIDE == CSP_PS5))
  #define ACTUAL_COLOUR_SPACE CSP_PS5
  #define FONT_BRIGHTNESS 2.03f

#else
  #define ACTUAL_COLOUR_SPACE CSP_UNKNOWN
  #define FONT_BRIGHTNESS 1.f
#endif

//#define K_BT709  float3(0.2126f, 0.7152f, 0.0722f)
//#define K_BT2020 float3(0.2627f, 0.6780f, 0.0593f)

#define K_BT709   float3(0.212636821677324, 0.715182981841251,  0.0721801964814255)
#define K_BT2020  float3(0.262698338956556, 0.678008765772817,  0.0592928952706273)
#define K_AP0_D65 float3(0.343163015452697, 0.734695029446046, -0.0778580448987425)

#define KB_BT709_HELPER   1.85563960703715
#define KR_BT709_HELPER   1.57472635664535
#define KG_BT709_HELPER   float2(0.187281345942859, 0.468194596334655)

#define KB_BT2020_HELPER  1.88141420945875
#define KR_BT2020_HELPER  1.47460332208689
#define KG_BT2020_HELPER  float2(0.164532527178987, 0.571343414550845)

#define KB_AP0_D65_HELPER 2.15571608979748
#define KR_AP0_D65_HELPER 1.31367396909461
#define KG_AP0_D65_HELPER float2(-0.228448313084334, 0.613593807618545)

//#define KB_BT709_HELPER 1.8556f //2 - 2 * 0.0722
//#define KR_BT709_HELPER 1.5748f //2 - 2 * 0.2126
//#define KG_BT709_HELPER float2(0.187324272930648, 0.468124272930648)
//(0.0722/0.7152)*(2-2*0.0722), (0.2126/0.7152)*(2-2*0.2126)

//#define KB_BT2020_HELPER 1.8814f //2 - 2 * 0.0593
//#define KR_BT2020_HELPER 1.4746f //2 - 2 * 0.2627
//#define KG_BT2020_HELPER float2(0.164553126843658, 0.571353126843658)
//(0.0593/0.6780)*(2-2*0.0593), (0.2627/0.6780)*(2-2*0.2627)


//static const float3x3 IDENTITY =
//  float3x3(1.f, 0.f, 0.f,
//           0.f, 1.f, 0.f,
//           0.f, 0.f, 1.f);
//
//struct colourspace
//{
//  bool     can_ycbcr;
//
//  float3   k;
//  float    kb_helper;
//  float    kr_helper;
//  float2   kg_helper;
//
//  float3x3 to_xyz;
//  float3x3 to_bt709;
//  float3x3 to_dci_p3;
//  float3x3 to_bt2020;
//  float3x3 to_ap1;
//  float3x3 to_ap1_d65;
//  float3x3 to_ap0;
//  float3x3 to_ap0_d65;
//  float3x3 to_lms;
//
//  float3x3 from_xyz;
//  float3x3 from_bt709;
//  float3x3 from_dci_p3;
//  float3x3 from_bt2020;
//  float3x3 from_ap1;
//  float3x3 from_ap1_d65;
//  float3x3 from_ap0;
//  float3x3 from_ap0_d65;
//  float3x3 from_lms;
//};

/*
default:
struct colourspace
{
  can_ycbcr    = false;

  k            = float3(0.f, 0.f, 0.f);
  kb_helper    = 0.f;
  kr_helper    = 0.f;
  kg_helper    = float2(0.f, 0.f);

  to_xyz       = IDENTITY;
  to_bt709     = IDENTITY;
  to_dci_p3    = IDENTITY;
  to_bt2020    = IDENTITY;
  to_ap1       = IDENTITY;
  to_ap1_d65   = IDENTITY;
  to_ap0       = IDENTITY;
  to_ap0_d65   = IDENTITY;
  to_lms       = IDENTITY;

  from_xyz     = IDENTITY;
  from_bt709   = IDENTITY;
  from_dci_p3  = IDENTITY;
  from_bt2020  = IDENTITY;
  from_ap1     = IDENTITY;
  from_ap1_d65 = IDENTITY;
  from_ap0     = IDENTITY;
  from_ap0_d65 = IDENTITY;
  from_lms     = IDENTITY;
};
*/

//float posPow(float x, float y)
//{
//  pow(abs(x), abs)
//}

//linear->gamma compressed = inverse EOTF -> ^(1 / 2.2)
//
//gamma compressed->display (also linear) = EOTF -> ^(2.2)

// IEC 61966-2-1
float sRGB_TRC(const float C)
{
  if (C <= 0.04045f)
    return C / 12.92f;
  else
    return pow(((C + 0.055f) / 1.055f), 2.4f);
}

float3 sRGB_TRC(const float3 Colour)
{
  return float3(
    sRGB_TRC(Colour.r),
    sRGB_TRC(Colour.g),
    sRGB_TRC(Colour.b));
}

float sRGB_Inverse_TRC(const float C)
{
  if (C <= 0.0031308f)
    return C * 12.92f;
  else
    return 1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
}

float3 sRGB_Inverse_TRC(const float3 Colour)
{
  return float3(
    sRGB_Inverse_TRC(Colour.r),
    sRGB_Inverse_TRC(Colour.g),
    sRGB_Inverse_TRC(Colour.b));
}

//#define X_sRGB_1 1.19417654368084505707
//#define X_sRGB_x 0.039815307380813555
//#define X_sRGB_y_adjust 1.21290538811
// extended sRGB gamma including above 1 and below -1
float Extended_sRGB_TRC(const float C)
{
  if (C < -1.f)
    return
      -1.055f * pow(-C, (1.f / 2.4f)) + 0.055f;
  else if (C < -0.04045f)
    return
      -pow((-C + 0.055f) / 1.055f, 2.4f);
  else if (C <= 0.04045f)
    return
      C / 12.92f;
  else if (C <= 1.f)
    return
      pow((C + 0.055f) / 1.055f, 2.4f);
  else
    return
      1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
}
//{
//  if (C < -X_sRGB_1)
//    return
//      -1.055f * (pow(-C - X_sRGB_1 + X_sRGB_x, (1.f / 2.4f)) + X_sRGB_y_adjust) + 0.055f;
//  else if (C < -0.04045f)
//    return
//      -pow((-C + 0.055f) / 1.055f, 2.4f);
//  else if (C <= 0.04045f)
//    return
//      C / 12.92f;
//  else if (C <= X_sRGB_1)
//    return
//      pow((C + 0.055f) / 1.055f, 2.4f);
//  else
//    return
//      1.055f * (pow(C - X_sRGB_1 + X_sRGB_x, (1.f / 2.4f)) + X_sRGB_y_adjust) - 0.055f;
//}

float3 Extended_sRGB_TRC(const float3 Colour)
{
  return float3(
    Extended_sRGB_TRC(Colour.r),
    Extended_sRGB_TRC(Colour.g),
    Extended_sRGB_TRC(Colour.b));
}


float Extended_Inverse_sRGB_TRC(const float C)
{
  if (C < -1.f)
    return
      -pow((-C + 0.055f) / 1.055f, 2.4f);
  else if (C < -0.0031308f)
    return
      -1.055f * pow(-C, (1.f / 2.4f)) + 0.055f;
  else if (C <= 0.0031308f)
    return
      C * 12.92f;
  else if (C <= 1.f)
    return
      1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
  else
    return
      pow((C + 0.055f) / 1.055f, 2.4f);
}

float3 Extended_Inverse_sRGB_TRC(const float3 Colour)
{
  return float3(
    Extended_Inverse_sRGB_TRC(Colour.r),
    Extended_Inverse_sRGB_TRC(Colour.g),
    Extended_Inverse_sRGB_TRC(Colour.b));
}

static const float GAMMA_22         = 2.2f;
static const float INVERSE_GAMMA_22 = 1.f / 2.2f;
//#define X_22_1 1.20237927370128566986
//#define X_22_x 0.0370133892172524
//#define X_22_y_adjust 1.5f - pow(X_22_x, INVERSE_GAMMA_22)
// extended gamma 2.2 including above 1 and below 0
float Extended_22_TRC(const float C)
{
  if (C < -1.f)
    return
      -pow(-C, INVERSE_GAMMA_22);
  else if (C < 0.f)
    return
      -pow(-C, GAMMA_22);
  else if (C <= 1.f)
    return
      pow(C, GAMMA_22);
  else
    return
      pow(C, INVERSE_GAMMA_22);
}
//{
//  if (C < -X_22_1)
//    return
//      -(pow(-C - X_22_1 + X_22_x, INVERSE_GAMMA_22) + X_22_y_adjust);
//  else if (C < 0)
//    return
//      -pow(-C, GAMMA_22);
//  else if (C <= X_22_1)
//    return
//      pow(C, GAMMA_22);
//  else
//    return
//      (pow(C - X_22_1 + X_22_x, INVERSE_GAMMA_22) + X_22_y_adjust);
//}

float3 Extended_22_TRC(const float3 Colour)
{
  return float3(
    Extended_22_TRC(Colour.r),
    Extended_22_TRC(Colour.g),
    Extended_22_TRC(Colour.b));
}

static const float GAMMA_24         = 2.4f;
static const float INVERSE_GAMMA_24 = 1.f / 2.4f;
//#define X_24_1 1.1840535873752085849
//#define X_24_x 0.033138075
//#define X_24_y_adjust 1.5f - pow(X_24_x, INVERSE_GAMMA_24)
// extended gamma 2.4 including above 1 and below 0
float Extended_24_TRC(const float C)
{
  if (C < -1.f)
    return
      -pow(-C, INVERSE_GAMMA_24);
  else if (C < 0.f)
    return
      -pow(-C, GAMMA_24);
  else if (C <= 1.f)
    return
      pow(C, GAMMA_24);
  else
    return
      pow(C, INVERSE_GAMMA_24);
}
//{
//  if (C < -X_24_1)
//    return
//      -(pow(-C - X_24_1 + X_24_x, INVERSE_GAMMA_24) + X_24_y_adjust);
//  else if (C < 0)
//    return
//      -pow(-C, GAMMA_24);
//  else if (C <= X_24_1)
//    return
//      pow(C, GAMMA_24);
//  else
//    return
//      (pow(C - X_24_1 + X_24_x, INVERSE_GAMMA_24) + X_24_y_adjust);
//}

float3 Extended_24_TRC(const float3 Colour)
{
  return float3(
    Extended_24_TRC(Colour.r),
    Extended_24_TRC(Colour.g),
    Extended_24_TRC(Colour.b));
}

//float X_power_TRC(const float C, const float pow_gamma)
//{
//  const float pow_Inverse_gamma = 1.f / pow_gamma;
//
//  if (C < -1)
//    return
//      -pow(-C, pow_Inverse_gamma);
//  else if (C < 0)
//    return
//      -pow(-C, pow_gamma);
//  else if (C <= 1)
//    return
//      pow(C, pow_gamma);
//  else
//    return
//      pow(C, pow_Inverse_gamma);
//}
//
//float3 X_power_TRC(const float3 Colour, const float pow_gamma)
//{
//  return float3(
//    X_power_TRC(Colour.r, pow_gamma),
//    X_power_TRC(Colour.g, pow_gamma),
//    X_power_TRC(Colour.b, pow_gamma));
//}

// gamma adjust including values above 1 and below 0
//TODO make this work like my custom gamma curve
float ExtendedGammaAdjust(const float C, const float adjust)
{
  const float inverse_adjust = 1.f / adjust;

  if (C < -1.f)
    return
      -pow(-C, inverse_adjust);
  else if (C < 0.f)
    return
      -pow(-C, adjust);
  else if (C <= 1.f)
    return
      pow(C, adjust);
  else
    return
      pow(C, inverse_adjust);
}

float3 ExtendedGammaAdjust(const float3 Colour, const float adjust)
{
  return float3(
    ExtendedGammaAdjust(Colour.r, adjust),
    ExtendedGammaAdjust(Colour.g, adjust),
    ExtendedGammaAdjust(Colour.b, adjust));
}

//ICtCp transforms

//L'M'S'->ICtCp
static const float3x3 LMS_To_ICtCp = float3x3(
  0.5,             0.5,             0.0,
  1.61376953125,  -3.323486328125,  1.709716796875,
  4.378173828125, -4.24560546875,  -0.132568359375);

//ICtCp->L'M'S'
static const float3x3 ICtCp_To_LMS = float3x3(
  1.0,  0.00860903703793276,  0.111029625003026,
  1.0, -0.00860903703793276, -0.111029625003026,
  1.0,  0.560031335710679,   -0.320627174987319);

//RGB BT.2020->LMS
static const float3x3 RGB_BT2020_To_LMS = float3x3(
  0.412109375,    0.52392578125,  0.06396484375,
  0.166748046875, 0.720458984375, 0.11279296875,
  0.024169921875, 0.075439453125, 0.900390625);

//LMS->RGB BT.2020
static const float3x3 LMS_To_RGB_BT2020 = float3x3(
   3.43660669433308,   -2.50645211865627,    0.0698454243231915,
  -0.791329555598929,   1.98360045179229,   -0.192270896193362,
  -0.0259498996905927, -0.0989137147117265,  1.12486361440232);

//AP0_D65 uses the highly accurate D65 white point
//RGB AP0_D65->LMS
static const float3x3 RGB_AP0_D65_To_LMS = float3x3(
  0.58056640625, 0.512451171875, -0.09326171875,
  0.19482421875, 0.80859375,     -0.00341796875,
  0.0322265625,  0.054931640625,  0.911865234375);

//LMS->RGB AP0_D65
static const float3x3 LMS_To_RGB_AP0_D65 = float3x3(
   2.17868661291322,   -1.39553812861432,    0.217595917625769,
  -0.525128926909500,   1.57276675119756,   -0.0478127217035243,
  -0.0453634871084088, -0.0454247619926001,  1.09184342737796);

//RGB transforms

static const float3x3 BT709_To_BT2020 = float3x3(
  0.627225305694944,  0.329476882715808,  0.0432978115892484,
  0.0690418812810714, 0.919605681354755,  0.0113524373641739,
  0.0163911702607078, 0.0880887513437058, 0.895520078395586);

static const float3x3 BT2020_To_BT709 = float3x3(
   1.66096379471340,   -0.588112737547978, -0.0728510571654192,
  -0.124477196529907,   1.13281946828499,  -0.00834227175508652,
  -0.0181571579858552, -0.100666415661988,  1.11882357364784);

static const float3x3 BT709_To_DCI_P3 = float3x3(
  0.822334429220561,  0.177665570779439,  0.000000000000000,
  0.0331661871416848, 0.966833812858315,  0.000000000000000,
  0.0170826010352503, 0.0724605600100221, 0.910456838954727);

//AP0 with highly accurate D65 white point instead of the custom white point from ACES which is around 6000K
static const float3x3 BT709_To_AP0_D65 = float3x3(
  0.433799790599445,  0.376466672803638, 0.189745486872426,
  0.0885578169506515, 0.809428607238992, 0.102029598497234,
  0.0177530102658331, 0.109561562944343, 0.872889419606533);

static const float3x3 AP0_D65_To_BT709 = float3x3(
   4.16950193129902,  -2.09370503254065,  -0.608571628857482,
  -1.94167536514522,   2.55258054992803,   0.220870975313075,
   0.158911053371001, -0.277807408599857,  0.952091082956431);

static const float3x3 BT2020_To_AP0_D65 = float3x3(
  0.670218991540148,    0.152244595663773,  0.177548363071589,
  0.0444833563345243,   0.854583550336869,  0.100949116015484,
  4.58778652504860e-17, 0.0258020508708645, 0.974401941945845);

static const float3x3 AP0_D65_To_BT2020 = float3x3(
   1.98236047774887,   -0.484236949434716,  -0.267716185193201,
  -1.49590141201230,    2.19966045032977,    0.171905827977915,
   0.0396112966005198, -0.0582467546449039,  0.862097728487838);

//to XYZ

static const float3x3 BT709_To_XYZ = float3x3(
  0.412135323426798,  0.357675002654190, 0.180356796374193,
  0.212507276141942,  0.715350005308380, 0.0721427185496773,
  0.0193188432856311, 0.119225000884730, 0.949879127570751);

static const float3x3 BT2020_To_XYZ = float3x3(
  0.636744702289598,    0.144643300793529,  0.168779119372055,
  0.262612221848252,    0.678121827837897,  0.0592659503138512,
  4.99243382266951e-17, 0.0280778172128614, 1.06034515452825);

static const float3x3 AP1_D65_To_XYZ = float3x3(
   0.647315060545070,   0.134403912261337,   0.168436775620020,
   0.266007451247834,   0.676092407132784,   0.0579001416193820,
  -0.00544725156138904, 0.00407284582610111, 1.08957539229201);

static const float3x3 AP0_D65_To_XYZ = float3x3(
  0.950054699026617, 0.000000000000000,  0.000101049399810263,
  0.343064531988242, 0.734743505865660, -0.0778080378539023,
  0.000000000000000, 0.000000000000000,  1.08820098655672);

//from XYZ

static const float3x3 XYZ_To_BT709 = float3x3(
   3.24297896532120,   -1.53833617585749,  -0.498919840818647,
  -0.968997952917093,   1.87549198225861,   0.0415445240532242,
   0.0556683243682128, -0.204117189350113,  1.05769816299604);

static const float3x3 XYZ_To_DCI_P3 = float3x3(
   2.49465568203257,   -0.931816447602876,  -0.402897930947739,
  -0.829302738210345,   1.76226831869698,    0.0236193817844718,
   0.0358679881475428, -0.0762194748135283,  0.957476016938569);

static const float3x3 XYZ_To_BT2020 = float3x3(
   1.71722636462073,   -0.355789953897356,  -0.253451173616083,
  -0.666562682837409,   1.61618623098933,    0.0157656680755665,
   0.0176505028477730, -0.0427964247130936,  0.942671667036796);

static const float3x3 XYZ_To_AP1 = float3x3(
   1.64102337969433,   -0.324803294184790,   -0.236424695237612,
  -0.663662858722983,   1.61533159165734,     0.0167563476855301,
   0.0117218943283754, -0.00828444199623741,  0.988394858539022);

static const float3x3 XYZ_To_AP0 = float3x3(
   1.04981101749797,  0.000000000000000, -0.0000974845405792529,
  -0.495903023077320, 1.37331304581571,   0.0982400360573100,
   0.000000000000000, 0.000000000000000,  0.991252018200499);


static const float3x3 myExp_BT709_To_BT2020 = float3x3(
  0.629890780672547,   0.335521112585212,  0.0345881067422407,
  0.0374561069306398,  0.956248944844986,  0.00629494822437453,
  0.00697011360541966, 0.0147953231418517, 0.978234563252729);

// START Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli
//static const float3x3 expanded_BT709_To_BT2020_matrix = float3x3(
//   0.6274040,  0.3292820, 0.0433136,
//   0.0457456,  0.941777,  0.0124772,
//  -0.00121055, 0.0176041, 0.983607);
// END Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli

//float3 ExpandColourGamutP3(float3 colour, float start, float stop)
//{
//  // The original Rec.709 colour, but rotated into the P3-D65 colour space
//  float3 Rec709 = mul(BT709_To_DCI_P3, colour);
//
//  // Treat the colour as if it was originally mastered in the P3 colour space
//  float3 P3 = colour;
//
//  // Interpolate between Rec.709 and P3-D65, but only for bright HDR values, we don't want to change the overall look of the image
//  float lum = max(max(colour.r, colour.g), colour.b);
//  float lerp = saturate((lum - start) / (stop - start));
//  float3 expandedColourInP3 = ((1.f - lerp) * Rec709) + (lerp * P3);
//
//  return expandedColourInP3;
//}
//
//float3 ExpandColourGamutBT2020(float3 colour, float start, float stop)
//{
//  // The original Rec.709 colour, but rotated into the BT2020 colour space
//  float3 Rec709 = mul(BT709_To_BT2020, colour);
//
//  // Treat the colour as if it was originally mastered in the BT2020 colour space
//  float3 BT2020 = colour;
//
//  // Interpolate between Rec.709 and BT2020, but only for bright HDR values, we don't want to change the overall look of the image
//  float lum = max(max(colour.r, colour.g), colour.b);
//  float lerp = saturate((lum - start) / (stop - start));
//  float3 expandedColourInBT2020 = ((1.f - lerp) * Rec709) + (lerp * BT2020);
//
//  return expandedColourInBT2020;
//}

//static const struct csp_bt709
//{
//  can_ycbcr    = true,
//
//  k            = float3(0.212636821677324, 0.715182981841251, 0.0721801964814255),
//  kb_helper    = 1.85563960703715,
//  kr_helper    = 1.57472635664535,
//  kg_helper    = float2(0.187281345942859, 0.468194596334655),
//
//  to_xyz       = float3x3(0.412386563252992,  0.357591490920625, 0.180450491203564,
//                          0.212636821677324,  0.715182981841251, 0.0721801964814255,
//                          0.0193306201524840, 0.119197163640208, 0.950372587005435),
//
//  to_bt709     = IDENTITY,
//
//  to_dci_p3    = float3x3(0.822457548511777,  0.177542451488222,  0.000000000000000,
//                          0.0331932273885255, 0.966806772611475,  0.000000000000000,
//                          0.0170850449332782, 0.0724098641777013, 0.910505090889021),
//
//  to_bt2020    = float3x3(0.627401924722236,  0.329291971755002,  0.0433061035227622,
//                          0.0690954897392608, 0.919544281267395,  0.0113602289933443,
//                          0.0163937090881632, 0.0880281623979006, 0.895578128513936),
//
//  to_ap1       = IDENTITY,
//  to_ap1_d65   = IDENTITY,
//  to_ap0       = IDENTITY,
//
//  to_ap0_d65   = float3x3(0.433939666226453,  0.376270757528954, 0.189789576244594,
//                          0.0886176490106605, 0.809293012830817, 0.102089338158523,
//                          0.0177524231517299, 0.109465628662465, 0.872781948185805),
//
//  to_lms       = IDENTITY,
//
//
//  from_xyz     = float3x3( 3.24100323297636,   -1.53739896948879,  -0.498615881996363,
//                          -0.969224252202516,   1.87592998369518,   0.0415542263400847,
//                           0.0556394198519755, -0.204011206123910,  1.05714897718753),
//
//  from_bt2020  = float3x3( 1.66049621914783,   -0.587656444131135, -0.0728397750166941,
//                          -0.124547095586012,   1.13289510924730,  -0.00834801366128445,
//                          -0.0181536813870718, -0.100597371685743,  1.11875105307281),
//
//  from_ap0_d65 = float3x3( 2.55243581004094,   -1.12951938115888,  -0.422916428882053,
//                          -0.277330603707685,   1.37823643460965,  -0.100905830901963,
//                          -0.0171334337475196, -0.149886019090529,  1.16701945283805),
//};
//
//
//static const colourspace csp_dci_p3 =
//{
//  can_ycbcr    = false;
//
//  k            = float3(0.f, 0.f, 0.f);
//  kb_helper    = 0.f;
//  kr_helper    = 0.f;
//  kg_helper    = float2(0.f, 0.f);
//
//  to_xyz       = IDENTITY;
//  to_bt709     = IDENTITY;
//  to_dci_p3    = IDENTITY;
//  to_bt2020    = IDENTITY;
//  to_ap1       = IDENTITY;
//  to_ap1_d65   = IDENTITY;
//  to_ap0       = IDENTITY;
//  to_ap0_d65   = IDENTITY;
//  to_lms       = IDENTITY;
//
//  from_xyz     = float3x3( 2.49350912393461,   -0.931388179404778,  -0.402712756741651,
//                          -0.829473213929555,   1.76263057960030,    0.0236242371055886,
//                           0.0358512644339181, -0.0761839369220759,  0.957029586694311);
//};
//
//
//static const colourspace csp_bt2020 =
//{
//  can_ycbcr    = true;
//
//  k            = float3(0.262698338956556, 0.678008765772817, 0.0592928952706273);
//  kb_helper    = 1.88141420945875;
//  kr_helper    = 1.47460332208689;
//  kg_helper    = float2(0.164532527178987, 0.571343414550845);
//
//  to_xyz       = float3x3(0.636953506785074,    0.144619184669233,  0.168855853922873,
//                          0.262698338956556,    0.678008765772817,  0.0592928952706273,
//                          4.99407096644439e-17, 0.0280731358475570, 1.06082723495057);
//
//  to_bt709     = bt709.from_bt2020;
//
//  to_dci_p3    = IDENTITY;
//  to_bt2020    = IDENTITY;
//  to_ap1       = IDENTITY;
//  to_ap1_d65   = IDENTITY;
//  to_ap0       = IDENTITY;
//
//  to_ap0_d65   = float3x3(0.670246365605384,    0.152175527191681,  0.177578107202935,
//                          0.0445008795878928,   0.854497444583291,  0.101001675828816,
//                          4.58634334267322e-17, 0.0257811794360767, 0.974218820563924);
//
//  to_lms       = float3x3(0.412109375,    0.52392578125,  0.06396484375,
//                          0.166748046875, 0.720458984375, 0.11279296875,
//                          0.024169921875, 0.075439453125, 0.900390625);
//
//
//  from_xyz     = float3x3( 1.71666342779588,   -0.355673319730140, -0.253368087890248,
//                          -0.666673836198887,   1.61645573982470,   0.0157682970961337,
//                           0.0176424817849772, -0.0427769763827532, 0.942243281018431);
//
//  from_bt709   = bt709.to_bt2020;
//
//  from_ap0_d65 = float3x3( 1.98120359851493,   -0.484110148394926,  -0.267481115328003,
//                          -1.49600189517300,    2.20017241853874,    0.171935552888793,
//                           0.0395893535231033, -0.0582241265671916,  0.861149547243843);
//
//  from_lms     = float3x3( 3.43660669433308,   -2.50645211865627,    0.0698454243231915,
//                          -0.791329555598929,   1.98360045179229,   -0.192270896193362,
//                          -0.0259498996905927, -0.0989137147117265,  1.12486361440232);
//};
//
//
//static const colourspace csp_ap1 =
//{
//  can_ycbcr    = false;
//
//  k            = float3(0.f, 0.f, 0.f);
//  kb_helper    = 0.f;
//  kr_helper    = 0.f;
//  kg_helper    = float2(0.f, 0.f);
//
//
//  from_xyz     = float3x3( 1.64102337969433,   -0.324803294184790,   -0.236424695237612,
//                          -0.663662858722983,   1.61533159165734,     0.0167563476855301,
//                           0.0117218943283754, -0.00828444199623741,  0.988394858539022);
//};
//
//
//static const colourspace csp_ap1_d65 =
//{
//  can_ycbcr    = false;
//
//  k            = float3(0.f, 0.f, 0.f);
//  kb_helper    = 0.f;
//  kr_helper    = 0.f;
//  kg_helper    = float2(0.f, 0.f);
//
//
//  from_xyz     = float3x3( 0.647502080944762,   0.134381221854532,   0.168545242577887,
//                           0.266084305353177,   0.675978267510674,   0.0579374271361486,
//                          -0.00544882536559402, 0.00407215823801611, 1.09027703792571);
//};
//
//static const colourspace csp_ap0 =
//{
//  can_ycbcr    = false;
//
//  k            = float3(0.f, 0.f, 0.f);
//  kb_helper    = 0.f;
//  kr_helper    = 0.f;
//  kg_helper    = float2(0.f, 0.f);
//
//
//  from_xyz     = float3x3( 1.04981101749797,  0.000000000000000, -0.0000974845405792529,
//                          -0.495903023077320, 1.37331304581571,   0.0982400360573100,
//                           0.000000000000000, 0.000000000000000,  0.991252018200499);
//};
//
//
//static const colourspace csp_ap0_d65 =
//{
//  can_ycbcr    = true;
//
//  k            = float3(0.343163015452697, 0.734695029446046, -0.0778580448987425);
//  kb_helper    = 2.15571608979748;
//  kr_helper    = 1.31367396909461;
//  kg_helper    = float2(-0.228448313084334, 0.613593807618545);
//
//  to_xyz       = float3x3(0.950327431033156, 0.000000000000000,  0.000101114344024341,
//                          0.343163015452697, 0.734695029446046, -0.0778580448987425,
//                          0.000000000000000, 0.000000000000000,  1.08890037079813);
//
//  to_lms       = float3x3(0.580810546875, 0.512451171875, -0.09326171875,
//                          0.195068359375, 0.808349609375, -0.00341796875,
//                          0.0322265625,   0.054931640625,  0.91259765625);
//
//  from_lms     = float3x3( 2.17845648544721,   -1.39580019302982,    0.217396782969079,
//                          -0.525889627357037,   1.57372643877619,   -0.0478484931801823,
//                          -0.0452731647735950, -0.0454368173474335,  1.09097633376501);
//};
//
//colourspace return_struct(float test)
//{
//  colourspace csp_bt709;
//  csp_bt709.can_ycbcr = true;
//  return csp_bt709;
//}
//


float3 YCbCr_BT709_To_RGB(const float3 Colour)
{
  return float3(Colour.x + KR_BT709_HELPER    * Colour.z,
                Colour.x - KG_BT709_HELPER[0] * Colour.y - KG_BT709_HELPER[1] * Colour.z,
                Colour.x + KB_BT709_HELPER    * Colour.y);
}

float3 YCbCr_BT2020_To_RGB(const float3 Colour)
{
  return float3(Colour.x + KR_BT2020_HELPER    * Colour.z,
                Colour.x - KG_BT2020_HELPER[0] * Colour.y - KG_BT2020_HELPER[1] * Colour.z,
                Colour.x + KB_BT2020_HELPER    * Colour.y);
}

float3 YCbCr_AP0_D65_To_RGB(const float3 Colour)
{
  return float3(Colour.x + KR_AP0_D65_HELPER    * Colour.z,
                Colour.x - KG_AP0_D65_HELPER[0] * Colour.y - KG_AP0_D65_HELPER[1] * Colour.z,
                Colour.x + KB_AP0_D65_HELPER    * Colour.y);
}

float3 RGB_BT709_To_YCbCr(const float3 Colour)
{
  const float Y = dot(Colour, K_BT709);
  return float3(Y,
                (Colour.b - Y) / KB_BT709_HELPER,
                (Colour.r - Y) / KR_BT709_HELPER);
}

float3 RGB_BT2020_To_YCbCr(const float3 Colour)
{
  const float Y = dot(Colour, K_BT2020);
  return float3(Y,
                (Colour.b - Y) / KB_BT2020_HELPER,
                (Colour.r - Y) / KR_BT2020_HELPER);
}

float3 RGB_AP0_D65_To_YCbCr(const float3 Colour)
{
  const float Y = dot(Colour, K_AP0_D65);
  return float3(Y,
                (Colour.b - Y) / KB_AP0_D65_HELPER,
                (Colour.r - Y) / KR_AP0_D65_HELPER);
}


// Rec. ITU-R BT.2100-2 Table 4
static const float PQ_m1 =  0.1593017578125; //1305.f / 8192.f;
static const float PQ_m2 = 78.84375;         //2523.f / 32.f;
static const float PQ_c1 =  0.8359375;       // 107.f / 128.f;
static const float PQ_c2 = 18.8515625;       //2413.f / 128.f;
static const float PQ_c3 = 18.6875;          //2392.f / 128.f;

static const float _1_div_PQ_m1 = 6.2773946360153256705;
static const float _1_div_PQ_m2 = 0.01268331351565596512;

// Rec. ITU-R BT.2100-2 Table 4
// takes normalised values as input
float3 PQ_EOTF(float3 E_)
{
  E_ = clamp(E_, 0.f, 65504.f);

  const float3 E_pow_1_div_m2 = pow(E_, _1_div_PQ_m2);

  //Y
  return pow(
             (max(E_pow_1_div_m2 - PQ_c1.xxx, 0.f.xxx)) /
             (PQ_c2.xxx - PQ_c3 * E_pow_1_div_m2)
         , _1_div_PQ_m1);
}

// takes normalised values as input
float PQ_EOTF(float E_)
{
  E_ = clamp(E_, 0.f, 65504.f);

  const float E_pow_1_div_m2 = pow(E_, _1_div_PQ_m2);

  //Y
  return pow(
             (max(E_pow_1_div_m2 - PQ_c1, 0.f)) /
             (PQ_c2 - PQ_c3 * E_pow_1_div_m2)
         , _1_div_PQ_m1);
}

// Rec. ITU-R BT.2100-2 Table 4 (end)
// takes normalised values as input
float3 PQ_Inverse_EOTF(float3 Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float3 Y_pow_m1 = pow(Y, PQ_m1);

  //E'
  return pow(
             (PQ_c1.xxx + PQ_c2.xxx * Y_pow_m1) /
             (  1.f.xxx + PQ_c3.xxx * Y_pow_m1)
         , PQ_m2);
}

// takes normalised values as input
float2 PQ_Inverse_EOTF(float2 Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float2 Y_pow_m1 = pow(Y, PQ_m1);

  //E'
  return pow(
             (PQ_c1.xx + PQ_c2.xx * Y_pow_m1) /
             (  1.f.xx + PQ_c3.xx * Y_pow_m1)
         , PQ_m2);
}

// takes normalised values as input
float PQ_Inverse_EOTF(float Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float Y_pow_m1 = pow(Y, PQ_m1);

  //E'
  return pow(
             (PQ_c1 + PQ_c2 * Y_pow_m1) /
             (  1.f + PQ_c3 * Y_pow_m1)
         , PQ_m2);
}

// takes nits as input
float3 PQ_OETF(const float3 Fd)
{
  const float3 Y = clamp(Fd / 10000.f, 0.f, 65504.f);

  const float3 Y_pow_m1 = pow(Y, PQ_m1);

  //E'
  return pow(
             (PQ_c1.xxx + PQ_c2 * Y_pow_m1) /
             (  1.f.xxx + PQ_c3 * Y_pow_m1)
         , PQ_m2);
}


float Normalised_To_I_BT2020(const float normalised)
{
  float3 normalised3 =
    float3(normalised, normalised, normalised);

  float2 LM_normalised = float2(
    dot(normalised3, RGB_BT2020_To_LMS[0]),
    dot(normalised3, RGB_BT2020_To_LMS[1]));

  float2 LM_PQ = PQ_Inverse_EOTF(LM_normalised);

  return 0.5f * LM_PQ.x + 0.5f * LM_PQ.y;
}

float Nits_To_I_BT2020(const float nits)
{
  return Normalised_To_I_BT2020(nits / 10000.f);
}

float Normalised_To_I_AP0_D65(const float normalised)
{
  float3 normalised3 =
    float3(normalised, normalised, normalised);

  float2 LM_normalised = float2(
    dot(normalised3, RGB_AP0_D65_To_LMS[0]),
    dot(normalised3, RGB_AP0_D65_To_LMS[1]));

  float2 LM_PQ = PQ_Inverse_EOTF(LM_normalised);

  return 0.5f * LM_PQ.x + 0.5f * LM_PQ.y;
}

float Nits_To_I_AP0_D65(const float nits)
{
  return Normalised_To_I_AP0_D65(nits / 10000.f);
}


// Rec. ITU-R BT.2100-2 Table 5
static const float HLG_a = 0.17883277;
static const float HLG_b = 0.28466892; // 1 - 4 * HLG_a
static const float HLG_c = 0.55991072952956202016; // 0.5 - HLG_a * ln(4 * HLG_a)

// Rec. ITU-R BT.2100-2 Table 5 (end)
float HLG_EOTF(float x)
{
  if (x <= 0.5f)
  {
    return pow(x, 2.f) / 3.f;
  }
  else
  {
    return (exp((x - HLG_c) / HLG_a) + HLG_b) / 12.f;
  }
}

float3 HLG_EOTF(float3 x)
{
  return float3(HLG_EOTF(x.r),
                HLG_EOTF(x.g),
                HLG_EOTF(x.b));
}

// takes normalised values as input
float HLG_Inverse_EOTF(const float E)
{
  if (E <= (1.f / 12.f))
  {
    return (3.f * E);
  }
  else
  {
    return HLG_a * log(12.f * E - HLG_b) + HLG_c;
  }
}

float3 HLG_Inverse_EOTF(const float3 E)
{
  return float3(HLG_Inverse_EOTF(E.r),
                HLG_Inverse_EOTF(E.g),
                HLG_Inverse_EOTF(E.b));
}

// Rec. ITU-R BT.2100-2 Table 5
// takes nits as input
float HLG_OETF(float E)
{
  E = E / 1000.f;

  if (E <= (1.f / 12.f))
  {
    return (3.f * E);
  }
  else
  {
    return HLG_a * log(12.f * E - HLG_b) + HLG_c;
  }
}

float3 HLG_OETF(const float3 E)
{
  return float3(HLG_OETF(E.r),
                HLG_OETF(E.g),
                HLG_OETF(E.b));
}


bool IsNAN(const float input)
{
  if (isnan(input) || isinf(input))
    return true;
  else
    return false;
}

float fixNAN(const float input)
{
  if (IsNAN(input))
    return 0.f;
  else
    return input;
}

float3 fixNAN(float3 input)
{
  if (IsNAN(input.r))
    input.r = 0.f;
  else if (IsNAN(input.g))
    input.g = 0.f;
  else if (IsNAN(input.b))
    input.b = 0.f;

  return input;
}
