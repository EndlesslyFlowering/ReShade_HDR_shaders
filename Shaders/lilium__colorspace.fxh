#define CSP_UNKNOWN 0
#define CSP_SRGB    1
#define CSP_SCRGB   2
#define CSP_PQ      3
#define CSP_HLG     4
#define CSP_PS5     5


//#define K_BT709  float3(0.2126f, 0.7152f, 0.0722f)
//#define K_BT2020 float3(0.2627f, 0.6780f, 0.0593f)

#define K_BT709   float3(0.212636821677324, 0.715182981841251,  0.0721801964814255)
#define K_BT2020  float3(0.262698338956556, 0.678008765772817,  0.0592928952706273)
#define K_AP0_D65 float3(0.343163015452697, 0.734695029446046, -0.0778580448987425)

#define KB_BT709_helper   1.85563960703715
#define KR_BT709_helper   1.57472635664535
#define KG_BT709_helper   float2(0.187281345942859, 0.468194596334655)

#define KB_BT2020_helper  1.88141420945875
#define KR_BT2020_helper  1.47460332208689
#define KG_BT2020_helper  float2(0.164532527178987, 0.571343414550845)

#define KB_AP0_D65_helper 2.15571608979748
#define KR_AP0_D65_helper 1.31367396909461
#define KG_AP0_D65_helper float2(-0.228448313084334, 0.613593807618545)

//#define KB_BT709_helper 1.8556f //2 - 2 * 0.0722
//#define KR_BT709_helper 1.5748f //2 - 2 * 0.2126
//#define KG_BT709_helper float2(0.187324272930648, 0.468124272930648)
//(0.0722/0.7152)*(2-2*0.0722), (0.2126/0.7152)*(2-2*0.2126)

//#define KB_BT2020_helper 1.8814f //2 - 2 * 0.0593
//#define KR_BT2020_helper 1.4746f //2 - 2 * 0.2627
//#define KG_BT2020_helper float2(0.164553126843658, 0.571353126843658)
//(0.0593/0.6780)*(2-2*0.0593), (0.2627/0.6780)*(2-2*0.2627)


//static const float3x3 IDENTITY =
//  float3x3(1.f, 0.f, 0.f,
//           0.f, 1.f, 0.f,
//           0.f, 0.f, 1.f);
//
//struct colorspace
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
struct colorspace
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
float sRGB_EOTF(const float C)
{
  if (C <= 0.04045f)
    return C / 12.92f;
  else
    return pow(((C + 0.055f) / 1.055f), 2.4f);
}

float3 sRGB_EOTF(const float3 colour)
{
  return float3(
    sRGB_EOTF(colour.r),
    sRGB_EOTF(colour.g),
    sRGB_EOTF(colour.b));
}

float sRGB_inverse_EOTF(const float C)
{
  if (C <= 0.0031308f)
    return C * 12.92f;
  else
    return 1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
}

float3 sRGB_inverse_EOTF(const float3 colour)
{
  return float3(
    sRGB_inverse_EOTF(colour.r),
    sRGB_inverse_EOTF(colour.g),
    sRGB_inverse_EOTF(colour.b));
}

//#define X_sRGB_1 1.19417654368084505707
//#define X_sRGB_x 0.039815307380813555
//#define X_sRGB_y_adjust 1.21290538811
// extended sRGB gamma including above 1 and below -1
float extended_sRGB_EOTF(const float C)
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

float3 extended_sRGB_EOTF(const float3 colour)
{
  return float3(
    extended_sRGB_EOTF(colour.r),
    extended_sRGB_EOTF(colour.g),
    extended_sRGB_EOTF(colour.b));
}

#define GAMMA_22 2.2f
#define INVERSE_GAMMA_22 1.f / GAMMA_22
//#define X_22_1 1.20237927370128566986
//#define X_22_x 0.0370133892172524
//#define X_22_y_adjust 1.5f - pow(X_22_x, INVERSE_GAMMA_22)
// extended gamma 2.2 including above 1 and below 0
float extended_22_EOTF(const float C)
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

float3 extended_22_EOTF(const float3 colour)
{
  return float3(
    extended_22_EOTF(colour.r),
    extended_22_EOTF(colour.g),
    extended_22_EOTF(colour.b));
}

#define GAMMA_24 2.4f
#define INVERSE_GAMMA_24 1.f / GAMMA_24
//#define X_24_1 1.1840535873752085849
//#define X_24_x 0.033138075
//#define X_24_y_adjust 1.5f - pow(X_24_x, INVERSE_GAMMA_24)
// extended gamma 2.4 including above 1 and below 0
float extended_24_EOTF(const float C)
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

float3 extended_24_EOTF(const float3 colour)
{
  return float3(
    extended_24_EOTF(colour.r),
    extended_24_EOTF(colour.g),
    extended_24_EOTF(colour.b));
}

//float X_power_EOTF(const float C, const float pow_gamma)
//{
//  const float pow_inverse_gamma = 1.f / pow_gamma;
//
//  if (C < -1)
//    return
//      -pow(-C, pow_inverse_gamma);
//  else if (C < 0)
//    return
//      -pow(-C, pow_gamma);
//  else if (C <= 1)
//    return
//      pow(C, pow_gamma);
//  else
//    return
//      pow(C, pow_inverse_gamma);
//}
//
//float3 X_power_EOTF(const float3 colour, const float pow_gamma)
//{
//  return float3(
//    X_power_EOTF(colour.r, pow_gamma),
//    X_power_EOTF(colour.g, pow_gamma),
//    X_power_EOTF(colour.b, pow_gamma));
//}

// gamma adjust including values above 1 and below 0
//TODO make this work like my custom gamma curve
float extended_gamma_adjust(const float C, const float adjust)
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

float3 extended_gamma_adjust(const float3 colour, const float adjust)
{
  return float3(
    extended_gamma_adjust(colour.r, adjust),
    extended_gamma_adjust(colour.g, adjust),
    extended_gamma_adjust(colour.b, adjust));
}

//ICtCp transforms

//L'M'S'->ICtCp
static const float3x3 LMS_to_ICtCp = float3x3(
  0.5,             0.5,             0.0,
  1.61376953125,  -3.323486328125,  1.709716796875,
  4.378173828125, -4.24560546875,  -0.132568359375);

//ICtCp->L'M'S'
static const float3x3 ICtCp_to_LMS = float3x3(
  1.0,  0.00860903703793276,  0.111029625003026,
  1.0, -0.00860903703793276, -0.111029625003026,
  1.0,  0.560031335710679,   -0.320627174987319);

//RGB BT.2020->LMS
static const float3x3 RGB_BT2020_to_LMS = float3x3(
  0.412109375,    0.52392578125,  0.06396484375,
  0.166748046875, 0.720458984375, 0.11279296875,
  0.024169921875, 0.075439453125, 0.900390625);

//LMS->RGB BT.2020
static const float3x3 LMS_to_RGB_BT2020 = float3x3(
   3.43660669433308,   -2.50645211865627,    0.0698454243231915,
  -0.791329555598929,   1.98360045179229,   -0.192270896193362,
  -0.0259498996905927, -0.0989137147117265,  1.12486361440232);

//AP0_D65 uses the highly accurate D65 white point
//RGB AP0_D65->LMS
static const float3x3 RGB_AP0_D65_to_LMS = float3x3(
  0.58056640625, 0.512451171875, -0.09326171875,
  0.19482421875, 0.80859375,     -0.00341796875,
  0.0322265625,  0.054931640625,  0.911865234375);

//LMS->RGB AP0_D65
static const float3x3 LMS_to_RGB_AP0_D65 = float3x3(
   2.17868661291322,   -1.39553812861432,    0.217595917625769,
  -0.525128926909500,   1.57276675119756,   -0.0478127217035243,
  -0.0453634871084088, -0.0454247619926001,  1.09184342737796);

//RGB transforms

static const float3x3 BT709_to_BT2020 = float3x3(
  0.627225305694944,  0.329476882715808,  0.0432978115892484,
  0.0690418812810714, 0.919605681354755,  0.0113524373641739,
  0.0163911702607078, 0.0880887513437058, 0.895520078395586);

static const float3x3 BT2020_to_BT709 = float3x3(
   1.66096379471340,   -0.588112737547978, -0.0728510571654192,
  -0.124477196529907,   1.13281946828499,  -0.00834227175508652,
  -0.0181571579858552, -0.100666415661988,  1.11882357364784);

static const float3x3 BT709_to_DCI_P3 = float3x3(
  0.822334429220561,  0.177665570779439,  0.000000000000000,
  0.0331661871416848, 0.966833812858315,  0.000000000000000,
  0.0170826010352503, 0.0724605600100221, 0.910456838954727);

//AP0 with highly accurate D65 white point instead of the custom white point from ACES which is around 6000K
static const float3x3 BT709_to_AP0_D65 = float3x3(
  0.433799790599445,  0.376466672803638, 0.189745486872426,
  0.0885578169506515, 0.809428607238992, 0.102029598497234,
  0.0177530102658331, 0.109561562944343, 0.872889419606533);

static const float3x3 AP0_D65_to_BT709 = float3x3(
   4.16950193129902,  -2.09370503254065,  -0.608571628857482,
  -1.94167536514522,   2.55258054992803,   0.220870975313075,
   0.158911053371001, -0.277807408599857,  0.952091082956431);

static const float3x3 BT2020_to_AP0_D65 = float3x3(
  0.670218991540148,    0.152244595663773,  0.177548363071589,
  0.0444833563345243,   0.854583550336869,  0.100949116015484,
  4.58778652504860e-17, 0.0258020508708645, 0.974401941945845);

static const float3x3 AP0_D65_to_BT2020 = float3x3(
   1.98236047774887,   -0.484236949434716,  -0.267716185193201,
  -1.49590141201230,    2.19966045032977,    0.171905827977915,
   0.0396112966005198, -0.0582467546449039,  0.862097728487838);

//to XYZ

static const float3x3 BT709_to_XYZ = float3x3(
  0.412135323426798,  0.357675002654190, 0.180356796374193,
  0.212507276141942,  0.715350005308380, 0.0721427185496773,
  0.0193188432856311, 0.119225000884730, 0.949879127570751);

static const float3x3 BT2020_to_XYZ = float3x3(
  0.636744702289598,    0.144643300793529,  0.168779119372055,
  0.262612221848252,    0.678121827837897,  0.0592659503138512,
  4.99243382266951e-17, 0.0280778172128614, 1.06034515452825);

static const float3x3 AP1_D65_to_XYZ = float3x3(
   0.647315060545070,   0.134403912261337,   0.168436775620020,
   0.266007451247834,   0.676092407132784,   0.0579001416193820,
  -0.00544725156138904, 0.00407284582610111, 1.08957539229201);

static const float3x3 AP0_D65_to_XYZ = float3x3(
  0.950054699026617, 0.000000000000000,  0.000101049399810263,
  0.343064531988242, 0.734743505865660, -0.0778080378539023,
  0.000000000000000, 0.000000000000000,  1.08820098655672);

//from XYZ

static const float3x3 XYZ_to_BT709 = float3x3(
   3.24297896532120,   -1.53833617585749,  -0.498919840818647,
  -0.968997952917093,   1.87549198225861,   0.0415445240532242,
   0.0556683243682128, -0.204117189350113,  1.05769816299604);

static const float3x3 XYZ_to_DCI_P3 = float3x3(
   2.49465568203257,   -0.931816447602876,  -0.402897930947739,
  -0.829302738210345,   1.76226831869698,    0.0236193817844718,
   0.0358679881475428, -0.0762194748135283,  0.957476016938569);

static const float3x3 XYZ_to_BT2020 = float3x3(
   1.71722636462073,   -0.355789953897356,  -0.253451173616083,
  -0.666562682837409,   1.61618623098933,    0.0157656680755665,
   0.0176505028477730, -0.0427964247130936,  0.942671667036796);

static const float3x3 XYZ_to_AP1 = float3x3(
   1.64102337969433,   -0.324803294184790,   -0.236424695237612,
  -0.663662858722983,   1.61533159165734,     0.0167563476855301,
   0.0117218943283754, -0.00828444199623741,  0.988394858539022);

static const float3x3 XYZ_to_AP0 = float3x3(
   1.04981101749797,  0.000000000000000, -0.0000974845405792529,
  -0.495903023077320, 1.37331304581571,   0.0982400360573100,
   0.000000000000000, 0.000000000000000,  0.991252018200499);


static const float3x3 myExp_BT709_to_BT2020 = float3x3(
  0.629890780672547,   0.335521112585212,  0.0345881067422407,
  0.0374561069306398,  0.956248944844986,  0.00629494822437453,
  0.00697011360541966, 0.0147953231418517, 0.978234563252729);

// START Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli
//static const float3x3 expanded_BT709_to_BT2020_matrix = float3x3(
//   0.6274040,  0.3292820, 0.0433136,
//   0.0457456,  0.941777,  0.0124772,
//  -0.00121055, 0.0176041, 0.983607);
// END Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli

//float3 ExpandColorGamutP3(float3 color, float start, float stop)
//{
//  // The original Rec.709 color, but rotated into the P3-D65 color space
//  float3 Rec709 = mul(BT709_to_DCI_P3, color);
//
//  // Treat the color as if it was originally mastered in the P3 color space
//  float3 P3 = color;
//
//  // Interpolate between Rec.709 and P3-D65, but only for bright HDR values, we don't want to change the overall look of the image
//  float lum = max(max(color.r, color.g), color.b);
//  float lerp = saturate((lum - start) / (stop - start));
//  float3 expandedColorInP3 = ((1.f - lerp) * Rec709) + (lerp * P3);
//
//  return expandedColorInP3;
//}
//
//float3 ExpandColorGamutBT2020(float3 color, float start, float stop)
//{
//  // The original Rec.709 color, but rotated into the BT2020 color space
//  float3 Rec709 = mul(BT709_to_BT2020, color);
//
//  // Treat the color as if it was originally mastered in the BT2020 color space
//  float3 BT2020 = color;
//
//  // Interpolate between Rec.709 and BT2020, but only for bright HDR values, we don't want to change the overall look of the image
//  float lum = max(max(color.r, color.g), color.b);
//  float lerp = saturate((lum - start) / (stop - start));
//  float3 expandedColorInBT2020 = ((1.f - lerp) * Rec709) + (lerp * BT2020);
//
//  return expandedColorInBT2020;
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
//static const colorspace csp_dci_p3 =
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
//static const colorspace csp_bt2020 =
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
//static const colorspace csp_ap1 =
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
//static const colorspace csp_ap1_d65 =
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
//static const colorspace csp_ap0 =
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
//static const colorspace csp_ap0_d65 =
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
//colorspace return_struct(float test)
//{
//  colorspace csp_bt709;
//  csp_bt709.can_ycbcr = true;
//  return csp_bt709;
//}
//


float3 ycbcr_bt2020_to_rgb(const float3 col)
{
  return float3(col.x + KR_BT2020_helper    * col.z,
                col.x - KG_BT2020_helper[0] * col.y - KG_BT2020_helper[1] * col.z,
                col.x + KB_BT2020_helper    * col.y);
}

float3 ycbcr_ap0_d65_to_rgb(const float3 col)
{
  return float3(col.x + KR_AP0_D65_helper    * col.z,
                col.x - KG_AP0_D65_helper[0] * col.y - KG_AP0_D65_helper[1] * col.z,
                col.x + KB_AP0_D65_helper    * col.y);
}

float3 rgb_bt2020_to_ycbcr(const float3 col)
{
  const float Y = dot(col, K_BT2020);
  return float3(Y,
                (col.b - Y) / KB_BT2020_helper,
                (col.r - Y) / KR_BT2020_helper);
}

float3 rgb_ap0_d65_to_ycbcr(const float3 col)
{
  const float Y = dot(col, K_AP0_D65);
  return float3(Y,
                (col.b - Y) / KB_AP0_D65_helper,
                (col.r - Y) / KR_AP0_D65_helper);
}


// Rec. ITU-R BT.2100-2 Table 4
static const float m1 =  0.1593017578125; //1305.f / 8192.f;
static const float m2 = 78.84375;         //2523.f / 32.f;
static const float c1 =  0.8359375;       // 107.f / 128.f;
static const float c2 = 18.8515625;       //2413.f / 128.f;
static const float c3 = 18.6875;          //2392.f / 128.f;

static const float one_div_m1 = 6.2773946360153256705;
static const float one_div_m2 = 0.01268331351565596512;

// Rec. ITU-R BT.2100-2 Table 4
// takes normalised values as input
float3 PQ_EOTF(float3 E_)
{
  E_ = clamp(E_, 0.f, 65504.f);

  const float3 E_pow_one_div_m2 = pow(E_, one_div_m2);

  //Y
  return pow(
             (max(E_pow_one_div_m2 - c1.xxx, 0.f.xxx)) /
             (c2.xxx - c3 * E_pow_one_div_m2)
         , one_div_m1);
}

// takes normalised values as input
float PQ_EOTF(float E_)
{
  E_ = clamp(E_, 0.f, 65504.f);

  const float E_pow_one_div_m2 = pow(E_, one_div_m2);

  //Y
  return pow(
             (max(E_pow_one_div_m2 - c1, 0.f)) /
             (c2 - c3 * E_pow_one_div_m2)
         , one_div_m1);
}

// Rec. ITU-R BT.2100-2 Table 4 (end)
// takes normalised values as input
float3 PQ_inverse_EOTF(float3 Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float3 Y_pow_m1 = pow(Y, m1);

  //E'
  return pow(
             ( c1.xxx + c2.xxx * Y_pow_m1) /
             (1.f.xxx + c3.xxx * Y_pow_m1)
         , m2);
}

// takes normalised values as input
float2 PQ_inverse_EOTF(float2 Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float2 Y_pow_m1 = pow(Y, m1);

  //E'
  return pow(
             ( c1.xx + c2.xx * Y_pow_m1) /
             (1.f.xx + c3.xx * Y_pow_m1)
         , m2);
}

// takes normalised values as input
float PQ_inverse_EOTF(float Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float Y_pow_m1 = pow(Y, m1);

  //E'
  return pow(
             ( c1 + c2 * Y_pow_m1) /
             (1.f + c3 * Y_pow_m1)
         , m2);
}

// takes nits as input
float3 PQ_OETF(const float3 Fd)
{
  const float3 Y = clamp(Fd / 10000.f, 0.f, 65504.f);

  const float3 Y_pow_m1 = pow(Y, m1);

  //E'
  return pow(
             (c1.xxx + c2 * Y_pow_m1) /
             (1.f.xxx + c3 * Y_pow_m1)
         , m2);
}

float nits_to_I(const float nits)
{
  float  nits_normalised  = nits / 10000.f;

  float3 nits_normalised3 =
    float3(nits_normalised, nits_normalised, nits_normalised);

  float2 LM_nits = float2(
    dot(nits_normalised3, RGB_AP0_D65_to_LMS[0]),
    dot(nits_normalised3, RGB_AP0_D65_to_LMS[1]));

  float2 LM_PQ = PQ_inverse_EOTF(LM_nits);

  return 0.5f * LM_PQ.x + 0.5f * LM_PQ.y;
}

float normalised_to_I(const float normalised)
{
  float3 normalised3 =
    float3(normalised, normalised, normalised);

  float2 LM_normalised = float2(
    dot(normalised3, RGB_AP0_D65_to_LMS[0]),
    dot(normalised3, RGB_AP0_D65_to_LMS[1]));

  float2 LM_PQ = PQ_inverse_EOTF(LM_normalised);

  return 0.5f * LM_PQ.x + 0.5f * LM_PQ.y;
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
