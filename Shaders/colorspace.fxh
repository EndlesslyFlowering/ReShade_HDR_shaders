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

//RGB AP0_D65->LMS
static const float3x3 RGB_AP0_D65_to_LMS = float3x3(
  0.580810546875, 0.512451171875, -0.09326171875,
  0.195068359375, 0.808349609375, -0.00341796875,
  0.0322265625,   0.054931640625,  0.91259765625);

//LMS->RGB AP0_D65
static const float3x3 LMS_to_RGB_AP0_D65 = float3x3(
   2.17845648544721,   -1.39580019302982,    0.217396782969079,
  -0.525889627357037,   1.57372643877619,   -0.0478484931801823,
  -0.0452731647735950, -0.0454368173474335,  1.09097633376501);

//RGB transforms

static const float3x3 BT709_to_BT2020 = float3x3(
  0.627401924722236,  0.329291971755002,  0.0433061035227622,
  0.0690954897392608, 0.919544281267395,  0.0113602289933443,
  0.0163937090881632, 0.0880281623979006, 0.895578128513936);

static const float3x3 BT2020_to_BT709 = float3x3(
   1.66049621914783,   -0.587656444131135, -0.0728397750166941,
  -0.124547095586012,   1.13289510924730,  -0.00834801366128445,
  -0.0181536813870718, -0.100597371685743,  1.11875105307281);

//AP0 with D65 white point instead of the custom white point from ACES which is around 6000K
static const float3x3 BT709_to_AP0_D65 = float3x3(
  0.433939666226453,  0.376270757528954, 0.189789576244594,
  0.0886176490106605, 0.809293012830817, 0.102089338158523,
  0.0177524231517299, 0.109465628662465, 0.872781948185805);

static const float3x3 AP0_D65_to_BT709 = float3x3(
   2.55243581004094,   -1.12951938115888,  -0.422916428882053,
  -0.277330603707685,   1.37823643460965,  -0.100905830901963,
  -0.0171334337475196, -0.149886019090529,  1.16701945283805);

static const float3x3 BT709_to_XYZ = float3x3(
  0.412386563252992,  0.357591490920625, 0.180450491203564,
  0.212636821677324,  0.715182981841251, 0.0721801964814255,
  0.0193306201524840, 0.119197163640208, 0.950372587005435);

static const float3x3 BT2020_to_XYZ = float3x3(
  0.636953506785074,    0.144619184669233,  0.168855853922873,
  0.262698338956556,    0.678008765772817,  0.0592928952706273,
  4.99407096644439e-17, 0.0280731358475570, 1.06082723495057);

static const float3x3 AP0_D65_to_XYZ = float3x3(
  0.950327431033156, 0.000000000000000,  0.000101114344024341,
  0.343163015452697, 0.734695029446046, -0.0778580448987425,
  0.000000000000000, 0.000000000000000,  1.08890037079813);

static const float3x3 AP1_D65_to_XYZ = float3x3(
   0.647502080944762,   0.134381221854532,   0.168545242577887,
   0.266084305353177,   0.675978267510674,   0.0579374271361486,
  -0.00544882536559402, 0.00407215823801611, 1.09027703792571);

static const float3x3 XYZ_to_BT709 = float3x3(
  3.24100323297636,   -1.53739896948879,  -0.498615881996363,
 -0.969224252202516,   1.87592998369518,   0.0415542263400847,
  0.0556394198519755, -0.204011206123910,  1.05714897718753);

static const float3x3 XYZ_to_DCI_P3 = float3x3(
  2.49350912393461,   -0.931388179404778,  -0.402712756741651,
 -0.829473213929555,   1.76263057960030,    0.0236242371055886,
  0.0358512644339181, -0.0761839369220759,  0.957029586694311);

static const float3x3 XYZ_to_BT2020 = float3x3(
   1.71666342779588,   -0.355673319730140, -0.253368087890248,
  -0.666673836198887,   1.61645573982470,   0.0157682970961337,
   0.0176424817849772, -0.0427769763827532, 0.942243281018431);

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
static const float3x3 expanded_BT709_to_BT2020_matrix = float3x3(
   0.6274040,  0.3292820, 0.0433136,
   0.0457456,  0.941777,  0.0124772,
  -0.00121055, 0.0176041, 0.983607);
// END Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli

static const float3x3 BT709_to_DCI_P3 = float3x3(
  0.822457548511777,  0.177542451488222,  0.000000000000000,
  0.0331932273885255, 0.966806772611475,  0.000000000000000,
  0.0170850449332782, 0.0724098641777013, 0.910505090889021);


float3 ExpandColorGamutP3(float3 color, float start, float stop)
{
  // The original Rec.709 color, but rotated into the P3-D65 color space
  float3 Rec709 = mul(BT709_to_DCI_P3, color);

  // Treat the color as if it was originally mastered in the P3 color space
  float3 P3 = color;

  // Interpolate between Rec.709 and P3-D65, but only for bright HDR values, we don't want to change the overall look of the image
  float lum = max(max(color.r, color.g), color.b);
  float lerp = saturate((lum - start) / (stop - start));
  float3 expandedColorInP3 = ((1.f - lerp) * Rec709) + (lerp * P3);

  return expandedColorInP3;
}

float3 ExpandColorGamutBT2020(float3 color, float start, float stop)
{
  // The original Rec.709 color, but rotated into the BT2020 color space
  float3 Rec709 = mul(BT709_to_BT2020, color);

  // Treat the color as if it was originally mastered in the BT2020 color space
  float3 BT2020 = color;

  // Interpolate between Rec.709 and BT2020, but only for bright HDR values, we don't want to change the overall look of the image
  float lum = max(max(color.r, color.g), color.b);
  float lerp = saturate((lum - start) / (stop - start));
  float3 expandedColorInBT2020 = ((1.f - lerp) * Rec709) + (lerp * BT2020);

  return expandedColorInBT2020;
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
             (c1.xxx + c2 * Y_pow_m1) /
             (1.f.xxx + c3 * Y_pow_m1)
         , m2);
}

// takes normalised values as input
float PQ_inverse_EOTF(float Y)
{
  Y = clamp(Y, 0.f, 65504.f);

  const float Y_pow_m1 = pow(Y, m1);

  //E'
  return pow(
             (c1 + c2 * Y_pow_m1) /
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
