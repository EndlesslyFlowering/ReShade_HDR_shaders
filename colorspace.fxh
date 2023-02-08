#undef CSP_UNKNOWN
#undef CSP_SRGB
#undef CSP_SCRGB
#undef CSP_PQ
#undef CSP_HLG

#ifdef BUFFER_COLOR_SPACE
  #if BUFFER_COLOR_SPACE == 0
    #define CSP_UNKNOWN true
    #define CSP_SRGB    false
    #define CSP_SCRGB   false
    #define CSP_PQ      false
    #define CSP_HLG     false
  #elif BUFFER_COLOR_SPACE == 1
    #define CSP_UNKNOWN false
    #define CSP_SRGB    true
    #define CSP_SCRGB   false
    #define CSP_PQ      false
    #define CSP_HLG     false
  #elif BUFFER_COLOR_SPACE == 2
    #define CSP_UNKNOWN false
    #define CSP_SRGB    false
    #define CSP_SCRGB   true
    #define CSP_PQ      false
    #define CSP_HLG     false
  #elif BUFFER_COLOR_SPACE == 3
    #define CSP_UNKNOWN false
    #define CSP_SRGB    false
    #define CSP_SCRGB   false
    #define CSP_PQ      true
    #define CSP_HLG     false
  #elif BUFFER_COLOR_SPACE == 4
    #define CSP_UNKNOWN false
    #define CSP_SRGB    false
    #define CSP_SCRGB   false
    #define CSP_PQ      false
    #define CSP_HLG     true
  #endif
#else
  #define CSP_UNKNOWN true
  #define CSP_SRGB    false
  #define CSP_SCRGB   false
  #define CSP_PQ      false
  #define CSP_HLG     false
#endif

#define K_BT709  float3(0.2126f, 0.7152f, 0.0722f)
#define K_BT2020 float3(0.2627f, 0.6780f, 0.0593f)


//float posPow(float x, float y)
//{
//  pow(abs(x), abs)
//}

// IEC 61966-2-1
float sRGB_inverse_EOTF_single(const float C)
{
  if (C <= 0.04045f)
    return C / 12.92f;
  else
    return pow(((C + 0.055f) / 1.055f), 2.4f);
}

float3 sRGB_inverse_EOTF(const float3 colour)
{
  return float3(
    sRGB_inverse_EOTF_single(colour.r),
    sRGB_inverse_EOTF_single(colour.g),
    sRGB_inverse_EOTF_single(colour.b));
}

float XsRGB_inverse_EOTF_single(const float C)
{
  if (C < -0.04045)
    return
      -pow((-C + 0.055) / 1.055, 2.4);
  else if (C <= 0.04045)
    return
      C / 12.92;
  else if (C <= 1)
    return
      pow((C + 0.055) / 1.055, 2.4);
  else
    return
      1.055 * pow(C, (1 / 2.4)) - 0.055;
}

float3 XsRGB_inverse_EOTF(const float3 colour)
{
  return float3(
    XsRGB_inverse_EOTF_single(colour.r),
    XsRGB_inverse_EOTF_single(colour.g),
    XsRGB_inverse_EOTF_single(colour.b));
}

// Rep. ITU-R BT.2407 �2.2 calculated in double
static const float3x3 bt709_to_bt2020_matrix = float3x3(
  0.627401924722236,  0.329291971755002,  0.0433061035227622,
  0.0690954897392608, 0.919544281267395,  0.0113602289933443,
  0.0163937090881632, 0.0880281623979006, 0.895578128513936);

static const float3x3 bt709_to_XYZ = float3x3(
  0.412386563252992,  0.357591490920625, 0.180450491203564,
  0.212636821677324,  0.715182981841251, 0.0721801964814255,
  0.0193306201524840, 0.119197163640208, 0.950372587005435);

static const float3x3 bt2020_to_XYZ = float3x3(
  0.636953506785074,    0.144619184669233,  0.168855853922873,
  0.262698338956556,    0.678008765772817,  0.0592928952706273,
  4.99407096644439e-17, 0.0280731358475570, 1.06082723495057);

static const float3x3 XYZ_to_bt2020 = float3x3(
   1.71666342779588,   -0.355673319730140, -0.253368087890248,
  -0.666673836198887,   1.61645573982470,   0.0157682970961337,
   0.0176424817849772, -0.0427769763827532, 0.942243281018431);

static const float3x3 myExp_bt709_to_bt2020 = float3x3(
  0.629890780672547,   0.335521112585212,  0.0345881067422407,
  0.0374561069306398,  0.956248944844986,  0.00629494822437453,
  0.00697011360541966, 0.0147953231418517, 0.978234563252729);

// START Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli
static const float3x3 expanded_bt709_to_bt2020_matrix = float3x3(
   0.6274040,  0.3292820, 0.0433136,
   0.0457456,  0.941777,  0.0124772,
  -0.00121055, 0.0176041, 0.983607);
// END Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli

// Rec. ITU-R BT.2087 Annex 1 M2 calculated in double
// clamp to (0, 1) afterwards
static const float3x3 bt2020_to_bt709_matrix = float3x3(
   1.66049621914783,   -0.587656444131135, -0.0728397750166941,
  -0.124547095586012,   1.13289510924730,  -0.00834801366128445,
  -0.0181536813870718, -0.100597371685743,  1.11875105307281);

static const float3x3 from709toP3_D65 = float3x3(
  0.822457548511778,  0.177542451488222,  0.000000000000000,
  0.0331932273885254, 0.966806772611475, -1.38777878078145e-17,
  0.0170850449332782, 0.0724098641777013, 0.910505090889021);


float3 ExpandColorGamutP3(float3 color, float start, float stop)
{
  // The original Rec.709 color, but rotated into the P3-D65 color space
  float3 Rec709 = mul(from709toP3_D65, color);

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
  float3 Rec709 = mul(bt709_to_bt2020_matrix, color);

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
float3 PQ_EOTF(const float3 E_)
{
  const float3 E_pow_one_div_m2 = pow(E_, one_div_m2);

  const float3 Y =
    pow(
        (max(E_pow_one_div_m2 - c1.xxx, 0.f.xxx)) /
        (c2.xxx - c3 * E_pow_one_div_m2)
    , one_div_m1);

  return Y;
}

float PQ_EOTF_single(const float E_)
{
  const float E_pow_one_div_m2 = pow(E_, one_div_m2);

  const float Y =
    pow(
        (max(E_pow_one_div_m2 - c1, 0.f)) /
        (c2 - c3 * E_pow_one_div_m2)
    , one_div_m1);

  return Y;
}

// Rec. ITU-R BT.2100-2 Table 4 (end)
float3 PQ_inverse_EOTF(const float3 Y)
{
  const float3 Y_pow_m1 = pow(Y, m1);

  //E'
  const float3 E_ =
    pow(
        (c1.xxx + c2 * Y_pow_m1) /
        (1.f.xxx + c3 * Y_pow_m1)
    , m2);

  return E_;
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