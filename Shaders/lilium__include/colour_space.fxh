#pragma once

#include "../ReShade.fxh"

#ifndef __RESHADE__
  #include "_no.fxh"
  #define BUFFER_WIDTH       3840
  #define BUFFER_HEIGHT      2160
  #define BUFFER_COLOR_SPACE    2
#endif


#define YES 1
#define NO  0

#define CSP_UNKNOWN 0
#define CSP_UNSET   CSP_UNKNOWN
#define CSP_SRGB    1
#define CSP_SCRGB   2
#define CSP_HDR10   3
#define CSP_HLG     4
#define CSP_PS5     5

#ifndef CSP_OVERRIDE
  #define CSP_OVERRIDE CSP_UNSET
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 8 || BUFFER_COLOR_BIT_DEPTH == 10 || BUFFER_COLOR_BIT_DEPTH == 16)
  #define IS_POSSIBLE_SRGB_BIT_DEPTH
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 16)
  #define IS_POSSIBLE_SCRGB_BIT_DEPTH
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 10)
  #define IS_POSSIBLE_HDR10_BIT_DEPTH
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 16)
  #define IS_POSSIBLE_PS5_BIT_DEPTH
#endif

#if ((BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))  \
  || (BUFFER_COLOR_SPACE != CSP_SCRGB && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))  \
  || (                                   CSP_OVERRIDE == CSP_SCRGB && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))  \
  || (BUFFER_COLOR_SPACE == CSP_SRGB  && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH)))

  #define ACTUAL_COLOUR_SPACE CSP_SCRGB
  #define FONT_BRIGHTNESS 2.5375f // 203.f / 80.f

#elif ((BUFFER_COLOR_SPACE == CSP_HDR10 && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_HDR10_BIT_DEPTH))  \
    || (                                   CSP_OVERRIDE == CSP_HDR10 && defined(IS_POSSIBLE_HDR10_BIT_DEPTH)))

  #define ACTUAL_COLOUR_SPACE CSP_HDR10
  #define FONT_BRIGHTNESS 0.58068888104160783796

#elif ((BUFFER_COLOR_SPACE == CSP_HLG && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_HDR10_BIT_DEPTH))  \
    || (                                 CSP_OVERRIDE == CSP_HLG   && defined(IS_POSSIBLE_HDR10_BIT_DEPTH)))

  #define ACTUAL_COLOUR_SPACE CSP_HLG
  #define FONT_BRIGHTNESS 0.69691214644230630735

#elif (CSP_OVERRIDE == CSP_PS5 && defined(IS_POSSIBLE_PS5_BIT_DEPTH))

  #define ACTUAL_COLOUR_SPACE CSP_PS5
  #define FONT_BRIGHTNESS 2.03f

#elif ((BUFFER_COLOR_SPACE == CSP_SRGB && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SRGB_BIT_DEPTH))  \
    || (                                  CSP_OVERRIDE == CSP_SRGB  && defined(IS_POSSIBLE_SRGB_BIT_DEPTH)))

  #define ACTUAL_COLOUR_SPACE CSP_SRGB
  #define FONT_BRIGHTNESS 1.f

#else
  #define ACTUAL_COLOUR_SPACE CSP_UNKNOWN
  #define FONT_BRIGHTNESS 1.f
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG   \
  || ACTUAL_COLOUR_SPACE == CSP_PS5   \
  || defined(IS_POSSIBLE_HDR10_BIT_DEPTH))

  #define IS_POSSIBLE_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define IS_FLOAT_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  #define IS_UNORM_HDR_CSP
#endif


#define CSP_SRGB_TEXT  "sRGB (sRGB transfer function / gamma 2.2 + BT.709 primaries)"
#define CSP_SCRGB_TEXT "scRGB (linear + BT.709 primaries)"
#define CSP_HDR10_TEXT "HDR10 (PQ + BT.2020 primaries)"
#define CSP_HLG_TEXT   "HLG (HLG + BT.2020 primaries)"

#if (BUFFER_COLOR_BIT_DEPTH == 8)
  #define BACK_BUFFER_FORMAT_TEXT "RGBA8_UNORM or BGRA8_UNORM"
#elif (BUFFER_COLOR_BIT_DEPTH == 10)
  // d3d11 and d3d12 only allow rgb10a2 to be used for HDR10
  #if (__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000)
    #define BACK_BUFFER_FORMAT_TEXT "RGB10A2_UNORM"
  #else
    #define BACK_BUFFER_FORMAT_TEXT "RGB10A2_UNORM or BGR10A2_UNORM"
  #endif
#elif (BUFFER_COLOR_BIT_DEPTH == 16)
  #define BACK_BUFFER_FORMAT_TEXT "RGBA16_FLOAT"
#else
  #define BACK_BUFFER_FORMAT_TEXT "unknown"
#endif


#define CSP_UNSET_TEXT "colour space unset! likely "

#if (BUFFER_COLOR_SPACE == CSP_SCRGB)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_SCRGB_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_HDR10)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_HDR10_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_HLG)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_HLG_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_SRGB \
    && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_SCRGB_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_UNKNOWN \
    && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_SCRGB_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_UNKNOWN \
    && defined(IS_POSSIBLE_HDR10_BIT_DEPTH))
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_HDR10_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_SRGB)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_SRGB_TEXT
#else
  #define BACK_BUFFER_COLOUR_SPACE_TEXT "unknown"
#endif


#if (CSP_OVERRIDE == CSP_SRGB)
  #define CSP_OVERRIDE_TEXT CSP_SRGB_TEXT
#elif (CSP_OVERRIDE == CSP_SCRGB)
  #define CSP_OVERRIDE_TEXT CSP_SCRGB_TEXT
#elif (CSP_OVERRIDE == CSP_HDR10)
  #define CSP_OVERRIDE_TEXT CSP_HDR10_TEXT
#elif (CSP_OVERRIDE == CSP_HLG)
  #define CSP_OVERRIDE_TEXT CSP_HLG_TEXT
#else
  #define CSP_OVERRIDE_TEXT "unset"
#endif


#if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
  #define ACTUAL_CSP_TEXT CSP_SRGB_TEXT
#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  #define ACTUAL_CSP_TEXT CSP_SCRGB_TEXT
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  #define ACTUAL_CSP_TEXT CSP_HDR10_TEXT
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
  #define ACTUAL_CSP_TEXT CSP_HLG_TEXT
#else
  #define ACTUAL_CSP_TEXT "unknown"
#endif

#ifndef HIDE_CSP_OVERRIDE_EXPLANATION
  #define HIDE_CSP_OVERRIDE_EXPLANATION NO
#endif


#if (HIDE_CSP_OVERRIDE_EXPLANATION == YES)
  #define INFO_TEXT \
         "detected back buffer format:       " BACK_BUFFER_FORMAT_TEXT           \
    "\n" "detected back buffer color space:  " BACK_BUFFER_COLOUR_SPACE_TEXT     \
    "\n" "colour space overwritten to:       " CSP_OVERRIDE_TEXT                 \
    "\n" "colour space in use by the shader: " ACTUAL_CSP_TEXT
#else
  #define INFO_TEXT \
         "detected back buffer format:       " BACK_BUFFER_FORMAT_TEXT           \
    "\n" "detected back buffer color space:  " BACK_BUFFER_COLOUR_SPACE_TEXT     \
    "\n" "colour space overwritten to:       " CSP_OVERRIDE_TEXT                 \
    "\n" "colour space in use by the shader: " ACTUAL_CSP_TEXT                   \
    "\n"                                                                         \
    "\n" "Use the \"Preprocessor definition\" 'CSP_OVERRIDE' below to override " \
         "the colour space in case the auto detection doesn't work."             \
    "\n" "Only overrides that make sense are allowed."                           \
    "\n" "Possible values are:"                                                  \
    "\n" "- 'CSP_HDR10'"                                                         \
    "\n" "- 'CSP_SCRGB'"                                                         \
    "\n" "Hit ENTER to apply."
#endif


uniform int GLOBAL_INFO
<
  ui_category = "info";
  ui_label    = " ";
  ui_type     = "radio";
  ui_text     = INFO_TEXT;
>;


#if (__RENDERER__ & 0x10000 \
  || __RENDERER__ < 0xB000)
  #define ERROR_TEXT "Only DirectX 11, 12 and Vulkan are supported!"
#else
  #define ERROR_TEXT "Only HDR colour spaces are supported!"
#endif

#define ERROR_STUFF             \
  uniform int ERROR_MESSAGE     \
    <                           \
      ui_category = "ERROR";    \
      ui_label    = " ";        \
      ui_type     = "radio";    \
      ui_text     = ERROR_TEXT; \
    >;                          \
                                \
    void CS_Error()             \
    {                           \
      return;                   \
    }


namespace Csp
{

  namespace KHelpers
  {

    //#define K_BT709  float3(0.2126f, 0.7152f, 0.0722f)
    //#define K_BT2020 float3(0.2627f, 0.6780f, 0.0593f)

    //#define KB_BT709_HELPER 1.8556f //2 - 2 * 0.0722
    //#define KR_BT709_HELPER 1.5748f //2 - 2 * 0.2126
    //#define KG_BT709_HELPER float2(0.187324272930648, 0.468124272930648)
    //(0.0722/0.7152)*(2-2*0.0722), (0.2126/0.7152)*(2-2*0.2126)

    //#define KB_BT2020_HELPER 1.8814f //2 - 2 * 0.0593
    //#define KR_BT2020_HELPER 1.4746f //2 - 2 * 0.2627
    //#define KG_BT2020_HELPER float2(0.164553126843658, 0.571353126843658)
    //(0.0593/0.6780)*(2-2*0.0593), (0.2627/0.6780)*(2-2*0.2627)

    namespace Bt709
    {
      static const float3 K  = float3(0.212636821677324, 0.715182981841251,  0.0721801964814255);

      static const float  Kb = 1.85563960703715;
      static const float  Kr = 1.57472635664535;
      static const float2 Kg = float2(0.187281345942859, 0.468194596334655);
    } //Bt709

    namespace Bt2020
    {
      static const float3 K  = float3(0.262698338956556, 0.678008765772817,  0.0592928952706273);

      static const float  Kb = 1.88141420945875;
      static const float  Kr = 1.47460332208689;
      static const float2 Kg = float2(0.164532527178987, 0.571343414550845);
    } //Bt2020

    namespace Ap0D65
    {
      static const float3 K  = float3(0.343163015452697, 0.734695029446046, -0.0778580448987425);

      static const float  Kb = 2.15571608979748;
      static const float  Kr = 1.31367396909461;
      static const float2 Kg = float2(-0.228448313084334, 0.613593807618545);
    } //AP0_D65

  } //KHelpers


  namespace Trc
  {

    //linear->gamma compressed = inverse EOTF -> ^(1 / 2.2)
    //
    //gamma compressed->display (also linear) = EOTF -> ^(2.2)

    // IEC 61966-2-1
    float FromSrgb(const float C)
    {
      if (C <= 0.04045f)
        return C / 12.92f;
      else
        return pow(((C + 0.055f) / 1.055f), 2.4f);
    }

    float3 FromSrgb(const float3 Colour)
    {
      return float3(
        FromSrgb(Colour.r),
        FromSrgb(Colour.g),
        FromSrgb(Colour.b));
    }

    float ToSrgb(const float C)
    {
      if (C <= 0.0031308f)
        return C * 12.92f;
      else
        return 1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
    }

    float3 ToSrgb(const float3 Colour)
    {
      return float3(
        ToSrgb(Colour.r),
        ToSrgb(Colour.g),
        ToSrgb(Colour.b));
    }

    //#define X_sRGB_1 1.19417654368084505707
    //#define X_sRGB_x 0.039815307380813555
    //#define X_sRGB_y_adjust 1.21290538811
    // extended sRGB gamma including above 1 and below -1
    float FromExtendedSrgb(const float C)
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

    float3 FromExtendedSrgb(const float3 Colour)
    {
      return float3(
        FromExtendedSrgb(Colour.r),
        FromExtendedSrgb(Colour.g),
        FromExtendedSrgb(Colour.b));
    }

    
    float ToExtendedSrgb(const float C)
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

    float3 ToExtendedSrgb(const float3 Colour)
    {
      return float3(
        ToExtendedSrgb(Colour.r),
        ToExtendedSrgb(Colour.g),
        ToExtendedSrgb(Colour.b));
    }

    static const float Gamma22        = 2.2f;
    static const float InverseGamma22 = 1.f / 2.2f;
    //#define X_22_1 1.20237927370128566986
    //#define X_22_x 0.0370133892172524
    //#define X_22_y_adjust 1.5f - pow(X_22_x, InverseGamma22)
    // extended gamma 2.2 including above 1 and below 0
    float FromExtendedGamma22(const float C)
    {
      if (C < -1.f)
        return
          -pow(-C, InverseGamma22);
      else if (C < 0.f)
        return
          -pow(-C, Gamma22);
      else if (C <= 1.f)
        return
          pow(C, Gamma22);
      else
        return
          pow(C, InverseGamma22);
    }
    //{
    //  if (C < -X_22_1)
    //    return
    //      -(pow(-C - X_22_1 + X_22_x, InverseGamma22) + X_22_y_adjust);
    //  else if (C < 0)
    //    return
    //      -pow(-C, Gamma22);
    //  else if (C <= X_22_1)
    //    return
    //      pow(C, Gamma22);
    //  else
    //    return
    //      (pow(C - X_22_1 + X_22_x, InverseGamma22) + X_22_y_adjust);
    //}

    float3 FromExtendedGamma22(const float3 Colour)
    {
      return float3(
        FromExtendedGamma22(Colour.r),
        FromExtendedGamma22(Colour.g),
        FromExtendedGamma22(Colour.b));
    }

    static const float Gamma24        = 2.4f;
    static const float InverseGamma24 = 1.f / 2.4f;
    //#define X_24_1 1.1840535873752085849
    //#define X_24_x 0.033138075
    //#define X_24_y_adjust 1.5f - pow(X_24_x, InverseGamma24)
    // extended gamma 2.4 including above 1 and below 0
    float FromExtendedGamma24(const float C)
    {
      if (C < -1.f)
        return
          -pow(-C, InverseGamma24);
      else if (C < 0.f)
        return
          -pow(-C, Gamma24);
      else if (C <= 1.f)
        return
          pow(C, Gamma24);
      else
        return
          pow(C, InverseGamma24);
    }
    //{
    //  if (C < -X_24_1)
    //    return
    //      -(pow(-C - X_24_1 + X_24_x, InverseGamma24) + X_24_y_adjust);
    //  else if (C < 0)
    //    return
    //      -pow(-C, Gamma24);
    //  else if (C <= X_24_1)
    //    return
    //      pow(C, Gamma24);
    //  else
    //    return
    //      (pow(C - X_24_1 + X_24_x, InverseGamma24) + X_24_y_adjust);
    //}

    float3 FromExtendedGamma24(const float3 Colour)
    {
      return float3(
        FromExtendedGamma24(Colour.r),
        FromExtendedGamma24(Colour.g),
        FromExtendedGamma24(Colour.b));
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
    float ExtendedGammaAdjust(const float C, const float Adjust)
    {
      /*static*/ const float inverseAdjust = 1.f / Adjust;
    
      if (C < -1.f)
        return
          -pow(-C, inverseAdjust);
      else if (C < 0.f)
        return
          -pow(-C, Adjust);
      else if (C <= 1.f)
        return
          pow(C, Adjust);
      else
        return
          pow(C, inverseAdjust);
    }

    float3 ExtendedGammaAdjust(const float3 Colour, const float Adjust)
    {
      return float3(
        ExtendedGammaAdjust(Colour.r, Adjust),
        ExtendedGammaAdjust(Colour.g, Adjust),
        ExtendedGammaAdjust(Colour.b, Adjust));
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
    // (EOTF) takes PQ values as input
    // outputs as 1 = 10000 nits
    float FromPq(float E_)
    {
      E_ = clamp(E_, 0.f, 65504.f);

      const float E_pow_1_div_m2 = pow(E_, _1_div_PQ_m2);

      //Y
      return pow(
                 (max(E_pow_1_div_m2 - PQ_c1, 0.f)) /
                 (PQ_c2 - PQ_c3 * E_pow_1_div_m2)
             , _1_div_PQ_m1);
    }

    // (EOTF) takes PQ values as input
    // outputs as 1 = 10000 nits
    float2 FromPq(float2 E_)
    {
      return float2(FromPq(E_.x),
                    FromPq(E_.y));
    }

    // (EOTF) takes PQ values as input
    // outputs as 1 = 10000 nits
    float3 FromPq(float3 E_)
    {
      return float3(FromPq(E_.r),
                    FromPq(E_.g),
                    FromPq(E_.b));
    }

    // (EOTF) takes PQ values as input
    // outputs nits
    float FromPqToNits(float E_)
    {
      return FromPq(E_) * 10000.f;
    }

    // (EOTF) takes PQ values as input
    // outputs nits
    float2 FromPqToNits(float2 E_)
    {
      return FromPq(E_) * 10000.f;
    }

    // (EOTF) takes PQ values as input
    // outputs nits
    float3 FromPqToNits(float3 E_)
    {
      return FromPq(E_) * 10000.f;
    }

    // Rec. ITU-R BT.2100-2 Table 4 (end)
    // (inverse EOTF) takes normalised to 10000 nits values as input
    float ToPq(float Y)
    {
      Y = clamp(Y, 0.f, 65504.f);

      const float Y_pow_m1 = pow(Y, PQ_m1);

      //E'
      return pow(
                 (PQ_c1 + PQ_c2 * Y_pow_m1) /
                 (  1.f + PQ_c3 * Y_pow_m1)
             , PQ_m2);
    }

    // (inverse EOTF) takes normalised to 10000 nits values as input
    float2 ToPq(float2 Y)
    {
      return float2(ToPq(Y.x),
                    ToPq(Y.y));
    }

    // (inverse EOTF) takes normalised to 10000 nits values as input
    float3 ToPq(float3 Y)
    {
      return float3(ToPq(Y.r),
                    ToPq(Y.g),
                    ToPq(Y.b));
    }

    // (OETF) takes nits as input
    float ToPqFromNits(const float Fd)
    {
      const float Y = clamp(Fd / 10000.f, 0.f, 65504.f);

      const float Y_pow_m1 = pow(Y, PQ_m1);

      //E'
      return pow(
                 (PQ_c1 + PQ_c2 * Y_pow_m1) /
                 (  1.f + PQ_c3 * Y_pow_m1)
             , PQ_m2);
    }

    // (OETF) takes nits as input
    float3 ToPqFromNits(float3 Fd)
    {
      return float3(ToPqFromNits(Fd.r),
                    ToPqFromNits(Fd.g),
                    ToPqFromNits(Fd.b));
    }


    // Rec. ITU-R BT.2100-2 Table 5
    static const float HLG_a = 0.17883277;
    static const float HLG_b = 0.28466892; // 1 - 4 * HLG_a
    static const float HLG_c = 0.55991072952956202016; // 0.5 - HLG_a * ln(4 * HLG_a)

    // Rec. ITU-R BT.2100-2 Table 5 (end)
    // (EOTF) takes HLG values as input
    float FromHlg(float X)
    {
      if (X <= 0.5f)
      {
        return pow(X, 2.f) / 3.f;
      }
      else
      {
        return (exp((X - HLG_c) / HLG_a) + HLG_b) / 12.f;
      }
    }

    // (EOTF) takes HLG values as input
    float3 FromHlg(float3 Rgb)
    {
      return float3(FromHlg(Rgb.r),
                    FromHlg(Rgb.g),
                    FromHlg(Rgb.b));
    }

    // Rec. ITU-R BT.2100-2 Table 5
    // (inverse EOTF) takes normalised to 1000 nits values as input
    float ToHlg(float E)
    {
      if (E <= (1.f / 12.f))
      {
        return sqrt(3.f * E);
      }
      else
      {
        return HLG_a * log(12.f * E - HLG_b) + HLG_c;
      }
    }

    // (inverse EOTF) takes normalised to 1000 nits values as input
    float3 ToHlg(float3 E)
    {
      return float3(ToHlg(E.r),
                    ToHlg(E.g),
                    ToHlg(E.b));
    }

    // Rec. ITU-R BT.2100-2 Table 5
    // (OETF) takes nits as input
    float ToHlgFromNits(float E)
    {
      E = E / 1000.f;

      if (E <= (1.f / 12.f))
      {
        return sqrt(3.f * E);
      }
      else
      {
        return HLG_a * log(12.f * E - HLG_b) + HLG_c;
      }
    }

    // (OETF) takes nits as input
    float3 ToHlgFromNits(float3 E)
    {
      return float3(ToHlgFromNits(E.r),
                    ToHlgFromNits(E.g),
                    ToHlgFromNits(E.b));
    }

  } //Trc


  namespace Ycbcr
  {

    namespace ToRgb
    {

      float3 Bt709(float3 Colour)
      {
        return float3(
          Colour.x + KHelpers::Bt709::Kr    * Colour.z,
          Colour.x - KHelpers::Bt709::Kg[0] * Colour.y - KHelpers::Bt709::Kg[1] * Colour.z,
          Colour.x + KHelpers::Bt709::Kb    * Colour.y);
      }

      float3 Bt2020(const float3 Colour)
      {
        return float3(
          Colour.x + KHelpers::Bt2020::Kr    * Colour.z,
          Colour.x - KHelpers::Bt2020::Kg[0] * Colour.y - KHelpers::Bt2020::Kg[1] * Colour.z,
          Colour.x + KHelpers::Bt2020::Kb    * Colour.y);
      }

      float3 Ap0D65(const float3 Colour)
      {
        return float3(
          Colour.x + KHelpers::Ap0D65::Kr    * Colour.z,
          Colour.x - KHelpers::Ap0D65::Kg[0] * Colour.y - KHelpers::Ap0D65::Kg[1] * Colour.z,
          Colour.x + KHelpers::Ap0D65::Kb    * Colour.y);
      }

    } //ToRgb

    namespace FromRgb
    {

      float3 Bt709(const float3 Colour)
      {
        const float Y = dot(Colour, KHelpers::Bt709::K);
        return float3(Y,
                      (Colour.b - Y) / KHelpers::Bt709::Kb,
                      (Colour.r - Y) / KHelpers::Bt709::Kr);
      }

      float3 Bt2020(const float3 Colour)
      {
        const float Y = dot(Colour, KHelpers::Bt2020::K);
        return float3(Y,
                      (Colour.b - Y) / KHelpers::Bt2020::Kb,
                      (Colour.r - Y) / KHelpers::Bt2020::Kr);
      }

      float3 Ap0D65(const float3 Colour)
      {
        const float Y = dot(Colour, KHelpers::Ap0D65::K);
        return float3(Y,
                      (Colour.b - Y) / KHelpers::Ap0D65::Kb,
                      (Colour.r - Y) / KHelpers::Ap0D65::Kr);
      }

    } //FromRgb

  } //Ycbcr


  namespace Mat
  {

    // AP0_D65 uses a highly accurate D65 white point

    //BT709 To
    static const float3x3 Bt709ToXYZ = float3x3(
      0.412135323426798,  0.357675002654190, 0.180356796374193,
      0.212507276141942,  0.715350005308380, 0.0721427185496773,
      0.0193188432856311, 0.119225000884730, 0.949879127570751);

    static const float3x3 Bt709ToDciP3 = float3x3(
      0.822334429220561,  0.177665570779439,  0.000000000000000,
      0.0331661871416848, 0.966833812858315,  0.000000000000000,
      0.0170826010352503, 0.0724605600100221, 0.910456838954727);

    static const float3x3 Bt709ToBt2020 = float3x3(
      0.627225305694944,  0.329476882715808,  0.0432978115892484,
      0.0690418812810714, 0.919605681354755,  0.0113524373641739,
      0.0163911702607078, 0.0880887513437058, 0.895520078395586);

    static const float3x3 Bt709ToAp0D65 = float3x3(
      0.433799790599445,  0.376466672803638, 0.189745486872426,
      0.0885578169506515, 0.809428607238992, 0.102029598497234,
      0.0177530102658331, 0.109561562944343, 0.872889419606533);

    
    //BT2020 To
    static const float3x3 Bt2020ToXYZ = float3x3(
      0.636744702289598,    0.144643300793529,  0.168779119372055,
      0.262612221848252,    0.678121827837897,  0.0592659503138512,
      4.99243382266951e-17, 0.0280778172128614, 1.06034515452825);

    static const float3x3 Bt2020ToBt709 = float3x3(
       1.66096379471340,   -0.588112737547978, -0.0728510571654192,
      -0.124477196529907,   1.13281946828499,  -0.00834227175508652,
      -0.0181571579858552, -0.100666415661988,  1.11882357364784);

    static const float3x3 Bt2020ToAp0D65 = float3x3(
      0.670218991540148,    0.152244595663773,  0.177548363071589,
      0.0444833563345243,   0.854583550336869,  0.100949116015484,
      4.58778652504860e-17, 0.0258020508708645, 0.974401941945845);


    //AP0_D65 To
    //AP0 To
    static const float3x3 Ap0D65ToXYZ = float3x3(
      0.950054699026617, 0.000000000000000,  0.000101049399810263,
      0.343064531988242, 0.734743505865660, -0.0778080378539023,
      0.000000000000000, 0.000000000000000,  1.08820098655672);

    static const float3x3 Ap0D65ToBt709 = float3x3(
       2.55325882463675,   -1.13028251504951,  -0.422902442510382,
      -0.277186279374805,   1.37800555426763,  -0.100817475751080,
      -0.0171374148821935, -0.149973779310547,  1.16687576769787);

    static const float3x3 Ap0D65ToBt2020 = float3x3(
       1.50940006298040,    -0.261414358078325,  -0.247948974278613,
      -0.0788148360852721,   1.18748233748891,   -0.108663419645253,
       0.00208700775574192, -0.0314443951322142,  1.02914792747532);


    //AP1_D65 To
    static const float3x3 Ap1D65ToXYZ = float3x3(
       0.647315060545070,   0.134403912261337,   0.168436775620020,
       0.266007451247834,   0.676092407132784,   0.0579001416193820,
      -0.00544725156138904, 0.00407284582610111, 1.08957539229201);


    //XYZ To
    static const float3x3 XYZToBt709 = float3x3(
       3.24297896532120,   -1.53833617585749,  -0.498919840818647,
      -0.968997952917093,   1.87549198225861,   0.0415445240532242,
       0.0556683243682128, -0.204117189350113,  1.05769816299604);

    static const float3x3 XYZToDciP3 = float3x3(
       2.49465568203257,   -0.931816447602876,  -0.402897930947739,
      -0.829302738210345,   1.76226831869698,    0.0236193817844718,
       0.0358679881475428, -0.0762194748135283,  0.957476016938569);

    static const float3x3 XYZToBt2020 = float3x3(
       1.71722636462073,   -0.355789953897356,  -0.253451173616083,
      -0.666562682837409,   1.61618623098933,    0.0157656680755665,
       0.0176505028477730, -0.0427964247130936,  0.942671667036796);

    static const float3x3 XYZToAp1 = float3x3(
       1.64102337969433,   -0.324803294184790,   -0.236424695237612,
      -0.663662858722983,   1.61533159165734,     0.0167563476855301,
       0.0117218943283754, -0.00828444199623741,  0.988394858539022);

    static const float3x3 XYZToAp0 = float3x3(
       1.04981101749797,  0.000000000000000, -0.0000974845405792529,
      -0.495903023077320, 1.37331304581571,   0.0982400360573100,
       0.000000000000000, 0.000000000000000,  0.991252018200499);


    namespace Bt709To
    {
      float3 XYZ(const float3 Colour)
      {
        return mul(Bt709ToXYZ, Colour);
      }

      float3 DciP3(const float3 Colour)
      {
        return mul(Bt709ToDciP3, Colour);
      }

      float3 Bt2020(const float3 Colour)
      {
        return mul(Bt709ToBt2020, Colour);
      }

      float3 Ap0D65(const float3 Colour)
      {
        return mul(Bt709ToAp0D65, Colour);
      }
    } //Bt709To

    namespace Bt2020To
    {
      float3 XYZ(const float3 Colour)
      {
        return mul(Bt2020ToXYZ, Colour);
      }

      float3 Bt709(const float3 Colour)
      {
        return mul(Bt2020ToBt709, Colour);
      }

      float3 Ap0D65(const float3 Colour)
      {
        return mul(Bt2020ToAp0D65, Colour);
      }
    } //Bt2020To

    namespace Ap0D65To
    {
      float3 XYZ(const float3 Colour)
      {
        return mul(Ap0D65ToXYZ, Colour);
      }

      float3 Bt709(const float3 Colour)
      {
        return mul(Ap0D65ToBt709, Colour);
      }

      float3 Bt2020(const float3 Colour)
      {
        return mul(Ap0D65ToBt2020, Colour);
      }
    } //Ap0D65To

    namespace AP1_D65To
    {
      float3 XYZ(const float3 Colour)
      {
        return mul(Ap1D65ToXYZ, Colour);
      }
    } //AP1_D65To

    namespace XYZTo
    {
      float3 Bt709(const float3 Colour)
      {
        return mul(XYZToBt709, Colour);
      }

      float3 DciP3(const float3 Colour)
      {
        return mul(XYZToDciP3, Colour);
      }

      float3 Bt2020(const float3 Colour)
      {
        return mul(XYZToBt2020, Colour);
      }

      float3 AP1(const float3 Colour)
      {
        return mul(XYZToAp1, Colour);
      }

      float3 AP0(const float3 Colour)
      {
        return mul(XYZToAp0, Colour);
      }
    } //XYZTo

  } //Mat


  namespace Ictcp
  {

    namespace Mat
    {

      //L'M'S'->ICtCp
      static const float3x3 PqLmsToIctcp = float3x3(
        0.5,             0.5,             0.0,
        1.61376953125,  -3.323486328125,  1.709716796875,
        4.378173828125, -4.24560546875,  -0.132568359375);

      //ICtCp->L'M'S'
      static const float3x3 IctcpToPqLms = float3x3(
        1.0,  0.00860903703793276,  0.111029625003026,
        1.0, -0.00860903703793276, -0.111029625003026,
        1.0,  0.560031335710679,   -0.320627174987319);


      //RGB BT.2020->LMS
      static const float3x3 Bt2020ToLms = float3x3(
        0.412109375,    0.52392578125,  0.06396484375,
        0.166748046875, 0.720458984375, 0.11279296875,
        0.024169921875, 0.075439453125, 0.900390625);

      //RGB AP0_D65->LMS
      static const float3x3 Ap0D65ToLms = float3x3(
        0.58056640625, 0.512451171875, -0.09326171875,
        0.19482421875, 0.80859375,     -0.00341796875,
        0.0322265625,  0.054931640625,  0.911865234375);


      //LMS->RGB BT.2020
      static const float3x3 LmsToBt2020 = float3x3(
         3.43660669433308,   -2.50645211865627,    0.0698454243231915,
        -0.791329555598929,   1.98360045179229,   -0.192270896193362,
        -0.0259498996905927, -0.0989137147117265,  1.12486361440232);

      //LMS->RGB AP0_D65
      static const float3x3 LmsToAp0D65 = float3x3(
         2.17868661291322,   -1.39553812861432,    0.217595917625769,
        -0.525128926909500,   1.57276675119756,   -0.0478127217035243,
        -0.0453634871084088, -0.0454247619926001,  1.09184342737796);

      namespace IctcpTo
      {
        //ICtCp->L'M'S'
        float3 PqLms(const float3 Colour)
        {
          return mul(IctcpToPqLms, Colour);
        }
      } //IctcpTo

      namespace PqLmsTo
      {
        //L'M'S'->ICtCp
        float3 Ictcp(const float3 Colour)
        {
          return mul(PqLmsToIctcp, Colour);
        }
      } //PqLmsTo

      namespace Bt2020To
      {
        //RGB BT.2020->LMS
        float3 Lms(const float3 Colour)
        {
          return mul(Bt2020ToLms, Colour);
        }
      } //Bt2020To

      namespace Ap0D65To
      {
        //RGB AP0_D65->LMS
        float3 Lms(const float3 Colour)
        {
          return mul(Ap0D65ToLms, Colour);
        }
      } //Ap0D65To

      namespace LmsTo
      {
        //LMS->RGB BT.2020
        float3 Bt2020(const float3 Colour)
        {
          return mul(LmsToBt2020, Colour);
        }

        //LMS->RGB AP0_D65
        float3 Ap0D65(const float3 Colour)
        {
          return mul(LmsToAp0D65, Colour);
        }
      } //LmsTo

    } //Mat

  } //ICtCp


  namespace Map
  {

    namespace Bt709Into
    {

      float3 Scrgb(const float3 Input)
      {
        return Input / 80.f;
      }

      float3 Hdr10(const float3 Input)
      {
        return Csp::Trc::ToPqFromNits(Csp::Mat::Bt709To::Bt2020(Input));
      }

      float3 Hlg(const float3 Input)
      {
        return Csp::Trc::ToHlgFromNits(Csp::Mat::Bt709To::Bt2020(Input));
      }

      float3 Ps5(const float3 Input)
      {
        return Csp::Mat::Bt709To::Bt2020(Input / 100.f);
      }

    } //Bt709Into

  }

}


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


// START Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli
//static const float3x3 expanded_BT709_To_BT2020_matrix = float3x3(
//   0.6274040,  0.3292820, 0.0433136,
//   0.0457456,  0.941777,  0.0124772,
//  -0.00121055, 0.0176041, 0.983607);
// END Converted from (Copyright (c) Microsoft Corporation - Licensed under the MIT License.)  https://github.com/microsoft/Xbox-GDK-Samples/blob/main/Kits/ATGTK/HDR/HDRCommon.hlsli

//float3 ExpandColourGamutP3(float3 colour, float start, float stop)
//{
//  // The original Rec.709 colour, but rotated into the P3-D65 colour space
//  float3 Rec709 = mul(Bt709ToDciP3, colour);
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
//  float3 Rec709 = mul(Bt709ToBt2020, colour);
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


bool IsNAN(const float Input)
{
  if (isnan(Input) || isinf(Input))
    return true;
  else
    return false;
}

float fixNAN(const float Input)
{
  if (IsNAN(Input))
    return 0.f;
  else
    return Input;
}

float3 fixNAN(float3 Input)
{
  if (IsNAN(Input.r))
    Input.r = 0.f;
  else if (IsNAN(Input.g))
    Input.g = 0.f;
  else if (IsNAN(Input.b))
    Input.b = 0.f;

  return Input;
}
