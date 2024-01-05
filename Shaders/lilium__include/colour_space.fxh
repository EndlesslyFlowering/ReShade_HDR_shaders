#pragma once

#include "../ReShade.fxh"

#ifndef __RESHADE__
  #include "_no.fxh"
  #define BUFFER_WIDTH       3840
  #define BUFFER_HEIGHT      2160
  #define BUFFER_COLOR_SPACE    2
#endif

#ifdef GAMESCOPE
  #ifndef GAMESCOPE_SDR_ON_HDR_NITS
    #define GAMESCOPE_SDR_ON_HDR_NITS 203.f
  #endif
#endif


#define STRINGIFY(x) #x
#define GET_UNKNOWN_NUMBER(x) "unknown (" STRINGIFY(x) ")"

#define VS_PostProcess PostProcessVS

#define YES 1
#define NO  0

#define CSP_UNKNOWN 0
#define CSP_UNSET   CSP_UNKNOWN
#define CSP_SRGB    1
#define CSP_SCRGB   2
#define CSP_HDR10   3
#define CSP_HLG     4
#define CSP_PS5     5
#define CSP_FAIL    255

#ifndef CSP_OVERRIDE
  #define CSP_OVERRIDE CSP_UNSET
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 8 || BUFFER_COLOR_BIT_DEPTH == 10 || BUFFER_COLOR_BIT_DEPTH == 16)
  #define IS_POSSIBLE_SRGB_BIT_DEPTH
#endif

#if (BUFFER_COLOR_BIT_DEPTH == 16 \
  || BUFFER_COLOR_BIT_DEPTH == 11)
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
  || (                                   CSP_OVERRIDE == CSP_SCRGB && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH)))

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

#elif (CSP_OVERRIDE != CSP_UNSET \
    && CSP_OVERRIDE != CSP_UNKNOWN)

  #define ACTUAL_COLOUR_SPACE CSP_FAIL
  #define FONT_BRIGHTNESS 1.f

#else
  #define ACTUAL_COLOUR_SPACE CSP_UNKNOWN
  #define FONT_BRIGHTNESS 1.f
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG   \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define IS_POSSIBLE_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define IS_FLOAT_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  #define IS_HDR10_LIKE_CSP
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
#elif (BUFFER_COLOR_BIT_DEPTH == 11)
  #define BACK_BUFFER_FORMAT_TEXT "R11G11B10_UFLOAT"
#elif (BUFFER_COLOR_BIT_DEPTH == 16)
  #define BACK_BUFFER_FORMAT_TEXT "RGBA16_SFLOAT"
#else
  #define BACK_BUFFER_FORMAT_TEXT GET_UNKNOWN_NUMBER(BUFFER_COLOR_BIT_DEPTH)
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
  #define BACK_BUFFER_COLOUR_SPACE_TEXT GET_UNKNOWN_NUMBER(BUFFER_COLOR_SPACE)
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
#elif (ACTUAL_COLOUR_SPACE == CSP_FAIL)
  #define ACTUAL_CSP_TEXT "failed override"
#else
  #define ACTUAL_CSP_TEXT GET_UNKNOWN_NUMBER(ACTUAL_COLOUR_SPACE)
#endif

#ifndef HIDE_CSP_OVERRIDE_EXPLANATION
  #define HIDE_CSP_OVERRIDE_EXPLANATION NO
#endif


#define INFO_TEXT_BACK_BUFFER \
       "detected back buffer format:       " BACK_BUFFER_FORMAT_TEXT           \
  "\n" "detected back buffer color space:  " BACK_BUFFER_COLOUR_SPACE_TEXT     \
  "\n" "colour space overwritten to:       " CSP_OVERRIDE_TEXT                 \
  "\n" "colour space in use by the shader: " ACTUAL_CSP_TEXT

#define INFO_TEXT_CSP_OVERRIDE \
  "\n"                                                                         \
  "\n" "Use the \"Preprocessor definition\" 'CSP_OVERRIDE' below to override " \
       "the colour space in case the auto detection doesn't work. "            \
       "Hit ENTER to apply."                                                   \
  "\n"                                                                         \
  "\n" "Currently allowed override:"                                           \
  "\n"

#if defined(IS_POSSIBLE_SCRGB_BIT_DEPTH)
  #define INFO_TEXT_ALLOWED_CSP_OVERRIDE "'CSP_SCRGB'"
#elif defined(IS_POSSIBLE_HDR10_BIT_DEPTH)
  #define INFO_TEXT_ALLOWED_CSP_OVERRIDE "'CSP_HDR10'"
#else
  #define INFO_TEXT_ALLOWED_CSP_OVERRIDE "none!"
#endif

#if (HIDE_CSP_OVERRIDE_EXPLANATION == YES \
  || defined(IS_FLOAT_HDR_CSP)            \
  || defined(IS_HDR10_LIKE_CSP))
  #define INFO_TEXT INFO_TEXT_BACK_BUFFER
#else
  #define INFO_TEXT INFO_TEXT_BACK_BUFFER          \
                    INFO_TEXT_CSP_OVERRIDE         \
                    INFO_TEXT_ALLOWED_CSP_OVERRIDE
#endif


uniform int GLOBAL_INFO
<
  ui_category = "Info";
  ui_label    = " ";
  ui_type     = "radio";
  ui_text     = INFO_TEXT;
>;


#if (!(__RENDERER__ & 0xB000)   \
  && !(__RENDERER__ & 0xC000)   \
  && !(__RENDERER__ & 0x10000)  \
  && !(__RENDERER__ & 0x20000))
  #define ERROR_TEXT "Only DirectX 11, 12, OpenGL and Vulkan are supported!"
#else
  #define IS_HDR_COMPATIBLE_API
  #define ERROR_TEXT "Only HDR colour spaces are supported!"
#endif

#define ERROR_STUFF           \
  uniform int ERROR_MESSAGE   \
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

#define CS_ERROR                      \
  {                                   \
    pass CS_Error                     \
    {                                 \
      ComputeShader = CS_Error<1, 1>; \
      DispatchSizeX = 1;              \
      DispatchSizeY = 1;              \
    }                                 \
  }


#define PI asfloat(0x40490FDB)

#define FP32_MAX asfloat(0x7F7FFFFF)

#define MIN3(A, B, C) min(A, min(B, C))

#define MAX3(A, B, C) max(A, max(B, C))

#define MAXRGB(Rgb) max(Rgb.r, max(Rgb.g, Rgb.b))


namespace Csp
{

  namespace Trc
  {

    //linear->gamma compressed = inverse EOTF -> ^(1 / 2.2)
    //
    //gamma compressed->display (also linear) = EOTF -> ^(2.2)

    namespace SrgbTo
    {
      // IEC 61966-2-1
      float Linear(float C)
      {
        if (C <= 0.04045f)
        {
          return C / 12.92f;
        }
        else
        {
          return pow(((C + 0.055f) / 1.055f), 2.4f);
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::SrgbTo::Linear(Colour.r),
                      Csp::Trc::SrgbTo::Linear(Colour.g),
                      Csp::Trc::SrgbTo::Linear(Colour.b));
      }
    } //SrgbTo


    namespace LinearTo
    {
      float Srgb(float C)
      {
        if (C <= 0.0031308f)
        {
          return C * 12.92f;
        }
        else
        {
          return 1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
        }
      }

      float3 Srgb(float3 Colour)
      {
        return float3(Csp::Trc::LinearTo::Srgb(Colour.r),
                      Csp::Trc::LinearTo::Srgb(Colour.g),
                      Csp::Trc::LinearTo::Srgb(Colour.b));
      }
    } //LinearTo


    namespace ExtendedSrgbTo
    {
      //#define X_sRGB_1 1.19417654368084505707
      //#define X_sRGB_x 0.039815307380813555
      //#define X_sRGB_y_adjust 1.21290538811
      // extended sRGB gamma including above 1 and below -1
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
        }
        else if (absC > 0.04045f)
        {
          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
        }
        else
        {
          return C / 12.92f;
        }
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

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedSrgbTo::Linear(Colour.r),
                      Csp::Trc::ExtendedSrgbTo::Linear(Colour.g),
                      Csp::Trc::ExtendedSrgbTo::Linear(Colour.b));
      }
    } //ExtendedSrgbTo


    namespace LinearTo
    {
      float ExtendedSrgb(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
        }
        else if (absC > 0.0031308f)
        {
          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
        }
        else
        {
          return C * 12.92f;
        }
      }

      float3 ExtendedSrgb(float3 Colour)
      {
        return float3(Csp::Trc::LinearTo::ExtendedSrgb(Colour.r),
                      Csp::Trc::LinearTo::ExtendedSrgb(Colour.g),
                      Csp::Trc::LinearTo::ExtendedSrgb(Colour.b));
      }
    }


    namespace SrgbAccurateTo
    {
      // accurate sRGB with no slope discontinuity
      #define SrgbX       asfloat(0x3D20EA0B) //  0.0392857
      #define SrgbPhi     asfloat(0x414EC578) // 12.92321
      #define SrgbXDivPhi asfloat(0x3B4739A5) //  0.003039935

      float Linear(float C)
      {
        if (C <= SrgbX)
        {
          return C / SrgbPhi;
        }
        else
        {
          return pow(((C + 0.055f) / 1.055f), 2.4f);
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::SrgbAccurateTo::Linear(Colour.r),
                      Csp::Trc::SrgbAccurateTo::Linear(Colour.g),
                      Csp::Trc::SrgbAccurateTo::Linear(Colour.b));
      }
    } //SrgbAccurateTo


    namespace LinearTo
    {
      float SrgbAccurate(float C)
      {
        if (C <= SrgbXDivPhi)
        {
          return C * SrgbPhi;
        }
        else
        {
          return 1.055f * pow(C, (1.f / 2.4f)) - 0.055f;
        }
      }

      float3 SrgbAccurate(float3 Colour)
      {
        return float3(Csp::Trc::LinearTo::SrgbAccurate(Colour.r),
                      Csp::Trc::LinearTo::SrgbAccurate(Colour.g),
                      Csp::Trc::LinearTo::SrgbAccurate(Colour.b));
      }
    } //LinearTo


    namespace ExtendedSrgbAccurateTo
    {
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
        }
        else if (absC > SrgbX)
        {
          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
        }
        else
        {
          return C / SrgbPhi;
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedSrgbAccurateTo::Linear(Colour.r),
                      Csp::Trc::ExtendedSrgbAccurateTo::Linear(Colour.g),
                      Csp::Trc::ExtendedSrgbAccurateTo::Linear(Colour.b));
      }
    } //ExtendedSrgbAccurateTo


    namespace LinearTo
    {
      float ExtendedSrgbAccurate(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
        }
        else if (absC > SrgbXDivPhi)
        {
          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
        }
        else
        {
          return C * SrgbPhi;
        }
      }

      float3 ExtendedSrgbAccurate(float3 Colour)
      {
        return float3(Csp::Trc::LinearTo::ExtendedSrgbAccurate(Colour.r),
                      Csp::Trc::LinearTo::ExtendedSrgbAccurate(Colour.g),
                      Csp::Trc::LinearTo::ExtendedSrgbAccurate(Colour.b));
      }
    } //LinearTo


    static const float RemoveGamma22 = 2.2f;
    static const float ApplyGamma22  = 1.f / RemoveGamma22;

    namespace ExtendedGamma22To
    {
      //#define X_22_1 1.20237927370128566986
      //#define X_22_x 0.0370133892172524
      //#define X_22_y_adjust 1.5f - pow(X_22_x, Csp::Trc::ApplyGamma22)
      // extended gamma 2.2 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * pow(absC, Csp::Trc::ApplyGamma22);
        }
        else
        {
          return signC * pow(absC, Csp::Trc::RemoveGamma22);
        }
      }
      //{
      //  if (C < -X_22_1)
      //    return
      //      -(pow(-C - X_22_1 + X_22_x, Csp::Trc::ApplyGamma22) + X_22_y_adjust);
      //  else if (C < 0)
      //    return
      //      -pow(-C, Csp::Trc::RemoveGamma22);
      //  else if (C <= X_22_1)
      //    return
      //      pow(C, Csp::Trc::RemoveGamma22);
      //  else
      //    return
      //      (pow(C - X_22_1 + X_22_x, Csp::Trc::ApplyGamma22) + X_22_y_adjust);
      //}

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma22To::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma22To::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma22To::Linear(Colour.b));
      }
    } //ExtendedGamma22To


    static const float RemoveGamma24 = 2.4f;
    static const float ApplyGamma24  = 1.f / RemoveGamma24;

    namespace ExtendedGamma24To
    {
      //#define X_24_1 1.1840535873752085849
      //#define X_24_x 0.033138075
      //#define X_24_y_adjust 1.5f - pow(X_24_x, Csp::Trc::ApplyGamma24)
      // extended gamma 2.4 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        if (absC > 1.f)
        {
          return signC * pow(absC, Csp::Trc::ApplyGamma24);
        }
        else
        {
          return signC * pow(absC, Csp::Trc::RemoveGamma24);
        }
      }
      //{
      //  if (C < -X_24_1)
      //    return
      //      -(pow(-C - X_24_1 + X_24_x, Csp::Trc::ApplyGamma24) + X_24_y_adjust);
      //  else if (C < 0)
      //    return
      //      -pow(-C, Csp::Trc::RemoveGamma24);
      //  else if (C <= X_24_1)
      //    return
      //      pow(C, Csp::Trc::RemoveGamma24);
      //  else
      //    return
      //      (pow(C - X_24_1 + X_24_x, Csp::Trc::ApplyGamma24) + X_24_y_adjust);
      //}

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma24To::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma24To::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma24To::Linear(Colour.b));
      }
    } //ExtendedGamma24To


    //float X_power_TRC(float C, float pow_gamma)
    //{
    //  float pow_Inverse_gamma = 1.f / pow_gamma;
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
    //float3 X_power_TRC(float3 Colour, float pow_gamma)
    //{
    //  return float3(
    //    X_power_TRC(Colour.r, pow_gamma),
    //    X_power_TRC(Colour.g, pow_gamma),
    //    X_power_TRC(Colour.b, pow_gamma));
    //}


    // gamma adjust including values above 1 and below 0
    float ExtendedGammaAdjust(float C, float Adjust)
    {
      float inverseAdjust = 1.f / Adjust;

      static const float absC = abs(C);
      static const float signC = sign(C);

      if (absC > 1.f)
      {
        return signC * pow(absC, inverseAdjust);
      }
      else
      {
        return signC * pow(absC, Adjust);
      }
    }

    float3 ExtendedGammaAdjust(float3 Colour, float Adjust)
    {
      return float3(ExtendedGammaAdjust(Colour.r, Adjust),
                    ExtendedGammaAdjust(Colour.g, Adjust),
                    ExtendedGammaAdjust(Colour.b, Adjust));
    }


    // Rec. ITU-R BT.2100-2 Table 4
    #define PQ_m1 asfloat(0x3E232000) //  0.1593017578125 = 1305.f / 8192.f;
    #define PQ_m2 asfloat(0x429DB000) // 78.84375         = 2523.f / 32.f;
    #define PQ_c1 asfloat(0x3F560000) //  0.8359375       =  107.f / 128.f;
    #define PQ_c2 asfloat(0x4196D000) // 18.8515625       = 2413.f / 128.f;
    #define PQ_c3 asfloat(0x41958000) // 18.6875          = 2392.f / 128.f;

    #define _1_div_PQ_m1 asfloat(0x40C8E06B)
    #define _1_div_PQ_m2 asfloat(0x3C4FCDAC)


    // Rec. ITU-R BT.2100-2 Table 4
    namespace PqTo
    {

      #define PQ_TO_LINEAR(T)                           \
        T Linear(T E_)                                  \
        {                                               \
          E_ = max(E_, 0.f);                            \
                                                        \
          T E_pow_1_div_m2 = pow(E_, _1_div_PQ_m2);     \
                                                        \
          /* Y */                                       \
          return pow(                                   \
                     (max(E_pow_1_div_m2 - PQ_c1, 0.f)) \
                   / (PQ_c2 - PQ_c3 * E_pow_1_div_m2)   \
                 , _1_div_PQ_m1);                       \
        }

      // (EOTF) takes PQ values as input
      // outputs as 1 = 10000 nits
      PQ_TO_LINEAR(float)

      // (EOTF) takes PQ values as input
      // outputs as 1 = 10000 nits
      PQ_TO_LINEAR(float2)

      // (EOTF) takes PQ values as input
      // outputs as 1 = 10000 nits
      PQ_TO_LINEAR(float3)

      // (EOTF) takes PQ values as input
      // outputs nits
      #define PQ_TO_NITS(T)                            \
        T Nits(T E_)                                   \
        {                                              \
          return Csp::Trc::PqTo::Linear(E_) * 10000.f; \
        }

      // (EOTF) takes PQ values as input
      // outputs nits
      PQ_TO_NITS(float)

      // (EOTF) takes PQ values as input
      // outputs nits
      PQ_TO_NITS(float2)

      // (EOTF) takes PQ values as input
      // outputs nits
      PQ_TO_NITS(float3)

    } //PqTo


    // Rec. ITU-R BT.2100-2 Table 4 (end)
    namespace LinearTo
    {
      #define LINEAR_TO_PQ(T)                     \
        T Pq(T Y)                                 \
        {                                         \
          Y = max(Y, 0.f);                        \
                                                  \
          T Y_pow_m1 = pow(Y, PQ_m1);             \
                                                  \
          /* E' */                                \
          return pow(                             \
                     (PQ_c1 + PQ_c2 * Y_pow_m1) / \
                     (  1.f + PQ_c3 * Y_pow_m1)   \
                 , PQ_m2);                        \
        }

      // (inverse EOTF) takes normalised to 10000 nits values as input
      LINEAR_TO_PQ(float)

      // (inverse EOTF) takes normalised to 10000 nits values as input
      LINEAR_TO_PQ(float2)

      // (inverse EOTF) takes normalised to 10000 nits values as input
      LINEAR_TO_PQ(float3)

    } //LinearTo


    // Rec. ITU-R BT.2100-2 Table 4 (end)
    namespace NitsTo
    {
      // (OETF) takes nits as input
      #define NITS_TO_PQ(T)                 \
        T Pq(T Fd)                          \
        {                                   \
          T Y = Fd / 10000.f;               \
                                            \
          return Csp::Trc::LinearTo::Pq(Y); \
        }

      // (OETF) takes nits as input
      NITS_TO_PQ(float)

      // (OETF) takes nits as input
      NITS_TO_PQ(float2)

      // (OETF) takes nits as input
      NITS_TO_PQ(float3)
    }


    // Rec. ITU-R BT.2100-2 Table 5
    namespace HlgTo
    {
      #define HLG_a asfloat(0x3E371FF0) //0.17883277
      #define HLG_b asfloat(0x3E91C020) //0.28466892 = 1 - 4 * HLG_a
      #define HLG_c asfloat(0x3F0F564F) //0.55991072952956202016 = 0.5 - HLG_a * ln(4 * HLG_a)

      // Rec. ITU-R BT.2100-2 Table 5 (end)
      // (EOTF) takes HLG values as input
      float Linear(float X)
      {
        if (X <= 0.5f)
        {
          return (X * X) / 3.f;
        }
        else
        {
          return (exp((X - HLG_c) / HLG_a) + HLG_b) / 12.f;
        }
      }

      // (EOTF) takes HLG values as input
      float3 Linear(float3 Rgb)
      {
        return float3(Csp::Trc::HlgTo::Linear(Rgb.r),
                      Csp::Trc::HlgTo::Linear(Rgb.g),
                      Csp::Trc::HlgTo::Linear(Rgb.b));
      }
    } //HlgTo


    namespace LinearTo
    {
      // Rec. ITU-R BT.2100-2 Table 5
      // (inverse EOTF) takes normalised to 1000 nits values as input
      float Hlg(float E)
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
      float3 Hlg(float3 E)
      {
        return float3(Csp::Trc::LinearTo::Hlg(E.r),
                      Csp::Trc::LinearTo::Hlg(E.g),
                      Csp::Trc::LinearTo::Hlg(E.b));
      }
    } //LinearTo


    namespace NitsTo
    {
      // Rec. ITU-R BT.2100-2 Table 5
      // (OETF) takes nits as input
      float Hlg(float E)
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
      float3 Hlg(float3 E)
      {
        return float3(Csp::Trc::NitsTo::Hlg(E.r),
                      Csp::Trc::NitsTo::Hlg(E.g),
                      Csp::Trc::NitsTo::Hlg(E.b));
      }
    }


  } //Trc


  namespace Ycbcr
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

      #define KBt709  float3(asfloat(0x3e599b82), asfloat(0x3f37212e), asfloat(0x3d93bf90))
      #define KbBt709 asfloat(0x3fed880e)
      #define KrBt709 asfloat(0x3fc99920)
      #define KgBt709 float2(asfloat(0xbe3fa3b7), asfloat(0xbeef8d95))

      #define KBt2020  float3(asfloat(0x3e86751c), asfloat(0x3f2d9964), asfloat(0x3d72c0da))
      #define KbBt2020 asfloat(0x3ff0d3f2)
      #define KrBt2020 asfloat(0x3fbcc572)
      #define KgBt2020 float2(asfloat(0xbe2861a9), asfloat(0xbf12356b))

      #define KAp0D65  float3(asfloat(0x3eafa6b1), asfloat(0x3f3c18ec), asfloat(0xbd9f6224))
      #define KbAp0D65 asfloat(0x4009f622)
      #define KrAp0D65 asfloat(0x3fa82ca7)
      #define KgAp0D65 float2(asfloat(0x3e69cd4c), asfloat(0xbf1d0be7))


    namespace YcbcrTo
    {

      float3 RgbBt709(float3 Colour)
      {
        return float3(
          Colour.x + KrBt709    * Colour.z,
          Colour.x + KgBt709[0] * Colour.y + KgBt709[1] * Colour.z,
          Colour.x + KbBt709    * Colour.y);
      }

      float3 RgbBt2020(float3 Colour)
      {
        return float3(
          Colour.x + KrBt2020    * Colour.z,
          Colour.x + KgBt2020[0] * Colour.y + KgBt2020[1] * Colour.z,
          Colour.x + KbBt2020    * Colour.y);
      }

      float3 RgbAp0D65(float3 Colour)
      {
        return float3(
          Colour.x + KrAp0D65    * Colour.z,
          Colour.x + KgAp0D65[0] * Colour.y + KgAp0D65[1] * Colour.z,
          Colour.x + KbAp0D65    * Colour.y);
      }

    } //YcbcrTo


    namespace RgbTo
    {

      float3 YcbcrBt709(float3 Colour)
      {
        float Y = dot(Colour, KBt709);
        return float3(Y,
                      (Colour.b - Y) / KbBt709,
                      (Colour.r - Y) / KrBt709);
      }

      float3 YcbcrBt2020(float3 Colour)
      {
        float Y = dot(Colour, KBt2020);
        return float3(Y,
                      (Colour.b - Y) / KbBt2020,
                      (Colour.r - Y) / KrBt2020);
      }

      float3 YcbcrAp0D65(float3 Colour)
      {
        float Y = dot(Colour, KAp0D65);
        return float3(Y,
                      (Colour.b - Y) / KbAp0D65,
                      (Colour.r - Y) / KrAp0D65);
      }

    } //RgbTo

  } //Ycbcr


  namespace Mat
  {

    //BT.709 To
    static const float3x3 Bt709ToXYZ = float3x3(
      0.4123907983303070068359375f,    0.3575843274593353271484375f,   0.18048079311847686767578125f,
      0.2126390039920806884765625f,    0.715168654918670654296875f,    0.072192318737506866455078125f,
      0.0193308182060718536376953125f, 0.119194783270359039306640625f, 0.950532138347625732421875f);

    static const float3x3 Bt709ToDciP3 = float3x3(
      0.82246196269989013671875f,    0.17753803730010986328125f,   0.f,
      0.03319419920444488525390625f, 0.96680581569671630859375f,   0.f,
      0.017082631587982177734375f,   0.0723974406719207763671875f, 0.91051995754241943359375f);

    static const float3x3 Bt709ToBt2020 = float3x3(
      0.627403914928436279296875f,      0.3292830288410186767578125f,  0.0433130674064159393310546875f,
      0.069097287952899932861328125f,   0.9195404052734375f,           0.011362315155565738677978515625f,
      0.01639143936336040496826171875f, 0.08801330626010894775390625f, 0.895595252513885498046875f);

    static const float3x3 Bt709ToAp1D65 = float3x3(
      0.61702883243560791015625f,       0.333867609500885009765625f,    0.04910354316234588623046875f,
      0.069922320544719696044921875f,   0.91734969615936279296875f,     0.012727967463433742523193359375f,
      0.02054978720843791961669921875f, 0.107552029192447662353515625f, 0.871898174285888671875f);

    static const float3x3 Bt709ToAp0D65 = float3x3(
      0.4339316189289093017578125f,   0.3762523829936981201171875f,   0.1898159682750701904296875f,
      0.088618390262126922607421875f, 0.809275329113006591796875f,    0.10210628807544708251953125f,
      0.01775003969669342041015625f,  0.109447620809078216552734375f, 0.872802317142486572265625f);


    //DCI-P3 To
    static const float3x3 DciP3ToXYZ = float3x3(
      0.48657095432281494140625f,    0.2656677067279815673828125f,    0.19821728765964508056640625f,
      0.22897456586360931396484375f, 0.691738545894622802734375f,     0.079286910593509674072265625f,
      0.f,                           0.0451133809983730316162109375f, 1.04394435882568359375f);

    static const float3x3 DciP3ToBt709 = float3x3(
       1.22494018077850341796875f,     -0.22494018077850341796875f,     0.f,
      -0.042056955397129058837890625f,  1.04205691814422607421875f,     0.f,
      -0.0196375548839569091796875f,   -0.078636042773723602294921875f, 1.09827363491058349609375f);

    static const float3x3 DciP3ToBt2020 = float3x3(
       0.7538330554962158203125f,         0.198597371578216552734375f,     0.047569595277309417724609375f,
       0.0457438491284847259521484375f,   0.94177722930908203125f,         0.01247893087565898895263671875f,
      -0.001210340298712253570556640625f, 0.0176017172634601593017578125f, 0.98360860347747802734375f);


    //BT.2020 To
    static const float3x3 Bt2020ToXYZ = float3x3(
      0.636958062648773193359375f, 0.144616901874542236328125f,    0.1688809692859649658203125f,
      0.26270020008087158203125f,  0.677998065948486328125f,       0.0593017153441905975341796875f,
      0.f,                         0.028072692453861236572265625f, 1.060985088348388671875f);

    static const float3x3 Bt2020ToBt709 = float3x3(
       1.66049098968505859375f,          -0.58764111995697021484375f,    -0.072849862277507781982421875f,
      -0.12455047667026519775390625f,     1.13289988040924072265625f,    -0.0083494223654270172119140625f,
      -0.01815076358616352081298828125f, -0.100578896701335906982421875f, 1.11872971057891845703125f);

    static const float3x3 Bt2020ToDciP3 = float3x3(
       1.34357821941375732421875f,        -0.2821796834468841552734375f,    -0.06139858067035675048828125f,
      -0.0652974545955657958984375f,       1.07578790187835693359375f,      -0.010490463115274906158447265625f,
       0.002821787260472774505615234375f, -0.0195984952151775360107421875f,  1.01677668094635009765625f);

    static const float3x3 Bt2020ToAp1D65 = float3x3(
      0.982096254825592041015625f,          0.010708245448768138885498046875f, 0.0071955197490751743316650390625f,
      0.001618025242350995540618896484375f, 0.996895968914031982421875f,       0.001485982094891369342803955078125f,
      0.00490146316587924957275390625f,     0.02207522280514240264892578125f,  0.97302329540252685546875f);

    static const float3x3 Bt2020ToAp0D65 = float3x3(
      0.67023181915283203125f,         0.152168750762939453125f,         0.17759941518306732177734375f,
      0.0445011146366596221923828125f, 0.854482352733612060546875f,      0.101016514003276824951171875f,
      0.f,                             0.02577704750001430511474609375f, 0.974222958087921142578125f);


    //AP1 D65 To
    static const float3x3 Ap1D65ToXYZ = float3x3(
       0.647507190704345703125f,         0.13437913358211517333984375f,     0.1685695946216583251953125f,
       0.266086399555206298828125f,      0.67596781253814697265625f,        0.057945795357227325439453125f,
      -0.00544886849820613861083984375f, 0.004072095267474651336669921875f, 1.090434551239013671875f);


    //AP0 D65 To
    static const float3x3 Ap0D65ToXYZ = float3x3(
      0.9503548145294189453125f,  0.f,                      0.000101128956885077059268951416015625f,
      0.34317290782928466796875f, 0.73469638824462890625f, -0.07786929607391357421875f,
      0.f,                        0.f,                      1.08905780315399169921875f);

    static const float3x3 Ap0D65ToBt709 = float3x3(
       2.552483081817626953125f,         -1.12950992584228515625f,      -0.422973215579986572265625f,
      -0.2773441374301910400390625f,      1.3782665729522705078125f,    -0.1009224355220794677734375f,
      -0.01713105104863643646240234375f, -0.14986114203929901123046875f, 1.1669921875f);

    static const float3x3 Ap0D65ToBt2020 = float3x3(
       1.50937116146087646484375f,         -0.261310040950775146484375f,     -0.24806107580661773681640625f,
      -0.078854121267795562744140625f,      1.18762290477752685546875f,      -0.10876882076263427734375f,
       0.0020864079706370830535888671875f, -0.0314234159886837005615234375f,  1.02933704853057861328125f);


    //XYZ To
    static const float3x3 XYZToBt709 = float3x3(
       3.2409698963165283203125f,      -1.53738319873809814453125f,  -0.4986107647418975830078125f,
      -0.96924364566802978515625f,      1.875967502593994140625f,     0.0415550582110881805419921875f,
       0.055630080401897430419921875f, -0.2039769589900970458984375f, 1.05697154998779296875f);

    static const float3x3 XYZToDciP3 = float3x3(
       2.49349689483642578125f,       -0.931383609771728515625f,      -0.40271079540252685546875f,
      -0.8294889926910400390625f,      1.7626640796661376953125f,      0.02362468652427196502685546875f,
       0.03584583103656768798828125f, -0.076172389090061187744140625f, 0.95688450336456298828125f);

    static const float3x3 XYZToBt2020 = float3x3(
       1.7166512012481689453125f,       -0.3556707799434661865234375f,   -0.253366291522979736328125f,
      -0.666684329509735107421875f,      1.61648118495941162109375f,      0.0157685466110706329345703125f,
       0.0176398567855358123779296875f, -0.0427706129848957061767578125f, 0.9421031475067138671875f);

    static const float3x3 XYZToAp1D65 = float3x3(
       1.67890453338623046875f,           -0.33230102062225341796875f,        -0.2418822944164276123046875f,
      -0.661811172962188720703125f,        1.6108245849609375f,                0.0167095959186553955078125f,
       0.010860889218747615814208984375f, -0.0076759266667068004608154296875f, 0.915794551372528076171875f);

    static const float3x3 XYZToAp0D65 = float3x3(
       1.05223858356475830078125f,   0.f,                      -0.0000977099625742994248867034912109375f,
      -0.4914952218532562255859375f, 1.361106395721435546875f,  0.097366832196712493896484375f,
       0.f,                          0.f,                       0.91822493076324462890625f);


    namespace Bt709To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Bt709ToXYZ, Colour);
      }

      float3 DciP3(float3 Colour)
      {
        return mul(Bt709ToDciP3, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(Bt709ToBt2020, Colour);
      }

      float3 Ap1D65(float3 Colour)
      {
        return mul(Bt709ToAp1D65, Colour);
      }

      float3 Ap0D65(float3 Colour)
      {
        return mul(Bt709ToAp0D65, Colour);
      }
    } //Bt709To

    namespace DciP3To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(DciP3ToXYZ, Colour);
      }

      float3 Bt709(float3 Colour)
      {
        return mul(DciP3ToBt709, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(DciP3ToBt2020, Colour);
      }
    } //DciP3To

    namespace Bt2020To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Bt2020ToXYZ, Colour);
      }

      float3 Bt709(float3 Colour)
      {
        return mul(Bt2020ToBt709, Colour);
      }

      float3 DciP3(float3 Colour)
      {
        return mul(Bt2020ToDciP3, Colour);
      }

      float3 Ap1D65(float3 Colour)
      {
        return mul(Bt2020ToAp1D65, Colour);
      }

      float3 Ap0D65(float3 Colour)
      {
        return mul(Bt2020ToAp0D65, Colour);
      }
    } //Bt2020To

    namespace Ap0D65To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Ap0D65ToXYZ, Colour);
      }

      float3 Bt709(float3 Colour)
      {
        return mul(Ap0D65ToBt709, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(Ap0D65ToBt2020, Colour);
      }
    } //Ap0D65To

    namespace Ap1D65To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Ap1D65ToXYZ, Colour);
      }
    } //Ap1D65To

    namespace XYZTo
    {
      float3 Bt709(float3 Colour)
      {
        return mul(XYZToBt709, Colour);
      }

      float3 DciP3(float3 Colour)
      {
        return mul(XYZToDciP3, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(XYZToBt2020, Colour);
      }

      float3 Ap1D65(float3 Colour)
      {
        return mul(XYZToAp1D65, Colour);
      }

      float3 Ap0D65(float3 Colour)
      {
        return mul(XYZToAp0D65, Colour);
      }
    } //XYZTo

  } //Mat


  namespace Ictcp
  {

    namespace Mat
    {

      //L'M'S'->ICtCp
      #define PqLmsToIctcp float3x3( \
        asfloat(0x3f000000), asfloat(0x3f000000), asfloat(0x00000000), \
        asfloat(0x3fce9000), asfloat(0xc054b400), asfloat(0x3fdad800), \
        asfloat(0x408c1a00), asfloat(0xc087dc00), asfloat(0xbe07c000)) \

      //ICtCp->L'M'S'
      #define IctcpToPqLms float3x3( \
        asfloat(0x3f800000), asfloat(0x3c0d0ceb), asfloat(0x3de36380), \
        asfloat(0x3f800000), asfloat(0xbc0d0ceb), asfloat(0xbde36380), \
        asfloat(0x3f800000), asfloat(0x3f0f5e37), asfloat(0xbea4293f)) \


      //RGB BT.709->LMS
      #define Bt709ToLms float3x3( \
        asfloat(0x3e976000), asfloat(0x3f1f9000), asfloat(0x3da60000), \
        asfloat(0x3e1fc000), asfloat(0x3f3a4000), asfloat(0x3dee8000), \
        asfloat(0x3d100000), asfloat(0x3e208000), asfloat(0x3f4ed000)) \

      //RGB DCI-P3->LMS
      #define DciP3ToLms float3x3( \
        asfloat(0x3eab2000), asfloat(0x3f139000), asfloat(0x3db68000), \
        asfloat(0x3e224000), asfloat(0x3f36b000), asfloat(0x3e030000), \
        asfloat(0x3ca80000), asfloat(0x3dbc0000), asfloat(0x3f632000)) \

      //RGB BT.2020->LMS
      #define Bt2020ToLms float3x3( \
        asfloat(0x3ed30000), asfloat(0x3f062000), asfloat(0x3d830000), \
        asfloat(0x3e2ac000), asfloat(0x3f387000), asfloat(0x3de70000), \
        asfloat(0x3cc60000), asfloat(0x3d9a8000), asfloat(0x3f668000)) \

      //RGB AP0_D65->LMS
      #define Ap0D65ToLms float3x3( \
        asfloat(0x3f14a000), asfloat(0x3f033000), asfloat(0xbdbf0000), \
        asfloat(0x3e478000), asfloat(0x3f4f0000), asfloat(0xbb600000), \
        asfloat(0x3d040000), asfloat(0x3d610000), asfloat(0x3f698000)) \


      //LMS->RGB BT.709
      #define LmsToBt709 float3x3( \
        asfloat(0x40c57ba5), asfloat(0xc0aa33fb), asfloat(0x3e171433), \
        asfloat(0xbfa92286), asfloat(0x4023ac35), asfloat(0xbe71be38), \
        asfloat(0xbc47d18b), asfloat(0xbe87882b), asfloat(0x3fa37be5)) \

      //LMS->RGB DCI-P3
      #define LmsToDciP3 float3x3( \
        asfloat(0x409b273c), asfloat(0xc07b4bdf), asfloat(0x3da22dbf), \
        asfloat(0xbf89c7ab), asfloat(0x40132ae0), asfloat(0xbe64d21d), \
        asfloat(0xba37da3e), asfloat(0xbe16b154), asfloat(0x3f92ff84)) \

      //LMS->RGB BT.2020
      #define LmsToBt2020 float3x3( \
        asfloat(0x405bf15d), asfloat(0xc02069b6), asfloat(0x3d8f0b1e), \
        asfloat(0xbf4a9493), asfloat(0x3ffde69f), asfloat(0xbe44e2a9), \
        asfloat(0xbcd494e2), asfloat(0xbdca9346), asfloat(0x3f8ffb88)) \

      //LMS->RGB AP0_D65
      #define LmsToAp0D65 float3x3( \
        asfloat(0x400b6fa4), asfloat(0xbfb2a0ea), asfloat(0x3e5ec243), \
        asfloat(0xbf066ee2), asfloat(0x3fc95067), asfloat(0xbd43c9e9), \
        asfloat(0xbd39c263), asfloat(0xbd3a029f), asfloat(0x3f8bb7fe)) \

      namespace IctcpTo
      {
        //ICtCp->L'M'S'
        float3 PqLms(float3 Colour)
        {
          return mul(IctcpToPqLms, Colour);
        }
      } //IctcpTo

      namespace PqLmsTo
      {
        //L'M'S'->ICtCp
        float3 Ictcp(float3 Colour)
        {
          return mul(PqLmsToIctcp, Colour);
        }
      } //PqLmsTo

      namespace Bt709To
      {
        //RGB BT.709->LMS
        float3 Lms(float3 Colour)
        {
          return mul(Bt709ToLms, Colour);
        }
      } //Bt709To

      namespace DciP3To
      {
        //RGB DCI-P3->LMS
        float3 Lms(float3 Colour)
        {
          return mul(DciP3ToLms, Colour);
        }
      } //DciP3To

      namespace Bt2020To
      {
        //RGB BT.2020->LMS
        float3 Lms(float3 Colour)
        {
          return mul(Bt2020ToLms, Colour);
        }
      } //Bt2020To

      namespace Ap0D65To
      {
        //RGB AP0_D65->LMS
        float3 Lms(float3 Colour)
        {
          return mul(Ap0D65ToLms, Colour);
        }
      } //Ap0D65To

      namespace LmsTo
      {
        //LMS->RGB BT.709
        float3 Bt709(float3 Colour)
        {
          return mul(LmsToBt709, Colour);
        }

        //LMS->RGB DCI-P3
        float3 DciP3(float3 Colour)
        {
          return mul(LmsToDciP3, Colour);
        }

        //LMS->RGB BT.2020
        float3 Bt2020(float3 Colour)
        {
          return mul(LmsToBt2020, Colour);
        }

        //LMS->RGB AP0_D65
        float3 Ap0D65(float3 Colour)
        {
          return mul(LmsToAp0D65, Colour);
        }
      } //LmsTo

    } //Mat

  } //ICtCp


  namespace OkLab
  {

//  Copyright (c) 2021 Björn Ottosson
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of
//  this software and associated documentation files (the "Software"), to deal in
//  the Software without restriction, including without limitation the rights to
//  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//  of the Software, and to permit persons to whom the Software is furnished to do
//  so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

    //RGB BT.709->OKLMS
    #define Bt709ToOkLms float3x3( \
      asfloat(0x3ed2e753), asfloat(0x3f095229), asfloat(0x3d528e15), \
      asfloat(0x3e58dc50), asfloat(0x3f2e4ed6), asfloat(0x3ddbcdc4), \
      asfloat(0x3db4d16f), asfloat(0x3e905888), asfloat(0x3f213db6))

    //RGB BT.2020->OKLMS
    #define Bt2020ToOkLms float3x3( \
      asfloat(0x3f1dd1c2), asfloat(0x3eb86f63), asfloat(0x3cbca825), \
      asfloat(0x3e87b4d1), asfloat(0x3f22cf19), asfloat(0x3dcab10c), \
      asfloat(0x3dcd0a32), asfloat(0x3e50f045), asfloat(0x3f3226d0))

    //OKL'M'S'->OKLab
    #define G3OkLmsToOkLab float3x3 ( \
      0.2104542553f,  0.7936177850f, -0.0040720468f, \
      1.9779984951f, -2.4285922050f,  0.4505937099f, \
      0.0259040371f,  0.7827717662f, -0.8086757660f)

    //OKLab->OKL'M'S'
    #define OkLabToG3OkLms float3x3 ( \
      1.f,  0.3963377774f,  0.2158037573f, \
      1.f, -0.1055613458f, -0.0638541728f, \
      1.f, -0.0894841775f, -1.2914855480f)

    //OKLMS->RGB BT.709
    #define OkLmsToBt709 float3x3 ( \
      asfloat(0x40828d05), asfloat(0xc053d1ae), asfloat(0x3e6c8bde), \
      asfloat(0xbfa2562d), asfloat(0x4026fa47), asfloat(0xbeaea0a2), \
      asfloat(0xbb899b59), asfloat(0xbf3431b1), asfloat(0x3fda9ebe))

    //OKLMS->RGB BT.2020
    #define OkLmsToBt2020 float3x3 ( \
      asfloat(0x400903cf), asfloat(0xbf9f9647), asfloat(0x3dda0b93), \
      asfloat(0xbf6279cc), asfloat(0x400a6af4), asfloat(0xbe8e7ec1), \
      asfloat(0xbd471993), asfloat(0xbee8d6fc), asfloat(0x3fc06aec))


// provided matrices
//
//    //RGB BT.709->OKLMS
//    #define  Bt709ToOkLms = float3x3(
//      0.4122214708f, 0.5363325363f, 0.0514459929f, \
//      0.2119034982f, 0.6806995451f, 0.1073969566f, \
//      0.0883024619f, 0.2817188376f, 0.6299787005f) \
//
//    //OKLMS->RGB BT.709
//    #define OkLmsToBt709 float3x3 ( \
//       4.0767416621f, -3.3077115913f,  0.2309699292f, \
//      -1.2684380046f,  2.6097574011f, -0.3413193965f, \
//      -0.0041960863f, -0.7034186147f,  1.7076147010f) \

    namespace OkLmsTo
    {

      //OKLMS->OKL'M'S'
      float3 G3OkLms(float3 Lms)
      {
        //apply gamma 3
        return sign(Lms) * pow(abs(Lms), 1.f / 3.f);
      }

    } //OkLmsTo

    namespace G3OkLmsTo
    {

      //OKL'M'S'->OKLab
      float3 OkLab(float3 G3Lms)
      {
        return mul(G3OkLmsToOkLab, G3Lms);
      }

      //OKL'M'S'->OKLMS
      float3 OkLms(float3 G3Lms)
      {
        //remove gamma 3
        return G3Lms * G3Lms * G3Lms;
      }

    } //G3OkLmsTo

    namespace OkLabTo
    {

      //OKLab->OKL'M'S'
      float3 G3OkLms(float3 Lab)
      {
        return mul(OkLabToG3OkLms, Lab);
      }

    } //OkLabTo

    namespace Bt709To
    {

      //RGB BT.709->OKLab
      float3 OkLab(float3 Rgb)
      {
        //to OKLMS
        float3 OkLms = mul(Bt709ToOkLms, Rgb);

        //to OKL'M'S'
        //apply gamma 3
        float3 g3OkLms = Csp::OkLab::OkLmsTo::G3OkLms(OkLms);

        //to OKLab
        return Csp::OkLab::G3OkLmsTo::OkLab(g3OkLms);
      }

    } //Bt709To

    namespace Bt2020To
    {

      //RGB BT.2020->OKLab
      float3 OkLab(float3 Rgb)
      {
        //to OKLMS
        float3 OkLms = mul(Bt2020ToOkLms, Rgb);

        //to OKL'M'S'
        //apply gamma 3
        float3 g3OkLms = Csp::OkLab::OkLmsTo::G3OkLms(OkLms);

        //to OKLab
        return Csp::OkLab::G3OkLmsTo::OkLab(g3OkLms);
      }

    } //Bt2020To

    namespace OkLabTo
    {

      //OKLab->RGB BT.709
      float3 Bt709(float3 Lab)
      {
        //to OKL'M'S'
        float3 g3OkLms = mul(OkLabToG3OkLms, Lab);

        //to OKLMS
        //remove gamma 3
        float3 okLms = Csp::OkLab::G3OkLmsTo::OkLms(g3OkLms);

        //to RGB BT.709
        return mul(OkLmsToBt709, okLms);
      }

      //OKLab->RGB BT.2020
      float3 Bt2020(float3 Lab)
      {
        //to OKL'M'S'
        float3 g3OkLms = mul(OkLabToG3OkLms, Lab);

        //to OKLMS
        //remove gamma 3
        float3 okLms = Csp::OkLab::G3OkLmsTo::OkLms(g3OkLms);

        //to RGB BT.2020
        return mul(OkLmsToBt2020, okLms);
      }

    } //OkLabTo


    // Finds the maximum saturation possible for a given hue that fits in sRGB
    // Saturation here is defined as S = C/L
    // a and b must be normalized so a^2 + b^2 == 1
    float ComputeMaxSaturation(float2 ab)
    {
      // Max saturation will be when one of r, g or b goes below zero.

      // Select different coefficients depending on which component goes below zero first
      float k0, k1, k2, k3, k4;

      float3 wLms;

      if (-1.88170328f * ab.x - 0.80936493f * ab.y > 1)
      {
        // Red component
        k0 = 1.19086277f;
        k1 = 1.76576728f;
        k2 = 0.59662641f;
        k3 = 0.75515197f;
        k4 = 0.56771245f;

        wLms.rgb = OkLmsToBt709[0].rgb;
      }
      else if (1.81444104f * ab.x - 1.19445276f * ab.y > 1)
      {
        // Green component
        k0 =  0.73956515f;
        k1 = -0.45954404f;
        k2 =  0.08285427f;
        k3 =  0.12541070f;
        k4 =  0.14503204f;

        wLms.rgb = OkLmsToBt709[1].rgb;
      }
      else
      {
        // Blue component
        k0 =  1.35733652f;
        k1 = -0.00915799f;
        k2 = -1.15130210f;
        k3 = -0.50559606f;
        k4 =  0.00692167f;

        wLms.rgb = OkLmsToBt709[2].rgb;
      }

      // Approximate max saturation using a polynomial:
      float s = k0
              + k1 * ab.x
              + k2 * ab.y
              + k3 * ab.x * ab.x
              + k4 * ab.x * ab.y;

      // Do one step Halley's method to get closer
      // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
      // this should be sufficient for most applications, otherwise do two/three steps

      float3 kLms = mul(OkLabToG3OkLms, float3(0.f, ab));

      {
        float3 g3Lms = 1.f + s * kLms;

        float3 intermediateLms = g3Lms * g3Lms;

        float3 lms = intermediateLms * g3Lms;

        float3 g3LmsdS  = 3.f * kLms * intermediateLms;
        float3 g3LmsdS2 = 6.f * kLms * kLms * g3Lms;

        float3 f = mul(float3x3(lms,
                                g3LmsdS,
                                g3LmsdS2), wLms);

        s = s
          - f.x * f.y
          / (f.y * f.y - 0.5f * f.x * f.z);
      }

      return s;
    }

    // finds L_cusp and C_cusp for a given hue
    // a and b must be normalized so a^2 + b^2 == 1
    float2 FindCusp(float2 ab)
    {
      // First, find the maximum saturation (saturation S = C/L)
      float sCusp = ComputeMaxSaturation(ab);

      float2 lcCusp;

      // Convert to linear sRGB to find the first point where at least one of r, g or b >= 1:
      float3 rgbAtMax = Csp::OkLab::OkLabTo::Bt709(float3(1.f, sCusp * ab));

      lcCusp.x = pow(1.f / max(max(rgbAtMax.r, rgbAtMax.g), rgbAtMax.b), 1.f / 3.f);
      lcCusp.y = lcCusp.x * sCusp;

      return lcCusp;
    }

    float2 ToSt(float2 LC)
    {
      return LC.y / float2(LC.x,
                           1.f - LC.x);
    }

    static const float ToeK1 = 0.206f;
    static const float ToeK2 = 0.03f;
    static const float ToeK3 = (1.f + ToeK1) / (1.f + ToeK2);

    // toe function for L_r
    float Toe(float X)
    {
      float i0 = ToeK3 * X - ToeK1;
      return 0.5f * (i0 + sqrt(i0 * i0 + 4.f * ToeK2 * ToeK3 * X));
    }

    // inverse toe function for L_r
    float ToeInv(float X)
    {
      return (X * X     + ToeK1 * X)
           / (X * ToeK3 + ToeK2 * ToeK3);
    }

    float2 GetStMid(float2 ab)
    {

      float s = 0.11516993f
              + 1.f / (7.44778970f
                     + 4.15901240f * ab.y
                     + ab.x * (-2.19557347f
                              + 1.75198401f * ab.y
                              + ab.x * (-2.13704948f
                                      - 10.02301043f * ab.y
                                      + ab.x * (-4.24894561f
                                               + 5.38770819f * ab.y
                                               + 4.69891013f * ab.x))));

      float t = 0.11239642f
              + 1.f / (1.61320320f
                     - 0.68124379f * ab.y
                     + ab.x * (0.40370612f
                             + 0.90148123f * ab.y
                             + ab.x * (-0.27087943f
                                      + 0.61223990f * ab.y
                                      + ab.x * (0.00299215f
                                              - 0.45399568f * ab.y
                                              - 0.14661872f * ab.x))));

      return float2(s, t);
    }

    float FindGamutIntersection(
      float2 ab,
      float  L1,
      float  C1,
      float  L0,
      float2 cusp)
    {
      // Find the intersection for upper and lower half seprately
      float t;

      if (((L1 - L0) * cusp.y - (cusp.x - L0) * C1) <= 0.f)
      {
        // Lower half
        t = cusp.y * L0 / (C1 * cusp.x + cusp.y * (L0 - L1));
      }
      else
      {
        // Upper half
        // First intersect with triangle
        t = cusp.y * (L0 - 1.f) / (C1 * (cusp.x - 1.f) + cusp.y * (L0 - L1));

        // Then one step Halley's method
        {
          float dL = L1 - L0;
          float dC = C1;

          float3 kLms = mul(OkLabToG3OkLms, float3(0.f, ab));

          float3 g3LmsDt = dL + dC * kLms;

          // If higher accuracy is required, 2 or 3 iterations of the following block can be used:
          {
            float L = L0 * (1.f - t) + t * L1;
            float C = t * C1;

            float3 g3Lms = L + C * kLms;

            float3 intermediateLms = g3Lms * g3Lms;

            float3 lms = g3Lms * intermediateLms;

            float3 lmsDt  = 3.f * g3LmsDt * intermediateLms;
            float3 lmsDt2 = 6.f * g3LmsDt * g3LmsDt * g3Lms;

            static const float3x3 iLms = float3x3(lms.xyz,
                                                  lmsDt.xyz,
                                                  lmsDt2.xyz);

            static const float3 ir = OkLmsToBt709[0].rgb;

            static const float3 ig = OkLmsToBt709[1].rgb;

            static const float3 ib = OkLmsToBt709[2].rgb;

            float3 r = mul(iLms, ir);

            r.x -= 1.f;

            float u_r = r.y / (r.y * r.y - 0.5f * r.x * r.z);
            float t_r = -r.x * u_r;

            float3 g = mul(iLms, ig);

            g.x -= 1.f;

            float u_g = g.y / (g.y * g.y - 0.5f * g.x * g.z);
            float t_g = -g.x * u_g;

            float3 b = mul(iLms, ib);

            b.x -= 1.f;

            float u_b = b.y / (b.y * b.y - 0.5f * b.x * b.z);
            float t_b = -b.x * u_b;

            t_r = u_r >= 0.f ? t_r : FP32_MAX;
            t_g = u_g >= 0.f ? t_g : FP32_MAX;
            t_b = u_b >= 0.f ? t_b : FP32_MAX;

            t += min(t_r, min(t_g, t_b));
          }
        }
      }

      return t;
    }

    float3 GetCs(float3 Lab)
    {
      float2 cusp = FindCusp(Lab.yz);

      float   C_max = FindGamutIntersection(Lab.yz, Lab.x, 1.f, Lab.x, cusp);
      float2 ST_max = ToSt(cusp);

      // Scale factor to compensate for the curved part of gamut shape:
      float k = C_max / min((Lab.x * ST_max.x),
                            (1.f - Lab.x) * ST_max.y);

      float C_mid;
      {
        float2 ST_mid = GetStMid(Lab.yz);

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        float2 C_ab = Lab.x * ST_mid;

        C_ab.y = ST_mid.y - C_ab.y;

        C_mid = 0.9f * k * sqrt(sqrt(1.f
                                   / (1.f / (C_ab.x * C_ab.x * C_ab.x * C_ab.x)
                                    + 1.f / (C_ab.y * C_ab.y * C_ab.y * C_ab.y))));
      }

      float C_0;
      {
        // for C_0, the shape is independent of hue, so ST are constant. Values picked to roughly be the average values of ST.
        float C_a = Lab.x * 0.4f;
        float C_b = (1.f - Lab.x) * 0.8f;

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        C_0 = sqrt(1.f
                 / (1.f / (C_a * C_a)
                  + 1.f / (C_b * C_b)));
      }

      return float3(C_0, C_mid, C_max);
    }

    namespace OkLabTo
    {

      //OKLab->OKHSV
      float3 OkHsv(float3 Lab)
      {
        float2 LC;

        LC.x = Lab.x;
        LC.y = sqrt(Lab.y * Lab.y
                  + Lab.z * Lab.z);

        float2 ab = Lab.yz / LC.y;

        float3 hsv;

        hsv.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) / PI;

        float2 cusp = Csp::OkLab::FindCusp(ab);

        float2 stMax = Csp::OkLab::ToSt(cusp);

        float s0 = 0.5f;
        float k = 1.f - s0 / stMax.x;

        // first we find L_v, C_v, L_vt and C_vt

        float t = stMax.y / (LC.y + LC.x * stMax.y);
        float2 LC_v = LC * t;

        float L_vt = ToeInv(LC_v.x);
        float C_vt = LC_v.y * L_vt / LC_v.x;

        // we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
        float3 rgbScale = Csp::OkLab::OkLabTo::Bt709(float3(L_vt,
                                                            ab * C_vt));

        float scaleL = pow(1.f / max(max(rgbScale.r, rgbScale.g), max(rgbScale.b, 0.f)), 1.f / 3.f);

        LC /= scaleL;

        float toeL = Toe(LC.x);

        LC.y = LC.y * toeL / LC.x;
        LC.x = toeL;

        // we can now compute v and s:

        hsv.z = LC.x / LC_v.x;
        hsv.y = (s0 + stMax.y) * LC_v.y / ((stMax.y * s0) + stMax.y * k * LC_v.y);

        return hsv;
      }

      //OKLab->OKHSL
      float3 OkHsl(float3 Lab)
      {
        float C;

        float2 ab = Lab.yz;

        C = sqrt(Lab.y * Lab.y
               + Lab.z * Lab.z);

        ab /= C;

        float3 hsl;

        hsl.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) / PI;

        float3 cs = GetCs(float3(Lab.x, ab));

        float C_0   = cs.x;
        float C_mid = cs.y;
        float C_max = cs.z;

        // Inverse of the interpolation in okhsl_to_srgb:
        float mid    = 0.8f;
        float midInv = 1.25f;

        float s;
        if (C < C_mid)
        {
          float k1 = mid * C_0;

          float k2 = 1.f
                   - k1 / C_mid;

          float t = C
                  / (k1 + k2 * C);

          hsl.y = t * mid;
        }
        else
        {
          float k0 = C_mid;

          float k1 = (1.f - mid)
                   * C_mid * C_mid
                   * midInv * midInv / C_0;

          float k2 = 1.f
                   - k1 / (C_max - C_mid);

          float t = (C - k0) / (k1 + k2 * (C - k0));

          hsl.y = mid + (1.f - mid) * t;
        }

        hsl.z = Csp::OkLab::Toe(Lab.x);

        return hsl;
      }

    }

    namespace Bt709To
    {

      //RGB BT.709->OKHSV
      float3 OkHsv(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt709To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsv(lab);
      }

      //RGB BT.709->OKHSL
      float3 OkHsl(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt709To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsl(lab);
      }

    } //Bt709To

    namespace Bt2020To
    {

      //RGB BT.2020->OKHSV
      float3 OkHsv(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt2020To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsv(lab);
      }

      //RGB BT.2020->OKHSL
      float3 OkHsl(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt2020To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsl(lab);
      }

    } //Bt2020To

    namespace OkHsvTo
    {
      //OKHSV->OKLab
      float3 OkLab(float3 Hsv)
      {
        float i_ab = 2.f * PI * Hsv.x;

        float2 ab = float2(cos(i_ab),
                           sin(i_ab));

        float2 cusp = Csp::OkLab::FindCusp(ab);

        float2 stMax = Csp::OkLab::ToSt(cusp);

        float s0 = 0.5f;
        float k = 1.f - s0 / stMax.x;

        // first we compute L and V as if the gamut is a perfect triangle:

        // L, C when v==1:
        float i_num = Hsv.y * s0;
        float i_den = s0 + stMax.y - stMax.y * k * Hsv.y;

        float i_div = i_num / i_den;

        float L_v =     1.f - i_div;
        float C_v = stMax.y * i_div;

        float2 LC = float2(L_v, C_v) * Hsv.z;

        // then we compensate for both toe and the curved top part of the triangle:
        float L_vt = Csp::OkLab::ToeInv(L_v);
        float C_vt = C_v * L_vt / L_v;

        float L_new = Csp::OkLab::ToeInv(LC.x);

        LC.y = LC.y * L_new / LC.x;
        LC.x = L_new;

        float3 rgbScale = Csp::OkLab::OkLabTo::Bt709(float3(L_vt,
                                                            ab * C_vt));

        float scaleL = pow(1.f / max(max(rgbScale.r, rgbScale.g), max(rgbScale.b, 0.f)), 1.f / 3.f);

        LC *= scaleL;

        return float3(LC.x,
                      ab * LC.y);
      }

      //OKHSV->BT.709
      float3 Bt709(float3 Hsv)
      {
        float3 lab = Csp::OkLab::OkHsvTo::OkLab(Hsv);

        return Csp::OkLab::OkLabTo::Bt709(lab);
      }

      //OKHSV->BT.2020
      float3 Bt2020(float3 Hsv)
      {
        float3 lab = Csp::OkLab::OkHsvTo::OkLab(Hsv);

        return Csp::OkLab::OkLabTo::Bt2020(lab);
      }
    } //OkHsvTo

    namespace OkHslTo
    {
      //OKHSL->OKLab
      float3 OkLab(float3 Hsl)
      {
        if (Hsl.z == 1.f)
        {
          return 1.f;
        }
        else if (Hsl.z == 0.f)
        {
          return 0.f;
        }

        float  L;
        float2 ab;

        float i_ab = 2.f * PI * Hsl.x;

        L = Csp::OkLab::ToeInv(Hsl.z);
        ab = float2(cos(i_ab),
                    sin(i_ab));

        float3 cs = Csp::OkLab::GetCs(float3(L, ab));

        float C_0   = cs.x;
        float C_mid = cs.y;
        float C_max = cs.z;

        // Interpolate the three values for C so that:
        // At s=0: dC/ds = C_0, C=0
        // At s=0.8: C=C_mid
        // At s=1.0: C=C_max

        float mid    = 0.8f;
        float midInv = 1.25f;

        float C,
              t,
              k0,
              k1,
              k2;

        if (Hsl.y < mid)
        {
          t = midInv * Hsl.y;

          k1 = mid * C_0;
          k2 = 1.f
             - k1 / C_mid;

          C = t * k1 / (1.f - k2 * t);
        }
        else
        {
          float i0 = 1.f - mid;

          t = (Hsl.y - mid) / i0;

          k0 = C_mid;

          k1 = i0
             * C_mid * C_mid
             * midInv * midInv
             / C_0;

          k2 = 1.f
             - k1 / (C_max - C_mid);

          C = k0
            + t * k1
            / (1.f - k2 * t);
        }

        return float3(L,
                      ab * C);
      }

      //OKHSL->BT.709
      float3 Bt709(float3 Hsl)
      {
        float3 lab = Csp::OkLab::OkHslTo::OkLab(Hsl);

        return Csp::OkLab::OkLabTo::Bt709(lab);
      }

      //OKHSL->BT.2020
      float3 Bt2020(float3 Hsl)
      {
        float3 lab = Csp::OkLab::OkHslTo::OkLab(Hsl);

        return Csp::OkLab::OkLabTo::Bt2020(lab);
      }
    } //OkHslTo

  } //OkLab


  namespace Map
  {

    namespace Bt709Into
    {

      float3 Scrgb(
        const float3 Colour,
        const float  Brightness)
      {
        return Colour / 80.f * Brightness;
      }

      float3 Hdr10(
        const float3 Colour,
        const float  Brightness)
      {
        return Csp::Trc::NitsTo::Pq(Csp::Mat::Bt709To::Bt2020(Colour) * Brightness);
      }

      float3 Hlg(
        const float3 Colour,
        const float  Brightness)
      {
        return Csp::Trc::NitsTo::Hlg(Csp::Mat::Bt709To::Bt2020(Colour) * Brightness);
      }

      float3 Ps5(
        const float3 Colour,
        const float  Brightness)
      {
        return Csp::Mat::Bt709To::Bt2020(Colour / 100.f) * Brightness;
      }

    } //Bt709Into

  }

}


struct Sxy
{
  float x;
  float y;
};


Sxy GetxyFromXYZ(float3 XYZ)
{
  const float xyz = XYZ.x + XYZ.y + XYZ.z;

  Sxy xy;

  xy.x = XYZ.x / xyz;

  xy.y = XYZ.y / xyz;

  return xy;
}

float3 GetXYZfromxyY(Sxy xy, float Y)
{
  float3 XYZ;

  XYZ.x = (xy.x / xy.y)
        * Y;

  XYZ.y = Y;

  XYZ.z = ((1.f - xy.x - xy.y) / xy.y)
        * Y;

  return XYZ;
}

float3 GetXYZfromxyY(float2 xy, float Y)
{
  float3 XYZ;

  XYZ.x = (xy.x / xy.y)
        * Y;

  XYZ.y = Y;

  XYZ.z = ((1.f - xy.x - xy.y) / xy.y)
        * Y;

  return XYZ;
}


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


bool IsNAN(float Input)
{
  if (isnan(Input) || isinf(Input))
    return true;
  else
    return false;
}

float fixNAN(float Input)
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
