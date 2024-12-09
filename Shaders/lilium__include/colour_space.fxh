#pragma once


#define RESHADE_MAJOR    6
#define RESHADE_MINOR    3
#define RESHADE_REVISION 3

#define RESHADE_VERSION (RESHADE_MAJOR * 10000 \
                       + RESHADE_MINOR *   100 \
                       + RESHADE_REVISION)

#if (__RESHADE__ < RESHADE_VERSION)
  #error "ReShade version too old! Please update!"
#endif


#pragma warning(disable : 3571) // disable warning about potentially using pow on a negative value


texture TextureBackBuffer : COLOR;

sampler SamplerBackBuffer
{
  Texture = TextureBackBuffer;
};


static const float2 PIXEL_SIZE = float2(BUFFER_RCP_WIDTH, BUFFER_RCP_HEIGHT);

static const  uint BUFFER_WIDTH_UINT   =  uint(BUFFER_WIDTH);
static const  uint BUFFER_HEIGHT_UINT  =  uint(BUFFER_HEIGHT);
static const   int BUFFER_WIDTH_INT    =   int(BUFFER_WIDTH);
static const   int BUFFER_HEIGHT_INT   =   int(BUFFER_HEIGHT);
static const float BUFFER_WIDTH_FLOAT  = float(BUFFER_WIDTH);
static const float BUFFER_HEIGHT_FLOAT = float(BUFFER_HEIGHT);

static const  uint2 BUFFER_SIZE_UINT  =  uint2(BUFFER_WIDTH_UINT,  BUFFER_HEIGHT_UINT);
static const   int2 BUFFER_SIZE_INT   =   int2(BUFFER_WIDTH_INT,   BUFFER_HEIGHT_INT);
static const float2 BUFFER_SIZE_FLOAT = float2(BUFFER_WIDTH_FLOAT, BUFFER_HEIGHT_FLOAT);

static const  uint PIXEL_COUNT_UINT  = BUFFER_WIDTH_UINT * BUFFER_HEIGHT_UINT;
static const float PIXEL_COUNT_FLOAT = float(PIXEL_COUNT_UINT);

static const uint BUFFER_WIDTH_MINUS_1_UINT  = BUFFER_WIDTH_UINT  - 1u;
static const uint BUFFER_HEIGHT_MINUS_1_UINT = BUFFER_HEIGHT_UINT - 1u;

static const uint BUFFER_WIDTH_MINUS_1_INT  = BUFFER_WIDTH_INT  - 1;
static const uint BUFFER_HEIGHT_MINUS_1_INT = BUFFER_HEIGHT_INT - 1;

static const float BUFFER_WIDTH_MINUS_1_FLOAT  = float(BUFFER_WIDTH_MINUS_1_UINT);
static const float BUFFER_HEIGHT_MINUS_1_FLOAT = float(BUFFER_HEIGHT_MINUS_1_UINT);

static const  uint2 BUFFER_SIZE_MINUS_1_UINT  =  uint2(BUFFER_WIDTH_MINUS_1_UINT,  BUFFER_HEIGHT_MINUS_1_UINT);
static const  uint2 BUFFER_SIZE_MINUS_1_INT   =   int2(BUFFER_WIDTH_MINUS_1_INT,   BUFFER_HEIGHT_MINUS_1_INT);
static const float2 BUFFER_SIZE_MINUS_1_FLOAT = float2(BUFFER_WIDTH_MINUS_1_FLOAT, BUFFER_HEIGHT_MINUS_1_FLOAT);


//#ifndef __RESHADE__
//  #include "_no.fxh"
//  #define BUFFER_WIDTH       3840
//  #define BUFFER_HEIGHT      2160
//  #define BUFFER_COLOR_SPACE    2
//#endif


#if (__RENDERER__ >= 0x9000  \
  && __RENDERER__ <  0xA000)
  #define API_IS_D3D9
#elif (__RENDERER__ >= 0xA000  \
    && __RENDERER__ <  0xB000)
  #define API_IS_D3D10
#elif (__RENDERER__ >= 0xB000  \
    && __RENDERER__ <  0xC000)
  #define API_IS_D3D11
#elif (__RENDERER__ >= 0xC000  \
    && __RENDERER__ <  0xD000)
  #define API_IS_D3D12
#elif (__RENDERER__ >= 0x10000  \
    && __RENDERER__ <  0x20000)
  #define API_IS_OPENGL
#elif (__RENDERER__ >= 0x20000  \
    && __RENDERER__ <  0x30000)
  #define API_IS_VULKAN
#endif


#if (BUFFER_WIDTH  >= 2560) \
 && (BUFFER_HEIGHT >= 1440)
  #define IS_QHD_OR_HIGHER_RES
#endif

#if (BUFFER_WIDTH  > 2560) \
 && (BUFFER_HEIGHT > 1440)
  #define IS_HIGHER_THAN_QHD_RES
#endif


#ifdef GAMESCOPE
  #ifndef GAMESCOPE_SDR_ON_HDR_NITS
    #define GAMESCOPE_SDR_ON_HDR_NITS 203.f
  #endif
#endif


#define STRINGIFY(x) #x
#define GET_UNKNOWN_NUMBER(x) "unknown (" STRINGIFY(x) ")"

// Vertex shader generating a triangle covering the entire screen
// See also https://www.reddit.com/r/gamedev/comments/2j17wk/a_slightly_faster_bufferless_vertex_shader_trick/
void VS_PostProcess
(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position,
  out float2 TexCoord : TEXCOORD0
)
{
	TexCoord.x = (VertexID == 2) ? 2.f : 0.f;
	TexCoord.y = (VertexID == 1) ? 2.f : 0.f;

	Position = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);
}

void VS_PostProcessWithoutTexCoord
(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position
)
{
  float2 texCoord;
  texCoord.x = (VertexID == 2) ? 2.f : 0.f;
  texCoord.y = (VertexID == 1) ? 2.f : 0.f;

  Position = float4(texCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);
}

#define YES 1
#define NO  0

#define CSP_UNKNOWN 0
#define CSP_SRGB    1
#define CSP_SCRGB   2
#define CSP_HDR10   3
#define CSP_HLG     4
#define CSP_PS5     5
#define CSP_UNSET   254
#define CSP_FAIL    255

#if (BUFFER_COLOR_BIT_DEPTH == 8 || BUFFER_COLOR_BIT_DEPTH == 10)
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


#if ((BUFFER_COLOR_SPACE != CSP_SCRGB && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH)) \
  || (BUFFER_COLOR_SPACE != CSP_HDR10 && defined(IS_POSSIBLE_HDR10_BIT_DEPTH)))

  #if (BUFFER_COLOR_SPACE == CSP_SRGB)
    #define ORG_BUFFER_COLOR_SPACE CSP_SRGB
  #else
    #define ORG_BUFFER_COLOR_SPACE CSP_UNKNOWN
  #endif

  #undef BUFFER_COLOR_SPACE
#endif

#ifndef BUFFER_COLOR_SPACE
  #ifdef IS_POSSIBLE_SCRGB_BIT_DEPTH
    #ifndef CSP_OVERRIDE
      #define CSP_OVERRIDE CSP_UNSET
    #endif
  #endif
  #ifdef IS_POSSIBLE_HDR10_BIT_DEPTH
    #ifndef CSP_OVERRIDE
      #define CSP_OVERRIDE CSP_UNSET
    #endif
  #endif
#else
  #undef CSP_OVERRIDE //for some reason this workaround is needed in DS1PTDE...
  #define CSP_OVERRIDE CSP_UNSET
#endif

#ifndef BUFFER_COLOR_SPACE
  #define BUFFER_COLOR_SPACE ORG_BUFFER_COLOR_SPACE
#endif

#if ((BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SCRGB_BIT_DEPTH))  \
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
    || (                                  CSP_OVERRIDE == CSP_UNSET && defined(IS_POSSIBLE_SRGB_BIT_DEPTH))  \
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

  #define IS_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define IS_FLOAT_HDR_CSP
#endif

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  #define IS_HDR10_LIKE_CSP
#endif

#define GAMMA_UNSET 0
#define GAMMA_SRGB  1
#define GAMMA_22    2
#define GAMMA_24    3

#ifndef IS_HDR_CSP
  #ifndef IS_POSSIBLE_SCRGB_BIT_DEPTH
    #ifndef OVERWRITE_SDR_GAMMA
      #define OVERWRITE_SDR_GAMMA GAMMA_UNSET
    #endif
  #endif
#endif

#define CSP_SRGB_DEFAULT_TEXT "sRGB (gamma 2.2 or sRGB transfer function + BT.709 primaries)"
#define CSP_GAMMA22_TEXT      "Gamma 2.2 (2.2 power gamma + BT.709 primaries)"
#define CSP_GAMMA24_TEXT      "Gamma 2.4 (2.4 power gamma + BT.709 primaries)"
#define CSP_SRGB_TEXT         "sRGB (sRGB transfer function + BT.709 primaries)"
#define CSP_SCRGB_TEXT        "scRGB (linear + BT.709 primaries)"
#define CSP_HDR10_TEXT        "HDR10 (PQ + BT.2020 primaries)"
#define CSP_HLG_TEXT          "HLG (HLG + BT.2020 primaries)"

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


#define CSP_UNSET_TEXT "colour space unset! could be: "

#if (BUFFER_COLOR_SPACE == CSP_SCRGB)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_SCRGB_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_HDR10)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_HDR10_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_HLG)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_HLG_TEXT
#elif (BUFFER_COLOR_SPACE == CSP_UNKNOWN)
  #if defined(IS_POSSIBLE_SCRGB_BIT_DEPTH)
    #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_SCRGB_TEXT
  #elif defined(IS_POSSIBLE_HDR10_BIT_DEPTH)
    #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_HDR10_TEXT
  #elif defined(IS_POSSIBLE_SRGB_BIT_DEPTH)
    #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_UNSET_TEXT CSP_SRGB_DEFAULT_TEXT
  #else
    #define BACK_BUFFER_COLOUR_SPACE_TEXT GET_UNKNOWN_NUMBER(BUFFER_COLOR_SPACE)
  #endif
#elif (BUFFER_COLOR_SPACE == CSP_SRGB)
  #define BACK_BUFFER_COLOUR_SPACE_TEXT CSP_SRGB_DEFAULT_TEXT
#else
  #define BACK_BUFFER_COLOUR_SPACE_TEXT GET_UNKNOWN_NUMBER(BUFFER_COLOR_SPACE)
#endif


#if (CSP_OVERRIDE == CSP_SCRGB)
  #define CSP_OVERRIDE_TEXT CSP_SCRGB_TEXT
#elif (CSP_OVERRIDE == CSP_HDR10)
  #define CSP_OVERRIDE_TEXT CSP_HDR10_TEXT
#elif (CSP_OVERRIDE == CSP_HLG)
  #define CSP_OVERRIDE_TEXT CSP_HLG_TEXT
#elif ((CSP_OVERRIDE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SRGB) \
    && OVERWRITE_SDR_GAMMA != GAMMA_UNSET)
  #if (OVERWRITE_SDR_GAMMA == GAMMA_24)
    #define CSP_OVERRIDE_TEXT CSP_GAMMA24_TEXT
  #elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)
    #define CSP_OVERRIDE_TEXT CSP_SRGB_TEXT
  #else
    #define CSP_OVERRIDE_TEXT CSP_GAMMA22_TEXT
  #endif
#else
  #define CSP_OVERRIDE_TEXT "unset"
#endif


#if (ACTUAL_COLOUR_SPACE == CSP_SRGB)
  #if (OVERWRITE_SDR_GAMMA == GAMMA_24)
    #define ACTUAL_CSP_TEXT CSP_GAMMA24_TEXT
  #elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)
    #define ACTUAL_CSP_TEXT CSP_SRGB_TEXT
  #else
    #define ACTUAL_CSP_TEXT CSP_GAMMA22_TEXT
  #endif
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
       "detected back buffer format:       " BACK_BUFFER_FORMAT_TEXT       \
  "\n" "detected back buffer color space:  " BACK_BUFFER_COLOUR_SPACE_TEXT \
  "\n" "colour space overwritten to:       " CSP_OVERRIDE_TEXT             \
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

#if ((HIDE_CSP_OVERRIDE_EXPLANATION == YES) \
  || defined(IS_HDR_CSP)                    \
  || (BUFFER_COLOR_BIT_DEPTH <= 8))
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


#if (defined(API_IS_D3D11)   \
  || defined(API_IS_D3D12)   \
  || defined(API_IS_OPENGL)  \
  || defined(API_IS_VULKAN))

  #define IS_COMPUTE_CAPABLE_API

#endif

#if (defined(API_IS_D3D10)   \
  || defined(API_IS_D3D11)   \
  || defined(API_IS_D3D12)   \
  || defined(API_IS_OPENGL)  \
  || defined(API_IS_VULKAN))

  #define IS_ANALYSIS_CAPABLE_API

#endif


#define ERROR_TEXT "Only HDR colour spaces are supported!"
#define ERROR_TEXT_1 "Only DirectX 11, 12, OpenGL and Vulkan are supported!"
#define ERROR_TEXT_2 "Only DirectX 10, 11, 12, OpenGL and Vulkan are supported!"


#define ERROR_STUFF                   \
  uniform int ERROR_MESSAGE           \
  <                                   \
    ui_category = "ERROR";            \
    ui_label    = " ";                \
    ui_type     = "radio";            \
    ui_text     = ERROR_TEXT;         \
  >;                                  \
                                      \
  void VS_Error                       \
  (                                   \
    out float4 Position : SV_Position \
  )                                   \
  {                                   \
    Position = -2.f;                  \
    return;                           \
  }                                   \
  void PS_Error()                     \
  {                                   \
    discard;                          \
  }

#define VS_ERROR                     \
  {                                  \
    pass Error                       \
    {                                \
      VertexShader      = VS_Error;  \
      PixelShader       = PS_Error;  \
      PrimitiveTopology = POINTLIST; \
      VertexCount       = 1;         \
    }                                \
  }


#if (__RESHADE_PERFORMANCE_MODE__ == 0)
  #define BRANCH()  [branch]
  #define FLATTEN() [flatten]
#else
  #define BRANCH()
  #define FLATTEN()
#endif


#define PI   3.1415927410125732421875f
#define PI_2 6.283185482025146484375f

#define _1_DIV_PI 0.3183098733425140380859375f

#define FP32_MIN asfloat(0x00800000)
#define FP32_MAX asfloat(0x7F7FFFFF)

#define UINT_MAX 4294967295u
#define  INT_MAX 2147483647

#define MIN3(A, B, C) min(A, min(B, C))

#define MIN4(A, B, C, D) min(A, min(B, max(C, D)))


#define MAX3(A, B, C) max(A, max(B, C))

#define MAX4(A, B, C, D) max(A, max(B, max(C, D)))

#define MAX5(A, B, C, D, E) max(A, max(B, max(C, max(D, E))))

#define MAX6(A, B, C, D, E, F) max(A, max(B, max(C, max(D, max(E, F)))))

#define MAX7(A, B, C, D, E, F, G) max(A, max(B, max(C, max(D, max(E, max(F, G))))))

#define MAX8(A, B, C, D, E, F, G, H) max(A, max(B, max(C, max(D, max(E, max(F, max(G, H)))))))

#define MAXRGB(Rgb) max(Rgb.r, max(Rgb.g, Rgb.b))
#define MINRGB(Rgb) min(Rgb.r, min(Rgb.g, Rgb.b))


struct s_xyY
{
  float2 xy;
  float  Y;
};

struct s_uvY
{
  float2 uv;
  float  Y;
};


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
      #define SRGB_TO_LINEAR(T)                                      \
        T Linear(T C)                                                \
        {                                                            \
          return C <= 0.04045f ? C / 12.92f                          \
                               : pow(((C + 0.055f) / 1.055f), 2.4f); \
        }

      SRGB_TO_LINEAR(float)
      SRGB_TO_LINEAR(float3)
    } //SrgbTo


    namespace LinearTo
    {
      #define LINEAR_TO_SRGB(T)                                          \
        T Srgb(T C)                                                      \
        {                                                                \
          return C <= 0.0031308f ? C * 12.92f                            \
                                 : 1.055f * pow(C, 1.f / 2.4f) - 0.055f; \
        }

      LINEAR_TO_SRGB(float)
      LINEAR_TO_SRGB(float3)
    } //LinearTo


    namespace ExtendedSrgbSCurveTo
    {
      //#define X_sRGB_1 1.19417654368084505707
      //#define X_sRGB_x 0.039815307380813555
      //#define X_sRGB_y_adjust 1.21290538811
      // extended sRGB gamma including above 1 and below -1
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC > 1.f)
        {
          return signC * ((1.055f * pow(absC - 0.940277040004730224609375f, (1.f / 2.4f)) - 0.055f) + 0.728929579257965087890625f);
        }
        else
        [branch]
        if (absC > 0.04045f)
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
        return float3(Csp::Trc::ExtendedSrgbSCurveTo::Linear(Colour.r),
                      Csp::Trc::ExtendedSrgbSCurveTo::Linear(Colour.g),
                      Csp::Trc::ExtendedSrgbSCurveTo::Linear(Colour.b));
      }
    } //ExtendedSrgbSCurveTo


    namespace ExtendedSrgbLinearTo
    {
      //#define X_sRGB_1 1.19417654368084505707
      //#define X_sRGB_x 0.039815307380813555
      //#define X_sRGB_y_adjust 1.21290538811
      // extended sRGB gamma including above 1 and below -1
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC > 1.f)
        {
          return C;
        }
        else
        [branch]
        if (absC > 0.04045f)
        {
          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
        }
        else
        {
          return C / 12.92f;
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedSrgbLinearTo::Linear(Colour.r),
                      Csp::Trc::ExtendedSrgbLinearTo::Linear(Colour.g),
                      Csp::Trc::ExtendedSrgbLinearTo::Linear(Colour.b));
      }
    } //ExtendedSrgbLinearTo


// DO NOT USE!!!
// it does not match the ExtendedSrgbSCurveToLinear version!
//
//    namespace LinearTo
//    {
//      float ExtendedSrgbSCurve(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * pow((absC - 0.728929579257965087890625f + 0.055f) / 1.055f, 2.4f) + 0.940277040004730224609375f;
//        }
//        else if (absC > 0.0031308f)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else
//        {
//          return C * 12.92f;
//        }
//      }
//
//      float3 ExtendedSrgbSCurve(float3 Colour)
//      {
//        return float3(Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.r),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.g),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.b));
//      }
//    }


    namespace SrgbAccurateTo
    {
      // accurate sRGB with no slope discontinuity
      #define SrgbX       asfloat(0x3D20EA0B) //  0.0392857
      #define SrgbPhi     asfloat(0x414EC578) // 12.92321
      #define SrgbXDivPhi asfloat(0x3B4739A5) //  0.003039935

      float Linear(float C)
      {
        [branch]
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
        [branch]
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


//    namespace ExtendedSrgbSCurveAccurateTo
//    {
//      float Linear(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else if (absC > SrgbX)
//        {
//          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
//        }
//        else
//        {
//          return C / SrgbPhi;
//        }
//      }
//
//      float3 Linear(float3 Colour)
//      {
//        return float3(Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.r),
//                      Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.g),
//                      Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.b));
//      }
//    } //ExtendedSrgbSCurveAccurateTo
//
//
//    namespace LinearTo
//    {
//      float ExtendedSrgbSCurveAccurate(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
//        }
//        else if (absC > SrgbXDivPhi)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else
//        {
//          return C * SrgbPhi;
//        }
//      }
//
//      float3 ExtendedSrgbSCurveAccurate(float3 Colour)
//      {
//        return float3(Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.r),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.g),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.b));
//      }
//    } //LinearTo


    namespace ExtendedGamma22SCurveTo
    {
      //#define X_22_1 1.20237927370128566986
      //#define X_22_x 0.0370133892172524
      //#define X_22_y_adjust 1.5f - pow(X_22_x, Csp::Trc::ApplyGamma22)
      // extended gamma 2.2 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC <= 1.f)
        {
          return signC * pow(absC, 2.2f);
        }
        else
        {
          return signC * (pow(absC - 0.944479882717132568359375f, 1.f / 2.2f) + 0.731282770633697509765625f);
        }
      }
      //{
      //  if (C < -X_22_1)
      //    return
      //      -(pow(-C - X_22_1 + X_22_x, 1.f / 2.2f) + X_22_y_adjust);
      //  else if (C < 0)
      //    return
      //      -pow(-C, 2.2f);
      //  else if (C <= X_22_1)
      //    return
      //      pow(C, 2.2f);
      //  else
      //    return
      //      (pow(C - X_22_1 + X_22_x, Csp::Trc::ApplyGamma22) + X_22_y_adjust);
      //}

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma22SCurveTo::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma22SCurveTo::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma22SCurveTo::Linear(Colour.b));
      }
    } //ExtendedGamma22SCurveTo


    namespace ExtendedGamma22LinearTo
    {
      //#define X_22_1 1.20237927370128566986
      //#define X_22_x 0.0370133892172524
      //#define X_22_y_adjust 1.5f - pow(X_22_x, Csp::Trc::ApplyGamma22)
      // extended gamma 2.2 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC <= 1.f)
        {
          return signC * pow(absC, 2.2f);
        }
        else
        {
          return C;
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma22LinearTo::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma22LinearTo::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma22LinearTo::Linear(Colour.b));
      }
    } //ExtendedGamma22LinearTo


    namespace ExtendedGamma24SCurveTo
    {
      //#define X_24_1 1.1840535873752085849
      //#define X_24_x 0.033138075
      //#define X_24_y_adjust 1.5f - pow(X_24_x, 1.f / 2.4f)
      // extended gamma 2.4 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC <= 1.f)
        {
          return signC * pow(absC, 2.4f);
        }
        else
        {
          return signC * (pow(absC - 0.950292885303497314453125f, 1.f / 2.4f) + 0.71368694305419921875f);
        }
      }
      //{
      //  if (C < -X_24_1)
      //    return
      //      -(pow(-C - X_24_1 + X_24_x, 1.f / 2.4f) + X_24_y_adjust);
      //  else if (C < 0)
      //    return
      //      -pow(-C, 2.4f);
      //  else if (C <= X_24_1)
      //    return
      //      pow(C, 2.4f);
      //  else
      //    return
      //      (pow(C - X_24_1 + X_24_x, 1.f / 2.4f) + X_24_y_adjust);
      //}

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma24SCurveTo::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma24SCurveTo::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma24SCurveTo::Linear(Colour.b));
      }
    } //ExtendedGamma24SCurveTo


    namespace ExtendedGamma24LinearTo
    {
      //#define X_24_1 1.1840535873752085849
      //#define X_24_x 0.033138075
      //#define X_24_y_adjust 1.5f - pow(X_24_x, 1.f / 2.4f)
      // extended gamma 2.4 including above 1 and below 0
      float Linear(float C)
      {
        static const float absC  = abs(C);
        static const float signC = sign(C);

        [branch]
        if (absC <= 1.f)
        {
          return signC * pow(absC, 2.4f);
        }
        else
        {
          return C;
        }
      }

      float3 Linear(float3 Colour)
      {
        return float3(Csp::Trc::ExtendedGamma24LinearTo::Linear(Colour.r),
                      Csp::Trc::ExtendedGamma24LinearTo::Linear(Colour.g),
                      Csp::Trc::ExtendedGamma24LinearTo::Linear(Colour.b));
      }
    } //ExtendedGamma24LinearTo


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

      [branch]
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
    static const float PQ_m1 =  0.1593017578125f; // = 1305 / 8192;
    static const float PQ_m2 = 78.84375f;         // = 2523 /   32;
    static const float PQ_c1 =  0.8359375f;       // =  107 /  128;
    static const float PQ_c2 = 18.8515625f;       // = 2413 /  128;
    static const float PQ_c3 = 18.6875f;          // = 2392 /  128;

    static const float _1_div_PQ_m1 = 6.27739477f;
    static const float _1_div_PQ_m2 = 0.0126833133f;


    // Rec. ITU-R BT.2100-2 Table 4
    namespace PqTo
    {

      #define PQ_TO_LINEAR(T)                       \
        T Linear(T E_)                              \
        {                                           \
          E_ = max(E_, 0.f);                        \
                                                    \
          T E_pow_1_div_m2 = pow(E_, _1_div_PQ_m2); \
                                                    \
          T num = max(E_pow_1_div_m2 - PQ_c1, 0.f); \
                                                    \
          T den = PQ_c2 - PQ_c3 * E_pow_1_div_m2;   \
                                                    \
          /* Y */                                   \
          return pow(num                            \
                   / den                            \
                 , _1_div_PQ_m1);                   \
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
      #define PQ_TO_NITS(T)                 \
        T Nits(T E_)                        \
        {                                   \
          return Csp::Trc::PqTo::Linear(E_) \
               * 10000.f;                   \
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
      #define LINEAR_TO_PQ(T)               \
        T Pq(T Y)                           \
        {                                   \
          Y = max(Y, 0.f);                  \
                                            \
          T Y_pow_m1 = pow(Y, PQ_m1);       \
                                            \
          T num = PQ_c1 + PQ_c2 * Y_pow_m1; \
                                            \
          T den =   1.f + PQ_c3 * Y_pow_m1; \
                                            \
          /* E' */                          \
          return pow(num                    \
                   / den                    \
                 , PQ_m2);                  \
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


    static const float HLG_a = 0.17883277;
    static const float HLG_b = 0.28466892; // = 1 - 4 * HLG_a
    static const float HLG_c = 0.55991072952956202016; //0.55991072952956202016 = 0.5 - HLG_a * ln(4 * HLG_a)

    // Rec. ITU-R BT.2100-2 Table 5
    namespace HlgTo
    {
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


  namespace CieXYZ
  {

    namespace XYZTo
    {
      s_xyY xyY(const float3 XYZ)
      {
        const float xyz = XYZ.x + XYZ.y + XYZ.z;

        s_xyY xyY;

        // for pure black (RGB(0,0,0) = XYZ(0,0,0)) there is a division by 0
        xyY.xy = xyz != 0.f ? XYZ.xy / xyz
                            : 0.f;

        xyY.Y = XYZ.y;

        return xyY;
      }

      s_uvY uvY(const float3 XYZ)
      {
        const float X15Y3Z = XYZ.x
                           + 15.f * XYZ.y
                           +  3.f * XYZ.z;

        s_uvY uvY;

        // for pure black (RGB(0,0,0) = XYZ(0,0,0)) there is a division by 0
        uvY.uv = X15Y3Z != 0.f ? float2(4.f, 9.f) * XYZ.xy / X15Y3Z
                               : 0.f;

        uvY.Y = XYZ.y;

        return uvY;
      }

      float2 xy(const float3 XYZ)
      {
        const float xyz = XYZ.x + XYZ.y + XYZ.z;

        // for pure black (RGB(0,0,0) = XYZ(0,0,0)) there is a division by 0
        float2 xy = xyz != 0.f ? XYZ.xy / xyz
                               : 0.f;

        return xy;
      }

      float2 uv(const float3 XYZ)
      {
        const float X15Y3Z = XYZ.x
                           + 15.f * XYZ.y
                           +  3.f * XYZ.z;

        // for pure black (RGB(0,0,0) = XYZ(0,0,0)) there is a division by 0
        float2 uv = X15Y3Z != 0.f ? float2(4.f, 9.f) * XYZ.xy / X15Y3Z
                                  : 0.f;

        return uv;
      }
    } //XYZTo

    namespace xyYTo
    {
      float3 XYZ(const s_xyY xyY)
      {
        float3 XYZ;

        XYZ.xz = float2(xyY.xy.x, (1.f - xyY.xy.x - xyY.xy.y))
               / xyY.xy.y
               * xyY.Y;

        XYZ.y = xyY.Y;

        return XYZ;
      }
    } //xyYTo

    namespace uvYTo
    {
      float3 XYZ(const s_uvY uvY)
      {
        float3 XYZ;

        XYZ.x = 9.f * uvY.uv[0];

        XYZ.z = 12.f
              - (3.f * uvY.uv[0])
              - (20.f * uvY.uv[1]);

        XYZ.xz = XYZ.xz
               / (4.f * uvY.uv[1])
               * uvY.Y;

        XYZ.y = uvY.Y;

        return XYZ;
      }
    } //uvYTo

    namespace xyTo
    {
      float3 XYZ(const float2 xy)
      {
        float3 XYZ;

        XYZ.xz = float2(xy.x, (1.f - xy.x - xy.y))
               / xy.y;

        XYZ.y = 1.f;

        return XYZ;
      }

      float2 uv(const float2 xy)
      {
        float m2x12y3 = -2.f * xy.x
                      + 12.f * xy.y
                      +  3.f;

        float2 uv = float2(4.f, 9.f)
                  * xy
                  / m2x12y3;

        return uv;
      }
    } //xyTo

    namespace uvTo
    {
      float3 XYZ(const float2 uv)
      {
        float3 XYZ;

        XYZ.x = 9.f * uv[0];

        XYZ.z = 12.f
              - ( 3.f * uv[0])
              - (20.f * uv[1]);

        XYZ.xz /= 4.f * uv[1];

        XYZ.y = 1.f;

        return XYZ;
      }
    } //uvTo

  }


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

      static const float3 KBt709  = float3(0.212639003f,
                                           0.715168654f,
                                           0.0721923187f);
      static const float  KbBt709 = 1.85561537f;
      static const float  KrBt709 = 1.57472193f;
      static const float2 KgBt709 = float2(0.187314093f,
                                           0.468207478f);

      static const float3 KBt2020  = float3(0.262700200f,
                                            0.677998065f,
                                            0.0593017153f);
      static const float  KbBt2020 = 1.88139653f;
      static const float  KrBt2020 = 1.47459959f;
      static const float2 KgBt2020 = float2(0.164558053f,
                                            0.571355044f);

    namespace YcbcrTo
    {

      float3 RgbBt709(float3 Colour)
      {
        return float3(
          Colour.x + KrBt709    * Colour.z,
          Colour.x - KgBt709[0] * Colour.y - KgBt709[1] * Colour.z,
          Colour.x + KbBt709    * Colour.y);
      }

      float3 RgbBt2020(float3 Colour)
      {
        return float3(
          Colour.x + KrBt2020    * Colour.z,
          Colour.x - KgBt2020[0] * Colour.y - KgBt2020[1] * Colour.z,
          Colour.x + KbBt2020    * Colour.y);
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

    } //RgbTo

  } //Ycbcr


  namespace Mat
  {

    //BT.709 To
    static const float3x3 Bt709ToXYZ =
      float3x3
      (
        0.412390798f,  0.357584327f, 0.180480793f,
        0.212639003f,  0.715168654f, 0.0721923187f,
        0.0193308182f, 0.119194783f, 0.950532138f
      );

    static const float3x3 Bt709ToDciP3 =
      float3x3
      (
        0.822461962f,  0.177538037f,  0.f,
        0.0331941992f, 0.966805815f,  0.f,
        0.0170826315f, 0.0723974406f, 0.910519957f
      );

    static const float3x3 Bt709ToBt2020 =
      float3x3
      (
        0.627403914f,  0.329283028f,  0.0433130674f,
        0.0690972879f, 0.919540405f,  0.0113623151f,
        0.0163914393f, 0.0880133062f, 0.895595252f
      );

    static const float3x3 Bt709ToAp1D65 =
      float3x3
      (
        0.617028832f,  0.333867609f, 0.0491035431f,
        0.0699223205f, 0.917349696f, 0.0127279674f,
        0.0205497872f, 0.107552029f, 0.871898174f
      );

    static const float3x3 Bt709ToAp0D65 =
      float3x3
      (
        0.433931618f,  0.376252382f, 0.189815968f,
        0.0886183902f, 0.809275329f, 0.102106288f,
        0.0177500396f, 0.109447620f, 0.872802317f
      );

    static const float3x3 Bt709ToAp1D60 =
      float3x3
      (
        0.613097429f,  0.339523136f, 0.0473794527f,
        0.0701937228f, 0.916353881f, 0.0134523985f,
        0.0206155925f, 0.109569773f, 0.869814634f
      );

    static const float3x3 Bt709ToAp0D60 =
      float3x3
      (
        0.439632982f,  0.382988691f, 0.177378326f,
        0.0897764414f, 0.813439428f, 0.0967841297f,
        0.0175411701f, 0.111546553f, 0.870912253f
      );


    //scRGB To
    static const float3x3 ScRgbToXYZ =
      float3x3
      (
        0.00329912640f,  0.00286067463f,  0.00144384626f,
        0.00170111202f,  0.00572134926f,  0.000577538507f,
        0.000154646550f, 0.000953558250f, 0.00760425720f
      );

    static const float3x3 ScRgbToBt2020Normalised =
      float3x3
      (
        0.00501923123f,  0.00263426429f,  0.000346504530f,
        0.000552778306f, 0.00735632330f,  0.0000908985239f,
        0.000131131513f, 0.000704106467f, 0.00716476188f
      );


    //DCI-P3 To
    static const float3x3 DciP3ToXYZ =
      float3x3
      (
        0.486570954f, 0.265667706f,  0.198217287f,
        0.228974565f, 0.691738545f,  0.0792869105f,
        0.f,          0.0451133809f, 1.04394435f
      );

    static const float3x3 DciP3ToBt709 =
      float3x3
      (
         1.22494018f,   -0.224940180f,  0.f,
        -0.0420569553f,  1.04205691f,   0.f,
        -0.0196375548f, -0.0786360427f, 1.09827363f
      );

    static const float3x3 DciP3ToBt2020 =
      float3x3
      (
         0.753833055f,   0.198597371f,  0.0475695952f,
         0.0457438491f,  0.941777229f,  0.0124789308f,
        -0.00121034029f, 0.0176017172f, 0.983608603f
      );


    //DCI-P3 80 To
    static const float3x3 DciP3_80ToXYZ =
      float3x3
      (
        0.00389256747f, 0.00212534144f,  0.00158573826f,
        0.00183179648f, 0.00553390802f,  0.000634295283f,
        0.f,            0.000360907055f, 0.00835155509f
      );

    static const float3x3 DciP3_80ToBt2020Normalised =
      float3x3
      (
         0.00603066431f,    0.00158877891f,  0.000380556768f,
         0.000365950778f,   0.00753421755f,  0.0000998314499f,
        -0.00000968272252f, 0.000140813732f, 0.00786886923f
      );


    //BT.2020 To
    static const float3x3 Bt2020ToXYZ =
      float3x3
      (
        0.636958062f, 0.144616901f,  0.168880969f,
        0.262700200f, 0.677998065f,  0.0593017153f,
        0.f,          0.0280726924f, 1.06098508f
      );

    static const float3x3 Bt2020ToBt709 =
      float3x3
      (
         1.66049098f,   -0.587641119f, -0.0728498622f,
        -0.124550476f,   1.13289988f,  -0.00834942236f,
        -0.0181507635f, -0.100578896f,  1.11872971f
      );

    static const float3x3 Bt2020ToDciP3 =
      float3x3
      (
         1.34357821f,    -0.282179683f,  -0.0613985806f,
        -0.0652974545f,   1.07578790f,   -0.0104904631f,
         0.00282178726f, -0.0195984952f,  1.01677668f
      );

    static const float3x3 Bt2020ToAp1D65 =
      float3x3
      (
        0.982096254f,   0.0107082454f, 0.00719551974f,
        0.00161802524f, 0.996895968f,  0.00148598209f,
        0.00490146316f, 0.0220752228f, 0.973023295f
      );

    static const float3x3 Bt2020ToAp0D65 =
      float3x3
      (
        0.670231819f,  0.152168750f,  0.177599415f,
        0.0445011146f, 0.854482352f,  0.101016514f,
        0.f,           0.0257770475f, 0.974222958f
      );

    static const float3x3 Bt2020ToAp1D60 =
      float3x3
      (
        0.974895000f,   0.0195991080f, 0.00550591340f,
        0.00217956281f, 0.995535492f,  0.00228496827f,
        0.00479723978f, 0.0245320163f, 0.970670759f
      );

    static const float3x3 Bt2020ToAp0D60 =
      float3x3
      (
         0.679085612f,    0.157700911f,  0.163213446f,
         0.0460020042f,   0.859054684f,  0.0949433222f,
        -0.000573943194f, 0.0284677688f, 0.972106158f
      );


    //BT.2020 normalised To
    static const float3x3 Bt2020NormalisedToScRgb =
      float3x3
      (
         207.561370f, -73.4551391f, -9.10623264f,
        -15.5688095f,  141.612487f, -1.04367780f,
        -2.26884531f, -12.5723619f,  139.841201f
      );


    //BT.2020 80 To
    static const float3x3 Bt2020_80ToXYZ =
      float3x3
      (
        0.00509566441f, 0.00115693523f,  0.00135104777f,
        0.00210160179f, 0.00542398449f,  0.000474413740f,
        0.f,            0.000224581541f, 0.00848788022f
      );


    //AP1 D65 To
    static const float3x3 Ap1D65ToXYZ =
      float3x3
      (
         0.647507190f,   0.134379133f,   0.168569594f,
         0.266086399f,   0.675967812f,   0.0579457953f,
        -0.00544886849f, 0.00407209526f, 1.09043455f
      );


    //AP0 D65 To
    static const float3x3 Ap0D65ToXYZ =
      float3x3
      (
        0.950354814f, 0.f,           0.000101128956f,
        0.343172907f, 0.734696388f, -0.0778692960f,
        0.f,          0.f,           1.08905780f
      );

    static const float3x3 Ap0D65ToBt709 =
      float3x3
      (
         2.55248308f,   -1.12950992f,  -0.422973215f,
        -0.277344137f,   1.37826657f,  -0.100922435f,
        -0.0171310510f, -0.149861142f,  1.16699218f
      );

    static const float3x3 Ap0D65ToBt2020 =
      float3x3
      (
         1.50937116f,    -0.261310040f,  -0.248061075f,
        -0.0788541212f,   1.18762290f,   -0.108768820f,
         0.00208640797f, -0.0314234159f,  1.02933704f
      );


    //AP1 D60 To
    static const float3x3 Ap1D60ToXYZ =
      float3x3
      (
         0.662454187f,   0.134004205f,   0.156187683f,
         0.272228717f,   0.674081742f,   0.0536895170f,
        -0.00557464966f, 0.00406073359f, 1.01033914f
      );

    static const float3x3 Ap1D60ToBt709 =
      float3x3
      (
         1.70505094f,   -0.621792137f, -0.0832588747f,
        -0.130256414f,   1.14080476f,  -0.0105483187f,
        -0.0240033566f, -0.128968968f,  1.15297234f
      );

    static const float3x3 Ap1D60ToBt2020 =
      float3x3
      (
         1.02582478f,    -0.0200531911f, -0.00577155686f,
        -0.00223436951f,  1.00458645f,   -0.00235213246f,
        -0.00501335132f, -0.0252900719f,  1.03030347f
      );

    static const float3x3 Ap1D60ToAp0D60 =
      float3x3
      (
         0.695452213f,   0.140678703f,   0.163869068f,
         0.0447945632f,  0.859671115f,   0.0955343171f,
        -0.00552588235f, 0.00402521016f, 1.00150072f
      );


    //AP0 D60 To
    static const float3x3 Ap0D60ToXYZ =
      float3x3
      (
        0.952552378f, 0.f,           0.0000936786309f,
        0.343966454f, 0.728166103f, -0.0721325427f,
        0.f,          0.f,           1.00882518f
      );

    static const float3x3 Ap0D60ToBt709 =
      float3x3
      (
         2.52168607f,   -1.13413095f,  -0.387555211f,
        -0.276479899f,   1.37271904f,  -0.0962391719f,
        -0.0153780654f, -0.152975335f,  1.16835343f
      );

    static const float3x3 Ap0D60ToBt2020 =
      float3x3
      (
         1.49040949f,    -0.266170918f,  -0.224238604f,
        -0.0801675021f,   1.18216717f,   -0.101999618f,
         0.00322763109f, -0.0347764752f,  1.03154885f
      );

    static const float3x3 Ap0D60ToAp1D60 =
      float3x3
      (
         1.45143926f,    -0.236510753f,   -0.214928567f,
        -0.0765537768f,   1.17622971f,    -0.0996759235f,
         0.00831614807f, -0.00603244965f,  0.997716307f
      );


    //XYZ To
    static const float3x3 XYZToBt709 =
      float3x3
      (
         3.24096989f,   -1.53738319f,  -0.498610764f,
        -0.969243645f,   1.87596750f,   0.0415550582f,
         0.0556300804f, -0.203976958f,  1.05697154f
      );

    static const float3x3 XYZToScRgb =
      float3x3
      (
         0.0259277597f,   -0.0122990654f,  -0.00398888625f,
        -0.00775394914f,   0.0150077398f,   0.000332440453f,
         0.000445040641f, -0.00163181568f,  0.00845577195f
      );

    static const float3x3 XYZToDciP3 =
      float3x3
      (
         2.49349689f,   -0.931383609f,  -0.402710795f,
        -0.829488992f,   1.76266407f,    0.0236246865f,
         0.0358458310f, -0.0761723890f,  0.956884503f
      );

    static const float3x3 XYZToBt2020 =
      float3x3
      (
         1.71665120f,   -0.355670779f,  -0.253366291f,
        -0.666684329f,   1.61648118f,    0.0157685466f,
         0.0176398567f, -0.0427706129f,  0.942103147f
      );

    static const float3x3 XYZToAp1D65 =
      float3x3
      (
         1.67890453f,   -0.332301020f,   -0.241882294f,
        -0.661811172f,   1.61082458f,     0.0167095959f,
         0.0108608892f, -0.00767592666f,  0.915794551f
      );

    static const float3x3 XYZToAp0D65 =
      float3x3
      (
         1.05223858f,  0.f,         -0.0000977099625f,
        -0.491495221f, 1.36110639f,  0.0973668321f,
         0.f,          0.f,          0.918224930f
      );


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

      float3 Ap1D60(float3 Colour)
      {
        return mul(Bt709ToAp1D60, Colour);
      }

      float3 Ap0D60(float3 Colour)
      {
        return mul(Bt709ToAp0D60, Colour);
      }
    } //Bt709To

    namespace ScRgbTo
    {
      float3 Bt2020Normalised(float3 Colour)
      {
        return mul(ScRgbToBt2020Normalised, Colour);
      }
    } //ScRgbTo

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

    namespace DciP3_80To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(DciP3_80ToXYZ, Colour);
      }

      float3 Bt2020Normalised(float3 Colour)
      {
        return mul(DciP3_80ToBt2020Normalised, Colour);
      }
    } //DciP3_80To

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

      float3 Ap1D60(float3 Colour)
      {
        return mul(Bt2020ToAp1D60, Colour);
      }

      float3 Ap0D60(float3 Colour)
      {
        return mul(Bt2020ToAp0D60, Colour);
      }
    } //Bt2020To

    namespace Bt2020NormalisedTo
    {
      float3 ScRgb(float3 Colour)
      {
        return mul(Bt2020NormalisedToScRgb, Colour);
      }
    } //Bt2020NormalisedTo

    namespace Bt2020_80To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Bt2020_80ToXYZ, Colour);
      }
    } //Bt2020NormalisedTo

    namespace Ap1D65To
    {
      float3 XYZ(float3 Colour)
      {
        return mul(Ap1D65ToXYZ, Colour);
      }
    } //Ap1D65To

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

    namespace Ap1D60To
    {
      float3 Bt709(float3 Colour)
      {
        return mul(Ap1D60ToBt709, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(Ap1D60ToBt2020, Colour);
      }

      float3 Ap0D60(float3 Colour)
      {
        return mul(Ap1D60ToAp0D60, Colour);
      }
    } //Ap1D60To

    namespace Ap0D60To
    {
      float3 Bt709(float3 Colour)
      {
        return mul(Ap0D60ToBt709, Colour);
      }

      float3 Bt2020(float3 Colour)
      {
        return mul(Ap0D60ToBt2020, Colour);
      }

      float3 Ap1D60(float3 Colour)
      {
        return mul(Ap0D60ToAp1D60, Colour);
      }
    } //Ap0D60To

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

    // The matrices use higher precision rather than being rounded to 12 bit values like the Dolby spec describes.
    // This is because I only use these for internal processing.

    //L'M'S'->ICtCp
    static const float3x3 PqLmsToIctcp =
      float3x3
      (
        0.5f,         0.5f,         0.f,
        1.61370003f, -3.32339620f,  1.70969617f,
        4.37806224f, -4.24553966f, -0.132522642f
      );

    //ICtCp->L'M'S'
    static const float3x3 IctcpToPqLms =
      float3x3
      (
        1.f,  0.00860647484f,  0.111033529f,
        1.f, -0.00860647484f, -0.111033529f,
        1.f,  0.560046315f,   -0.320631951f
      );


    //RGB BT.709->LMS
    static const float3x3 Bt709ToLms =
      float3x3
      (
        0.295764088f,  0.623072445f, 0.0811667516f,
        0.156191974f,  0.727251648f, 0.116557933f,
        0.0351022854f, 0.156589955f, 0.808302998f
      );

    //scRGB->LMS
    static const float3x3 ScRgbToLms =
      float3x3
      (
        0.00236611254f,  0.00498457951f, 0.000649333989f,
        0.00124953582f,  0.00581801310f, 0.000932463502f,
        0.000280818290f, 0.00125271955f, 0.00646642409f
      );

    //RGB DCI-P3->LMS
    static const float3x3 DciP3ToLms =
      float3x3
      (
        0.334494858f,  0.576365113f,  0.0891432985f,
        0.158450931f,  0.713538110f,  0.128012508f,
        0.0205394085f, 0.0917179808f, 0.887737870f
      );

    //RGB BT.2020->LMS
    static const float3x3 Bt2020ToLms =
      float3x3
      (
        0.412036389f,  0.523911893f,  0.0640549808f,
        0.166660219f,  0.720395207f,  0.112946122f,
        0.0241123586f, 0.0754749625f, 0.900407910f
      );


    //LMS->RGB BT.709
    static const float3x3 LmsToBt709 =
      float3x3
      (
         6.17353248f,   -5.32089900f,   0.147354885f,
        -1.32403194f,    2.56026983f,  -0.236238613f,
        -0.0115983877f, -0.264921456f,  1.27652633f
      );

    //LMS->scRGB
    static const float3x3 LmsToScRgb =
      float3x3
      (
         771.691589f,   -665.112365f,    18.4193611f,
        -165.503982f,    320.033721f,   -29.5298271f,
          -1.44979846f,  -33.1151809f,  159.565795f
      );

    //LMS->RGB DCI-P3
    static const float3x3 LmsToDciP3 =
      float3x3
      (
         4.84242963f,     -3.92169165f,   0.0792524516f,
        -1.07515621f,      2.29866075f,  -0.223505541f,
        -0.000956906354f, -0.146754235f,  1.14771676f
      );

    //LMS->RGB BT.2020
    static const float3x3 LmsToBt2020 =
      float3x3
      (
         3.43681478f,   -2.50677371f,    0.0699519291f,
        -0.791058242f,   1.98360168f,   -0.192544832f,
        -0.0257268063f, -0.0991417691f,  1.12487411f
      );

    namespace IctcpTo
    {
      //ICtCp->L'M'S'
      float3 PqLms(float3 Ictcp)
      {
        return mul(IctcpToPqLms, Ictcp);
      }
    } //IctcpTo

    namespace PqLmsTo
    {
      //L'M'S'->ICtCp
      float3 Ictcp(float3 PqLms)
      {
        return mul(PqLmsToIctcp, PqLms);
      }

      //L'M'S'->LMS
      float3 Lms(float3 PqLms)
      {
        return Csp::Trc::PqTo::Linear(PqLms);
      }
    } //PqLmsTo

    namespace IctcpTo
    {
      //ICtCp->LMS
      float3 Lms(float3 Ictcp)
      {
        float3 pqLms = IctcpTo::PqLms(Ictcp);

        //LMS
        return PqLmsTo::Lms(pqLms);
      }
    } //IctcpTo

    namespace LmsTo
    {
      //LMS->L'M'S'
      float3 PqLms(float3 Lms)
      {
        return Csp::Trc::LinearTo::Pq(Lms);
      }

      //LMS->ICtCp
      float3 Ictcp(float3 Lms)
      {
        float3 pqLms = LmsTo::PqLms(Lms);

        //ICtCp
        return PqLmsTo::Ictcp(pqLms);
      }

      //LMS->RGB BT.709
      float3 Bt709(float3 Colour)
      {
        return mul(LmsToBt709, Colour);
      }

      //LMS->scRGB
      float3 ScRgb(float3 Colour)
      {
        return mul(LmsToScRgb, Colour);
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
    } //LmsTo

    namespace PqLmsTo
    {
      //L'M'S'->RGB BT.709
      float3 Bt709(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //BT.709
        return LmsTo::Bt709(lms);
      }

      //L'M'S'->scRGB
      float3 ScRgb(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //scRGB
        return LmsTo::ScRgb(lms);
      }

      //L'M'S'->RGB DCI-P3
      float3 DciP3(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //DCI-P3
        return LmsTo::DciP3(lms);
      }

      //L'M'S'->RGB BT.2020
      float3 Bt2020(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //BT.2020
        return LmsTo::Bt2020(lms);
      }
    } //PqLmsTo

    namespace IctcpTo
    {
      //ICtCp->RGB BT.709
      float3 Bt709(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.709
        return LmsTo::Bt709(lms);
      }

      //ICtCp->scRGB
      float3 ScRgb(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //scRGB
        return LmsTo::ScRgb(lms);
      }

      //ICtCp->RGB DCI-P3
      float3 DciP3(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.709
        return LmsTo::DciP3(lms);
      }

      //ICtCp->RGB BT.2020
      float3 Bt2020(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.2020
        return LmsTo::Bt2020(lms);
      }
    } //IctcpTo

    namespace Bt709To
    {
      //RGB BT.709->LMS
      float3 Lms(float3 Colour)
      {
        return mul(Bt709ToLms, Colour);
      }

      //RGB BT.709->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = Bt709To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB BT.709->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = Bt709To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //Bt709To

    namespace ScRgbTo
    {
      //scRGB->LMS
      float3 Lms(float3 Colour)
      {
        return mul(ScRgbToLms, Colour);
      }

      //scRGB->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = ScRgbTo::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //scRGB->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = ScRgbTo::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //ScRgbTo

    namespace DciP3To
    {
      //RGB DCI-P3->LMS
      float3 Lms(float3 Colour)
      {
        return mul(DciP3ToLms, Colour);
      }

      //RGB DCI-P3->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = DciP3To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB DCI-P3->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = DciP3To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //DciP3To

    namespace Bt2020To
    {
      //RGB BT.2020->LMS
      float3 Lms(float3 Colour)
      {
        return mul(Bt2020ToLms, Colour);
      }

      //RGB BT.2020->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = Bt2020To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB BT.2020->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = Bt2020To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //Bt2020To

  } //ICtCp


  namespace Jzazbz
  {

// https://doi.org/10.1364/OE.25.015131

    static const float b  =   1.15f;
    static const float g  =   0.66f;
    static const float p  = 134.034375f; // 1.7 * 2523 / 2^5 (2523 / 2^5 = 2523 / 4096 * 128, which is the PQ constant m2)
    static const float d  =  -0.56f;
    static const float d0 =   0.0000000000162954996f; // 1.6295499532821566 * 10^-11

    static const float bMinus1 =  0.15f;
    static const float gMinus1 = -0.34f;

    static const float dPlus1 = 0.44f;

    static const float _1_div_p = 0.00746077252f;

    static const float3x3 NonLinearXYAndLinearZToJzazbzLms =
    float3x3
    (
       0.41478972f, 0.579999f, 0.014648f,
      -0.20151f,    1.120649f, 0.0531008f,
      -0.0166008f,  0.2648f,   0.6684799f
    );

    static const float3x3 JzazbzLmsToNonLinearXYAndLinearZ =
    float3x3
    (
       1.92422640f,   -1.00479233f,   0.0376514047f,
       0.350316762f,   0.726481199f, -0.0653844252f,
      -0.0909828096f, -0.312728285f,  1.52276659f
    );

    static const float3x3 PqJzazbzLmsToIzazbz =
    float3x3
    (
      0.5f,       0.5f,       0.f,
      3.524f,    -4.066708f,  0.542708f,
      0.199076f,  1.096799f, -1.295875f
    );

    static const float3x3 IzazbzToPqJzazbzLms =
    float3x3
    (
      1.f,  0.138605043f,   0.0580473169f,
      1.f, -0.138605043f,  -0.0580473169f,
      1.f, -0.0960192456f, -0.811891913f
    );

    namespace XYZTo
    {
      //XYZ->X'Y'Z
      float3 NonLinearXYAndLinearZ(float3 XYZ)
      {
        float2 val0 = float2(b * XYZ.x,
                             g * XYZ.y);

        float2 val1 = float2(bMinus1 * XYZ.z,
                             gMinus1 * XYZ.x);

        return float3(val0 - val1, XYZ.z);
      }
    } //XYZTo

    namespace NonLinearXYAndLinearZTo
    {
      //X'Y'Z->XYZ
      float3 XYZ(float3 NonLinearXYAndLinearZ)
      {
        float X = (NonLinearXYAndLinearZ.x + (bMinus1 * NonLinearXYAndLinearZ.z))
                / b;

        float Y = (NonLinearXYAndLinearZ.y + (gMinus1 * X))
                / g;

        return float3(X, Y, NonLinearXYAndLinearZ.z);
      }

      //X'Y'Z->LMS (Jzazbz variant)
      float3 JzazbzLms(float3 NonLinearXYAndLinearZ)
      {
        return mul(NonLinearXYAndLinearZToJzazbzLms, NonLinearXYAndLinearZ);
      }
    } //NonLinearXYAndLinearZTo

    namespace JzazbzLmsTo
    {
      //LMS (Jzazbz variant)->X'Y'Z
      float3 NonLinearXYAndLinearZ(float3 JzazbzLms)
      {
        return mul(JzazbzLmsToNonLinearXYAndLinearZ, JzazbzLms);
      }

      //LMS (Jzazbz variant)->L'M'S' (Jzazbz variant)
      float3 PqJzazbzLms(float3 JzazbzLms)
      {
        float3 powJzazbzLms = pow(JzazbzLms, PQ_m1);

        float3 numerator    = PQ_c1 + PQ_c2 * powJzazbzLms;

        float3 denominator  = 1.f   + PQ_c3 * powJzazbzLms;

        return pow(numerator / denominator, p);
      }
    } //JzazbzLmsTo

    namespace PqJzazbzLmsTo
    {
      //L'M'S' (Jzazbz variant)->Izazbz
      float3 Izazbz(float3 PqJzazbzLms)
      {
        return mul(PqJzazbzLmsToIzazbz, PqJzazbzLms);
      }

      //L'M'S' (Jzazbz variant)->LMS (Jzazbz variant)
      float3 JzazbzLms(float3 PqJzazbzLms)
      {
        float3 powPqJzazbzLms = pow(PqJzazbzLms, _1_div_p);

        float3 numerator      = PQ_c1 - powPqJzazbzLms;

        float3 denominator    = PQ_c3 * powPqJzazbzLms - PQ_c2;

        return pow(numerator / denominator, _1_div_PQ_m1);
      }
    } //PqJzazbzLmsTo

    namespace IzazbzTo
    {
      //Izazbz->Jzazbz
      float3 Jzazbz(float3 Izazbz)
      {
        float numerator   = dPlus1 * Izazbz.x;

        float denominator = 1.f + (d * Izazbz.x);

        float Jz = (numerator / denominator) - d0;

        return float3(Jz, Izazbz.yz);
      }

      //Izazbz->L'M'S' (Jzazbz variant)
      float3 PqJzazbzLms(float3 Izazbz)
      {
        return mul(IzazbzToPqJzazbzLms, Izazbz);
      }
    } //IzazbzTo

    namespace JzazbzTo
    {
      //Jzazbz->Izazbz
      float3 Izazbz(float3 Jzazbz)
      {
        float numerator   = Jzazbz.x + d0;

        float denominator = dPlus1 - (d * (Jzazbz.x + d0));

        float Iz = numerator / denominator;

        return float3(Iz, Jzazbz.yz);
      }
    } //JzazbzTo

    namespace XYZTo
    {
      //XYZ->Jzazbz
      float3 Jzazbz(float3 XYZ)
      {
        float3 NonLinearXYAndLinearZ = XYZTo::NonLinearXYAndLinearZ(XYZ);

        float3 JzazbzLms             = NonLinearXYAndLinearZTo::JzazbzLms(NonLinearXYAndLinearZ);

        float3 PqJzazbzLms           = JzazbzLmsTo::PqJzazbzLms(JzazbzLms);

        float3 Izazbz                = PqJzazbzLmsTo::Izazbz(PqJzazbzLms);

        //Jzazbz
        return IzazbzTo::Jzazbz(Izazbz);
      }
    } //XYZTo

    namespace JzazbzTo
    {
      //Jzazbz->XYZ
      float3 XYZ(float3 Jzazbz)
      {
        float3 Izazbz                = JzazbzTo::Izazbz(Jzazbz);

        float3 PqJzazbzLms           = IzazbzTo::PqJzazbzLms(Izazbz);

        float3 JzazbzLms             = PqJzazbzLmsTo::JzazbzLms(PqJzazbzLms);

        float3 NonLinearXYAndLinearZ = JzazbzLmsTo::NonLinearXYAndLinearZ(JzazbzLms);

        //XYZ
        return NonLinearXYAndLinearZTo::XYZ(NonLinearXYAndLinearZ);
      }
    } //JzazbzTo

    namespace Bt709To
    {
      //RGB BT.709->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::Bt709To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //Bt709To

    namespace DciP3To
    {
      //RGB DCI-P3->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::DciP3To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //DciP3To

    namespace Bt2020To
    {
      //RGB BT.2020->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::Bt2020To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //Bt2020To

    namespace JzazbzTo
    {
      //Jzazbz->RGB BT.709
      float3 Bt709(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB BT.709
        return Csp::Mat::XYZTo::Bt709(XYZ);
      }

      //Jzazbz->RGB DCI-P3
      float3 DciP3(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB DCI-P3
        return Csp::Mat::XYZTo::DciP3(XYZ);
      }

      //Jzazbz->RGB BT.2020
      float3 Bt2020(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB BT.2020
        return Csp::Mat::XYZTo::Bt2020(XYZ);
      }
    } //JzazbzTo

  }


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
    static const float3x3 Bt709ToOkLms =
      float3x3
      (
        0.412176460f,  0.536273956f, 0.0514403730f,
        0.211909204f,  0.680717885f, 0.107399843f,
        0.0883448123f, 0.281853973f, 0.630280852f
      );

    //RGB BT.2020->OKLMS
    static const float3x3 Bt2020ToOkLms =
      float3x3
      (
        0.616688430f, 0.360159069f, 0.0230432935f,
        0.265140205f, 0.635856509f, 0.0990302339f,
        0.100150644f, 0.204004317f, 0.696324706f
      );

    //OKL'M'S'->OKLab
    static const float3x3 G3OkLmsToOkLab =
      float3x3
      (
        0.2104542553f,  0.7936177850f, -0.0040720468f,
        1.9779984951f, -2.4285922050f,  0.4505937099f,
        0.0259040371f,  0.7827717662f, -0.8086757660f
      );

    //OKLab->OKL'M'S'
    static const float3x3 OkLabToG3OkLms =
      float3x3
      (
        1.f,  0.3963377774f,  0.2158037573f,
        1.f, -0.1055613458f, -0.0638541728f,
        1.f, -0.0894841775f, -1.2914855480f
      );

    //OKLMS->RGB BT.709
    static const float3x3 OkLmsToBt709 =
      float3x3
      (
         4.07718706f,    -3.30762243f,   0.230859190f,
        -1.26857650f,     2.60968708f,  -0.341155737f,
        -0.00419654231f, -0.703399658f,  1.70679605f
      );

    //OKLMS->RGB BT.2020
    static const float3x3 OkLmsToBt2020 =
      float3x3
      (
         2.14014029f,   -1.24635589f,   0.106431722f,
        -0.884832441f,   2.16317272f,  -0.278361588f,
        -0.0485790595f, -0.454490900f,  1.50235629f
      );


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

    namespace OkLabTo
    {

      //OKLab->OKLCh°
      float3 OkLch(const float3 OkLab)
      {
        float C = sqrt(OkLab.y * OkLab.y
                     + OkLab.z * OkLab.z);

        float h = atan2(OkLab.z, OkLab.y);

        return float3(OkLab.x, C, h);
      }

    } //OkLabTo

    namespace OkLchTo
    {

      //OKLCh°->OKLab
      float3 OkLab(const float3 OkLch)
      {
        float2 ab;
        sincos(OkLch[2], ab[0], ab[1]);

        ab *= OkLch.y;

        return float3(OkLch.x, ab);
      }

    } //OkLchTo

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

        hsv.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) * _1_DIV_PI;

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

        hsl.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) * _1_DIV_PI;

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
        float i_ab = PI_2 * Hsv.x;

        float2 ab;
        sincos(i_ab, ab[0], ab[1]);

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

        float i_ab = PI_2 * Hsl.x;

        L = Csp::OkLab::ToeInv(Hsl.z);

        sincos(i_ab, ab[0], ab[1]);

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


  namespace DarktableUcs
  {

    // -  UV is pure colour coordinates
    // - JCH is lightness (J), chroma     (C) and hue        (H)  [Helmholtz-Kohlrausch effect is corrected]
    // - HSB is hue       (H), saturation (S) and brightness (B)  [derived from JCH so HKE is corrected]
    // - HCB is hue       (H), chroma     (C) and brightness (B)  [derived from JCH so HKE is corrected]
    // - HPW is hue       (H), purity     (P) and whiteness  (W)  [derived from HSB/HCB so HKE is corrected]
    //
    // -  UV is a Lab/Luv like colour space
    // - JCH is a LCh like colour space
    //
    // -  UV is for perceptually uniform gamut mapping
    // - JCH gets you chroma amount and hue angles and is adjusted for the Helmholtz-Kohlrausch effect
    // - HSB and HCB are intermediate colour spaces to get to specifc HPW versions
    // - HPW is for purity artistic grading (painter's saturation)
    //
    // there are 3 implementations of HPW, each with different numerical issues:
    // - method 1: when (W^2 - S^2) is smaller than 0 it needs to be clipped
    // - method 2: slightly improves the issues of method 1
    // - method 3: adjusting Purity or Whiteness can have issues when converting back if I understand it correctly?


    namespace YTo
    {
      float LStar(const float Y)
      {
        float YHat = pow(Y, 0.631651345306265);

        float LStar = 2.098883786377
                    * YHat
                    / (YHat + 1.12426773749357);

        return LStar;
      }
    } //YTo

    namespace LStarTo
    {
      float Y(const float LStar)
      {
        float Y = pow(-1.12426773749357
                    * LStar
                    / (LStar - 2.098883786377)
                  , 1.5831518565279648);

        return Y;
      }
    } //LStarTo

//#define TTTTTT

    namespace xyTo
    {
      float2 UV(const float2 xy)
      {
#ifndef TTTTTT
        static const float3x3 xyToUVD =
          float3x3
          (
            -0.783941002840055,  0.277512987809202,  0.153836578598858,
             0.745273540913283, -0.205375866083878, -0.165478376301988,
             0.318707282433486,  2.16743692732158,   0.291320554395942
          );

        float3 UVD = mul(xyToUVD, float3(xy, 1.f));
#else
        static const float3x2 xyToUVD =
          float3x2
          (
            -0.783941002840055,  0.277512987809202,
             0.745273540913283, -0.205375866083878,
             0.318707282433486,  2.16743692732158
          );

        float3 UVD = mul(xyToUVD, xy) + float3(0.153836578598858, -0.165478376301988, 0.291320554395942);
#endif

        float2 UV = UVD.xy / UVD.z;

        float2 UVStar = float2(1.39656225667, 1.4513954287)
                      * UV
                      / (abs(UV) + float2(1.49217352929, 1.52488637914));

        static const float2x2 UVStarToUVStarPrime =
          float2x2
          (
            -1.124983854323892, -0.980483721769325,
             1.86323315098672,   1.971853092390862
          );

        float2 UVStarPrime = mul(UVStarToUVStarPrime, UVStar);

        return UVStarPrime;
      }
    } //xyTo

    namespace UVTo
    {
      float2 xy
      (
        const float2 UVStarPrime
      )
      {
        static const float2x2 UVStarPrimeToUVStar =
          float2x2
          (
            -5.037522385190711, -2.504856328185843,
             4.760029407436461,  2.874012963239247
          );

        float2 UVStar = mul(UVStarPrimeToUVStar, UVStarPrime);

        float2 UV = float2(-1.49217352929, -1.52488637914)
                  * UVStar
                  / (abs(UVStar) - float2(1.39656225667, 1.4513954287));

#ifndef TTTTTT
        static const float3x3 UVToxyD =
          float3x3
          (
             0.167171472114775,  0.141299802443708, -0.00801531300850582,
            -0.150959086409163, -0.155185060382272, -0.00843312433578007,
             0.940254742367256,  1.0,               -0.0256325967652889
          );

        float3 xyD = mul(UVToxyD, float3(UV, 1.f));
#else
        static const float2x2 UVToxyD =
          float2x2
          (
             0.167171472114775,  0.141299802443708,
            -0.150959086409163, -0.155185060382272
          );

        float2 xy = mul(UVToxyD, UV);

        float3 xyD = float3(xy, UV[0] * 0.940254742367256 + UV[1])
                   + float3(-0.00801531300850582, -0.00843312433578007, -0.0256325967652889);
#endif

        float2 xy = xyD.xy / xyD.z;

        return xy;
      }
    } //UVTo

    namespace xyYTo
    {
      float3 LUV
      (
        const s_xyY xyY
      )
      {
        float LStar = YTo::LStar(xyY.Y);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        return float3(LStar, UVStarPrime);
      }

      float3 LCH
      (
        const s_xyY xyY
      )
      {
        float LStar = YTo::LStar(xyY.Y);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        float C = sqrt(UVStarPrime.x * UVStarPrime.x
                     + UVStarPrime.y * UVStarPrime.y);

        float H = atan2(UVStarPrime.y, UVStarPrime.x);

        return float3(LStar, C, H);
      }

      float3 JCH
      (
        const s_xyY xyY,
        const float YWhite,
        const float cz
      )
      {
        //input:
        //  * xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
        //  * LWhite the lightness of white as dt UCS L* lightness. [is this the max white you want to display? like 10000 nits?]
        //  * cz: c * z
        //    * n = ratio of background luminance and the luminance of white,
        //    * z = 1 + sqrt(n)
        //    * c = 0.69 for average surround lighting
        //          0.59 for dim surround lighting (sRGB standard)
        //          0.525 for dark surround lighting
        //    * cz = 1 for standard pre-print proofing conditions with average surround and n = 20 %
        //          (background = middle grey, white = perfect diffuse white)
        //range:
        //  * xy in [0; 1]
        //  * Y normalized for perfect diffuse white = 1


        float LStar  = YTo::LStar(xyY.Y);
        float LWhite = YTo::LStar(YWhite);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        float M2 = UVStarPrime.x * UVStarPrime.x
                 + UVStarPrime.y * UVStarPrime.y;

        float C = 15.932993652962535
                * pow(LStar, 0.6523997524738018)
                * pow(M2,    0.6007557017508491)
                / LWhite;

        float J = pow(LStar / LWhite, cz);

        float H = atan2(UVStarPrime.y, UVStarPrime.x);

        return float3(J, C, H);
      }
    } //xyYTo

    namespace LUVTo
    {
      s_xyY xyY
      (
        const float3 LUV
      )
      {
        s_xyY xyY;

        xyY.xy = UVTo::xy(LUV.yz);

        xyY.Y = LStarTo::Y(LUV[0]);

        return xyY;
      }
    } //LUVTo

    namespace LCHTo
    {
      s_xyY xyY
      (
        const float3 LCH
      )
      {
        float2 UVStarPrime;
        sincos(LCH[2], UVStarPrime[0], UVStarPrime[1]);

        UVStarPrime *= LCH[1];

        s_xyY xyY;

        xyY.xy = UVTo::xy(UVStarPrime);

        xyY.Y = LStarTo::Y(LCH[0]);

        return xyY;
      }
    } //LCHTo

    namespace JCHTo
    {
      s_xyY xyY
      (
        const float3 JCH,
        const float  YWhite,
        const float  cz
      )
      {
        //output: xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
        //range:
        //  * xy in [0; 1]
        //  * Y normalized for perfect diffuse white = 1

        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float LWhite = YTo::LStar(YWhite);

        float LStar = pow(J, (1.f / cz)) * LWhite;

        float M = pow((C
                     * LWhite
                     / (15.932993652962535 * pow(LStar,
                                                 0.6523997524738018)))
                  , 0.8322850678616855);

        float2 UVStarPrime;
        sincos(H, UVStarPrime[0], UVStarPrime[1]);

        UVStarPrime *= M;

        s_xyY xyY;

        xyY.xy = UVTo::xy(UVStarPrime);

        xyY.Y = LStarTo::Y(LStar);

        return xyY;
      }

      float3 HCB(const float3 JCH)
      {
        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float B = J
                * (pow(C, 1.33654221029386) + 1.f);

        return float3(H, C, B);
      }

      float3 HSB(const float3 JCH)
      {
        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float B = J
                * (pow(C, 1.33654221029386) + 1.f);

        float S = C / B;

        return float3(H, S, B);
      }
    } //JCHTo

    namespace HCBTo
    {
      float3 JCH(const float3 HCB)
      {
        float H = HCB[0];
        float C = HCB[1];
        float B = HCB[2];

        float J = B
                / (pow(C, 1.33654221029386) + 1.f);

        return float3(J, C, H);
      }
    } //HCBTo

    namespace HSBTo
    {
      float3 JCH(const float3 HSB)
      {
        float H = HSB[0];
        float S = HSB[1];
        float B = HSB[2];

        float C = S * B;
        float J = B
                / (pow(C, 1.33654221029386) + 1.f);

        return float3(J, C, H);
      }
    } //HSBTo

    // has numerical issues:
    // when (W^2 - S^2) is smaller than 0 it needs to be clipped
    namespace Method1
    {

      namespace HSBTo
      {
        float3 HPW(const float3 HSB)
        {
          float H = HSB[0];
          float S = HSB[1];
          float B = HSB[2];

          float W = sqrt(S * S + B * B);
          float P = S / W;

          return float3(H, P, W);
        }
      } //HSBTo

      namespace HPWTo
      {
        float3 HSB(const float3 HPW)
        {
          float H = HPW[0];
          float P = HPW[1];
          float W = HPW[2];

          float S = P;
          float B = sqrt(W * W - S * S);

          return float3(H, S, B);
        }
      } //HPWTo

    } //Method1

    namespace Method2
    {

      namespace HSBTo
      {
        float3 HPW(const float3 HSB)
        {
          float H = HSB[0];
          float S = HSB[1];
          float B = HSB[2];

          float W = B;
          float P = W / B;

          return float3(H, P, W);
        }
      } //HSBTo

      namespace HPWTo
      {
        float3 HSB(const float3 HPW)
        {
          float H = HPW[0];
          float P = HPW[1];
          float W = HPW[2];

          float S = W * P;
          float B = W;

          return float3(H, S, B);
        }
      } //HPWTo

    } //Method2

    namespace Method3
    {

      namespace HCBTo
      {
        float3 HPWcc(const float3 HCB)
        {
          float H = HCB[0];
          float C = HCB[1];
          float B = HCB[2];

          float W = sqrt(2.f * (C * C) + (B * B));
          float P = 2.f * C / W;

          return float3(H, P, W);
        }
      } //HCBTo

      namespace HPWccTo
      {
        float3 HCB(const float3 HPWcc)
        {
          float H = HPWcc[0];
          float P = HPWcc[1];
          float W = HPWcc[2];

          float C = P * W / 2.f;
          float B = sqrt((W * W) - 2.f * (C * C));

          return float3(H, C, B);
        }
      } //HPWccTo

    } //Method3

  } //DarktableUcs


  namespace Map
  {

    namespace Bt709Into
    {

      float3 ScRgb(
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

  } //Map

}


#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  #define HDR10_TO_LINEAR_LUT()                                    \
    texture1D TextureHdr10ToLinearLUT                              \
    {                                                              \
      Width  = 1024;                                               \
      Format = R32F;                                               \
    };                                                             \
                                                                   \
    sampler1D<float> SamplerHdr10ToLinearLUT                       \
    {                                                              \
      Texture = TextureHdr10ToLinearLUT;                           \
    };                                                             \
                                                                   \
    storage1D<float> StorageHdr10ToLinearLUT                       \
    {                                                              \
      Texture = TextureHdr10ToLinearLUT;                           \
    };                                                             \
                                                                   \
    void CS_CreateHdr10ToLinearLUT                                 \
    (                                                              \
      uint3 DTID : SV_DispatchThreadID                             \
    )                                                              \
    {                                                              \
      float curr = Csp::Trc::PqTo::Linear(float(DTID.x) / 1023.f); \
                                                                   \
      tex1Dstore(StorageHdr10ToLinearLUT, DTID.x, curr);           \
                                                                   \
      return;                                                      \
    }                                                              \
                                                                   \
    float FetchFromHdr10ToLinearLUT                                \
    (                                                              \
      float Channel                                                \
    )                                                              \
    {                                                              \
      const int lookup = int(Channel * 1023.f + 0.5f);             \
                                                                   \
      Channel = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup);       \
                                                                   \
      return Channel;                                              \
    }                                                              \
                                                                   \
    float2 FetchFromHdr10ToLinearLUT                               \
    (                                                              \
      float2 Channels                                              \
    )                                                              \
    {                                                              \
      const int2 lookup = int2(Channels * 1023.f + 0.5f);          \
                                                                   \
      Channels.x = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.x);  \
      Channels.y = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.y);  \
                                                                   \
      return Channels;                                             \
    }                                                              \
                                                                   \
    float3 FetchFromHdr10ToLinearLUT                               \
    (                                                              \
      float3 Colour                                                \
    )                                                              \
    {                                                              \
      const int3 lookup = int3(Colour * 1023.f + 0.5f);            \
                                                                   \
      Colour.r = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.r);    \
      Colour.g = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.g);    \
      Colour.b = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.b);    \
                                                                   \
      return Colour;                                               \
    }                                                              \
                                                                   \
    technique lilium__create_hdr10_to_linear_lut                   \
    <                                                              \
      enabled = true;                                              \
      hidden  = true;                                              \
      timeout = 1;                                                 \
    >                                                              \
    {                                                              \
      pass CreateHdr10ToLinearLUT                                  \
      {                                                            \
        ComputeShader = CS_CreateHdr10ToLinearLUT <16, 1>;         \
        DispatchSizeX = 64; /* (1024 / 16) */                      \
        DispatchSizeY = 1;                                         \
      }                                                            \
    }

#else

  #define HDR10_TO_LINEAR_LUT()

#endif


#if (OVERWRITE_SDR_GAMMA == GAMMA_24)

  #define ENCODE_SDR(COLOUR) \
            pow(COLOUR, 1.f / 2.4f)

  #define DECODE_SDR(COLOUR) \
            pow(COLOUR, 2.4f)

#elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)

  #define ENCODE_SDR(COLOUR) \
            Csp::Trc::LinearTo::Srgb(COLOUR)

  #define DECODE_SDR(COLOUR) \
            Csp::Trc::SrgbTo::Linear(COLOUR)

#else

  #define ENCODE_SDR(COLOUR) \
            pow(COLOUR, 1.f / 2.2f)

  #define DECODE_SDR(COLOUR) \
            pow(COLOUR, 2.2f)

#endif


// convert BT.709 to BT.2020
float3 ConditionallyConvertBt709ToBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
#endif
  return Colour;
}

// convert DCI-P3 to BT.709
float3 ConditionallyConvertDciP3ToBt709(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::DciP3To::Bt709(Colour);
#endif
  return Colour;
}

// convert DCI-P3 to BT.2020
float3 ConditionallyConvertDciP3ToBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Mat::DciP3To::Bt2020(Colour);
#endif
  return Colour;
}

// convert BT.2020 to BT.709
float3 ConditionallyConvertBt2020ToBt709(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
#endif
  return Colour;
}

// normalise so that 10000 = 1
float3 ConditionallyNormaliseScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour /= 125.f;
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
float3 ConditionallyConvertScRgbToNormalisedBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::ScRgbTo::Bt2020Normalised(Colour);
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
// - convert normalised to PQ
float3 ConditionallyConvertScRgbToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = ConditionallyConvertScRgbToNormalisedBt2020(Colour);
  Colour = Csp::Trc::LinearTo::Pq(Colour);
#endif
  return Colour;
}

// convert HDR10 to normalised BT.2020
float3 ConditionallyLineariseHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::PqTo::Linear(Colour);
#endif
  return Colour;
}

// convert normalised BT.2020 to HDR10
float3 ConditionallyConvertNormalisedBt2020ToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::LinearTo::Pq(Colour);
#endif
  return Colour;
}

// convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt709ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour *= 125.f;
#endif
  return Colour;
}

// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt2020ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020NormalisedTo::ScRgb(Colour);
#endif
  return Colour;
}

// - convert HDR10 to normalised BT.2020
// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertHdr10ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Trc::PqTo::Linear(Colour);
  Colour = ConditionallyConvertNormalisedBt2020ToScRgb(Colour);
#endif
  return Colour;
}

// get luminance for the current colour space
float GetLuminance(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  return dot(Colour, Csp::Mat::Bt709ToXYZ[1]);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  return dot(Colour, Csp::Mat::Bt2020ToXYZ[1]);
#else
  return 0;
#endif
}


#ifdef ENABLE_COLOUR_OBJECT

  HDR10_TO_LINEAR_LUT()

  namespace CO
  {
    #define PRIM_BT709  0
    #define PRIM_DCI_P3 1
    #define PRIM_BT2020 2

    #define TRC_LINEAR_NORMALISED 0
    #define TRC_LINEAR_80         1
    #define TRC_PQ                2

    struct ColourObject
    {
      float3 RGB;
      uint   prim;
      uint   trc;
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      bool   isUntouched;
#endif
    };


    namespace GetLuminance
    {
      float LinearNormalised
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                return dot(CO.RGB, Csp::Mat::Bt709ToXYZ[1]);
              }
              case PRIM_DCI_P3:
              {
                return dot(CO.RGB, Csp::Mat::DciP3ToXYZ[1]);
              }
              case PRIM_BT2020:
              {
                return dot(CO.RGB, Csp::Mat::Bt2020ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                return dot(CO.RGB, Csp::Mat::ScRgbToXYZ[1]);
              }
              case PRIM_DCI_P3:
              {
                return dot(CO.RGB, Csp::Mat::DciP3_80ToXYZ[1]);
              }
              case PRIM_BT2020:
              {
                return dot(CO.RGB, Csp::Mat::Bt2020_80ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
                float3 linearColour;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  linearColour = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  linearColour = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                return dot(linearColour, Csp::Mat::Bt2020ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          default:
            return 0.f;
        }
      }

//      float Linear80
//      (
//        ColourObject CO
//      )
//      {
//        [forcecase]
//        switch(CO.trc)
//        {
//          case TRC_LINEAR_NORMALISED:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT709:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt709ToXYZ[1]);
//              }
//              case PRIM_DCI_P3:
//              {
//                return dot(CO.RGB, Csp::Mat::DciP3ToXYZ[1]);
//              }
//              case PRIM_BT2020:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt2020ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//          case TRC_LINEAR_80:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT709:
//              {
//                return dot(CO.RGB, Csp::Mat::ScRgbToXYZ[1]);
//              }
//              case PRIM_DCI_P3:
//              {
//                return dot(CO.RGB, Csp::Mat::DciP3_80ToXYZ[1]);
//              }
//              case PRIM_BT2020:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt2020_80ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//          case TRC_PQ:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT2020:
//              {
//                return dot(Csp::Trc::PqTo::Linear(CO.RGB), Csp::Mat::Bt2020ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//        }
//      }
    }


    namespace ConvertCspTo
    {
      ColourObject ScRgb
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
                break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB = Csp::Mat::Bt2020NormalisedTo::ScRgb(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT709;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Hdr10
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::ScRgbTo::Bt2020Normalised(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3_80To::Bt2020Normalised(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB /= 125.f;

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT2020;
        CO.trc  = TRC_PQ;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject DciP3_80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
                break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Bt2020_80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }

    namespace ConvertPrimariesTo
    {
      ColourObject Bt709
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
                break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT709;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject DciP3
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
                break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Bt2020
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT2020;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }


    namespace ConvertTrcTo
    {
      ColourObject LinearNormalised
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
            break;
          case TRC_LINEAR_80:
          {
            CO.RGB /= 125.f;
          }
          break;
          case TRC_PQ:
          {
            [branch]
            if (CO.prim == PRIM_BT2020)
            {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
              if (CO.isUntouched)
              {
                CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
              }
              else
#endif
              {
                CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
              }
            }
            else
            {
              CO.RGB = 0.f;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.trc = TRC_LINEAR_NORMALISED;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Linear80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            CO.RGB *= 125.f;
          }
          break;
          case TRC_LINEAR_80:
            break;
          case TRC_PQ:
          {
            [branch]
            if (CO.prim == PRIM_BT2020)
            {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
              if (CO.isUntouched)
              {
                CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
              }
              else
#endif
              {
                CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
              }

              CO.RGB *= 125.f;
            }
            else
            {
              CO.RGB = 0.f;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.trc = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Pq
      (
        ColourObject CO
      )
      {
        [branch]
        if (CO.prim == PRIM_BT2020)
        {
          [forcecase]
          switch(CO.trc)
          {
            case TRC_LINEAR_NORMALISED:
            {
              CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
            }
            break;
            case TRC_LINEAR_80:
            {
              CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB / 125.f);
            }
            break;
            default:
              CO.RGB = 0.f;
              break;
          }
        }
        else
        {
          CO.RGB = 0.f;
        }

        CO.trc = TRC_PQ;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }

  }

#endif //ENABLE_COLOUR_OBJECT


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


//bool IsNAN(float Input)
//{
//  if (isnan(Input) || isinf(Input))
//    return true;
//  else
//    return false;
//}
//
//float fixNAN(float Input)
//{
//  if (IsNAN(Input))
//    return 0.f;
//  else
//    return Input;
//}
//
//float3 fixNAN(float3 Input)
//{
//  if (IsNAN(Input.r))
//    Input.r = 0.f;
//  else if (IsNAN(Input.g))
//    Input.g = 0.f;
//  else if (IsNAN(Input.b))
//    Input.b = 0.f;
//
//  return Input;
//}
