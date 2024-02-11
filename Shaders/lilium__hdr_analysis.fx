#include "lilium__include/colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_HDR_CSP))

#define GAMMA_SRGB 0
#define GAMMA_22   1
#define GAMMA_24   2

#ifndef OVERWRITE_SDR_GAMMA
  #define OVERWRITE_SDR_GAMMA GAMMA_22
#endif

#undef TEXT_BRIGHTNESS

#ifndef GAMESCOPE
  //#define _DEBUG
  //#define _TESTY
#endif

#if (BUFFER_WIDTH  >= 2560) \
 && (BUFFER_HEIGHT >= 1440)
  #define IS_QHD_OR_HIGHER_RES
#endif


#if defined(IS_HDR_CSP)
  #define DEFAULT_ALPHA_LEVEL 80.f
#else
  #define DEFAULT_ALPHA_LEVEL 50.f
#endif

uniform int2 MOUSE_POSITION
<
  source = "mousepoint";
>;

uniform float2 NIT_PINGPONG0
<
  source    = "pingpong";
  min       = 0.f;
  max       = 2.f;
  step      = 1.f;
  smoothing = 0.f;
>;

uniform float2 NIT_PINGPONG1
<
  source    = "pingpong";
  min       =  0.f;
  max       =  3.f;
  step      =  1.f;
  smoothing =  0.f;
>;

uniform float2 NIT_PINGPONG2
<
  source    = "pingpong";
  min       = 0.f;
  max       = 1.f;
  step      = 3.f;
  smoothing = 0.f;
>;

#if (defined(GAMESCOPE) \
  && BUFFER_HEIGHT <= 800)
  static const uint TEXT_SIZE_DEFAULT = 0;
#else
  // 2160 height gives 42
  //  800 height gives 16
  //  720 height gives 14
  static const uint TEXT_SIZE_DEFAULT = (uint((BUFFER_HEIGHT / 2160.f) * 42.f + 0.5f) - 12) / 2;
#endif


uniform uint TEXT_SIZE
<
  ui_category = "Global";
  ui_label    = "text size";
  ui_type     = "combo";
  ui_items    = "13\0"
                "14\0"
                "16\0"
                "18\0"
                "20\0"
                "22\0"
                "24\0"
                "26\0"
                "28\0"
                "30\0"
                "32\0"
                "34\0"
                "36\0"
                "38\0"
                "40\0"
                "42\0"
                "44\0"
                "46\0"
                "48\0"
                "50\0"
                "52\0"
                "54\0"
                "56\0"
                "58\0";
> = TEXT_SIZE_DEFAULT;

uniform float TEXT_BRIGHTNESS
<
  ui_category = "Global";
  ui_label    = "text brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
#ifdef GAMESCOPE
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = 140.f;
#endif

uniform float TEXT_BG_ALPHA
<
  ui_category = "Global";
  ui_label    = "text background transparency";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.5f;
> = DEFAULT_ALPHA_LEVEL;

#define TEXT_POSITION_TOP_LEFT  0
#define TEXT_POSITION_TOP_RIGHT 1

uniform uint TEXT_POSITION
<
  ui_category = "Global";
  ui_label    = "text position";
  ui_type     = "combo";
  ui_items    = "top left\0"
                "top right\0";
> = 0;


// Active Area
uniform bool ACTIVE_AREA_ENABLE
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "enable setting the active area";
> = false;

uniform float ACTIVE_AREA_CROP_LEFT
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the left side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float ACTIVE_AREA_CROP_TOP
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the top side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float ACTIVE_AREA_CROP_RIGHT
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the right side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float ACTIVE_AREA_CROP_BOTTOM
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the bottom side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;


// Nit Values
uniform bool SHOW_NITS_VALUES
<
  ui_category = "Luminance analysis";
  ui_label    = "show max/avg/min luminance values";
> = true;

uniform bool SHOW_NITS_FROM_CURSOR
<
  ui_category = "Luminance analysis";
  ui_label    = "show luminance value from cursor position";
> = true;


// TextureCsps
uniform bool SHOW_CSPS
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour spaces used";
  ui_tooltip  = "in %";
> = true;

uniform bool SHOW_CSP_FROM_CURSOR
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour space from cursor position";
> = true;

uniform bool SHOW_CSP_MAP
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour space map";
  ui_tooltip  = "        colours:"
           "\n" "black and white: BT.709"
           "\n" "         yellow: DCI-P3"
           "\n" "           blue: BT.2020"
           "\n" "            red: AP0"
           "\n" "           pink: invalid";
> = false;

// CIE
uniform bool SHOW_CIE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show CIE diagram";
> = true;

#define CIE_1931 0
#define CIE_1976 1

uniform uint CIE_DIAGRAM_TYPE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram type";
  ui_type     = "combo";
  ui_items    = "CIE 1931 xy\0"
                "CIE 1976 UCS u'v'\0";
> = CIE_1931;

uniform float CIE_DIAGRAM_BRIGHTNESS
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
#ifdef GAMESCOPE
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = 80.f;
#endif

uniform float CIE_DIAGRAM_ALPHA
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram transparency";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.5f;
> = DEFAULT_ALPHA_LEVEL;

#ifdef IS_QHD_OR_HIGHER_RES
  #define CIE_TEXTURE_FILE_NAME   "lilium__cie_1000x1000_consolidated.png"
  #define CIE_TEXTURE_WIDTH       5016
  #define CIE_TEXTURE_HEIGHT      1626
  #define CIE_BG_BORDER             50

  #define CIE_ORIGINAL_DIM        1000

  #define CIE_1931_WIDTH           736
  #define CIE_1931_HEIGHT          837
  #define CIE_1931_BG_WIDTH        836
  #define CIE_1931_BG_HEIGHT       937

  #define CIE_1976_WIDTH           625
  #define CIE_1976_HEIGHT          589
  #define CIE_1976_BG_WIDTH        725
  #define CIE_1976_BG_HEIGHT       689
#else
  #define CIE_TEXTURE_FILE_NAME   "lilium__cie_500x500_consolidated.png"
  #define CIE_TEXTURE_WIDTH       2520
  #define CIE_TEXTURE_HEIGHT       817
  #define CIE_BG_BORDER             25

  #define CIE_ORIGINAL_DIM         500

  #define CIE_1931_WIDTH           370
  #define CIE_1931_HEIGHT          420
  #define CIE_1931_BG_WIDTH        420
  #define CIE_1931_BG_HEIGHT       470

  #define CIE_1976_WIDTH           314
  #define CIE_1976_HEIGHT          297
  #define CIE_1976_BG_WIDTH        364
  #define CIE_1976_BG_HEIGHT       347
#endif

static const float2 CIE_CONSOLIDATED_TEXTURE_SIZE = float2(CIE_TEXTURE_WIDTH, CIE_TEXTURE_HEIGHT);

static const int2 CIE_1931_SIZE = int2(CIE_1931_WIDTH, CIE_1931_HEIGHT);
static const int2 CIE_1976_SIZE = int2(CIE_1976_WIDTH, CIE_1976_HEIGHT);

static const float CIE_BG_WIDTH[2]  = { CIE_1931_BG_WIDTH,  CIE_1976_BG_WIDTH };
static const float CIE_BG_HEIGHT[2] = { CIE_1931_BG_HEIGHT, CIE_1976_BG_HEIGHT };

static const int CIE_BG_WIDTH_AS_INT[2]  = { CIE_1931_BG_WIDTH,  CIE_1976_BG_WIDTH };
static const int CIE_BG_HEIGHT_AS_INT[2] = { CIE_1931_BG_HEIGHT, CIE_1976_BG_HEIGHT };

static const float CIE_DIAGRAM_DEFAULT_SIZE = (float(BUFFER_HEIGHT) * 0.375f)
                                              / CIE_1931_BG_HEIGHT
                                              * 100.f;

uniform float CIE_DIAGRAM_SIZE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = CIE_DIAGRAM_DEFAULT_SIZE;

uniform bool SHOW_CIE_CSP_BT709_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show BT.709 colour space outline";
> = true;

uniform bool SHOW_CIE_CSP_DCI_P3_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show DCI-P3 colour space outline";
> = true;

uniform bool SHOW_CIE_CSP_BT2020_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show BT.2020 colour space outline";
> = true;

#ifdef IS_FLOAT_HDR_CSP

uniform bool SHOW_CIE_CSP_AP0_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show AP0 colour space outline";
> = true;

#endif

// heatmap
uniform bool SHOW_HEATMAP
<
  ui_category = "Heatmap visualisation";
  ui_label    = "show heatmap";
> = false;

uniform float HEATMAP_BRIGHTNESS
<
  ui_category = "Heatmap visualisation";
  ui_label    = "heatmap brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

uniform uint HEATMAP_CUTOFF_POINT
<
  ui_category = "Heatmap visualisation";
  ui_label    = "heatmap cutoff point";
  ui_type     = "combo";
  ui_items    = "10000 nits\0"
                " 4000 nits\0"
                " 2000 nits\0"
                " 1000 nits\0";
  ui_tooltip  = "         colours: | 10000 nits: | 4000 nits: | 2000 nits: | 1000 nits:"
           "\n" "------------------|-------------|------------|------------|-----------"
           "\n" " black to white:  |     0-  100 |     0- 100 |     0- 100 |     0- 100"
           "\n" "  teal to green:  |   100-  203 |   100- 203 |   100- 203 |   100- 203"
           "\n" " green to yellow: |   203-  400 |   203- 400 |   203- 400 |   203- 400"
           "\n" "yellow to red:    |   400- 1000 |   400-1000 |   400-1000 |   400- 600"
           "\n" "   red to pink:   |  1000- 4000 |  1000-2000 |  1000-1500 |   600- 800"
           "\n" "  pink to blue:   |  4000-10000 |  2000-4000 |  1500-2000 |   800-1000"
           "\n" "----------------------------------------------------------------------"
           "\n"
           "\n" "extra cases:"
           "\n" "highly saturated red:  above the cutoff point"
           "\n" "highly saturated blue: below 0 nits";
> = 0;

uniform bool SHOW_LUMINANCE_WAVEFORM
<
  ui_category = "Luminance waveform";
  ui_label    = "show luminance waveform";
  ui_tooltip  = "Luminance waveform paid for by Aemony.";
> = true;

uniform uint LUMINANCE_WAVEFORM_CUTOFF_POINT
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform cutoff point";
  ui_type     = "combo";
  ui_items    = "10000 nits\0"
                " 4000 nits\0"
                " 2000 nits\0"
                " 1000 nits\0";
> = 0;

uniform float LUMINANCE_WAVEFORM_BRIGHTNESS
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
#ifdef GAMESCOPE
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = 80.f;
#endif

uniform float LUMINANCE_WAVEFORM_ALPHA
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform transparency";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.5f;
> = DEFAULT_ALPHA_LEVEL;

static const uint TEXTURE_LUMINANCE_WAVEFORM_WIDTH = uint(float(BUFFER_WIDTH) / 4.f) * 2;

#if (BUFFER_HEIGHT <= (512 * 5 / 2))
  static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 512;
#elif (BUFFER_HEIGHT <= (1024 * 5 / 2))
  static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1024;
#elif (BUFFER_HEIGHT <= (2048 * 5 / 2))
  static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 2048;
#elif (BUFFER_HEIGHT <= (4096 * 5 / 2))
  static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 4096;
#else
  static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = (float(BUFFER_HEIGHT) / 2.f) + 0.5f;
#endif

static const uint TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT = TEXTURE_LUMINANCE_WAVEFORM_HEIGHT - 1;

static const float LUMINANCE_WAVEFORM_DEFAULT_HEIGHT = (float(BUFFER_HEIGHT) * 0.35f)
                                                       / float(TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT)
                                                       * 100.f;

uniform float2 LUMINANCE_WAVEFORM_SIZE
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = float2(70.f, LUMINANCE_WAVEFORM_DEFAULT_HEIGHT);

uniform bool LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE
<
  ui_category = "Luminance waveform";
  ui_label    = "show the minimum nits line";
  ui_tooltip  = "Show a horizontal line where the minimum nits is on the waveform."
           "\n" "The line is invisible when the minimum nits hits 0 nits.";
> = true;

uniform bool LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE
<
  ui_category = "Luminance waveform";
  ui_label    = "show the maximum nits line";
  ui_tooltip  = "Show a horizontal line where the maximum nits is on the waveform."
           "\n" "The line is invisible when the maximum nits hits above 10000 nits.";
> = true;

// highlight a certain nit range
uniform bool HIGHLIGHT_NIT_RANGE
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "enable highlighting brightness levels in a certain range";
  ui_tooltip  = "in nits";
> = false;

uniform float HIGHLIGHT_NIT_RANGE_BRIGHTNESS
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "highlighted range brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

uniform float HIGHLIGHT_NIT_RANGE_START_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range starting point";
  ui_tooltip  = "CTRL + LEFT CLICK on the value to input an exact value.";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 0.f;

uniform float HIGHLIGHT_NIT_RANGE_END_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range end point";
  ui_tooltip  = "CTRL + LEFT CLICK on the value to input an exact value.";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 0.f;

// draw pixels as black depending on their nits
uniform bool DRAW_ABOVE_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "enable drawing above this brightness as black";
> = false;

uniform float ABOVE_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "draw above this brightness as black";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 10000.f;

uniform bool DRAW_BELOW_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "enable drawing below this brightness as black";
> = false;

uniform float BELOW_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "draw below this brightness as black";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 1.f;
> = 0.f;


#ifdef _TESTY
uniform bool ENABLE_TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "enable test thingy";
> = false;

uniform float2 TEST_AREA_SIZE
<
  ui_category = "TESTY";
  ui_label    = "test area size";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = BUFFER_WIDTH;
  ui_step     = 1.f;
> = float2(100.f, 100.f);

#define TESTY_MODE_RGB_BT709     0
#define TESTY_MODE_RGB_DCI_P3    1
#define TESTY_MODE_RGB_BT2020    2
#define TESTY_MODE_RGB_AP0D65    3
#define TESTY_MODE_xyY           4
#define TESTY_MODE_POS_INF_ON_R  5
#define TESTY_MODE_NEG_INF_ON_G  6
#define TESTY_MODE_NAN_ON_B      7

uniform uint TEST_MODE
<
  ui_category = "TESTY";
  ui_label    = "mode";
  ui_type     = "combo";
  ui_items    = " RGB BT.709\0"
                " RGB DCI-P3\0"
                " RGB BT.2020\0"
                " ACES AP0 (D65 white point)\0"
                " xyY\0"
                "+Inf (0x7F800000) on R (else is 0)\0"
                "-Inf (0xFF800000) on G (else is 0)\0"
                " NaN (0xFFFFFFFF) on B (else is 0)\0";
> = TESTY_MODE_RGB_BT709;

precise uniform float3 TEST_THINGY_RGB
<
  ui_category = "TESTY";
  ui_label    = "RGB (linear)";
  ui_type     = "drag";
  ui_min      = -10000.f;
  ui_max      =  10000.f;
  ui_step     =      0.000001f;
> = 80.f;

precise uniform float2 TEST_THINGY_xy
<
  ui_category = "TESTY";
  ui_label    = "xy";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.000001f;
> = 0.f;

precise uniform float TEST_THINGY_Y
<
  ui_category = "TESTY";
  ui_label    = "Y";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      =     0.f;
  ui_max      = 10000.f;
  ui_step     =     0.000001f;
> = 80.f;


void VS_Testy(
  in                  uint   Id         : SV_VertexID,
  out                 float4 VPos       : SV_Position,
  out                 float2 TexCoord   : TEXCOORD0,
  out nointerpolation float4 TestyStuff : TestyStuff)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

  float2 testAreaSizeDiv2 = TEST_AREA_SIZE / 2.f;

  TestyStuff.x = BUFFER_WIDTH  / 2.f - testAreaSizeDiv2.x - 1.f;
  TestyStuff.y = BUFFER_HEIGHT / 2.f - testAreaSizeDiv2.y - 1.f;
  TestyStuff.z = BUFFER_WIDTH  - TestyStuff.x;
  TestyStuff.w = BUFFER_HEIGHT - TestyStuff.y;
}

void PS_Testy(
  in                          float4 VPos       : SV_Position,
  in                          float2 TexCoord   : TEXCOORD0,
  in  nointerpolation         float4 TestyStuff : TestyStuff,
  out                 precise float4 Output     : SV_Target0)
{
  const float2 pureCoord = floor(VPos.xy);

  if(ENABLE_TEST_THINGY
  && pureCoord.x > TestyStuff.x
  && pureCoord.x < TestyStuff.z
  && pureCoord.y > TestyStuff.y
  && pureCoord.y < TestyStuff.w)
  {
#if (ACTUAL_COLOUR_SPACE != CSP_SCRGB) \
 && (ACTUAL_COLOUR_SPACE != CSP_HDR10)
  Output = float4(0.f, 0.f, 0.f, 1.f);
  return;
#endif
    if (TEST_MODE == TESTY_MODE_RGB_BT709)
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      Output = float4(TEST_THINGY_RGB / 80.f, 1.f);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      Output = float4(Csp::Trc::LinearTo::Pq(Csp::Mat::Bt709To::Bt2020(TEST_THINGY_RGB / 10000.f)), 1.f);
#endif
      return;
    }
    else if (TEST_MODE == TESTY_MODE_RGB_DCI_P3)
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      Output = float4(Csp::Mat::DciP3To::Bt709(TEST_THINGY_RGB / 80.f), 1.f);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      Output = float4(Csp::Trc::LinearTo::Pq(Csp::Mat::DciP3To::Bt2020(TEST_THINGY_RGB / 10000.f)), 1.f);
#endif
      return;
    }
    else if (TEST_MODE == TESTY_MODE_RGB_BT2020)
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      Output = float4(Csp::Mat::Bt2020To::Bt709(TEST_THINGY_RGB / 80.f), 1.f);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      Output = float4(Csp::Trc::LinearTo::Pq(TEST_THINGY_RGB / 10000.f), 1.f);
#endif
      return;
    }
    else if (TEST_MODE == TESTY_MODE_RGB_AP0D65)
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      Output = float4(Csp::Mat::Ap0D65To::Bt709(TEST_THINGY_RGB / 80.f), 1.f);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      Output = float4(Csp::Trc::LinearTo::Pq(Csp::Mat::Ap0D65To::Bt2020(TEST_THINGY_RGB / 10000.f)), 1.f);
#endif
      return;
    }
    else if (TEST_MODE == TESTY_MODE_xyY)
    {

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
      precise float3 XYZ = GetXYZfromxyY(TEST_THINGY_xy, TEST_THINGY_Y / 80.f);
      Output = float4(Csp::Mat::XYZTo::Bt709(XYZ), 1.f);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      precise float3 XYZ = GetXYZfromxyY(TEST_THINGY_xy, TEST_THINGY_Y / 10000.f);
      Output = float4(Csp::Trc::LinearTo::Pq(Csp::Mat::XYZTo::Bt2020(XYZ)), 1.f);
#endif
      return;
    }
    else if (TEST_MODE == TESTY_MODE_POS_INF_ON_R)
    {
      static const precise float asFloat = asfloat(0x7F800000);
      Output = float4(asFloat, 0.f, 0.f, 1.f);
      return;
    }
    else if (TEST_MODE == TESTY_MODE_NEG_INF_ON_G)
    {
      static const precise float asFloat = asfloat(0xFF800000);
      Output = float4(0.f, asFloat, 0.f, 1.f);
      return;
    }
    else if (TEST_MODE == TESTY_MODE_NAN_ON_B)
    {
      static const precise float asFloat = asfloat(0xFFFFFFFF);
      Output = float4(0.f, 0.f, asFloat, 1.f);
      return;
    }
    else
    {
      Output = float4(0.f, 0.f, 0.f, 1.f);
      return;
    }
  }
  Output = float4(0.f, 0.f, 0.f, 1.f);
  return;
}
#endif //_TESTY

#define CSP_DESC_SPACING_MULTIPLIER 0.5f
#define SPACING_MULTIPLIER          0.3f

#define SHOW_NITS_VALUES_LINE_COUNT      3
#define SHOW_NITS_FROM_CURSOR_LINE_COUNT 1

#if defined(IS_HDR10_LIKE_CSP)

  #define SHOW_CSPS_LINE_COUNT 3

#else //IS_HDR10_LIKE_CSP

  #define SHOW_CSPS_LINE_COUNT 5

#endif //IS_HDR10_LIKE_CSP

#define SHOW_CSP_FROM_CURSOR_LINE_COUNT 1



#define HDR_ANALYSIS_ENABLE

#include "lilium__include/hdr_analysis.fxh"


// outer spacing is half the size of a character rounded up
uint GetOuterSpacing(const uint CharXDimension)
{
  float charXDimAsFloat = float(CharXDimension);

  return uint(charXDimAsFloat / 2.f + 0.5f);
}

uint GetActiveLines()
{
  return 1
       + (SHOW_NITS_VALUES ? SHOW_NITS_VALUES_LINE_COUNT
                           : 0)
       + (SHOW_NITS_FROM_CURSOR ? SHOW_NITS_FROM_CURSOR_LINE_COUNT
                                : 0)
       + (SHOW_CSPS ? SHOW_CSPS_LINE_COUNT
                    : 0)
       + (SHOW_CSP_FROM_CURSOR ? SHOW_CSP_FROM_CURSOR_LINE_COUNT
                               : 0);
}

#ifdef IS_HDR_CSP
  #define CSP_DESC_TEXT_LENGTH 20
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
  #if (OVERWRITE_SDR_GAMMA == GAMMA_22 \
    || OVERWRITE_SDR_GAMMA == GAMMA_24)
    #define CSP_DESC_TEXT_LENGTH 24
  #elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)
    #define CSP_DESC_TEXT_LENGTH 19
  #else
    #define CSP_DESC_TEXT_LENGTH 23
  #endif
#endif

uint GetActiveCharacters()
{
//  return MAX5(CSP_DESC_TEXT_LENGTH,
//              (SHOW_NITS_VALUES ? 21
//                                :  0),
//              (SHOW_NITS_FROM_CURSOR ? 24
//                                     :  0),
//              (SHOW_CSPS ? 16
//                         :  0),
//              (SHOW_CSP_FROM_CURSOR ? 18
//                                    :  0));
  return MAX3(CSP_DESC_TEXT_LENGTH,
              (SHOW_NITS_VALUES ? 21
                                :  0),
              (SHOW_NITS_FROM_CURSOR ? 24
                                     :  0));
}

uint GetAtlasEntry()
{
  return 23 - TEXT_SIZE;
}

uint GetCharArrayEntry()
{
  return GetAtlasEntry() * 2;
}

void CS_PrepareOverlay(uint3 ID : SV_DispatchThreadID)
{
  //convert UI inputs into floats for comparisons
  const float showNitsValues     = SHOW_NITS_VALUES;
  const float showNitsFromCrusor = SHOW_NITS_FROM_CURSOR;

  const float showCsps          = SHOW_CSPS;
  const float showCspFromCursor = SHOW_CSP_FROM_CURSOR;

  const float fontSize          = TEXT_SIZE;

  //get last UI values from the consolidated texture
  const float showNitsLast       = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  const float showCursorNitsLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

  const float showCspsLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  const float showCursorCspLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);

  const float fontSizeLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);

  if (showNitsLast       != showNitsValues
   || showCursorNitsLast != showNitsFromCrusor
   || showCspsLast       != showCsps
   || showCursorCspLast  != showCspFromCursor
   || fontSizeLast       != fontSize)
  {
    //store all current UI values
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW,  1.f);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, showNitsValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, showNitsFromCrusor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, showCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, showCspFromCursor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, fontSize);

    //calculate offset for the cursor nits text in the overlay
    float cursorNitsYOffset = (!SHOW_NITS_VALUES
                             ? -SHOW_NITS_VALUES_LINE_COUNT
                             : SPACING_MULTIPLIER)
                            + CSP_DESC_SPACING_MULTIPLIER;

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS,
               cursorNitsYOffset);


    //calculate offset for the colour spaces text in the overlay
    float cspsYOffset = ((!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR)
                       ? -(SHOW_NITS_VALUES_LINE_COUNT
                         - SPACING_MULTIPLIER)

                       : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR)
                       ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                         - SPACING_MULTIPLIER)

                       : (!SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR)
                       ? -(SHOW_NITS_VALUES_LINE_COUNT
                         + SHOW_NITS_FROM_CURSOR_LINE_COUNT)

                       : SPACING_MULTIPLIER * 2)
                      + CSP_DESC_SPACING_MULTIPLIER;

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS,
               cspsYOffset);


    //calculate offset for the cursorCSP text in the overlay
    float cursorCspYOffset = ((!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR  && SHOW_CSPS)
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR && SHOW_CSPS)
                            ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_NITS_VALUES  && SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (!SHOW_NITS_VALUES && !SHOW_NITS_FROM_CURSOR && SHOW_CSPS)
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              + SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              + SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR && !SHOW_CSPS)
                            ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              + SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (!SHOW_NITS_VALUES && !SHOW_NITS_FROM_CURSOR && !SHOW_CSPS)
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              + SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              + SHOW_CSPS_LINE_COUNT)

#if defined(IS_HDR10_LIKE_CSP)
                            : SPACING_MULTIPLIER * 3) - 2
#else //IS_HDR10_LIKE_CSP
                            : SPACING_MULTIPLIER * 3)
#endif //IS_HDR10_LIKE_CSP
                           + CSP_DESC_SPACING_MULTIPLIER;

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP,
               cursorCspYOffset);


    float4 bgCol = tex2Dfetch(StorageFontAtlasConsolidated, int2(0, 0)).rgba;

    uint activeLines = GetActiveLines();

    uint activeCharacters = GetActiveCharacters();

    static const uint charArrayEntry = GetCharArrayEntry();

    uint2 charSize = GetCharSize(charArrayEntry);

    uint outerSpacing = GetOuterSpacing(charSize.x);

    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);

    activeTextArea.y += uint(max(SHOW_NITS_VALUES
                               + SHOW_NITS_FROM_CURSOR
                               + SHOW_CSPS
                               + SHOW_CSP_FROM_CURSOR
                               - 1, 0)
                           * charSize.y * SPACING_MULTIPLIER);

    activeTextArea.y += charSize.y * CSP_DESC_SPACING_MULTIPLIER;

    activeTextArea += outerSpacing + outerSpacing;

    //draw active background for the overlay
    for (int y = 0; y < TEXTURE_OVERLAY_HEIGHT; y++)
    {
      for (int x = 0; x < TEXTURE_OVERLAY_WIDTH; x++)
      {
        int2 xy = int2(x, y);

        if (all(xy < activeTextArea))
        {
          tex2Dstore(StorageTextOverlay, xy, bgCol);
        }
        else
        {
          tex2Dstore(StorageTextOverlay, xy, float4(0.f, 0.f, 0.f, 0.f));
        }
      }
    }
  }

#ifdef _DEBUG

  else
  {
    for (uint x = 0; x < 20; x++)
    {
      for (uint y = 0; y < 20; y++)
      {
        tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(1.f, 1.f, 1.f, 1.f));
      }
    }
  }

#endif //_DEBUG
}


void DrawChar(uint Char, float2 DrawOffset)
{
  uint atlasEntry     = GetAtlasEntry();
  uint charArrayEntry = atlasEntry * 2;

  uint2 charSize     = uint2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);
  uint  atlasXOffset = AtlasXOffset[atlasEntry];
  uint2 charOffset   = uint2(atlasXOffset, Char * charSize.y);

  uint outerSpacing = GetOuterSpacing(charSize.x);

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      float4 pixel = tex2Dfetch(StorageFontAtlasConsolidated, charOffset + currentOffset).rgba;
      tex2Dstore(StorageTextOverlay, uint2(DrawOffset * charSize) + outerSpacing + currentOffset, pixel);
    }
  }
}


void DrawSpace(float2 DrawOffset)
{
  uint charArrayEntry = GetCharArrayEntry();

  uint2 charSize = uint2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);

  uint outerSpacing = GetOuterSpacing(charSize.x);

  float4 emptyPixel = tex2Dfetch(StorageFontAtlasConsolidated, int2(0, 0)).rgba;

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      tex2Dstore(StorageTextOverlay, uint2(DrawOffset * charSize) + outerSpacing + currentOffset, emptyPixel);
    }
  }
}


void CS_DrawTextToOverlay(uint3 ID : SV_DispatchThreadID)
{

  if (tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW) != 0.f)
  {

    static const float showNitsValuesYOffset0 =  1.f + CSP_DESC_SPACING_MULTIPLIER;
    static const float showNitsValuesYOffset1 =  2.f + CSP_DESC_SPACING_MULTIPLIER;
    static const float showNitsValuesYOffset2 =  3.f + CSP_DESC_SPACING_MULTIPLIER;
    static const float cursorNitsYOffset      =  4.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS);
    static const float cspsYOffset0           =  5.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS);
    static const float cspsYOffset1           =  1.f + cspsYOffset0;
    static const float cspsYOffset2           =  2.f + cspsYOffset0;
#ifdef IS_FLOAT_HDR_CSP
    static const float cspsYOffset3           =  3.f + cspsYOffset0;
    static const float cspsYOffset4           =  4.f + cspsYOffset0;
#endif
    static const float cursorCspYOffset       = 10.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP);

#ifdef _DEBUG

    for (uint x = 0; x < 20; x++)
    {
      for (uint y = 0; y < 20; y++)
      {
        if ((x < 5 || x > 14)
         || (y < 5 || y > 14))
        {
          tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(1.f, 1.f, 1.f, 1.f));
        }
        else
        {
          tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(0.f, 0.f, 0.f, 1.f));
        }
      }
    }

#endif //_DEBUG

    // draw header
#ifdef IS_HDR_CSP
    DrawChar(_H, float2( 0, 0));
    DrawChar(_D, float2( 1, 0));
    DrawChar(_R, float2( 2, 0));
#else
    DrawChar(_S, float2( 0, 0));
    DrawChar(_D, float2( 1, 0));
    DrawChar(_R, float2( 2, 0));
#endif
    DrawChar(_A, float2( 4, 0));
    DrawChar(_n, float2( 5, 0));
    DrawChar(_a, float2( 6, 0));
    DrawChar(_l, float2( 7, 0));
    DrawChar(_y, float2( 8, 0));
    DrawChar(_s, float2( 9, 0));
    DrawChar(_i, float2(10, 0));
    DrawChar(_s, float2(11, 0));

    DrawChar(_roundBracketOpen, float2(13, 0));
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    DrawChar(_s, float2(14, 0));
    DrawChar(_c, float2(15, 0));
    DrawChar(_R, float2(16, 0));
    DrawChar(_G, float2(17, 0));
    DrawChar(_B, float2(18, 0));
    DrawChar(_roundBracketClose, float2(19, 0));
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    DrawChar(_H, float2(14, 0));
    DrawChar(_D, float2(15, 0));
    DrawChar(_R, float2(16, 0));
    DrawChar(_1, float2(17, 0));
    DrawChar(_0, float2(18, 0));
    DrawChar(_roundBracketClose, float2(19, 0));
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
  #if (OVERWRITE_SDR_GAMMA == GAMMA_22 \
    || OVERWRITE_SDR_GAMMA == GAMMA_24)
    DrawChar(_g,   float2(14, 0));
    DrawChar(_a,   float2(15, 0));
    DrawChar(_m,   float2(16, 0));
    DrawChar(_m,   float2(17, 0));
    DrawChar(_a,   float2(18, 0));
    DrawChar(_2,   float2(20, 0));
    DrawChar(_dot, float2(21, 0));
    #if (OVERWRITE_SDR_GAMMA == GAMMA_22)
      DrawChar(_2, float2(22, 0));
    #elif (OVERWRITE_SDR_GAMMA == GAMMA_24)
      DrawChar(_4, float2(22, 0));
    #endif
    DrawChar(_roundBracketClose, float2(23, 0));
  #elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)
    DrawChar(_s, float2(14, 0));
    DrawChar(_R, float2(15, 0));
    DrawChar(_G, float2(16, 0));
    DrawChar(_B, float2(17, 0));
    DrawChar(_roundBracketClose, float2(18, 0));
  #else
    DrawChar(_u, float2(14, 0));
    DrawChar(_n, float2(15, 0));
    DrawChar(_k, float2(16, 0));
    DrawChar(_n, float2(17, 0));
    DrawChar(_o, float2(18, 0));
    DrawChar(_w, float2(20, 0));
    DrawChar(_n, float2(21, 0));
    DrawChar(_roundBracketClose, float2(22, 0));
  #endif
#endif

    // max/avg/min Nits
    if (SHOW_NITS_VALUES)
    {
      // maxNits:
      DrawChar(_m,     float2( 0, showNitsValuesYOffset0));
      DrawChar(_a,     float2( 1, showNitsValuesYOffset0));
      DrawChar(_x,     float2( 2, showNitsValuesYOffset0));
      DrawChar(_N,     float2( 3, showNitsValuesYOffset0));
      DrawChar(_i,     float2( 4, showNitsValuesYOffset0));
      DrawChar(_t,     float2( 5, showNitsValuesYOffset0));
      DrawChar(_s,     float2( 6, showNitsValuesYOffset0));
      DrawChar(_colon, float2( 7, showNitsValuesYOffset0));
      DrawChar(_dot,   float2(14, showNitsValuesYOffset0)); // five figure number
      // avgNits:
      DrawChar(_a,     float2( 0, showNitsValuesYOffset1));
      DrawChar(_v,     float2( 1, showNitsValuesYOffset1));
      DrawChar(_g,     float2( 2, showNitsValuesYOffset1));
      DrawChar(_N,     float2( 3, showNitsValuesYOffset1));
      DrawChar(_i,     float2( 4, showNitsValuesYOffset1));
      DrawChar(_t,     float2( 5, showNitsValuesYOffset1));
      DrawChar(_s,     float2( 6, showNitsValuesYOffset1));
      DrawChar(_colon, float2( 7, showNitsValuesYOffset1));
      DrawChar(_dot,   float2(14, showNitsValuesYOffset1)); // five figure number
      // minNits:
      DrawChar(_m,     float2( 0, showNitsValuesYOffset2));
      DrawChar(_i,     float2( 1, showNitsValuesYOffset2));
      DrawChar(_n,     float2( 2, showNitsValuesYOffset2));
      DrawChar(_N,     float2( 3, showNitsValuesYOffset2));
      DrawChar(_i,     float2( 4, showNitsValuesYOffset2));
      DrawChar(_t,     float2( 5, showNitsValuesYOffset2));
      DrawChar(_s,     float2( 6, showNitsValuesYOffset2));
      DrawChar(_colon, float2( 7, showNitsValuesYOffset2));
      DrawChar(_dot,   float2(14, showNitsValuesYOffset2)); // five figure number
    }

    // cursorNits:
    if (SHOW_NITS_FROM_CURSOR)
    {
      DrawChar(_c,     float2( 0, cursorNitsYOffset));
      DrawChar(_u,     float2( 1, cursorNitsYOffset));
      DrawChar(_r,     float2( 2, cursorNitsYOffset));
      DrawChar(_s,     float2( 3, cursorNitsYOffset));
      DrawChar(_o,     float2( 4, cursorNitsYOffset));
      DrawChar(_r,     float2( 5, cursorNitsYOffset));
      DrawChar(_N,     float2( 6, cursorNitsYOffset));
      DrawChar(_i,     float2( 7, cursorNitsYOffset));
      DrawChar(_t,     float2( 8, cursorNitsYOffset));
      DrawChar(_s,     float2( 9, cursorNitsYOffset));
      DrawChar(_colon, float2(10, cursorNitsYOffset));
      DrawChar(_dot,   float2(17, cursorNitsYOffset)); // five figure number
    }

    // CSPs
    if (SHOW_CSPS)
    {
      // BT.709:
      DrawChar(_B,       float2( 0, cspsYOffset0));
      DrawChar(_T,       float2( 1, cspsYOffset0));
      DrawChar(_dot,     float2( 2, cspsYOffset0));
      DrawChar(_7,       float2( 3, cspsYOffset0));
      DrawChar(_0,       float2( 4, cspsYOffset0));
      DrawChar(_9,       float2( 5, cspsYOffset0));
      DrawChar(_colon,   float2( 6, cspsYOffset0));
      DrawChar(_dot,     float2(12, cspsYOffset0));
      DrawChar(_percent, float2(15, cspsYOffset0));
      // DCI-P3:
      DrawChar(_D,       float2( 0, cspsYOffset1));
      DrawChar(_C,       float2( 1, cspsYOffset1));
      DrawChar(_I,       float2( 2, cspsYOffset1));
      DrawChar(_minus,   float2( 3, cspsYOffset1));
      DrawChar(_P,       float2( 4, cspsYOffset1));
      DrawChar(_3,       float2( 5, cspsYOffset1));
      DrawChar(_colon,   float2( 6, cspsYOffset1));
      DrawChar(_dot,     float2(12, cspsYOffset1));
      DrawChar(_percent, float2(15, cspsYOffset1));
      // BT.2020:
      DrawChar(_B,       float2( 0, cspsYOffset2));
      DrawChar(_T,       float2( 1, cspsYOffset2));
      DrawChar(_dot,     float2( 2, cspsYOffset2));
      DrawChar(_2,       float2( 3, cspsYOffset2));
      DrawChar(_0,       float2( 4, cspsYOffset2));
      DrawChar(_2,       float2( 5, cspsYOffset2));
      DrawChar(_0,       float2( 6, cspsYOffset2));
      DrawChar(_colon,   float2( 7, cspsYOffset2));
      DrawChar(_dot,     float2(12, cspsYOffset2));
      DrawChar(_percent, float2(15, cspsYOffset2));
#ifdef IS_FLOAT_HDR_CSP
      // AP0:
      DrawChar(_A,       float2( 0, cspsYOffset3));
      DrawChar(_P,       float2( 1, cspsYOffset3));
      DrawChar(_0,       float2( 2, cspsYOffset3));
      DrawChar(_colon,   float2( 3, cspsYOffset3));
      DrawChar(_dot,     float2(12, cspsYOffset3));
      DrawChar(_percent, float2(15, cspsYOffset3));
      // invalid:
      DrawChar(_i,       float2( 0, cspsYOffset4));
      DrawChar(_n,       float2( 1, cspsYOffset4));
      DrawChar(_v,       float2( 2, cspsYOffset4));
      DrawChar(_a,       float2( 3, cspsYOffset4));
      DrawChar(_l,       float2( 4, cspsYOffset4));
      DrawChar(_i,       float2( 5, cspsYOffset4));
      DrawChar(_d,       float2( 6, cspsYOffset4));
      DrawChar(_colon,   float2( 7, cspsYOffset4));
      DrawChar(_dot,     float2(12, cspsYOffset4));
      DrawChar(_percent, float2(15, cspsYOffset4));
#endif //IS_FLOAT_HDR_CSP
    }

    // cursorCSP:
    if (SHOW_CSP_FROM_CURSOR)
    {
      DrawChar(_c,     float2(0, cursorCspYOffset));
      DrawChar(_u,     float2(1, cursorCspYOffset));
      DrawChar(_r,     float2(2, cursorCspYOffset));
      DrawChar(_s,     float2(3, cursorCspYOffset));
      DrawChar(_o,     float2(4, cursorCspYOffset));
      DrawChar(_r,     float2(5, cursorCspYOffset));
      DrawChar(_C,     float2(6, cursorCspYOffset));
      DrawChar(_S,     float2(7, cursorCspYOffset));
      DrawChar(_P,     float2(8, cursorCspYOffset));
      DrawChar(_colon, float2(9, cursorCspYOffset));
    }

    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW, 0.f);
  }
  return;
}


//extract all digits without causing float issues
#define _6th(Val) Val / 100000.f
#define _5th(Val) Val /  10000.f
#define _4th(Val) Val /   1000.f
#define _3rd(Val) Val /    100.f
#define _2nd(Val) Val /     10.f
#define _1st(Val) Val %     10.f

#define d1st(Val) Val % 1.f *     10.f
#define d2nd(Val) Val % 1.f *    100.f % 10.f
#define d3rd(Val) Val % 1.f *   1000.f % 10.f
#define d4th(Val) Val % 1.f *  10000.f % 10.f
#define d5th(Val) Val % 1.f * 100000.f % 10.f
#define d6th(Val) Val % 1.f * 100000.f %  1.f  *  10.f
#define d7th(Val) Val % 1.f * 100000.f %  0.1f * 100.f


void DrawNumberAboveZero(precise uint CurNumber, float2 Offset)
{
  if (CurNumber > 0)
  {
    DrawChar(CurNumber % 10, Offset);
  }
  else
  {
    DrawSpace(Offset);
  }
}


#define showNitsValuesYOffset0  1.f + CSP_DESC_SPACING_MULTIPLIER
#define showNitsValuesYOffset1  2.f + CSP_DESC_SPACING_MULTIPLIER
#define showNitsValuesYOffset2  3.f + CSP_DESC_SPACING_MULTIPLIER
#define cursorNitsYOffset0      4.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS)
#define cspsYOffset0            5.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsYOffset1            6.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsYOffset2            7.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsYOffset3            8.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsYOffset4            9.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cursorCspYOffset       10.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP)

void CS_DrawValuesToOverlay(uint3 ID : SV_DispatchThreadID)
{
  switch(ID.x)
  {
    // max/avg/min Nits
    // maxNits:
    case 0:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _5th(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(9, showNitsValuesYOffset0));
      }
      return;
    }
    case 1:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _4th(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, showNitsValuesYOffset0));
      }
      return;
    }
    case 2:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _3rd(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, showNitsValuesYOffset0));
      }
      return;
    }
    case 3:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _2nd(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, showNitsValuesYOffset0));
      }
      return;
    }
    case 4:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _1st(maxNitsShow);
        DrawChar(curNumber, float2(13, showNitsValuesYOffset0));
      }
      return;
    }
    case 5:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d1st(maxNitsShow);
        DrawChar(curNumber, float2(15, showNitsValuesYOffset0));
      }
      return;
    }
    case 6:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d2nd(maxNitsShow);
        DrawChar(curNumber, float2(16, showNitsValuesYOffset0));
      }
      return;
    }
    case 7:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d3rd(maxNitsShow);
        DrawChar(curNumber, float2(17, showNitsValuesYOffset0));
      }
      return;
    }
    // avgNits:
    case 8:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _5th(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(9, showNitsValuesYOffset1));
      }
      return;
    }
    case 9:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _4th(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, showNitsValuesYOffset1));
      }
      return;
    }
    case 10:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _3rd(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, showNitsValuesYOffset1));
      }
      return;
    }
    case 11:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _2nd(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, showNitsValuesYOffset1));
      }
      return;
    }
    case 12:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _1st(avgNitsShow);
        DrawChar(curNumber, float2(13, showNitsValuesYOffset1));
      }
      return;
    }
    case 13:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d1st(avgNitsShow);
        DrawChar(curNumber, float2(15, showNitsValuesYOffset1));
      }
      return;
    }
    case 14:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d2nd(avgNitsShow);
        DrawChar(curNumber, float2(16, showNitsValuesYOffset1));
      }
      return;
    }
    case 15:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d2nd(avgNitsShow);
        DrawChar(curNumber, float2(17, showNitsValuesYOffset1));
      }
      return;
    }
    // minNits:
    case 16:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _5th(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(9, showNitsValuesYOffset2));
      }
      return;
    }
    case 17:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _4th(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, showNitsValuesYOffset2));
      }
      return;
    }
    case 18:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _3rd(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, showNitsValuesYOffset2));
      }
      return;
    }
    case 19:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _2nd(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, showNitsValuesYOffset2));
      }
      return;
    }
    case 20:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _1st(minNitsShow);
        DrawChar(curNumber, float2(13, showNitsValuesYOffset2));
      }
      return;
    }
    case 21:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d1st(minNitsShow);
        DrawChar(curNumber, float2(15, showNitsValuesYOffset2));
      }
      return;
    }
    case 22:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d2nd(minNitsShow);
        DrawChar(curNumber, float2(16, showNitsValuesYOffset2));
      }
      return;
    }
    case 23:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d3rd(minNitsShow);
        DrawChar(curNumber, float2(17, showNitsValuesYOffset2));
      }
      return;
    }
    case 24:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d4th(minNitsShow);
        DrawChar(curNumber, float2(18, showNitsValuesYOffset2));
      }
      return;
    }
    case 25:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d5th(minNitsShow);
        DrawChar(curNumber, float2(19, showNitsValuesYOffset2));
      }
      return;
    }
    case 26:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d6th(minNitsShow);
        DrawChar(curNumber, float2(20, showNitsValuesYOffset2));
      }
      return;
    }
    // cursorNits:
    case 27:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _5th(cursorNits);
        DrawNumberAboveZero(curNumber, float2(12, cursorNitsYOffset0));
      }
      return;
    }
    case 28:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _4th(cursorNits);
        DrawNumberAboveZero(curNumber, float2(13, cursorNitsYOffset0));
      }
      return;
    }
    case 29:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _3rd(cursorNits);
        DrawNumberAboveZero(curNumber, float2(14, cursorNitsYOffset0));
      }
      return;
    }
    case 30:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _2nd(cursorNits);
        DrawNumberAboveZero(curNumber, float2(15, cursorNitsYOffset0));
      }
      return;
    }
    case 31:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _1st(cursorNits);
        DrawChar(curNumber, float2(16, cursorNitsYOffset0));
      }
      return;
    }
    case 32:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d1st(cursorNits);
        DrawChar(curNumber, float2(18, cursorNitsYOffset0));
      }
      return;
    }
    case 33:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d2nd(cursorNits);
        DrawChar(curNumber, float2(19, cursorNitsYOffset0));
      }
      return;
    }
    case 34:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d3rd(cursorNits);
        DrawChar(curNumber, float2(20, cursorNitsYOffset0));
      }
      return;
    }
    case 35:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber = d4th(cursorNits);
        DrawChar(curNumber, float2(21, cursorNitsYOffset0));
      }
      return;
    }
    case 36:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d5th(cursorNits);
        DrawChar(curNumber, float2(22, cursorNitsYOffset0));
      }
      return;
    }
    case 37:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d6th(cursorNits);
        DrawChar(curNumber, float2(23, cursorNitsYOffset0));
      }
      return;
    }


    // show CSPs
    // BT.709:
    case 38:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _3rd(precentageBt709);
        DrawNumberAboveZero(curNumber, float2(9, cspsYOffset0));
      }
      return;
    }
    case 39:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _2nd(precentageBt709);
        DrawNumberAboveZero(curNumber, float2(10, cspsYOffset0));
      }
      return;
    }
    case 40:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _1st(precentageBt709);
        DrawChar(curNumber, float2(11, cspsYOffset0));
      }
      return;
    }
    case 41:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d1st(precentageBt709);
        DrawChar(curNumber, float2(13, cspsYOffset0));
      }
      return;
    }
    case 42:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d2nd(precentageBt709);
        DrawChar(curNumber, float2(14, cspsYOffset0));
      }
      return;
    }
    // DCI-P3:
    case 43:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _3rd(precentageDciP3);
        DrawNumberAboveZero(curNumber, float2(9, cspsYOffset1));
      }
      return;
    }
    case 44:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _2nd(precentageDciP3);
        DrawNumberAboveZero(curNumber, float2(10, cspsYOffset1));
      }
      return;
    }
    case 45:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _1st(precentageDciP3);
        DrawChar(curNumber, float2(11, cspsYOffset1));
      }
      return;
    }
    case 46:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d1st(precentageDciP3);
        DrawChar(curNumber, float2(13, cspsYOffset1));
      }
      return;
    }
    case 47:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d2nd(precentageDciP3);
        DrawChar(curNumber, float2(14, cspsYOffset1));
      }
      return;
    }
    // BT.2020:
    case 48:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _3rd(precentageBt2020);
        DrawNumberAboveZero(curNumber, float2(9, cspsYOffset2));
      }
      return;
    }
    case 49:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _2nd(precentageBt2020);
        DrawNumberAboveZero(curNumber, float2(10, cspsYOffset2));
      }
      return;
    }
    case 50:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _1st(precentageBt2020);
        DrawChar(curNumber, float2(11, cspsYOffset2));
      }
      return;
    }
    case 51:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d1st(precentageBt2020);
        DrawChar(curNumber, float2(13, cspsYOffset2));
      }
      return;
    }
    case 52:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d2nd(precentageBt2020);
        DrawChar(curNumber, float2(14, cspsYOffset2));
      }
      return;
    }
#ifdef IS_FLOAT_HDR_CSP
    // AP0:
    case 53:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _3rd(precentageAp0);
        DrawNumberAboveZero(curNumber, float2(9, cspsYOffset3));
      }
      return;
    }
    case 54:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _2nd(precentageAp0);
        DrawNumberAboveZero(curNumber, float2(10, cspsYOffset3));
      }
      return;
    }
    case 55:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _1st(precentageAp0);
        DrawChar(curNumber, float2(11, cspsYOffset3));
      }
      return;
    }
    case 56:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d1st(precentageAp0);
        DrawChar(curNumber, float2(13, cspsYOffset3));
      }
      return;
    }
    case 57:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d2nd(precentageAp0);
        DrawChar(curNumber, float2(14, cspsYOffset3));
      }
      return;
    }
    // invalid:
    case 58:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _3rd(precentageInvalid);
        DrawNumberAboveZero(curNumber, float2(9, cspsYOffset4));
      }
      return;
    }
    case 59:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _2nd(precentageInvalid);
        DrawNumberAboveZero(curNumber, float2(10, cspsYOffset4));
      }
      return;
    }
    case 60:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _1st(precentageInvalid);
        DrawChar(curNumber, float2(11, cspsYOffset4));
      }
      return;
    }
    case 61:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d1st(precentageInvalid);
        DrawChar(curNumber, float2(13, cspsYOffset4));
      }
      return;
    }
    case 62:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d2nd(precentageInvalid);
        DrawChar(curNumber, float2(14, cspsYOffset4));
      }
      return;
    }
#endif
    //cursorCSP:
    case 63:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(11, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_B, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_D, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_B, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_A, currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_i, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 64:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(12, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_T, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_C, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_T, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_P, currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_n, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 65:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(13, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_dot, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_I, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_dot, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_v, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 66:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(14, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_7, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_minus, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_a, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 67:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(15, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_P, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_l, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 68:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(16, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_9, currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_3, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_i, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    case 69:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(17, cursorCspYOffset);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
        else //invalid
        {
          DrawChar(_d, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
    default:
      return;
  }
  return;
}

#undef showNitsValuesYOffset0
#undef showNitsValuesYOffset1
#undef showNitsValuesYOffset2
#undef cursorNitsYOffset0
#undef cspsYOffset0
#undef cspsYOffset1
#undef cspsYOffset2
#undef cspsYOffset3
#undef cspsYOffset4
#undef cursorCspYOffset


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareSetActiveArea(
  in                  uint   Id                : SV_VertexID,
#if (defined(API_IS_D3D11) \
  || defined(API_IS_D3D12))
  out                 float4 VPos              : SV_Position,
#else
  out                 float2 VPos              : SV_Position,
#endif
  out nointerpolation float4 PercentagesToCrop : PercentagesToCrop)
{
  float2 TexCoord;
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
#if (defined(API_IS_D3D11) \
  || defined(API_IS_D3D12))
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);
#else
  VPos = TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f);
#endif


#define percentageToCropFromLeft   PercentagesToCrop.x
#define percentageToCropFromTop    PercentagesToCrop.y
#define percentageToCropFromRight  PercentagesToCrop.z
#define percentageToCropFromBottom PercentagesToCrop.w

  float fractionCropLeft   = ACTIVE_AREA_CROP_LEFT   / 100.f;
  float fractionCropTop    = ACTIVE_AREA_CROP_TOP    / 100.f;
  float fractionCropRight  = ACTIVE_AREA_CROP_RIGHT  / 100.f;
  float fractionCropBottom = ACTIVE_AREA_CROP_BOTTOM / 100.f;

  percentageToCropFromLeft   =                 fractionCropLeft   * BUFFER_WIDTH;
  percentageToCropFromTop    =                 fractionCropTop    * BUFFER_HEIGHT;
  percentageToCropFromRight  = BUFFER_WIDTH  - fractionCropRight  * BUFFER_WIDTH;
  percentageToCropFromBottom = BUFFER_HEIGHT - fractionCropBottom * BUFFER_HEIGHT;

}

void PS_SetActiveArea(
#if (defined(API_IS_D3D11) \
  || defined(API_IS_D3D12))
  in                  float4 VPos              : SV_Position,
#else
  in                  float2 VPos              : SV_Position,
#endif
  in  nointerpolation float4 PercentagesToCrop : PercentagesToCrop,
  out                 float4 Output            : SV_Target0)
{
  Output = 0.f;

  if (ACTIVE_AREA_ENABLE)
  {
    if (VPos.x > percentageToCropFromLeft
     && VPos.y > percentageToCropFromTop
     && VPos.x < percentageToCropFromRight
     && VPos.y < percentageToCropFromBottom)
    {
      discard;
    }
    else
    {
      Output = float4(0.f, 0.f, 0.f, 1.f);
      return;
    }
  }

  discard;
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrAnalysis(
  in                  uint   Id                : SV_VertexID,
  out                 float4 VPos              : SV_Position,
  out                 float2 TexCoord          : TEXCOORD0,
  out nointerpolation bool2  PingPongChecks    : PingPongChecks,
  out nointerpolation float4 HighlightNitRange : HighlightNitRange,
#ifndef GAMESCOPE
  out nointerpolation int4   TextureDisplaySizes : TextureDisplaySizes,
#else
  out nointerpolation int2   CurrentActiveOverlayArea                 : CurrentActiveOverlayArea,
  out nointerpolation int2   LuminanceWaveformTextureDisplayAreaBegin : LuminanceWaveformTextureDisplayAreaBegin,
#endif
  out nointerpolation float2 CieDiagramTextureActiveSize  : CieDiagramTextureActiveSize,
  out nointerpolation float2 CieDiagramTextureDisplaySize : CieDiagramTextureDisplaySize)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#define pingpong0Above1   PingPongChecks[0]
#define breathingIsActive PingPongChecks[1]

#define highlightNitRangeOut HighlightNitRange.rgb
#define breathing            HighlightNitRange.w

#ifndef GAMESCOPE
  #define CurrentActiveOverlayArea                 TextureDisplaySizes.xy
  #define LuminanceWaveformTextureDisplayAreaBegin TextureDisplaySizes.zw
#endif

  pingpong0Above1                          = false;
  breathingIsActive                        = false;
  HighlightNitRange                        = 0.f;
  CurrentActiveOverlayArea                 = 0;
  LuminanceWaveformTextureDisplayAreaBegin = 0;
  CieDiagramTextureActiveSize              = 0.f;
  CieDiagramTextureDisplaySize             = 0.f;

  if (HIGHLIGHT_NIT_RANGE)
  {
    float pingpong0 = NIT_PINGPONG0.x + 0.5f;

    pingpong0Above1 = pingpong0 >= 1.f;

    if (pingpong0Above1)
    {
      breathing = saturate(pingpong0 - 1.f);
      //breathing = 1.f;

      breathingIsActive = breathing > 0.f;

      float pingpong1 = NIT_PINGPONG1.y == 1 ?       NIT_PINGPONG1.x
                                             : 6.f - NIT_PINGPONG1.x;

      if (pingpong1 <= 1.f)
      {
        highlightNitRangeOut = float3(1.f, NIT_PINGPONG2.x, 0.f);
      }
      else if (pingpong1 <= 2.f)
      {
        highlightNitRangeOut = float3(NIT_PINGPONG2.x, 1.f, 0.f);
      }
      else if (pingpong1 <= 3.f)
      {
        highlightNitRangeOut = float3(0.f, 1.f, NIT_PINGPONG2.x);
      }
      else if (pingpong1 <= 4.f)
      {
        highlightNitRangeOut = float3(0.f, NIT_PINGPONG2.x, 1.f);
      }
      else if (pingpong1 <= 5.f)
      {
        highlightNitRangeOut = float3(NIT_PINGPONG2.x, 0.f, 1.f);
      }
      else //if (pingpong1 <= 6.f)
      {
        highlightNitRangeOut = float3(1.f, 0.f, NIT_PINGPONG2.x);
      }

      highlightNitRangeOut *= breathing;

      highlightNitRangeOut = MapBt709IntoCurrentCsp(highlightNitRangeOut, HIGHLIGHT_NIT_RANGE_BRIGHTNESS);
    }
  }

  if (SHOW_LUMINANCE_WAVEFORM)
  {
    LuminanceWaveformTextureDisplayAreaBegin = int2(BUFFER_WIDTH, BUFFER_HEIGHT) - Waveform::GetActiveArea();
  }

  if (SHOW_CIE)
  {
    float cieDiagramSizeFrac = CIE_DIAGRAM_SIZE / 100.f;

    CieDiagramTextureActiveSize =
      round(float2(CIE_BG_WIDTH[CIE_DIAGRAM_TYPE], CIE_BG_HEIGHT[CIE_DIAGRAM_TYPE]) * cieDiagramSizeFrac);

    CieDiagramTextureActiveSize.y = float(BUFFER_HEIGHT) - CieDiagramTextureActiveSize.y;

    CieDiagramTextureDisplaySize =
      float2(CIE_1931_BG_WIDTH, CIE_1931_BG_HEIGHT) * cieDiagramSizeFrac;
  }

  {
    uint activeLines = GetActiveLines();

    uint activeCharacters = GetActiveCharacters();

    static const uint charArrayEntry = GetCharArrayEntry();

    uint2 charSize = GetCharSize(charArrayEntry);

    uint outerSpacing = GetOuterSpacing(charSize.x);

    uint2 currentOverlayDimensions = charSize
                                   * uint2(activeCharacters, activeLines);

    currentOverlayDimensions.y += uint(max(SHOW_NITS_VALUES
                                         + SHOW_NITS_FROM_CURSOR
                                         + SHOW_CSPS
                                         + SHOW_CSP_FROM_CURSOR
                                         - 1, 0)
                                     * charSize.y * SPACING_MULTIPLIER);

    if (SHOW_NITS_VALUES
     || SHOW_NITS_FROM_CURSOR
     || SHOW_CSPS
     || SHOW_CSP_FROM_CURSOR)
    {
      currentOverlayDimensions.y += charSize.y * CSP_DESC_SPACING_MULTIPLIER;
    }

    currentOverlayDimensions += outerSpacing + outerSpacing;

    CurrentActiveOverlayArea = int2(currentOverlayDimensions);

    if (TEXT_POSITION == TEXT_POSITION_TOP_RIGHT)
    {
      CurrentActiveOverlayArea.x = int(BUFFER_WIDTH) - CurrentActiveOverlayArea.x;
    }
  }

}


void ExtendedReinhardTmo(
  inout float3 Colour,
  in    float  WhitePoint)
{
  float maxWhite = 10000.f / WhitePoint;

  Colour = (Colour * (1.f + (Colour / (maxWhite * maxWhite))))
         / (1.f + Colour);
}

void MergeOverlay(
  inout float3 Output,
  in    float3 Overlay,
  in    float  OverlayBrightness,
  in    float  Alpha)
{
  Overlay = Csp::Mat::Bt709To::Bt2020(Overlay);

  // tone map pixels below the overlay area
  //
  // first set 1.0 to be equal to OverlayBrightness
  float adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  adjustFactor = OverlayBrightness / 80.f;

  Output.rgb = Csp::Mat::Bt709To::Bt2020(Output.rgb / adjustFactor);

  // safety clamp colours outside of BT.2020
  Output.rgb = max(Output.rgb, 0.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  adjustFactor = OverlayBrightness / 10000.f;

  Output =  Csp::Trc::PqTo::Linear(Output);
  Output /= adjustFactor;

#endif

  // then tone map to 1.0 at max
  ExtendedReinhardTmo(Output, OverlayBrightness);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  // safety clamp for the case that there are values that represent above 10000 nits
  Output.rgb = min(Output.rgb, 1.f);

#endif

  // apply the overlay
  Output = lerp(Output, Overlay, Alpha);

  // map everything back to the used colour space
  Output *= adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Output = Csp::Mat::Bt2020To::Bt709(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Output = Csp::Trc::LinearTo::Pq(Output);
#endif
}


void PS_HdrAnalysis(
  in                  float4 VPos              : SV_Position,
  in                  float2 TexCoord          : TEXCOORD0,
  in  nointerpolation bool2  PingPongChecks    : PingPongChecks,
  in  nointerpolation float4 HighlightNitRange : HighlightNitRange,
#ifndef GAMESCOPE
  in  nointerpolation int4   TextureDisplaySizes : TextureDisplaySizes,
#else
  in  nointerpolation int2   CurrentActiveOverlayArea                 : CurrentActiveOverlayArea,
  in  nointerpolation int2   LuminanceWaveformTextureDisplayAreaBegin : LuminanceWaveformTextureDisplayAreaBegin,
#endif
  in  nointerpolation float2 CieDiagramTextureActiveSize  : CieDiagramTextureActiveSize,
  in  nointerpolation float2 CieDiagramTextureDisplaySize : CieDiagramTextureDisplaySize,
  out                 float4 Output                       : SV_Target0)
{
  const int2 pureCoordAsInt = int2(VPos.xy);

  Output = tex2Dfetch(ReShade::BackBuffer, pureCoordAsInt);

  if (SHOW_CSP_MAP
   || SHOW_HEATMAP
   || HIGHLIGHT_NIT_RANGE
   || DRAW_ABOVE_NITS_AS_BLACK
   || DRAW_BELOW_NITS_AS_BLACK)
  {
    const float pixelNits = tex2Dfetch(SamplerNitsValues, pureCoordAsInt).r;

    if (SHOW_CSP_MAP)
    {
      Output.rgb = CreateCspMap(tex2Dfetch(SamplerCsps, pureCoordAsInt).x * 255.f, pixelNits);
    }

    if (SHOW_HEATMAP)
    {
      Output.rgb = HeatmapRgbValues(pixelNits,
                                    HEATMAP_CUTOFF_POINT,
                                    false);

      Output.rgb = MapBt709IntoCurrentCsp(Output.rgb, HEATMAP_BRIGHTNESS);
    }

    if (HIGHLIGHT_NIT_RANGE
     && pixelNits >= HIGHLIGHT_NIT_RANGE_START_POINT
     && pixelNits <= HIGHLIGHT_NIT_RANGE_END_POINT
     && pingpong0Above1
     && breathingIsActive)
    {
      //Output.rgb = HighlightNitRangeOut;
      Output.rgb = lerp(Output.rgb, highlightNitRangeOut, breathing);
    }

    if (DRAW_ABOVE_NITS_AS_BLACK)
    {
      if (pixelNits > ABOVE_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }

    if (DRAW_BELOW_NITS_AS_BLACK)
    {
      if (pixelNits < BELOW_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }
  }

  if (SHOW_CIE)
  {
    // draw the diagram in the bottom left corner
    if (VPos.x <  CieDiagramTextureActiveSize.x
     && VPos.y >= CieDiagramTextureActiveSize.y)
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(VPos.x,
                                           VPos.y - CieDiagramTextureActiveSize.y);

      float2 currentCieSamplerCoords = currentSamplerCoords / CieDiagramTextureDisplaySize;

      float3 currentPixelToDisplay = pow(tex2D(SamplerCieCurrent, currentCieSamplerCoords).rgb, 2.2f);

      float alpha = min(ceil(MAXRGB(currentPixelToDisplay)) + CIE_DIAGRAM_ALPHA / 100.f, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay,
                   CIE_DIAGRAM_BRIGHTNESS,
                   alpha);
    }
  }

  if (SHOW_LUMINANCE_WAVEFORM)
  {
    // draw the waveform in the bottom right corner
    if (all(pureCoordAsInt.xy >= LuminanceWaveformTextureDisplayAreaBegin))
    {
      // get fetch coords
      int2 currentFetchCoords = pureCoordAsInt.xy - LuminanceWaveformTextureDisplayAreaBegin;

      float4 currentPixelToDisplay =
        tex2Dfetch(SamplerLuminanceWaveformFinal, currentFetchCoords);

      float alpha = min(LUMINANCE_WAVEFORM_ALPHA / 100.f + currentPixelToDisplay.a, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay.rgb,
                   LUMINANCE_WAVEFORM_BRIGHTNESS,
                   alpha);
    }
  }

  {
    if (TEXT_POSITION == TEXT_POSITION_TOP_LEFT)
    {
      if (all(pureCoordAsInt <= CurrentActiveOverlayArea))
      {
        float4 overlay = tex2Dfetch(SamplerTextOverlay, pureCoordAsInt);

        float alpha = min(TEXT_BG_ALPHA / 100.f + overlay.a, 1.f);

        MergeOverlay(Output.rgb,
                     overlay.rgb,
                     TEXT_BRIGHTNESS,
                     alpha);
      }
    }
    else
    {
      if (pureCoordAsInt.x >= CurrentActiveOverlayArea.x
       && pureCoordAsInt.y <= CurrentActiveOverlayArea.y)
      {
        float4 overlay = tex2Dfetch(SamplerTextOverlay,
                                    int2(pureCoordAsInt.x - CurrentActiveOverlayArea.x, pureCoordAsInt.y));

        float alpha = min(TEXT_BG_ALPHA / 100.f + overlay.a, 1.f);

        MergeOverlay(Output.rgb,
                     overlay.rgb,
                     TEXT_BRIGHTNESS,
                     alpha);
      }
    }
  }

}

//technique lilium__HDR_analysis_CLL_OLD
//<
//  enabled = false;
//>
//{
//  pass PS_CalcNitsPerPixel
//  {
//    VertexShader = VS_PostProcess;
//     PixelShader = PS_CalcNitsPerPixel;
//    RenderTarget = TextureNitsValues;
//  }
//
//  pass CS_GetMaxAvgMinCll0
//  {
//    ComputeShader = CS_GetMaxAvgMinCll0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMaxAvgMinCll1
//  {
//    ComputeShader = CS_GetMaxAvgMinCll1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMaxCll0
//  {
//    ComputeShader = CS_GetMaxCll0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMaxCll1
//  {
//    ComputeShader = CS_GetMaxCll1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetAvgCll0
//  {
//    ComputeShader = CS_GetAvgCll0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetAvgCll1
//  {
//    ComputeShader = CS_GetAvgCll1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMinCll0
//  {
//    ComputeShader = CS_GetMinCll0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMinCll1
//  {
//    ComputeShader = CS_GetMinCll1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//}

#ifdef _TESTY
technique lilium__hdr_analysis_TESTY
<
  ui_label = "Lilium's TESTY";
>
{
  pass PS_Testy
  {
    VertexShader       = VS_Testy;
     PixelShader       = PS_Testy;
    ClearRenderTargets = true;
  }
}
#endif //_TESTY

void CS_MakeOverlayBgAndWaveformScaleRedraw()
{
  tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, 3.f);
  tex2Dstore(StorageConsolidated, COORDS_LUMINANCE_WAVEFORM_LAST_SIZE_X, 0.f);
  return;
}

technique lilium__make_overlay_bg_redraw
<
  enabled = true;
  hidden  = true;
  timeout = 1;
>
{
  pass CS_MakeOverlayBgAndWaveformScaleRedraw
  {
    ComputeShader = CS_MakeOverlayBgAndWaveformScaleRedraw <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique lilium__hdr_analysis
<
  ui_label = "Lilium's HDR analysis";
>
{

//Active Area
  pass PS_SetActiveArea
  {
    VertexShader = VS_PrepareSetActiveArea;
     PixelShader = PS_SetActiveArea;
  }


//Luminance Values
  pass PS_CalcNitsPerPixel
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CalcNitsPerPixel;
    RenderTarget = TextureNitsValues;
  }

  pass CS_GetMaxAvgMinNits0_NEW
  {
    ComputeShader = CS_GetMaxAvgMinNits0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass CS_GetMaxAvgMinNits1_NEW
  {
    ComputeShader = CS_GetMaxAvgMinNits1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass CS_GetFinalMaxAvgMinNits_NEW
  {
    ComputeShader = CS_GetFinalMaxAvgMinNits_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass CS_RenderLuminanceWaveformScale
  {
    ComputeShader = CS_RenderLuminanceWaveformScale <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass PS_ClearLuminanceWaveformTexture
  {
    VertexShader       = VS_PostProcess;
     PixelShader       = PS_ClearLuminanceWaveformTexture;
    RenderTarget       = TextureLuminanceWaveform;
    ClearRenderTargets = true;
  }

  pass CS_RenderLuminanceWaveform
  {
    ComputeShader = CS_RenderLuminanceWaveform <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass PS_RenderLuminanceWaveformToScale
  {
    VertexShader = VS_PrepareRenderLuminanceWaveformToScale;
     PixelShader = PS_RenderLuminanceWaveformToScale;
    RenderTarget = TextureLuminanceWaveformFinal;
  }


//CIE
  pass PS_CopyCieBgAndOutlines
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CopyCieBgAndOutlines;
    RenderTarget = TextureCieCurrent;
  }

  pass CS_GenerateCieDiagram
  {
    ComputeShader = CS_GenerateCieDiagram <THREAD_SIZE1, THREAD_SIZE1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = DISPATCH_Y1;
  }


//CSP
  pass PS_CalcCsps
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CalcCsps;
    RenderTarget = TextureCsps;
  }

  pass CS_CountCspsY
  {
    ComputeShader = CS_CountCspsY <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass CS_CountCspsX
  {
    ComputeShader = CS_CountCspsX <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }


  pass CS_CopyShowValues
  {
    ComputeShader = ShowValuesCopy <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }


  pass CS_PrepareOverlay
  {
    ComputeShader = CS_PrepareOverlay <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass CS_DrawTextToOverlay
  {
    ComputeShader = CS_DrawTextToOverlay <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass CS_DrawValuesToOverlay
  {
    ComputeShader = CS_DrawValuesToOverlay <1, 1>;
    DispatchSizeX = 70;
    DispatchSizeY = 1;
  }

  pass PS_HdrAnalysis
  {
    VertexShader = VS_PrepareHdrAnalysis;
     PixelShader = PS_HdrAnalysis;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__hdr_analysis
<
  ui_label = "Lilium's HDR analysis (ERROR)";
>
CS_ERROR

#endif //is hdr API and hdr colour space
