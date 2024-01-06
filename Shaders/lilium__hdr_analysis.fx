#include "lilium__include/colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_POSSIBLE_HDR_CSP))


#undef TEXT_BRIGHTNESS

#ifndef GAMESCOPE
  //#define _DEBUG
  //#define _TESTY
#endif

#if (BUFFER_WIDTH  >= 2560) \
 && (BUFFER_HEIGHT >= 1440)
  #define IS_QHD_OR_HIGHER_RES
#endif


#if defined(IS_POSSIBLE_HDR_CSP)
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
  max       = 1.75f;
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

#ifndef GAMESCOPE
uniform int GLOBAL_SPACER_0
<
  ui_category = "Global";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif


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

#ifndef GAMESCOPE
uniform int ACTIVE_AREA_SPACER_0
<
  ui_category = "Set Active Area for analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif


// Nit Values
uniform bool SHOW_NITS_VALUES
<
  ui_category = "Luminance analysis";
  ui_label    = "show luminance values";
  ui_tooltip  = "Shows max/avg/min Luminance Levels.";
> = true;

uniform bool SHOW_NITS_FROM_CURSOR
<
  ui_category = "Luminance analysis";
  ui_label    = "show luminance value from cursor position";
> = true;

#ifndef GAMESCOPE
uniform int NITS_SPACER_0
<
  ui_category = "Luminance analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif


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

#ifndef GAMESCOPE
uniform int CSP_SPACER_0
<
  ui_category = "Colour Space analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int CIE_SPACER_0
<
  ui_category = "CIE diagram visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int CIE_SPACER_1
<
  ui_category = "CIE diagram visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;
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

#ifndef GAMESCOPE
uniform int HEATMAP_SPACER_0
<
  ui_category = "Heatmap visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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
                                                       / TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT
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

#ifndef GAMESCOPE
uniform int LUMINANCE_WAVEFORM_SPACER_0
<
  ui_category = "Luminance waveform";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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
  ui_label    = "show max nits line";
  ui_tooltip  = "Show a horizontal line where the maximum nits is on the waveform."
           "\n" "The line is invisible when the maximum nits hits above 10000 nits.";
> = true;

#ifndef GAMESCOPE
uniform int LUMINANCE_WAVEFORM_SPACER_1
<
  ui_category = "Luminance waveform";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int HIGHLIGHT_NIT_RANGE_SPACER_0
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int HIGHLIGHT_NIT_RANGE_SPACER_1
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int DRAW_AS_BLACK_SPACER_0
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int DRAW_AS_BLACK_SPACER_1
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif


#define HDR_ANALYSIS_ENABLE

#include "lilium__include/hdr_analysis.fxh"


#ifdef _TESTY
uniform bool ENABLE_TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "enable test thingy";
> = false;

uniform uint TEST_MODE
<
  ui_category = "TESTY";
  ui_label    = "mode";
  ui_type     = "combo";
  ui_items    = " RGB\0"
                " xyY\0"
                "+Inf (0x7F800000) on R (else is 0)\0"
                "-Inf (0xFF800000) on G (else is 0)\0"
                " NaN (0xFFFFFFFF) on B (else is 0)\0";
> = 0;

precise uniform float TEST_THINGY_R
<
  ui_category = "TESTY";
  ui_label    = "R";
  ui_type     = "drag";
  ui_units    = " R";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;

precise uniform float TEST_THINGY_G
<
  ui_category = "TESTY";
  ui_label    = "G";
  ui_type     = "drag";
  ui_units    = " G";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;

precise uniform float TEST_THINGY_B
<
  ui_category = "TESTY";
  ui_label    = "B";
  ui_type     = "drag";
  ui_units    = " B";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;

precise uniform float TEST_THINGY_x
<
  ui_category = "TESTY";
  ui_label    = "x";
  ui_type     = "drag";
  ui_units    = " x";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.001f;
> = 0.f;

precise uniform float TEST_THINGY_y
<
  ui_category = "TESTY";
  ui_label    = "y";
  ui_type     = "drag";
  ui_units    = " y";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.001f;
> = 0.f;

precise uniform float TEST_THINGY_Y
<
  ui_category = "TESTY";
  ui_label    = "Y";
  ui_type     = "drag";
  ui_units    = " Y";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.001f;
> = 0.f;
#endif //_TESTY


//void draw_maxCLL(float4 position : POSITION, float2 txcoord : TEXCOORD) : COLOR
//void draw_maxCLL(float4 VPos : SV_Position, float2 TexCoord : TEXCOORD, out float4 fragment : SV_Target0)
//{
//  uint int_maxCLL = int(round(maxCLL));
//  uint digit1;
//  uint digit2;
//  uint digit3;
//  uint digit4;
//  uint digit5;
//
//  if (maxCLL < 10)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = 0;
//    digit4 = 0;
//    digit5 = int_maxCLL;
//  }
//  else if (maxCLL < 100)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = 0;
//    digit4 = int_maxCLL / 10;
//    digit5 = int_maxCLL % 10;
//  }
//  else if (maxCLL < 1000)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = int_maxCLL / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }
//  else if (maxCLL < 10000)
//  {
//    digit1 = 0;
//    digit2 = int_maxCLL / 1000;
//    digit3 = (int_maxCLL % 1000) / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }
//  else
//  {
//    digit1 = int_maxCLL / 10000;
//    digit2 = (int_maxCLL % 10000) / 1000;
//    digit3 = (int_maxCLL % 1000) / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }

  //res += tex2D(samplerText, (frac(uv) + float2(index % 14.0, trunc(index / 14.0))) /
  //            float2(_DRAWTEXT_GRID_X, _DRAWTEXT_GRID_Y)).x;

//  float4 hud = tex2D(samplerNumbers, TexCoord);
//  fragment = lerp(tex2Dfetch(ReShade::BackBuffer, TexCoord), hud, 1.f);
//
//}

#ifdef _TESTY
void Testy(
  in          float4 VPos     : SV_Position,
  in          float2 TexCoord : TEXCOORD,
  out precise float4 Output   : SV_Target0)
{
  Output = 0.f;

  if(ENABLE_TEST_THINGY == true)
  {
    float xxx = BUFFER_WIDTH  / 2.f - 100.f;
    float xxe = (BUFFER_WIDTH  - xxx);
    float yyy = BUFFER_HEIGHT / 2.f - 100.f;
    float yye = (BUFFER_HEIGHT - yyy);
    if (TexCoord.x > xxx / BUFFER_WIDTH
     && TexCoord.x < xxe / BUFFER_WIDTH
     && TexCoord.y > yyy / BUFFER_HEIGHT
     && TexCoord.y < yye / BUFFER_HEIGHT)
    {
      if (TEST_MODE == 0)
      {
        Output = float4(TEST_THINGY_R, TEST_THINGY_G, TEST_THINGY_B, 1.f);
        return;
      }
      else if (TEST_MODE == 1)
      {
        precise float3 XYZ = float3(
                       TEST_THINGY_x / TEST_THINGY_y * TEST_THINGY_Y,
                       TEST_THINGY_Y,
                       (1.f - TEST_THINGY_x - TEST_THINGY_y) / TEST_THINGY_y * TEST_THINGY_Y);
        Output = float4(mul(XYZToBt709, XYZ), 1.f);
        return;
      }
      else if (TEST_MODE == 2)
      {
        precise float asFloat = asfloat(0x7F800000);
        Output = float4(asFloat, 0.f, 0.f, 1.f);
        return;
      }
      else if (TEST_MODE == 3)
      {
        precise float asFloat = asfloat(0xFF800000);
        Output = float4(0.f, asFloat, 0.f, 1.f);
        return;
      }
      else if (TEST_MODE == 4)
      {
        precise float asFloat = asfloat(0xFFFFFFFF);
        Output = float4(0.f, 0.f, asFloat, 1.f);
        return;
      }
    }
    else
    {
      Output = float4(0.f, 0.f, 0.f, 0.f);
      return;
    }
  }
  // else
  discard;
}
#endif //_TESTY


#define SPACING_MULTIPLIER   0.3f
#define OUTER_SPACING       15.f
#define OUTER_SPACING_X2    (2.f * OUTER_SPACING)

static const float ShowNitsValuesLineCount     = 3;
static const float ShowNitsFromCursorLineCount = 1;

#if defined(IS_HDR10_LIKE_CSP)

  static const float ShowCspsLineCount = 3;

#else //IS_HDR10_LIKE_CSP

  static const float ShowCspsLineCount = 5;

#endif //IS_HDR10_LIKE_CSP


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
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW,  1);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, showNitsValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, showNitsFromCrusor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, showCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, showCspFromCursor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, fontSize);

    //calculate offset for the cursor nits text in the overlay
    float cursorNitsYOffset = (!SHOW_NITS_VALUES
                             ? -ShowNitsValuesLineCount
                             : SPACING_MULTIPLIER);

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS,
               cursorNitsYOffset);


    //calculate offset for the colour spaces text in the overlay
    float cspsYOffset = ((!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR)
                       ? -(ShowNitsValuesLineCount
                         - SPACING_MULTIPLIER)

                       : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR)
                       ? -(ShowNitsFromCursorLineCount
                         - SPACING_MULTIPLIER)

                       : (!SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR)
                       ? -(ShowNitsValuesLineCount
                         + ShowNitsFromCursorLineCount)

                       : SPACING_MULTIPLIER * 2);

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS,
               cspsYOffset);


    //calculate offset for the cursorCSP text in the overlay
    float cursorCspYOffset = ((!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR  && SHOW_CSPS)
                            ? -(ShowNitsValuesLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowNitsFromCursorLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_NITS_VALUES  && SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowCspsLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (!SHOW_NITS_VALUES && !SHOW_NITS_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowNitsValuesLineCount
                              + ShowNitsFromCursorLineCount
                              - SPACING_MULTIPLIER)

                            : (!SHOW_NITS_VALUES && SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowNitsValuesLineCount
                              + ShowCspsLineCount
                              - SPACING_MULTIPLIER)

                            : (SHOW_NITS_VALUES  && !SHOW_NITS_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowNitsFromCursorLineCount
                              + ShowCspsLineCount
                              - SPACING_MULTIPLIER)

                            : (!SHOW_NITS_VALUES && !SHOW_NITS_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowNitsValuesLineCount
                              + ShowNitsFromCursorLineCount
                              + ShowCspsLineCount)

#if defined(IS_HDR10_LIKE_CSP)
                            : SPACING_MULTIPLIER * 3) - 2;
#else //IS_HDR10_LIKE_CSP
                            : SPACING_MULTIPLIER * 3);
#endif //IS_HDR10_LIKE_CSP

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP,
               cursorCspYOffset);


    float4 bgCol = tex2Dfetch(StorageFontAtlasConsolidated, int2(0, 0)).rgba;

    uint activeLines = (SHOW_NITS_VALUES ? ShowNitsValuesLineCount
                                        : 0)
                     + (SHOW_NITS_FROM_CURSOR ? ShowNitsFromCursorLineCount
                                             : 0)
                     + (SHOW_CSPS ? ShowCspsLineCount
                                  : 0)
                     + (SHOW_CSP_FROM_CURSOR ? 1
                                             : 0);

    uint activeCharacters =
      max(max(max((SHOW_NITS_VALUES ? 21
                                   : 0),
                  (SHOW_NITS_FROM_CURSOR ? 24
                                        : 0)),
                  (SHOW_CSPS ? 16
                             : 0)),
                  (SHOW_CSP_FROM_CURSOR ? 18
                                        : 0));

    static const uint charSizeArrayOffsetX = (23 - TEXT_SIZE) * 2;

    uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);

    activeTextArea.y += (max(SHOW_NITS_VALUES
                           + SHOW_NITS_FROM_CURSOR
                           + SHOW_CSPS
                           + SHOW_CSP_FROM_CURSOR
                           - 1, 0) * charSize.y * SPACING_MULTIPLIER);

    activeTextArea += OUTER_SPACING_X2;

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
  uint atlasEntry     = 23 - TEXT_SIZE;
  uint charArrayEntry = atlasEntry * 2;

  uint2 charSize     = uint2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);
  uint  atlasXOffset = AtlasXOffset[atlasEntry];
  uint2 charOffset   = uint2(atlasXOffset, Char * charSize.y);

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      float4 pixel = tex2Dfetch(StorageFontAtlasConsolidated, charOffset + currentOffset).rgba;
      tex2Dstore(StorageTextOverlay, DrawOffset * charSize + OUTER_SPACING + currentOffset, pixel);
    }
  }
}


void DrawSpace(float2 DrawOffset)
{
  uint charArrayEntry = (23 - TEXT_SIZE) * 2;

  uint2 charSize = uint2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);

  float4 emptyPixel = tex2Dfetch(StorageFontAtlasConsolidated, int2(0, 0)).rgba;

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      tex2Dstore(StorageTextOverlay, DrawOffset * charSize + OUTER_SPACING + currentOffset, emptyPixel);
    }
  }
}


void CS_DrawTextToOverlay(uint3 ID : SV_DispatchThreadID)
{

  if (tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW))
  {

    static const float cursorNitsYOffset = 3.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS);
    static const float cspsYOffset0      = 4.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS);
    static const float cspsYOffset1      = 1.f + cspsYOffset0;
    static const float cspsYOffset2      = 2.f + cspsYOffset0;
#ifdef IS_FLOAT_HDR_CSP
    static const float cspsYOffset3      = 3.f + cspsYOffset0;
    static const float cspsYOffset4      = 4.f + cspsYOffset0;
#endif
    static const float cursorCspYOffset  = 9.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP);

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

    // max/avg/min Nits
    if (SHOW_NITS_VALUES)
    {
      // maxNits:
      DrawChar(_m,     float2( 0, 0));
      DrawChar(_a,     float2( 1, 0));
      DrawChar(_x,     float2( 2, 0));
      DrawChar(_N,     float2( 3, 0));
      DrawChar(_i,     float2( 4, 0));
      DrawChar(_t,     float2( 5, 0));
      DrawChar(_s,     float2( 6, 0));
      DrawChar(_colon, float2( 7, 0));
      DrawChar(_dot,   float2(14, 0)); // five figure number
      // avgNits:
      DrawChar(_a,     float2( 0, 1));
      DrawChar(_v,     float2( 1, 1));
      DrawChar(_g,     float2( 2, 1));
      DrawChar(_N,     float2( 3, 1));
      DrawChar(_i,     float2( 4, 1));
      DrawChar(_t,     float2( 5, 1));
      DrawChar(_s,     float2( 6, 1));
      DrawChar(_colon, float2( 7, 1));
      DrawChar(_dot,   float2(14, 1)); // five figure number
      // minNits:
      DrawChar(_m,     float2( 0, 2));
      DrawChar(_i,     float2( 1, 2));
      DrawChar(_n,     float2( 2, 2));
      DrawChar(_N,     float2( 3, 2));
      DrawChar(_i,     float2( 4, 2));
      DrawChar(_t,     float2( 5, 2));
      DrawChar(_s,     float2( 6, 2));
      DrawChar(_colon, float2( 7, 2));
      DrawChar(_dot,   float2(14, 2)); // five figure number
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

    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW, 0);
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


#define cursorNitsYOffset tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS)
#define cspsYOffset       tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cursorCspYOffset  tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP)

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
        DrawNumberAboveZero(curNumber, float2(9, 0));
      }
      return;
    }
    case 1:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _4th(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, 0));
      }
      return;
    }
    case 2:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _3rd(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, 0));
      }
      return;
    }
    case 3:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _2nd(maxNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, 0));
      }
      return;
    }
    case 4:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = _1st(maxNitsShow);
        DrawChar(curNumber, float2(13, 0));
      }
      return;
    }
    case 5:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d1st(maxNitsShow);
        DrawChar(curNumber, float2(15, 0));
      }
      return;
    }
    case 6:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d2nd(maxNitsShow);
        DrawChar(curNumber, float2(16, 0));
      }
      return;
    }
    case 7:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float maxNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAX_NITS);
        precise uint  curNumber   = d3rd(maxNitsShow);
        DrawChar(curNumber, float2(17, 0));
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
        DrawNumberAboveZero(curNumber, float2(9, 1));
      }
      return;
    }
    case 9:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _4th(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, 1));
      }
      return;
    }
    case 10:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _3rd(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, 1));
      }
      return;
    }
    case 11:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _2nd(avgNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, 1));
      }
      return;
    }
    case 12:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = _1st(avgNitsShow);
        DrawChar(curNumber, float2(13, 1));
      }
      return;
    }
    case 13:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d1st(avgNitsShow);
        DrawChar(curNumber, float2(15, 1));
      }
      return;
    }
    case 14:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d2nd(avgNitsShow);
        DrawChar(curNumber, float2(16, 1));
      }
      return;
    }
    case 15:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float avgNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVG_NITS);
        precise uint  curNumber   = d2nd(avgNitsShow);
        DrawChar(curNumber, float2(17, 1));
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
        DrawNumberAboveZero(curNumber, float2(9, 2));
      }
      return;
    }
    case 17:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _4th(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(10, 2));
      }
      return;
    }
    case 18:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _3rd(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(11, 2));
      }
      return;
    }
    case 19:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _2nd(minNitsShow);
        DrawNumberAboveZero(curNumber, float2(12, 2));
      }
      return;
    }
    case 20:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = _1st(minNitsShow);
        DrawChar(curNumber, float2(13, 2));
      }
      return;
    }
    case 21:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d1st(minNitsShow);
        DrawChar(curNumber, float2(15, 2));
      }
      return;
    }
    case 22:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d2nd(minNitsShow);
        DrawChar(curNumber, float2(16, 2));
      }
      return;
    }
    case 23:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d3rd(minNitsShow);
        DrawChar(curNumber, float2(17, 2));
      }
      return;
    }
    case 24:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d4th(minNitsShow);
        DrawChar(curNumber, float2(18, 2));
      }
      return;
    }
    case 25:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d5th(minNitsShow);
        DrawChar(curNumber, float2(19, 2));
      }
      return;
    }
    case 26:
    {
      if (SHOW_NITS_VALUES)
      {
        precise float minNitsShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MIN_NITS);
        precise uint  curNumber   = d6th(minNitsShow);
        DrawChar(curNumber, float2(20, 2));
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
        DrawNumberAboveZero(curNumber, float2(12, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 28:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _4th(cursorNits);
        DrawNumberAboveZero(curNumber, float2(13, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 29:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _3rd(cursorNits);
        DrawNumberAboveZero(curNumber, float2(14, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 30:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _2nd(cursorNits);
        DrawNumberAboveZero(curNumber, float2(15, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 31:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = _1st(cursorNits);
        DrawChar(curNumber, float2(16, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 32:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d1st(cursorNits);
        DrawChar(curNumber, float2(18, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 33:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d2nd(cursorNits);
        DrawChar(curNumber, float2(19, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 34:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d3rd(cursorNits);
        DrawChar(curNumber, float2(20, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 35:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber = d4th(cursorNits);
        DrawChar(curNumber, float2(21, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 36:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d5th(cursorNits);
        DrawChar(curNumber, float2(22, 3 + cursorNitsYOffset));
      }
      return;
    }
    case 37:
    {
      if (SHOW_NITS_FROM_CURSOR)
      {
        precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION).r;
        precise uint  curNumber  = d6th(cursorNits);
        DrawChar(curNumber, float2(23, 3 + cursorNitsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 4 + cspsYOffset));
      }
      return;
    }
    case 39:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _2nd(precentageBt709);
        DrawNumberAboveZero(curNumber, float2(10, 4 + cspsYOffset));
      }
      return;
    }
    case 40:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _1st(precentageBt709);
        DrawChar(curNumber, float2(11, 4 + cspsYOffset));
      }
      return;
    }
    case 41:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d1st(precentageBt709);
        DrawChar(curNumber, float2(13, 4 + cspsYOffset));
      }
      return;
    }
    case 42:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d2nd(precentageBt709);
        DrawChar(curNumber, float2(14, 4 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 5 + cspsYOffset));
      }
      return;
    }
    case 44:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _2nd(precentageDciP3);
        DrawNumberAboveZero(curNumber, float2(10, 5 + cspsYOffset));
      }
      return;
    }
    case 45:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _1st(precentageDciP3);
        DrawChar(curNumber, float2(11, 5 + cspsYOffset));
      }
      return;
    }
    case 46:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d1st(precentageDciP3);
        DrawChar(curNumber, float2(13, 5 + cspsYOffset));
      }
      return;
    }
    case 47:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d2nd(precentageDciP3);
        DrawChar(curNumber, float2(14, 5 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 6 + cspsYOffset));
      }
      return;
    }
    case 49:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _2nd(precentageBt2020);
        DrawNumberAboveZero(curNumber, float2(10, 6 + cspsYOffset));
      }
      return;
    }
    case 50:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _1st(precentageBt2020);
        DrawChar(curNumber, float2(11, 6 + cspsYOffset));
      }
      return;
    }
    case 51:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d1st(precentageBt2020);
        DrawChar(curNumber, float2(13, 6 + cspsYOffset));
      }
      return;
    }
    case 52:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d2nd(precentageBt2020);
        DrawChar(curNumber, float2(14, 6 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 7 + cspsYOffset));
      }
      return;
    }
    case 54:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _2nd(precentageAp0);
        DrawNumberAboveZero(curNumber, float2(10, 7 + cspsYOffset));
      }
      return;
    }
    case 55:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _1st(precentageAp0);
        DrawChar(curNumber, float2(11, 7 + cspsYOffset));
      }
      return;
    }
    case 56:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d1st(precentageAp0);
        DrawChar(curNumber, float2(13, 7 + cspsYOffset));
      }
      return;
    }
    case 57:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d2nd(precentageAp0);
        DrawChar(curNumber, float2(14, 7 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 8 + cspsYOffset));
      }
      return;
    }
    case 59:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _2nd(precentageInvalid);
        DrawNumberAboveZero(curNumber, float2(10, 8 + cspsYOffset));
      }
      return;
    }
    case 60:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _1st(precentageInvalid);
        DrawChar(curNumber, float2(11, 8 + cspsYOffset));
      }
      return;
    }
    case 61:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d1st(precentageInvalid);
        DrawChar(curNumber, float2(13, 8 + cspsYOffset));
      }
      return;
    }
    case 62:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d2nd(precentageInvalid);
        DrawChar(curNumber, float2(14, 8 + cspsYOffset));
      }
      return;
    }
#endif
    //cursorCSP:
    case 63:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(11, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(12, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(13, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(14, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(15, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(16, 9 + cursorCspYOffset);

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
        float2 currentCursorCspOffset = float2(17, 9 + cursorCspYOffset);

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

#undef cursorNitsYOffset
#undef cspsYOffset
#undef cursorCspYOffset


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareSetActiveArea(
  in                  uint   Id                : SV_VertexID,
  out                 float4 VPos              : SV_Position,
  out                 float2 TexCoord          : TEXCOORD0,
  out nointerpolation float4 PercentagesToCrop : PercentagesToCrop)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);


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
  in                  float4 VPos              : SV_Position,
  in                  float2 TexCoord          : TEXCOORD,
  in  nointerpolation float4 PercentagesToCrop : PercentagesToCrop,
  out                 float4 Output            : SV_Target0)
{
  Output = 0.f;

  if (ACTIVE_AREA_ENABLE)
  {
    float2 pureCoord = TexCoord * ReShade::ScreenSize;

    if (pureCoord.x > percentageToCropFromLeft
     && pureCoord.y > percentageToCropFromTop
     && pureCoord.x < percentageToCropFromRight
     && pureCoord.y < percentageToCropFromBottom)
    {
      discard;
    }
    else
    {
      Output = float4(0.f, 0.f, 0.f, 1.f);
    }
  }

  discard;
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrAnalysis(
  in                  uint   Id                   : SV_VertexID,
  out                 float4 VPos                 : SV_Position,
  out                 float2 TexCoord             : TEXCOORD0,
  out nointerpolation bool   PingPongChecks[2]    : PingPongChecks,
  out nointerpolation float4 HighlightNitRange    : HighlightNitRange,
#ifndef GAMESCOPE
  out nointerpolation float4 TextureDisplaySizes0 : TextureDisplaySizes0,
  out nointerpolation float4 TextureDisplaySizes1 : TextureDisplaySizes1,
  out nointerpolation float4 TextureDisplaySizes2 : TextureDisplaySizes2)
#else
  out nointerpolation float2 LuminanceWaveformTextureDisplaySize : LuminanceWaveformTextureDisplaySize,
  out nointerpolation float2 CieDiagramTextureActiveSize         : CieDiagramTextureActiveSize,
  out nointerpolation float2 CieDiagramTextureDisplaySize        : CieDiagramTextureDisplaySize,
  out nointerpolation float2 CieDiagramConsolidatedActiveSize    : CieDiagramConsolidatedActiveSize,
  out nointerpolation float2 CieOutlinesSamplerOffset            : CieOutlinesSamplerOffset,
  out nointerpolation float2 CurrentActiveOverlayArea            : CurrentActiveOverlayArea)
#endif
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
  #define LuminanceWaveformTextureDisplaySize TextureDisplaySizes0.xy
  #define CieDiagramTextureActiveSize         TextureDisplaySizes0.zw
  #define CieDiagramTextureDisplaySize        TextureDisplaySizes1.xy
  #define CieDiagramConsolidatedActiveSize    TextureDisplaySizes1.zw
  #define CieOutlinesSamplerOffset            TextureDisplaySizes2.xy
  #define CurrentActiveOverlayArea            TextureDisplaySizes2.zw
#endif

  pingpong0Above1      = false;
  breathingIsActive    = false;
  HighlightNitRange    = 0.f;
#ifndef GAMESCOPE
  TextureDisplaySizes0 = 0.f;
  TextureDisplaySizes1 = 0.f;
  TextureDisplaySizes2 = 0.f;
#else
  LuminanceWaveformTextureDisplaySize = 0.f;
  CieDiagramTextureActiveSize         = 0.f;
  CieDiagramTextureDisplaySize        = 0.f;
  CieDiagramConsolidatedActiveSize    = 0.f;
  CieOutlinesSamplerOffset            = 0.f;
  CurrentActiveOverlayArea            = 0.f;
#endif

  if (HIGHLIGHT_NIT_RANGE)
  {
    float pingpong0 = NIT_PINGPONG0.x + 0.25f;

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
    LuminanceWaveformTextureDisplaySize = Waveform::GetActiveArea();
  }


  if (SHOW_CIE)
  {
    float cieDiagramSizeFrac = CIE_DIAGRAM_SIZE / 100.f;

    CieDiagramTextureActiveSize =
      round(float2(CIE_BG_WIDTH[CIE_DIAGRAM_TYPE], CIE_BG_HEIGHT[CIE_DIAGRAM_TYPE]) * cieDiagramSizeFrac);

    CieDiagramTextureDisplaySize =
      float2(CIE_1931_BG_WIDTH, CIE_1931_BG_HEIGHT) * cieDiagramSizeFrac;

    CieDiagramConsolidatedActiveSize = CIE_CONSOLIDATED_TEXTURE_SIZE * cieDiagramSizeFrac;

    CieOutlinesSamplerOffset =
      float2(CIE_BG_WIDTH[CIE_DIAGRAM_TYPE],
             float(CIE_1931_BG_HEIGHT) * float(CIE_DIAGRAM_TYPE)) * cieDiagramSizeFrac;
  }

  if (SHOW_NITS_VALUES
   || SHOW_NITS_FROM_CURSOR
   || SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    float activeLines = (SHOW_NITS_VALUES ? ShowNitsValuesLineCount
                                         : 0.f)
                      + (SHOW_NITS_FROM_CURSOR ? ShowNitsFromCursorLineCount
                                              : 0.f)
                      + (SHOW_CSPS ? ShowCspsLineCount
                                   : 0.f)
                      + (SHOW_CSP_FROM_CURSOR ? 1.f
                                              : 0.f);

    float activeCharacters = max(max(max((SHOW_NITS_VALUES ? 21.f
                                                          :  0.f),
                                         (SHOW_NITS_FROM_CURSOR ? 24.f
                                                               :  0.f)),
                                         (SHOW_CSPS ? 16.f
                                                    :  0.f)),
                                         (SHOW_CSP_FROM_CURSOR ? 18.f
                                                               :  0.f));

    static const uint charArrayEntry = (23 - TEXT_SIZE) * 2;

    float2 charSize = float2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);

    float2 currentOverlayDimensions = charSize
                                    * float2(activeCharacters, activeLines);

    currentOverlayDimensions.y += (max(SHOW_NITS_VALUES
                                     + SHOW_NITS_FROM_CURSOR
                                     + SHOW_CSPS
                                     + SHOW_CSP_FROM_CURSOR
                                     - 1, 0) * charSize.y * SPACING_MULTIPLIER);

    currentOverlayDimensions += OUTER_SPACING_X2;

    CurrentActiveOverlayArea = (currentOverlayDimensions - 1.f + 0.5f)
                             / ReShade::ScreenSize;

    if (TEXT_POSITION == TEXT_POSITION_TOP_RIGHT)
    {
      CurrentActiveOverlayArea.x = 1.f - CurrentActiveOverlayArea.x;
    }
  }

}


void ExtendedReinhardTmo(
  inout float3 Colour,
  in    float  WhitePoint)
{
  float maxWhite = 10000.f / WhitePoint;

  Colour =  (Colour * (1.f + (Colour / (maxWhite * maxWhite))))
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

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  adjustFactor = OverlayBrightness / 10000.f;

  Output =  Csp::Trc::PqTo::Linear(Output);
  Output /= adjustFactor;

#endif

  // then tone map to 1.0 at max
  ExtendedReinhardTmo(Output, OverlayBrightness);

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
  in                  float4 VPos                 : SV_Position,
  in                  float2 TexCoord             : TEXCOORD,
  in  nointerpolation bool   PingPongChecks[2]    : PingPongChecks,
  in  nointerpolation float4 HighlightNitRange    : HighlightNitRange,
#ifndef GAMESCOPE
  in  nointerpolation float4 TextureDisplaySizes0 : TextureDisplaySizes0,
  in  nointerpolation float4 TextureDisplaySizes1 : TextureDisplaySizes1,
  in  nointerpolation float4 TextureDisplaySizes2 : TextureDisplaySizes2,
#else
  in  nointerpolation float2 LuminanceWaveformTextureDisplaySize : LuminanceWaveformTextureDisplaySize,
  in  nointerpolation float2 CieDiagramTextureActiveSize         : CieDiagramTextureActiveSize,
  in  nointerpolation float2 CieDiagramTextureDisplaySize        : CieDiagramTextureDisplaySize,
  in  nointerpolation float2 CieDiagramConsolidatedActiveSize    : CieDiagramConsolidatedActiveSize,
  in  nointerpolation float2 CieOutlinesSamplerOffset            : CieOutlinesSamplerOffset,
  in  nointerpolation float2 CurrentActiveOverlayArea            : CurrentActiveOverlayArea,
#endif
  out                 float4 Output               : SV_Target0)
{
  Output = float4(tex2D(ReShade::BackBuffer, TexCoord).rgb, 1.f);

  const float2 pureCoord = VPos.xy - 0.5f;

  if (SHOW_CSP_MAP
   || SHOW_HEATMAP
   || HIGHLIGHT_NIT_RANGE
   || DRAW_ABOVE_NITS_AS_BLACK
   || DRAW_BELOW_NITS_AS_BLACK)
  {
    const float pixelNits = tex2D(SamplerNitsValues, TexCoord).r;

    if (SHOW_CSP_MAP)
    {
      Output.rgb = CreateCspMap(tex2D(SamplerCsps, TexCoord).r * 255.f, pixelNits);
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
    float textureDisplayAreaYBegin = BUFFER_HEIGHT - CieDiagramTextureActiveSize.y;

    // draw the diagram in the bottom left corner
    if (pureCoord.x <  CieDiagramTextureActiveSize.x
     && pureCoord.y >= textureDisplayAreaYBegin)
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(pureCoord.x,
                                           pureCoord.y - textureDisplayAreaYBegin);

      currentSamplerCoords += 0.5f;

      float2 currentCieSamplerCoords = currentSamplerCoords / CieDiagramTextureDisplaySize;

      float3 currentPixelToDisplay = pow(tex2D(SamplerCieCurrent, currentCieSamplerCoords).rgb, 2.2f);

      float3 cspOutlineOverlay = float3(0.f, 0.f, 0.f);

      if (SHOW_CIE_CSP_BT709_OUTLINE)
      {
        float2 outlineSamplerCoords =
          float2(currentSamplerCoords.x + CieOutlinesSamplerOffset.x * float(CIE_TEXTURE_ENTRY_BT709_OUTLINE),
                 currentSamplerCoords.y + CieOutlinesSamplerOffset.y);

        outlineSamplerCoords /= CieDiagramConsolidatedActiveSize;

        cspOutlineOverlay += pow(tex2D(SamplerCieConsolidated, outlineSamplerCoords).rgb, 2.2f);
      }
      if (SHOW_CIE_CSP_DCI_P3_OUTLINE)
      {
        float2 outlineSamplerCoords =
          float2(currentSamplerCoords.x + CieOutlinesSamplerOffset.x * float(CIE_TEXTURE_ENTRY_DCI_P3_OUTLINE),
                 currentSamplerCoords.y + CieOutlinesSamplerOffset.y);

        outlineSamplerCoords /= CieDiagramConsolidatedActiveSize;

        cspOutlineOverlay += pow(tex2D(SamplerCieConsolidated, outlineSamplerCoords).rgb, 2.2f);
      }
      if (SHOW_CIE_CSP_BT2020_OUTLINE)
      {
        float2 outlineSamplerCoords =
          float2(currentSamplerCoords.x + CieOutlinesSamplerOffset.x * float(CIE_TEXTURE_ENTRY_BT2020_OUTLINE),
                 currentSamplerCoords.y + CieOutlinesSamplerOffset.y);

        outlineSamplerCoords /= CieDiagramConsolidatedActiveSize;

        cspOutlineOverlay += pow(tex2D(SamplerCieConsolidated, outlineSamplerCoords).rgb, 2.2f);
      }
#ifdef IS_FLOAT_HDR_CSP
      if (SHOW_CIE_CSP_AP0_OUTLINE)
      {
        float2 outlineSamplerCoords =
          float2(currentSamplerCoords.x + CieOutlinesSamplerOffset.x * float(CIE_TEXTURE_ENTRY_AP0_OUTLINE),
                 currentSamplerCoords.y + CieOutlinesSamplerOffset.y);

        outlineSamplerCoords /= CieDiagramConsolidatedActiveSize;

        cspOutlineOverlay += pow(tex2D(SamplerCieConsolidated, outlineSamplerCoords).rgb, 2.2f);
      }
#endif

      cspOutlineOverlay = min(cspOutlineOverlay, 1.25f);

      currentPixelToDisplay = currentPixelToDisplay + cspOutlineOverlay;

      float alpha = min(ceil(MAXRGB(currentPixelToDisplay)) + CIE_DIAGRAM_ALPHA / 100.f, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay,
                   CIE_DIAGRAM_BRIGHTNESS,
                   alpha);

    }
  }

  if (SHOW_LUMINANCE_WAVEFORM)
  {
    float2 textureDisplayAreaBegin = ReShade::ScreenSize - LuminanceWaveformTextureDisplaySize;

    // draw the waveform in the bottom right corner
    if (all(pureCoord >= textureDisplayAreaBegin))
    {
      // get coords for the sampler
      float2 currentSamplerCoords = pureCoord - textureDisplayAreaBegin;

      currentSamplerCoords += 0.5f;
      currentSamplerCoords /= float2(TEXTURE_LUMINANCE_WAVEFORM_SCALE_WIDTH, TEXTURE_LUMINANCE_WAVEFORM_SCALE_HEIGHT);

      float3 currentPixelToDisplay =
        tex2D(SamplerLuminanceWaveformFinal, currentSamplerCoords).rgb;

      float alpha = min(ceil(MAXRGB(currentPixelToDisplay)) + LUMINANCE_WAVEFORM_ALPHA / 100.f, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay,
                   LUMINANCE_WAVEFORM_BRIGHTNESS,
                   alpha);

    }
  }

  if (SHOW_NITS_VALUES
   || SHOW_NITS_FROM_CURSOR
   || SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    if (TEXT_POSITION == TEXT_POSITION_TOP_LEFT)
    {
      if (all(TexCoord <= CurrentActiveOverlayArea))
      {
        float4 overlay = tex2D(SamplerTextOverlay, (TexCoord
                                                  * ReShade::ScreenSize
                                                  / float2(TEXTURE_OVERLAY_WIDTH, TEXTURE_OVERLAY_HEIGHT))).rgba;

        overlay.rgb = Csp::Mat::Bt709To::Bt2020(overlay.rgb);

        float alpha = min(TEXT_BG_ALPHA / 100.f + overlay.a, 1.f);

        MergeOverlay(Output.rgb,
                     overlay.rgb,
                     TEXT_BRIGHTNESS,
                     alpha);
      }
    }
    else
    {
      if (TexCoord.x >= CurrentActiveOverlayArea.x
       && TexCoord.y <= CurrentActiveOverlayArea.y)
      {
        float4 overlay = tex2D(SamplerTextOverlay, float2(TexCoord.x - CurrentActiveOverlayArea.x, TexCoord.y)
                                                 * ReShade::ScreenSize
                                                 / float2(TEXTURE_OVERLAY_WIDTH, TEXTURE_OVERLAY_HEIGHT)).rgba;

        overlay = float4(MapBt709IntoCurrentCsp(overlay.rgb, TEXT_BRIGHTNESS), overlay.a);

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

void CS_MakeOverlayBgRedraw()
{
  tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, 3.f);
  return;
}

technique lilium__make_overlay_bg_redraw
<
  enabled = true;
  hidden  = true;
  timeout = 1;
>
{
  pass CS_MakeOverlayBgRedraw
  {
    ComputeShader = CS_MakeOverlayBgRedraw <1, 1>;
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
  pass PS_CopyCieBg
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CopyCieBg;
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
