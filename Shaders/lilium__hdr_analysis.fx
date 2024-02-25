#include "lilium__include/colour_space.fxh"


#if defined(IS_HDR_COMPATIBLE_API)

#undef TEXT_BRIGHTNESS

#if (defined(API_IS_VULKAN) \
  && (BUFFER_WIDTH > 1280   \
   || BUFFER_HEIGHT > 800))
  #warning "If you are on the Steam Deck and want to use this shader, you need to switch to a resolution that is 1280x800 or lower!"
  #warning "The Steam Deck also has my HDR analysis shaders built in. They are available throught developer options."
#endif

#ifndef GAMESCOPE
  //#define _DEBUG
  //#define _TESTY
#endif

#if (BUFFER_WIDTH  >= 2560) \
 && (BUFFER_HEIGHT >= 1440)
  #define IS_QHD_OR_HIGHER_RES
#endif


#define DEFAULT_BRIGHTNESS 80.f

#define DEFAULT_ALPHA_LEVEL 75.f

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

#if ((defined(GAMESCOPE) || defined(POSSIBLE_DECK_VULKAN_USAGE)) \
  && BUFFER_HEIGHT <= 800)
  static const uint TEXT_SIZE_DEFAULT = 0;
#else
  // 2160 height gives 42
  //  800 height gives 16
  //  720 height gives 14
  static const uint TEXT_SIZE_DEFAULT = (uint((BUFFER_HEIGHT / 2160.f) * 42.f + 0.5f) - 12) / 2;
#endif


#ifndef IS_HDR_CSP
  #define _TEXT_SIZE                              SDR_TEXT_SIZE
  #define _TEXT_BRIGHTNESS                        SDR_TEXT_BRIGHTNESS
  #define _TEXT_BG_ALPHA                          SDR_TEXT_BG_ALPHA
  #define _TEXT_POSITION                          SDR_TEXT_POSITION
  #define _ACTIVE_AREA_ENABLE                     SDR_ACTIVE_AREA_ENABLE
  #define _TACTIVE_AREA_CROP_LEFT                 SDR_ACTIVE_AREA_CROP_LEFT
  #define _ACTIVE_AREA_CROP_TOP                   SDR_ACTIVE_AREA_CROP_TOP
  #define _ACTIVE_AREA_CROP_RIGHT                 SDR_ACTIVE_AREA_CROP_RIGHT
  #define _ACTIVE_AREA_CROP_BOTTOM                SDR_ACTIVE_AREA_CROP_BOTTOM
  #define _SHOW_NITS_VALUES                       SDR_SHOW_NITS_VALUES
  #define _SHOW_NITS_FROM_CURSOR                  SDR_SHOW_NITS_FROM_CURSOR
  #define _SHOW_CIE                               SDR_SHOW_CIE
  #define _CIE_DIAGRAM_TYPE                       SDR_CIE_DIAGRAM_TYPE
  #define _CIE_DIAGRAM_BRIGHTNESS                 SDR_CIE_DIAGRAM_BRIGHTNESS
  #define _CIE_DIAGRAM_ALPHA                      SDR_CIE_DIAGRAM_ALPHA
  #define _CIE_DIAGRAM_SIZE                       SDR_CIE_DIAGRAM_SIZE
  #define _CIE_SHOW_CIE_CSP_BT709_OUTLINE         SDR_SHOW_CIE_CSP_BT709_OUTLINE
  #define _SHOW_HEATMAP                           SDR_SHOW_HEATMAP
  #define _HEATMAP_BRIGHTNESS                     SDR_HEATMAP_BRIGHTNESS
  #define _SHOW_LUMINANCE_WAVEFORM                SDR_SHOW_LUMINANCE_WAVEFORM
  #define _LUMINANCE_WAVEFORM_BRIGHTNESS          SDR_LUMINANCE_WAVEFORM_BRIGHTNESS
  #define _LUMINANCE_WAVEFORM_ALPHA               SDR_LUMINANCE_WAVEFORM_ALPHA
  #define _LUMINANCE_WAVEFORM_SIZE                SDR_LUMINANCE_WAVEFORM_SIZE
  #define _LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE  SDR_LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE
  #define _LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE  SDR_LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE
  #define _HIGHLIGHT_NIT_RANGE                    SDR_HIGHLIGHT_NIT_RANGE
  #define _HIGHLIGHT_NIT_RANGE_BRIGHTNESS         SDR_HIGHLIGHT_NIT_RANGE_BRIGHTNESS
  #define _HIGHLIGHT_NIT_RANGE_START_POINT        SDR_HIGHLIGHT_NIT_RANGE_START_POINT
  #define _HIGHLIGHT_NIT_RANGE_END_POINT          SDR_HIGHLIGHT_NIT_RANGE_END_POINT
  #define _DRAW_ABOVE_NITS_AS_BLACK               SDR_DRAW_ABOVE_NITS_AS_BLACK
  #define _ABOVE_NITS_AS_BLACK                    SDR_ABOVE_NITS_AS_BLACK
  #define _DRAW_BELOW_NITS_AS_BLACK               SDR_DRAW_BELOW_NITS_AS_BLACK
  #define _BELOW_NITS_AS_BLACK                    SDR_BELOW_NITS_AS_BLACK
#else
  #define _TEXT_SIZE                              TEXT_SIZE
  #define _TEXT_BRIGHTNESS                        TEXT_BRIGHTNESS
  #define _TEXT_BG_ALPHA                          TEXT_BG_ALPHA
  #define _TEXT_POSITION                          TEXT_POSITION
  #define _ACTIVE_AREA_ENABLE                     ACTIVE_AREA_ENABLE
  #define _ACTIVE_AREA_CROP_LEFT                  ACTIVE_AREA_CROP_LEFT
  #define _ACTIVE_AREA_CROP_TOP                   ACTIVE_AREA_CROP_TOP
  #define _ACTIVE_AREA_CROP_RIGHT                 ACTIVE_AREA_CROP_RIGHT
  #define _ACTIVE_AREA_CROP_BOTTOM                ACTIVE_AREA_CROP_BOTTOM
  #define _SHOW_NITS_VALUES                       SHOW_NITS_VALUES
  #define _SHOW_NITS_FROM_CURSOR                  SHOW_NITS_FROM_CURSOR
  #define _SHOW_CIE                               SHOW_CIE
  #define _CIE_DIAGRAM_TYPE                       CIE_DIAGRAM_TYPE
  #define _CIE_DIAGRAM_BRIGHTNESS                 CIE_DIAGRAM_BRIGHTNESS
  #define _CIE_DIAGRAM_ALPHA                      CIE_DIAGRAM_ALPHA
  #define _CIE_DIAGRAM_SIZE                       CIE_DIAGRAM_SIZE
  #define _SHOW_CIE_CSP_BT709_OUTLINE             SHOW_CIE_CSP_BT709_OUTLINE
  #define _SHOW_HEATMAP                           SHOW_HEATMAP
  #define _HEATMAP_BRIGHTNESS                     HEATMAP_BRIGHTNESS
  #define _SHOW_LUMINANCE_WAVEFORM                SHOW_LUMINANCE_WAVEFORM
  #define _LUMINANCE_WAVEFORM_BRIGHTNESS          LUMINANCE_WAVEFORM_BRIGHTNESS
  #define _LUMINANCE_WAVEFORM_ALPHA               LUMINANCE_WAVEFORM_ALPHA
  #define _LUMINANCE_WAVEFORM_SIZE                LUMINANCE_WAVEFORM_SIZE
  #define _LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE  LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE
  #define _LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE  LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE
  #define _HIGHLIGHT_NIT_RANGE                    HIGHLIGHT_NIT_RANGE
  #define _HIGHLIGHT_NIT_RANGE_BRIGHTNESS         HIGHLIGHT_NIT_RANGE_BRIGHTNESS
  #define _HIGHLIGHT_NIT_RANGE_START_POINT        HIGHLIGHT_NIT_RANGE_START_POINT
  #define _HIGHLIGHT_NIT_RANGE_END_POINT          HIGHLIGHT_NIT_RANGE_END_POINT
  #define _DRAW_ABOVE_NITS_AS_BLACK               DRAW_ABOVE_NITS_AS_BLACK
  #define _ABOVE_NITS_AS_BLACK                    ABOVE_NITS_AS_BLACK
  #define _DRAW_BELOW_NITS_AS_BLACK               DRAW_BELOW_NITS_AS_BLACK
  #define _BELOW_NITS_AS_BLACK                    BELOW_NITS_AS_BLACK
#endif


uniform uint _TEXT_SIZE
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

uniform float _TEXT_BRIGHTNESS
<
  ui_category = "Global";
  ui_label    = "text brightness";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
#else
  ui_units    = "%%";
  ui_min      = 10.f;
  ui_max      = 100.f;
#endif
  ui_step     = 0.5f;
#if defined(IS_HDR_CSP)
> = 140.f;
#elif (defined(GAMESCOPE) \
    && defined(IS_HDR_CSP))
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = DEFAULT_BRIGHTNESS;
#endif

uniform float _TEXT_BG_ALPHA
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

uniform uint _TEXT_POSITION
<
  ui_category = "Global";
  ui_label    = "text position";
  ui_type     = "combo";
  ui_items    = "top left\0"
                "top right\0";
> = 0;


// Active Area
uniform bool _ACTIVE_AREA_ENABLE
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "enable setting the active area";
> = false;

uniform float _ACTIVE_AREA_CROP_LEFT
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the left side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float _ACTIVE_AREA_CROP_TOP
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the top side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float _ACTIVE_AREA_CROP_RIGHT
<
  ui_category = "Set Active Area for analysis";
  ui_label    = "% to crop from the right side";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 0.f;

uniform float _ACTIVE_AREA_CROP_BOTTOM
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
uniform bool _SHOW_NITS_VALUES
<
  ui_category = "Luminance analysis";
  ui_label    = "show max/avg/min luminance values";
> = true;

uniform bool _SHOW_NITS_FROM_CURSOR
<
  ui_category = "Luminance analysis";
  ui_label    = "show luminance value from cursor position";
> = true;


// TextureCsps
#ifdef IS_HDR_CSP
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
#endif // IS_HDR_CSP

// CIE
uniform bool _SHOW_CIE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show CIE diagram";
> = true;

#define CIE_1931 0
#define CIE_1976 1

uniform uint _CIE_DIAGRAM_TYPE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram type";
  ui_type     = "combo";
  ui_items    = "CIE 1931 xy\0"
                "CIE 1976 UCS u'v'\0";
> = CIE_1931;

uniform float _CIE_DIAGRAM_BRIGHTNESS
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram brightness";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
#else
  ui_units    = "%%";
  ui_min      = 10.f;
  ui_max      = 100.f;
#endif
  ui_step     = 0.5f;
#if (defined(GAMESCOPE) \
  && defined(IS_HDR_CSP))
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = DEFAULT_BRIGHTNESS;
#endif

uniform float _CIE_DIAGRAM_ALPHA
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

uniform float _CIE_DIAGRAM_SIZE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = CIE_DIAGRAM_DEFAULT_SIZE;

uniform bool _SHOW_CIE_CSP_BT709_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show BT.709 colour space outline";
> = true;

#ifdef IS_HDR_CSP
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
#endif //IS_FLOAT_HDR_CSP
#endif //IS_HDR_CSP

// heatmap
uniform bool _SHOW_HEATMAP
<
  ui_category = "Heatmap visualisation";
  ui_label    = "show heatmap";
#ifndef IS_HDR_CSP
  ui_tooltip  = "         colours: |     % nits:"
           "\n" "------------------|------------"
           "\n" " black to white:  |   0.0-  1.0"
           "\n" "  teal to green:  |   1.0- 18.0"
           "\n" " green to yellow: |  18.0- 50.0"
           "\n" "yellow to red:    |  50.0- 75.0"
           "\n" "   red to pink:   |  75.0- 87.5"
           "\n" "  pink to blue:   |  87.5-100.0"
           "\n" "-------------------------------"
           "\n"
           "\n" "extra cases:"
           "\n" "highly saturated red:  above the cutoff point"
           "\n" "highly saturated blue: below 0 nits";
#endif
> = false;

uniform float _HEATMAP_BRIGHTNESS
<
  ui_category = "Heatmap visualisation";
  ui_label    = "heatmap brightness";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
#else
  ui_units    = "%%";
  ui_min      = 10.f;
  ui_max      = 100.f;
#endif
  ui_step     = 0.5f;
> = DEFAULT_BRIGHTNESS;

#ifdef IS_HDR_CSP
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
#endif //IS_HDR_CSP

uniform bool _SHOW_LUMINANCE_WAVEFORM
<
  ui_category = "Luminance waveform";
  ui_label    = "show luminance waveform";
  ui_tooltip  = "Luminance waveform paid for by Aemony.";
> = true;

#ifdef IS_HDR_CSP
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
#endif //IS_HDR_CSP

uniform float _LUMINANCE_WAVEFORM_BRIGHTNESS
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform brightness";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
#else
  ui_units    = "%%";
  ui_min      = 10.f;
  ui_max      = 100.f;
#endif
  ui_step     = 0.5f;
#if (defined(GAMESCOPE) \
  && defined(IS_HDR_CSP))
> = GAMESCOPE_SDR_ON_HDR_NITS;
#else
> = DEFAULT_BRIGHTNESS;
#endif

uniform float _LUMINANCE_WAVEFORM_ALPHA
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

#ifdef IS_HDR_CSP
  #if (BUFFER_HEIGHT <= (512 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 512;
  #elif (BUFFER_HEIGHT <= (768 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 768;
  #elif (BUFFER_HEIGHT <= (1024 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1024;
  #elif (BUFFER_HEIGHT <= (1536 * 5 / 2) \
      || defined(IS_HDR10_LIKE_CSP))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1536;
  #elif (BUFFER_HEIGHT <= (2048 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 2048;
  #elif (BUFFER_HEIGHT <= (3072 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 3072;
  #else //(BUFFER_HEIGHT <= (4096 * 5 / 2))
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 4096;
  #endif
#else
  #if (BUFFER_COLOR_BIT_DEPTH == 10)
    #if (BUFFER_HEIGHT <= (512 * 5 / 2))
      static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 512;
    #elif (BUFFER_HEIGHT <= (768 * 5 / 2))
      static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 768;
    #elif (BUFFER_HEIGHT <= (1024 * 5 / 2))
      static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1024;
    #elif (BUFFER_HEIGHT <= (1280 * 5 / 2))
      static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1280;
    #else //(BUFFER_HEIGHT <= (1536 * 5 / 2))
      static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 1536;
    #endif
  #else
    static const uint TEXTURE_LUMINANCE_WAVEFORM_HEIGHT = 384;
  #endif
#endif

static const uint TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT = TEXTURE_LUMINANCE_WAVEFORM_HEIGHT - 1;

static const int UGH = uint(float(BUFFER_HEIGHT) * 0.35f
                          / float(TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT)
                          * 10000.f);
// "minimum of 2 variables" without using functions...
// https://guru.multimedia.cx/category/optimization/
#ifdef IS_HDR_CSP
  static const uint UGH2 = int(10000) + ((UGH - int(10000)) & ((UGH - int(10000)) >> 31));
#else
  static const uint UGH2 = int(20000) + ((UGH - int(20000)) & ((UGH - int(20000)) >> 31));
#endif

static const precise float LUMINANCE_WAVEFORM_DEFAULT_HEIGHT = UGH2
                                                             / 100.f;

uniform float2 _LUMINANCE_WAVEFORM_SIZE
<
  ui_category = "Luminance waveform";
  ui_label    = "luminance waveform size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      =  50.f;
#ifdef IS_HDR_CSP
  ui_max      = 100.f;
#else
  ui_max      = 200.f;
#endif
  ui_step     =   0.1f;
> = float2(70.f, LUMINANCE_WAVEFORM_DEFAULT_HEIGHT);

uniform bool _LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE
<
  ui_category = "Luminance waveform";
  ui_label    = "show the minimum nits line";
  ui_tooltip  = "Show a horizontal line where the minimum nits is on the waveform."
           "\n" "The line is invisible when the minimum nits hits 0 nits.";
> = true;

uniform bool _LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE
<
  ui_category = "Luminance waveform";
  ui_label    = "show the maximum nits line";
  ui_tooltip  = "Show a horizontal line where the maximum nits is on the waveform."
           "\n" "The line is invisible when the maximum nits hits above 10000 nits.";
> = true;

// highlight a certain nit range
uniform bool _HIGHLIGHT_NIT_RANGE
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "enable highlighting brightness levels in a certain range";
  ui_tooltip  = "in nits";
> = false;

uniform float _HIGHLIGHT_NIT_RANGE_BRIGHTNESS
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "highlighted range brightness";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
#else
  ui_units    = "%%";
  ui_min      = 10.f;
  ui_max      = 100.f;
#endif
> = DEFAULT_BRIGHTNESS;

uniform float _HIGHLIGHT_NIT_RANGE_START_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range starting point";
  ui_tooltip  = "CTRL + LEFT CLICK on the value to input an exact value.";
  ui_type     = "drag";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
#else
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.00001f;
#endif
> = 0.f;

uniform float _HIGHLIGHT_NIT_RANGE_END_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range end point";
  ui_tooltip  = "CTRL + LEFT CLICK on the value to input an exact value.";
  ui_type     = "drag";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
#else
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.00001f;
#endif
> = 0.f;

// draw pixels as black depending on their nits
uniform bool _DRAW_ABOVE_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "enable drawing above this brightness as black";
> = false;

uniform float _ABOVE_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "draw above this brightness as black";
  ui_type     = "slider";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
#else
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.00001f;
#endif
#ifdef IS_HDR_CSP
> = 10000.f;
#else
> = 100.f;
#endif

uniform bool _DRAW_BELOW_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "enable drawing below this brightness as black";
> = false;

uniform float _BELOW_NITS_AS_BLACK
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = "draw below this brightness as black";
  ui_type     = "drag";
#ifdef IS_HDR_CSP
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 1.f;
#else
  ui_units    = "%%";
  ui_min      = 0.f;
  ui_max      = 100.f;
  ui_step     = 0.5f;
#endif
#ifdef IS_HDR_CSP
> = 10000.f;
#else
> = 100.f;
#endif


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

#elif defined(IS_HDR_CSP)

  #define SHOW_CSPS_LINE_COUNT 5

#else

  #define SHOW_CSPS_LINE_COUNT 0

#endif //IS_HDR10_LIKE_CSP

#if defined(IS_HDR_CSP)
  #define SHOW_CSP_FROM_CURSOR_LINE_COUNT 1
#else
  #define SHOW_CSP_FROM_CURSOR_LINE_COUNT 0
#endif



#define ANALYSIS_ENABLE

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
       + (_SHOW_NITS_VALUES ? SHOW_NITS_VALUES_LINE_COUNT
                            : 0)
       + (_SHOW_NITS_FROM_CURSOR ? SHOW_NITS_FROM_CURSOR_LINE_COUNT
                                 : 0)
#ifdef IS_HDR_CSP
       + (SHOW_CSPS ? SHOW_CSPS_LINE_COUNT
                    : 0)
       + (SHOW_CSP_FROM_CURSOR ? SHOW_CSP_FROM_CURSOR_LINE_COUNT
                               : 0)
#endif
                                   ;
}

#ifdef IS_HDR_CSP
  #define CSP_DESC_TEXT_LENGTH 20
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
  #if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
    || OVERWRITE_SDR_GAMMA == GAMMA_22    \
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
//              (_SHOW_NITS_VALUES ? 21
//                                 :  0),
//              (_SHOW_NITS_FROM_CURSOR ? 24
//                                      :  0),
//              (SHOW_CSPS ? 16
//                         :  0),
//              (SHOW_CSP_FROM_CURSOR ? 18
//                                    :  0));
  return MAX3(CSP_DESC_TEXT_LENGTH,
              (_SHOW_NITS_VALUES ? 21
                                 :  0),
              (_SHOW_NITS_FROM_CURSOR ? 24
                                      :  0));
}

uint GetAtlasEntry()
{
  return 23 - _TEXT_SIZE;
}

uint GetCharArrayEntry()
{
  return GetAtlasEntry() * 2;
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
      float4 pixel = tex2Dfetch(SamplerFontAtlasConsolidated, charOffset + currentOffset);
      tex2Dstore(StorageTextOverlay, uint2(DrawOffset * charSize) + outerSpacing + currentOffset, pixel);
    }
  }
}

void DrawSpace(float2 DrawOffset)
{
  uint charArrayEntry = GetCharArrayEntry();

  uint2 charSize = uint2(CharSize[charArrayEntry], CharSize[charArrayEntry + 1]);

  uint outerSpacing = GetOuterSpacing(charSize.x);

  float4 emptyPixel = tex2Dfetch(SamplerFontAtlasConsolidated, int2(0, 0));

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
  //convert UI inputs into floats for comparisons
  const float showNitsValues     = _SHOW_NITS_VALUES;
  const float showNitsFromCrusor = _SHOW_NITS_FROM_CURSOR;

#ifdef IS_HDR_CSP
  const float showCsps          = SHOW_CSPS;
  const float showCspFromCursor = SHOW_CSP_FROM_CURSOR;
#endif

  const float fontSize          = _TEXT_SIZE;

  //get last UI values from the consolidated texture
  const float showNitsLast       = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  const float showCursorNitsLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

#ifdef IS_HDR_CSP
  const float showCspsLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  const float showCursorCspLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);
#endif

  const float fontSizeLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);

  if (showNitsLast       != showNitsValues
   || showCursorNitsLast != showNitsFromCrusor
#ifdef IS_HDR_CSP
   || showCspsLast       != showCsps
   || showCursorCspLast  != showCspFromCursor
#endif
   || fontSizeLast       != fontSize)
  {
    //store all current UI values
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, showNitsValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, showNitsFromCrusor);
#ifdef IS_HDR_CSP
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, showCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, showCspFromCursor);
#endif
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, fontSize);

    //calculate offset for the cursor nits text in the overlay
    float cursorNitsYOffset = (!_SHOW_NITS_VALUES
                             ? -SHOW_NITS_VALUES_LINE_COUNT
                             : SPACING_MULTIPLIER)
                            + CSP_DESC_SPACING_MULTIPLIER;

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS,
               cursorNitsYOffset);


#ifdef IS_HDR_CSP
    //calculate offset for the colour spaces text in the overlay
    float cspsYOffset = ((!_SHOW_NITS_VALUES) && _SHOW_NITS_FROM_CURSOR
                       ? -(SHOW_NITS_VALUES_LINE_COUNT
                         - SPACING_MULTIPLIER)

                       : (_SHOW_NITS_VALUES)  && !_SHOW_NITS_FROM_CURSOR
                       ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                         - SPACING_MULTIPLIER)

                       : (!_SHOW_NITS_VALUES)  && !_SHOW_NITS_FROM_CURSOR
                       ? -(SHOW_NITS_VALUES_LINE_COUNT
                         + SHOW_NITS_FROM_CURSOR_LINE_COUNT)

                       : SPACING_MULTIPLIER * 2)
                      + CSP_DESC_SPACING_MULTIPLIER;

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS,
               cspsYOffset);


    //calculate offset for the cursorCSP text in the overlay
    float cursorCspYOffset = ((!_SHOW_NITS_VALUES) && _SHOW_NITS_FROM_CURSOR  && SHOW_CSPS
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (_SHOW_NITS_VALUES)  && !_SHOW_NITS_FROM_CURSOR && SHOW_CSPS
                            ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (_SHOW_NITS_VALUES)  && _SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS
                            ? -(SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER * 2)

                            : (!_SHOW_NITS_VALUES) && !_SHOW_NITS_FROM_CURSOR && SHOW_CSPS
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              + SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (!_SHOW_NITS_VALUES) && _SHOW_NITS_FROM_CURSOR  && !SHOW_CSPS
                            ? -(SHOW_NITS_VALUES_LINE_COUNT
                              + SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (_SHOW_NITS_VALUES)  && !_SHOW_NITS_FROM_CURSOR && !SHOW_CSPS
                            ? -(SHOW_NITS_FROM_CURSOR_LINE_COUNT
                              + SHOW_CSPS_LINE_COUNT
                              - SPACING_MULTIPLIER)

                            : (!_SHOW_NITS_VALUES) && !_SHOW_NITS_FROM_CURSOR && !SHOW_CSPS
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
#endif //IS_HDR_CSP

    float4 bgCol = tex2Dfetch(SamplerFontAtlasConsolidated, int2(0, 0));

    uint activeLines = GetActiveLines();

    uint activeCharacters = GetActiveCharacters();

    static const uint charArrayEntry = GetCharArrayEntry();

    uint2 charSize = GetCharSize(charArrayEntry);

    uint outerSpacing = GetOuterSpacing(charSize.x);

    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);

    activeTextArea.y += uint(max(_SHOW_NITS_VALUES
                               + _SHOW_NITS_FROM_CURSOR
#ifdef IS_HDR_CSP
                               + SHOW_CSPS
                               + SHOW_CSP_FROM_CURSOR
#endif
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

    static const float showMaxNitsValueYOffset      =  1.f + CSP_DESC_SPACING_MULTIPLIER;
    static const float showAvgNitsValueYOffset      =  2.f + CSP_DESC_SPACING_MULTIPLIER;
    static const float showMinNitsValueYOffset      =  3.f + CSP_DESC_SPACING_MULTIPLIER;
                       cursorNitsYOffset            =  4.f + cursorNitsYOffset;
#ifdef IS_HDR_CSP
    static const float cspsBt709PercentageYOffset   =  5.f + cspsYOffset;
    static const float cspsDciP3PercentageYOffset   =  6.f + cspsYOffset;
    static const float cspsBt2020PercentageYOffset  =  7.f + cspsYOffset;
#ifdef IS_FLOAT_HDR_CSP
    static const float cspsAp0PercentageYOffset     =  8.f + cspsYOffset;
    static const float cspsInvalidPercentageYOffset =  9.f + cspsYOffset;
#endif
                       cursorCspYOffset             = 10.f + cursorCspYOffset;
#endif

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
  #if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
    || OVERWRITE_SDR_GAMMA == GAMMA_22    \
    || OVERWRITE_SDR_GAMMA == GAMMA_24)
    DrawChar(_g,   float2(14, 0));
    DrawChar(_a,   float2(15, 0));
    DrawChar(_m,   float2(16, 0));
    DrawChar(_m,   float2(17, 0));
    DrawChar(_a,   float2(18, 0));
    DrawChar(_2,   float2(20, 0));
    DrawChar(_dot, float2(21, 0));
    #if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
      || OVERWRITE_SDR_GAMMA == GAMMA_22)
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

#ifdef IS_HDR_CSP
  #define SHOW_NITS_VALUES_DOT_X_OFFSET 14 // 5 figure number
#else
  #define SHOW_NITS_VALUES_DOT_X_OFFSET 12 // 3 figure number

  #define SHOW_MAX_AVG_NITS_VALUES_PERCENT_X_OFFSET (SHOW_NITS_VALUES_DOT_X_OFFSET + 4)
  #define SHOW_MIN_NITS_VALUES_PERCENT_X_OFFSET     (SHOW_NITS_VALUES_DOT_X_OFFSET + 7)
#endif

    // max/avg/min Nits
    if (_SHOW_NITS_VALUES)
    {
      // maxNits:
      DrawChar(_m,       float2( 0,                                        showMaxNitsValueYOffset));
      DrawChar(_a,       float2( 1,                                        showMaxNitsValueYOffset));
      DrawChar(_x,       float2( 2,                                        showMaxNitsValueYOffset));
      DrawChar(_N,       float2( 3,                                        showMaxNitsValueYOffset));
      DrawChar(_i,       float2( 4,                                        showMaxNitsValueYOffset));
      DrawChar(_t,       float2( 5,                                        showMaxNitsValueYOffset));
      DrawChar(_s,       float2( 6,                                        showMaxNitsValueYOffset));
      DrawChar(_colon,   float2( 7,                                        showMaxNitsValueYOffset));
      DrawChar(_dot,     float2(SHOW_NITS_VALUES_DOT_X_OFFSET,             showMaxNitsValueYOffset));
#ifndef IS_HDR_CSP
      DrawChar(_percent, float2(SHOW_MAX_AVG_NITS_VALUES_PERCENT_X_OFFSET, showMaxNitsValueYOffset));
#endif
      // avgNits:
      DrawChar(_a,       float2( 0,                                        showAvgNitsValueYOffset));
      DrawChar(_v,       float2( 1,                                        showAvgNitsValueYOffset));
      DrawChar(_g,       float2( 2,                                        showAvgNitsValueYOffset));
      DrawChar(_N,       float2( 3,                                        showAvgNitsValueYOffset));
      DrawChar(_i,       float2( 4,                                        showAvgNitsValueYOffset));
      DrawChar(_t,       float2( 5,                                        showAvgNitsValueYOffset));
      DrawChar(_s,       float2( 6,                                        showAvgNitsValueYOffset));
      DrawChar(_colon,   float2( 7,                                        showAvgNitsValueYOffset));
      DrawChar(_dot,     float2(SHOW_NITS_VALUES_DOT_X_OFFSET,             showAvgNitsValueYOffset));
#ifndef IS_HDR_CSP
      DrawChar(_percent, float2(SHOW_MAX_AVG_NITS_VALUES_PERCENT_X_OFFSET, showAvgNitsValueYOffset));
#endif
      // minNits:
      DrawChar(_m,       float2( 0,                                    showMinNitsValueYOffset));
      DrawChar(_i,       float2( 1,                                    showMinNitsValueYOffset));
      DrawChar(_n,       float2( 2,                                    showMinNitsValueYOffset));
      DrawChar(_N,       float2( 3,                                    showMinNitsValueYOffset));
      DrawChar(_i,       float2( 4,                                    showMinNitsValueYOffset));
      DrawChar(_t,       float2( 5,                                    showMinNitsValueYOffset));
      DrawChar(_s,       float2( 6,                                    showMinNitsValueYOffset));
      DrawChar(_colon,   float2( 7,                                    showMinNitsValueYOffset));
      DrawChar(_dot,     float2(SHOW_NITS_VALUES_DOT_X_OFFSET,         showMinNitsValueYOffset));
#ifndef IS_HDR_CSP
      DrawChar(_percent, float2(SHOW_MIN_NITS_VALUES_PERCENT_X_OFFSET, showMinNitsValueYOffset));
#endif
    }

#ifdef IS_HDR_CSP
  #define SHOW_NITS_FROM_CURSOR_DOT_X_OFFSET 17 // 5 figure number
#else
  #define SHOW_NITS_FROM_CURSOR_DOT_X_OFFSET 15 // 3 figure number
#endif

    // cursorNits:
    if (_SHOW_NITS_FROM_CURSOR)
    {
      DrawChar(_c,       float2( 0,                                     cursorNitsYOffset));
      DrawChar(_u,       float2( 1,                                     cursorNitsYOffset));
      DrawChar(_r,       float2( 2,                                     cursorNitsYOffset));
      DrawChar(_s,       float2( 3,                                     cursorNitsYOffset));
      DrawChar(_o,       float2( 4,                                     cursorNitsYOffset));
      DrawChar(_r,       float2( 5,                                     cursorNitsYOffset));
      DrawChar(_N,       float2( 6,                                     cursorNitsYOffset));
      DrawChar(_i,       float2( 7,                                     cursorNitsYOffset));
      DrawChar(_t,       float2( 8,                                     cursorNitsYOffset));
      DrawChar(_s,       float2( 9,                                     cursorNitsYOffset));
      DrawChar(_colon,   float2(10,                                     cursorNitsYOffset));
      DrawChar(_dot,     float2(SHOW_NITS_FROM_CURSOR_DOT_X_OFFSET,     cursorNitsYOffset));
#ifndef IS_HDR_CSP
      DrawChar(_percent, float2(SHOW_NITS_FROM_CURSOR_DOT_X_OFFSET + 7, cursorNitsYOffset));
#endif
    }

#ifdef IS_HDR_CSP
    // CSPs
    if (SHOW_CSPS)
    {
      // BT.709:
      DrawChar(_B,       float2( 0, cspsBt709PercentageYOffset));
      DrawChar(_T,       float2( 1, cspsBt709PercentageYOffset));
      DrawChar(_dot,     float2( 2, cspsBt709PercentageYOffset));
      DrawChar(_7,       float2( 3, cspsBt709PercentageYOffset));
      DrawChar(_0,       float2( 4, cspsBt709PercentageYOffset));
      DrawChar(_9,       float2( 5, cspsBt709PercentageYOffset));
      DrawChar(_colon,   float2( 6, cspsBt709PercentageYOffset));
      DrawChar(_dot,     float2(12, cspsBt709PercentageYOffset));
      DrawChar(_percent, float2(15, cspsBt709PercentageYOffset));
      // DCI-P3:
      DrawChar(_D,       float2( 0, cspsDciP3PercentageYOffset));
      DrawChar(_C,       float2( 1, cspsDciP3PercentageYOffset));
      DrawChar(_I,       float2( 2, cspsDciP3PercentageYOffset));
      DrawChar(_minus,   float2( 3, cspsDciP3PercentageYOffset));
      DrawChar(_P,       float2( 4, cspsDciP3PercentageYOffset));
      DrawChar(_3,       float2( 5, cspsDciP3PercentageYOffset));
      DrawChar(_colon,   float2( 6, cspsDciP3PercentageYOffset));
      DrawChar(_dot,     float2(12, cspsDciP3PercentageYOffset));
      DrawChar(_percent, float2(15, cspsDciP3PercentageYOffset));
      // BT.2020:
      DrawChar(_B,       float2( 0, cspsBt2020PercentageYOffset));
      DrawChar(_T,       float2( 1, cspsBt2020PercentageYOffset));
      DrawChar(_dot,     float2( 2, cspsBt2020PercentageYOffset));
      DrawChar(_2,       float2( 3, cspsBt2020PercentageYOffset));
      DrawChar(_0,       float2( 4, cspsBt2020PercentageYOffset));
      DrawChar(_2,       float2( 5, cspsBt2020PercentageYOffset));
      DrawChar(_0,       float2( 6, cspsBt2020PercentageYOffset));
      DrawChar(_colon,   float2( 7, cspsBt2020PercentageYOffset));
      DrawChar(_dot,     float2(12, cspsBt2020PercentageYOffset));
      DrawChar(_percent, float2(15, cspsBt2020PercentageYOffset));
#ifdef IS_FLOAT_HDR_CSP
      // AP0:
      DrawChar(_A,       float2( 0, cspsAp0PercentageYOffset));
      DrawChar(_P,       float2( 1, cspsAp0PercentageYOffset));
      DrawChar(_0,       float2( 2, cspsAp0PercentageYOffset));
      DrawChar(_colon,   float2( 3, cspsAp0PercentageYOffset));
      DrawChar(_dot,     float2(12, cspsAp0PercentageYOffset));
      DrawChar(_percent, float2(15, cspsAp0PercentageYOffset));
      // invalid:
      DrawChar(_i,       float2( 0, cspsInvalidPercentageYOffset));
      DrawChar(_n,       float2( 1, cspsInvalidPercentageYOffset));
      DrawChar(_v,       float2( 2, cspsInvalidPercentageYOffset));
      DrawChar(_a,       float2( 3, cspsInvalidPercentageYOffset));
      DrawChar(_l,       float2( 4, cspsInvalidPercentageYOffset));
      DrawChar(_i,       float2( 5, cspsInvalidPercentageYOffset));
      DrawChar(_d,       float2( 6, cspsInvalidPercentageYOffset));
      DrawChar(_colon,   float2( 7, cspsInvalidPercentageYOffset));
      DrawChar(_dot,     float2(12, cspsInvalidPercentageYOffset));
      DrawChar(_percent, float2(15, cspsInvalidPercentageYOffset));
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
#endif

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


#define showMaxNitsValueYOffset       1.f + CSP_DESC_SPACING_MULTIPLIER
#define showAvgNitsValueYOffset       2.f + CSP_DESC_SPACING_MULTIPLIER
#define showMinNitsValueYOffset       3.f + CSP_DESC_SPACING_MULTIPLIER
#define cursorNitsYOffset             4.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS)
#define cspsBt709PercentageYOffset    5.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsDciP3PercentageYOffset    6.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsBt2020PercentageYOffset   7.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsAp0PercentageYOffset      8.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsInvalidPercentageYOffset  9.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cursorCspYOffset             10.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP)


#define DRAW_NUMBERS(SHOW_IT, NUMBER, FETCH_COORDS, DRAW_TYPE, DRAW_OFFSET)   \
  if (SHOW_IT)                                                                \
  {                                                                           \
    precise float fullNumber = tex2Dfetch(StorageConsolidated, FETCH_COORDS); \
    precise uint  curNumber  = NUMBER(fullNumber);                            \
    DRAW_TYPE(curNumber, DRAW_OFFSET);                                        \
  }                                                                           \
  return

#define DRAW_CURSOR_NITS(NUMBER, X_OFFSET, DRAW_TYPE)                         \
  if (_SHOW_NITS_FROM_CURSOR)                                                  \
  {                                                                           \
    precise float cursorNits = tex2Dfetch(StorageNitsValues, MOUSE_POSITION); \
    precise uint  curNumber  = NUMBER(cursorNits);                            \
    DRAW_TYPE(curNumber, float2(X_OFFSET, cursorNitsYOffset));                \
  }                                                                           \
  return


#ifdef IS_HDR_CSP
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET0   9
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET1  10
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET2  11
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET3  12
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET4  13
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET5  15
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET6  16
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET7  17
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET8  18
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET9  19
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET10 20

  #define DRAW_CURSOR_NITS_OFFSET0  12
  #define DRAW_CURSOR_NITS_OFFSET1  13
  #define DRAW_CURSOR_NITS_OFFSET2  14
  #define DRAW_CURSOR_NITS_OFFSET3  15
  #define DRAW_CURSOR_NITS_OFFSET4  16
  #define DRAW_CURSOR_NITS_OFFSET5  18
  #define DRAW_CURSOR_NITS_OFFSET6  19
  #define DRAW_CURSOR_NITS_OFFSET7  20
  #define DRAW_CURSOR_NITS_OFFSET8  21
  #define DRAW_CURSOR_NITS_OFFSET9  22
  #define DRAW_CURSOR_NITS_OFFSET10 23
#else
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET2   9
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET3  10
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET4  11
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET5  13
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET6  14
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET7  15
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET8  16
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET9  17
  #define DRAW_MAX_AVG_MIN_NITS_OFFSET10 18

  #define DRAW_CURSOR_NITS_OFFSET2  12
  #define DRAW_CURSOR_NITS_OFFSET3  13
  #define DRAW_CURSOR_NITS_OFFSET4  14
  #define DRAW_CURSOR_NITS_OFFSET5  16
  #define DRAW_CURSOR_NITS_OFFSET6  17
  #define DRAW_CURSOR_NITS_OFFSET7  18
  #define DRAW_CURSOR_NITS_OFFSET8  19
  #define DRAW_CURSOR_NITS_OFFSET9  20
  #define DRAW_CURSOR_NITS_OFFSET10 21
#endif


void CS_DrawValuesToOverlay(uint3 ID : SV_DispatchThreadID)
{
  switch(ID.x)
  {
    // max/avg/min Nits
    // maxNits:
#ifdef IS_HDR_CSP
    case 0:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _5th, COORDS_SHOW_MAX_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET0, showMaxNitsValueYOffset));
    }
    case 1:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _4th, COORDS_SHOW_MAX_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET1, showMaxNitsValueYOffset));
    }
#endif
    case 2:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _3rd, COORDS_SHOW_MAX_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET2, showMaxNitsValueYOffset));
    }
    case 3:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _2nd, COORDS_SHOW_MAX_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET3, showMaxNitsValueYOffset));
    }
    case 4:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _1st, COORDS_SHOW_MAX_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET4, showMaxNitsValueYOffset));
    }
    case 5:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d1st, COORDS_SHOW_MAX_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET5, showMaxNitsValueYOffset));
    }
    case 6:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d2nd, COORDS_SHOW_MAX_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET6, showMaxNitsValueYOffset));
    }
    case 7:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d3rd, COORDS_SHOW_MAX_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET7, showMaxNitsValueYOffset));
    }
    // avgNits:
#ifdef IS_HDR_CSP
    case 8:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _5th, COORDS_SHOW_AVG_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET0, showAvgNitsValueYOffset));
    }
    case 9:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _4th, COORDS_SHOW_AVG_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET1, showAvgNitsValueYOffset));
    }
#endif
    case 10:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _3rd, COORDS_SHOW_AVG_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET2, showAvgNitsValueYOffset));
    }
    case 11:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _2nd, COORDS_SHOW_AVG_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET3, showAvgNitsValueYOffset));
    }
    case 12:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _1st, COORDS_SHOW_AVG_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET4, showAvgNitsValueYOffset));
    }
    case 13:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d1st, COORDS_SHOW_AVG_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET5, showAvgNitsValueYOffset));
    }
    case 14:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d2nd, COORDS_SHOW_AVG_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET6, showAvgNitsValueYOffset));
    }
    case 15:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d3rd, COORDS_SHOW_AVG_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET7, showAvgNitsValueYOffset));
    }
#ifdef IS_HDR_CSP
    // minNits:
    case 16:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _5th, COORDS_SHOW_MIN_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET0,  showMinNitsValueYOffset));
    }
    case 17:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _4th, COORDS_SHOW_MIN_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET1,  showMinNitsValueYOffset));
    }
#endif
    case 18:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _3rd, COORDS_SHOW_MIN_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET2,  showMinNitsValueYOffset));
    }
    case 19:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _2nd, COORDS_SHOW_MIN_NITS, DrawNumberAboveZero, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET3,  showMinNitsValueYOffset));
    }
    case 20:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, _1st, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET4,  showMinNitsValueYOffset));
    }
    case 21:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d1st, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET5,  showMinNitsValueYOffset));
    }
    case 22:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d2nd, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET6,  showMinNitsValueYOffset));
    }
    case 23:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d3rd, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET7,  showMinNitsValueYOffset));
    }
    case 24:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d4th, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET8,  showMinNitsValueYOffset));
    }
    case 25:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d5th, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET9,  showMinNitsValueYOffset));
    }
    case 26:
    {
      DRAW_NUMBERS(_SHOW_NITS_VALUES, d6th, COORDS_SHOW_MIN_NITS, DrawChar, float2(DRAW_MAX_AVG_MIN_NITS_OFFSET10, showMinNitsValueYOffset));
    }
    // cursorNits:
#ifdef IS_HDR_CSP
    case 27:
    {
      DRAW_CURSOR_NITS(_5th, DRAW_CURSOR_NITS_OFFSET0, DrawNumberAboveZero);
    }
    case 28:
    {
      DRAW_CURSOR_NITS(_4th, DRAW_CURSOR_NITS_OFFSET1, DrawNumberAboveZero);
    }
#endif
    case 29:
    {
      DRAW_CURSOR_NITS(_3rd, DRAW_CURSOR_NITS_OFFSET2, DrawNumberAboveZero);
    }
    case 30:
    {
      DRAW_CURSOR_NITS(_2nd, DRAW_CURSOR_NITS_OFFSET3, DrawNumberAboveZero);
    }
    case 31:
    {
      DRAW_CURSOR_NITS(_1st, DRAW_CURSOR_NITS_OFFSET4, DrawChar);
    }
    case 32:
    {
      DRAW_CURSOR_NITS(d1st, DRAW_CURSOR_NITS_OFFSET5, DrawChar);
    }
    case 33:
    {
      DRAW_CURSOR_NITS(d2nd, DRAW_CURSOR_NITS_OFFSET6, DrawChar);
    }
    case 34:
    {
      DRAW_CURSOR_NITS(d3rd, DRAW_CURSOR_NITS_OFFSET7, DrawChar);
    }
    case 35:
    {
      DRAW_CURSOR_NITS(d4th, DRAW_CURSOR_NITS_OFFSET8, DrawChar);
    }
    case 36:
    {
      DRAW_CURSOR_NITS(d5th, DRAW_CURSOR_NITS_OFFSET9, DrawChar);
    }
    case 37:
    {
      DRAW_CURSOR_NITS(d6th, DRAW_CURSOR_NITS_OFFSET10, DrawChar);
    }


#ifdef IS_HDR_CSP
    // show CSPs
    // BT.709:
    case 38:
    {
      DRAW_NUMBERS(SHOW_CSPS, _3rd, COORDS_SHOW_PERCENTAGE_BT709, DrawNumberAboveZero, float2( 9, cspsBt709PercentageYOffset));
    }
    case 39:
    {
      DRAW_NUMBERS(SHOW_CSPS, _2nd, COORDS_SHOW_PERCENTAGE_BT709, DrawNumberAboveZero, float2(10, cspsBt709PercentageYOffset));
    }
    case 40:
    {
      DRAW_NUMBERS(SHOW_CSPS, _1st, COORDS_SHOW_PERCENTAGE_BT709, DrawChar, float2(11, cspsBt709PercentageYOffset));
    }
    case 41:
    {
      DRAW_NUMBERS(SHOW_CSPS, d1st, COORDS_SHOW_PERCENTAGE_BT709, DrawChar, float2(13, cspsBt709PercentageYOffset));
    }
    case 42:
    {
      DRAW_NUMBERS(SHOW_CSPS, d2nd, COORDS_SHOW_PERCENTAGE_BT709, DrawChar, float2(14, cspsBt709PercentageYOffset));
    }
    // DCI-P3:
    case 43:
    {
      DRAW_NUMBERS(SHOW_CSPS, _3rd, COORDS_SHOW_PERCENTAGE_DCI_P3, DrawNumberAboveZero, float2( 9, cspsDciP3PercentageYOffset));
    }
    case 44:
    {
      DRAW_NUMBERS(SHOW_CSPS, _2nd, COORDS_SHOW_PERCENTAGE_DCI_P3, DrawNumberAboveZero, float2(10, cspsDciP3PercentageYOffset));
    }
    case 45:
    {
      DRAW_NUMBERS(SHOW_CSPS, _1st, COORDS_SHOW_PERCENTAGE_DCI_P3, DrawChar, float2(11, cspsDciP3PercentageYOffset));
    }
    case 46:
    {
      DRAW_NUMBERS(SHOW_CSPS, d1st, COORDS_SHOW_PERCENTAGE_DCI_P3, DrawChar, float2(13, cspsDciP3PercentageYOffset));
    }
    case 47:
    {
      DRAW_NUMBERS(SHOW_CSPS, d2nd, COORDS_SHOW_PERCENTAGE_DCI_P3, DrawChar, float2(14, cspsDciP3PercentageYOffset));
    }
    // BT.2020:
    case 48:
    {
      DRAW_NUMBERS(SHOW_CSPS, _3rd, COORDS_SHOW_PERCENTAGE_BT2020, DrawNumberAboveZero, float2( 9, cspsBt2020PercentageYOffset));
    }
    case 49:
    {
      DRAW_NUMBERS(SHOW_CSPS, _2nd, COORDS_SHOW_PERCENTAGE_BT2020, DrawNumberAboveZero, float2(10, cspsBt2020PercentageYOffset));
    }
    case 50:
    {
      DRAW_NUMBERS(SHOW_CSPS, _1st, COORDS_SHOW_PERCENTAGE_BT2020, DrawChar, float2(11, cspsBt2020PercentageYOffset));
    }
    case 51:
    {
      DRAW_NUMBERS(SHOW_CSPS, d1st, COORDS_SHOW_PERCENTAGE_BT2020, DrawChar, float2(13, cspsBt2020PercentageYOffset));
    }
    case 52:
    {
      DRAW_NUMBERS(SHOW_CSPS, d2nd, COORDS_SHOW_PERCENTAGE_BT2020, DrawChar, float2(14, cspsBt2020PercentageYOffset));
    }
#ifdef IS_FLOAT_HDR_CSP
    // AP0:
    case 53:
    {
      DRAW_NUMBERS(SHOW_CSPS, _3rd, COORDS_SHOW_PERCENTAGE_AP0, DrawNumberAboveZero, float2( 9, cspsAp0PercentageYOffset));
    }
    case 54:
    {
      DRAW_NUMBERS(SHOW_CSPS, _2nd, COORDS_SHOW_PERCENTAGE_AP0, DrawNumberAboveZero, float2(10, cspsAp0PercentageYOffset));
    }
    case 55:
    {
      DRAW_NUMBERS(SHOW_CSPS, _1st, COORDS_SHOW_PERCENTAGE_AP0, DrawChar, float2(11, cspsAp0PercentageYOffset));
    }
    case 56:
    {
      DRAW_NUMBERS(SHOW_CSPS, d1st, COORDS_SHOW_PERCENTAGE_AP0, DrawChar, float2(13, cspsAp0PercentageYOffset));
    }
    case 57:
    {
      DRAW_NUMBERS(SHOW_CSPS, d2nd, COORDS_SHOW_PERCENTAGE_AP0, DrawChar, float2(14, cspsAp0PercentageYOffset));
    }
    // invalid:
    case 58:
    {
      DRAW_NUMBERS(SHOW_CSPS, _3rd, COORDS_SHOW_PERCENTAGE_INVALID, DrawNumberAboveZero, float2( 9, cspsInvalidPercentageYOffset));
    }
    case 59:
    {
      DRAW_NUMBERS(SHOW_CSPS, _2nd, COORDS_SHOW_PERCENTAGE_INVALID, DrawNumberAboveZero, float2(10, cspsInvalidPercentageYOffset));
    }
    case 60:
    {
      DRAW_NUMBERS(SHOW_CSPS, _1st, COORDS_SHOW_PERCENTAGE_INVALID, DrawChar, float2(11, cspsInvalidPercentageYOffset));
    }
    case 61:
    {
      DRAW_NUMBERS(SHOW_CSPS, d1st, COORDS_SHOW_PERCENTAGE_INVALID, DrawChar, float2(13, cspsInvalidPercentageYOffset));
    }
    case 62:
    {
      DRAW_NUMBERS(SHOW_CSPS, d2nd, COORDS_SHOW_PERCENTAGE_INVALID, DrawChar, float2(14, cspsInvalidPercentageYOffset));
    }
#endif
    //cursorCSP:
    case 63:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = float2(11, cursorCspYOffset);

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_B, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_D, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_B, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_T, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_C, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_T, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_dot, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_I, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_dot, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_7, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_minus, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_P, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709)
        {
          DrawChar(_9, currentCursorCspOffset);
          return;
        }
        else if (mousePosCsp == IS_CSP_DCI_P3)
        {
          DrawChar(_3, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_AP0)
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

        uint mousePosCsp = uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f);

        if (mousePosCsp == IS_CSP_BT709
         || mousePosCsp == IS_CSP_DCI_P3
#ifdef IS_FLOAT_HDR_CSP
         || mousePosCsp == IS_CSP_AP0
#endif
        )
        {
          DrawSpace(currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (mousePosCsp == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset);
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else //invalid
        {
          DrawChar(_d, currentCursorCspOffset);
          return;
        }
#endif
      }
      return;
    }
#endif
    default:
      return;
  }
  return;
}

#undef showMaxNitsValueYOffset
#undef showAvgNitsValueYOffset
#undef showMinNitsValueYOffset
#undef cursorNitsYOffset
#undef cspsBt709PercentageYOffset
#undef cspsDciP3PercentageYOffset
#undef cspsBt2020PercentageYOffset
#undef cspsAp0PercentageYOffset
#undef cspsInvalidPercentageYOffset
#undef cursorCspYOffset


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareSetActiveArea(
  in                  uint   Id                : SV_VertexID,
  out                 float4 VPos              : SV_Position,
  out nointerpolation float4 PercentagesToCrop : PercentagesToCrop)
{
  float2 TexCoord;
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);


#define percentageToCropFromLeft   PercentagesToCrop.x
#define percentageToCropFromTop    PercentagesToCrop.y
#define percentageToCropFromRight  PercentagesToCrop.z
#define percentageToCropFromBottom PercentagesToCrop.w

  float fractionCropLeft   = _ACTIVE_AREA_CROP_LEFT   / 100.f;
  float fractionCropTop    = _ACTIVE_AREA_CROP_TOP    / 100.f;
  float fractionCropRight  = _ACTIVE_AREA_CROP_RIGHT  / 100.f;
  float fractionCropBottom = _ACTIVE_AREA_CROP_BOTTOM / 100.f;

  percentageToCropFromLeft   =                 fractionCropLeft   * BUFFER_WIDTH;
  percentageToCropFromTop    =                 fractionCropTop    * BUFFER_HEIGHT;
  percentageToCropFromRight  = BUFFER_WIDTH  - fractionCropRight  * BUFFER_WIDTH;
  percentageToCropFromBottom = BUFFER_HEIGHT - fractionCropBottom * BUFFER_HEIGHT;

}

void PS_SetActiveArea(
  in                  float4 VPos              : SV_Position,
  in  nointerpolation float4 PercentagesToCrop : PercentagesToCrop,
  out                 float4 Output            : SV_Target0)
{
  Output = 0.f;

  if (_ACTIVE_AREA_ENABLE)
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
#if (!defined(GAMESCOPE) \
  && !defined(POSSIBLE_DECK_VULKAN_USAGE))
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

#if (!defined(GAMESCOPE) \
  && !defined(POSSIBLE_DECK_VULKAN_USAGE))
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

  if (_HIGHLIGHT_NIT_RANGE)
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

      highlightNitRangeOut = MapBt709IntoCurrentCsp(highlightNitRangeOut, _HIGHLIGHT_NIT_RANGE_BRIGHTNESS);
    }
  }

  if (_SHOW_LUMINANCE_WAVEFORM)
  {
    LuminanceWaveformTextureDisplayAreaBegin = int2(BUFFER_WIDTH, BUFFER_HEIGHT) - Waveform::GetActiveArea();
  }

  if (_SHOW_CIE)
  {
    float cieDiagramSizeFrac = _CIE_DIAGRAM_SIZE / 100.f;

    CieDiagramTextureActiveSize =
      round(float2(CIE_BG_WIDTH[_CIE_DIAGRAM_TYPE], CIE_BG_HEIGHT[_CIE_DIAGRAM_TYPE]) * cieDiagramSizeFrac);

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

    currentOverlayDimensions.y += uint(max(_SHOW_NITS_VALUES
                                         + _SHOW_NITS_FROM_CURSOR
#ifdef IS_HDR_CSP
                                         + SHOW_CSPS
                                         + SHOW_CSP_FROM_CURSOR
#endif
                                         - 1, 0)
                                     * charSize.y * SPACING_MULTIPLIER);

    if (_SHOW_NITS_VALUES
     || _SHOW_NITS_FROM_CURSOR
#ifdef IS_HDR_CSP
     || SHOW_CSPS
     || SHOW_CSP_FROM_CURSOR
#endif
    )
    {
      currentOverlayDimensions.y += charSize.y * CSP_DESC_SPACING_MULTIPLIER;
    }

    currentOverlayDimensions += outerSpacing + outerSpacing;

    if (_TEXT_POSITION == TEXT_POSITION_TOP_RIGHT)
    {
      currentOverlayDimensions.x = uint(BUFFER_WIDTH) - currentOverlayDimensions.x;
    }

    CurrentActiveOverlayArea = int2(currentOverlayDimensions);
  }

}


void ExtendedReinhardTmo(
  inout float3 Colour,
  in    float  WhitePoint)
{
#ifdef IS_HDR_CSP
  float maxWhite = 10000.f / WhitePoint;
#else
  float maxWhite = 100.f / WhitePoint;
#endif

  Colour = (Colour * (1.f + (Colour / (maxWhite * maxWhite))))
         / (1.f + Colour);
}

void MergeOverlay(
  inout float3 Output,
  in    float3 Overlay,
  in    float  OverlayBrightness,
  in    float  Alpha)
{
#ifdef IS_HDR_CSP
  Overlay = Csp::Mat::Bt709To::Bt2020(Overlay);
#endif

  // tone map pixels below the overlay area
  //
  // first set 1.0 to be equal to OverlayBrightness
  float adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  adjustFactor = OverlayBrightness / 80.f;

  Output = Csp::Mat::Bt709To::Bt2020(Output / adjustFactor);

  // safety clamp colours outside of BT.2020
  Output = max(Output, 0.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  adjustFactor = OverlayBrightness / 10000.f;

  Output = Csp::Trc::PqTo::Linear(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  adjustFactor = OverlayBrightness / 100.f;

  Output = DECODE_SDR(Output);

#endif

#if (ACTUAL_COLOUR_SPACE != CSP_SCRGB)

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

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  Output = ENCODE_SDR(Output);

#endif
}


void PS_HdrAnalysis(
  in                  float4 VPos              : SV_Position,
  in                  float2 TexCoord          : TEXCOORD0,
  in  nointerpolation bool2  PingPongChecks    : PingPongChecks,
  in  nointerpolation float4 HighlightNitRange : HighlightNitRange,
#if (!defined(GAMESCOPE) \
  && !defined(POSSIBLE_DECK_VULKAN_USAGE))
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

  if (_SHOW_HEATMAP
#ifdef IS_HDR_CSP
   || SHOW_CSP_MAP
#endif
   || _HIGHLIGHT_NIT_RANGE
   || _DRAW_ABOVE_NITS_AS_BLACK
   || _DRAW_BELOW_NITS_AS_BLACK)
  {
    const float pixelNits = tex2Dfetch(SamplerNitsValues, pureCoordAsInt);

    if (_SHOW_HEATMAP)
    {
      Output.rgb = HeatmapRgbValues(pixelNits,
#ifdef IS_HDR_CSP
                                    HEATMAP_CUTOFF_POINT,
#endif
                                    false);

      Output.rgb = MapBt709IntoCurrentCsp(Output.rgb, _HEATMAP_BRIGHTNESS);
    }

#ifdef IS_HDR_CSP
    if (SHOW_CSP_MAP)
    {
      Output.rgb = CreateCspMap(tex2Dfetch(SamplerCsps, pureCoordAsInt).x * 255.f, pixelNits);
    }
#endif

    if (_HIGHLIGHT_NIT_RANGE
     && pixelNits >= _HIGHLIGHT_NIT_RANGE_START_POINT
     && pixelNits <= _HIGHLIGHT_NIT_RANGE_END_POINT
     && pingpong0Above1
     && breathingIsActive)
    {
      //Output.rgb = HighlightNitRangeOut;
      Output.rgb = lerp(Output.rgb, highlightNitRangeOut, breathing);
    }

    if (_DRAW_ABOVE_NITS_AS_BLACK)
    {
      if (pixelNits > _ABOVE_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }

    if (_DRAW_BELOW_NITS_AS_BLACK)
    {
      if (pixelNits < _BELOW_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }
  }

  if (_SHOW_CIE)
  {
    // draw the diagram in the bottom left corner
    if (VPos.x <  CieDiagramTextureActiveSize.x
     && VPos.y >= CieDiagramTextureActiveSize.y)
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(VPos.x,
                                           VPos.y - CieDiagramTextureActiveSize.y);

      float2 currentCieSamplerCoords = currentSamplerCoords / CieDiagramTextureDisplaySize;

      float3 currentPixelToDisplay = tex2D(SamplerCieCurrent, currentCieSamplerCoords).rgb;

      // using gamma 2 as intermediate gamma space
      currentPixelToDisplay *= currentPixelToDisplay;

      float alpha = min(ceil(MAXRGB(currentPixelToDisplay)) + _CIE_DIAGRAM_ALPHA / 100.f, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay,
                   _CIE_DIAGRAM_BRIGHTNESS,
                   alpha);
    }
  }

  if (_SHOW_LUMINANCE_WAVEFORM)
  {
    // draw the waveform in the bottom right corner
    if (all(pureCoordAsInt.xy >= LuminanceWaveformTextureDisplayAreaBegin))
    {
      // get fetch coords
      int2 currentFetchCoords = pureCoordAsInt.xy - LuminanceWaveformTextureDisplayAreaBegin;

      float4 currentPixelToDisplay =
        tex2Dfetch(SamplerLuminanceWaveformFinal, currentFetchCoords);

      // using gamma 2 as intermediate gamma space
      currentPixelToDisplay.rgb *= currentPixelToDisplay.rgb;

      float alpha = min(_LUMINANCE_WAVEFORM_ALPHA / 100.f + currentPixelToDisplay.a, 1.f);

      MergeOverlay(Output.rgb,
                   currentPixelToDisplay.rgb,
                   _LUMINANCE_WAVEFORM_BRIGHTNESS,
                   alpha);
    }
  }

  {
    if (_TEXT_POSITION == TEXT_POSITION_TOP_LEFT)
    {
      if (all(pureCoordAsInt <= CurrentActiveOverlayArea))
      {
        float4 overlay = tex2Dfetch(SamplerTextOverlay, pureCoordAsInt).rrrg;

        // using gamma 2 as intermediate gamma space
        overlay.rgb *= overlay.rgb;

        float alpha = min(_TEXT_BG_ALPHA / 100.f + overlay.a, 1.f);

        MergeOverlay(Output.rgb,
                     overlay.rgb,
                     _TEXT_BRIGHTNESS,
                     alpha);
      }
    }
    else
    {
      if (pureCoordAsInt.x >= CurrentActiveOverlayArea.x
       && pureCoordAsInt.y <= CurrentActiveOverlayArea.y)
      {
        float4 overlay = tex2Dfetch(SamplerTextOverlay,
                                    int2(pureCoordAsInt.x - CurrentActiveOverlayArea.x, pureCoordAsInt.y)).rrrg;

        // using gamma 2 as intermediate gamma space
        overlay.rgb *= overlay.rgb;

        float alpha = min(_TEXT_BG_ALPHA / 100.f + overlay.a, 1.f);

        MergeOverlay(Output.rgb,
                     overlay.rgb,
                     _TEXT_BRIGHTNESS,
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
#ifdef IS_HDR_CSP
  ui_label = "Lilium's HDR analysis";
#else
  ui_label = "Lilium's SDR analysis";
#endif
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
    VertexShader       = VS_PostProcessWithoutTexCoord;
     PixelShader       = PS_ClearLuminanceWaveformTexture;
    RenderTarget       = TextureLuminanceWaveform;
    ClearRenderTargets = true;
  }

  pass CS_RenderLuminanceWaveform
  {
    ComputeShader = CS_RenderLuminanceWaveform <8, 8>;
    DispatchSizeX = RENDER_WAVEFORM_DISPATCH_X;
    DispatchSizeY = RENDER_WAVEFORM_DISPATCH_Y;
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
#ifdef IS_HDR_CSP
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
#endif


  pass CS_CopyShowValues
  {
    ComputeShader = ShowValuesCopy <1, 1>;
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
#ifdef IS_HDR_CSP
    DispatchSizeX = 70;
#else
    DispatchSizeX = 38;
#endif
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
#ifdef IS_HDR_CSP
  ui_label = "Lilium's HDR analysis (ERROR)";
#else
  ui_label = "Lilium's SDR analysis (ERROR)";
#endif
>
CS_ERROR

#endif //is hdr API and hdr colour space
