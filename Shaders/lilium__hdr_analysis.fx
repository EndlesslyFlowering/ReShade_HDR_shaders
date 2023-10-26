#include "lilium__include/colour_space.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


#undef TEXT_BRIGHTNESS

//#define _DEBUG
//#define _TESTY


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


uniform uint TEXT_SIZE
<
  ui_category = "Global";
  ui_label    = "text size";
  ui_type     = "combo";
  ui_items    = "18\0"
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
                "56\0";
> = 0;

uniform float TEXT_BRIGHTNESS
<
  ui_category = "Global";
  ui_label    = "text brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 140.f;

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

uniform int GLOBAL_SPACER_0
<
  ui_category = "Global";
  ui_label    = " ";
  ui_type     = "radio";
>;


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

uniform int ACTIVE_AREA_SPACER_0
<
  ui_category = "Set Active Area for analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;


// CLL
uniform bool SHOW_CLL_VALUES
<
  ui_category = "Content Light Level analysis";
  ui_label    = "show CLL values";
  ui_tooltip  = "Shows max/avg/min Content Light Levels.";
> = true;

uniform bool SHOW_CLL_FROM_CURSOR
<
  ui_category = "Content Light Level analysis";
  ui_label    = "show CLL value from cursor position";
> = true;

uniform int CLL_SPACER_0
<
  ui_category = "Content Light Level analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;


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

uniform int CSP_SPACER_0
<
  ui_category = "Colour Space analysis";
  ui_label    = " ";
  ui_type     = "radio";
>;


// CIE
uniform bool SHOW_CIE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show CIE diagram";
  ui_tooltip  = "Change diagram type via the \"Preprocessor definition\" 'CIE_DIAGRAM' below."
           "\n" "Possible values are:"
           "\n" "- 'CIE_1931' for the CIE 1931 xy diagram"
           "\n" "- 'CIE_1976' for the CIE 1976 UCS u'v' diagram";
> = true;

uniform float CIE_DIAGRAM_BRIGHTNESS
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

uniform float CIE_DIAGRAM_SIZE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "CIE diagram size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 100.f;

uniform int CIE_SPACER_0
<
  ui_category = "CIE diagram visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;

uniform bool SHOW_CIE_CSP_BT709_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show BT.709 colour space outline";
> = true;

uniform bool SHOW_CIE_CSP_DCI_P3_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show DCI-P3 colour space outline";
> = false;

uniform bool SHOW_CIE_CSP_BT2020_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show BT.2020 colour space outline";
> = false;

#ifdef IS_FLOAT_HDR_CSP

uniform bool SHOW_CIE_CSP_AP0_OUTLINE
<
  ui_category = "CIE diagram visualisation";
  ui_label    = "show AP0 colour space outline";
> = false;

#endif

uniform int CIE_SPACER_1
<
  ui_category = "CIE diagram visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;


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

uniform int HEATMAP_SPACER_0
<
  ui_category = "Heatmap visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;

uniform bool SHOW_BRIGHTNESS_HISTOGRAM
<
  ui_category = "Brightness histogram";
  ui_label    = "show brightness histogram";
  ui_tooltip  = "Brightness histogram paid for by Aemony.";
> = true;

uniform float BRIGHTNESS_HISTOGRAM_BRIGHTNESS
<
  ui_category = "Brightness histogram";
  ui_label    = "brightness histogram brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

uniform float BRIGHTNESS_HISTOGRAM_SIZE
<
  ui_category = "Brightness histogram";
  ui_label    = "brightness histogram size";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 70.f;

uniform int BRIGHTNESS_HISTOGRAM_SPACER_0
<
  ui_category = "Brightness histogram";
  ui_label    = " ";
  ui_type     = "radio";
>;

uniform bool BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE
<
  ui_category = "Brightness histogram";
  ui_label    = "show minCLL line";
  ui_tooltip  = "Show a horizontal line where minCLL is on the histogram."
           "\n" "The line is invisible when minCLL hits 0 nits.";
> = true;

uniform bool BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE
<
  ui_category = "Brightness histogram";
  ui_label    = "show maxCLL line";
  ui_tooltip  = "Show a horizontal line where maxCLL is on the histogram."
           "\n" "The line is invisible when maxCLL hits above 10000 nits.";
> = true;

uniform int BRIGHTNESS_HISTOGRAM_SPACER_1
<
  ui_category = "Brightness histogram";
  ui_label    = " ";
  ui_type     = "radio";
>;

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

uniform int HIGHLIGHT_NIT_RANGE_SPACER_0
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;

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

uniform int HIGHLIGHT_NIT_RANGE_SPACER_1
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = " ";
  ui_type     = "radio";
>;

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

uniform int DRAW_AS_BLACK_SPACER_0
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = " ";
  ui_type     = "radio";
>;

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

uniform int DRAW_AS_BLACK_SPACER_1
<
  ui_category = "Draw certain brightness levels as black";
  ui_label    = " ";
  ui_type     = "radio";
>;


#define HDR_ANALYSIS_ENABLE

#include "lilium__include/hdr_analysis.fxh"

#if (CIE_DIAGRAM == CIE_1931)
  static const uint CIE_BG_WIDTH  = CIE_1931_BG_WIDTH;
  static const uint CIE_BG_HEIGHT = CIE_1931_BG_HEIGHT;
#else
  static const uint CIE_BG_WIDTH  = CIE_1976_BG_WIDTH;
  static const uint CIE_BG_HEIGHT = CIE_1976_BG_HEIGHT;
#endif


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
        Output = float4(mul(Csp::Mat::XYZToBt709, XYZ), 1.f);
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


#define SPACING_MULTIPLIER  0.3f
#define OUTER_SPACING      15.f
#define OUTER_SPACING_X2    2.f * OUTER_SPACING

static const float ShowCllValuesLineCount     = 3;
static const float ShowCllFromCursorLineCount = 1;

#if defined(IS_HDR10_LIKE_CSP)

  static const float ShowCspsLineCount = 3;

#else //IS_HDR10_LIKE_CSP

  static const float ShowCspsLineCount = 5;

#endif //IS_HDR10_LIKE_CSP


void CS_PrepareOverlay(uint3 ID : SV_DispatchThreadID)
{
  float drawCllLast       = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  float drawcursorCllLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

  float floatShowCllValues     = SHOW_CLL_VALUES;
  float floatShowCllFromCrusor = SHOW_CLL_FROM_CURSOR;

  float drawCspsLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  float drawcursorCspLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);

  float floatShowCsps          = SHOW_CSPS;
  float floatShowCspFromCursor = SHOW_CSP_FROM_CURSOR;

  uint fontSizeLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);


  if (
      floatShowCllValues     != drawCllLast
   || floatShowCllFromCrusor != drawcursorCllLast
   || floatShowCsps          != drawCspsLast
   || floatShowCspFromCursor != drawcursorCspLast
   || TEXT_SIZE              != fontSizeLast)
  {
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW,  1);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, floatShowCllValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, floatShowCllFromCrusor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, floatShowCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, floatShowCspFromCursor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, TEXT_SIZE);



    float cursorCllYOffset = (!SHOW_CLL_VALUES
                            ? -ShowCllValuesLineCount
                            : SPACING_MULTIPLIER);
    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL,
               cursorCllYOffset);

    float cspsYOffset = ((!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllValuesLineCount
                         - SPACING_MULTIPLIER)

                       : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllFromCursorLineCount
                         - SPACING_MULTIPLIER)

                       : (!SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllValuesLineCount
                         + ShowCllFromCursorLineCount)

                       : SPACING_MULTIPLIER * 2);

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS,
               cspsYOffset);

    float cursorCspYOffset = ((!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR  && SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowCllFromCursorLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (SHOW_CLL_VALUES  && SHOW_CLL_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowCspsLineCount
                              - SPACING_MULTIPLIER * 2)

                            : (!SHOW_CLL_VALUES && !SHOW_CLL_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCllFromCursorLineCount
                              - SPACING_MULTIPLIER)

                            : (!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCspsLineCount
                              - SPACING_MULTIPLIER)

                            : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowCllFromCursorLineCount
                              + ShowCspsLineCount
                              - SPACING_MULTIPLIER)

                            : (!SHOW_CLL_VALUES && !SHOW_CLL_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCllFromCursorLineCount
                              + ShowCspsLineCount)

#if defined(IS_HDR10_LIKE_CSP)
                            : SPACING_MULTIPLIER * 3) - 2;
#else //IS_HDR10_LIKE_CSP
                            : SPACING_MULTIPLIER * 3);
#endif //IS_HDR10_LIKE_CSP

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP,
               cursorCspYOffset);


    float4 bgCol = tex2Dfetch(SamplerFontAtlasConsolidated, int2(0, 0)).rgba;

    uint activeLines = (SHOW_CLL_VALUES ? ShowCllValuesLineCount
                                        : 0)
                     + (SHOW_CLL_FROM_CURSOR ? ShowCllFromCursorLineCount
                                             : 0)
                     + (SHOW_CSPS ? ShowCspsLineCount
                                  : 0)
                     + (SHOW_CSP_FROM_CURSOR ? 1
                                             : 0);

    uint activeCharacters =
      max(max(max((SHOW_CLL_VALUES ? 25
                                   : 0),
                  (SHOW_CLL_FROM_CURSOR ? 28
                                        : 0)),
                  (SHOW_CSPS ? 16
                             : 0)),
                  (SHOW_CSP_FROM_CURSOR ? 18
                                        : 0));

    static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

    uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);
    activeTextArea.x += OUTER_SPACING_X2;

    activeTextArea.y += (max(SHOW_CLL_VALUES
                           + SHOW_CLL_FROM_CURSOR
                           + SHOW_CSPS
                           + SHOW_CSP_FROM_CURSOR
                           - 1, 0) * charSize.y * SPACING_MULTIPLIER + OUTER_SPACING_X2);

    for (int y = 0; y < TEXTURE_OVERLAY_HEIGHT; y++)
    {
      for (int x = 0; x < TEXTURE_OVERLAY_WIDTH; x++)
      {
        if (!(x < activeTextArea.x
           && y < activeTextArea.y))
        {
          tex2Dstore(StorageTextOverlay, int2(x, y), float4(0.f, 0.f, 0.f, 0.f));
        }
        else
        {
          tex2Dstore(StorageTextOverlay, int2(x, y), bgCol);
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


void DrawChar(uint2 Char, float2 DrawOffset, float2 Id)
{
  static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

  uint2 charSize   = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);
  uint  fontSize   = 19 - TEXT_SIZE;
  uint2 atlasXY    = uint2(fontSize % 5, fontSize / 5) * uint2(FONT_ATLAS_OFFSET_X, FONT_ATLAS_OFFSET_Y);
  uint2 charOffset = Char * charSize + atlasXY;
  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      float4 pixel = tex2Dfetch(SamplerFontAtlasConsolidated, charOffset + currentOffset).rgba;
      tex2Dstore(StorageTextOverlay, (Id + DrawOffset) * charSize + OUTER_SPACING + currentOffset, pixel);
    }
  }
}


void DrawSpace(float2 DrawOffset, float2 Id)
{
  static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

  uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);
  uint  fontSize = 19 - TEXT_SIZE;

  float4 emptyPixel = tex2Dfetch(SamplerFontAtlasConsolidated, int2(0, 0)).rgba;

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      tex2Dstore(StorageTextOverlay, (Id + DrawOffset) * charSize + OUTER_SPACING + currentOffset, emptyPixel);
    }
  }
}


#define cursorCllOffset float2(0.f, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL))
#define cspsOffset      float2(0.f, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS))
#define cursorCspOffset float2(0.f, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP))


void CS_DrawTextToOverlay(uint3 ID : SV_DispatchThreadID)
{

  if (tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW))
  {

#ifdef _DEBUG

    if(ID.x == 0 && ID.y == 0 && ID.z == 0)
    {
      for (uint x = 0; x < 20; x++)
      {
        for (uint y = 0; y < 20; y++)
        {
          if ((x < 5 || x > 14)
           || (y < 5 || y > 14))
          {
            tex2Dstore(StorageTextOverlay, int2(x + 220, y), float54(1.f, 1.f, 1.f, 1.f));
          }
          else
          {
            tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(0.f, 0.f, 0.f, 1.f));
          }
        }
      }
    }

#endif //_DEBUG

    switch(ID.x)
    {
      // max/avg/min CLL
      // maxCLL:
      case 0:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_m, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 1:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_a, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 2:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_x, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 3:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_C, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 4:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 5:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 6:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_colon, float2(0, 0), float2(ID.x, ID.y));
        }
        return;
      }
      case 7:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_dot, float2(6, 0), float2(ID.x, ID.y)); // five figure number
        }
        return;
      }
      case 8:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_n, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 9:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_i, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 10:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_t, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 11:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_s, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      // avgCLL:
      case 12:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_a, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 13:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_v, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 14:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_g, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 15:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_C, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 16:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 17:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 18:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_colon, float2(-12, 1), float2(ID.x, ID.y));
        }
        return;
      }
      case 19:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_dot, float2(6 - 12, 1), float2(ID.x, ID.y)); // five figure number
        }
        return;
      }
      case 20:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_n, float2(7 + 2 - 12, 1), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 21:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_i, float2(7 + 2 - 12, 1), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 22:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_t, float2(7 + 2 - 12, 1), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 23:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_s, float2(7 + 2 - 12, 1), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      // minCLL:
      case 24:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_m, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 25:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_i, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 26:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_n, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 27:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_C, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 28:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 29:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_L, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 30:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_colon, float2(-24, 2), float2(ID.x, ID.y));
        }
        return;
      }
      case 31:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_dot, float2(6 - 24, 2), float2(ID.x, ID.y)); // five figure number
        }
        return;
      }
      case 32:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_n, float2(7 + 6 - 24, 2), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 33:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_i, float2(7 + 6 - 24, 2), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 34:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_t, float2(7 + 6 - 24, 2), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 35:
      {
        if (SHOW_CLL_VALUES)
        {
          DrawChar(_s, float2(7 + 6 - 24, 2), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      // cursorCLL:
      case 36:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_c, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 37:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_u, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 38:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_r, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 39:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_s, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 40:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_o, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 41:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_r, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 42:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_C, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 43:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_L, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 44:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_L, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 45:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_colon, cursorCllOffset + float2(-36, 3), float2(ID.x, ID.y));
        }
        return;
      }
      case 46:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_dot, cursorCllOffset + float2(6 - 36, 3), float2(ID.x, ID.y)); // five figure number
        }
        return;
      }
      case 47:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_n, cursorCllOffset + float2(7 + 6 - 36, 3), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 48:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_i, cursorCllOffset + float2(7 + 6 - 36, 3), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 49:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_t, cursorCllOffset + float2(7 + 6 - 36, 3), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }
      case 50:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          DrawChar(_s, cursorCllOffset + float2(7 + 6 - 36, 3), float2(ID.x, ID.y)); // decimal places offset
        }
        return;
      }


      // CSPs
      // BT.709:
      case 51:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_B, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 52:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_T, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 53:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 54:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_7, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 55:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_0, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 56:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_9, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 57:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_colon, cspsOffset + float2(-51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 58:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(5 - 51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      case 59:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_percent, cspsOffset + float2(7 - 51, 4), float2(ID.x, ID.y));
        }
        return;
      }
      // DCI-P3:
      case 60:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_D, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 61:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_C, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 62:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_I, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 63:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_minus, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 64:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_P, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 65:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_3, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 66:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_colon, cspsOffset + float2(-60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 67:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(5 - 60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      case 68:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_percent, cspsOffset + float2(7 - 60, 5), float2(ID.x, ID.y));
        }
        return;
      }
      // BT.2020:
      case 69:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_B, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 70:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_T, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 71:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 72:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_2, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 73:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_0, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 74:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_2, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 75:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_0, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 76:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_colon, cspsOffset + float2(-69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 77:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(4 - 69, 6), float2(ID.x, ID.y));
        }
        return;
      }
      case 78:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_percent, cspsOffset + float2(6 - 69, 6), float2(ID.x, ID.y));
        }
        return;
      }
#ifdef IS_FLOAT_HDR_CSP
      // AP0:
      case 79:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_A, cspsOffset + float2(-79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      case 80:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_P, cspsOffset + float2(-79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      case 81:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_0, cspsOffset + float2(-79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      case 82:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_colon, cspsOffset + float2(-79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      case 83:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(8 - 79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      case 84:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_percent, cspsOffset + float2(10 - 79, 7), float2(ID.x, ID.y));
        }
        return;
      }
      // invalid:
      case 85:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_i, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 86:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_n, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 87:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_v, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 88:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_a, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 89:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_l, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 90:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_i, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 91:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_d, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 92:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_colon, cspsOffset + float2(-85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 93:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_dot, cspsOffset + float2(4 - 85, 8), float2(ID.x, ID.y));
        }
        return;
      }
      case 94:
      {
        if (SHOW_CSPS)
        {
          DrawChar(_percent, cspsOffset + float2(6 - 85, 8), float2(ID.x, ID.y));
        }
        return;
      }
#endif //IS_FLOAT_HDR_CSP

      // cursorCSP:
      case 95:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_c, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 96:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_u, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 97:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_r, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 98:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_s, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 99:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_o, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 100:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_r, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 101:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_C, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 102:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_S, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 103:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_P, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }
      case 104:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          DrawChar(_colon, cursorCspOffset + float2(-95, 9), float2(ID.x, ID.y));
        }
        return;
      }

      case 105:
      {
        tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW, 0);
        return;
      }
      default:
      {
        return;
      }
    }
  }
  return;
}


#define _6th(Val) Val / 100000.f
#define _5th(Val) Val / 10000.f
#define _4th(Val) Val / 1000.f
#define _3rd(Val) Val / 100.f
#define _2nd(Val) Val / 10.f
#define _1st(Val) Val % 10.f
#define d1st(Val) Val % 1.f *     10.f
#define d2nd(Val) Val % 1.f *    100.f % 10.f
#define d3rd(Val) Val % 1.f *   1000.f % 10.f
#define d4th(Val) Val % 1.f *  10000.f % 10.f
#define d5th(Val) Val % 1.f * 100000.f % 10.f
#define d6th(Val) Val % 1.f * 100000.f % 1.f  * 10.f
#define d7th(Val) Val % 1.f * 100000.f % 0.1f * 100.f


void DrawNumberAboveZero(precise uint CurNumber, float2 Offset, float2 Id)
{
  if (CurNumber > 0)
  {
    DrawChar(uint2(CurNumber % 10, 0), Offset, Id);
  }
  else
  {
    DrawSpace(Offset, Id);
  }
}


void CS_DrawValuesToOverlay(uint3 ID : SV_DispatchThreadID)
{
  switch(ID.x)
  {
    // max/avg/min CLL
    // maxCLL:
    case 0:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _5th(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
      }
      return;
    }
    case 1:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _4th(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
      }
      return;
    }
    case 2:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _3rd(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
      }
      return;
    }
    case 3:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _2nd(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
      }
      return;
    }
    case 4:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _1st(maxCllShow);
        DrawChar(uint2(curNumber, 0), float2(8, 0), float2(ID.x, ID.y));
      }
      return;
    }
    case 5:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = d1st(maxCllShow);
        DrawChar(uint2(curNumber, 0), float2(9, 0), float2(ID.x, ID.y));
      }
      return;
    }
    // avgCLL:
    case 6:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _5th(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 7:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _4th(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 8:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _3rd(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 9:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _2nd(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 10:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _1st(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(8 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 11:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = d1st(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    case 12:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = d2nd(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 6, 1), float2(ID.x, ID.y));
      }
      return;
    }
    // minCLL:
    case 13:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _5th(minCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 14:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _4th(minCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 15:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _3rd(minCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 16:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _2nd(minCllShow);
        DrawNumberAboveZero(curNumber, float2(8 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 17:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _1st(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(8 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 18:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d1st(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 19:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d2nd(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 20:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d3rd(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 21:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d4th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 22:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d5th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    case 23:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d6th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(9 - 13, 2), float2(ID.x, ID.y));
      }
      return;
    }
    // cursorCLL:
    case 24:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _5th(cursorCll);
        DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 25:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _4th(cursorCll);
        DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 26:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _3rd(cursorCll);
        DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 27:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _2nd(cursorCll);
        DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 28:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _1st(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 29:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d1st(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 30:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d2nd(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 31:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d3rd(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 32:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d4th(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 33:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d5th(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }
    case 34:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d6th(cursorCll);
        DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12 - 24, 3), float2(ID.x, ID.y));
      }
      return;
    }


    // show CSPs
    // BT.709:
    case 35:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _3rd(precentageBt709);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 35, 4), float2(ID.x, ID.y));
      }
      return;
    }
    case 36:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _2nd(precentageBt709);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 35, 4), float2(ID.x, ID.y));
      }
      return;
    }
    case 37:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _1st(precentageBt709);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 35, 4), float2(ID.x, ID.y));
      }
      return;
    }
    case 38:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d1st(precentageBt709);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 35, 4), float2(ID.x, ID.y));
      }
      return;
    }
    case 39:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d2nd(precentageBt709);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 35, 4), float2(ID.x, ID.y));
      }
      return;
    }
    // DCI-P3:
    case 40:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _3rd(precentageDciP3);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 40, 5), float2(ID.x, ID.y));
      }
      return;
    }
    case 41:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _2nd(precentageDciP3);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 40, 5), float2(ID.x, ID.y));
      }
      return;
    }
    case 42:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _1st(precentageDciP3);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 40, 5), float2(ID.x, ID.y));
      }
      return;
    }
    case 43:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d1st(precentageDciP3);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 40, 5), float2(ID.x, ID.y));
      }
      return;
    }
    case 44:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d2nd(precentageDciP3);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 40, 5), float2(ID.x, ID.y));
      }
      return;
    }
    // BT.2020:
    case 45:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _3rd(precentageBt2020);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 45, 6), float2(ID.x, ID.y));
      }
      return;
    }
    case 46:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _2nd(precentageBt2020);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 45, 6), float2(ID.x, ID.y));
      }
      return;
    }
    case 47:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _1st(precentageBt2020);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 45, 6), float2(ID.x, ID.y));
      }
      return;
    }
    case 48:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d1st(precentageBt2020);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 45, 6), float2(ID.x, ID.y));
      }
      return;
    }
    case 49:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d2nd(precentageBt2020);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 45, 6), float2(ID.x, ID.y));
      }
      return;
    }
#ifdef IS_FLOAT_HDR_CSP
    // AP0:
    case 50:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _3rd(precentageAp0);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 50, 7), float2(ID.x, ID.y));
      }
      return;
    }
    case 51:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _2nd(precentageAp0);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 50, 7), float2(ID.x, ID.y));
      }
      return;
    }
    case 52:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _1st(precentageAp0);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 50, 7), float2(ID.x, ID.y));
      }
      return;
    }
    case 53:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d1st(precentageAp0);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 50, 7), float2(ID.x, ID.y));
      }
      return;
    }
    case 54:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d2nd(precentageAp0);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 50, 7), float2(ID.x, ID.y));
      }
      return;
    }
    // invalid:
    case 55:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _3rd(precentageInvalid);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 55, 8), float2(ID.x, ID.y));
      }
      return;
    }
    case 56:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _2nd(precentageInvalid);
        DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 55, 8), float2(ID.x, ID.y));
      }
      return;
    }
    case 57:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _1st(precentageInvalid);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 55, 8), float2(ID.x, ID.y));
      }
      return;
    }
    case 58:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d1st(precentageInvalid);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 55, 8), float2(ID.x, ID.y));
      }
      return;
    }
    case 59:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d2nd(precentageInvalid);
        DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 55, 8), float2(ID.x, ID.y));
      }
      return;
    }
#endif
    //cursorCSP:
    case 60:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_B, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_D, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_B, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_A, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_i, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#endif
      }
      return;
    }
    case 61:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_T, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_C, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_T, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_P, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_n, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#endif
      }
      return;
    }
    case 62:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_dot, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_I, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_dot, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawChar(_0, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_v, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#endif
      }
      return;
    }
    case 63:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_7, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_minus, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_a, currentCursorCspOffset, float2(ID.x, ID.y));
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
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_0, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_P, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_l, currentCursorCspOffset, float2(ID.x, ID.y));
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
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawChar(_9, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawChar(_3, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_2, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_i, currentCursorCspOffset, float2(ID.x, ID.y));
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
        float2 currentCursorCspOffset = cursorCspOffset + float2(11 - 60, 9);

        if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT709)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_DCI_P3)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_BT2020)
#else
        else
#endif
        {
          DrawChar(_0, currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
#ifdef IS_FLOAT_HDR_CSP
        else if (uint(tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f) == IS_CSP_AP0)
        {
          DrawSpace(currentCursorCspOffset, float2(ID.x, ID.y));
          return;
        }
        else //invalid
        {
          DrawChar(_d, currentCursorCspOffset, float2(ID.x, ID.y));
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


#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  #define MAP_INTO_CSP Scrgb

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  #define MAP_INTO_CSP Hdr10

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  #define MAP_INTO_CSP Hlg

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define MAP_INTO_CSP Ps5

#else //ACTUAL_COLOUR_SPACE ==
// FIX THIS someday...
  #define MAP_INTO_CSP Scrgb

#endif //ACTUAL_COLOUR_SPACE ==


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareSetActiveArea(
  in  uint   Id                : SV_VertexID,
  out float4 VPos              : SV_Position,
  out float2 TexCoord          : TEXCOORD0,
  out float4 PercentagesToCrop : PercentagesToCrop)
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
  in  float4 VPos              : SV_Position,
  in  float2 TexCoord          : TEXCOORD,
  out float4 Output            : SV_Target0,
  in  float4 PercentagesToCrop : PercentagesToCrop)
{
  if (ACTIVE_AREA_ENABLE)
  {
    float2 pureCoord = TexCoord * float2(BUFFER_WIDTH, BUFFER_HEIGHT);

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
  else
  {
    discard;
  }
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrAnalysis(
  in                  uint   Id                                    : SV_VertexID,
  out                 float4 VPos                                  : SV_Position,
  out                 float2 TexCoord                              : TEXCOORD0,
  out                 bool   PingPongChecks[2]                     : PingPongChecks,
  out                 float4 HighlightNitRange                     : HighlightNitRange,
  out nointerpolation uint2  BrightnessHistogramTextureDisplaySize : BrightnessHistogramTextureDisplaySize,
  out nointerpolation uint2  CieDiagramTextureDisplaySize          : CieDiagramTextureDisplaySize,
  out                 float2 CurrentActiveOverlayArea              : CurrentActiveOverlayArea)
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


  pingpong0Above1   = false;
  breathingIsActive = false;


  if (HIGHLIGHT_NIT_RANGE)
  {
    float pingpong0 = NIT_PINGPONG0.x + 0.25f;

    pingpong0Above1 = pingpong0 >= 1.f;

    if (pingpong0Above1)
    {
      breathing = saturate(pingpong0 - 1.f);
      //breathing = 1.f;

      breathingIsActive = breathing > 0.f;

      float pingpong1 = NIT_PINGPONG1.y == 1 ? NIT_PINGPONG1.x
                                             : 6.f - NIT_PINGPONG1.x;

      if (pingpong1 >= 0.f
       && pingpong1 <= 1.f)
      {
        highlightNitRangeOut = float3(1.f, NIT_PINGPONG2.x, 0.f);
      }
      else if (pingpong1 > 1.f
            && pingpong1 <= 2.f)
      {
        highlightNitRangeOut = float3(NIT_PINGPONG2.x, 1.f, 0.f);
      }
      else if (pingpong1 > 2.f
            && pingpong1 <= 3.f)
      {
        highlightNitRangeOut = float3(0.f, 1.f, NIT_PINGPONG2.x);
      }
      else if (pingpong1 > 3.f
            && pingpong1 <= 4.f)
      {
        highlightNitRangeOut = float3(0.f, NIT_PINGPONG2.x, 1.f);
      }
      else if (pingpong1 > 4.f
            && pingpong1 <= 5.f)
      {
        highlightNitRangeOut = float3(NIT_PINGPONG2.x, 0.f, 1.f);
      }
      else /*if (pingpong1 > 5.f
              && pingpong1 <= 6.f)*/
      {
        highlightNitRangeOut = float3(1.f, 0.f, NIT_PINGPONG2.x);
      }

      highlightNitRangeOut *= breathing * HIGHLIGHT_NIT_RANGE_BRIGHTNESS;

      highlightNitRangeOut = Csp::Map::Bt709Into::MAP_INTO_CSP(highlightNitRangeOut);
    }
  }

  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    BrightnessHistogramTextureDisplaySize =
      uint2(round(float(TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH)  * BRIGHTNESS_HISTOGRAM_SIZE / 100.f),
            round(float(TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT) * BRIGHTNESS_HISTOGRAM_SIZE / 100.f));
  }

  if (SHOW_CIE)
  {
    CieDiagramTextureDisplaySize = 
      uint2(round(float(CIE_BG_WIDTH ) * CIE_DIAGRAM_SIZE / 100.f),
            round(float(CIE_BG_HEIGHT) * CIE_DIAGRAM_SIZE / 100.f));
  }

  if (SHOW_CLL_VALUES
   || SHOW_CLL_FROM_CURSOR
   || SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    uint activeLines = (SHOW_CLL_VALUES ? ShowCllValuesLineCount
                                        : 0)
                     + (SHOW_CLL_FROM_CURSOR ? ShowCllFromCursorLineCount
                                             : 0)
                     + (SHOW_CSPS ? ShowCspsLineCount
                                  : 0)
                     + (SHOW_CSP_FROM_CURSOR ? 1
                                             : 0);

    uint activeCharacters =
      max(max(max((SHOW_CLL_VALUES ? 25
                                   : 0),
                  (SHOW_CLL_FROM_CURSOR ? 28
                                        : 0)),
                  (SHOW_CSPS ? 16
                             : 0)),
                  (SHOW_CSP_FROM_CURSOR ? 18
                                        : 0));

    static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

    uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

    uint2 currentOverlayDimensions = charSize
                                   * uint2(activeCharacters, activeLines);

    currentOverlayDimensions.x += OUTER_SPACING_X2;
    currentOverlayDimensions.y += (max(SHOW_CLL_VALUES
                                     + SHOW_CLL_FROM_CURSOR
                                     + SHOW_CSPS
                                     + SHOW_CSP_FROM_CURSOR
                                     - 1, 0) * charSize.y * SPACING_MULTIPLIER + OUTER_SPACING_X2);

    float2 bufferDimInFloat = float2(BUFFER_WIDTH, BUFFER_HEIGHT);

    CurrentActiveOverlayArea = (currentOverlayDimensions - 1 + 0.5f)
                             / bufferDimInFloat;

    if (TEXT_POSITION == TEXT_POSITION_TOP_RIGHT)
    {
      CurrentActiveOverlayArea.x = 1.f - CurrentActiveOverlayArea.x;
    }
  }

}


void PS_HdrAnalysis(
  in                  float4 VPos                                  : SV_Position,
  in                  float2 TexCoord                              : TEXCOORD,
  out                 float4 Output                                : SV_Target0,
  in                  bool   PingPongChecks[2]                     : PingPongChecks,
  in                  float4 HighlightNitRange                     : HighlightNitRange,
  in  nointerpolation uint2  BrightnessHistogramTextureDisplaySize : BrightnessHistogramTextureDisplaySize,
  in  nointerpolation uint2  CieDiagramTextureDisplaySize          : CieDiagramTextureDisplaySize,
  in                  float2 CurrentActiveOverlayArea              : CurrentActiveOverlayArea)
{
  Output = tex2D(ReShade::BackBuffer, TexCoord).rgba;


  if (SHOW_CSP_MAP
   || SHOW_HEATMAP
   || HIGHLIGHT_NIT_RANGE)
  {
    float pixelCLL = tex2D(SamplerCllValues, TexCoord).r;

    if (SHOW_CSP_MAP)
    {
      Output = float4(Create_CSP_Map(tex2D(SamplerCsps, TexCoord).r * 255.f,
                                     pixelCLL), 1.f);
    }

    if (SHOW_HEATMAP)
    {
      Output = float4(HeatmapRgbValues(pixelCLL,
                                         HEATMAP_CUTOFF_POINT,
                                         HEATMAP_BRIGHTNESS,
                                         false), 1.f);
    }

    if (HIGHLIGHT_NIT_RANGE
     && pixelCLL >= HIGHLIGHT_NIT_RANGE_START_POINT
     && pixelCLL <= HIGHLIGHT_NIT_RANGE_END_POINT
     && pingpong0Above1
     && breathingIsActive)
    {
      //Output = float4(HighlightNitRangeOut, 1.f);
      Output = float4(lerp(Output.rgb, highlightNitRangeOut, breathing), 1.f);
    }

  }

  if (DRAW_ABOVE_NITS_AS_BLACK)
  {
    float pixelCLL = tex2D(SamplerCllValues, TexCoord).r;
    if (pixelCLL > ABOVE_NITS_AS_BLACK)
    {
      Output = (0.f, 0.f, 0.f, 0.f);
    }
  }
  if (DRAW_BELOW_NITS_AS_BLACK)
  {
    float pixelCLL = tex2D(SamplerCllValues, TexCoord).r;
    if (pixelCLL < BELOW_NITS_AS_BLACK)
    {
      Output = (0.f, 0.f, 0.f, 0.f);
    }
  }

  if (SHOW_CIE)
  {
    uint current_x_coord = TexCoord.x * BUFFER_WIDTH;  // expand to actual pixel values
    uint current_y_coord = TexCoord.y * BUFFER_HEIGHT; // ^

    // draw the diagram in the bottom left corner
    if (current_x_coord <  CieDiagramTextureDisplaySize.x
     && current_y_coord >= (BUFFER_HEIGHT - CieDiagramTextureDisplaySize.y))
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(
        (current_x_coord + 0.5f) / CieDiagramTextureDisplaySize.x,
        (current_y_coord - (BUFFER_HEIGHT - CieDiagramTextureDisplaySize.y) + 0.5f) / CieDiagramTextureDisplaySize.y);

#if (CIE_DIAGRAM == CIE_1931)
  #define CIE_SAMPLER SamplerCie1931Current
  #define CIE_TRIANGLE_SAMPLER_BT709  SamplerCie1931CspTriangleBt709
  #define CIE_TRIANGLE_SAMPLER_DCI_P3 SamplerCie1931CspTriangleDciP3
  #define CIE_TRIANGLE_SAMPLER_BT2020 SamplerCie1931CspTriangleBt2020
  #define CIE_TRIANGLE_SAMPLER_AP0    SamplerCie1931CspTriangleAp0
#else
  #define CIE_SAMPLER SamplerCie1976Current
  #define CIE_TRIANGLE_SAMPLER_BT709  SamplerCie1976CspTriangleBt709
  #define CIE_TRIANGLE_SAMPLER_DCI_P3 SamplerCie1976CspTriangleDciP3
  #define CIE_TRIANGLE_SAMPLER_BT2020 SamplerCie1976CspTriangleBt2020
  #define CIE_TRIANGLE_SAMPLER_AP0    SamplerCie1976CspTriangleAp0
#endif

      float3 currentPixelToDisplay =
        pow(tex2D(CIE_SAMPLER, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;

      float3 cspOutlineOverlay = float3(0.f, 0.f, 0.f);

      if (SHOW_CIE_CSP_BT709_OUTLINE)
      {
        cspOutlineOverlay +=
          pow(tex2D(CIE_TRIANGLE_SAMPLER_BT709, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;
      }
      if (SHOW_CIE_CSP_DCI_P3_OUTLINE)
      {
        cspOutlineOverlay +=
          pow(tex2D(CIE_TRIANGLE_SAMPLER_DCI_P3, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;
      }
      if (SHOW_CIE_CSP_BT2020_OUTLINE)
      {
        cspOutlineOverlay +=
          pow(tex2D(CIE_TRIANGLE_SAMPLER_BT2020, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;
      }
#ifdef IS_FLOAT_HDR_CSP
      if (SHOW_CIE_CSP_AP0_OUTLINE)
      {
        cspOutlineOverlay +=
          pow(tex2D(CIE_TRIANGLE_SAMPLER_AP0, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;
      }
#endif

      Output = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay + cspOutlineOverlay), 1.f);

#undef CIE_SAMPLER
#undef CIE_TRIANGLE_SAMPLER

    }
  }

  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {

    uint current_x_coord = TexCoord.x * BUFFER_WIDTH;  // expand to actual pixel values
    uint current_y_coord = TexCoord.y * BUFFER_HEIGHT; // ^

    // draw the histogram in the bottom right corner
    if (current_x_coord >= (BUFFER_WIDTH  - BrightnessHistogramTextureDisplaySize.x)
     && current_y_coord >= (BUFFER_HEIGHT - BrightnessHistogramTextureDisplaySize.y))
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(
        (BrightnessHistogramTextureDisplaySize.x - (BUFFER_WIDTH - current_x_coord)  + 0.5f) / BrightnessHistogramTextureDisplaySize.x,
        (current_y_coord - (BUFFER_HEIGHT - BrightnessHistogramTextureDisplaySize.y) + 0.5f) / BrightnessHistogramTextureDisplaySize.y);

      float3 currentPixelToDisplay =
        tex2D(SamplerBrightnessHistogramFinal, currentSamplerCoords).rgb;

      Output = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay * BRIGHTNESS_HISTOGRAM_BRIGHTNESS), 1.f);

    }
  }

  if (SHOW_CLL_VALUES
   || SHOW_CLL_FROM_CURSOR
   || SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    float4 overlay;
    if (TEXT_POSITION == TEXT_POSITION_TOP_LEFT)
    {
      if (TexCoord.x <= CurrentActiveOverlayArea.x
       && TexCoord.y <= CurrentActiveOverlayArea.y)
      {
        overlay = tex2D(SamplerTextOverlay, (TexCoord
                                          * float2(BUFFER_WIDTH, BUFFER_HEIGHT)
                                          / float2(TEXTURE_OVERLAY_WIDTH, TEXTURE_OVERLAY_HEIGHT))).rgba;
      }
    }
    else
    {
      if (TexCoord.x >= CurrentActiveOverlayArea.x
       && TexCoord.y <= CurrentActiveOverlayArea.y)
      {
        overlay = tex2D(SamplerTextOverlay, float2(TexCoord.x - CurrentActiveOverlayArea.x, TexCoord.y)
                                          * float2(BUFFER_WIDTH, BUFFER_HEIGHT)
                                          / float2(TEXTURE_OVERLAY_WIDTH, TEXTURE_OVERLAY_HEIGHT)).rgba;
      }
    }

    overlay = pow(overlay, 2.2f);

    overlay = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(overlay.rgb * TEXT_BRIGHTNESS),
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
                     pow(overlay.a, 1.f / 2.6f)
#else
                     overlay.a
#endif
                    );

    Output = float4(lerp(Output.rgb, overlay.rgb, overlay.a), 1.f);
  }

}

//technique lilium__HDR_analysis_CLL_OLD
//<
//  enabled = false;
//>
//{
//  pass PS_CalcCllPerPixel
//  {
//    VertexShader = VS_PostProcess;
//     PixelShader = PS_CalcCllPerPixel;
//    RenderTarget = TextureCllValues;
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
  pass TESTY
  {
    VertexShader = VS_PostProcess;
     PixelShader = Testy;
  }
}
#endif //_TESTY

void CS_MakeOverlayBgRedraw(uint3 ID : SV_DispatchThreadID)
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


//CLL
  pass PS_CalcCllPerPixel
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CalcCllPerPixel;
    RenderTarget = TextureCllValues;
  }

  pass CS_GetMaxAvgMinCLL0_NEW
  {
    ComputeShader = CS_GetMaxAvgMinCLL0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass CS_GetMaxAvgMinCLL1_NEW
  {
    ComputeShader = CS_GetMaxAvgMinCLL1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass CS_GetFinalMaxAvgMinCLL_NEW
  {
    ComputeShader = CS_GetFinalMaxAvgMinCLL_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass PS_ClearBrightnessHistogramTexture
  {
    VertexShader       = VS_PostProcess;
     PixelShader       = PS_ClearBrightnessHistogramTexture;
    RenderTarget       = TextureBrightnessHistogram;
    ClearRenderTargets = true;
  }

  pass CS_RenderBrightnessHistogram
  {
    ComputeShader = CS_RenderBrightnessHistogram <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass PS_RenderBrightnessHistogramToScale
  {
    VertexShader = VS_PrepareRenderBrightnessHistogramToScale;
     PixelShader = PS_RenderBrightnessHistogramToScale;
    RenderTarget = TextureBrightnessHistogramFinal;
  }


//CIE
#if (CIE_DIAGRAM == CIE_1931)
  pass PS_CopyCie1931Bg
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CopyCie1931Bg;
    RenderTarget = TextureCie1931Current;
  }
#endif //CIE_DIAGRAM == CIE_1931

#if (CIE_DIAGRAM == CIE_1976)
  pass PS_CopyCie1976Bg
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CopyCie1976Bg;
    RenderTarget = TextureCie1976Current;
  }
#endif //CIE_DIAGRAM == CIE_1976

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
    DispatchSizeX = 106;
    DispatchSizeY = 1;
  }

  pass CS_DrawValuesToOverlay
  {
    ComputeShader = CS_DrawValuesToOverlay <1, 1>;
    DispatchSizeX = 67;
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
