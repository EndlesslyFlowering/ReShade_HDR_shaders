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
  ui_items    = "12\0"
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
> = 15;

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

#ifndef GAMESCOPE
uniform int CLL_SPACER_0
<
  ui_category = "Content Light Level analysis";
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

#ifndef GAMESCOPE
uniform int BRIGHTNESS_HISTOGRAM_SPACER_0
<
  ui_category = "Brightness histogram";
  ui_label    = " ";
  ui_type     = "radio";
>;
#endif

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

#ifndef GAMESCOPE
uniform int BRIGHTNESS_HISTOGRAM_SPACER_1
<
  ui_category = "Brightness histogram";
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

static const float ShowCllValuesLineCount     = 3;
static const float ShowCllFromCursorLineCount = 1;

#if defined(IS_HDR10_LIKE_CSP)

  static const float ShowCspsLineCount = 3;

#else //IS_HDR10_LIKE_CSP

  static const float ShowCspsLineCount = 5;

#endif //IS_HDR10_LIKE_CSP


void CS_PrepareOverlay(uint3 ID : SV_DispatchThreadID)
{
  //convert UI inputs into floats for comparisons
  float showCllValues     = SHOW_CLL_VALUES;
  float showCllFromCrusor = SHOW_CLL_FROM_CURSOR;

  float showCsps          = SHOW_CSPS;
  float showCspFromCursor = SHOW_CSP_FROM_CURSOR;

  float fontSize          = TEXT_SIZE;

  //get last UI values from the consolidated texture
  float showCllLast       = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  float showCursorCllLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

  float showCspsLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  float showCursorCspLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);

  float fontSizeLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);

  if (showCllLast       != showCllValues
   || showCursorCllLast != showCllFromCrusor
   || showCspsLast      != showCsps
   || showCursorCspLast != showCspFromCursor
   || fontSizeLast      != fontSize)
  {
    //store all current UI values
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW,  1);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, showCllValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, showCllFromCrusor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, showCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, showCspFromCursor);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, fontSize);

    //calculate offset for the cursorCLL text in the overlay
    float cursorCllYOffset = (!SHOW_CLL_VALUES
                            ? -ShowCllValuesLineCount
                            : SPACING_MULTIPLIER);

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL,
               cursorCllYOffset);


    //calculate offset for the colour spaces text in the overlay
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


    //calculate offset for the cursorCSP text in the overlay
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
      max(max(max((SHOW_CLL_VALUES ? 21
                                   : 0),
                  (SHOW_CLL_FROM_CURSOR ? 24
                                        : 0)),
                  (SHOW_CSPS ? 16
                             : 0)),
                  (SHOW_CSP_FROM_CURSOR ? 18
                                        : 0));

    static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

    uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);

    activeTextArea.y += (max(SHOW_CLL_VALUES
                           + SHOW_CLL_FROM_CURSOR
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


void DrawChar(uint2 Char, float2 DrawOffset)
{
  static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

  uint2 charSize   = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);
  uint  fontSize   = 23 - TEXT_SIZE;
  uint2 atlasXY    = uint2(fontSize % 4, fontSize / 4) * uint2(FONT_ATLAS_OFFSET_X, FONT_ATLAS_OFFSET_Y);
  uint2 charOffset = Char * charSize + atlasXY;

  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      float4 pixel = tex2Dfetch(SamplerFontAtlasConsolidated, charOffset + currentOffset).rgba;
      tex2Dstore(StorageTextOverlay, DrawOffset * charSize + OUTER_SPACING + currentOffset, pixel);
    }
  }
}


void DrawSpace(float2 DrawOffset)
{
  static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

  uint2 charSize = uint2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

  float4 emptyPixel = tex2Dfetch(SamplerFontAtlasConsolidated, int2(0, 0)).rgba;

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

    static const float cursorCllYOffset = 3.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL);
    static const float cspsYOffset0     = 4.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS);
    static const float cspsYOffset1     = 1.f + cspsYOffset0;
    static const float cspsYOffset2     = 2.f + cspsYOffset0;
#ifdef IS_FLOAT_HDR_CSP
    static const float cspsYOffset3     = 3.f + cspsYOffset0;
    static const float cspsYOffset4     = 4.f + cspsYOffset0;
#endif
    static const float cursorCspYOffset = 9.f + tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP);

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
    if (SHOW_CLL_VALUES)
    {
      // maxNits:
      DrawChar(_m,     float2( 0, 0));
      DrawChar(_a,     float2( 1, 0));
      DrawChar(_x,     float2( 2, 0));
      DrawChar(_n,     float2( 3, 0));
      DrawChar(_i,     float2( 4, 0));
      DrawChar(_t,     float2( 5, 0));
      DrawChar(_s,     float2( 6, 0));
      DrawChar(_colon, float2( 7, 0));
      DrawChar(_dot,   float2(14, 0)); // five figure number
      // avgNits:
      DrawChar(_a,     float2( 0, 1));
      DrawChar(_v,     float2( 1, 1));
      DrawChar(_g,     float2( 2, 1));
      DrawChar(_n,     float2( 3, 1));
      DrawChar(_i,     float2( 4, 1));
      DrawChar(_t,     float2( 5, 1));
      DrawChar(_s,     float2( 6, 1));
      DrawChar(_colon, float2( 7, 1));
      DrawChar(_dot,   float2(14, 1)); // five figure number
      // minNits:
      DrawChar(_m,     float2( 0, 2));
      DrawChar(_i,     float2( 1, 2));
      DrawChar(_n,     float2( 2, 2));
      DrawChar(_n,     float2( 3, 2));
      DrawChar(_i,     float2( 4, 2));
      DrawChar(_t,     float2( 5, 2));
      DrawChar(_s,     float2( 6, 2));
      DrawChar(_colon, float2( 7, 2));
      DrawChar(_dot,   float2(14, 2)); // five figure number
    }

    // cursorNits:
    if (SHOW_CLL_FROM_CURSOR)
    {
      DrawChar(_c,     float2( 0, cursorCllYOffset));
      DrawChar(_u,     float2( 1, cursorCllYOffset));
      DrawChar(_r,     float2( 2, cursorCllYOffset));
      DrawChar(_s,     float2( 3, cursorCllYOffset));
      DrawChar(_o,     float2( 4, cursorCllYOffset));
      DrawChar(_r,     float2( 5, cursorCllYOffset));
      DrawChar(_n,     float2( 6, cursorCllYOffset));
      DrawChar(_i,     float2( 7, cursorCllYOffset));
      DrawChar(_t,     float2( 8, cursorCllYOffset));
      DrawChar(_s,     float2( 9, cursorCllYOffset));
      DrawChar(_colon, float2(10, cursorCllYOffset));
      DrawChar(_dot,   float2(17, cursorCllYOffset)); // five figure number
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
    DrawChar(uint2(CurNumber % 10, 0), Offset);
  }
  else
  {
    DrawSpace(Offset);
  }
}


#define cursorCllYOffset tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL)
#define cspsYOffset      tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cursorCspYOffset tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP)

void CS_DrawValuesToOverlay(uint3 ID : SV_DispatchThreadID)
{
  switch(ID.x)
  {
    // max/avg/min Nits
    // maxNits:
    case 0:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _5th(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(9, 0));
      }
      return;
    }
    case 1:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _4th(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(10, 0));
      }
      return;
    }
    case 2:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _3rd(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(11, 0));
      }
      return;
    }
    case 3:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _2nd(maxCllShow);
        DrawNumberAboveZero(curNumber, float2(12, 0));
      }
      return;
    }
    case 4:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = _1st(maxCllShow);
        DrawChar(uint2(curNumber, 0), float2(13, 0));
      }
      return;
    }
    case 5:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
        precise uint  curNumber  = d1st(maxCllShow);
        DrawChar(uint2(curNumber, 0), float2(15, 0));
      }
      return;
    }
    // avgNits:
    case 6:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _5th(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(9, 1));
      }
      return;
    }
    case 7:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _4th(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(10, 1));
      }
      return;
    }
    case 8:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _3rd(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(11, 1));
      }
      return;
    }
    case 9:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _2nd(avgCllShow);
        DrawNumberAboveZero(curNumber, float2(12, 1));
      }
      return;
    }
    case 10:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = _1st(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(13, 1));
      }
      return;
    }
    case 11:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = d1st(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(15, 1));
      }
      return;
    }
    case 12:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
        precise uint  curNumber  = d2nd(avgCllShow);
        DrawChar(uint2(curNumber, 0), float2(16, 1));
      }
      return;
    }
    // minNits:
    case 13:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _5th(minCllShow);
        DrawNumberAboveZero(curNumber, float2(9, 2));
      }
      return;
    }
    case 14:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _4th(minCllShow);
        DrawNumberAboveZero(curNumber, float2(10, 2));
      }
      return;
    }
    case 15:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _3rd(minCllShow);
        DrawNumberAboveZero(curNumber, float2(11, 2));
      }
      return;
    }
    case 16:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _2nd(minCllShow);
        DrawNumberAboveZero(curNumber, float2(12, 2));
      }
      return;
    }
    case 17:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = _1st(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(13, 2));
      }
      return;
    }
    case 18:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d1st(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(15, 2));
      }
      return;
    }
    case 19:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d2nd(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(16, 2));
      }
      return;
    }
    case 20:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d3rd(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(17, 2));
      }
      return;
    }
    case 21:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d4th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(18, 2));
      }
      return;
    }
    case 22:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d5th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(19, 2));
      }
      return;
    }
    case 23:
    {
      if (SHOW_CLL_VALUES)
      {
        precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
        precise uint  curNumber  = d6th(minCllShow);
        DrawChar(uint2(curNumber, 0), float2(20, 2));
      }
      return;
    }
    // cursorNits:
    case 24:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _5th(cursorCll);
        DrawNumberAboveZero(curNumber, float2(12, 3 + cursorCllYOffset));
      }
      return;
    }
    case 25:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _4th(cursorCll);
        DrawNumberAboveZero(curNumber, float2(13, 3 + cursorCllYOffset));
      }
      return;
    }
    case 26:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _3rd(cursorCll);
        DrawNumberAboveZero(curNumber, float2(14, 3 + cursorCllYOffset));
      }
      return;
    }
    case 27:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _2nd(cursorCll);
        DrawNumberAboveZero(curNumber, float2(15, 3 + cursorCllYOffset));
      }
      return;
    }
    case 28:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = _1st(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(16, 3 + cursorCllYOffset));
      }
      return;
    }
    case 29:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d1st(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(18, 3 + cursorCllYOffset));
      }
      return;
    }
    case 30:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d2nd(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(19, 3 + cursorCllYOffset));
      }
      return;
    }
    case 31:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d3rd(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(20, 3 + cursorCllYOffset));
      }
      return;
    }
    case 32:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d4th(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(21, 3 + cursorCllYOffset));
      }
      return;
    }
    case 33:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d5th(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(22, 3 + cursorCllYOffset));
      }
      return;
    }
    case 34:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
        precise uint  curNumber = d6th(cursorCll);
        DrawChar(uint2(curNumber, 0), float2(23, 3 + cursorCllYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 4 + cspsYOffset));
      }
      return;
    }
    case 36:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _2nd(precentageBt709);
        DrawNumberAboveZero(curNumber, float2(10, 4 + cspsYOffset));
      }
      return;
    }
    case 37:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = _1st(precentageBt709);
        DrawChar(uint2(curNumber, 0), float2(11, 4 + cspsYOffset));
      }
      return;
    }
    case 38:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d1st(precentageBt709);
        DrawChar(uint2(curNumber, 0), float2(13, 4 + cspsYOffset));
      }
      return;
    }
    case 39:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
        precise uint  curNumber       = d2nd(precentageBt709);
        DrawChar(uint2(curNumber, 0), float2(14, 4 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 5 + cspsYOffset));
      }
      return;
    }
    case 41:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _2nd(precentageDciP3);
        DrawNumberAboveZero(curNumber, float2(10, 5 + cspsYOffset));
      }
      return;
    }
    case 42:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = _1st(precentageDciP3);
        DrawChar(uint2(curNumber, 0), float2(11, 5 + cspsYOffset));
      }
      return;
    }
    case 43:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d1st(precentageDciP3);
        DrawChar(uint2(curNumber, 0), float2(13, 5 + cspsYOffset));
      }
      return;
    }
    case 44:
    {
      if (SHOW_CSPS)
      {
        precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
        precise uint  curNumber       = d2nd(precentageDciP3);
        DrawChar(uint2(curNumber, 0), float2(14, 5 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 6 + cspsYOffset));
      }
      return;
    }
    case 46:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _2nd(precentageBt2020);
        DrawNumberAboveZero(curNumber, float2(10, 6 + cspsYOffset));
      }
      return;
    }
    case 47:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = _1st(precentageBt2020);
        DrawChar(uint2(curNumber, 0), float2(11, 6 + cspsYOffset));
      }
      return;
    }
    case 48:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d1st(precentageBt2020);
        DrawChar(uint2(curNumber, 0), float2(13, 6 + cspsYOffset));
      }
      return;
    }
    case 49:
    {
      if (SHOW_CSPS)
      {
        precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
        precise uint  curNumber        = d2nd(precentageBt2020);
        DrawChar(uint2(curNumber, 0), float2(14, 6 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 7 + cspsYOffset));
      }
      return;
    }
    case 51:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _2nd(precentageAp0);
        DrawNumberAboveZero(curNumber, float2(10, 7 + cspsYOffset));
      }
      return;
    }
    case 52:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = _1st(precentageAp0);
        DrawChar(uint2(curNumber, 0), float2(11, 7 + cspsYOffset));
      }
      return;
    }
    case 53:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d1st(precentageAp0);
        DrawChar(uint2(curNumber, 0), float2(13, 7 + cspsYOffset));
      }
      return;
    }
    case 54:
    {
      if (SHOW_CSPS)
      {
        precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
        precise uint  curNumber     = d2nd(precentageAp0);
        DrawChar(uint2(curNumber, 0), float2(14, 7 + cspsYOffset));
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
        DrawNumberAboveZero(curNumber, float2(9, 8 + cspsYOffset));
      }
      return;
    }
    case 56:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _2nd(precentageInvalid);
        DrawNumberAboveZero(curNumber, float2(10, 8 + cspsYOffset));
      }
      return;
    }
    case 57:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = _1st(precentageInvalid);
        DrawChar(uint2(curNumber, 0), float2(11, 8 + cspsYOffset));
      }
      return;
    }
    case 58:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d1st(precentageInvalid);
        DrawChar(uint2(curNumber, 0), float2(13, 8 + cspsYOffset));
      }
      return;
    }
    case 59:
    {
      if (SHOW_CSPS)
      {
        precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
        precise uint  curNumber         = d2nd(precentageInvalid);
        DrawChar(uint2(curNumber, 0), float2(14, 8 + cspsYOffset));
      }
      return;
    }
#endif
    //cursorCSP:
    case 60:
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
    case 61:
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
    case 62:
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
    case 63:
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
    case 64:
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
    case 65:
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
    case 66:
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

#undef cursorCllYOffset
#undef cspsYOffset
#undef cursorCspYOffset


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
  else
  {
    discard;
  }
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrAnalysis(
  in                  uint   Id                       : SV_VertexID,
  out                 float4 VPos                     : SV_Position,
  out                 float2 TexCoord                 : TEXCOORD0,
  out nointerpolation bool   PingPongChecks[2]        : PingPongChecks,
  out nointerpolation float4 HighlightNitRange        : HighlightNitRange,
  out nointerpolation float4 TextureDisplaySizes      : TextureDisplaySizes,
  out nointerpolation float2 CurrentActiveOverlayArea : CurrentActiveOverlayArea)
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

      highlightNitRangeOut *= breathing * HIGHLIGHT_NIT_RANGE_BRIGHTNESS;

      highlightNitRangeOut = Csp::Map::Bt709Into::MAP_INTO_CSP(highlightNitRangeOut);
    }
  }

#define BrightnessHistogramTextureDisplaySize TextureDisplaySizes.xy

  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    float brightnessHistogramSizeFrac = BRIGHTNESS_HISTOGRAM_SIZE / 100.f;

    BrightnessHistogramTextureDisplaySize =
      float2(TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH ,TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT)
    * brightnessHistogramSizeFrac;
  }

#define CieDiagramTextureDisplaySize TextureDisplaySizes.zw

  if (SHOW_CIE)
  {
    float cieDiagramSizeFrac = CIE_DIAGRAM_SIZE / 100.f;

    CieDiagramTextureDisplaySize = float2(CIE_BG_WIDTH, CIE_BG_HEIGHT) * cieDiagramSizeFrac;
  }

  if (SHOW_CLL_VALUES
   || SHOW_CLL_FROM_CURSOR
   || SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    float activeLines = (SHOW_CLL_VALUES ? ShowCllValuesLineCount
                                         : 0.f)
                      + (SHOW_CLL_FROM_CURSOR ? ShowCllFromCursorLineCount
                                              : 0.f)
                      + (SHOW_CSPS ? ShowCspsLineCount
                                   : 0.f)
                      + (SHOW_CSP_FROM_CURSOR ? 1.f
                                              : 0.f);

    float activeCharacters = max(max(max((SHOW_CLL_VALUES ? 21.f
                                                          :  0.f),
                                         (SHOW_CLL_FROM_CURSOR ? 24.f
                                                               :  0.f)),
                                         (SHOW_CSPS ? 16.f
                                                    :  0.f)),
                                         (SHOW_CSP_FROM_CURSOR ? 18.f
                                                               :  0.f));

    static const uint charSizeArrayOffsetX = TEXT_SIZE * 2;

    float2 charSize = float2(CharSize[charSizeArrayOffsetX], CharSize[charSizeArrayOffsetX + 1]);

    float2 currentOverlayDimensions = charSize
                                    * float2(activeCharacters, activeLines);

    currentOverlayDimensions.y += (max(SHOW_CLL_VALUES
                                     + SHOW_CLL_FROM_CURSOR
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


void PS_HdrAnalysis(
  in                  float4 VPos                     : SV_Position,
  in                  float2 TexCoord                 : TEXCOORD,
  in  nointerpolation bool   PingPongChecks[2]        : PingPongChecks,
  in  nointerpolation float4 HighlightNitRange        : HighlightNitRange,
  in  nointerpolation float4 TextureDisplaySizes      : TextureDisplaySizes,
  in  nointerpolation float2 CurrentActiveOverlayArea : CurrentActiveOverlayArea,
  out                 float4 Output                   : SV_Target0)
{
  Output = float4(tex2D(ReShade::BackBuffer, TexCoord).rgb, 1.f);


  if (SHOW_CSP_MAP
   || SHOW_HEATMAP
   || HIGHLIGHT_NIT_RANGE
   || DRAW_ABOVE_NITS_AS_BLACK
   || DRAW_BELOW_NITS_AS_BLACK)
  {
    float pixelCll = tex2D(SamplerCllValues, TexCoord).r;

    if (SHOW_CSP_MAP)
    {
      Output.rgb = CreateCspMap(tex2D(SamplerCsps, TexCoord).r * 255.f, pixelCll);
    }

    if (SHOW_HEATMAP)
    {
      Output.rgb = HeatmapRgbValues(pixelCll,
                                    HEATMAP_CUTOFF_POINT,
                                    HEATMAP_BRIGHTNESS,
                                    false);
    }

    if (HIGHLIGHT_NIT_RANGE
     && pixelCll >= HIGHLIGHT_NIT_RANGE_START_POINT
     && pixelCll <= HIGHLIGHT_NIT_RANGE_END_POINT
     && pingpong0Above1
     && breathingIsActive)
    {
      //Output.rgb = HighlightNitRangeOut;
      Output.rgb = lerp(Output.rgb, highlightNitRangeOut, breathing);
    }

    if (DRAW_ABOVE_NITS_AS_BLACK)
    {
      if (pixelCll > ABOVE_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }

    if (DRAW_BELOW_NITS_AS_BLACK)
    {
      if (pixelCll < BELOW_NITS_AS_BLACK)
      {
        Output.rgba = 0.f;
      }
    }
  }

  if (SHOW_CIE)
  {
    float2 pureCoord = TexCoord * ReShade::ScreenSize; // expand to actual pixel values

    float textureDisplayAreaYBegin = BUFFER_HEIGHT - CieDiagramTextureDisplaySize.y;

    // draw the diagram in the bottom left corner
    if (pureCoord.x <  CieDiagramTextureDisplaySize.x
     && pureCoord.y >= textureDisplayAreaYBegin)
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(pureCoord.x,
                                           pureCoord.y - textureDisplayAreaYBegin);

      currentSamplerCoords += 0.5f;
      currentSamplerCoords /= CieDiagramTextureDisplaySize;

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

      Output.rgb = Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay + cspOutlineOverlay);

#undef CIE_SAMPLER
#undef CIE_TRIANGLE_SAMPLER

    }
  }

  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    float2 pureCoord = TexCoord * ReShade::ScreenSize; // expand to actual pixel values

    float2 textureDisplayAreaBegin = float2(BUFFER_WIDTH  - BrightnessHistogramTextureDisplaySize.x,
                                            BUFFER_HEIGHT - BrightnessHistogramTextureDisplaySize.y);

    // draw the histogram in the bottom right corner
    if (all(pureCoord >= textureDisplayAreaBegin))
    {
      // get coords for the sampler
      float2 currentSamplerCoords = float2(pureCoord - textureDisplayAreaBegin);

      currentSamplerCoords += 0.5f;
      currentSamplerCoords /= BrightnessHistogramTextureDisplaySize;

      float3 currentPixelToDisplay =
        tex2D(SamplerBrightnessHistogramFinal, currentSamplerCoords).rgb;

      Output.rgb = Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay * BRIGHTNESS_HISTOGRAM_BRIGHTNESS);

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
      if (all(TexCoord <= CurrentActiveOverlayArea))
      {
        overlay = tex2D(SamplerTextOverlay, (TexCoord
                                          * ReShade::ScreenSize
                                          / float2(TEXTURE_OVERLAY_WIDTH, TEXTURE_OVERLAY_HEIGHT))).rgba;
      }
    }
    else
    {
      if (TexCoord.x >= CurrentActiveOverlayArea.x
       && TexCoord.y <= CurrentActiveOverlayArea.y)
      {
        overlay = tex2D(SamplerTextOverlay, float2(TexCoord.x - CurrentActiveOverlayArea.x, TexCoord.y)
                                          * ReShade::ScreenSize
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

    Output.rgb = lerp(Output.rgb, overlay.rgb, overlay.a);
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
    DispatchSizeX = 1;
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
