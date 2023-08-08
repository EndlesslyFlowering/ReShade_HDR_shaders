#include "lilium__include/colour_space.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


// TODO:
// - redo font atlas texture
// - bring overlay alpha "in line" with single trigger technique for HDR10 output
// - improve vertex shader for clearing the brightness histogram with barebones Vertex shader?
// - add namespace for UI
// - fix 1x1 pixel offset in CIE textures
// - add primaries triangles to show on the CIE diagram for all CSPs



#undef TEXT_BRIGHTNESS

//#define _DEBUG
//#define _TESTY

#ifndef ENABLE_CLL_FEATURES
  #define ENABLE_CLL_FEATURES YES
#endif

#ifndef ENABLE_CIE_FEATURES
  #define ENABLE_CIE_FEATURES YES
#endif

#ifndef ENABLE_CSP_FEATURES
  #define ENABLE_CSP_FEATURES YES
#endif


#if (ENABLE_CLL_FEATURES == YES \
  || ENABLE_CSP_FEATURES == YES)
uniform int2 MOUSE_POSITION
<
  source = "mousepoint";
>;
#else //ENABLE_CLL_FEATURES == YES || ENABLE_CSP_FEATURES == YES
  static const int2 MOUSE_POSITION = float2(0.f, 0.f);
#endif //ENABLE_CLL_FEATURES == YES || ENABLE_CSP_FEATURES == YES

#if (ENABLE_CLL_FEATURES == YES)
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
#else //ENABLE_CLL_FEATURES == YES
  static const float2 NIT_PINGPONG0 = float2(0.f, 0.f);
  static const float2 NIT_PINGPONG1 = float2(0.f, 0.f);
  static const float2 NIT_PINGPONG2 = float2(0.f, 0.f);
#endif //ENABLE_CLL_FEATURES == YES


uniform uint TEXT_SIZE
<
  ui_category = "global";
  ui_label    = "text size";
  ui_type     = "combo";
  ui_items    = "32\0"
                "34\0"
                "36\0"
                "38\0"
                "40\0"
                "42\0"
                "44\0"
                "46\0"
                "48\0";
> = 0;

uniform float TEXT_BRIGHTNESS
<
  ui_category = "global";
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
  ui_category = "global";
  ui_label    = "text position";
  ui_type     = "combo";
  ui_items    = "top left\0"
                "top right\0";
> = 0;

// CLL
#if (ENABLE_CLL_FEATURES == YES)
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
#else //ENABLE_CLL_FEATURES == YES
  static const bool SHOW_CLL_VALUES      = false;
  static const bool SHOW_CLL_FROM_CURSOR = false;
#endif //ENABLE_CLL_FEATURES == YES

// TextureCsps
#if (ENABLE_CSP_FEATURES == YES)
uniform bool SHOW_CSPS
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour spaces used";
  ui_tooltip  = "in %";
> = true;

uniform bool SHOW_CSP_MAP
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour space map";
  ui_tooltip  = "        colours:"
           "\n" "black and white: BT.709"
           "\n" "           teal: DCI-P3"
           "\n" "         yellow: BT.2020"
           "\n" "           blue: AP1"
           "\n" "            red: AP0"
           "\n" "           pink: invalid";
> = false;

uniform bool SHOW_CSP_FROM_CURSOR
<
  ui_category = "Colour Space analysis";
  ui_label    = "show colour space from cursor position";
> = true;
#else //ENABLE_CSP_FEATURES == YES
  static const bool SHOW_CSPS            = false;
  static const bool SHOW_CSP_MAP         = false;
  static const bool SHOW_CSP_FROM_CURSOR = false;
#endif //ENABLE_CSP_FEATURES == YES

// CIE
#if (ENABLE_CIE_FEATURES == YES)
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
#else //ENABLE_CIE_FEATURES == YES
  static const bool  SHOW_CIE               = false;
  static const float CIE_DIAGRAM_BRIGHTNESS = 0.f;
  static const float CIE_DIAGRAM_SIZE       = 0.f;
#endif //ENABLE_CIE_FEATURES == YES

// heatmap
#if (ENABLE_CLL_FEATURES == YES)
uniform bool SHOW_HEATMAP
<
  ui_category = "Heatmap visualisation";
  ui_label    = "show heatmap";
  ui_tooltip  = "         colours:   10000 nits:   1000 nits:"
           "\n" " black and white:       0-  100       0- 100"
           "\n" "  teal to green:      100-  203     100- 203"
           "\n" " green to yellow:     203-  400     203- 400"
           "\n" "yellow to red:        400- 1000     400- 600"
           "\n" "   red to pink:      1000- 4000     600- 800"
           "\n" "  pink to blue:      4000-10000     800-1000";
> = false;

uniform uint HEATMAP_CUTOFF_POINT
<
  ui_category = "Heatmap visualisation";
  ui_label    = "heatmap cutoff point";
  ui_type     = "combo";
  ui_items    = "10000 nits\0"
                " 1000 nits\0";
> = 0;

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

// highlight a certain nit range
uniform bool HIGHLIGHT_NIT_RANGE
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "enable highlighting brightness levels in a certain range";
  ui_tooltip  = "in nits";
> = false;

uniform float HIGHLIGHT_NIT_RANGE_START_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range starting point";
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
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 0.f;

uniform float HIGHLIGHT_NIT_RANGE_BRIGHTNESS
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range brightness";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

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
#else //ENABLE_CLL_FEATURES == YES
  static const bool  SHOW_HEATMAP                          = false;
  static const uint  HEATMAP_CUTOFF_POINT                  = 0;
  static const float HEATMAP_BRIGHTNESS                    = 0.f;
  static const bool  SHOW_BRIGHTNESS_HISTOGRAM             = false;
  static const float BRIGHTNESS_HISTOGRAM_BRIGHTNESS       = 0.f;
  static const float BRIGHTNESS_HISTOGRAM_SIZE             = 0.f;
  static const bool  BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE = false;
  static const bool  BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE = false;
  static const bool  HIGHLIGHT_NIT_RANGE                   = false;
  static const float HIGHLIGHT_NIT_RANGE_START_POINT       = 0.f;
  static const float HIGHLIGHT_NIT_RANGE_END_POINT         = 0.f;
  static const float HIGHLIGHT_NIT_RANGE_BRIGHTNESS        = 0.f;
  static const bool  DRAW_ABOVE_NITS_AS_BLACK              = false;
  static const float ABOVE_NITS_AS_BLACK                   = 0.f;
  static const bool  DRAW_BELOW_NITS_AS_BLACK              = false;
  static const float BELOW_NITS_AS_BLACK                   = 0.f;
#endif //ENABLE_CLL_FEATURES == YES


#define HDR_ANALYSIS_ENABLE

#include "lilium__include/hdr_analysis.fxh"

#if (CIE_DIAGRAM == CIE_1931)
  static const uint CIE_BG_X = CIE_1931_BG_X;
  static const uint CIE_BG_Y = CIE_1931_BG_Y;
#else
  static const uint CIE_BG_X = CIE_1976_BG_X;
  static const uint CIE_BG_Y = CIE_1976_BG_Y;
#endif


#ifdef _TESTY
uniform bool ENABLE_TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "enable test thingy";
> = false;

uniform float TEST_THINGY_R
<
  ui_category = "TESTY";
  ui_label    = "R";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;

uniform float TEST_THINGY_G
<
  ui_category = "TESTY";
  ui_label    = "G";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;

uniform float TEST_THINGY_B
<
  ui_category = "TESTY";
  ui_label    = "B";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.00000001f;
> = 0.f;
#endif //_TESTY


//void draw_maxCLL(float4 position : POSITION, float2 txcoord : TEXCOORD) : COLOR
//void draw_maxCLL(float4 VPos : SV_Position, float2 TexCoord : TEXCOORD, out float4 fragment : SV_Target0)
//{
//  const uint int_maxCLL = int(round(maxCLL));
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
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 Output   : SV_Target0)
{
  if(ENABLE_TEST_THINGY == true)
  {
    const float xxx = BUFFER_WIDTH  / 2.f - 100.f;
    const float xxe = (BUFFER_WIDTH  - xxx);
    const float yyy = BUFFER_HEIGHT / 2.f - 100.f;
    const float yye = (BUFFER_HEIGHT - yyy);
    if (TexCoord.x > xxx / BUFFER_WIDTH
     && TexCoord.x < xxe / BUFFER_WIDTH
     && TexCoord.y > yyy / BUFFER_HEIGHT
     && TexCoord.y < yye / BUFFER_HEIGHT)
      Output = float4(TEST_THINGY_R, TEST_THINGY_G, TEST_THINGY_B, 1.f);
    else
      Output = float4(0.f, 0.f, 0.f, 0.f);
  }
  else
    Output = float4(tex2D(ReShade::BackBuffer, TexCoord).rgb, 1.f);
}
#endif //_TESTY


uint2 GetCharSize()
{
  switch(TEXT_SIZE)
  {
    case 8:
    {
      return FONT_ATLAS_SIZE_48_CHAR_DIM;
    }
    case 7:
    {
      return FONT_ATLAS_SIZE_46_CHAR_DIM;
    }
    case 6:
    {
      return FONT_ATLAS_SIZE_44_CHAR_DIM;
    }
    case 5:
    {
      return FONT_ATLAS_SIZE_42_CHAR_DIM;
    }
    case 4:
    {
      return FONT_ATLAS_SIZE_40_CHAR_DIM;
    }
    case 3:
    {
      return FONT_ATLAS_SIZE_38_CHAR_DIM;
    }
    case 2:
    {
      return FONT_ATLAS_SIZE_36_CHAR_DIM;
    }
    case 1:
    {
      return FONT_ATLAS_SIZE_34_CHAR_DIM;
    }
    default:
    {
      return FONT_ATLAS_SIZE_32_CHAR_DIM;
    }
  }
}

#define SPACING 0.3f

static const float ShowCllValuesLineCount     = 3;
static const float ShowCllFromCursorLineCount = 1;

#if defined(IS_HDR10_LIKE_CSP)

  static const float ShowCspsLineCount = 3;

#else //IS_HDR10_LIKE_CSP

  static const float ShowCspsLineCount = 6;

#endif //IS_HDR10_LIKE_CSP


void CS_PrepareOverlay(uint3 ID : SV_DispatchThreadID)
{
#if (ENABLE_CLL_FEATURES == YES)
  float drawCllLast       = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  float drawcursorCllLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

  float floatShowCllValues     = SHOW_CLL_VALUES;
  float floatShowCllFromCrusor = SHOW_CLL_FROM_CURSOR;
#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CSP_FEATURES == YES)
  float drawCspsLast      = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  float drawcursorCspLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);

  float floatShowCsps          = SHOW_CSPS;
  float floatShowCspFromCursor = SHOW_CSP_FROM_CURSOR;
#endif //ENABLE_CSP_FEATURES == YES

  uint fontSizeLast = tex2Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);


  if (
#if (ENABLE_CLL_FEATURES == YES)
      floatShowCllValues     != drawCllLast
   || floatShowCllFromCrusor != drawcursorCllLast
   ||
#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CSP_FEATURES == YES)
      floatShowCsps          != drawCspsLast
   || floatShowCspFromCursor != drawcursorCspLast
   ||
#endif //ENABLE_CSP_FEATURES == YES

      TEXT_SIZE              != fontSizeLast)
  {
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW,  1);
#if (ENABLE_CLL_FEATURES == YES)
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, floatShowCllValues);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, floatShowCllFromCrusor);
#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CSP_FEATURES == YES)
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, floatShowCsps);
    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, floatShowCspFromCursor);
#endif //ENABLE_CSP_FEATURES == YES

    tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, TEXT_SIZE);



#if (ENABLE_CLL_FEATURES == YES)
    float cursorCllYOffset = (!SHOW_CLL_VALUES
                            ? -ShowCllValuesLineCount
                            : SPACING);
    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL,
               cursorCllYOffset);
#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CSP_FEATURES == YES)
    float cspsYOffset = ((!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllValuesLineCount
                         - SPACING)

                       : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllFromCursorLineCount
                         - SPACING)

                       : (!SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR)
                       ? -(ShowCllValuesLineCount
                         + ShowCllFromCursorLineCount)

                       : SPACING * 2);

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS,
               cspsYOffset);

    float cursorCspYOffset = ((!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR  && SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              - SPACING * 2)

                            : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowCllFromCursorLineCount
                              - SPACING * 2)

                            : (SHOW_CLL_VALUES  && SHOW_CLL_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowCspsLineCount
                              - SPACING * 2)

                            : (!SHOW_CLL_VALUES && !SHOW_CLL_FROM_CURSOR && SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCllFromCursorLineCount
                              - SPACING)

                            : (!SHOW_CLL_VALUES && SHOW_CLL_FROM_CURSOR  && !SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCspsLineCount
                              - SPACING)

                            : (SHOW_CLL_VALUES  && !SHOW_CLL_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowCllFromCursorLineCount
                              + ShowCspsLineCount
                              - SPACING)

                            : (!SHOW_CLL_VALUES && !SHOW_CLL_FROM_CURSOR && !SHOW_CSPS)
                            ? -(ShowCllValuesLineCount
                              + ShowCllFromCursorLineCount
                              + ShowCspsLineCount)

#if defined(IS_HDR10_LIKE_CSP)
                            : SPACING * 3) - 3;
#else //IS_HDR10_LIKE_CSP
                            : SPACING * 3);
#endif //IS_HDR10_LIKE_CSP

    tex2Dstore(StorageConsolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP,
               cursorCspYOffset);

#endif //ENABLE_CSP_FEATURES == YES


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

    uint2 charSize = GetCharSize();
    uint2 activeTextArea = charSize
                         * uint2(activeCharacters, activeLines);
    activeTextArea.y += max(SHOW_CLL_VALUES
                          + SHOW_CLL_FROM_CURSOR
                          + SHOW_CSPS
                          + SHOW_CSP_FROM_CURSOR
                          - 1, 0) * charSize.y * SPACING;

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
  uint2 charSize   = GetCharSize();
  uint  fontSize   = 8 - TEXT_SIZE;
  uint2 atlasXY    = uint2(fontSize % 3, fontSize / 3) * FONT_ATLAS_OFFSET;
  uint2 charOffset = Char * charSize + atlasXY;
  for (uint y = 0; y < charSize.y; y++)
  {
    for (uint x = 0; x < charSize.x; x++)
    {
      uint2 currentOffset = uint2(x, y);
      float4 pixel = tex2Dfetch(SamplerFontAtlasConsolidated, charOffset + currentOffset).rgba;
      tex2Dstore(StorageTextOverlay, (Id + DrawOffset) * charSize + currentOffset, pixel);
    }
  }
}


#define cursorCllOffset float2(0, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL))
#define cspsOffset      float2(0, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS))
#define cursorCspOffset float2(0, tex2Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP))


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

    switch(ID.y)
    {
#if (ENABLE_CLL_FEATURES == YES)
      // max/avg/min CLL
      case 0:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            // maxCLL:
            case 0:
            {
              DrawChar(_m, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_a, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_x, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_C, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_L, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_L, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_colon, float2(0, 0), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_dot, float2(6, 0), float2(ID.x, ID.y)); // five figure number
              return;
            }
            case 8:
            {
              DrawChar(_n, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 9:
            {
              DrawChar(_i, float2(7 + 1, 0), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 1:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            // maxCLL:
            case 0:
            {
              DrawChar(_t, float2(10 + 7 + 1, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 1:
            {
              DrawChar(_s, float2(10 + 7 + 1, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            // avgCLL:
            case 2:
            {
              DrawChar(_a, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_v, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_g, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_C, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_L, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_L, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_colon, float2(-2, 0), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_dot, float2(6 - 2, 0), float2(ID.x, ID.y)); // five figure number
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 2:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            // avgCLL:
            case 0:
            {
              DrawChar(_n, float2(8 + 7 + 2, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 1:
            {
              DrawChar(_i, float2(8 + 7 + 2, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 2:
            {
              DrawChar(_t, float2(8 + 7 + 2, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 3:
            {
              DrawChar(_s, float2(8 + 7 + 2, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            // minCLL:
            case 4:
            {
              DrawChar(_m, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_i, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_n, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_C, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_L, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_L, float2(-4, 0), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 3:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            // minCLL:
            case 0:
            {
              DrawChar(_colon, float2(6, -1), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_dot, float2(6 + 6, -1), float2(ID.x, ID.y)); // five figure number
              return;
            }
            case 2:
            {
              DrawChar(_n, float2(6 + 6 + 7, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 3:
            {
              DrawChar(_i, float2(6 + 6 + 7, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 4:
            {
              DrawChar(_t, float2(6 + 6 + 7, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 5:
            {
              DrawChar(_s, float2(6 + 6 + 7, -1), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            default:
              return;
          }
        }
        return;
      }

      // cursorCLL:
      case 4:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_c, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_u, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_r, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_s, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_o, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_r, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_C, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_L, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_L, cursorCllOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_colon, cursorCllOffset, float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 5:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_dot, cursorCllOffset + float2(10 + 6, -2), float2(ID.x, ID.y)); // five figure number
              return;
            }
            case 1:
            {
              DrawChar(_n, cursorCllOffset + float2(10 + 6 + 7, -2), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 2:
            {
              DrawChar(_i, cursorCllOffset + float2(10 + 6 + 7, -2), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 3:
            {
              DrawChar(_t, cursorCllOffset + float2(10 + 6 + 7, -2), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 4:
            {
              DrawChar(_s, cursorCllOffset + float2(10 + 6 + 7, -2), float2(ID.x, ID.y)); // decimal places offset
              return;
            }
            case 5:
              break; // break here for the storage of the redraw bool
            default:
              return;
          }
        }
        else
        {
          switch(ID.x)
          {
            case 5:
              break; // break here for the storage of the redraw bool
            default:
              return;
          }
        }
      } break;
#else //ENABLE_CLL_FEATURES == YES
      case 5:
      {
        switch(ID.x)
        {
          case 5:
            break; // break here for the storage of the redraw bool
          default:
            return;
        }
      } break;
#endif //ENABLE_CLL_FEATURES == YES


#if (ENABLE_CSP_FEATURES == YES)
      // CSPs
      case 6:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            // BT.709:
            case 0:
            {
              DrawChar(_B, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_T, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_dot, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_7, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_0, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_9, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_colon, cspsOffset + float2(0, -2), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 2, -2), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_percent, cspsOffset + float2(5 + 2, -2), float2(ID.x, ID.y));
              return;
            }
            // DCI-P3:
            case 9:
            {
              DrawChar(_D, cspsOffset + float2(-9, -1), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 7:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            // DCI-P3:
            case 0:
            {
              DrawChar(_C, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_I, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_minus, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_P, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_3, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_colon, cspsOffset + float2(1, -2), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_dot, cspsOffset + float2(1 + 3 + 2, -2), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_percent, cspsOffset + float2(1 + 5 + 2, -2), float2(ID.x, ID.y));
              return;
            }
            // BT.2020:
            case 8:
            {
              DrawChar(_B, cspsOffset + float2(-8, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_T, cspsOffset + float2(-8, -1), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 8:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            // BT.2020:
            case 0:
            {
              DrawChar(_dot, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_2, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_0, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_2, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_0, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_colon, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_dot, cspsOffset + float2(2 + 3 + 1, -2), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_percent, cspsOffset + float2(2 + 5 + 1, -2), float2(ID.x, ID.y));
              return;
            }
#ifdef IS_FLOAT_HDR_CSP
            // AP1:
            case 8:
            {
              DrawChar(_A, cspsOffset + float2(-8, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_P, cspsOffset + float2(-8, -1), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      case 9:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            // AP1:
            case 0:
            {
              DrawChar(_1, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_colon, cspsOffset + float2(2, -2), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_dot, cspsOffset + float2(2 + 3 + 5, -2), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_percent, cspsOffset + float2(2 + 5 + 5, -2), float2(ID.x, ID.y));
              return;
            }
            // AP0:
            case 4:
            {
              DrawChar(_A, cspsOffset + float2(-4, -1), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_P, cspsOffset + float2(-4, -1), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_0, cspsOffset + float2(-4, -1), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_colon, cspsOffset + float2(-4, -1), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_dot, cspsOffset + float2(-4 + 3 + 5, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_percent, cspsOffset + float2(-4 + 5 + 5, -1), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // invalid:
      case 10:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_i, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_n, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_v, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_a, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_l, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_i, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_d, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_colon, cspsOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 1, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_percent, cspsOffset + float2(5 + 1, -1), float2(ID.x, ID.y));
              return;
            }
#endif //IS_FLOAT_HDR_CSP
            default:
              return;
          }
        }
        return;
      }

      // cursorCSP
      case 11:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_c, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_u, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_r, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_s, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_o, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_r, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_C, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 7:
            {
              DrawChar(_S, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 8:
            {
              DrawChar(_P, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            case 9:
            {
              DrawChar(_colon, cursorCspOffset + float2(0, -1), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
        }
        return;
      }
#endif //ENABLE_CSP_FEATURES == YES

      default:
      {
        return;
      }
    }

    if (ID.x == 5 && ID.y == 5 && ID.z == 0)
    {
      tex2Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW, 0);
      return;
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
    DrawChar(_space, Offset, Id);
  }
}


void CS_DrawValuesToOverlay(uint3 ID : SV_DispatchThreadID)
{

  switch(ID.y)
  {
#if (ENABLE_CLL_FEATURES == YES)
    // max/avg/min CLL
    case 0:
    {
      if (SHOW_CLL_VALUES)
      {
        switch(ID.x)
        {
          // maxCLL:
          case 0:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = _5th(maxCllShow);
            DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
            return;
          }
          case 1:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = _4th(maxCllShow);
            DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = _3rd(maxCllShow);
            DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = _2nd(maxCllShow);
            DrawNumberAboveZero(curNumber, float2(8, 0), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = _1st(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(8, 0), float2(ID.x, ID.y));
            return;
          }
          case 5:
          {
            precise float maxCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MAXCLL);
            precise uint  curNumber  = d1st(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0), float2(ID.x, ID.y));
            return;
          }
          // avgCLL:
          case 6:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = _5th(avgCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = _4th(avgCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = _3rd(avgCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = _2nd(avgCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          case 10:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = _1st(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(8 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          case 11:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = d1st(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 6, 1), float2(ID.x, ID.y));
            return;
          }
          default:
            return;
        }
      }
      return;
    }
    case 1:
    {
      if (SHOW_CLL_VALUES)
      {
        switch(ID.x)
        {
          // avgCLL:
          case 0:
          {
            precise float avgCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_AVGCLL);
            precise uint  curNumber  = d2nd(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 + 6, 0), float2(ID.x, ID.y));
            return;
          }
          // minCLL:
          case 1:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = _5th(minCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = _4th(minCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = _3rd(minCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = _2nd(minCllShow);
            DrawNumberAboveZero(curNumber, float2(8 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 5:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = _1st(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(8 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 6:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d1st(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d2nd(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d3rd(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d4th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 10:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d5th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          case 11:
          {
            precise float minCllShow = tex2Dfetch(StorageConsolidated, COORDS_SHOW_MINCLL);
            precise uint  curNumber  = d6th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9 - 1, 1), float2(ID.x, ID.y));
            return;
          }
          default:
            return;
        }
      }
      return;
    }

    // cursorCLL:
    case 2:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = _5th(cursorCll);
            DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11, 1), float2(ID.x, ID.y));
            return;
          }
          case 1:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = _4th(cursorCll);
            DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11, 1), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = _3rd(cursorCll);
            DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11, 1), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = _2nd(cursorCll);
            DrawNumberAboveZero(curNumber, cursorCllOffset + float2(11, 1), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = _1st(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 1), float2(ID.x, ID.y));
            return;
          }
          case 5:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d1st(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          case 6:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d2nd(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d3rd(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d4th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d5th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          case 10:
          {
            precise float cursorCll = tex2Dfetch(SamplerCllValues, MOUSE_POSITION).r;
            precise uint  curNumber = d6th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 1), float2(ID.x, ID.y));
            return;
          }
          default:
            return;
        }
      }
      return;
    }
#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CSP_FEATURES == YES)
    // show CSPs
    case 3:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          // BT.709:
          case 0:
          {
            precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
            precise uint  curNumber       = _3rd(precentageBt709);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 1), float2(ID.x, ID.y));
            return;
          }
          case 1:
          {
            precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
            precise uint  curNumber       = _2nd(precentageBt709);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 1), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
            precise uint  curNumber       = _1st(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 1), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
            precise uint  curNumber       = d1st(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 1), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float precentageBt709 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
            precise uint  curNumber       = d2nd(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 1), float2(ID.x, ID.y));
            return;
          }
          // DCI-P3:
          case 5:
          {
            precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
            precise uint  curNumber       = _3rd(precentageDciP3);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 2), float2(ID.x, ID.y));
            return;
          }
          case 6:
          {
            precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
            precise uint  curNumber       = _2nd(precentageDciP3);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 2), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
            precise uint  curNumber       = _1st(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 5, 2), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
            precise uint  curNumber       = d1st(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 2), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float precentageDciP3 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
            precise uint  curNumber       = d2nd(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 2), float2(ID.x, ID.y));
            return;
          }
          default:
            return;
        }
      }
      return;
    }
    case 4:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          // BT.2020:
          case 0:
          {
            precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
            precise uint  curNumber        = _3rd(precentageBt2020);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 2), float2(ID.x, ID.y));
            return;
          }
          case 1:
          {
            precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
            precise uint  curNumber        = _2nd(precentageBt2020);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 2), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
            precise uint  curNumber        = _1st(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 2), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
            precise uint  curNumber        = d1st(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 2), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float precentageBt2020 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
            precise uint  curNumber        = d2nd(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 2), float2(ID.x, ID.y));
            return;
          }
#ifdef IS_FLOAT_HDR_CSP
          // AP1:
          case 5:
          {
            precise float precentageAp1 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP1);
            precise uint  curNumber     = _3rd(precentageAp1);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 3), float2(ID.x, ID.y));
            return;
          }
          case 6:
          {
            precise float precentageAp1 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP1);
            precise uint  curNumber     = _2nd(precentageAp1);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 3), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float precentageAp1 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP1);
            precise uint  curNumber     = _1st(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 5, 3), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float precentageAp1 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP1);
            precise uint  curNumber     = d1st(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 3), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float precentageAp1 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP1);
            precise uint  curNumber     = d2nd(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 3), float2(ID.x, ID.y));
            return;
          }
          default:
            return;
        }
      }
      return;
    }

    case 5:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          // AP0:
          case 0:
          {
            precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
            precise uint  curNumber     = _3rd(precentageAp0);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 3), float2(ID.x, ID.y));
            return;
          }
          case 1:
          {
            precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
            precise uint  curNumber     = _2nd(precentageAp0);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9, 3), float2(ID.x, ID.y));
            return;
          }
          case 2:
          {
            precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
            precise uint  curNumber     = _1st(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 3), float2(ID.x, ID.y));
            return;
          }
          case 3:
          {
            precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
            precise uint  curNumber     = d1st(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 3), float2(ID.x, ID.y));
            return;
          }
          case 4:
          {
            precise float precentageAp0 = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
            precise uint  curNumber     = d2nd(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 3), float2(ID.x, ID.y));
            return;
          }
          // invalid:
          case 5:
          {
            precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
            precise uint  curNumber         = _3rd(precentageInvalid);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 4), float2(ID.x, ID.y));
            return;
          }
          case 6:
          {
            precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
            precise uint  curNumber         = _2nd(precentageInvalid);
            DrawNumberAboveZero(curNumber, cspsOffset + float2(9 - 5, 4), float2(ID.x, ID.y));
            return;
          }
          case 7:
          {
            precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
            precise uint  curNumber         = _1st(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9 - 5, 4), float2(ID.x, ID.y));
            return;
          }
          case 8:
          {
            precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
            precise uint  curNumber         = d1st(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 4), float2(ID.x, ID.y));
            return;
          }
          case 9:
          {
            precise float precentageInvalid = tex2Dfetch(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
            precise uint  curNumber         = d2nd(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10 - 5, 4), float2(ID.x, ID.y));
            return;
          }
#endif //IS_FLOAT_HDR_CSP
          default:
            return;
        }
      }
      return;
    }

    // cursorCSP:
    case 6:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        const uint cursorCSP = tex2Dfetch(SamplerCsps, MOUSE_POSITION).r * 255.f;

        if (cursorCSP == IS_CSP_BT709)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_T, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_dot, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_7, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_9, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }
        else if (cursorCSP == IS_CSP_DCI_P3)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_D, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_C, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_I, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_minus, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_3, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }
        else if (cursorCSP == IS_CSP_BT2020)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_T, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_dot, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_2, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_2, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }

#ifdef IS_FLOAT_HDR_CSP

        else if (cursorCSP == IS_CSP_AP1)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_A, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_1, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }
        else if (cursorCSP == IS_CSP_AP0)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_A, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_space, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }
        else
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_i, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 1:
            {
              DrawChar(_n, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 2:
            {
              DrawChar(_v, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 3:
            {
              DrawChar(_a, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 4:
            {
              DrawChar(_l, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 5:
            {
              DrawChar(_i, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            case 6:
            {
              DrawChar(_d, cursorCspOffset + float2(11, 4), float2(ID.x, ID.y));
              return;
            }
            default:
              return;
          }
          return;
        }
#endif //IS_FLOAT_HDR_CSP

#endif //ENABLE_CSP_FEATURES == YES

      }
      return;
    }

    default:
    {
      return;
    }
  }
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


#if (ENABLE_CLL_FEATURES == YES)

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

#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CIE_FEATURES == YES)

  if (SHOW_CIE)
  {
    CieDiagramTextureDisplaySize = 
      uint2(round(float(CIE_BG_X) * CIE_DIAGRAM_SIZE / 100.f),
            round(float(CIE_BG_Y) * CIE_DIAGRAM_SIZE / 100.f));
  }

#endif //ENABLE_CIE_FEATURES == YES

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

    uint2 charSize = GetCharSize();
    uint2 currentOverlayDimensions = charSize
                                   * uint2(activeCharacters, activeLines);
    currentOverlayDimensions.y += max(SHOW_CLL_VALUES
                                    + SHOW_CLL_FROM_CURSOR
                                    + SHOW_CSPS
                                    + SHOW_CSP_FROM_CURSOR
                                    - 1, 0) * charSize.y * SPACING;

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

#if (ENABLE_CSP_FEATURES == YES)

    if (SHOW_CSP_MAP)
    {
      Output = float4(Create_CSP_Map(tex2D(SamplerCsps, TexCoord).r * 255.f,
                                     pixelCLL), 1.f);
    }

#endif //ENABLE_CSP_FEATURES == YES

#if (ENABLE_CLL_FEATURES == YES)

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

#endif //ENABLE_CLL_FEATURES == YES
  }

#if (ENABLE_CLL_FEATURES == YES)

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

#endif //ENABLE_CLL_FEATURES == YES

#if (ENABLE_CIE_FEATURES == YES)

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
#else
  #define CIE_SAMPLER SamplerCie1931Current
#endif

      float3 currentPixelToDisplay =
        pow(tex2D(CIE_SAMPLER, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;

#undef CIE_SAMPLER

      Output = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay), 1.f);

    }
  }

#endif //ENABLE_CIE_FEATURES == YES

#if (ENABLE_CLL_FEATURES == YES)

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

#endif //ENABLE_CLL_FEATURES == YES


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

technique lilium__hdr_analysis
<
  ui_label = "Lilium's HDR analysis";
>
{

//CLL
#if (ENABLE_CLL_FEATURES == YES)

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
#endif //ENABLE_CLL_FEATURES == YES


//CIE
#if (ENABLE_CIE_FEATURES == YES)

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

#endif //ENABLE_CIE_FEATURES == YES


//CSP
#if (ENABLE_CSP_FEATURES == YES \
  && ACTUAL_COLOUR_SPACE != CSP_SRGB)

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

#endif //ENABLE_CSP_FEATURES == YES && ACTUAL_COLOUR_SPACE != CSP_SRGB

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
    DispatchSizeX = 10;
    DispatchSizeY = 12;
  }

  pass CS_DrawValuesToOverlay
  {
    ComputeShader = CS_DrawValuesToOverlay <1, 1>;
    DispatchSizeX = 12;
    DispatchSizeY = 7;
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
