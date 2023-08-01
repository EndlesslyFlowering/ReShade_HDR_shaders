#include "lilium__include\hdr_analysis.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


// TODO:
// - redo font atlas texture
// - rename all shaders to either VS/PS/CS
// - make "GLOBAL_INFO" about API support a macro in colour_space
// - bring overlay alpha "in line" with single trigger technique for HDR10 output



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

#ifndef DRAW_HIGH_ACCURACY
  #define DRAW_HIGH_ACCURACY NO
#endif

#if (DRAW_HIGH_ACCURACY == YES)
  #define CLL_DECIMAL_PLACES 7
#else
  #define CLL_DECIMAL_PLACES 2
#endif

#if (DRAW_HIGH_ACCURACY == YES)
  #define CSP_PERCENT_OFFSET 7
#else
  #define CSP_PERCENT_OFFSET 5
#endif

#if (ENABLE_CLL_FEATURES == YES \
  || ENABLE_CSP_FEATURES == YES)
uniform float2 MOUSE_POSITION
<
  source = "mousepoint";
>;
#else
  static const float2 MOUSE_POSITION = float2(0.f, 0.f);
#endif

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
#else
  static const float2 NIT_PINGPONG0 = float2(0.f, 0.f);
  static const float2 NIT_PINGPONG1 = float2(0.f, 0.f);
  static const float2 NIT_PINGPONG2 = float2(0.f, 0.f);
#endif


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
#else
  static const bool SHOW_CLL_VALUES      = false;
  static const bool SHOW_CLL_FROM_CURSOR = false;
#endif

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

#if (CIE_DIAGRAM == CIE_1931)
  static const uint CIE_BG_X = CIE_1931_BG_X;
  static const uint CIE_BG_Y = CIE_1931_BG_Y;
#else
  static const uint CIE_BG_X = CIE_1976_BG_X;
  static const uint CIE_BG_Y = CIE_1976_BG_Y;
#endif

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
  ui_label    = "CIE diagram size (in %)";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 100.f;
#else
  static const bool  SHOW_CIE               = false;
  static const float CIE_DIAGRAM_BRIGHTNESS = 0.f;
  static const float CIE_DIAGRAM_SIZE       = 0.f;
#endif

// Texture_CSPs
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
#else
  static const bool SHOW_CSPS            = false;
  static const bool SHOW_CSP_MAP         = false;
  static const bool SHOW_CSP_FROM_CURSOR = false;
#endif

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
  ui_label    = "heatmap brightness (in nits)";
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
  ui_label    = "brightness histogram brightness (in nits)";
  ui_type     = "slider";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 0.5f;
> = 80.f;

uniform float BRIGHTNESS_HISTOGRAM_SIZE
<
  ui_category = "Brightness histogram";
  ui_label    = "brightness histogram size (in %)";
  ui_type     = "slider";
  ui_units    = "%%";
  ui_min      = 50.f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 70.f;

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
  ui_label    = "range starting point (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 0.f;

uniform float HIGHLIGHT_NIT_RANGE_END_POINT
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range end point (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.0000001f;
> = 0.f;

uniform float HIGHLIGHT_NIT_RANGE_BRIGHTNESS
<
  ui_category = "Highlight brightness range visualisation";
  ui_label    = "range brightness (in nits)";
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
  ui_label    = "draw above this brightness as black (in nits)";
  ui_type     = "drag";
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
  ui_label    = "draw below this brightness as black (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 1.f;
> = 0.f;
#else
  static const bool  SHOW_HEATMAP                    = false;
  static const uint  HEATMAP_CUTOFF_POINT            = 0;
  static const float HEATMAP_BRIGHTNESS              = 0.f;
  static const bool  SHOW_BRIGHTNESS_HISTOGRAM       = false;
  static const float BRIGHTNESS_HISTOGRAM_BRIGHTNESS = 0.f;
  static const float BRIGHTNESS_HISTOGRAM_SIZE       = 0.f;
  static const bool  HIGHLIGHT_NIT_RANGE             = false;
  static const float HIGHLIGHT_NIT_RANGE_START_POINT = 0.f;
  static const float HIGHLIGHT_NIT_RANGE_END_POINT   = 0.f;
  static const float HIGHLIGHT_NIT_RANGE_BRIGHTNESS  = 0.f;
  static const bool  DRAW_ABOVE_NITS_AS_BLACK        = false;
  static const float ABOVE_NITS_AS_BLACK             = 0.f;
  static const bool  DRAW_BELOW_NITS_AS_BLACK        = false;
  static const float BELOW_NITS_AS_BLACK             = 0.f;
#endif

#ifdef _TESTY
uniform bool ENABLE_TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "enable test thingy";
> = false;

uniform float TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "test thingy";
  ui_type     = "drag";
  ui_min      = -125.f;
  ui_max      = 125.f;
  ui_step     = 0.000000001f;
> = 0.f;
#endif


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
    const float xTest = TEST_THINGY;
    const float xxx = BUFFER_WIDTH  / 2.f - 100.f;
    const float xxe = (BUFFER_WIDTH  - xxx);
    const float yyy = BUFFER_HEIGHT / 2.f - 100.f;
    const float yye = (BUFFER_HEIGHT - yyy);
    if (TexCoord.x > xxx / BUFFER_WIDTH
     && TexCoord.x < xxe / BUFFER_WIDTH
     && TexCoord.y > yyy / BUFFER_HEIGHT
     && TexCoord.y < yye / BUFFER_HEIGHT)
      Output = float4(xTest, xTest, xTest, 1.f);
    else
      Output = float4(0.f, 0.f, 0.f, 0.f);
  }
  else
    Output = float4(tex2D(ReShade::BackBuffer, TexCoord).rgb, 1.f);
}
#endif


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
static const float ShowCllFromCursorLineCount = 6;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  static const float ShowCspsLineCount = 3;

#else

  static const float ShowCspsLineCount = 6;

#endif


void PrepareOverlay(uint3 ID : SV_DispatchThreadID)
{
#if (ENABLE_CLL_FEATURES == YES)
  float drawCllLast        = tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW0).r;
  float drawcursorCllLast  = tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW1).r;

  float floatShowCllValues     = SHOW_CLL_VALUES;
  float floatShowCllFromCrusor = SHOW_CLL_FROM_CURSOR;
#endif

#if (ENABLE_CSP_FEATURES == YES)
  float drawCspsLast       = tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW2).r;
  float drawcursorCspLast  = tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW3).r;

  float floatShowCsps          = SHOW_CSPS;
  float floatShowCspFromCursor = SHOW_CSP_FROM_CURSOR;
#endif

  uint  fontSizeLast       = tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW4).r;


  if (
#if (ENABLE_CLL_FEATURES == YES)
      floatShowCllValues     != drawCllLast
   || floatShowCllFromCrusor != drawcursorCllLast
   ||
#endif

#if (ENABLE_CSP_FEATURES == YES)
      floatShowCsps          != drawCspsLast
   || floatShowCspFromCursor != drawcursorCspLast
   ||
#endif

      TEXT_SIZE              != fontSizeLast)
  {
    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW,  1);
#if (ENABLE_CLL_FEATURES == YES)
    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW0, floatShowCllValues);
    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW1, floatShowCllFromCrusor);
#endif

#if (ENABLE_CSP_FEATURES == YES)
    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW2, floatShowCsps);
    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW3, floatShowCspFromCursor);
#endif

    tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW4, TEXT_SIZE);



#if (ENABLE_CLL_FEATURES == YES)
    float cursorCllYOffset = (!SHOW_CLL_VALUES
                            ? -ShowCllValuesLineCount
                            : SPACING);
    tex2Dstore(Storage_Consolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL,
               cursorCllYOffset);
#endif

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

    tex2Dstore(Storage_Consolidated,
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

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)
                            : SPACING * 3) - 3;
#else
                            : SPACING * 3);
#endif

    tex2Dstore(Storage_Consolidated,
               COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP,
               cursorCspYOffset);

#endif


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
      max(max(max((SHOW_CLL_VALUES ? 26
                                   : 0),
                  (SHOW_CLL_FROM_CURSOR ? 29
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

#endif
}


#define FetchAndStoreSinglePixelOfChar(CurrentOffset, DrawOffset)                           \
  float4 pixel = tex2Dfetch(SamplerFontAtlasConsolidated, charOffset + CurrentOffset).rgba; \
  tex2Dstore(StorageTextOverlay, (float2(ID.x, ID.y) + DrawOffset) * charSize + CurrentOffset, pixel)

#define DrawChar(Char, DrawOffset)                                          \
  uint2 charSize   = GetCharSize();                                         \
  uint  fontSize   = 8 - TEXT_SIZE;                                         \
  uint2 atlasXY    = uint2(fontSize % 3, fontSize / 3) * FONT_ATLAS_OFFSET; \
  uint2 charOffset = Char * charSize + atlasXY;                             \
  for (int y = 0; y < charSize.y; y++)                                      \
  {                                                                         \
    for (int x = 0; x < charSize.x; x++)                                    \
    {                                                                       \
      FetchAndStoreSinglePixelOfChar(uint2(x, y), DrawOffset);              \
    }                                                                       \
  }                                                                         \


void DrawOverlay(uint3 ID : SV_DispatchThreadID)
{

  if (tex2Dfetch(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW).r)
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
            tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(1.f, 1.f, 1.f, 1.f));
          }
          else
          {
            tex2Dstore(StorageTextOverlay, int2(x + 220, y), float4(0.f, 0.f, 0.f, 1.f));
          }
        }
      }
    }

#endif


#define cursorCllOffset float2(0, tex2Dfetch(Storage_Consolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL).r)
#define cspsOffset      float2(0, tex2Dfetch(Storage_Consolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS).r)
#define cursorCspOffset float2(0, tex2Dfetch(Storage_Consolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP).r)


    switch(ID.y)
    {
#if (ENABLE_CLL_FEATURES == YES)
      // max/avg/min CLL
      // maxCLL:
      case 0:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_m, float2(0, 0))
              return;
            }
            case 1:
            {
              DrawChar(_a, float2(0, 0))
              return;
            }
            case 2:
            {
              DrawChar(_x, float2(0, 0))
              return;
            }
            case 3:
            {
              DrawChar(_C, float2(0, 0))
              return;
            }
            case 4:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 5:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 6:
            {
              DrawChar(_colon, float2(0, 0))
              return;
            }
            case 7:
            {
              DrawChar(_dot, float2(6, 0)) // five figure number
              return;
            }
            case 8:
            {
              DrawChar(_n, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 9:
            {
              DrawChar(_i, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 10:
            {
              DrawChar(_t, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 11:
            {
              DrawChar(_s, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 12:
              break; // break here for the storage of the redraw bool
            default:
              return;
          }
        }
        else
        {
          switch(ID.x)
          {
            case 12:
              break; // break here for the storage of the redraw bool
            default:
              return;
          }
        }
      } break;
      // avgCLL:
      case 1:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_a, float2(0, 0))
              return;
            }
            case 1:
            {
              DrawChar(_v, float2(0, 0))
              return;
            }
            case 2:
            {
              DrawChar(_g, float2(0, 0))
              return;
            }
            case 3:
            {
              DrawChar(_C, float2(0, 0))
              return;
            }
            case 4:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 5:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 6:
            {
              DrawChar(_colon, float2(0, 0))
              return;
            }
            case 7:
            {
              DrawChar(_dot, float2(6, 0)) // five figure number
              return;
            }
            case 8:
            {
              DrawChar(_n, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 9:
            {
              DrawChar(_i, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 10:
            {
              DrawChar(_t, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            case 11:
            {
              DrawChar(_s, float2(7 + CLL_DECIMAL_PLACES, 0)) // decimal places offset
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // minCLL:
      case 2:
      {
        if (SHOW_CLL_VALUES)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_m, float2(0, 0))
              return;
            }
            case 1:
            {
              DrawChar(_i, float2(0, 0))
              return;
            }
            case 2:
            {
              DrawChar(_n, float2(0, 0))
              return;
            }
            case 3:
            {
              DrawChar(_C, float2(0, 0))
              return;
            }
            case 4:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 5:
            {
              DrawChar(_L, float2(0, 0))
              return;
            }
            case 6:
            {
              DrawChar(_colon, float2(0, 0))
              return;
            }
            case 7:
            {
              DrawChar(_dot, float2(6, 0)) // five figure number
              return;
            }
            case 8:
            {
              DrawChar(_n, float2(14, 0)) // decimal places offset
              return;
            }
            case 9:
            {
              DrawChar(_i, float2(14, 0)) // decimal places offset
              return;
            }
            case 10:
            {
              DrawChar(_t, float2(14, 0)) // decimal places offset
              return;
            }
            case 11:
            {
              DrawChar(_s, float2(14, 0)) // decimal places offset
              return;
            }
            default:
              return;
          }
        }
        return;
      }

      // cursorCLL
      // x:
      case 3:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_x, cursorCllOffset + float2(8, 0))
              return;
            }
            case 1:
            {
              DrawChar(_colon, cursorCllOffset + float2(8, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // y:
      case 4:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_y, cursorCllOffset + float2(8, 0))
              return;
            }
            case 1:
            {
              DrawChar(_colon, cursorCllOffset + float2(8, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // cursorCLL:
      case 5:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_c, cursorCllOffset)
              return;
            }
            case 1:
            {
              DrawChar(_u, cursorCllOffset)
              return;
            }
            case 2:
            {
              DrawChar(_r, cursorCllOffset)
              return;
            }
            case 3:
            {
              DrawChar(_s, cursorCllOffset)
              return;
            }
            case 4:
            {
              DrawChar(_o, cursorCllOffset)
              return;
            }
            case 5:
            {
              DrawChar(_r, cursorCllOffset)
              return;
            }
            case 6:
            {
              DrawChar(_C, cursorCllOffset)
              return;
            }
            case 7:
            {
              DrawChar(_L, cursorCllOffset)
              return;
            }
            case 8:
            {
              DrawChar(_L, cursorCllOffset)
              return;
            }
            case 9:
            {
              DrawChar(_colon, cursorCllOffset)
              return;
            }
            case 10:
            {
              DrawChar(_dot, cursorCllOffset + float2(6, 0)) // five figure number
              return;
            }
            case 11:
            {
              DrawChar(_n, cursorCllOffset + float2(7 + 7, 0)) // 7 decimal places
              return;
            }
            case 12:
            {
              DrawChar(_i, cursorCllOffset + float2(7 + 7, 0)) // 7 decimal places
              return;
            }
            case 13:
            {
              DrawChar(_t, cursorCllOffset + float2(7 + 7, 0)) // 7 decimal places
              return;
            }
            case 14:
            {
              DrawChar(_s, cursorCllOffset + float2(7 + 7, 0)) // 7 decimal places
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // R:
      case 6:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_R, cursorCllOffset + float2(8, 0))
              return;
            }
            case 1:
            {
              DrawChar(_colon, cursorCllOffset + float2(8, 0))
              return;
            }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 1, 0))
              return;
            }

#else

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 3, 0))
              return;
            }

#endif

            default:
              return;
          }
        }
        return;
      }
      // G:
      case 7:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_G, cursorCllOffset + float2(8, 0))
              return;
            }
            case 1:
            {
              DrawChar(_colon, cursorCllOffset + float2(8, 0))
              return;
            }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 1, 0))
              return;
            }

#else

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 3, 0))
              return;
            }

#endif

            default:
              return;
          }
        }
        return;
      }
      // B:
      case 8:
      {
        if (SHOW_CLL_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cursorCllOffset + float2(8, 0))
              return;
            }
            case 1:
            {
              DrawChar(_colon, cursorCllOffset + float2(8, 0))
              return;
            }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 1, 0))
              return;
            }

#else

            case 2:
            {
              DrawChar(_dot, cursorCllOffset + float2(9 + 3, 0))
              return;
            }

#endif

            default:
              return;
          }
        }
        return;
      }
#endif

#if (ENABLE_CSP_FEATURES == YES)
      // CSPs
      // BT.709:
      case 9:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_T, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_dot, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_7, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_0, cspsOffset)
              return;
            }
            case 5:
            {
              DrawChar(_9, cspsOffset)
              return;
            }
            case 6:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 7:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 2, 0))
              return;
            }
            case 8:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 2, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // DCI-P3:
      case 10:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_D, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_C, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_I, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_minus, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_P, cspsOffset)
              return;
            }
            case 5:
            {
              DrawChar(_3, cspsOffset)
              return;
            }
            case 6:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 7:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 2, 0))
              return;
            }
            case 8:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 2, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // BT.2020:
      case 11:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_T, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_dot, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_2, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_0, cspsOffset)
              return;
            }
            case 5:
            {
              DrawChar(_2, cspsOffset)
              return;
            }
            case 6:
            {
              DrawChar(_0, cspsOffset)
              return;
            }
            case 7:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 8:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 1, 0))
              return;
            }
            case 9:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 1, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

      // AP1:
      case 12:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_A, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_P, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_1, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 5, 0))
              return;
            }
            case 5:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 5, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // AP0:
      case 13:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_A, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_P, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_0, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 5, 0))
              return;
            }
            case 5:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 5, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }
      // invalid:
      case 14:
      {
        if (SHOW_CSPS)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_i, cspsOffset)
              return;
            }
            case 1:
            {
              DrawChar(_n, cspsOffset)
              return;
            }
            case 2:
            {
              DrawChar(_v, cspsOffset)
              return;
            }
            case 3:
            {
              DrawChar(_a, cspsOffset)
              return;
            }
            case 4:
            {
              DrawChar(_l, cspsOffset)
              return;
            }
            case 5:
            {
              DrawChar(_i, cspsOffset)
              return;
            }
            case 6:
            {
              DrawChar(_d, cspsOffset)
              return;
            }
            case 7:
            {
              DrawChar(_colon, cspsOffset)
              return;
            }
            case 8:
            {
              DrawChar(_dot, cspsOffset + float2(3 + 1, 0))
              return;
            }
            case 9:
            {
              DrawChar(_percent, cspsOffset + float2(CSP_PERCENT_OFFSET + 1, 0))
              return;
            }
            default:
              return;
          }
        }
        return;
      }

#endif

      // cursorCSP
      case 15:
      {
        if (SHOW_CSP_FROM_CURSOR)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_c, cursorCspOffset)
              return;
            }
            case 1:
            {
              DrawChar(_u, cursorCspOffset)
              return;
            }
            case 2:
            {
              DrawChar(_r, cursorCspOffset)
              return;
            }
            case 3:
            {
              DrawChar(_s, cursorCspOffset)
              return;
            }
            case 4:
            {
              DrawChar(_o, cursorCspOffset)
              return;
            }
            case 5:
            {
              DrawChar(_r, cursorCspOffset)
              return;
            }
            case 6:
            {
              DrawChar(_C, cursorCspOffset)
              return;
            }
            case 7:
            {
              DrawChar(_S, cursorCspOffset)
              return;
            }
            case 8:
            {
              DrawChar(_P, cursorCspOffset)
              return;
            }
            case 9:
            {
              DrawChar(_colon, cursorCspOffset)
              return;
            }
            default:
              return;
          }
        }
        return;
      }
#endif

      default:
      {
        return;
      }
    }

    if (ID.x == 12 && ID.y == 0 && ID.z == 0)
    {
      tex2Dstore(Storage_Consolidated, COORDS_CHECK_OVERLAY_REDRAW, 0);
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

#define DrawNumberAboveZero(Offset)            \
  if (curNumber > 0)                           \
  {                                            \
    DrawChar(uint2(curNumber % 10, 0), Offset) \
  }                                            \
  else                                         \
  {                                            \
    DrawChar(_space, Offset)                   \
  }


void DrawNumbersToOverlay(uint3 ID : SV_DispatchThreadID)
{

  switch(ID.y)
  {
#if (ENABLE_CLL_FEATURES == YES)
    // max/avg/min CLL
    // maxCLL:
    case 0:
    {
      if (SHOW_CLL_VALUES)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = _5th(maxCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 1:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = _4th(maxCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 2:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = _3rd(maxCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 3:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = _2nd(maxCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 4:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = _1st(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(8, 0))
            return;
          }
          case 5:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d1st(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 6:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d2nd(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 7:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d3rd(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 8:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d4th(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 9:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d5th(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 10:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d6th(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 11:
          {
            precise float maxCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MAXCLL).r;
            precise uint  curNumber  = d7th(maxCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // avgCLL:
    case 1:
    {
      if (SHOW_CLL_VALUES)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = _5th(avgCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 1:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = _4th(avgCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 2:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = _3rd(avgCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 3:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = _2nd(avgCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 4:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = _1st(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(8, 0))
            return;
          }
          case 5:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d1st(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 6:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d2nd(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 7:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d3rd(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 8:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d4th(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 9:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d5th(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 10:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d6th(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 11:
          {
            precise float avgCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_AVGCLL).r;
            precise uint  curNumber  = d7th(avgCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // minCLL:
    case 2:
    {
      if (SHOW_CLL_VALUES)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = _5th(minCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 1:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = _4th(minCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 2:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = _3rd(minCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 3:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = _2nd(minCllShow);
            DrawNumberAboveZero(float2(8, 0))
            return;
          }
          case 4:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = _1st(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(8, 0))
            return;
          }
          case 5:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d1st(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 6:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d2nd(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 7:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d3rd(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 8:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d4th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 9:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d5th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 10:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d6th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
            return;
          }
          case 11:
          {
            precise float minCllShow = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_MINCLL).r;
            precise uint  curNumber  = d7th(minCllShow);
            DrawChar(uint2(curNumber, 0), float2(9, 0))
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

    // cursorCLL
    // x:
    case 3:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        switch(ID.x)
        {

#define mPosX clamp(MOUSE_POSITION.x, 0.f, BUFFER_WIDTH - 1)

          case 0:
          {
            precise uint curNumber = _4th(mPosX);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise uint curNumber = _3rd(mPosX);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise uint curNumber = _2nd(mPosX);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise uint curNumber = _1st(mPosX);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
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
    // y:
    case 4:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {

#define mPosY clamp(MOUSE_POSITION.y, 0.f, BUFFER_HEIGHT - 1)

        switch(ID.x)
        {
          case 0:
          {
            precise uint curNumber = _4th(mPosY);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise uint curNumber = _3rd(mPosY);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise uint curNumber = _2nd(mPosY);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise uint curNumber = _1st(mPosY);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
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
    // cursorCLL:
    case 5:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {

#define mPos int2(clamp(MOUSE_POSITION.x, 0.f, BUFFER_WIDTH  - 1), \
                  clamp(MOUSE_POSITION.y, 0.f, BUFFER_HEIGHT - 1))

        switch(ID.x)
        {
          case 0:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = _5th(cursorCll);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = _4th(cursorCll);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = _3rd(cursorCll);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = _2nd(cursorCll);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 4:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = _1st(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 5:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d1st(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 6:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d2nd(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 7:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d3rd(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 8:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d4th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 9:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d5th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 10:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d6th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 11:
          {
            precise float cursorCll = tex2Dfetch(Sampler_CLL_Values, mPos).r;
            precise uint  curNumber = d7th(cursorCll);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
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
    // R:
    case 6:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        switch(ID.x)
        {

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

          case 0:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = _1st(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d1st(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 2:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d2nd(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 3:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d3rd(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d4th(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#else

          case 0:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = _3rd(cursorR);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = _2nd(cursorR);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = _1st(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d1st(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d2nd(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 5:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d3rd(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 6:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d4th(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 7:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d5th(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 8:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d6th(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 9:
          {
            precise float cursorR   = tex2Dfetch(ReShade::BackBuffer, mPos).r;
            precise uint  curNumber = d7th(cursorR);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#endif

          default:
          {
            return;
          }
        }
      }
      return;
    }
    // G:
    case 7:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        switch(ID.x)
        {

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

          case 0:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = _1st(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d1st(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 2:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d2nd(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 3:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d3rd(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d4th(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#else

          case 0:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = _3rd(cursorG);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = _2nd(cursorG);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = _1st(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d1st(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d2nd(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 5:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d3rd(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 6:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d4th(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 7:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d5th(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 8:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d6th(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 9:
          {
            precise float cursorG   = tex2Dfetch(ReShade::BackBuffer, mPos).g;
            precise uint  curNumber = d7th(cursorG);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#endif

          default:
          {
            return;
          }
        }
      }
      return;
    }
    // B:
    case 8:
    {
      if (SHOW_CLL_FROM_CURSOR)
      {
        switch(ID.x)
        {

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

          case 0:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = _1st(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d1st(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 2:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d2nd(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 3:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d3rd(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d4th(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#else

          case 0:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = _3rd(cursorB);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 1:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = _2nd(cursorB);
            DrawNumberAboveZero(cursorCllOffset + float2(11, 0))
            return;
          }
          case 2:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = _1st(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(11, 0))
            return;
          }
          case 3:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d1st(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 4:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d2nd(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 5:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d3rd(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 6:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d4th(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 7:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d5th(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 8:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d6th(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }
          case 9:
          {
            precise float cursorB   = tex2Dfetch(ReShade::BackBuffer, mPos).b;
            precise uint  curNumber = d7th(cursorB);
            DrawChar(uint2(curNumber, 0), cursorCllOffset + float2(12, 0))
            return;
          }

#endif

          default:
          {
            return;
          }
        }
      }
      return;
    }
#endif

#if (ENABLE_CSP_FEATURES == YES)
    // show CSPs
    // BT.709:
    case 9:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = _3rd(precentageBt709);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = _2nd(precentageBt709);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = _1st(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = d1st(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = d2nd(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = d3rd(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageBt709 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709).r;
            precise uint  curNumber       = d4th(precentageBt709);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // DCI-P3:
    case 10:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = _3rd(precentageDciP3);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = _2nd(precentageDciP3);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = _1st(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = d1st(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = d2nd(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = d3rd(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageDciP3 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3).r;
            precise uint  curNumber       = d4th(precentageDciP3);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // BT.2020:
    case 11:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = _3rd(precentageBt2020);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = _2nd(precentageBt2020);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = _1st(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = d1st(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = d2nd(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = d3rd(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageBt2020 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020).r;
            precise uint  curNumber        = d4th(precentageBt2020);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

    // AP1:
    case 12:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = _3rd(precentageAp1);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = _2nd(precentageAp1);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = _1st(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = d1st(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = d2nd(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = d3rd(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageAp1 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1).r;
            precise uint  curNumber     = d4th(precentageAp1);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // AP0:
    case 13:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = _3rd(precentageAp0);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = _2nd(precentageAp0);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = _1st(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = d1st(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = d2nd(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = d3rd(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageAp0 = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0).r;
            precise uint  curNumber     = d4th(precentageAp0);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }
    // invalid:
    case 14:
    {
      if (SHOW_CSPS)
      {
        switch(ID.x)
        {
          case 0:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = _3rd(precentageInvalid);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 1:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = _2nd(precentageInvalid);
            DrawNumberAboveZero(cspsOffset + float2(9, 0))
            return;
          }
          case 2:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = _1st(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(9, 0))
            return;
          }
          case 3:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = d1st(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 4:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = d2nd(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#if (DRAW_HIGH_ACCURACY == YES)
          case 5:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = d3rd(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
          case 6:
          {
            precise float precentageInvalid = tex2Dfetch(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID).r;
            precise uint  curNumber         = d4th(precentageInvalid);
            DrawChar(uint2(curNumber, 0), cspsOffset + float2(10, 0))
            return;
          }
#endif
          default:
          {
            return;
          }
        }
      }
      return;
    }

#endif

    // cursorCSP:
    case 15:
    {
      if (SHOW_CSP_FROM_CURSOR)
      {
        const uint cursorCSP = tex2Dfetch(Sampler_CSPs, mPos).r * 255.f;

        if (cursorCSP == IS_CSP_BT709)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_B, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_T, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_dot, cursorCspOffset + float2(11, 0))
              return;
            }
            case 3:
            {
              DrawChar(_7, cursorCspOffset + float2(11, 0))
              return;
            }
            case 4:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 0))
              return;
            }
            case 5:
            {
              DrawChar(_9, cursorCspOffset + float2(11, 0))
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
              DrawChar(_D, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_C, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_I, cursorCspOffset + float2(11, 0))
              return;
            }
            case 3:
            {
              DrawChar(_minus, cursorCspOffset + float2(11, 0))
              return;
            }
            case 4:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 0))
              return;
            }
            case 5:
            {
              DrawChar(_3, cursorCspOffset + float2(11, 0))
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
              DrawChar(_B, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_T, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_dot, cursorCspOffset + float2(11, 0))
              return;
            }
            case 3:
            {
              DrawChar(_2, cursorCspOffset + float2(11, 0))
              return;
            }
            case 4:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 0))
              return;
            }
            case 5:
            {
              DrawChar(_2, cursorCspOffset + float2(11, 0))
              return;
            }
            case 6:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 0))
              return;
            }
            default:
              return;
          }
          return;
        }

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

        else if (cursorCSP == IS_CSP_AP1)
        {
          switch(ID.x)
          {
            case 0:
            {
              DrawChar(_A, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_1, cursorCspOffset + float2(11, 0))
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
              DrawChar(_A, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_P, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_0, cursorCspOffset + float2(11, 0))
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
              DrawChar(_i, cursorCspOffset + float2(11, 0))
              return;
            }
            case 1:
            {
              DrawChar(_n, cursorCspOffset + float2(11, 0))
              return;
            }
            case 2:
            {
              DrawChar(_v, cursorCspOffset + float2(11, 0))
              return;
            }
            case 3:
            {
              DrawChar(_a, cursorCspOffset + float2(11, 0))
              return;
            }
            case 4:
            {
              DrawChar(_l, cursorCspOffset + float2(11, 0))
              return;
            }
            case 5:
            {
              DrawChar(_i, cursorCspOffset + float2(11, 0))
              return;
            }
            case 6:
            {
              DrawChar(_d, cursorCspOffset + float2(11, 0))
              return;
            }
            default:
              return;
          }
          return;
        }
#endif

#endif

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

#else
// FIX THIS someday...
  #define MAP_INTO_CSP Scrgb

#endif


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

#endif

#if (ENABLE_CIE_FEATURES == YES)

  if (SHOW_CIE)
  {
    CieDiagramTextureDisplaySize = 
      uint2(round(float(CIE_BG_X) * CIE_DIAGRAM_SIZE / 100.f),
            round(float(CIE_BG_Y) * CIE_DIAGRAM_SIZE / 100.f));
  }

#endif

  uint activeLines = (SHOW_CLL_VALUES ? ShowCllValuesLineCount
                                      : 0)
                   + (SHOW_CLL_FROM_CURSOR ? ShowCllFromCursorLineCount
                                           : 0)
                   + (SHOW_CSPS ? ShowCspsLineCount
                                : 0)
                   + (SHOW_CSP_FROM_CURSOR ? 1
                                           : 0);

  uint activeCharacters =
    max(max(max((SHOW_CLL_VALUES ? 26
                                 : 0),
                (SHOW_CLL_FROM_CURSOR ? 29
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
  Output = float4(tex2D(ReShade::BackBuffer, TexCoord).rgb, 1.f);


  if (SHOW_CSP_MAP
   || SHOW_HEATMAP
   || HIGHLIGHT_NIT_RANGE)
  {
    float pixelCLL = tex2D(Sampler_CLL_Values, TexCoord).r;

#if (ENABLE_CSP_FEATURES == YES)

    if (SHOW_CSP_MAP)
    {
      Output = float4(Create_CSP_Map(tex2D(Sampler_CSPs, TexCoord).r * 255.f,
                                     pixelCLL), 1.f);
    }

#endif

#if (ENABLE_CLL_FEATURES == YES)

    if (SHOW_HEATMAP)
    {
      Output = float4(Heatmap_RGB_Values(pixelCLL,
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

#endif
  }

#if (ENABLE_CLL_FEATURES == YES)

  if (DRAW_ABOVE_NITS_AS_BLACK)
  {
    float pixelCLL = tex2D(Sampler_CLL_Values, TexCoord).r;
    if (pixelCLL > ABOVE_NITS_AS_BLACK)
    {
      Output = (0.f, 0.f, 0.f, 0.f);
    }
  }
  if (DRAW_BELOW_NITS_AS_BLACK)
  {
    float pixelCLL = tex2D(Sampler_CLL_Values, TexCoord).r;
    if (pixelCLL < BELOW_NITS_AS_BLACK)
    {
      Output = (0.f, 0.f, 0.f, 0.f);
    }
  }

#endif

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
  #define CIE_SAMPLER Sampler_CIE_1931_Current
#else
  #define CIE_SAMPLER Sampler_CIE_1976_Current
#endif

      float3 currentPixelToDisplay =
        pow(tex2D(CIE_SAMPLER, currentSamplerCoords).rgb, 2.2f) * CIE_DIAGRAM_BRIGHTNESS;

#undef CIE_SAMPLER

      Output = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay), 1.f);

    }
  }

#endif

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
        tex2D(Sampler_Brightness_Histogram_Final, currentSamplerCoords).rgb;

      Output = float4(Csp::Map::Bt709Into::MAP_INTO_CSP(currentPixelToDisplay * BRIGHTNESS_HISTOGRAM_BRIGHTNESS), 1.f);

    }
  }

#endif


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

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  overlay = float4(Csp::Map::Bt709Into::Scrgb(overlay.rgb * TEXT_BRIGHTNESS), pow(overlay.a, 1.f / 2.6f));

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  overlay = float4(Csp::Map::Bt709Into::Hdr10(overlay.rgb * TEXT_BRIGHTNESS), overlay.a);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  overlay = float4(Csp::Map::Bt709Into::Hlg(overlay.rgb * TEXT_BRIGHTNESS), overlay.a);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  overlay = float4(Csp::Map::Bt709Into::Ps5(overlay.rgb * TEXT_BRIGHTNESS), pow(overlay.a, 1.f / 2.6f));

#endif

  Output = float4(lerp(Output.rgb, overlay.rgb, overlay.a), 1.f);

}

//technique lilium__HDR_analysis_CLL_OLD
//<
//  enabled = false;
//>
//{
//  pass CalcCLLvalues
//  {
//    VertexShader = PostProcessVS;
//     PixelShader = CalcCLL;
//    RenderTarget = Texture_CLL_Values;
//  }
//
//  pass GetMaxAvgMinCLLvalue0
//  {
//    ComputeShader = GetMaxAvgMinCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMaxAvgMinCLLvalue1
//  {
//    ComputeShader = GetMaxAvgMinCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMaxCLLvalue0
//  {
//    ComputeShader = getMaxCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMaxCLLvalue1
//  {
//    ComputeShader = getMaxCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetAvgCLLvalue0
//  {
//    ComputeShader = getAvgCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetAvgCLLvalue1
//  {
//    ComputeShader = getAvgCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMinCLLvalue0
//  {
//    ComputeShader = getMinCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMinCLLvalue1
//  {
//    ComputeShader = getMinCLL1 <1, 1>;
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
    VertexShader = PostProcessVS;
     PixelShader = Testy;
  }
}
#endif

technique lilium__hdr_analysis
<
  ui_label = "Lilium's HDR analysis";
>
{

//CLL
#if (ENABLE_CLL_FEATURES == YES)

  pass CalcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = CalcCLL;
    RenderTarget = Texture_CLL_Values;
  }

  pass GetMaxAvgMinCLL0_NEW
  {
    ComputeShader = GetMaxAvgMinCLL0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass GetMaxAvgMinCLL1_NEW
  {
    ComputeShader = GetMaxAvgMinCLL1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass GetFinalMaxAvgMinCLL_NEW
  {
    ComputeShader = GetFinalMaxAvgMinCLL_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass ClearBrightnessHistogramTexture
  {
    VertexShader       = PostProcessVS;
     PixelShader       = ClearBrightnessHistogramTexture;
    RenderTarget       = Texture_Brightness_Histogram;
    ClearRenderTargets = true;
  }

  pass ComputeBrightnessHistogram
  {
    ComputeShader = ComputeBrightnessHistogram <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass RenderBrightnessHistogramToScale
  {
    VertexShader = PostProcessVS;
     PixelShader = RenderBrightnessHistogramToScale;
    RenderTarget = Texture_Brightness_Histogram_Final;
  }
#endif


//CIE
#if (ENABLE_CIE_FEATURES == YES)

#if (CIE_DIAGRAM == CIE_1931)
  pass Copy_CIE_1931_BG
  {
    VertexShader = PostProcessVS;
     PixelShader = Copy_CIE_1931_BG;
    RenderTarget = Texture_CIE_1931_Current;
  }
#endif

#if (CIE_DIAGRAM == CIE_1976)
  pass Copy_CIE_1976_BG
  {
    VertexShader = PostProcessVS;
     PixelShader = Copy_CIE_1976_BG;
    RenderTarget = Texture_CIE_1976_Current;
  }
#endif

  pass Generate_CIE_Diagram
  {
    ComputeShader = Generate_CIE_Diagram <THREAD_SIZE1, THREAD_SIZE1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = DISPATCH_Y1;
  }

#endif


//CSP
#if (ENABLE_CSP_FEATURES == YES \
  && ACTUAL_COLOUR_SPACE != CSP_SRGB)

  pass CalcCSPs
  {
    VertexShader = PostProcessVS;
     PixelShader = CalcCSPs;
    RenderTarget = Texture_CSPs;
  }

  pass CountCSPs_y
  {
    ComputeShader = CountCSPs_y <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass CountCSPs_x
  {
    ComputeShader = CountCSPs_x <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

#endif

  pass CopyShowValues
  {
    ComputeShader = ShowValuesCopy <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }


  pass PrepareOverlay
  {
    ComputeShader = PrepareOverlay <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass DrawOverlay
  {
    ComputeShader = DrawOverlay <1, 1>;
    DispatchSizeX = 15;
    DispatchSizeY = 16;
  }

  pass DrawNumbersToOverlay
  {
    ComputeShader = DrawNumbersToOverlay <1, 1>;
    DispatchSizeX = 12;
    DispatchSizeY = 16;
  }

  pass PS_HdrAnalysis
  {
    VertexShader = VS_PrepareHdrAnalysis;
     PixelShader = PS_HdrAnalysis;
  }
}

#else

ERROR_STUFF

technique lilium__hdr_analysis
<
  ui_label = "Lilium's HDR analysis (ERROR)";
>
{
  pass CS_Error
  {
    ComputeShader = CS_Error<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

#endif
