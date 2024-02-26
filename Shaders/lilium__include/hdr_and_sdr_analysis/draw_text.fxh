#pragma once


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

#define TEXTURE_OVERLAY_WIDTH  FONT_SIZE_56_CHAR_DIM.x * 26
#define TEXTURE_OVERLAY_HEIGHT FONT_SIZE_56_CHAR_DIM.y * (1                                \
                                                        + SHOW_NITS_VALUES_LINE_COUNT      \
                                                        + SHOW_NITS_FROM_CURSOR_LINE_COUNT \
                                                        + SHOW_CSPS_LINE_COUNT             \
                                                        + SHOW_CSP_FROM_CURSOR_LINE_COUNT  \
                                                        + 3)


texture2D TextureTextOverlay
<
  pooled = true;
>
{
  Width  = TEXTURE_OVERLAY_WIDTH;
  Height = TEXTURE_OVERLAY_HEIGHT;
  Format = RG8;
};

sampler2D<float4> SamplerTextOverlay
{
  Texture = TextureTextOverlay;
};

storage2D<float4> StorageTextOverlay
{
  Texture = TextureTextOverlay;
};


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


void CS_DrawTextToOverlay()
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
  const float showNitsLast       = tex1Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0);
  const float showCursorNitsLast = tex1Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1);

#ifdef IS_HDR_CSP
  const float showCspsLast      = tex1Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2);
  const float showCursorCspLast = tex1Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3);
#endif

  const float fontSizeLast      = tex1Dfetch(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4);

  if (showNitsLast       != showNitsValues
   || showCursorNitsLast != showNitsFromCrusor
#ifdef IS_HDR_CSP
   || showCspsLast       != showCsps
   || showCursorCspLast  != showCspFromCursor
#endif
   || fontSizeLast       != fontSize)
  {
    //store all current UI values
    tex1Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW0, showNitsValues);
    tex1Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW1, showNitsFromCrusor);
#ifdef IS_HDR_CSP
    tex1Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW2, showCsps);
    tex1Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW3, showCspFromCursor);
#endif
    tex1Dstore(StorageConsolidated, COORDS_CHECK_OVERLAY_REDRAW4, fontSize);

    //calculate offset for the cursor nits text in the overlay
    float cursorNitsYOffset = (!_SHOW_NITS_VALUES
                             ? -SHOW_NITS_VALUES_LINE_COUNT
                             : SPACING_MULTIPLIER)
                            + CSP_DESC_SPACING_MULTIPLIER;

    tex1Dstore(StorageConsolidated,
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

    tex1Dstore(StorageConsolidated,
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

    tex1Dstore(StorageConsolidated,
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
#define cursorNitsYOffset             4.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS)
#define cspsBt709PercentageYOffset    5.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsDciP3PercentageYOffset    6.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsBt2020PercentageYOffset   7.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsAp0PercentageYOffset      8.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cspsInvalidPercentageYOffset  9.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS)
#define cursorCspYOffset             10.f + tex1Dfetch(StorageConsolidated, COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP)


#define DRAW_NUMBERS(SHOW_IT, NUMBER, FETCH_COORDS, DRAW_TYPE, DRAW_OFFSET)   \
  if (SHOW_IT)                                                                \
  {                                                                           \
    precise float fullNumber = tex1Dfetch(StorageConsolidated, FETCH_COORDS); \
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
