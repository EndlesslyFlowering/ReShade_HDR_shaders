#pragma once


// outer spacing is half the size of a character rounded up
uint GetOuterSpacing
(
  const uint CharXDimension
)
{
  float charXDimAsFloat = float(CharXDimension);

  return uint(charXDimAsFloat / 2.f + 0.5f);
}


float2 GetTexCoordsFromRegularCoords(const float2 TexCoordOffset)
{
  return TexCoordOffset / float2(FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT);
}

float2 GetPositonCoordsFromRegularCoords
(
  const float2 RegularCoords,
  const float2 TextureSize
)
{
  float2 positionCoords = RegularCoords / TextureSize * 2;

  return float2(positionCoords.x - 1.f,
                1.f              - positionCoords.y);
}


//extract all digits without causing float issues
uint _6th
(
  precise const float Float
)
{
  return uint(Float) / 100000u;
}

uint _5th
(
  precise const float Float
)
{
  return uint(Float) /  10000u;
}

uint _4th
(
  precise const float Float
)
{
  return uint(Float) /   1000u;
}

uint _3rd
(
  precise const float Float
)
{
  return uint(Float) /    100u;
}

uint _2nd
(
  precise const float Float
)
{
  return uint(Float) /     10u;
}

uint _1st
(
  precise const float Float
)
{
  return uint(Float) %     10u;
}


uint d1st
(
  precise const float Float
)
{
  return uint((Float % 1.f) *       10.f) % 10u;
}

uint d2nd
(
  precise const float Float
)
{
  return uint((Float % 1.f) *      100.f) % 10u;
}

uint d3rd
(
  precise const float Float
)
{
  if (Float < 10000.f)
  {
    return uint((Float % 1.f) *     1000.f) % 10u;
  }
  return 11u;
}

uint _d3rd
(
  precise const float Float
)
{
  return uint((Float % 1.f) *     1000.f) % 10u;
}

uint d4th
(
  precise const float Float
)
{
  if (Float < 1000.f)
  {
    return uint((Float % 1.f) *    10000.f) % 10u;
  }
  return 11u;
}

uint d5th
(
  precise const float Float
)
{
  if (Float < 100.f)
  {
    return uint((Float % 1.f) *   100000.f) % 10u;
  }
  return 11u;
}

uint d6th
(
  precise const float Float
)
{
  if (Float < 10.f)
  {
    return uint((Float % 1.f) *  1000000.f) % 10u;
  }
  return 11u;
}

uint d7th
(
  precise const float Float
)
{
  if (Float < 1.f)
  {
    return uint((Float % 1.f) * 10000000.f) % 10u;
  }
  return 11u;
}


uint GetNumberAboveZero
(
  precise uint CurNumber
)
{
  if (CurNumber > 0)
  {
    return CurNumber % 10u;
  }
  else
  {
    return 11u;
  }
}

#ifdef IS_COMPUTE_CAPABLE_API

void DrawText()
{
#ifdef IS_HDR_CSP

  #define ARRAY_SIZE_CHAR_LIST_HEADER 20

  #define CHAR_ANALYSIS _H

  #if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    #define TEXT_CSP _s, _c, _R, _G, _B

  #else //CSP_HDR10

    #define TEXT_CSP _H, _D, _R, _1, _0

  #endif

#else //IS_HDR_CSP

  #if (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)

    #define ARRAY_SIZE_CHAR_LIST_HEADER 19

  #else

    #define ARRAY_SIZE_CHAR_LIST_HEADER 23

  #endif

  #define CHAR_ANALYSIS _S

  #if (OVERWRITE_SDR_GAMMA == GAMMA_24)

    #define TEXT_CSP _g, _a, _m, _m, _a, _2, _dot, _4

  #elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)

    #define TEXT_CSP _s, _R, _G, _B

  #else

    #define TEXT_CSP _g, _a, _m, _m, _a, _2, _dot, _2

  #endif

#endif //IS_HDR_CSP

  static const uint2 charListHeaderOffset = uint2(0, 2);

  static const uint2 charListHeader[ARRAY_SIZE_CHAR_LIST_HEADER] =
  {
    CHAR_ANALYSIS, _D, _R, _space,

    _A, _n, _a, _l, _y, _s, _i, _s, _space,

    _roundBracketOpen, TEXT_CSP, _roundBracketClose
  };


#ifdef IS_HDR_CSP

  //8
  #define SPACES_NITS    _space, _space, _space, _space, _space, _space, _space, _space
  //4
  #define SPACES_RED     _space, _space, _space
  //2
  #define SPACES_GREEN() _space,

  #define SPACES_NITS_COUNT  8
  #define SPACES_RED_COUNT   3
  #define SPACES_GREEN_COUNT 1

#else

  //7
  #define SPACES_NITS    _space, _space, _space, _space, _space, _space, _space
  //2
  #define SPACES_RED     _space, _space
  //0
  #define SPACES_GREEN()

  #define SPACES_NITS_COUNT  7
  #define SPACES_RED_COUNT   2
  #define SPACES_GREEN_COUNT 0

#endif

#define CHARS_CLL_IN_BRACKETS _space, _roundBracketOpen, _C, _L, _L, _roundBracketClose

#define CHARS_COUNT_CLL_IN_BRACKETS 6

#define ARRAY_SIZE_CHAR_LIST_TEXT_NITS_RGB (1 + 4                               + SPACES_NITS_COUNT  \
                                          + 1 + 3 + CHARS_COUNT_CLL_IN_BRACKETS + SPACES_RED_COUNT   \
                                          + 1 + 5 + CHARS_COUNT_CLL_IN_BRACKETS + SPACES_GREEN_COUNT \
                                          + 1 + 4 + CHARS_COUNT_CLL_IN_BRACKETS)

  static const uint2 charListTextNitsRGBOffset = uint2(0, 3);

  static const uint2 charListTextNitsRGB[ARRAY_SIZE_CHAR_LIST_TEXT_NITS_RGB] =
  {
    _verticalLine, _n, _i, _t, _s,                             SPACES_NITS,
    _verticalLine, _r, _e, _d,         CHARS_CLL_IN_BRACKETS,  SPACES_RED,
    _verticalLine, _g, _r, _e, _e, _n, CHARS_CLL_IN_BRACKETS,  SPACES_GREEN()
    _verticalLine, _b, _l, _u, _e,     CHARS_CLL_IN_BRACKETS,
  };


#ifdef IS_HDR_CSP

  //5
  #define SPACES_PRE_DOT   _space, _space, _space, _space, _space
  //6
  #define SPACES_AFTER_DOT _space, _space, _space, _space, _space, _space

  #define SPACES_PRE_DOT_COUNT   5
  #define SPACES_AFTER_DOT_COUNT 6

#else

  //3
  #define SPACES_PRE_DOT   _space, _space, _space
  //6
  #define SPACES_AFTER_DOT _space, _space, _space, _space, _space, _space

  #define SPACES_PRE_DOT_COUNT   3
  #define SPACES_AFTER_DOT_COUNT 6

#endif

#ifdef IS_HDR_CSP

  #define SDR_PERCENT()

  #define SDR_PERCENT_COUNT 0

  #define SDR_SPACES_FOR_LAST_PERCENT()

  #define SDR_SPACES_FOR_LAST_PERCENT_COUNT 0

#else //IS_HDR_CSP

  #define SDR_PERCENT() _percent,

  #define SDR_PERCENT_COUNT 1

  #define SDR_SPACES_FOR_LAST_PERCENT() SPACES_AFTER_DOT ## ,

  #define SDR_SPACES_FOR_LAST_PERCENT_COUNT SPACES_AFTER_DOT_COUNT

#endif //IS_HDR_CSP

#define ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB                                        \
  (3                                                                                     \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SDR_SPACES_FOR_LAST_PERCENT_COUNT + SDR_PERCENT_COUNT)

  static const uint2 charListMaxNitsRGBOffset = uint2(0, 4);

  static const uint2 charListMaxNitsRGB[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB] =
  {
    _m, _a, _x,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //red
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //green
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //blue
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

  static const uint2 charListAvgNitsRGBOffset = uint2(0, 5);

  static const uint2 charListAvgNitsRGB[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB] =
  {
    _a, _v, _g,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //red
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //green
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //blue
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

  static const uint2 charListMinNitsRGBOffset = uint2(0, 6);

  static const uint2 charListMinNitsRGB[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB] =
  {
    _m, _i, _n,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //red
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //green
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //blue
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

#define ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_RGB                                       \
  (6                                                                               \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SDR_SPACES_FOR_LAST_PERCENT_COUNT + SDR_PERCENT_COUNT)

  static const uint2 charListCursorNitsRGBOffset = uint2(0, 7);

  static const uint2 charListCursorNitsRGB[ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_RGB] =
  {
    _c, _u, _r, _s, _o, _r,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //red
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //green
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //blue
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };


#define ARRAY_SIZE_CHAR_LIST_TEXT_NITS_CLL (1 + 4 + SPACES_NITS_COUNT \
                                          + 1 + 3)

#ifdef IS_HDR_CSP
  static const uint2 charListTextNitsCllBaseOffset = uint2(24, 8);
#else
  static const uint2 charListTextNitsCllBaseOffset = uint2(0, 8);
#endif

  static const uint2 charListTextNitsCllOffset = charListTextNitsCllBaseOffset;

  static const uint2 charListTextNitsCll[ARRAY_SIZE_CHAR_LIST_TEXT_NITS_CLL] =
  {
    _verticalLine, _n, _i, _t, _s, SPACES_NITS,
    _verticalLine, _C, _L, _L
  };

#define ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL                                        \
  (3                                                                                     \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SDR_SPACES_FOR_LAST_PERCENT_COUNT + SDR_PERCENT_COUNT)

  static const uint2 charListMaxNitsCllOffset = charListTextNitsCllBaseOffset + uint2(0, 1);

  static const uint2 charListMaxNitsCll[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL] =
  {
    _m, _a, _x,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //CLL
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

  static const uint2 charListAvgNitsCllOffset = charListTextNitsCllBaseOffset + uint2(0, 2);

  static const uint2 charListAvgNitsCll[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL] =
  {
    _a, _v, _g,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //CLL
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

  static const uint2 charListMinNitsCllOffset = charListTextNitsCllBaseOffset + uint2(0, 3);

  static const uint2 charListMinNitsCll[ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL] =
  {
    _m, _i, _n,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //CLL
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };

#define ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_CLL                                             \
  (6                                                                                     \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SPACES_AFTER_DOT_COUNT            + SDR_PERCENT_COUNT  \
 + 1 + SPACES_PRE_DOT_COUNT + 1 + SDR_SPACES_FOR_LAST_PERCENT_COUNT + SDR_PERCENT_COUNT)

  static const uint2 charListCursorNitsCllOffset = charListTextNitsCllBaseOffset + uint2(0, 4);

  static const uint2 charListCursorNitsCll[ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_CLL] =
  {
    _c, _u, _r, _s, _o, _r,

    //nits
    _verticalLine, SPACES_PRE_DOT, _dot, SPACES_AFTER_DOT,             SDR_PERCENT()
    //CLL
    _verticalLine, SPACES_PRE_DOT, _dot, SDR_SPACES_FOR_LAST_PERCENT() SDR_PERCENT()
  };


#ifdef IS_HDR_CSP

  #define SPACES_GAMUTS _space, _space, _space, _space, _dot, _space, _space, _space

  static const uint2 charListGamutsBaseOffset = uint2(0, charListTextNitsCllOffset.y);

  static const uint2 charListGamutBt709Offset = charListGamutsBaseOffset + uint2(1, 0);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_BT709 16

  static const uint2 charListGamutBt709[ARRAY_SIZE_CHAR_LIST_GAMUT_BT709] =
  {
    _B, _T, _dot, _7, _0, _9, _colon, SPACES_GAMUTS, _percent
  };

  static const uint2 charListGamutDciP3Offset = charListGamutsBaseOffset + uint2(1, 1);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_DCI_P3 16

  static const uint2 charListGamutDciP3[ARRAY_SIZE_CHAR_LIST_GAMUT_DCI_P3] =
  {
    _D, _C, _I, _minus, _P, _3, _colon, SPACES_GAMUTS, _percent
  };

  static const uint2 charListGamutBt2020Offset = charListGamutsBaseOffset + uint2(0, 2);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_BT2020 17

  static const uint2 charListGamutBt2020[ARRAY_SIZE_CHAR_LIST_GAMUT_BT2020] =
  {
    _B, _T, _dot, _2, _0, _2, _0, _colon, SPACES_GAMUTS, _percent
  };

  static const uint2 charListGamutAp0Offset = charListGamutsBaseOffset + uint2(4, 3);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_AP0 13

  static const uint2 charListGamutAp0[ARRAY_SIZE_CHAR_LIST_GAMUT_AP0] =
  {
    _A, _P, _0, _colon, SPACES_GAMUTS, _percent
  };

  static const uint2 charListGamutInvalidOffset = charListGamutsBaseOffset + uint2(0, 4);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_INVALID 17

  static const uint2 charListGamutInvalid[ARRAY_SIZE_CHAR_LIST_GAMUT_INVALID] =
  {
    _i, _n, _v, _a, _l, _i, _d, _colon, SPACES_GAMUTS, _percent
  };

  static const uint2 charListGamutCursorOffset = uint2(ARRAY_SIZE_CHAR_LIST_HEADER + 1,
                                                       charListHeaderOffset.y);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR 7

  static const uint2 charListGamutCursor[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR] =
  {
    _c, _u, _r, _s, _o, _r, _colon
  };


  static const uint2 charListGamutCursorTextBt709Offset = uint2(17, charListGamutsBaseOffset.y);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT709 6

  static const uint2 charListGamutCursorTextBt709[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT709] =
  {
    _B, _T, _dot, _7, _0, _9
  };

  static const uint2 charListGamutCursorTextDciP3Offset = charListGamutCursorTextBt709Offset + uint2(0, 1);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_DCI_P3 6

  static const uint2 charListGamutCursorTextDciP3[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_DCI_P3] =
  {
    _D, _C, _I, _minus, _P, _3
  };

  static const uint2 charListGamutCursorTextBt2020Offset = charListGamutCursorTextBt709Offset + uint2(0, 2);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT2020 7

  static const uint2 charListGamutCursorTextBt2020[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT2020] =
  {
    _B, _T, _dot, _2, _0, _2, _0
  };

  static const uint2 charListGamutCursorTextAp0Offset = charListGamutCursorTextBt709Offset + uint2(0, 3);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_AP0 3

  static const uint2 charListGamutCursorTextAp0[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_AP0] =
  {
    _A, _P, _0
  };

  static const uint2 charListGamutCursorTextInvalidOffset = charListGamutCursorTextBt709Offset + uint2(0, 4);

#define ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_INVALID 7

  static const uint2 charListGamutCursorTextInvalid[ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_INVALID] =
  {
    _i, _n, _v, _a, _l, _i, _d
  };

#endif //IS_HDR_CSP


#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  #define COUNT_CHAR_LISTS 22

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  #define COUNT_CHAR_LISTS 18

#else

  #define COUNT_CHAR_LISTS 11

#endif


  static const int2 offsets[COUNT_CHAR_LISTS] =
  {
    charListHeaderOffset,
    charListTextNitsRGBOffset,
    charListMaxNitsRGBOffset,
    charListAvgNitsRGBOffset,
    charListMinNitsRGBOffset,
    charListCursorNitsRGBOffset,
    charListTextNitsCllOffset,
    charListMaxNitsCllOffset,
    charListAvgNitsCllOffset,
    charListMinNitsCllOffset,
    charListCursorNitsCllOffset,
#ifdef IS_HDR_CSP
    charListGamutBt709Offset,
    charListGamutDciP3Offset,
    charListGamutBt2020Offset,
    charListGamutCursorOffset,
    charListGamutCursorTextBt709Offset,
    charListGamutCursorTextDciP3Offset,
    charListGamutCursorTextBt2020Offset,
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    charListGamutAp0Offset,
    charListGamutInvalidOffset,
    charListGamutCursorTextAp0Offset,
    charListGamutCursorTextInvalidOffset,
#endif
#endif
  };


  static const uint arraySizes[COUNT_CHAR_LISTS] =
  {
    ARRAY_SIZE_CHAR_LIST_HEADER,
    ARRAY_SIZE_CHAR_LIST_TEXT_NITS_RGB,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_RGB,
    ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_RGB,
    ARRAY_SIZE_CHAR_LIST_TEXT_NITS_CLL,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL,
    ARRAY_SIZE_CHAR_LIST_MAX_AVG_MIN_NITS_CLL,
    ARRAY_SIZE_CHAR_LIST_CURSOR_NITS_CLL,
#ifdef IS_HDR_CSP
    ARRAY_SIZE_CHAR_LIST_GAMUT_BT709,
    ARRAY_SIZE_CHAR_LIST_GAMUT_DCI_P3,
    ARRAY_SIZE_CHAR_LIST_GAMUT_BT2020,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT709,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_DCI_P3,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_BT2020,
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    ARRAY_SIZE_CHAR_LIST_GAMUT_AP0,
    ARRAY_SIZE_CHAR_LIST_GAMUT_INVALID,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_AP0,
    ARRAY_SIZE_CHAR_LIST_GAMUT_CURSOR_TEXT_INVALID
#endif
#endif
  };


  [loop]
  for (uint i = 0; i < COUNT_CHAR_LISTS; i++)
  {

    [loop]
    for (uint j = 0; j < arraySizes[i]; j++)
    {

      uint2 currentChar;

      [forcecase]
      switch(i)
      {
        case 0:
        {
          currentChar = charListHeader[j];
        }
        break;
        case 1:
        {
          currentChar = charListTextNitsRGB[j];
        }
        break;
        case 2:
        {
          currentChar = charListMaxNitsRGB[j];
        }
        break;
        case 3:
        {
          currentChar = charListAvgNitsRGB[j];
        }
        break;
        case 4:
        {
          currentChar = charListMinNitsRGB[j];
        }
        break;
        case 5:
        {
          currentChar = charListCursorNitsRGB[j];
        }
        break;
        case 6:
        {
          currentChar = charListTextNitsCll[j];
        }
        break;
        case 7:
        {
          currentChar = charListMaxNitsCll[j];
        }
        break;
        case 8:
        {
          currentChar = charListAvgNitsCll[j];
        }
        break;
        case 9:
        {
          currentChar = charListMinNitsCll[j];
        }
        break;
#ifdef IS_HDR_CSP
        case 10:
#else
        default:
#endif
        {
          currentChar = charListCursorNitsCll[j];
        }
        break;
#ifdef IS_HDR_CSP
        case 11:
        {
          currentChar = charListGamutBt709[j];
        }
        break;
        case 12:
        {
          currentChar = charListGamutDciP3[j];
        }
        break;
        case 13:
        {
          currentChar = charListGamutBt2020[j];
        }
        break;
        case 14:
        {
          currentChar = charListGamutCursor[j];
        }
        break;
        case 15:
        {
          currentChar = charListGamutCursorTextBt709[j];
        }
        break;
        case 16:
        {
          currentChar = charListGamutCursorTextDciP3[j];
        }
        break;
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        case 17:
#else
        default:
#endif
        {
          currentChar = charListGamutCursorTextBt2020[j];
        }
        break;
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
        case 18:
        {
          currentChar = charListGamutAp0[j];
        }
        break;
        case 19:
        {
          currentChar = charListGamutInvalid[j];
        }
        break;
        case 20:
        {
          currentChar = charListGamutCursorTextAp0[j];
        }
        break;
        default:
        {
          currentChar = charListGamutCursorTextInvalid[j];
        }
        break;
#endif
#endif
      }

      [branch]
      if (any(currentChar != _space))
      {
        const uint2 storeOffset = (offsets[i] + uint2(j, 0))
                                * CHAR_DIM_UINT;

        const uint2 currentCharOffset = currentChar
                                      * CHAR_DIM_UINT;
        [loop]
        for (int x = 0; x < CHAR_DIM_UINT.x; x++)
        {
          [loop]
          for (int y = 0; y < CHAR_DIM_UINT.y; y++)
          {
            const int2 xy = int2(x, y);

            const int2 currentFetchOffset = int2(currentCharOffset)
                                          + xy;

            const int2 currentStoreOffset = int2(storeOffset)
                                          + xy;

            const float4 currentPixel = tex2Dfetch(StorageFontAtlasConsolidated, currentFetchOffset);

            tex2Dstore(StorageFontAtlasConsolidated, currentStoreOffset, currentPixel);
          }
        }
      }
    }
  }
}


void CS_GetNitNumbers
(
  uint3 GID  : SV_GroupID,
  uint3 GTID : SV_GroupThreadID,
  uint3 DTID : SV_DispatchThreadID
)
{
  static const bool showCllValues = _SHOW_RGB_OR_CLL == SHOW_CLL_VALUES;

  [branch]
  if (showCllValues
   && GID.x > 1)
  {
    return;
  }

  static const int2 storePos = int2(DTID.xy);

  float nits;

  [branch]
  if (GID.y < 3
   && _SHOW_NITS_VALUES)
  {
    int fetchPos = COORDS_SHOW_MAX_NITS + GID.y;

    fetchPos += showCllValues ? (12 * GID.x)
                              :  (3 * GID.x);

    nits = tex1Dfetch(SamplerConsolidated, fetchPos);
  }
  else
  BRANCH()
  if (_SHOW_NITS_FROM_CURSOR)
  {
    const int2 mousePosition = clamp(MOUSE_POSITION, 0, BUFFER_SIZE_MINUS_1_INT);
    //loading into groupshared memory is not worth it
    const float4 RgbNits = CalcNitsAndCll(tex2Dfetch(SamplerBackBuffer, mousePosition).rgb);
    const float  Cll     = MAXRGB(RgbNits.rgb);

    nits = showCllValues ? GID.x == 0u ? RgbNits.w : Cll
                         : RgbNits[(GID.x + 3) % 4];
  }
  else
  {
    nits = 0.f;
  }

#ifdef IS_FLOAT_HDR_CSP
  static /*const*/ uint minus0 = uint(0x80000000); //massaging the ReShade compiler...
  static const bool isMinus0 = asuint(nits) == minus0;

  static const uint negSignPos =  nits <= -1000.f  ? 5
                               :  nits <=  -100.f  ? 4
                               :  nits <=   -10.f  ? 3
                               : (isMinus0
                               || nits <      0.f) ? 2
                               :                     9;


  // avoid max nits above or below these values looking cut off in the overlay
  nits = clamp(nits, -9999.999f, 99999.99f);
  nits = abs(nits);
#endif

//  bool4 RgbNitsAbove100000 = RgbNits >= 100000.f;
//
//  if (RgbNitsAbove100000.w)
//  {
//    tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePosNits, 9u);
//  }
//  if (RgbNitsAbove100000.r)
//  {
//    tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePosR, 9u);
//  }
//  if (RgbNitsAbove100000.g)
//  {
//    tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePosG, 9u);
//  }
//  if (RgbNitsAbove100000.b)
//  {
//    tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePosB, 9u);
//  }
//
//  [branch]
//  if (all(RgbNitsAbove100000))
//  {
//    return;
//  }

#ifdef IS_HDR_CSP

  #define CASE2 2
  #define CASE3 3
  #define CASE4 4
  #define CASE5 5
  #define CASE6 6
  #define CASE7 7
  #define CASE8 8
  #define CASE9 9

#else

  #define CASE2 0
  #define CASE3 1
  #define CASE4 2
  #define CASE5 3
  #define CASE6 4
  #define CASE7 5
  #define CASE8 6
  #define CASE9 7

#endif

  switch(GTID.x)
  {
#ifdef IS_HDR_CSP
    case 0:
    {
      precise uint _00;

#ifdef IS_FLOAT_HDR_CSP
      if (negSignPos == 5)
      {
        _00 = _minus.x;
      }
      else
      {
#endif
        _00 = GetNumberAboveZero(_5th(nits));
#ifdef IS_FLOAT_HDR_CSP
      }
#endif

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _00);
    }
    break;
    case 1:
    {
      precise uint _01;

#ifdef IS_FLOAT_HDR_CSP
      if (negSignPos == 4)
      {
        _01 = _minus.x;
      }
      else
      {
#endif
        _01 = GetNumberAboveZero(_4th(nits));
#ifdef IS_FLOAT_HDR_CSP
      }
#endif

      if (GID.y == 3 && _01 == 9u)
        _01 = 1u;

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _01);
    }
    break;
#endif
    case CASE2:
    {
      precise uint _02;

#ifdef IS_FLOAT_HDR_CSP
      if (negSignPos == 3)
      {
        _02 = _minus.x;
      }
      else
      {
#endif
        _02 = GetNumberAboveZero(_3rd(nits));
#ifdef IS_FLOAT_HDR_CSP
      }
#endif

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _02);
    }
    break;
    case CASE3:
    {
      precise uint _03;

#ifdef IS_FLOAT_HDR_CSP
      if (negSignPos == 2)
      {
        _03 = _minus.x;
      }
      else
      {
#endif
        _03 = GetNumberAboveZero(_2nd(nits));
#ifdef IS_FLOAT_HDR_CSP
      }
#endif

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _03);
    }
    break;
    case CASE4:
    {
      precise const uint _04 = _1st(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _04);
    }
    break;
    case CASE5:
    {
      precise const uint _05 = d1st(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _05);
    }
    break;
    case CASE6:
    {
      precise const uint _06 = d2nd(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _06);
    }
    break;
    case CASE7:
    {
      precise const uint _07 = d3rd(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _07);
    }
    break;
    case CASE8:
    {
      precise const uint _08 = d4th(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _08);
    }
    break;
    case CASE9:
    {
      precise const uint _09 = d5th(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _09);
    }
    break;
    default:
    {
      precise const uint _10 = d6th(nits);

      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _10);
    }
    break;
  }

}

#ifdef IS_HDR_CSP
void CS_GetGamutNumbers
(
  uint3 GID  : SV_GroupID,
  uint3 GTID : SV_GroupThreadID,
  uint3 DTID : SV_DispatchThreadID
)
{
  static const int2 storePos = int2(DTID.x + NITS_NUMBERS_PER_ROW, GID.y);

  const float curGamutCounter = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_BT709 + GID.y);

  switch(DTID.x)
  {
    case 0:
    {
      precise const uint _00 = GetNumberAboveZero(_3rd(curGamutCounter));
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _00);
    }
    break;
    case 1:
    {
      precise const uint _01 = GetNumberAboveZero(_2nd(curGamutCounter));
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _01);
    }
    break;
    case 2:
    {
      precise const uint _02 = _1st(curGamutCounter);
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _02);
    }
    break;
    case 3:
    {
      precise const uint _03 = d1st(curGamutCounter);
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _03);
    }
    break;
    case 4:
    {
      precise const uint _04 = d2nd(curGamutCounter);
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _04);
    }
    break;
    default:
    {
      precise const uint _05 = _d3rd(curGamutCounter);
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, _05);
    }
    break;
  }
}
#endif //IS_HDR_CSP

#else //IS_COMPUTE_CAPABLE_API

static const float2 ShowNumbersTextureSize =
 float2(TEXTURE_MAX_AVG_MIN_NITS_AND_GAMUT_COUNTER_AND_SHOW_NUMBERS_WIDTH,
        TEXTURE_MAX_AVG_MIN_NITS_AND_GAMUT_COUNTER_AND_SHOW_NUMBERS_HEIGHT);

void VS_PrepareGetNumbersNits
(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position
)
{
  static const float2 positions[3] =
  {
    GetPositonCoordsFromRegularCoords(float2( NITS_NUMBERS_COUNT,  NITS_NUMBERS_ROWS), ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2(-NITS_NUMBERS_COUNT,  NITS_NUMBERS_ROWS), ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2( NITS_NUMBERS_COUNT, -NITS_NUMBERS_ROWS), ShowNumbersTextureSize)
  };

  Position = float4(positions[VertexID], 0.f, 1.f);

  return;
}

void PS_GetNumbersNits
(
  in  float4 Position : SV_Position,
  out float  Number   : SV_Target0
)
{
  const int2 positionAsInt2 = int2(Position.xy);

  precise float nitsValue;

  if (positionAsInt2.y < 3)
  {
    nitsValue = tex2Dfetch(SamplerTransfer, int2(COORDS_SHOW_MAX_NITS + positionAsInt2.y, 0)).x;
  }
  else
  {
    const int2 mousePosition = clamp(MOUSE_POSITION, 0, BUFFER_SIZE_MINUS_1_INT);
    nitsValue = CalcNits(tex2Dfetch(SamplerBackBuffer, mousePosition).rgb);
  }

  precise uint number;

#ifdef IS_HDR_CSP
  switch(positionAsInt2.x)
#else
  switch(positionAsInt2.x + 2)
#endif
  {
    case 0:
    {
      number = GetNumberAboveZero(_5th(nitsValue));
    }
    break;
    case 1:
    {
      number = GetNumberAboveZero(_4th(nitsValue));
    }
    break;
    case 2:
    {
      number = GetNumberAboveZero(_3rd(nitsValue));
    }
    break;
    case 3:
    {
      number = GetNumberAboveZero(_2nd(nitsValue));
    }
    break;
    case 4:
    {
      number = _1st(nitsValue);
    }
    break;
    case 5:
    {
      number = d1st(nitsValue);
    }
    break;
    case 6:
    {
      number = d2nd(nitsValue);
    }
    break;
    case 7:
    {
      number = d3rd(nitsValue);
    }
    break;
    case 8:
    {
      number = d4th(nitsValue);
    }
    break;
    case 9:
    {
      number = d5th(nitsValue);
    }
    break;
    default:
    {
      number = d6th(nitsValue);
    }
    break;
  }

  Number = float(number) / 254.f; // /254 for safety
}


#ifdef IS_HDR_CSP
void VS_PrepareGetGamutNumbers
(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position
)
{
  static const float2 positions[3] =
  {
    GetPositonCoordsFromRegularCoords(float2( GAMUTS_NUMBERS_COUNT, GAMUTS_Y_OFFSET),                             ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2(-GAMUTS_NUMBERS_COUNT, GAMUTS_Y_OFFSET),                             ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2( GAMUTS_NUMBERS_COUNT, (GAMUTS_Y_OFFSET + GAMUTS_NUMBERS_ROWS * 2)), ShowNumbersTextureSize)
  };

  Position = float4(positions[VertexID], 0.f, 1.f);

  return;
}


void PS_GetGamutNumbers
(
  in  float4 Position : SV_Position,
  out float  Number   : SV_Target0
)
{
  const int2 positionAsInt2 = int2(Position.xy);

  precise const float gamutCount = tex2Dfetch(SamplerTransfer,
                                              int2(COORDS_SHOW_PERCENTAGE_BT709 - 4 + positionAsInt2.y, 0)).x;

  precise uint number;

  switch(positionAsInt2.x)
  {
    case 0:
    {
      number = GetNumberAboveZero(_3rd(gamutCount));
    }
    break;
    case 1:
    {
      number = GetNumberAboveZero(_2nd(gamutCount));
    }
    break;
    case 2:
    {
      number = _1st(gamutCount);
    }
    break;
    case 3:
    {
      number = d1st(gamutCount);
    }
    break;
    case 4:
    {
      number = d2nd(gamutCount);
    }
    break;
    default:
    {
      number = _d3rd(gamutCount);
    }
    break;
  }

  Number = float(number) / 254.f; // /254 for safety
}
#endif //IS_HDR_CSP

#endif //IS_COMPUTE_CAPABLE_API


float3 MergeText
(
  float3 Output,
  float4 Mtsdf,
  float  ScreenPixelRange
)
{
  const float2 opacities = Msdf::GetTextOpacities(Mtsdf, ScreenPixelRange);

  const float innerOpacity = opacities[0];

  const float outerOpacity = opacities[1];

  // tone map pixels below the overlay area
  [branch]
  if (_TEXT_BG_ALPHA > 0.f
   || innerOpacity   > 0.f
   || outerOpacity   > 0.f)
  {
    // first set 1.0 to be equal to _TEXT_BRIGHTNESS
    float adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    adjustFactor = _TEXT_BRIGHTNESS / 80.f;

    Output = Csp::Mat::Bt709To::Bt2020(Output);

    // safety clamp colours outside of BT.2020
    Output = max(Output, 0.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    adjustFactor = _TEXT_BRIGHTNESS / 10000.f;

    Output = FetchFromHdr10ToLinearLUT(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    adjustFactor = _TEXT_BRIGHTNESS / 100.f;

    Output = DECODE_SDR(Output);

#endif

    Output /= adjustFactor;

    // then tone map to 1.0 at max
    ExtendedReinhardTmo(Output, _TEXT_BRIGHTNESS);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    // safety clamp for the case that there are values that represent above 10000 nits
    Output.rgb = min(Output.rgb, 1.f);

#endif

    // apply the background
    Output = lerp(Output, 0.f, _TEXT_BG_ALPHA / 100.f);

    // apply the text
    Output = lerp(Output, 0.f, outerOpacity);
    Output = lerp(Output, 1.f, innerOpacity);

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

  return Output;
}


struct VertexCoordsAndTexCoords
{
  float2 vertexCoords;
  float2 texCoords;
};

VertexCoordsAndTexCoords ReturnOffScreen()
{
  VertexCoordsAndTexCoords ret;

  ret.vertexCoords = float2(-2.f, -2.f);
  ret.texCoords    = float2(-2.f, -2.f);

  return ret;
}

#define MAX_LINES_NITS_RGB_CLL 6

#if defined(IS_FLOAT_HDR_CSP)
  #define MAX_LINES_GAMUT 6
#elif defined(IS_HDR10_LIKE_CSP)
  #define MAX_LINES_GAMUT 4
#endif

#define MAX_CHARS_NITS_RGB_CLL TEXT_BLOCK_SIZE_ANALYIS_HEADER.x

#ifdef IS_HDR_CSP
  #define NITS_EXTRA_CHARS 6
#else
  #define NITS_EXTRA_CHARS 0
#endif

struct MaxCharsAndMaxLines
{
  uint maxChars;
  uint maxLines;
};

float GetMaxCharsForNitsRgbCll()
{
  float maxChars = MAX_CHARS_NITS_RGB_CLL;

  FLATTEN()
  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    float textBlockSizeNitsRgbCursor = TEXT_BLOCK_SIZE_NITS_RGB_CURSOR.x;

    FLATTEN()
    if (_SHOW_RGB_OR_CLL == SHOW_CLL_VALUES)
    {
#ifdef IS_HDR_CSP
      textBlockSizeNitsRgbCursor -= 26.f;
#else
      textBlockSizeNitsRgbCursor -= 24.f;
#endif
    }

    FLATTEN()
    if (!_SHOW_NITS_FROM_CURSOR)
    {
      textBlockSizeNitsRgbCursor -= 3.f;
    }

    maxChars = max(TEXT_BLOCK_SIZE_ANALYIS_HEADER.x, textBlockSizeNitsRgbCursor + NITS_EXTRA_CHARS);
  }

  return maxChars;
}

#ifdef IS_HDR_CSP
float GetMaxCharsForGamut()
{
  float maxChars = 0.f;

  FLATTEN()
  if (SHOW_GAMUTS
   || SHOW_GAMUT_FROM_CURSOR)
  {
    maxChars = max(maxChars, TEXT_BLOCK_SIZE_GAMUT_PERCENTAGES.x + TEXT_BLOCK_DRAW_X_OFFSET[3]);
  }

  return maxChars;
}
#endif

MaxCharsAndMaxLines GetMaxCharsAndMaxLinesForNitsRgbCll()
{
  MaxCharsAndMaxLines ret;

  ret.maxChars = MAX_CHARS_NITS_RGB_CLL;
  ret.maxLines = MAX_LINES_NITS_RGB_CLL;

  FLATTEN()
  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    uint textOffsetNitsCursor = uint(TEXT_BLOCK_SIZE_NITS_RGB_CURSOR.x);

    FLATTEN()
    if (_SHOW_RGB_OR_CLL == SHOW_CLL_VALUES)
    {
#ifdef IS_HDR_CSP
      textOffsetNitsCursor -= 26u;
#else
      textOffsetNitsCursor -= 24u;
#endif
    }

    FLATTEN()
    if (!_SHOW_NITS_FROM_CURSOR)
    {
      textOffsetNitsCursor -= 3u;
    }

    ret.maxChars = max(ret.maxChars, textOffsetNitsCursor + NITS_EXTRA_CHARS);
  }

  FLATTEN()
  if (!_SHOW_NITS_VALUES)
  {
    ret.maxLines -= 3;
  }

  FLATTEN()
  if (!_SHOW_NITS_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

  FLATTEN()
  if (!_SHOW_NITS_VALUES
   && !_SHOW_NITS_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

  return ret;
}


#ifdef IS_HDR_CSP
MaxCharsAndMaxLines GetMaxCharsAndMaxLinesForGamut()
{
  MaxCharsAndMaxLines ret;

  ret.maxChars = uint(TEXT_BLOCK_SIZE_GAMUT_PERCENTAGES.x) + uint(TEXT_BLOCK_DRAW_X_OFFSET[3]);
  ret.maxLines = MAX_LINES_GAMUT;

  FLATTEN()
  if (!SHOW_GAMUTS)
  {
    ret.maxLines -= GAMUT_PERCENTAGES_LINES;
  }

  FLATTEN()
  if (!SHOW_GAMUT_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

  return ret;
}
#endif

VertexCoordsAndTexCoords GetVertexCoordsAndTexCoordsForTextBlocks
(
  const uint   VertexID,
  const float2 CharSize
)
{
  VertexCoordsAndTexCoords ret;

  switch(VertexID)
  {
    case 0u:
    case 1u:
    case 2u:
#ifdef IS_HDR_CSP
    case 3u:
    case 4u:
    case 5u:
#endif
    {
      MaxCharsAndMaxLines _max = GetMaxCharsAndMaxLinesForNitsRgbCll();

#ifdef IS_HDR_CSP
      [branch]
      if (VertexID > 2u)
      {
        MaxCharsAndMaxLines _max1 = GetMaxCharsAndMaxLinesForGamut();

        _max.maxChars  = _max1.maxChars;
        // this works because the 3rd vertex left or right of the target area
        // it's above but that is fine
        _max.maxLines += _max1.maxLines;
      }
#endif

      float2 pos = float2(_max.maxChars * CharSize.x,
                          _max.maxLines * CharSize.y);

      [flatten]
      if (VertexID % 3u == 1u)
      {
        pos.x = -pos.x;
      }
      else
      [flatten]
      if (VertexID % 3u == 2u)
      {
        pos.y = -pos.y;
      }

      [flatten]
      if (_TEXT_POSITION != 0u)
      {
        pos.x = BUFFER_WIDTH_FLOAT - pos.x;
      }

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(pos, BUFFER_SIZE_FLOAT);
//      ret.vertexCoords = float2(-2.f, -2.f);
      ret.texCoords    = float2(-1.f, -1.f);
    }
    break;
    default:
    {
#ifdef IS_HDR_CSP
      const uint textBlockVertexID = VertexID - 6u;
#else
      const uint textBlockVertexID = VertexID - 3u;
#endif

      const uint localVertexID = textBlockVertexID % 6u;

      uint currentTextBlockID = textBlockVertexID / 6u;

      float2 vertexOffset = (float2)0;

      float2 texCoordOffset;

      bool calcOffsets;

      //Analysis Header
      if (currentTextBlockID == 0)
      {
        calcOffsets = true;

        vertexOffset.y = 0.f;
      }
      //nits|red(CLL)|green(CLL)|blue(CLL)
      else if (currentTextBlockID == 1
            && (_SHOW_NITS_VALUES || _SHOW_NITS_FROM_CURSOR))
      {
        calcOffsets = true;

        vertexOffset.y = 1.f;
      }
      //max:
      //avg:
      //min:
      else if (currentTextBlockID == 2
            && _SHOW_NITS_VALUES)
      {
        calcOffsets = true;

        vertexOffset.y = 2.f;
      }
      //cursor:
      else if (currentTextBlockID == 3
            && _SHOW_NITS_FROM_CURSOR)
      {
        calcOffsets = true;

        vertexOffset.y = _SHOW_NITS_VALUES ? 5.f
                                           : 2.f;
      }
#ifdef IS_HDR_CSP
      //gamut percentages
      else if (currentTextBlockID == 4
            && SHOW_GAMUTS)
      {
        calcOffsets = true;

        vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                       : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                       : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                         : 1.f;
      }
      //cursor gamut
      else if (currentTextBlockID == 5
            && SHOW_GAMUT_FROM_CURSOR)
      {
        calcOffsets = true;

        vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                       : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                       : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                         : 1.f;

        [flatten]
        if (SHOW_GAMUTS)
        {
          vertexOffset.y += GAMUT_PERCENTAGES_LINES;
        }
        else
        {
          vertexOffset.x -= 1.f;
        }
      }
#endif
      else
      {
        calcOffsets = false;
      }

      [branch]
      if (calcOffsets)
      {
        vertexOffset.x += TEXT_BLOCK_DRAW_X_OFFSET[currentTextBlockID];

        float2 size = TEXT_BLOCK_SIZES[currentTextBlockID];

        texCoordOffset = TEXT_BLOCK_FETCH_OFFSETS[currentTextBlockID];

        [flatten]
        if (currentTextBlockID > 0
         && currentTextBlockID < 4)
        {
          [flatten]
          if (_SHOW_RGB_OR_CLL == SHOW_CLL_VALUES)
          {
#ifdef IS_HDR_CSP
            size.x -= 26.f;

            texCoordOffset.x += 24.f;
#else
            size.x -= 24.f;
#endif
            texCoordOffset.y += 5.f;
          }

          [flatten]
          if (!_SHOW_NITS_FROM_CURSOR)
          {
            vertexOffset.x -= 3.f;
          }
        }

        vertexOffset *= CharSize;

        texCoordOffset *= CHAR_DIM_FLOAT;

        const float2   vertexOffset2 = size * CharSize       +   vertexOffset;
        const float2 texCoordOffset2 = size * CHAR_DIM_FLOAT + texCoordOffset;

        [flatten]
        if (localVertexID == 1)
        {
            vertexOffset.y =   vertexOffset2.y;
          texCoordOffset.y = texCoordOffset2.y;
        }
        else
        [flatten]
        if (localVertexID == 4)
        {
            vertexOffset.x =   vertexOffset2.x;
          texCoordOffset.x = texCoordOffset2.x;
        }
        else
        [flatten]
        if (localVertexID == 2
         || localVertexID == 5)
        {
            vertexOffset =   vertexOffset2;
          texCoordOffset = texCoordOffset2;
        }

        FLATTEN()
        if (_TEXT_POSITION != 0u)
        {
          float maxChars;

#ifdef IS_HDR_CSP

          [flatten]
          if (currentTextBlockID == 4
           || currentTextBlockID == 5)
          {
            maxChars = GetMaxCharsForGamut();
          }
          else
          {
            maxChars = GetMaxCharsForNitsRgbCll();
          }
#else
          maxChars = GetMaxCharsForNitsRgbCll();
#endif
          vertexOffset.x += BUFFER_WIDTH_FLOAT - (maxChars * CharSize.x);
        }

        ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_FLOAT);
        ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);
      }
      else
      {
        ret = ReturnOffScreen();
      }
    }
    break;
  }

  return ret;
}


void VS_RenderText
(
  in                  uint   VertexID         : SV_VertexID,
  out                 float4 Position         : SV_Position,
  out                 float2 TexCoord         : TEXCOORD0,
  out nointerpolation float  ScreenPixelRange : ScreenPixelRange
)
{
  static const float2 charSize = CHAR_DIM_FLOAT * _TEXT_SIZE;

  const VertexCoordsAndTexCoords vertexCoordsAndTexCoords = GetVertexCoordsAndTexCoordsForTextBlocks(VertexID, charSize);

  Position = float4(vertexCoordsAndTexCoords.vertexCoords, 0.f, 1.f);

  TexCoord = vertexCoordsAndTexCoords.texCoords;

  ScreenPixelRange = Msdf::GetScreenPixelRange(_TEXT_SIZE);

  return;
}

void PS_RenderText
(
  in                  float4 Position         : SV_Position,
  in                  float2 TexCoord         : TEXCOORD0,
  in  nointerpolation float  ScreenPixelRange : ScreenPixelRange,
  out                 float4 Output           : SV_Target0
)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  Output.a = inputColour.a;

  float4 mtsdf = tex2D(SamplerFontAtlasConsolidated, TexCoord);

  Output.rgb = MergeText(inputColour.rgb,
                         mtsdf,
                         ScreenPixelRange);
}

VertexCoordsAndTexCoords GetVertexCoordsAndTexCoordsForNumbers
(
  const uint   VertexID,
  const float2 CharSize
)
{
  uint currentNumberID = VertexID / 6u;

  int2 fetchPos;

#ifdef IS_HDR_CSP
  [branch]
  if (currentNumberID < NITS_NUMBERS_TOTAL)
  {
#endif
    fetchPos = int2(currentNumberID % NITS_NUMBERS_PER_ROW,
                    currentNumberID / NITS_NUMBERS_PER_ROW);
#ifdef IS_HDR_CSP
  }
  else
  {
    uint currentGamutNumberID = currentNumberID - NITS_NUMBERS_TOTAL;

    fetchPos = int2(currentGamutNumberID % GAMUTS_NUMBERS_COUNT,
                    currentGamutNumberID / GAMUTS_NUMBERS_COUNT);

#ifdef IS_COMPUTE_CAPABLE_API
    fetchPos.x += NITS_NUMBERS_PER_ROW;
#else
    fetchPos.y += 4;
#endif
  }
#endif //IS_HDR_CSP

  static const uint curNumber = tex2Dfetch(SamplerMaxAvgMinNitsAndGamutCounterAndShowNumbers, fetchPos)
#ifndef IS_COMPUTE_CAPABLE_API
                              * 256.f /* *256 for safety */
#endif
                                     ;

#define SHOW_NITS_VALUES_NUMBER_ID_MAX      (NITS_NUMBERS_PER_ROW * 3)
#define SHOW_NITS_FROM_CURSOR_NUMBER_ID_MAX (NITS_NUMBERS_PER_ROW * 4)

#ifdef IS_FLOAT_HDR_CSP
  #define SHOW_GAMUTS_NUMBER_ID_MAX (NITS_NUMBERS_PER_ROW * 3 \
                                   + NITS_NUMBERS_PER_ROW * 1 \
                                   + 6 * 5)
#else
  #define SHOW_GAMUTS_NUMBER_ID_MAX (NITS_NUMBERS_PER_ROW * 3 \
                                   + NITS_NUMBERS_PER_ROW * 1 \
                                   + 6 * 3)
#endif

  VertexCoordsAndTexCoords ret;
  ret.vertexCoords = float2(0.f, 0.f);
  ret.texCoords    = float2(0.f, 0.f);

  [branch]
  if (curNumber < 11u)
  {
    const uint currentVertexID = VertexID % 6u;

    float2 vertexOffset;

#ifdef IS_HDR_CSP

  #define DOT_OFFSET_DIV 6u

#else

  #define DOT_OFFSET_DIV 4u

#endif

    static const bool drawMaxRbgOrMaxCll =
      ((_SHOW_RGB_OR_CLL == SHOW_CLL_VALUES && (currentNumberID % NITS_NUMBERS_PER_ROW) < (NITS_NUMBERS_COUNT * 2))
     || _SHOW_RGB_OR_CLL == SHOW_RGB_VALUES);

    bool calcOffsets = false;

    //max/avg/min nits
    [branch]
    if (_SHOW_NITS_VALUES
     && currentNumberID < SHOW_NITS_VALUES_NUMBER_ID_MAX
     && drawMaxRbgOrMaxCll)
    {
      calcOffsets = true;

      uint a = currentNumberID % NITS_NUMBERS_PER_ROW;

      uint b = a / NITS_NUMBERS_COUNT;

      uint spacerOffset = (currentNumberID / NITS_NUMBERS_COUNT) % NITS_NUMBERS_ROWS;

      uint dotOffset = ((currentNumberID % NITS_NUMBERS_COUNT) + 1u) / DOT_OFFSET_DIV;

#ifndef IS_HDR_CSP

      spacerOffset *= 2u;

      dotOffset = min(dotOffset, 1u);

#endif

      uint drawOffset = _SHOW_NITS_FROM_CURSOR ? 7u
                                               : 4u;

      vertexOffset.x = float(drawOffset + a + b + spacerOffset + dotOffset);

      vertexOffset.y = float(currentNumberID / NITS_NUMBERS_PER_ROW + 2u);
    }

    //cursor nits
    else
    [branch]
    if (_SHOW_NITS_FROM_CURSOR
     && currentNumberID >= SHOW_NITS_VALUES_NUMBER_ID_MAX
     && currentNumberID <  SHOW_NITS_FROM_CURSOR_NUMBER_ID_MAX
     && drawMaxRbgOrMaxCll)
    {
      calcOffsets = true;

      uint a = currentNumberID % NITS_NUMBERS_PER_ROW;

      uint b = a / NITS_NUMBERS_COUNT;

      uint spacerOffset = (currentNumberID / NITS_NUMBERS_COUNT) % NITS_NUMBERS_ROWS;

      uint dotOffset = ((currentNumberID % NITS_NUMBERS_COUNT) + 1u) / DOT_OFFSET_DIV;

#ifndef IS_HDR_CSP

      spacerOffset *= 2u;

      dotOffset = min(dotOffset, 1u);

#endif

      uint drawOffset = _SHOW_NITS_FROM_CURSOR ? 7u
                                               : 4u;

      vertexOffset.x = float(drawOffset + a + b + spacerOffset + dotOffset);

      vertexOffset.y = _SHOW_NITS_VALUES ? 5.f
                                         : 2.f;
    }

#ifdef IS_HDR_CSP
    else
    [branch]
    if (SHOW_GAMUTS
     && currentNumberID >= SHOW_NITS_FROM_CURSOR_NUMBER_ID_MAX
     && currentNumberID <  SHOW_GAMUTS_NUMBER_ID_MAX)
    //gamut percentages
    {
      calcOffsets = true;

      uint localNumberID = currentNumberID
                         - (NITS_NUMBERS_PER_ROW * 3u
                          + NITS_NUMBERS_PER_ROW * 1u);

      uint a = localNumberID % 6u;

      vertexOffset.x = float(a + 9u);

      vertexOffset.x += float(a / 3u);

      vertexOffset.y = float(localNumberID / 6u);

      vertexOffset.y += ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                      : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                      : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                        : 1.f;
    }
#endif //IS_HDR_CSP
    else
    {
      calcOffsets = false;
    }

    [branch]
    if (calcOffsets)
    {
      vertexOffset *= CharSize;

      float2 texCoordOffset = float2(CHAR_DIM_FLOAT.x * curNumber, 0.f);

      const float2   vertexOffset2 =   vertexOffset + CharSize;
      const float2 texCoordOffset2 = texCoordOffset + CHAR_DIM_FLOAT;

      [flatten]
      if (currentVertexID == 1u)
      {
          vertexOffset.y =   vertexOffset2.y;
        texCoordOffset.y = texCoordOffset2.y;
      }
      else
      [flatten]
      if (currentVertexID == 4u)
      {
          vertexOffset.x =   vertexOffset2.x;
        texCoordOffset.x = texCoordOffset2.x;
      }
      else
      [flatten]
      if (currentVertexID == 2u
       || currentVertexID == 5u)
      {
          vertexOffset =   vertexOffset2;
        texCoordOffset = texCoordOffset2;
      }

      FLATTEN()
      if (_TEXT_POSITION != 0u)
      {
        float maxChars;

#ifdef IS_HDR_CSP

        [flatten]
        if (SHOW_GAMUTS
         && currentNumberID >= SHOW_NITS_FROM_CURSOR_NUMBER_ID_MAX
         && currentNumberID <  SHOW_GAMUTS_NUMBER_ID_MAX)
        {
          maxChars = GetMaxCharsForGamut();
        }
        else
        {
          maxChars = GetMaxCharsForNitsRgbCll();
        }
#else
        maxChars = GetMaxCharsForNitsRgbCll();
#endif
        vertexOffset.x += BUFFER_WIDTH_FLOAT - (maxChars * CharSize.x);
      }

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_FLOAT);
      ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);
    }
    else
    {
      ret = ReturnOffScreen();
    }
  }
  else
  {
    ret = ReturnOffScreen();
  }

  return ret;
}

void VS_RenderNumbers
(
  in                  uint   VertexID         : SV_VertexID,
  out                 float4 Position         : SV_Position,
  out                 float2 TexCoord         : TEXCOORD0,
  out nointerpolation float  ScreenPixelRange : ScreenPixelRange
)
{
  static const float2 charSize = CHAR_DIM_FLOAT * _TEXT_SIZE;

  VertexCoordsAndTexCoords vertexCoordsAndTexCoords;

#ifdef IS_HDR_CSP
  [branch]
  if (VertexID < (NUMBERS_COUNT - 1u) * 6u)
  {
#endif
    vertexCoordsAndTexCoords = GetVertexCoordsAndTexCoordsForNumbers(VertexID, charSize);
#ifdef IS_HDR_CSP
  }
  //cursor gamut
  else
  BRANCH()
  if (SHOW_GAMUT_FROM_CURSOR)
  {
    const int2 mousePosition = clamp(MOUSE_POSITION, 0, BUFFER_SIZE_MINUS_1_INT);
    const float gamut = floor(tex2Dfetch(SamplerGamuts, mousePosition) * 256.f); // *256 for safety

    const uint currentVertexID = VertexID % 6u;

    float2 vertexOffset;

    vertexOffset.x = 10.f;

    vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                   : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                   : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                     : 1.f;

    FLATTEN()
    if (SHOW_GAMUTS)
    {
      vertexOffset.y += GAMUT_PERCENTAGES_LINES;
    }
    else
    {
      vertexOffset.x -= 1.f;
    }

    vertexOffset *= charSize;

    float2 texCoordOffset = float2(0.f, TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_BT709.y + gamut);

    texCoordOffset.y *= CHAR_DIM_FLOAT.y;

    const float2 x2 = float2(charSize.x, CHAR_DIM_FLOAT.x)
                    * 7.f
                    + float2(vertexOffset.x, texCoordOffset.x);

    const float2 y2 = float2(charSize.y,     CHAR_DIM_FLOAT.y)
                    + float2(vertexOffset.y, texCoordOffset.y);

    const float2   vertexOffset2 = float2(x2[0], y2[0]);
    const float2 texCoordOffset2 = float2(x2[1], y2[1]);

    [flatten]
    if (currentVertexID == 1u)
    {
        vertexOffset.y =   vertexOffset2.y;
      texCoordOffset.y = texCoordOffset2.y;
    }
    else
    [flatten]
    if (currentVertexID == 4u)
    {
        vertexOffset.x =   vertexOffset2.x;
      texCoordOffset.x = texCoordOffset2.x;
    }
    else
    [flatten]
    if (currentVertexID == 2u
     || currentVertexID == 5u)
    {
        vertexOffset =   vertexOffset2;
      texCoordOffset = texCoordOffset2;
    }

    FLATTEN()
    if (_TEXT_POSITION != 0u)
    {
      vertexOffset.x += (BUFFER_WIDTH_FLOAT - (GetMaxCharsForGamut() * charSize.x));
    }

    vertexCoordsAndTexCoords.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_FLOAT);
    vertexCoordsAndTexCoords.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);
  }
  else
  {
    vertexCoordsAndTexCoords = ReturnOffScreen();
  }
#endif

  Position = float4(vertexCoordsAndTexCoords.vertexCoords, 0.f, 1.f);

  TexCoord = vertexCoordsAndTexCoords.texCoords;

  ScreenPixelRange = Msdf::GetScreenPixelRange(_TEXT_SIZE);

  return;
}


void PS_RenderNumbers
(
  in                  float4 Position         : SV_Position,
  in                  float2 TexCoord         : TEXCOORD0,
  in  nointerpolation float  ScreenPixelRange : ScreenPixelRange,
  out                 float4 Output           : SV_Target0
)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  Output.a = inputColour.a;

  float4 mtsdf = tex2D(SamplerFontAtlasConsolidated, TexCoord);

  const float2 opacities = Msdf::GetTextOpacities(mtsdf, ScreenPixelRange);

  const float innerOpacity = opacities[0];

  const float outerOpacity = opacities[1];

  [branch]
  if (_TEXT_BG_ALPHA > 0.f
   || innerOpacity   > 0.f
   || outerOpacity   > 0.f)
  {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    Output.rgb = inputColour.rgb / 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    Output.rgb = FetchFromHdr10ToLinearLUT(inputColour.rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    Output.rgb = DECODE_SDR(inputColour.rgb);

#endif

    float textBrightness;

#ifdef IS_HDR_CSP

    textBrightness = (_TEXT_BRIGHTNESS / 10000.f);

#else

    textBrightness = (_TEXT_BRIGHTNESS / 100.f);

#endif

    Output = lerp(Output, 0.f, outerOpacity);
    Output = lerp(Output, textBrightness, innerOpacity);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    Output.rgb *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    Output.rgb = Csp::Trc::LinearTo::Pq(Output.rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    Output.rgb = ENCODE_SDR(Output.rgb);

#endif
  }
  else
  {
    Output.rgb = inputColour.rgb;
  }
}
