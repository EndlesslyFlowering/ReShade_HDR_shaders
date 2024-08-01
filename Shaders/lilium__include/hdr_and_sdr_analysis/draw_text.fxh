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

void CS_GetNitNumbers
(
  uint3 GID  : SV_GroupID,
  uint3 GTID : SV_GroupThreadID,
  uint3 DTID : SV_DispatchThreadID
)
{
  static const int2 storePos = int2(DTID.xy);

  float nits;

  [branch]
  if (GID.y < 3
   && _SHOW_NITS_VALUES)
  {
    nits = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_MAX_NITS + (3 * GID.x) + GID.y);
  }
  else
  [branch]
  if (_SHOW_NITS_FROM_CURSOR)
  {
    const int2 mousePosition = clamp(MOUSE_POSITION, 0, BUFFER_SIZE_MINUS_1_INT);
    nits = tex2Dfetch(SamplerNitsValues, mousePosition)[(GID.x + 3) % 4];
  }
  else
  {
    nits = 0.f;
  }

#ifdef IS_FLOAT_HDR_CSP
  static const bool isMinus0 = asuint(nits) == uint(0x80000000);

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
        _00 = _minus;
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
        _01 = _minus;
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
        _02 = _minus;
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
        _03 = _minus;
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
    nitsValue = tex2Dfetch(SamplerNitsValues, MOUSE_POSITION);
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
  const float sd = GetMedian(Mtsdf.rgb);

  const float screenPixelDistance = ScreenPixelRange * (sd - 0.5f);

  const float opacity = saturate(screenPixelDistance + 0.5f);

  const float outline = smoothstep(0.f, 0.1f, (opacity + Mtsdf.a) / 2.f);

  // tone map pixels below the overlay area
  //
  // first set 1.0 to be equal to _TEXT_BRIGHTNESS
  float adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  adjustFactor = _TEXT_BRIGHTNESS / 80.f;

  Output = Csp::Mat::Bt709To::Bt2020(Output / adjustFactor);

  // safety clamp colours outside of BT.2020
  Output = max(Output, 0.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  adjustFactor = _TEXT_BRIGHTNESS / 10000.f;

  Output = Csp::Trc::PqTo::Linear(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  adjustFactor = _TEXT_BRIGHTNESS / 100.f;

  Output = DECODE_SDR(Output);

#endif

#if (ACTUAL_COLOUR_SPACE != CSP_SCRGB)

  Output /= adjustFactor;

#endif

  // then tone map to 1.0 at max
  ExtendedReinhardTmo(Output, _TEXT_BRIGHTNESS);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  // safety clamp for the case that there are values that represent above 10000 nits
  Output.rgb = min(Output.rgb, 1.f);

#endif

  Output = lerp(Output, 0.f, _TEXT_BG_ALPHA / 100.f);

  // apply the text
  Output = lerp(Output, 0.f, outline);
  Output = lerp(Output, 1.f, opacity);

  // map everything back to the used colour space
  Output *= adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Output = Csp::Mat::Bt2020To::Bt709(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Output = Csp::Trc::LinearTo::Pq(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  Output = ENCODE_SDR(Output);

#endif

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

#if defined(IS_FLOAT_HDR_CSP)
  #define MAX_LINES 12
#elif defined(IS_HDR10_LIKE_CSP)
  #define MAX_LINES 10
#else
  #define MAX_LINES  6
#endif

#define MAX_CHARS TEXT_OFFSET_ANALYIS_HEADER.x

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

float GetMaxChars()
{
  float maxChars = MAX_CHARS;

  [flatten]
  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    maxChars = max(TEXT_OFFSET_ANALYIS_HEADER.x, TEXT_OFFSET_NITS_CURSOR.x + NITS_EXTRA_CHARS);
  }

#ifdef IS_HDR_CSP
  [flatten]
  if (SHOW_GAMUTS
   || SHOW_GAMUT_FROM_CURSOR)
  {
    maxChars = max(maxChars, TEXT_OFFSET_GAMUT_PERCENTAGES.x + TEXT_BLOCK_DRAW_X_OFFSET[3]);
  }
#endif

  return maxChars;
}

MaxCharsAndMaxLines GetMaxCharsAndMaxLines()
{
  MaxCharsAndMaxLines ret;

  ret.maxChars = MAX_CHARS;
  ret.maxLines = MAX_LINES;

  [flatten]
  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    ret.maxChars = max(ret.maxChars, uint(TEXT_OFFSET_NITS_CURSOR.x) + NITS_EXTRA_CHARS);
  }

  [flatten]
  if (!_SHOW_NITS_VALUES)
  {
    ret.maxLines -= 3;
  }

  [flatten]
  if (!_SHOW_NITS_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

#ifdef IS_HDR_CSP

  [flatten]
  if (SHOW_GAMUTS
   || SHOW_GAMUT_FROM_CURSOR)
  {
    ret.maxChars = max(ret.maxChars, uint(TEXT_OFFSET_GAMUT_PERCENTAGES.x) + uint(TEXT_BLOCK_DRAW_X_OFFSET[3]));
  }

  [flatten]
  if (!SHOW_GAMUTS)
  {
    ret.maxLines -= GAMUT_PERCENTAGES_LINES;
  }

  [flatten]
  if (!SHOW_GAMUT_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

#endif

  return ret;
}

VertexCoordsAndTexCoords GetVertexCoordsAndTexCoordsForTextBlocks
(
  const uint   VertexID,
  const float2 CharSize
)
{
  switch(VertexID)
  {
    case 0:
    case 1:
    case 2:
    {
      MaxCharsAndMaxLines _max = GetMaxCharsAndMaxLines();

      float2 pos = float2(_max.maxChars * CharSize.x,
                          _max.maxLines * CharSize.y);

      [flatten]
      if (VertexID == 1)
      {
        pos.x = -pos.x;
      }
      else
      [flatten]
      if (VertexID == 2)
      {
        pos.y = -pos.y;
      }

      [flatten]
      if (_TEXT_POSITION != 0)
      {
        pos.x = BUFFER_WIDTH_FLOAT - pos.x;
      }

      VertexCoordsAndTexCoords ret;

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(pos, BUFFER_SIZE_FLOAT);
//      ret.vertexCoords = float2(-2.f, -2.f);
      ret.texCoords    = float2(-1.f, -1.f);

      return ret;
    }
    default:
    {
      const uint textBlockVertexID = VertexID - 3;

      const uint localVertexID = textBlockVertexID % 6;

      uint currentTextBlockID = textBlockVertexID / 6;

      float2 vertexOffset;

      float2 texCoordOffset;

      static const bool specificallyNeedNitsRgbHeader = currentTextBlockID == 1
                                                     && !_SHOW_NITS_VALUES
                                                     && _SHOW_NITS_FROM_CURSOR;

      //Analysis Header
      if (currentTextBlockID == 0)
      {
        vertexOffset.y = 0.f;
      }
      //max:
      //avg:
      //min:
      else if (currentTextBlockID == 1
            && (_SHOW_NITS_VALUES
             || specificallyNeedNitsRgbHeader))
      {
        vertexOffset.y = 1.f;
      }
      //cursor:
      else if (currentTextBlockID == 2
            && _SHOW_NITS_FROM_CURSOR)
      {
        vertexOffset.y = _SHOW_NITS_VALUES ? 5.f
                                           : 2.f;
      }
#ifdef IS_HDR_CSP
      //gamut percentages
      //cursor gamut
      else
      {
        const bool isShowGamuts          = currentTextBlockID == 3 && SHOW_GAMUTS;
        const bool isShowGamutFromCursor = currentTextBlockID == 4 && SHOW_GAMUT_FROM_CURSOR;

        if (isShowGamuts
         || isShowGamutFromCursor)
        {
          vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                         : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                         : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                           : 2.f;

          [flatten]
          if (isShowGamutFromCursor
           && SHOW_GAMUTS)
          {
            vertexOffset.y += GAMUT_PERCENTAGES_LINES;
          }
        }
#endif
        else
        {
          return ReturnOffScreen();
        }
#ifdef IS_HDR_CSP
      }
#endif

      vertexOffset.x = TEXT_BLOCK_DRAW_X_OFFSET[currentTextBlockID];

      float2 size = TEXT_BLOCK_SIZES[currentTextBlockID];

      [flatten]
      if (specificallyNeedNitsRgbHeader)
      {
        size.y -= 3.f;
      }

      vertexOffset *= CharSize;

      texCoordOffset.x = 0.f;
      texCoordOffset.y = CHAR_DIM_FLOAT.y * TEXT_BLOCK_FETCH_Y_OFFSET[currentTextBlockID];

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

      [flatten]
      if (_TEXT_POSITION != 0)
      {
        vertexOffset.x += BUFFER_WIDTH_FLOAT - (GetMaxChars() * CharSize.x);
      }

      VertexCoordsAndTexCoords ret;

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_FLOAT);
      ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);

      return ret;
    }
  }
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

  ScreenPixelRange = GetScreenPixelRange(_TEXT_SIZE, RANGE);

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
  uint currentNumberID = VertexID / 6;

  int2 fetchPos;

#ifdef IS_HDR_CSP
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

  //[branch] can't branch here iirc?
  if (curNumber < 11u)
  {
    const uint currentVertexID = VertexID % 6;

    float2 vertexOffset;

#ifdef IS_HDR_CSP

  #define DOT_OFFSET_DIV 6

#else

  #define DOT_OFFSET_DIV 4

#endif

    //max/avg/min nits
    [flatten]
    if (currentNumberID < (NITS_NUMBERS_PER_ROW * 3)
     && _SHOW_NITS_VALUES)
    {
      uint a = currentNumberID % NITS_NUMBERS_PER_ROW;

      uint b = a / NITS_NUMBERS_COUNT;

      uint spacerOffset = (currentNumberID / NITS_NUMBERS_COUNT) % NITS_NUMBERS_ROWS;

      uint dotOffset = ((currentNumberID % NITS_NUMBERS_COUNT) + 1) / DOT_OFFSET_DIV;

#ifndef IS_HDR_CSP

      spacerOffset *= 2;

      dotOffset = min(dotOffset, 1);

#endif

      vertexOffset.x = 7 + a + b + spacerOffset + dotOffset;

      vertexOffset.y = currentNumberID / NITS_NUMBERS_PER_ROW + 2;
    }
    //cursor nits
    else
    [flatten]
    if (currentNumberID < (NITS_NUMBERS_PER_ROW * 4)
     && _SHOW_NITS_FROM_CURSOR)
    {
      uint a = currentNumberID % NITS_NUMBERS_PER_ROW;

      uint b = a / NITS_NUMBERS_COUNT;

      uint spacerOffset = (currentNumberID / NITS_NUMBERS_COUNT) % NITS_NUMBERS_ROWS;

      uint dotOffset = ((currentNumberID % NITS_NUMBERS_COUNT) + 1) / DOT_OFFSET_DIV;

#ifndef IS_HDR_CSP

      spacerOffset *= 2;

      dotOffset = min(dotOffset, 1);

#endif

      vertexOffset.x = 7 + a + b + spacerOffset + dotOffset;

      vertexOffset.y = _SHOW_NITS_VALUES ? 5
                                         : 2;
    }

#ifdef IS_HDR_CSP
    else
    [flatten]
    if (currentNumberID < (NITS_NUMBERS_PER_ROW * 3
                         + NITS_NUMBERS_PER_ROW * 1
                         + 6 *
#ifdef IS_FLOAT_HDR_CSP
                               5
#else
                               3
#endif
                          ) && SHOW_GAMUTS)
    //gamut percentages
    {
      currentNumberID -= (NITS_NUMBERS_PER_ROW * 3
                        + NITS_NUMBERS_PER_ROW * 1);

      uint a = (currentNumberID % 6);

      vertexOffset.x = a + 9;

      vertexOffset.x += a / 3;

      vertexOffset.y = currentNumberID / 6;

      vertexOffset.y += ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                      : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                      : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                        : 2.f;
    }
#endif //IS_HDR_CSP
    else
    {
      return ReturnOffScreen();
    }

    vertexOffset *= CharSize;

    float2 texCoordOffset = float2(CHAR_DIM_FLOAT.x * curNumber, 0.f);

    const float2   vertexOffset2 =   vertexOffset + CharSize;
    const float2 texCoordOffset2 = texCoordOffset + CHAR_DIM_FLOAT;

    [flatten]
    if (currentVertexID == 1)
    {
        vertexOffset.y =   vertexOffset2.y;
      texCoordOffset.y = texCoordOffset2.y;
    }
    else
    [flatten]
    if (currentVertexID == 4)
    {
        vertexOffset.x =   vertexOffset2.x;
      texCoordOffset.x = texCoordOffset2.x;
    }
    else
    [flatten]
    if (currentVertexID == 2
     || currentVertexID == 5)
    {
        vertexOffset =   vertexOffset2;
      texCoordOffset = texCoordOffset2;
    }

    [flatten]
    if (_TEXT_POSITION != 0)
    {
      vertexOffset.x += BUFFER_WIDTH_FLOAT - (GetMaxChars() * CharSize.x);
    }

    VertexCoordsAndTexCoords ret;

    ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_FLOAT);
    ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);

    return ret;
  }
  else
  {
    return ReturnOffScreen();
  }
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
  //[branch] can't branch here iirc?
  if (VertexID < (NUMBERS_COUNT - 1) * 6)
  {
#endif
    vertexCoordsAndTexCoords = GetVertexCoordsAndTexCoordsForNumbers(VertexID, charSize);
#ifdef IS_HDR_CSP
  }
  //cursor gamut
  else
  BRANCH(x)
  if (SHOW_GAMUT_FROM_CURSOR)
  {
    const float gamut = floor(tex2Dfetch(SamplerGamuts, MOUSE_POSITION) * 256.f); // *256 for safety

    const uint currentVertexID = VertexID % 6;

    float2 vertexOffset;

    vertexOffset.x = 10;

    vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 6.f
                   : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 3.f
                   : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 5.f
                                                                     : 2.f;

    [flatten]
    if (SHOW_GAMUTS)
    {
      vertexOffset.y += GAMUT_PERCENTAGES_LINES;
    }

    vertexOffset *= charSize;

    float2 texCoordOffset = float2(0.f, TEXT_OFFSET_GAMUT_CURSOR_BT709.y + gamut);

    texCoordOffset.y *= CHAR_DIM_FLOAT.y;

    const float2 x2 = float2(charSize.x, CHAR_DIM_FLOAT.x)
                    * 7.f
                    + float2(vertexOffset.x, texCoordOffset.x);

    const float2 y2 = float2(charSize.y,     CHAR_DIM_FLOAT.y)
                    + float2(vertexOffset.y, texCoordOffset.y);

    const float2   vertexOffset2 = float2(x2[0], y2[0]);
    const float2 texCoordOffset2 = float2(x2[1], y2[1]);

    [flatten]
    if (currentVertexID == 1)
    {
        vertexOffset.y =   vertexOffset2.y;
      texCoordOffset.y = texCoordOffset2.y;
    }
    else
    [flatten]
    if (currentVertexID == 4)
    {
        vertexOffset.x =   vertexOffset2.x;
      texCoordOffset.x = texCoordOffset2.x;
    }
    else
    [flatten]
    if (currentVertexID == 2
     || currentVertexID == 5)
    {
        vertexOffset =   vertexOffset2;
      texCoordOffset = texCoordOffset2;
    }

    [flatten]
    if (_TEXT_POSITION != 0)
    {
      vertexOffset.x += (BUFFER_WIDTH_FLOAT - (GetMaxChars() * charSize.x));
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

  ScreenPixelRange = GetScreenPixelRange(_TEXT_SIZE, RANGE);

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

  const float sd = GetMedian(mtsdf.rgb);

  const float screenPixelDistance = ScreenPixelRange * (sd - 0.5f);

  const float opacity = saturate(screenPixelDistance + 0.5f);

  const float outline = smoothstep(0.f, 0.1f, (opacity + mtsdf.a) / 2.f); //higher range for better outline?

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Output.rgb = inputColour.rgb / 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Output.rgb = Csp::Trc::PqTo::Linear(inputColour.rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  Output.rgb = DECODE_SDR(inputColour.rgb);

#endif

  float textBrightness;

#ifdef IS_HDR_CSP

  textBrightness = (_TEXT_BRIGHTNESS / 10000.f);

#else

  textBrightness = (_TEXT_BRIGHTNESS / 100.f);

#endif

  Output.rgb = lerp(Output.rgb, 0.f, outline);
  Output.rgb = lerp(Output.rgb, textBrightness, opacity);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Output.rgb *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Output.rgb = Csp::Trc::LinearTo::Pq(Output.rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  Output.rgb = ENCODE_SDR(Output.rgb);

#endif
}
