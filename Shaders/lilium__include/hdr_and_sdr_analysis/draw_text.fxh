#pragma once


// outer spacing is half the size of a character rounded up
uint GetOuterSpacing(const uint CharXDimension)
{
  float charXDimAsFloat = float(CharXDimension);

  return uint(charXDimAsFloat / 2.f + 0.5f);
}


float2 GetTexCoordsFromRegularCoords(const float2 TexCoordOffset)
{
  return TexCoordOffset / float2(FONT_TEXTURE_WIDTH - 1, FONT_TEXTURE_HEIGHT - 1);
}

float2 GetPositonCoordsFromRegularCoords(
  const float2 RegularCoords,
  const float2 TextureSize)
{
  float2 positionCoords = RegularCoords / TextureSize * 2;

  return float2(positionCoords.x - 1.f,
                1.f              - positionCoords.y);
}


//extract all digits without causing float issues
uint _6th(precise const float Float)
{
  return uint(Float) / 100000;
}

uint _5th(precise const float Float)
{
  return uint(Float) /  10000;
}

uint _4th(precise const float Float)
{
  return uint(Float) /   1000;
}

uint _3rd(precise const float Float)
{
  return uint(Float) /    100;
}

uint _2nd(precise const float Float)
{
  return uint(Float) /     10;
}

uint _1st(precise const float Float)
{
  return uint(Float) %     10;
}


uint d1st(precise const float Float)
{
  return uint(Float % 1.f *       10.f) % 10;
}

uint d2nd(precise const float Float)
{
  return uint(Float % 1.f *      100.f) % 10;
}

uint d3rd(precise const float Float)
{
  if (Float < 10000.f)
  {
    return uint(Float % 1.f *     1000.f) % 10;
  }
  return 10;
}

uint _d3rd(precise const float Float)
{
  return uint(Float % 1.f *     1000.f) % 10;
}

uint d4th(precise const float Float)
{
  if (Float < 1000.f)
  {
    return uint(Float % 1.f *    10000.f) % 10;
  }
  return 10;
}

uint d5th(precise const float Float)
{
  if (Float < 100.f)
  {
    return uint(Float % 1.f *   100000.f) % 10;
  }
  return 10;
}

uint d6th(precise const float Float)
{
  if (Float < 10.f)
  {
    return uint(Float % 1.f *  1000000.f) % 10;
  }
  return 10;
}

uint d7th(precise const float Float)
{
  if (Float < 1.f)
  {
    return uint(Float % 1.f * 10000000.f) % 10;
  }
  return 10;
}


uint GetNumberAboveZero(precise uint CurNumber)
{
  if (CurNumber > 0)
  {
    return CurNumber % 10;
  }
  else
  {
    return 10;
  }
}

#ifdef IS_HDR_CSP
  #define MAX_NUMBERS_NITS 11
#else
  #define MAX_NUMBERS_NITS  9
#endif

#ifdef IS_COMPUTE_CAPABLE_API

groupshared float GroupNits;
void CS_GetNumbersNits(uint3 GID  : SV_GroupID,
                       uint3 GTID : SV_GroupThreadID,
                       uint3 DTID : SV_DispatchThreadID)
{
  static const int storeYPos = GID.y + 16;

  if (GID.y ==  0)
  {
    GroupNits = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_MAX_NITS);
  }
  else if (GID.y ==  1)
  {
    GroupNits = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_AVG_NITS);
  }
  else if (GID.y ==  2)
  {
    GroupNits = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_MIN_NITS);
  }
  else //if (GID.y ==  3)
  {
    GroupNits = tex2Dfetch(SamplerNitsValues, MOUSE_POSITION);
  }

  barrier();

#ifdef IS_HDR_CSP
  switch(DTID.x)
#else
  switch(DTID.x + 2)
#endif
  {
    case 0:
    {
      precise const uint _00 = GetNumberAboveZero(_5th(GroupNits));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _00);
    }
    break;
    case 1:
    {
      precise const uint _01 = GetNumberAboveZero(_4th(GroupNits));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _01);
    }
    break;
    case 2:
    {
      precise const uint _02 = GetNumberAboveZero(_3rd(GroupNits));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _02);
    }
    break;
    case 3:
    {
      precise const uint _03 = GetNumberAboveZero(_2nd(GroupNits));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _03);
    }
    break;
    case 4:
    {
      precise const uint _04 = _1st(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _04);
    }
    break;
    case 5:
    {
      precise const uint _05 = d1st(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _05);
    }
    break;
    case 6:
    {
      precise const uint _06 = d2nd(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _06);
    }
    break;
    case 7:
    {
      precise const uint _07 = d3rd(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _07);
    }
    break;
    case 8:
    {
      precise const uint _08 = d4th(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _08);
    }
    break;
    case 9:
    {
      precise const uint _09 = d5th(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _09);
    }
    break;
    default:
    {
      precise const uint _10 = d6th(GroupNits);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, int2(DTID.x, storeYPos), _10);
    }
    break;
  }

}

#ifdef IS_HDR_CSP
groupshared float GroupCsp;
void CS_GetNumbersCsps(uint3 GID  : SV_GroupID,
                       uint3 GTID : SV_GroupThreadID,
                       uint3 DTID : SV_DispatchThreadID)
{
  static const int2 storePosOffset = int2(11, GID.y + 16);

  if (GID.y == 0)
  {
    GroupCsp = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_BT709);
  }
  else if (GID.y == 1)
  {
    GroupCsp = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3);
  }
#ifdef IS_HDR10_LIKE_CSP
  else
#else
  else if (GID.y == 2)
#endif
  {
    GroupCsp = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_BT2020);
  }
#ifdef IS_FLOAT_HDR_CSP
  else if (GID.y == 3)
  {
    GroupCsp = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_AP0);
  }
  else // if (GID.y == 4)
  {
    GroupCsp = tex1Dfetch(SamplerConsolidated, COORDS_SHOW_PERCENTAGE_INVALID);
  }
#endif

  barrier();

  switch(DTID.x)
  {
    case 0:
    {
      precise const uint _00 = GetNumberAboveZero(_3rd(GroupCsp));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _00);
    }
    break;
    case 1:
    {
      precise const uint _01 = GetNumberAboveZero(_2nd(GroupCsp));
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _01);
    }
    break;
    case 2:
    {
      precise const uint _02 = _1st(GroupCsp);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _02);
    }
    break;
    case 3:
    {
      precise const uint _03 = d1st(GroupCsp);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _03);
    }
    break;
    case 4:
    {
      precise const uint _04 = d2nd(GroupCsp);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _04);
    }
    break;
    default:
    {
      precise const uint _05 = _d3rd(GroupCsp);
      tex2Dstore(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, storePosOffset + int2(DTID.x, 0), _05);
    }
    break;
  }
}
#endif //IS_HDR_CSP

#else //IS_COMPUTE_CAPABLE_API

static const float2 ShowNumbersTextureSize =
 float2(TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_WIDTH,
        TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_HEIGHT);

void VS_PrepareGetNumbersNits(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position)
{
  static const float2 positions[3] =
  {
    GetPositonCoordsFromRegularCoords(float2( NITS_WIDTH,  NITS_HEIGHT), ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2(-NITS_WIDTH,  NITS_HEIGHT), ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2( NITS_WIDTH, -NITS_HEIGHT), ShowNumbersTextureSize)
  };

  Position = float4(positions[VertexID], 0.f, 1.f);

  return;
}

void PS_GetNumbersNits(
  in  float4 Position : SV_Position,
  out float  Number   : SV_Target0)
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

  Number = float(number) / 255.f;
}


#ifdef IS_HDR_CSP
void VS_PrepareGetNumbersCsps(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position)
{
  static const float2 positions[3] =
  {
    GetPositonCoordsFromRegularCoords(float2( CSPS_WIDTH, CSPS_Y_OFFSET),                      ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2(-CSPS_WIDTH, CSPS_Y_OFFSET),                      ShowNumbersTextureSize),
    GetPositonCoordsFromRegularCoords(float2( CSPS_WIDTH, (CSPS_Y_OFFSET + CSPS_NUMBERS * 2)), ShowNumbersTextureSize)
  };

  Position = float4(positions[VertexID], 0.f, 1.f);

  return;
}


void PS_GetNumbersCsps(
  in  float4 Position : SV_Position,
  out float  Number   : SV_Target0)
{
  const int2 positionAsInt2 = int2(Position.xy);

  precise const float cspCount = tex2Dfetch(SamplerTransfer,
                                            int2(COORDS_SHOW_PERCENTAGE_BT709 - 4 + positionAsInt2.y, 0)).x;

  precise uint number;

  switch(positionAsInt2.x)
  {
    case 0:
    {
      number = GetNumberAboveZero(_3rd(cspCount));
    }
    break;
    case 1:
    {
      number = GetNumberAboveZero(_2nd(cspCount));
    }
    break;
    case 2:
    {
      number = _1st(cspCount);
    }
    break;
    case 3:
    {
      number = d1st(cspCount);
    }
    break;
    case 4:
    {
      number = d2nd(cspCount);
    }
    break;
    default:
    {
      number = _d3rd(cspCount);
    }
    break;
  }

  Number = float(number) / 255.f;
}
#endif //IS_HDR_CSP

#endif //IS_COMPUTE_CAPABLE_API


float3 MergeText(
  float3 Output,
  float4 Mtsdf,
  float  ScreenPixelRange)
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
  #define MAX_LINES 11
#elif defined(IS_HDR10_LIKE_CSP)
  #define MAX_LINES  9
#else
  #define MAX_LINES  5
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

  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    maxChars = max(TEXT_OFFSET_ANALYIS_HEADER.x, TEXT_OFFSET_NITS_CURSOR.x + NITS_EXTRA_CHARS);
  }

#ifdef IS_HDR_CSP
  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
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

  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR)
  {
    ret.maxChars = max(ret.maxChars, uint(TEXT_OFFSET_NITS_CURSOR.x) + NITS_EXTRA_CHARS);
  }

  if (!_SHOW_NITS_VALUES)
  {
    ret.maxLines -= 3;
  }

  if (!_SHOW_NITS_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

#ifdef IS_HDR_CSP

  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR)
  {
    ret.maxChars = max(ret.maxChars, uint(TEXT_OFFSET_GAMUT_PERCENTAGES.x) + uint(TEXT_BLOCK_DRAW_X_OFFSET[3]));
  }

  if (!SHOW_CSPS)
  {
    ret.maxLines -= GAMUT_PERCENTAGES_LINES;
  }

  if (!SHOW_CSP_FROM_CURSOR)
  {
    ret.maxLines -= 1;
  }

#endif

  return ret;
}

VertexCoordsAndTexCoords GetVertexCoordsAndTexCoordsForTextBlocks(
  const uint   VertexID,
  const float2 CharSize)
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

      [branch]
      if (VertexID == 1)
      {
        pos.x = -pos.x;
      }
      else if (VertexID == 2)
      {
        pos.y = -pos.y;
      }

      if (_TEXT_POSITION != 0)
      {
        pos.x = BUFFER_WIDTH_MINUS_1_FLOAT - pos.x;
      }

      VertexCoordsAndTexCoords ret;

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(pos, BUFFER_SIZE_MINUS_1_FLOAT);
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

      //Analysis Header
      BRANCH(x)
      if (currentTextBlockID == 0)
      {
        vertexOffset.y = 0.f;
      }
      //maxNits:
      //avgNits:
      //minNits:
      else if (currentTextBlockID == 1)
      {
        BRANCH(x)
        if (_SHOW_NITS_VALUES)
        {
          vertexOffset.y = 1.f;
        }
        else
        {
          return ReturnOffScreen();
        }
      }
      //cursorNits:
#ifdef IS_HDR_CSP
      else if (currentTextBlockID == 2)
#else
      else
#endif
      {
        BRANCH(x)
        if (_SHOW_NITS_FROM_CURSOR)
        {
          vertexOffset.y = _SHOW_NITS_VALUES ? 4.f
                                             : 1.f;
        }
        else
        {
          return ReturnOffScreen();
        }
      }
#ifdef IS_HDR_CSP
      //gamut percentages
      else if (currentTextBlockID == 3)
      {
        BRANCH(x)
        if (SHOW_CSPS)
        {
          vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 5.f
                         : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 2.f
                         : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 4.f
                                                                           : 1.f;
        }
        else
        {
          return ReturnOffScreen();
        }
      }
      //cursor gamut
      else //if (currentTextBlockID == 4)
      {
        BRANCH(x)
        if (SHOW_CSP_FROM_CURSOR)
        {
          vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 5.f
                         : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 2.f
                         : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 4.f
                                                                           : 1.f;

          if (SHOW_CSPS)
          {
            vertexOffset.y += GAMUT_PERCENTAGES_LINES;
          }
        }
        else
        {
          return ReturnOffScreen();
        }
      }
#endif

      vertexOffset.x = TEXT_BLOCK_DRAW_X_OFFSET[currentTextBlockID];

      float2 size = TEXT_BLOCK_SIZES[currentTextBlockID];

      vertexOffset *= CharSize;

      texCoordOffset.x = 0.f;
      texCoordOffset.y = CHAR_DIM_FLOAT.y * TEXT_BLOCK_FETCH_Y_OFFSET[currentTextBlockID];

      BRANCH(x)
      if (localVertexID == 1)
      {
        vertexOffset.y += size.y * CharSize.y;

        texCoordOffset.y += size.y * CHAR_DIM_FLOAT.y;
      }
      else if (localVertexID == 4)
      {
        vertexOffset.x += size.x * CharSize.x;

        texCoordOffset.x += size.x * CHAR_DIM_FLOAT.x;
      }
      else if (localVertexID == 2 || localVertexID == 5)
      {
        vertexOffset += size * CharSize;

        texCoordOffset += size * CHAR_DIM_FLOAT;
      }

      vertexOffset   -= 1.f;
      texCoordOffset -= 1.f;

      vertexOffset   = max(vertexOffset,   0.f);
      texCoordOffset = max(texCoordOffset, 0.f);

      if (_TEXT_POSITION != 0)
      {
        vertexOffset.x += (BUFFER_WIDTH_MINUS_1_FLOAT - (GetMaxChars() * CharSize.x - 1.f));
      }

      VertexCoordsAndTexCoords ret;

      ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_MINUS_1_FLOAT);
      ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);

      return ret;
    }
  }
}


void VS_RenderText(
  in                  uint   VertexID         : SV_VertexID,
  out                 float4 Position         : SV_Position,
  out                 float2 TexCoord         : TEXCOORD0,
  out nointerpolation float  ScreenPixelRange : ScreenPixelRange)
{
  static const float2 charSize = CHAR_DIM_FLOAT * _TEXT_SIZE;

  const VertexCoordsAndTexCoords vertexCoordsAndTexCoords = GetVertexCoordsAndTexCoordsForTextBlocks(VertexID, charSize);

  Position = float4(vertexCoordsAndTexCoords.vertexCoords, 0.f, 1.f);

  TexCoord = vertexCoordsAndTexCoords.texCoords;

  ScreenPixelRange = GetScreenPixelRange(_TEXT_SIZE);

  return;
}

void PS_RenderText(
  in                  float4 Position         : SV_Position,
  in                  float2 TexCoord         : TEXCOORD0,
  in  nointerpolation float  ScreenPixelRange : ScreenPixelRange,
  out                 float4 Output           : SV_Target0)
{
  float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  Output.a = inputColour.a;

  float4 mtsdf = tex2D(SamplerFontAtlasConsolidated, TexCoord);

  Output.rgb = MergeText(inputColour.rgb,
                         mtsdf,
                         ScreenPixelRange);
}

VertexCoordsAndTexCoords GetVertexCoordsAndTexCoordsForNumbers(
  const uint   VertexID,
  const float2 CharSize)
{
  uint currentNumberID = VertexID / 6;

  int2 fetchPos;

#ifdef IS_HDR_CSP
  [branch]
  if (currentNumberID < NITS_NUMBERS)
  {
    fetchPos = int2(currentNumberID % 11, currentNumberID / 11);
  }
  else
  {
    uint currentCspNumberID = currentNumberID - NITS_NUMBERS;

    fetchPos = int2(currentCspNumberID % 6, currentCspNumberID / 6);

#ifdef IS_COMPUTE_CAPABLE_API
    fetchPos.x += 11;
#else
    fetchPos.y += 4;
#endif
  }
#else
  fetchPos = int2(currentNumberID % 9, currentNumberID / 9);
#endif

#ifdef IS_COMPUTE_CAPABLE_API
  fetchPos.y += 16;
#endif

  static const uint curNumber = tex2Dfetch(SamplerMaxAvgMinNitsAndCspCounterAndShowNumbers, fetchPos)
#ifndef IS_COMPUTE_CAPABLE_API
                              * 255.f
#endif
                                     ;

  BRANCH(x)
  if (curNumber < 10)
  {
    const uint currentVertexID = VertexID % 6;

    float2 vertexOffset;

    //max/avg/min nits
    BRANCH(x)
    if (currentNumberID < (MAX_NUMBERS_NITS * 3))
    {
      if (_SHOW_NITS_VALUES)
      {

        vertexOffset.x = currentNumberID % MAX_NUMBERS_NITS + 12;

        vertexOffset.y = currentNumberID / MAX_NUMBERS_NITS +  1;

#ifdef IS_HDR_CSP
        if (vertexOffset.x > 16)
#else
        if (vertexOffset.x > 14)
#endif
        {
          vertexOffset.x++;
        }
      }
      else
      {
        return ReturnOffScreen();
      }
    }
    //cursor nits
    else if (currentNumberID < (MAX_NUMBERS_NITS * 3 + MAX_NUMBERS_NITS * 1))
    {
      if (_SHOW_NITS_FROM_CURSOR)
      {
        currentNumberID -= (MAX_NUMBERS_NITS * 3);

        vertexOffset.x = currentNumberID % MAX_NUMBERS_NITS + 12;

        vertexOffset.y = _SHOW_NITS_VALUES ? 4
                                           : 1;

#ifdef IS_HDR_CSP
        if (vertexOffset.x > 16)
#else
        if (vertexOffset.x > 14)
#endif
        {
          vertexOffset.x++;
        }
      }
      else
      {
        return ReturnOffScreen();
      }
    }

#ifdef IS_HDR_CSP

#ifdef IS_FLOAT_HDR_CSP
    else //if (currentNumberID < (MAX_NUMBERS_NITS * 3 + MAX_NUMBERS_NITS * 1 + 6 * 5))
#else
    else //if (currentNumberID < (MAX_NUMBERS_NITS * 3 + MAX_NUMBERS_NITS * 1 + 6 * 3))
#endif
    //gamut percentages
    {
      if (SHOW_CSPS)
      {
        currentNumberID -= (MAX_NUMBERS_NITS * 3 + MAX_NUMBERS_NITS * 1);

        vertexOffset.x = currentNumberID % 6 + 14;

        vertexOffset.y = currentNumberID / 6;

        vertexOffset.y += ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 5.f
                        : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 2.f
                        : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 4.f
                                                                          : 1.f;

        if (vertexOffset.x > 16)
        {
          vertexOffset.x++;
        }
      }
      else
      {
        return ReturnOffScreen();
      }
    }
#endif //IS_HDR_CSP

    vertexOffset *= CharSize;

    float2 texCoordOffset = float2(CHAR_DIM_FLOAT.x * curNumber, 0.f);

    if (currentVertexID == 1)
    {
      vertexOffset.y += CharSize.y;

      texCoordOffset.y += CHAR_DIM_FLOAT.y;
    }
    else if (currentVertexID == 4)
    {
      vertexOffset.x += CharSize.x;

      texCoordOffset.x += CHAR_DIM_FLOAT.x;
    }
    else if (currentVertexID == 2 || currentVertexID == 5)
    {
      vertexOffset += CharSize;

      texCoordOffset += CHAR_DIM_FLOAT;
    }

    vertexOffset   -= 1.f;
    texCoordOffset -= 1.f;

    vertexOffset   = max(vertexOffset,   0.f);
    texCoordOffset = max(texCoordOffset, 0.f);

    if (_TEXT_POSITION != 0)
    {
      vertexOffset.x += (BUFFER_WIDTH_MINUS_1_FLOAT - (GetMaxChars() * CharSize.x - 1.f));
    }

    VertexCoordsAndTexCoords ret;

    ret.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_MINUS_1_FLOAT);
    ret.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);

    return ret;
  }
  else
  {
    return ReturnOffScreen();
  }
}

#ifdef IS_FLOAT_HDR_CSP
  #define NUMBERS_COUNT (4 * 11 + 5 * 6 + 1)
#elif defined(IS_HDR10_LIKE_CSP)
  #define NUMBERS_COUNT (4 * 11 + 3 * 6 + 1)
#else
  #define NUMBERS_COUNT (4 * 9)
#endif

void VS_RenderNumbers(
  in                  uint   VertexID         : SV_VertexID,
  out                 float4 Position         : SV_Position,
  out                 float2 TexCoord         : TEXCOORD0,
  out nointerpolation float  ScreenPixelRange : ScreenPixelRange)
{
  static const float2 charSize = CHAR_DIM_FLOAT * _TEXT_SIZE;

  VertexCoordsAndTexCoords vertexCoordsAndTexCoords;

#ifdef IS_HDR_CSP
  BRANCH(x)
  if (VertexID < (NUMBERS_COUNT - 1) * 6)
  {
#endif
    vertexCoordsAndTexCoords = GetVertexCoordsAndTexCoordsForNumbers(VertexID, charSize);
#ifdef IS_HDR_CSP
  }
  //cursor gamut
  else
  {
    BRANCH(x)
    if (SHOW_CSP_FROM_CURSOR)
    {
      const uint csp = tex2Dfetch(SamplerCsps, MOUSE_POSITION) * 255.f;

      const uint currentVertexID = VertexID % 6;

      float2 vertexOffset;

      vertexOffset.x = 15;

      vertexOffset.y = ( _SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 5.f
                     : (!_SHOW_NITS_VALUES &&  _SHOW_NITS_FROM_CURSOR) ? 2.f
                     : ( _SHOW_NITS_VALUES && !_SHOW_NITS_FROM_CURSOR) ? 4.f
                                                                       : 1.f;

      if (SHOW_CSPS)
      {
        vertexOffset.y += GAMUT_PERCENTAGES_LINES;
      }

      vertexOffset *= charSize;

      float2 texCoordOffset = float2(0.f, TEXT_OFFSET_GAMUT_CURSOR_BT709.y + float(csp));

      texCoordOffset.y *= CHAR_DIM_FLOAT.y;

      if (currentVertexID == 1)
      {
        vertexOffset.y += charSize.y;

        texCoordOffset.y += CHAR_DIM_FLOAT.y;
      }
      else if (currentVertexID == 4)
      {
        vertexOffset.x += charSize.x * 7.f;

        texCoordOffset.x += CHAR_DIM_FLOAT.x * 7.f;
      }
      else if (currentVertexID == 2 || currentVertexID == 5)
      {
        vertexOffset += float2(charSize.x * 7.f, charSize.y);

        texCoordOffset += float2(CHAR_DIM_FLOAT.x * 7.f, CHAR_DIM_FLOAT.y);
      }

      vertexOffset   -= 1.f;
      texCoordOffset -= 1.f;

      vertexOffset   = max(vertexOffset,   0.f);
      texCoordOffset = max(texCoordOffset, 0.f);

      if (_TEXT_POSITION != 0)
      {
        vertexOffset.x += (BUFFER_WIDTH_MINUS_1_FLOAT - (GetMaxChars() * charSize.x - 1.f));
      }

      vertexCoordsAndTexCoords.vertexCoords = GetPositonCoordsFromRegularCoords(vertexOffset, BUFFER_SIZE_MINUS_1_FLOAT);
      vertexCoordsAndTexCoords.texCoords    = GetTexCoordsFromRegularCoords(texCoordOffset);
    }
    else
    {
      vertexCoordsAndTexCoords = ReturnOffScreen();
    }
  }
#endif

  Position = float4(vertexCoordsAndTexCoords.vertexCoords, 0.f, 1.f);

  TexCoord = vertexCoordsAndTexCoords.texCoords;

  ScreenPixelRange = GetScreenPixelRange(_TEXT_SIZE);

  return;
}


void PS_RenderNumbers(
  in                  float4 Position         : SV_Position,
  in                  float2 TexCoord         : TEXCOORD0,
  in  nointerpolation float  ScreenPixelRange : ScreenPixelRange,
  out                 float4 Output           : SV_Target0)
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
