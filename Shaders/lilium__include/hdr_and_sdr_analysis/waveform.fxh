#pragma once


static const float TEXTURE_WAVEFORM_BUFFER_WIDTH_FACTOR = float(TEXTURE_WAVEFORM_WIDTH - 1)
                                                        / BUFFER_WIDTH_MINUS_1_FLOAT;

static const float TEXTURE_WAVEFORM_BUFFER_FACTOR = (BUFFER_WIDTH_FLOAT  / 3840.f
                                                   + BUFFER_HEIGHT_FLOAT / 2160.f)
                                                  / 2.f;

static const uint TEXTURE_WAVEFORM_SCALE_BORDER = TEXTURE_WAVEFORM_BUFFER_FACTOR * 35.f + 0.5f;
static const uint TEXTURE_WAVEFORM_SCALE_FRAME  = TEXTURE_WAVEFORM_BUFFER_FACTOR *  7.f + 0.5f;

//static const uint TEXTURE_WAVEFORM_FONT_SIZE =
//  clamp(uint(round(TEXTURE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f)), 14, 32);

static const uint TEXTURE_WAVEFORM_SCALE_WIDTH = TEXTURE_WAVEFORM_WIDTH
                                               + (CHAR_DIM_UINT.x * 8u) //8 chars for 10000.00
                                               + uint(CHAR_DIM_FLOAT.x / 2.f + 0.5f)
                                               + (TEXTURE_WAVEFORM_SCALE_BORDER * 2)
                                               + (TEXTURE_WAVEFORM_SCALE_FRAME  * 3);

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
  #define MAX_WAVEFORM_HEIGHT_FACTOR 4
#else
  #define MAX_WAVEFORM_HEIGHT_FACTOR 1
#endif

static const uint TEXTURE_WAVEFORM_SCALE_HEIGHT = TEXTURE_WAVEFORM_USED_HEIGHT * MAX_WAVEFORM_HEIGHT_FACTOR
                                                + uint(CHAR_DIM_FLOAT.y / 2.f - TEXTURE_WAVEFORM_SCALE_FRAME + 0.5f) * 2u
                                                + (TEXTURE_WAVEFORM_SCALE_BORDER * 2)
                                                + (TEXTURE_WAVEFORM_SCALE_FRAME  * 2);

static const float TEXTURE_WAVEFORM_SCALE_FACTOR_X = (TEXTURE_WAVEFORM_SCALE_WIDTH - 1.f)
                                                   / float(TEXTURE_WAVEFORM_WIDTH  - 1);

static const float TEXTURE_WAVEFORM_SCALE_FACTOR_Y = (TEXTURE_WAVEFORM_SCALE_HEIGHT - 1.f)
                                                   / float(TEXTURE_WAVEFORM_HEIGHT  - 1);


#if (TEXTURE_WAVEFORM_WIDTH % WAVE64_THREAD_SIZE_X == 0)
  #define TEXTURE_WAVEFORM_COUNTER_DISPATCH_X (TEXTURE_WAVEFORM_WIDTH / WAVE64_THREAD_SIZE_X)
#else
  #define TEXTURE_WAVEFORM_COUNTER_DISPATCH_X (TEXTURE_WAVEFORM_WIDTH / WAVE64_THREAD_SIZE_X + 1)

  #define GET_MAX_WAVEFORM_VALUE_NEEDS_X_CHECK
#endif
#if (TEXTURE_WAVEFORM_HEIGHT % WAVE64_THREAD_SIZE_Y == 0)
  #define TEXTURE_WAVEFORM_COUNTER_DISPATCH_Y (TEXTURE_WAVEFORM_HEIGHT / WAVE64_THREAD_SIZE_Y)
#else
  #define TEXTURE_WAVEFORM_COUNTER_DISPATCH_Y (TEXTURE_WAVEFORM_HEIGHT / WAVE64_THREAD_SIZE_Y + 1)

  #define GET_MAX_WAVEFORM_VALUE_NEEDS_Y_CHECK
#endif


texture2D TextureWaveformScale
<
  pooled = true;
>
{
  Width  = TEXTURE_WAVEFORM_SCALE_WIDTH;
  Height = TEXTURE_WAVEFORM_SCALE_HEIGHT;
  Format = RG8;
};

sampler2D<float4> SamplerWaveformScale
{
  Texture = TextureWaveformScale;
};

storage2D<float4> StorageWaveformScale
{
  Texture = TextureWaveformScale;
};


texture3D TextureWaveform
<
  pooled = true;
>
{
  Width  = TEXTURE_WAVEFORM_WIDTH;
  Height = TEXTURE_WAVEFORM_HEIGHT;
  Depth  = 3;
  Format = R32U;
};

sampler3D<uint> SamplerWaveform
{
  Texture = TextureWaveform;
};

storage3D<uint> StorageWaveform
{
  Texture = TextureWaveform;
};


texture2D TextureWaveformFinal
<
  pooled = true;
>
{
  Width  = TEXTURE_WAVEFORM_SCALE_WIDTH;
  Height = TEXTURE_WAVEFORM_SCALE_HEIGHT;
  Format = RGB10A2;
};

sampler2D<float4> SamplerWaveformFinal
{
  Texture = TextureWaveformFinal;
};


namespace Waveform
{

  struct SWaveformData
  {
    int    borderSize;
    int    frameSize;
    float2 charDimensions;
#ifndef IS_HDR_CSP
    float  charDimensionXForPercent;
#endif
    int2   waveformArea;
#ifdef IS_HDR_CSP
    int    cutoffOffset;
    #define WAVEDAT_CUTOFF_OFFSET waveDat.cutoffOffset
    int    tickPoints[16];
#else
    #define WAVEDAT_CUTOFF_OFFSET 0
    int    tickPoints[14];
#endif
#ifdef IS_HDR_CSP
    int    cutoffOffsetAt100PercentScale;
#endif
    int    fontSpacer;
    int2   offsetToFrame;
    int2   textOffset;
    int    tickXOffset;
    int    lowerFrameStart;
    int2   endXY;
    int    endYminus1;
  };

  SWaveformData GetData()
  {
    SWaveformData waveDat;

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
  #define WAVEFORM_SCALE_FACTOR_CLAMP_MIN 0.5f
  #define WAVEFORM_SCALE_FACTOR_CLAMP_MAX float2(1.f, 4.f)
#else
  #define WAVEFORM_SCALE_FACTOR_CLAMP_MIN 0.5f
  #define WAVEFORM_SCALE_FACTOR_CLAMP_MAX 1.f.xx
#endif

    float2 waveformScaleFactorXY = clamp(_WAVEFORM_SIZE / DIV_100, 0.5f, 1.f);

    const float waveformScaleFactor =
      (waveformScaleFactorXY.x + waveformScaleFactorXY.y) / 2.f;

    const float borderAndFrameSizeFactor = max(waveformScaleFactor, 0.75f);

    const float fontSizeFactor = max(waveformScaleFactor, 0.85f);

    static const int maxBorderSize = TEXTURE_WAVEFORM_SCALE_BORDER;
    static const int maxFrameSize  = TEXTURE_WAVEFORM_SCALE_FRAME;

    waveDat.borderSize = clamp(int(TEXTURE_WAVEFORM_BUFFER_FACTOR * 35.f * borderAndFrameSizeFactor + 0.5f), 10, maxBorderSize);
    waveDat.frameSize  = clamp(int(TEXTURE_WAVEFORM_BUFFER_FACTOR *  7.f * borderAndFrameSizeFactor + 0.5f),  4, maxFrameSize);

    static const float minFontSize = 0.375f;

    static const float maxFontSize =
      max(((TEXTURE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f) / 2.f) * 2.f / 32.f * FONT_SIZE_MULTIPLIER, minFontSize);

    const float fontSize =
      clamp(((TEXTURE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f) / 2.f * fontSizeFactor) * 2.f / 32.f * FONT_SIZE_MULTIPLIER * _WAVEFORM_TEXT_SIZE_ADJUST,
            minFontSize,
            maxFontSize);

#ifndef IS_HDR_CSP
    waveDat.charDimensionXForPercent = CHAR_DIM_FLOAT.x * fontSize.x;
#endif

    waveDat.charDimensions = float2(CHAR_DIM_FLOAT.x - 1.f, CHAR_DIM_FLOAT.y) * fontSize;

#ifdef IS_HDR_CSP
    const int maxChars = WAVEFORM_CUTOFF_POINT == 0 ? 8
                                                    : 7;
#else
    const int maxChars = 7;
#endif

    const int textWidth  = waveDat.charDimensions.x * maxChars;
    const int tickSpacer = int(waveDat.charDimensions.x / 2.f + 0.5f);

    waveDat.fontSpacer = int(waveDat.charDimensions.y / 2.f - float(waveDat.frameSize) + 0.5f);

    waveDat.offsetToFrame = int2(waveDat.borderSize + textWidth + tickSpacer + waveDat.frameSize,
                                 waveDat.borderSize + waveDat.fontSpacer);

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
    waveformScaleFactorXY.y += (3.f - ((8.f / 9.f) - waveformScaleFactorXY.y) * 6.f);
    waveformScaleFactorXY.y *= (3.f / 3.5f);
#endif

#ifdef IS_HDR_CSP

    waveDat.tickPoints =
    {
      int(0),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(4000.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(2000.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(1000.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq( 400.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq( 203.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq( 100.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(  50.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(  25.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(  10.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(   5.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(   2.5f ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(   1.f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(   0.25f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(   0.05f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int(                                                                         float(TEXTURE_WAVEFORM_USED_HEIGHT)   * waveformScaleFactorXY.y + 0.5f)
    };

    waveDat.cutoffOffset = waveDat.tickPoints[WAVEFORM_CUTOFF_POINT];

    waveDat.tickPoints[ 1] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 2] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 3] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 4] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 5] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 6] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 7] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 8] -= waveDat.cutoffOffset;
    waveDat.tickPoints[ 9] -= waveDat.cutoffOffset;
    waveDat.tickPoints[10] -= waveDat.cutoffOffset;
    waveDat.tickPoints[11] -= waveDat.cutoffOffset;
    waveDat.tickPoints[12] -= waveDat.cutoffOffset;
    waveDat.tickPoints[13] -= waveDat.cutoffOffset;
    waveDat.tickPoints[14] -= waveDat.cutoffOffset;
    waveDat.tickPoints[15] -= waveDat.cutoffOffset;

#ifdef IS_HDR_CSP
    static const int tick_points_at_100_percent_scale[4] =
    {
      int(0),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(4000.f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(2000.f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (Csp::Trc::NitsTo::Pq(1000.f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) + 0.5f)
    };

    waveDat.cutoffOffsetAt100PercentScale = tick_points_at_100_percent_scale[WAVEFORM_CUTOFF_POINT];
#endif

    if (WAVEFORM_CUTOFF_POINT > 0u)
    {
      waveDat.tickPoints[0] = -100;

      if (WAVEFORM_CUTOFF_POINT > 1u)
      {
        waveDat.tickPoints[1] = -100;

        if (WAVEFORM_CUTOFF_POINT > 2u)
        {
          waveDat.tickPoints[2] = -100;
        }
      }
    }

#else

    waveDat.tickPoints =
    {
      int(0),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.875f ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.75f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.6f   ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.5f   ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.35f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.25f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.18f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.1f   ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.05f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.025f ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.01f  ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.0025f) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
#else
      int((float(TEXTURE_WAVEFORM_USED_HEIGHT) - (ENCODE_SDR(0.004f ) * float(TEXTURE_WAVEFORM_USED_HEIGHT))) * waveformScaleFactorXY.y + 0.5f),
#endif
      int(                                                              float(TEXTURE_WAVEFORM_USED_HEIGHT)   * waveformScaleFactorXY.y + 0.5f)
    };

#endif

    waveDat.waveformArea =
      int2(uint(TEXTURE_WAVEFORM_WIDTH * waveformScaleFactorXY.x),
#ifdef IS_HDR_CSP
           waveDat.tickPoints[15]
#else
           waveDat.tickPoints[13]
#endif
           );

    waveDat.textOffset = int2(0, int(float(waveDat.charDimensions.y) / 2.f + 0.5f));

    waveDat.tickXOffset = waveDat.borderSize
                        + textWidth
                        + tickSpacer;

    waveDat.lowerFrameStart = waveDat.frameSize
                            + waveDat.waveformArea.y;

    waveDat.endXY = waveDat.frameSize
                  * 2
                  + waveDat.waveformArea;

    waveDat.endYminus1 = waveDat.endXY.y
                       - 1;

    return waveDat;
  }

  int2 GetActiveArea()
  {
    SWaveformData waveDat = GetData();

    return waveDat.offsetToFrame
         + waveDat.frameSize
         + waveDat.waveformArea
         + waveDat.frameSize
         + int2(0, waveDat.fontSpacer)
         + waveDat.borderSize;
  }

  int2 GetNitsOffset
  (
    const int ActiveBorderSize,
    const int ActiveFrameSize,
    const int ActiveFontSpacer,
    const int YOffset
  )
  {
    return int2(ActiveBorderSize,
                ActiveBorderSize + ActiveFontSpacer + ActiveFrameSize + YOffset);
  } //GetNitsOffset


  void DrawCharToScale
  (
    const int    Unrolling_Be_Gone_Int,
    const uint2  Char,
    const float2 CharDim,
    const int2   Pos,
    const int    CharCount
  )
  {
    const float2 charFetchPos = float2(Char * CHAR_DIM_UINT);

    const int2 currentDrawPos = Pos + int2(CharCount * CharDim.x, 0);

    const float charDimFactor = CharDim.x / CHAR_DIM_FLOAT.x;

    const float screenPixelRange = Msdf::GetScreenPixelRange(charDimFactor);

    const float2 char_dim_factor = min(CHAR_DIM_FLOAT / CharDim, 2.f);

    const int2 floorCharDim = floor(CharDim);

    int2 currentOffset = int2(0, 0);

    [loop]
    while (currentOffset.x < (floorCharDim.x + Unrolling_Be_Gone_Int))
    {
      [loop]
      while (currentOffset.y < (floorCharDim.y + Unrolling_Be_Gone_Int))
      {
        float2 currentSamplePos = charFetchPos
                                + float2(currentOffset)
                                * char_dim_factor
                                + 0.5f;

        float2 fract = frac(currentSamplePos);

        int2 currentFetchPos = floor(currentSamplePos);

        float4 a = tex2Dfetch(SamplerFontAtlasConsolidated, currentFetchPos);
        float4 b = tex2Dfetch(SamplerFontAtlasConsolidated, int2(currentFetchPos.x + 1, currentFetchPos.y));
        float4 c = tex2Dfetch(SamplerFontAtlasConsolidated, int2(currentFetchPos.x,     currentFetchPos.y + 1));
        float4 d = tex2Dfetch(SamplerFontAtlasConsolidated, currentFetchPos + 1);

        float4 mtsdf = lerp(lerp(a, b, fract.x),
                            lerp(c, d, fract.x),
                            fract.y);

        const float2 opacities = Msdf::GetTextOpacities(mtsdf, screenPixelRange);

        const float innerOpacity = opacities[0];

        const float outerOpacity = opacities[1];

        float grey = lerp(0, 0.5f, innerOpacity);

        float alpha = innerOpacity + outerOpacity;

        int2 currentDrawOffset = currentDrawPos + currentOffset;

        tex2Dstore(StorageWaveformScale, currentDrawOffset, float4(sqrt(grey), alpha, 0.f, 0.f));

        currentOffset.y++;
      }
      currentOffset.x++;
      currentOffset.y = 0;
    }

    return;
  } //DrawCharToScale

}


void RenderWaveform
(
  const int2 Fetch_Pos
)
{
  const float3 pixel = tex2Dfetch(SamplerBackBuffer, Fetch_Pos).rgb;

  BRANCH()
  if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
  {
    float pixel_nits_encoded;

    float pixel_nits = Calc_Nits_Normalised(pixel);

#ifdef IS_HDR_CSP

    pixel_nits_encoded = Csp::Trc::LinearTo::Pq(pixel_nits);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    pixel_nits_encoded = ENCODE_SDR(pixel_nits);

#endif

    int coord_x = float(Fetch_Pos.x)
                * 0.5f;

    int coord_y = float(TEXTURE_WAVEFORM_USED_HEIGHT)
                - pixel_nits_encoded * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                + 0.5f;

    atomicAdd(StorageWaveform, int3(coord_x, coord_y, 0), 1u);
  }
  else
  BRANCH()
  if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL)
  {
    float pixel_max_cll_encoded;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    float3 pixel_cll = Calc_Cll_Normalised(pixel);

    pixel_max_cll_encoded = Csp::Trc::LinearTo::Pq(MAXRGB(pixel_cll));

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
    || ACTUAL_COLOUR_SPACE == CSP_SRGB)

    //this is more performant to do
    pixel_max_cll_encoded = MAXRGB(pixel);

#endif

    int coord_x = float(Fetch_Pos.x)
                * 0.5f;

    int coord_y = float(TEXTURE_WAVEFORM_USED_HEIGHT)
                - pixel_max_cll_encoded * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                + 0.5f;

    atomicAdd(StorageWaveform, int3(coord_x, coord_y, 0), 1u);
  }
  else
  BRANCH()
  if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_COMBINED)
  {
    float3 pixel_cll_encoded;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    float3 pixel_cll = Calc_Cll_Normalised(pixel);

    pixel_cll_encoded = Csp::Trc::LinearTo::Pq(pixel_cll);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
    || ACTUAL_COLOUR_SPACE == CSP_SRGB)

    //this is more performant to do
    pixel_cll_encoded = pixel;

#endif

    int coord_x = float(Fetch_Pos.x)
                * 0.5f;

    int3 coords_y = float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  - pixel_cll_encoded * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  + 0.5f;

    atomicAdd(StorageWaveform, int3(coord_x, coords_y.r, 0), 1u);
    atomicAdd(StorageWaveform, int3(coord_x, coords_y.g, 1), 1u);
    atomicAdd(StorageWaveform, int3(coord_x, coords_y.b, 2), 1u);
  }
  else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
  {
    float3 pixel_cll_encoded;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    float3 pixel_cll = Calc_Cll_Normalised(pixel);

    pixel_cll_encoded = Csp::Trc::LinearTo::Pq(pixel_cll);

    //this is more performant to do
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
    || ACTUAL_COLOUR_SPACE == CSP_SRGB)

    pixel_cll_encoded = pixel;

#endif

    float x_normalised = float(Fetch_Pos.x)
                       / BUFFER_WIDTH_MINUS_1_FLOAT;

    const uint waveform_width_div3 = uint(TEXTURE_WAVEFORM_WIDTH) / 3u;

    int3 coords_x;

    coords_x.r = x_normalised * float(waveform_width_div3);

    coords_x.g = coords_x.r + waveform_width_div3;
    coords_x.b = coords_x.g + waveform_width_div3;

    int3 coords_y = float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  - pixel_cll_encoded * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  + 0.5f;

    atomicAdd(StorageWaveform, int3(coords_x.r, coords_y.r, 0), 1u);
    atomicAdd(StorageWaveform, int3(coords_x.g, coords_y.g, 0), 1u);
    atomicAdd(StorageWaveform, int3(coords_x.b, coords_y.b, 0), 1u);
  }

  return;
}


groupshared uint Group_Max;
void CS_Get_Max_Waveform_Value
(
  uint3 GTID : SV_GroupThreadID,
  uint3 DTID : SV_DispatchThreadID
)
{
  BRANCH()
  if (_SHOW_WAVEFORM)
  {
    const bool is_dtid_00 = all(DTID.xy == 0u);

    [branch]
    if (is_dtid_00)
    {
      tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_WAVEFORM_COUNTER_MAX, 0);
    }

    const bool is_gtid_00 = all(GTID.xy == 0u);

    [branch]
    if (is_gtid_00)
    {
      Group_Max = 0u;
    }

    barrier();

    // worth to do performance wise
#if (defined(GET_MAX_WAVEFORM_VALUE_NEEDS_X_CHECK) \
  && defined(GET_MAX_WAVEFORM_VALUE_NEEDS_Y_CHECK))

    const bool is_active_area = all(DTID.xy < uint2(TEXTURE_WAVEFORM_WIDTH, TEXTURE_WAVEFORM_HEIGHT));

#elif (defined(GET_MAX_WAVEFORM_VALUE_NEEDS_X_CHECK))

    const bool is_active_area = DTID.x < uint(TEXTURE_WAVEFORM_WIDTH);

#elif (defined(GET_MAX_WAVEFORM_VALUE_NEEDS_Y_CHECK))

    const bool is_active_area = DTID.y < uint(TEXTURE_WAVEFORM_HEIGHT);

#endif

#if (defined(GET_MAX_WAVEFORM_VALUE_NEEDS_X_CHECK) \
  || defined(GET_MAX_WAVEFORM_VALUE_NEEDS_Y_CHECK))
    [branch]
    if (is_active_area)
#endif
    {
      static const int2 dtid_as_int = int2(DTID.xy);

      BRANCH()
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE
       || _WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL
       || _WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
      {
        uint local_value = tex3Dfetch(StorageWaveform, int3(dtid_as_int, 0));

        atomicMax(Group_Max, local_value);
      }
      else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_COMBINED)
      {
        uint3 local_waveform;

        local_waveform.r = tex3Dfetch(StorageWaveform, int3(dtid_as_int, 0));
        local_waveform.g = tex3Dfetch(StorageWaveform, int3(dtid_as_int, 1));
        local_waveform.b = tex3Dfetch(StorageWaveform, int3(dtid_as_int, 2));

        atomicMax(Group_Max, MAXRGB(local_waveform));
      }

      groupMemoryBarrier();

      [branch]
      if (is_gtid_00)
      {
#ifdef IS_FLOAT_HDR_CSP
        atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_WAVEFORM_COUNTER_MAX, int(Group_Max));
#else
        atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_WAVEFORM_COUNTER_MAX, Group_Max);
#endif
      }
    }
  }

  return;
}


void RenderWaveformScale
(
  const float Unrolling_Be_Gone_Float,
  const uint  Unrolling_Be_Gone_Uint,
  const int   Unrolling_Be_Gone_Int
)
{
  [branch]
  if (tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_X)           != _WAVEFORM_SIZE.x
   || tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_Y)           != _WAVEFORM_SIZE.y
   || tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_TEXT_SIZE_ADJUST) != _WAVEFORM_TEXT_SIZE_ADJUST
#ifdef IS_HDR_CSP
   || tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_CUTOFF_POINT)     != WAVEFORM_CUTOFF_POINT
#endif
  )
  {
    //make background all black
    [loop]
    for (int x = 0; x < (TEXTURE_WAVEFORM_SCALE_WIDTH + Unrolling_Be_Gone_Int); x++)
    {
      [loop]
      for (int y = 0; y < (TEXTURE_WAVEFORM_SCALE_HEIGHT + Unrolling_Be_Gone_Int); y++)
      {
        tex2Dstore(StorageWaveformScale, int2(x, y), float4(0.f, 0.f, 0.f, 0.f));
      }
    }

    memoryBarrier();

    Waveform::SWaveformData waveDat = Waveform::GetData();

#ifdef IS_HDR_CSP

    int2 nitsOffsets[16];

    [loop]
    for (uint i = 0u; i < (16u + Unrolling_Be_Gone_Uint); i++)
    {
      nitsOffsets[i] = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[i]);
    }

    #define nits10000_00Offset nitsOffsets[ 0]
    #define nits_4000_00Offset nitsOffsets[ 1]
    #define nits_2000_00Offset nitsOffsets[ 2]
    #define nits_1000_00Offset nitsOffsets[ 3]
    #define nits__400_00Offset nitsOffsets[ 4]
    #define nits__203_00Offset nitsOffsets[ 5]
    #define nits__100_00Offset nitsOffsets[ 6]
    #define nits___50_00Offset nitsOffsets[ 7]
    #define nits___25_00Offset nitsOffsets[ 8]
    #define nits___10_00Offset nitsOffsets[ 9]
    #define nits____5_00Offset nitsOffsets[10]
    #define nits____2_50Offset nitsOffsets[11]
    #define nits____1_00Offset nitsOffsets[12]
    #define nits____0_25Offset nitsOffsets[13]
    #define nits____0_05Offset nitsOffsets[14]
    #define nits____0_00Offset nitsOffsets[15]


    const int2 text10000_00Offset = nits10000_00Offset - waveDat.textOffset;
    const int2 text_4000_00Offset = nits_4000_00Offset - waveDat.textOffset;
    const int2 text_2000_00Offset = nits_2000_00Offset - waveDat.textOffset;
    const int2 text_1000_00Offset = nits_1000_00Offset - waveDat.textOffset;
    const int2 text__400_00Offset = nits__400_00Offset - waveDat.textOffset;
    const int2 text__203_00Offset = nits__203_00Offset - waveDat.textOffset;
    const int2 text__100_00Offset = nits__100_00Offset - waveDat.textOffset;
    const int2 text___50_00Offset = nits___50_00Offset - waveDat.textOffset;
    const int2 text___25_00Offset = nits___25_00Offset - waveDat.textOffset;
    const int2 text___10_00Offset = nits___10_00Offset - waveDat.textOffset;
    const int2 text____5_00Offset = nits____5_00Offset - waveDat.textOffset;
    const int2 text____2_50Offset = nits____2_50Offset - waveDat.textOffset;
    const int2 text____1_00Offset = nits____1_00Offset - waveDat.textOffset;
    const int2 text____0_25Offset = nits____0_25Offset - waveDat.textOffset;
    const int2 text____0_05Offset = nits____0_05Offset - waveDat.textOffset;
    const int2 text____0_00Offset = nits____0_00Offset - waveDat.textOffset;


    int charOffsets[8];

    if (WAVEFORM_CUTOFF_POINT == 0)
    {
      charOffsets = {
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7 };
    }
    else //if (WAVEFORM_CUTOFF_POINT > 0)
    {
      charOffsets = {
        0,
        0,
        1,
        2,
        3,
        4,
        5,
        6 };
    }


    static const uint2 charList10000_00[8] = { _1, _0, _0, _0, _0, _dot, _0, _0 };
    static const uint2 charList_4000_00[7] = {     _4, _0, _0, _0, _dot, _0, _0 };
    static const uint2 charList_2000_00[7] = {     _2, _0, _0, _0, _dot, _0, _0 };
    static const uint2 charList_1000_00[7] = {     _1, _0, _0, _0, _dot, _0, _0 };
    static const uint2 charList__400_00[6] = {         _4, _0, _0, _dot, _0, _0 };
    static const uint2 charList__203_00[6] = {         _2, _0, _3, _dot, _0, _0 };
    static const uint2 charList__100_00[6] = {         _1, _0, _0, _dot, _0, _0 };
    static const uint2 charList___50_00[5] = {             _5, _0, _dot, _0, _0 };
    static const uint2 charList___25_00[5] = {             _2, _5, _dot, _0, _0 };
    static const uint2 charList___10_00[5] = {             _1, _0, _dot, _0, _0 };
    static const uint2 charList____5_00[4] = {                 _5, _dot, _0, _0 };
    static const uint2 charList____2_50[4] = {                 _2, _dot, _5, _0 };
    static const uint2 charList____1_00[4] = {                 _1, _dot, _0, _0 };
    static const uint2 charList____0_25[4] = {                 _0, _dot, _2, _5 };
    static const uint2 charList____0_05[4] = {                 _0, _dot, _0, _5 };
    static const uint2 charList____0_00[4] = {                 _0, _dot, _0, _0 };

    static const uint charListsCount = 16;

    [loop]
    for (uint i = 0u; i < (charListsCount + Unrolling_Be_Gone_Uint); i++)
    {
      uint2 currentNumber;

      int2 currentTextOffset;

      int currentCharOffset;

      bool needsDrawing = true;


      [loop]
      for (int j = 0u; j < (8u + Unrolling_Be_Gone_Uint); j++)
      {
        const int minj6 = min(j, 6);
        const int minj5 = min(j, 5);
        const int minj4 = min(j, 4);
        const int minj3 = min(j, 3);

        [forcecase]
        switch(i)
        {
          case 0:
          {
            currentNumber = charList10000_00[j];

            currentTextOffset = text10000_00Offset;

            currentCharOffset = 0;

            needsDrawing = WAVEFORM_CUTOFF_POINT == 0;
          }
          break;
          case 1:
          {
            currentNumber = charList_4000_00[minj6];

            currentTextOffset = text_4000_00Offset;

            currentCharOffset = 1;

            needsDrawing = WAVEFORM_CUTOFF_POINT <= 1;
          }
          break;
          case 2:
          {
            currentNumber = charList_2000_00[minj6];

            currentTextOffset = text_2000_00Offset;

            currentCharOffset = 1;

            needsDrawing = WAVEFORM_CUTOFF_POINT <= 2;
          }
          break;
          case 3:
          {
            currentNumber = charList_1000_00[minj6];

            currentTextOffset = text_1000_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 4:
          {
            currentNumber = charList__400_00[minj5];

            currentTextOffset = text__400_00Offset;

            currentCharOffset = 2;
          }
          break;
          case 5:
          {
            currentNumber = charList__203_00[minj5];

            currentTextOffset = text__203_00Offset;

            currentCharOffset = 2;
          }
          break;
          case 6:
          {
            currentNumber = charList__100_00[minj5];

            currentTextOffset = text__100_00Offset;

            currentCharOffset = 2;
          }
          break;
          case 7:
          {
            currentNumber = charList___50_00[minj4];

            currentTextOffset = text___50_00Offset;

            currentCharOffset = 3;
          }
          break;
          case 8:
          {
            currentNumber = charList___25_00[minj4];

            currentTextOffset = text___25_00Offset;

            currentCharOffset = 3;
          }
          break;
          case 9:
          {
            currentNumber = charList___10_00[minj4];

            currentTextOffset = text___10_00Offset;

            currentCharOffset = 3;
          }
          break;
          case 10:
          {
            currentNumber = charList____5_00[minj3];

            currentTextOffset = text____5_00Offset;

            currentCharOffset = 4;
          }
          break;
          case 11:
          {
            currentNumber = charList____2_50[minj3];

            currentTextOffset = text____2_50Offset;

            currentCharOffset = 4;
          }
          break;
          case 12:
          {
            currentNumber = charList____1_00[minj3];

            currentTextOffset = text____1_00Offset;

            currentCharOffset = 4;
          }
          break;
          case 13:
          {
            currentNumber = charList____0_25[minj3];

            currentTextOffset = text____0_25Offset;

            currentCharOffset = 4;
          }
          break;
          case 14:
          {
            currentNumber = charList____0_05[minj3];

            currentTextOffset = text____0_05Offset;

            currentCharOffset = 4;
          }
          break;
          default: //case 15:
          {
            currentNumber = charList____0_00[minj3];

            currentTextOffset = text____0_00Offset;

            currentCharOffset = 4;
          }
          break;
        }

        [branch]
        if (needsDrawing
         && (-(j - 7) >= currentCharOffset))
        {
          Waveform::DrawCharToScale(Unrolling_Be_Gone_Int,
                                    currentNumber,
                                    waveDat.charDimensions,
                                    currentTextOffset,
                                    charOffsets[j + currentCharOffset]);
        }
      }
    }

#else

    int2 nitsOffsets[14];

    [loop]
    for (uint i = 0u; i < (14u + Unrolling_Be_Gone_Uint); i++)
    {
      nitsOffsets[i] = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[i]);
    }

    #define nits100_00Offset nitsOffsets[ 0]
    #define nits_87_50Offset nitsOffsets[ 1]
    #define nits_75_00Offset nitsOffsets[ 2]
    #define nits_60_00Offset nitsOffsets[ 3]
    #define nits_50_00Offset nitsOffsets[ 4]
    #define nits_35_00Offset nitsOffsets[ 5]
    #define nits_25_00Offset nitsOffsets[ 6]
    #define nits_18_00Offset nitsOffsets[ 7]
    #define nits_10_00Offset nitsOffsets[ 8]
    #define nits__5_00Offset nitsOffsets[ 9]
    #define nits__2_50Offset nitsOffsets[10]
    #define nits__1_00Offset nitsOffsets[11]
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
    #define nits__0_25Offset nitsOffsets[12]
#else
    #define nits__0_40Offset nitsOffsets[12]
#endif
    #define nits__0_00Offset nitsOffsets[13]

    const int2 text100_00Offset = nits100_00Offset - waveDat.textOffset;
    const int2 text_87_50Offset = nits_87_50Offset - waveDat.textOffset;
    const int2 text_75_00Offset = nits_75_00Offset - waveDat.textOffset;
    const int2 text_60_00Offset = nits_60_00Offset - waveDat.textOffset;
    const int2 text_50_00Offset = nits_50_00Offset - waveDat.textOffset;
    const int2 text_35_00Offset = nits_35_00Offset - waveDat.textOffset;
    const int2 text_25_00Offset = nits_25_00Offset - waveDat.textOffset;
    const int2 text_18_00Offset = nits_18_00Offset - waveDat.textOffset;
    const int2 text_10_00Offset = nits_10_00Offset - waveDat.textOffset;
    const int2 text__5_00Offset = nits__5_00Offset - waveDat.textOffset;
    const int2 text__2_50Offset = nits__2_50Offset - waveDat.textOffset;
    const int2 text__1_00Offset = nits__1_00Offset - waveDat.textOffset;
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
    const int2 text__0_25Offset = nits__0_25Offset - waveDat.textOffset;
#else
    const int2 text__0_40Offset = nits__0_40Offset - waveDat.textOffset;
#endif
    const int2 text__0_00Offset = nits__0_00Offset - waveDat.textOffset;


    const int charOffsets[7] = {0, 1, 2, 3, 4, 5, 6};


    static const uint2 charList100_00[7] = { _1, _0, _0, _dot, _0, _0, _percent };
    static const uint2 charList_87_50[6] = {     _8, _7, _dot, _5, _0, _percent };
    static const uint2 charList_75_00[6] = {     _7, _5, _dot, _0, _0, _percent };
    static const uint2 charList_60_00[6] = {     _6, _0, _dot, _0, _0, _percent };
    static const uint2 charList_50_00[6] = {     _5, _0, _dot, _0, _0, _percent };
    static const uint2 charList_35_00[6] = {     _3, _5, _dot, _0, _0, _percent };
    static const uint2 charList_25_00[6] = {     _2, _5, _dot, _0, _0, _percent };
    static const uint2 charList_18_00[6] = {     _1, _8, _dot, _0, _0, _percent };
    static const uint2 charList_10_00[6] = {     _1, _0, _dot, _0, _0, _percent };
    static const uint2 charList__5_00[5] = {         _5, _dot, _0, _0, _percent };
    static const uint2 charList__2_50[5] = {         _2, _dot, _5, _0, _percent };
    static const uint2 charList__1_00[5] = {         _1, _dot, _0, _0, _percent };
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
    static const uint2 charList__0_25[5] = {         _0, _dot, _2, _5, _percent };
#else
    static const uint2 charList__0_40[5] = {         _0, _dot, _4, _0, _percent };
#endif
    static const uint2 charList__0_00[5] = {         _0, _dot, _0, _0, _percent };

    static const uint charListsCount = 14;

    float2 charDims;

    charDims.y = waveDat.charDimensions.y;

    [loop]
    for (uint i = 0; i < (charListsCount + Unrolling_Be_Gone_Uint); i++)
    {
      uint2 currentNumber;

      int2 currentTextOffset;

      int currentCharOffset;

      [loop]
      for (int j = 0; j < (7 + Unrolling_Be_Gone_Int); j++)
      {
        const int minj5 = min(j, 5);
        const int minj4 = min(j, 4);

        [forcecase]
        switch(i)
        {
          case 0:
          {
            currentNumber = charList100_00[j];

            currentTextOffset = text100_00Offset;

            currentCharOffset = 0;
          }
          break;
          case 1:
          {
            currentNumber = charList_87_50[minj5];

            currentTextOffset = text_87_50Offset;

            currentCharOffset = 1;
          }
          break;
          case 2:
          {
            currentNumber = charList_75_00[minj5];

            currentTextOffset = text_75_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 3:
          {
            currentNumber = charList_60_00[minj5];

            currentTextOffset = text_60_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 4:
          {
            currentNumber = charList_50_00[minj5];

            currentTextOffset = text_50_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 5:
          {
            currentNumber = charList_35_00[minj5];

            currentTextOffset = text_35_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 6:
          {
            currentNumber = charList_25_00[minj5];

            currentTextOffset = text_25_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 7:
          {
            currentNumber = charList_18_00[minj5];

            currentTextOffset = text_18_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 8:
          {
            currentNumber = charList_10_00[minj5];

            currentTextOffset = text_10_00Offset;

            currentCharOffset = 1;
          }
          break;
          case 9:
          {
            currentNumber = charList__5_00[minj4];

            currentTextOffset = text__5_00Offset;

            currentCharOffset = 2;
          }
          break;
          case 10:
          {
            currentNumber = charList__2_50[minj4];

            currentTextOffset = text__2_50Offset;

            currentCharOffset = 2;
          }
          break;
          case 11:
          {
            currentNumber = charList__1_00[minj4];

            currentTextOffset = text__1_00Offset;

            currentCharOffset = 2;
          }
          break;
          case 12:
          {
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
            currentNumber = charList__0_25[minj4];

            currentTextOffset = text__0_25Offset;
#else
            currentNumber = charList__0_40[minj4];

            currentTextOffset = text__0_40Offset;
#endif
            currentCharOffset = 2;
          }
          break;
          default: //case 13:
          {
            currentNumber = charList__0_00[minj4];

            currentTextOffset = text__0_00Offset;

            currentCharOffset = 2;
          }
          break;
        }

        charDims.x = currentNumber != _percent ? waveDat.charDimensions.x
                                               : waveDat.charDimensionXForPercent;

        [branch]
        if (-(j - 7) > currentCharOffset)
        {
          Waveform::DrawCharToScale(Unrolling_Be_Gone_Int,
                                    currentNumber,
                                    charDims,
                                    currentTextOffset,
                                    charOffsets[j + currentCharOffset]);
        }
      }
    }

#endif

    // draw the frame, ticks and horizontal lines
    [loop]
    for (int y = 0; y < (waveDat.endXY.y + Unrolling_Be_Gone_Int); y++)
    {
      int2 curPos = waveDat.offsetToFrame
                  + int2(0, y);

      float curGrey = lerp(0.23f, 0.13f, (float(y + WAVEDAT_CUTOFF_OFFSET) / float(waveDat.endYminus1 + WAVEDAT_CUTOFF_OFFSET)));
      // using gamma 2 as intermediate gamma space
      curGrey = sqrt(curGrey);

      float4 curColour = float4(curGrey, 1.f.xxx);

      // draw top and bottom part of the frame
      [branch]
      if (y <  waveDat.frameSize
       || y >= waveDat.lowerFrameStart)
      {
        [loop]
        for (int x = 0; x < (waveDat.endXY.x + Unrolling_Be_Gone_Int); x++)
        {
          int2 curXPos = int2(curPos.x + x,
                              curPos.y);
          tex2Dstore(StorageWaveformScale, curXPos, curColour);
        }
      }
      // draw left and right part of the frame
      else
      {
        [loop]
        for (int x = 0; x < (waveDat.frameSize + Unrolling_Be_Gone_Int); x++)
        {
          int2 curLeftPos  = int2(curPos.x + x,
                                  curPos.y);
          int2 curRightPos = int2(curLeftPos.x + waveDat.waveformArea.x + waveDat.frameSize, curLeftPos.y);
          tex2Dstore(StorageWaveformScale, curLeftPos,  curColour);
          tex2Dstore(StorageWaveformScale, curRightPos, curColour);
        }
      }

      // draw top tick and bottom tick
      [branch]
#ifdef IS_HDR_CSP
  #ifdef IS_QHD_OR_HIGHER_RES
      if ((WAVEFORM_CUTOFF_POINT == 0 && ((nits10000_00Offset.y - 1) == curPos.y || nits10000_00Offset.y == curPos.y || (nits10000_00Offset.y + 1) == curPos.y))
       || (WAVEFORM_CUTOFF_POINT == 1 && ((nits_4000_00Offset.y - 1) == curPos.y || nits_4000_00Offset.y == curPos.y || (nits_4000_00Offset.y + 1) == curPos.y))
       || (WAVEFORM_CUTOFF_POINT == 2 && ((nits_2000_00Offset.y - 1) == curPos.y || nits_2000_00Offset.y == curPos.y || (nits_2000_00Offset.y + 1) == curPos.y))
       || (WAVEFORM_CUTOFF_POINT == 3 && ((nits_1000_00Offset.y - 1) == curPos.y || nits_1000_00Offset.y == curPos.y || (nits_1000_00Offset.y + 1) == curPos.y))
       || (nits____0_00Offset.y - 1) == curPos.y || nits____0_00Offset.y == curPos.y || (nits____0_00Offset.y + 1) == curPos.y)
  #else
      if ((WAVEFORM_CUTOFF_POINT == 0 && nits10000_00Offset.y == curPos.y)
       || (WAVEFORM_CUTOFF_POINT == 1 && nits_4000_00Offset.y == curPos.y)
       || (WAVEFORM_CUTOFF_POINT == 2 && nits_2000_00Offset.y == curPos.y)
       || (WAVEFORM_CUTOFF_POINT == 3 && nits_1000_00Offset.y == curPos.y)
       || nits____0_00Offset.y == curPos.y)
  #endif
#else
  #ifdef IS_QHD_OR_HIGHER_RES
      if ((nits100_00Offset.y - 1) == curPos.y || nits100_00Offset.y == curPos.y || (nits100_00Offset.y + 1) == curPos.y
       || (nits__0_00Offset.y - 1) == curPos.y || nits__0_00Offset.y == curPos.y || (nits__0_00Offset.y + 1) == curPos.y)
  #else
      if (nits100_00Offset.y == curPos.y
       || nits__0_00Offset.y == curPos.y)
  #endif
#endif
      {
        [loop]
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.frameSize + Unrolling_Be_Gone_Int); x++)
        {
          int2 curTickPos = int2(x,
                                 curPos.y);
          tex2Dstore(StorageWaveformScale, curTickPos, curColour);
        }
      }

      // draw ticks + draw horizontal lines
      [branch]
#ifdef IS_HDR_CSP
  #ifdef IS_QHD_OR_HIGHER_RES
      if ((WAVEFORM_CUTOFF_POINT < 1 && ((nits_4000_00Offset.y - 1) == curPos.y || nits_4000_00Offset.y == curPos.y || (nits_4000_00Offset.y + 1) == curPos.y))
       || (WAVEFORM_CUTOFF_POINT < 2 && ((nits_2000_00Offset.y - 1) == curPos.y || nits_2000_00Offset.y == curPos.y || (nits_2000_00Offset.y + 1) == curPos.y))
       || (WAVEFORM_CUTOFF_POINT < 3 && ((nits_1000_00Offset.y - 1) == curPos.y || nits_1000_00Offset.y == curPos.y || (nits_1000_00Offset.y + 1) == curPos.y))
       || (nits__400_00Offset.y - 1) == curPos.y || nits__400_00Offset.y == curPos.y || (nits__400_00Offset.y + 1) == curPos.y
       || (nits__203_00Offset.y - 1) == curPos.y || nits__203_00Offset.y == curPos.y || (nits__203_00Offset.y + 1) == curPos.y
       || (nits__100_00Offset.y - 1) == curPos.y || nits__100_00Offset.y == curPos.y || (nits__100_00Offset.y + 1) == curPos.y
       || (nits___50_00Offset.y - 1) == curPos.y || nits___50_00Offset.y == curPos.y || (nits___50_00Offset.y + 1) == curPos.y
       || (nits___25_00Offset.y - 1) == curPos.y || nits___25_00Offset.y == curPos.y || (nits___25_00Offset.y + 1) == curPos.y
       || (nits___10_00Offset.y - 1) == curPos.y || nits___10_00Offset.y == curPos.y || (nits___10_00Offset.y + 1) == curPos.y
       || (nits____5_00Offset.y - 1) == curPos.y || nits____5_00Offset.y == curPos.y || (nits____5_00Offset.y + 1) == curPos.y
       || (nits____2_50Offset.y - 1) == curPos.y || nits____2_50Offset.y == curPos.y || (nits____2_50Offset.y + 1) == curPos.y
       || (nits____1_00Offset.y - 1) == curPos.y || nits____1_00Offset.y == curPos.y || (nits____1_00Offset.y + 1) == curPos.y
       || (nits____0_25Offset.y - 1) == curPos.y || nits____0_25Offset.y == curPos.y || (nits____0_25Offset.y + 1) == curPos.y
       || (nits____0_05Offset.y - 1) == curPos.y || nits____0_05Offset.y == curPos.y || (nits____0_05Offset.y + 1) == curPos.y)
  #else
      if ((WAVEFORM_CUTOFF_POINT < 1 && nits_4000_00Offset.y == curPos.y)
       || (WAVEFORM_CUTOFF_POINT < 2 && nits_2000_00Offset.y == curPos.y)
       || (WAVEFORM_CUTOFF_POINT < 3 && nits_1000_00Offset.y == curPos.y)
       || nits__400_00Offset.y == curPos.y
       || nits__203_00Offset.y == curPos.y
       || nits__100_00Offset.y == curPos.y
       || nits___50_00Offset.y == curPos.y
       || nits___25_00Offset.y == curPos.y
       || nits___10_00Offset.y == curPos.y
       || nits____5_00Offset.y == curPos.y
       || nits____2_50Offset.y == curPos.y
       || nits____1_00Offset.y == curPos.y
       || nits____0_25Offset.y == curPos.y
       || nits____0_05Offset.y == curPos.y)
  #endif
#else
  #ifdef IS_QHD_OR_HIGHER_RES
      if ((nits_87_50Offset.y - 1) == curPos.y || nits_87_50Offset.y == curPos.y || (nits_87_50Offset.y + 1) == curPos.y
       || (nits_75_00Offset.y - 1) == curPos.y || nits_75_00Offset.y == curPos.y || (nits_75_00Offset.y + 1) == curPos.y
       || (nits_60_00Offset.y - 1) == curPos.y || nits_60_00Offset.y == curPos.y || (nits_60_00Offset.y + 1) == curPos.y
       || (nits_50_00Offset.y - 1) == curPos.y || nits_50_00Offset.y == curPos.y || (nits_50_00Offset.y + 1) == curPos.y
       || (nits_35_00Offset.y - 1) == curPos.y || nits_35_00Offset.y == curPos.y || (nits_35_00Offset.y + 1) == curPos.y
       || (nits_25_00Offset.y - 1) == curPos.y || nits_25_00Offset.y == curPos.y || (nits_25_00Offset.y + 1) == curPos.y
       || (nits_18_00Offset.y - 1) == curPos.y || nits_18_00Offset.y == curPos.y || (nits_18_00Offset.y + 1) == curPos.y
       || (nits_10_00Offset.y - 1) == curPos.y || nits_10_00Offset.y == curPos.y || (nits_10_00Offset.y + 1) == curPos.y
       || (nits__5_00Offset.y - 1) == curPos.y || nits__5_00Offset.y == curPos.y || (nits__5_00Offset.y + 1) == curPos.y
       || (nits__2_50Offset.y - 1) == curPos.y || nits__2_50Offset.y == curPos.y || (nits__2_50Offset.y + 1) == curPos.y
       || (nits__1_00Offset.y - 1) == curPos.y || nits__1_00Offset.y == curPos.y || (nits__1_00Offset.y + 1) == curPos.y
    #if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
      || OVERWRITE_SDR_GAMMA == GAMMA_22    \
      || OVERWRITE_SDR_GAMMA == GAMMA_24)
       || (nits__0_25Offset.y - 1) == curPos.y || nits__0_25Offset.y == curPos.y || (nits__0_25Offset.y + 1) == curPos.y
    #else
       || (nits__0_40Offset.y - 1) == curPos.y || nits__0_40Offset.y == curPos.y || (nits__0_40Offset.y + 1) == curPos.y
    #endif
      )
  #else
      if (nits_87_50Offset.y == curPos.y
       || nits_75_00Offset.y == curPos.y
       || nits_60_00Offset.y == curPos.y
       || nits_50_00Offset.y == curPos.y
       || nits_35_00Offset.y == curPos.y
       || nits_25_00Offset.y == curPos.y
       || nits_18_00Offset.y == curPos.y
       || nits_10_00Offset.y == curPos.y
       || nits__5_00Offset.y == curPos.y
       || nits__2_50Offset.y == curPos.y
       || nits__1_00Offset.y == curPos.y
    #if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
      || OVERWRITE_SDR_GAMMA == GAMMA_22    \
      || OVERWRITE_SDR_GAMMA == GAMMA_24)
       || nits__0_25Offset.y == curPos.y
    #else
       || nits__0_40Offset.y == curPos.y
    #endif
      )
  #endif
#endif
      {
        [loop]
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.endXY.x + Unrolling_Be_Gone_Int); x++)
        {
          int2 curTickPos = int2(x,
                                 curPos.y);
          tex2Dstore(StorageWaveformScale, curTickPos, curColour);
        }
      }
    }

    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_X,           _WAVEFORM_SIZE.x);
    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_Y,           _WAVEFORM_SIZE.y);
    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_TEXT_SIZE_ADJUST, _WAVEFORM_TEXT_SIZE_ADJUST);
#ifdef IS_HDR_CSP
    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_CUTOFF_POINT,     WAVEFORM_CUTOFF_POINT);
#endif
  }

  return;
}


int GetNitsLine
(
  const float Nits,
  const float WaveformScaleFactorY
#ifdef IS_HDR_CSP
                                  ,
  const int   WaveDatCutoffOffset
#endif
)
{
#ifdef IS_HDR_CSP
  float encodedNits = Csp::Trc::NitsTo::Pq(Nits);
#else
  float encodedNits = ENCODE_SDR(Nits / 100.f);
#endif

  return int((float(TEXTURE_WAVEFORM_USED_HEIGHT)
            - (encodedNits * float(TEXTURE_WAVEFORM_USED_HEIGHT)))
           * WaveformScaleFactorY + 0.5f)
#ifdef IS_HDR_CSP
       - WaveDatCutoffOffset
#endif
                            ;
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareRenderWaveformToScale
(
  in                  uint   VertexID : SV_VertexID,
  out                 float4 Position : SV_Position,
  out nointerpolation int4   WaveDat0 : WaveDat0,
  out nointerpolation int4   WaveDat1 : WaveDat1,
  out nointerpolation int4   WaveDat2 : WaveDat2
#ifdef IS_HDR_CSP
                                                ,
  out nointerpolation int    WaveDat3 : WaveDat3
#endif
                                                ,
#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
  out nointerpolation float2 WaveDat4 : WaveDat4
#else
  out nointerpolation float  WaveDat4 : WaveDat4
#endif
)
{
  float2 texCoord;
  texCoord.x = (VertexID == 2) ? 2.f
                               : 0.f;
  texCoord.y = (VertexID == 1) ? 2.f
                               : 0.f;
  Position = float4(texCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#define WaveformActiveArea   WaveDat0.xy
#define OffsetToWaveformArea WaveDat0.zw

#define MinNitsLineY WaveDat1.x
#define MaxNitsLineY WaveDat2.x
#define MinRLineY    WaveDat1.r
#define MinGLineY    WaveDat1.g
#define MinBLineY    WaveDat1.b
#define MaxRLineY    WaveDat2.r
#define MaxGLineY    WaveDat2.g
#define MaxBLineY    WaveDat2.b
#define GLinePartX   WaveDat1.w
#define BLinePartX   WaveDat2.w
#define Waveform_Max WaveDat4.x

  WaveDat0    =  0;
  MinRLineY   =  INT_MAX;
  MinGLineY   =  INT_MAX;
  MinBLineY   =  INT_MAX;
  MaxRLineY   = -INT_MAX;
  MaxGLineY   = -INT_MAX;
  MaxBLineY   = -INT_MAX;
  GLinePartX  = -INT_MAX;
  BLinePartX  = -INT_MAX;
  WaveDat4    = 0.f;

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)

  #define WaveformSizeYFactor WaveDat4.y
#endif

#ifdef IS_HDR_CSP
  #define WaveformCutoffOffset WaveDat3

  WaveformCutoffOffset = 0;
#else
  #define WaveformCutoffOffset 0
#endif

  BRANCH()
  if (_SHOW_WAVEFORM)
  {
    Waveform::SWaveformData waveDat = Waveform::GetData();

    WaveformActiveArea = waveDat.waveformArea;

    OffsetToWaveformArea = waveDat.offsetToFrame
                         + waveDat.frameSize;

#ifdef IS_HDR_CSP
    WaveformCutoffOffset = waveDat.cutoffOffsetAt100PercentScale;
#endif

    float waveformSizeY = _WAVEFORM_SIZE.y / DIV_100;

#ifndef IS_HDR_CSP
    waveformSizeY += 3.f - (1.f - waveformSizeY) * 6.f;
#endif

    const float waveformScaleFactorY = clamp(waveformSizeY, WAVEFORM_SCALE_FACTOR_CLAMP_MIN, WAVEFORM_SCALE_FACTOR_CLAMP_MAX.y);

#ifdef IS_HDR_CSP
  #define MAX_NITS_LINE_CUTOFF 10000.f
#else
  #define MAX_NITS_LINE_CUTOFF 100.f
#endif

    BRANCH()
    if (_SHOW_WAVEFORM)
    {
      BRANCH()
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
      {
        const float minNits = tex1Dfetch(SamplerConsolidated, COORDS_MIN_NITS_VALUE);

        [branch]
        if (minNits > 0.f
         && minNits < MAX_NITS_LINE_CUTOFF)
        {
          MinNitsLineY = GetNitsLine(minNits,
                                     waveformScaleFactorY
#ifdef IS_HDR_CSP
                                   , waveDat.cutoffOffset
#endif
                                    );
        }
      }
      else
      BRANCH()
      if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL
       || _WAVEFORM_MODE == WAVEFORM_MODE_RGB_COMBINED)
      {
        const float minCll = tex1Dfetch(SamplerConsolidated, COORDS_MIN_CLL_VALUE);

        [branch]
        if (minCll > 0.f
         && minCll < MAX_NITS_LINE_CUTOFF)
        {
          MinNitsLineY = GetNitsLine(minCll,
                                     waveformScaleFactorY
#ifdef IS_HDR_CSP
                                   , waveDat.cutoffOffset
#endif
                                    );
        }
      }
      else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
      {
        const float3 minRgb = float3(tex1Dfetch(SamplerConsolidated, COORDS_MIN_R_VALUE),
                                     tex1Dfetch(SamplerConsolidated, COORDS_MIN_G_VALUE),
                                     tex1Dfetch(SamplerConsolidated, COORDS_MIN_B_VALUE));

        const bool3 drawMinRgbLine = minRgb > 0.f
                                  && minRgb < MAX_NITS_LINE_CUTOFF;

        [branch]
        if (drawMinRgbLine.r)
        {
          MinRLineY = GetNitsLine(minRgb.r,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMinRgbLine.g)
        {
          MinGLineY = GetNitsLine(minRgb.g,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMinRgbLine.b)
        {
          MinBLineY = GetNitsLine(minRgb.b,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
      }

      BRANCH()
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
      {
        const float maxNits = tex1Dfetch(SamplerConsolidated, COORDS_MAX_NITS_VALUE);

        [branch]
        if (maxNits > 0.f
         && maxNits < MAX_NITS_LINE_CUTOFF)
        {
          MaxNitsLineY = GetNitsLine(maxNits,
                                     waveformScaleFactorY
#ifdef IS_HDR_CSP
                                   , waveDat.cutoffOffset
#endif
                                    );
        }
      }
      else
      BRANCH()
      if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL
       || _WAVEFORM_MODE == WAVEFORM_MODE_RGB_COMBINED)
      {
        const float maxCll = tex1Dfetch(SamplerConsolidated, COORDS_MAX_CLL_VALUE);

        [branch]
        if (maxCll > 0.f
         && maxCll < MAX_NITS_LINE_CUTOFF)
        {
          MaxNitsLineY = GetNitsLine(maxCll,
                                     waveformScaleFactorY
#ifdef IS_HDR_CSP
                                   , waveDat.cutoffOffset
#endif
                                    );
        }
      }
      else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
      {
        const float3 maxRgb = float3(tex1Dfetch(SamplerConsolidated, COORDS_MAX_R_VALUE),
                                     tex1Dfetch(SamplerConsolidated, COORDS_MAX_G_VALUE),
                                     tex1Dfetch(SamplerConsolidated, COORDS_MAX_B_VALUE));

        const bool3 drawMaxRgbLine = maxRgb > 0.f
                                  && maxRgb < MAX_NITS_LINE_CUTOFF;

        [branch]
        if (drawMaxRgbLine.r)
        {
          MaxRLineY = GetNitsLine(maxRgb.r,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMaxRgbLine.g)
        {
          MaxGLineY = GetNitsLine(maxRgb.g,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMaxRgbLine.b)
        {
          MaxBLineY = GetNitsLine(maxRgb.b,
                                  waveformScaleFactorY
#ifdef IS_HDR_CSP
                                , waveDat.cutoffOffset
#endif
                                 );
        }
      }
    }

    BRANCH()
    if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
    {
      GLinePartX = uint(waveDat.waveformArea.x) / 3u;
      BLinePartX = GLinePartX + GLinePartX;
    }

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)

    WaveformSizeYFactor  = _WAVEFORM_SIZE.y + (300.f - ((800.f / 9.f) - _WAVEFORM_SIZE.y) * 6.f);
    WaveformSizeYFactor *= (3.f / 350.f);
    WaveformSizeYFactor  = 1.f / WaveformSizeYFactor;

#endif

    float waveform_max = (float)tex2Dfetch(SamplerMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_WAVEFORM_COUNTER_MAX);

    Waveform_Max = waveform_max;
  }
}

void PS_RenderWaveformToScale
(
  in                  float4 Position : SV_Position,
  in  nointerpolation int4   WaveDat0 : WaveDat0,
  in  nointerpolation int4   WaveDat1 : WaveDat1,
  in  nointerpolation int4   WaveDat2 : WaveDat2,
#ifdef IS_HDR_CSP
  in  nointerpolation int    WaveDat3 : WaveDat3,
#endif
#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
  in  nointerpolation float2 WaveDat4 : WaveDat4,
#else
  in  nointerpolation float  WaveDat4 : WaveDat4,
#endif
  out                 float4 Out      : SV_Target0
)
{
  Out = 0.f;

  BRANCH()
  if (_SHOW_WAVEFORM)
  {
    const int2 pureCoordAsInt = int2(Position.xy);

    const int2 scaleCoords = pureCoordAsInt;

    const int2 waveformCoords = pureCoordAsInt - OffsetToWaveformArea;

    float2 scale_colour = tex2Dfetch(SamplerWaveformScale, scaleCoords).xy;

    float4 yccrccbc_out = float4(scale_colour.x, (127.f / 255.f).xx, scale_colour.y);

    [branch]
    if (all(waveformCoords >= 0)
     && all(waveformCoords < WaveformActiveArea))
    {
      static const bool isRPart = waveformCoords.x  < GLinePartX;
      static const bool isGPart = waveformCoords.x  < BLinePartX;
      static const bool isBPart = waveformCoords.x >= BLinePartX;

      int minLineY;
      int maxLineY;

      FLATTEN()
      if (_WAVEFORM_MODE < WAVEFORM_MODE_RGB_INDIVIDUALLY) // WAVEFORM_MODE_LUMINANCE or WAVEFORM_MODE_MAX_CLL
      {
        minLineY = MinNitsLineY;
        maxLineY = MaxNitsLineY;
      }
      else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
      {
        [flatten]
        if (isRPart)
        {
          minLineY = MinRLineY;
          maxLineY = MaxRLineY;
        }
        else
        [flatten]
        if (isGPart)
        {
          minLineY = MinGLineY;
          maxLineY = MaxGLineY;
        }
        else
        {
          minLineY = MinBLineY;
          maxLineY = MaxBLineY;
        }
      }

      [branch]
      if
      (
        (
          _WAVEFORM_SHOW_MIN_NITS_LINE
#ifdef IS_QHD_OR_HIGHER_RES
        && (waveformCoords.y == minLineY
         || waveformCoords.y == minLineY - 1)
#else
        && waveformCoords.y == minLineY
#endif
        )
     || (
          _WAVEFORM_SHOW_MAX_NITS_LINE
#ifdef IS_QHD_OR_HIGHER_RES
        && (waveformCoords.y == maxLineY
         || waveformCoords.y == maxLineY + 1)
#else
        && waveformCoords.y == maxLineY
#endif
        )
      )
      {
        Out = float4(1.f, (127.f / 255.f).xx, 1.f);
        return;
      }
      else
      {
        bool waveformCoordsGTEMaxLine;
        bool waveformCoordsSTEMinLine;

        FLATTEN()
        if (_WAVEFORM_MODE < WAVEFORM_MODE_RGB_INDIVIDUALLY) // WAVEFORM_MODE_LUMINANCE or WAVEFORM_MODE_MAX_CLL
        {
          waveformCoordsGTEMaxLine = waveformCoords.y >= MaxNitsLineY;
          waveformCoordsSTEMinLine = waveformCoords.y <= MinNitsLineY;
        }
        else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
        {
          [flatten]
          if (isRPart)
          {
            waveformCoordsGTEMaxLine = waveformCoords.y >= MaxRLineY;
            waveformCoordsSTEMinLine = waveformCoords.y <= MinRLineY;
          }
          else
          [flatten]
          if (isGPart)
          {
            waveformCoordsGTEMaxLine = waveformCoords.y >= MaxGLineY;
            waveformCoordsSTEMinLine = waveformCoords.y <= MinGLineY;
          }
          else
          {
            waveformCoordsGTEMaxLine = waveformCoords.y >= MaxBLineY;
            waveformCoordsSTEMinLine = waveformCoords.y <= MinBLineY;
          }
        }

        yccrccbc_out.a = 1.f;

        [branch]
        if (waveformCoordsGTEMaxLine
         && waveformCoordsSTEMinLine)
        {
          int2 waveform_fetch_coords = waveformCoords;

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)

          waveform_fetch_coords.x = float(waveform_fetch_coords.x)
                                  / float(WaveformActiveArea.x)
                                  * float(TEXTURE_WAVEFORM_WIDTH);

          waveform_fetch_coords.y = float(waveform_fetch_coords.y)
                                  * WaveformSizeYFactor;

#else

          waveform_fetch_coords = float2(waveform_fetch_coords)
                                / float2(WaveformActiveArea.x, WaveformActiveArea.y)
                                * float2(TEXTURE_WAVEFORM_WIDTH, TEXTURE_WAVEFORM_USED_HEIGHT - WaveformCutoffOffset);

          waveform_fetch_coords.y += WaveformCutoffOffset;

#endif

          BRANCH()
          if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE
           || _WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL)
          {
            uint waveform_current = tex3Dfetch(SamplerWaveform, int3(waveform_fetch_coords, 0));

            [branch]
            if (waveform_current > 0u)
            {
              float waveform_colour = float(waveform_current)
                                    / Waveform_Max;

              float scale_colour = yccrccbc_out[0] * yccrccbc_out[0];

              waveform_colour = max(pow(waveform_colour, 1.f / 1.25f), 0.f);

              static const float plus_value = (1.f / float(255u * 255u)) * 1.05f;

              waveform_colour += plus_value;

              float waveform_out = scale_colour
                                 + waveform_colour;

              float yc = sqrt(waveform_out);

              yccrccbc_out[0] = yc;
            }
          }
          else
          BRANCH()
          if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_COMBINED)
          {
            uint3 waveform_current;

            waveform_current.r = tex3Dfetch(SamplerWaveform, int3(waveform_fetch_coords, 0));
            waveform_current.g = tex3Dfetch(SamplerWaveform, int3(waveform_fetch_coords, 1));
            waveform_current.b = tex3Dfetch(SamplerWaveform, int3(waveform_fetch_coords, 2));

            const bool3 waveform_current_above_0 = waveform_current.rgb > 0u;

            [branch]
            if (any(waveform_current_above_0))
            {
              float3 waveform_colour = float3(waveform_current.rgb)
                                     / Waveform_Max;

              float scale_colour = yccrccbc_out[0] * yccrccbc_out[0];

              float waveform_luminance = dot(Csp::Mat::Bt709ToXYZ[1], waveform_colour);

              waveform_colour *= pow(waveform_luminance, 1.f / 1.1f)
                               / waveform_luminance;

              static const float3 plus_values = (1.f / float(255u * 255u)) * 1.05f;

              waveform_colour += waveform_current_above_0 ? plus_values
                                                          : 0.f;

              float3 colour_out = saturate(waveform_colour + scale_colour.xxx);

              float3 yccrccbc;

              yccrccbc[0] = sqrt(dot(colour_out, Csp::Ycbcr::K_Bt709));

              yccrccbc.yz = sqrt(colour_out.rb) - yccrccbc[0];

              yccrccbc.yz *= yccrccbc.yz <= 0.f ? float2(Csp::Ycbcr::PR_NR_Bt709_g2_enc[1], Csp::Ycbcr::PB_NB_Bt709_g2_enc[1])
                                                : float2(Csp::Ycbcr::PR_NR_Bt709_g2_enc[0], Csp::Ycbcr::PB_NB_Bt709_g2_enc[0]);

              yccrccbc.yz += (127.f / 255.f);

              yccrccbc_out.xyz = yccrccbc;
            }
          }
          else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
          {
            uint waveform_current = tex3Dfetch(SamplerWaveform, uint3(waveform_fetch_coords, 0));

            [branch]
            if (waveform_current > 0u)
            {
              float waveform_colour_channel = float(waveform_current)
                                            / Waveform_Max;

              float scale_colour = yccrccbc_out[0] * yccrccbc_out[0];

              waveform_colour_channel = pow(waveform_colour_channel, 1.f / 1.25f);

              static const float plus_value = (1.f / float(255u * 255u)) * 1.05f;

              waveform_colour_channel += plus_value;

              float3 waveform_colour = (float3)0.f;
              [branch]
              if (isRPart)
              {
                waveform_colour.r  = waveform_colour_channel;
                waveform_colour.r *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][0];
                waveform_colour.g  = (waveform_colour.r - 1.f) * (Csp::Mat::Bt709ToXYZ[1][0] / Csp::Mat::Bt709ToXYZ[1][1]);
                waveform_colour    = saturate(waveform_colour);
              }
              else
              [branch]
              if (isGPart)
              {
                waveform_colour.g = waveform_colour_channel;
              }
              else
              {
                waveform_colour.b  = waveform_colour_channel;
                waveform_colour.b *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][2];
                waveform_colour.g  = (waveform_colour.b - 1.f) * (Csp::Mat::Bt709ToXYZ[1][2] / Csp::Mat::Bt709ToXYZ[1][1]);
                waveform_colour    = saturate(waveform_colour);
              }


              float3 colour_out = saturate(waveform_colour + scale_colour.xxx);

              float3 yccrccbc;

              yccrccbc[0] = sqrt(dot(colour_out, Csp::Ycbcr::K_Bt709));

              yccrccbc.yz = sqrt(colour_out.rb) - yccrccbc[0];

              yccrccbc.yz *= yccrccbc.yz <= 0.f ? float2(Csp::Ycbcr::PR_NR_Bt709_g2_enc[1], Csp::Ycbcr::PB_NB_Bt709_g2_enc[1])
                                                : float2(Csp::Ycbcr::PR_NR_Bt709_g2_enc[0], Csp::Ycbcr::PB_NB_Bt709_g2_enc[0]);

              yccrccbc.yz += (127.f / 255.f);

              yccrccbc_out.xyz = yccrccbc;
            }
          }
        }
      }
    }

    Out = yccrccbc_out;
    return;
  }
  else
  {
    discard;
  }
}


void CS_Clear_Texture_Waveform
(
  uint3 DTID : SV_DispatchThreadID
)
{
  BRANCH()
  if (_SHOW_WAVEFORM)
  {
    tex3Dstore(StorageWaveform, DTID, 0u);
  }

  return;
}
