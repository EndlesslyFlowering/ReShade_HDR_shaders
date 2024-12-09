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
                                               + (CHAR_DIM_FLOAT.x * 8) //8 chars for 10000.00
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
                                                + uint(CHAR_DIM_FLOAT.y / 2.f - TEXTURE_WAVEFORM_SCALE_FRAME + 0.5f)
                                                + (TEXTURE_WAVEFORM_SCALE_BORDER * 2)
                                                + (TEXTURE_WAVEFORM_SCALE_FRAME  * 2);

static const float TEXTURE_WAVEFORM_SCALE_FACTOR_X = (TEXTURE_WAVEFORM_SCALE_WIDTH - 1.f)
                                                   / float(TEXTURE_WAVEFORM_WIDTH  - 1);

static const float TEXTURE_WAVEFORM_SCALE_FACTOR_Y = (TEXTURE_WAVEFORM_SCALE_HEIGHT - 1.f)
                                                   / float(TEXTURE_WAVEFORM_HEIGHT  - 1);


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


texture2D TextureWaveform
<
  pooled = true;
>
{
  Width  = TEXTURE_WAVEFORM_WIDTH;
  Height = TEXTURE_WAVEFORM_HEIGHT;
  Format = RGB10A2;
};

sampler2D<float4> SamplerWaveform
{
  Texture = TextureWaveform;
};

storage2D<float4> StorageWaveform
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

    float2 waveformScaleFactorXY = clamp(_WAVEFORM_SIZE / 100.f, 0.5f, 1.f);

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
      int2(uint(TEXTURE_WAVEFORM_WIDTH * waveformScaleFactorXY.x * 3.f) / 3u,
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

// workaround so that the driver shader compiler doesn't
// unroll the loops and it taking 14 minutes or longer
#if ((!defined(API_IS_D3D11) && !defined(API_IS_D3D12)) \
  || ((defined(API_IS_D3D11) || defined(API_IS_D3D12)) && (BUFFER_WIDTH >= 1600 || BUFFER_HEIGHT >= 900) && __RESHADE_PERFORMANCE_MODE__ == 0))

  #define NO_WORKAROUND_NEEDED

#endif

  void DrawCharToScale
  (
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


    const int2 floorCharDim = floor(CharDim);

    int2 currentOffset = int2(0, 0);

    [loop]
    while (currentOffset.x < floorCharDim.x)
    {
#ifndef NO_WORKAROUND_NEEDED
      currentOffset.x += floor(FRAMETIME / 100000.f + 0.1f);
#endif
      [loop]
      while (currentOffset.y < floorCharDim.y)
      {
#ifndef NO_WORKAROUND_NEEDED
        currentOffset.y += floor(FRAMETIME / 100000.f);
#endif
        float2 currentSamplePos = charFetchPos
                                + float2(currentOffset)
                                * (min(CHAR_DIM_FLOAT / CharDim, 2.f))
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
  const int2 FetchPos
)
{
#ifdef IS_HDR_CSP
  static const float2 waveformSizeFactor = _WAVEFORM_SIZE / 100.f;
#else
  static const float2 waveformSizeFactor = float2(_WAVEFORM_SIZE.x / 100.f, 1.f);
#endif

  static const float2 coordFactors = float2(TEXTURE_WAVEFORM_BUFFER_WIDTH_FACTOR, float(TEXTURE_WAVEFORM_USED_HEIGHT))
                                   * waveformSizeFactor;

  BRANCH()
  if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
  {
    float pixelNits = CalcNits(tex2Dfetch(SamplerBackBuffer, FetchPos).rgb);

#ifdef IS_HDR_CSP
    float pixelEncoded = Csp::Trc::NitsTo::Pq(pixelNits);
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    float pixelEncoded = ENCODE_SDR(pixelNits / 100.f);
#endif

    const int2 coords = float2(float(FetchPos.x)
                             * coordFactors.x,
                               (float(TEXTURE_WAVEFORM_USED_HEIGHT)
                              - pixelEncoded * float(TEXTURE_WAVEFORM_USED_HEIGHT))
                             * waveformSizeFactor.y + 0.5f);

    const float waveformEncoded = 1.f
                                - (ceil(float(coords.y)
                                      / waveformSizeFactor.y)
                                 / float(TEXTURE_WAVEFORM_USED_HEIGHT));

#ifdef IS_HDR_CSP
    float3 waveformColour = WaveformRgbValues(Csp::Trc::PqTo::Nits(waveformEncoded));
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    float3 waveformColour = WaveformRgbValues(ENCODE_SDR(waveformEncoded));
#endif
    waveformColour = sqrt(waveformColour);

    tex2Dstore(StorageWaveform,
               coords,
               float4(waveformColour, 1.f));

    return;
  }
  else
  BRANCH()
  if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL)
  {
    float pixelCll;
    float pixelEncoded;

    float3 pixel = tex2Dfetch(SamplerBackBuffer, FetchPos).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    float3 pixelOpticalBt2020 = CalcCll(pixel);

    pixelCll = MAXRGB(pixelOpticalBt2020);

    pixelEncoded = Csp::Trc::NitsTo::Pq(pixelCll);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    pixelEncoded = MAXRGB(pixel);

    pixelCll = FetchFromHdr10ToLinearLUT(pixelEncoded) * 10000.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    pixelEncoded = MAXRGB(pixel);

    pixelCll = DECODE_SDR(pixelEncoded) * 100.f;

#endif

    const int2 coords = float2(float(FetchPos.x)
                             * coordFactors.x,
                               (float(TEXTURE_WAVEFORM_USED_HEIGHT)
                              - pixelEncoded * float(TEXTURE_WAVEFORM_USED_HEIGHT))
                             * waveformSizeFactor.y + 0.5f);

    const float waveformEncoded = 1.f
                                - (ceil(float(coords.y)
                                      / waveformSizeFactor.y)
                                 / float(TEXTURE_WAVEFORM_USED_HEIGHT));

#ifdef IS_HDR_CSP
    float3 waveformColour = WaveformRgbValues(Csp::Trc::PqTo::Nits(waveformEncoded));
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    float3 waveformColour = WaveformRgbValues(ENCODE_SDR(waveformEncoded));
#endif
    waveformColour = sqrt(waveformColour);

    tex2Dstore(StorageWaveform,
               coords,
               float4(waveformColour, 1.f));

    return;
  }
  else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
  {
    float3 pixelEncoded;
    float3 pixelRgb;
    float3 waveformColour;

    float waveformColourRG;
    float waveformColourBG;

#ifdef IS_HDR_CSP

    #if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      pixelRgb = CalcCll(tex2Dfetch(SamplerBackBuffer, FetchPos).rgb);

      pixelEncoded = Csp::Trc::NitsTo::Pq(pixelRgb);

    //this is more performant to do
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      pixelEncoded = tex2Dfetch(SamplerBackBuffer, FetchPos).rgb;

      pixelRgb = CalcCll(pixelEncoded);

    #endif

    waveformColour  = pixelRgb - 100.f;
    waveformColour  = max(waveformColour, 0.f);
    waveformColour += 600.f;
    waveformColour /= 10500.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    //this is more performant to do
    pixelEncoded = tex2Dfetch(SamplerBackBuffer, FetchPos).rgb;

    pixelRgb = DECODE_SDR(pixelEncoded);

    waveformColour  = pixelRgb - 0.1f;
    waveformColour  = max(waveformColour, 0.f);
    waveformColour += 0.1f;

#endif

    waveformColour.r *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][0];
    waveformColour.b *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][2];

    waveformColourRG = (waveformColour.r - 1.f) * Csp::Mat::Bt709ToXYZ[1][0] / Csp::Mat::Bt709ToXYZ[1][1];
    waveformColourBG = (waveformColour.b - 1.f) * Csp::Mat::Bt709ToXYZ[1][2] / Csp::Mat::Bt709ToXYZ[1][1];

    waveformColour = floor(waveformColour
                         * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                         * waveformSizeFactor.y)
                   / waveformSizeFactor.y
                   / float(TEXTURE_WAVEFORM_USED_HEIGHT);

    waveformColourRG = floor(waveformColourRG
                           * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                           * waveformSizeFactor.y)
                     / waveformSizeFactor.y
                     / float(TEXTURE_WAVEFORM_USED_HEIGHT);

    waveformColourBG = floor(waveformColourBG
                           * float(TEXTURE_WAVEFORM_USED_HEIGHT)
                           * waveformSizeFactor.y)
                     / waveformSizeFactor.y
                     / float(TEXTURE_WAVEFORM_USED_HEIGHT);

    waveformColour = sqrt(waveformColour);

    waveformColourRG = sqrt(waveformColourRG);
    waveformColourBG = sqrt(waveformColourBG);

    float normalisedX = float(FetchPos.x)
                      / BUFFER_WIDTH_MINUS_1_FLOAT;

    Waveform::SWaveformData waveDat = Waveform::GetData();

    static const uint currentWaveformWidth = waveDat.waveformArea.x;

    static const uint textureWaveformWidthDiv3 = currentWaveformWidth / 3u;

    int xCoord0 = normalisedX * float(textureWaveformWidthDiv3 - 1u);

    int xCoord1 = xCoord0 + textureWaveformWidthDiv3;
    int xCoord2 = xCoord1 + textureWaveformWidthDiv3;

    int3 yCoords = (float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  - pixelEncoded * float(TEXTURE_WAVEFORM_USED_HEIGHT))
                 * waveformSizeFactor.y + 0.5f;

    tex2Dstore(StorageWaveform,
               int2(xCoord0, yCoords[0]),
               float4(waveformColour.r, waveformColourRG, 0.f, 1.f));

    tex2Dstore(StorageWaveform,
               int2(xCoord1, yCoords[1]),
               float4(0.f, waveformColour.g, 0.f, 1.f));

    tex2Dstore(StorageWaveform,
               int2(xCoord2, yCoords[2]),
               float4(0.f, waveformColourBG, waveformColour.b, 1.f));

    return;
  }
}


void RenderWaveformScale()
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
    for (int x = 0; x < TEXTURE_WAVEFORM_SCALE_WIDTH; x++)
    {
      [loop]
      for (int y = 0; y < TEXTURE_WAVEFORM_SCALE_HEIGHT; y++)
      {
        tex2Dstore(StorageWaveformScale, int2(x, y), float4(0.f, 0.f, 0.f, 0.f));
      }
    }

    memoryBarrier();

    Waveform::SWaveformData waveDat = Waveform::GetData();

#ifdef IS_HDR_CSP

    int2 nitsOffsets[16];

    [loop]
    for (uint i = 0u; i < 16u; i++)
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
    for (uint i = 0u; i < charListsCount; i++)
    {
      uint2 currentNumber;

      int2 currentTextOffset;

      int currentCharOffset;

      bool needsDrawing = true;


      [loop]
      for (int j = 0u; j < 8u; j++)
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
          Waveform::DrawCharToScale(currentNumber,
                                    waveDat.charDimensions,
                                    currentTextOffset,
                                    charOffsets[j + currentCharOffset]);
        }
      }



    }

#else

    int2 nitsOffsets[14];

    [loop]
    for (uint i = 0u; i < 14u; i++)
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
    for (uint i = 0; i < charListsCount; i++)
    {
      uint2 currentNumber;

      int2 currentTextOffset;

      int currentCharOffset;

      [loop]
      for (int j = 0; j < 7; j++)
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
          Waveform::DrawCharToScale(currentNumber,
                                    charDims,
                                    currentTextOffset,
                                    charOffsets[j + currentCharOffset]);
        }
      }
    }

#endif

    // draw the frame, ticks and horizontal lines
    [loop]
    for (int y = 0; y < waveDat.endXY.y; y++)
    {
      int2 curPos = waveDat.offsetToFrame
                  + int2(0, y);

      float curGrey = lerp(0.5f, 0.4f, (float(y + WAVEDAT_CUTOFF_OFFSET) / float(waveDat.endYminus1 + WAVEDAT_CUTOFF_OFFSET)));
      curGrey = pow(curGrey, 2.2f);
      // using gamma 2 as intermediate gamma space
      curGrey = sqrt(curGrey);

      float4 curColour = float4(curGrey, 1.f.xxx);

      // draw top and bottom part of the frame
      [branch]
      if (y <  waveDat.frameSize
       || y >= waveDat.lowerFrameStart)
      {
        [loop]
        for (int x = 0; x < waveDat.endXY.x; x++)
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
        for (int x = 0; x < waveDat.frameSize; x++)
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
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.frameSize); x++)
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
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.endXY.x); x++)
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
#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
                                                ,
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

  WaveDat0    =  0;
  MinRLineY   =  INT_MAX;
  MinGLineY   =  INT_MAX;
  MinBLineY   =  INT_MAX;
  MaxRLineY   = -INT_MAX;
  MaxGLineY   = -INT_MAX;
  MaxBLineY   = -INT_MAX;
  GLinePartX  = -INT_MAX;
  BLinePartX  = -INT_MAX;

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)

  #define WaveformSizeYFactor WaveDat4

  WaveformSizeYFactor = 0.f;
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
    WaveformCutoffOffset = WAVEDAT_CUTOFF_OFFSET;
#endif

    float waveformSizeY = _WAVEFORM_SIZE.y / 100.f;

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
    if (_WAVEFORM_SHOW_MIN_NITS_LINE)
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
      if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL)
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
    }

    BRANCH()
    if (_WAVEFORM_SHOW_MAX_NITS_LINE)
    {
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
      if (_WAVEFORM_MODE == WAVEFORM_MODE_MAX_CLL)
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

    float2 scaleColour = tex2Dfetch(SamplerWaveformScale, scaleCoords).rg;

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
#ifdef IS_QHD_OR_HIGHER_RES
      if (waveformCoords.y == minLineY
       || waveformCoords.y == minLineY - 1)
#else
      if (waveformCoords.y == minLineY)
#endif
      {
        Out = float4(1.f, 1.f, 1.f, 1.f);
        return;
      }
      else
      [branch]
#ifdef IS_QHD_OR_HIGHER_RES
      if (waveformCoords.y == maxLineY
       || waveformCoords.y == maxLineY + 1)
#else
      if (waveformCoords.y == maxLineY)
#endif
      {
        Out = float4(1.f, 1.f, 0.f, 1.f);
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

        const bool showMaxLineActive = waveformCoordsGTEMaxLine && _WAVEFORM_SHOW_MAX_NITS_LINE;
        const bool showMinLineActive = waveformCoordsSTEMinLine && _WAVEFORM_SHOW_MIN_NITS_LINE;

        [branch]
        if (( showMaxLineActive            &&  showMinLineActive)
         || (!_WAVEFORM_SHOW_MAX_NITS_LINE &&  showMinLineActive)
         || ( showMaxLineActive            && !_WAVEFORM_SHOW_MIN_NITS_LINE)
         || (!_WAVEFORM_SHOW_MAX_NITS_LINE && !_WAVEFORM_SHOW_MIN_NITS_LINE))
        {
          int2 waveformFetchCoords;

          waveformFetchCoords.x = waveformCoords.x;
#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)

          waveformFetchCoords.y = int(float(waveformCoords.y + WaveformCutoffOffset)
                                          * WaveformSizeYFactor);
#else
          waveformFetchCoords.y = waveformCoords.y + WaveformCutoffOffset;
#endif

          // using gamma 2 as intermediate gamma space
          scaleColour.r *= scaleColour.r;

          float4 waveformColour = tex2Dfetch(SamplerWaveform, waveformFetchCoords);
          // using gamma 2 as intermediate gamma space
          waveformColour.rgb *= waveformColour.rgb;

          float4 colourOut = scaleColour.rrrg
                           + waveformColour;

          // using gamma 2 as intermediate gamma space
          colourOut.rgb = sqrt(colourOut.rgb);

          Out = colourOut;
          return;
        }
        else
        {
          Out = scaleColour.rrrg;
          return;
        }
      }
    }
    else
    {
      Out = scaleColour.rrrg;
      return;
    }
  }
  else
  {
    discard;
  }
}
