#pragma once


static const float TEXTURE_WAVEFORM_BUFFER_WIDTH_FACTOR = BUFFER_WIDTH_FLOAT
                                                        / float(TEXTURE_WAVEFORM_WIDTH);

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
  Format = RGBA8;
};

sampler2D<float4> SamplerWaveform
{
  Texture = TextureWaveform;
  MagFilter = POINT;
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
  Format = RGBA8;
};

sampler2D<float4> SamplerWaveformFinal
{
  Texture   = TextureWaveformFinal;
  MagFilter = POINT;
};


void RenderWaveform(
  const int2 FetchPos)
{
  BRANCH(x)
  if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
  {
    float curPixelNits = tex2Dfetch(StorageNitsValues, FetchPos).w;

#ifdef IS_HDR_CSP
    float encodedPixel = Csp::Trc::NitsTo::Pq(curPixelNits);
#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)
    float encodedPixel = ENCODE_SDR(curPixelNits / 100.f);
#endif

    int2 coord = float2(float(FetchPos.x)
                    / TEXTURE_WAVEFORM_BUFFER_WIDTH_FACTOR,
                      float(TEXTURE_WAVEFORM_USED_HEIGHT)
                    - (encodedPixel * float(TEXTURE_WAVEFORM_USED_HEIGHT))) + 0.5f;

    float3 waveformColour = WaveformRgbValues(curPixelNits);
    waveformColour = sqrt(waveformColour);

    tex2Dstore(StorageWaveform,
               coord,
               float4(waveformColour, 1.f));

    return;
  }
  else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
  {
    float3 encodedPixel;
    float3 waveformColour;

#ifdef IS_HDR_CSP

    float3 curPixelRgb;

    #if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      curPixelRgb = tex2Dfetch(StorageNitsValues, FetchPos).rgb;

      encodedPixel = Csp::Trc::NitsTo::Pq(curPixelRgb);

    //this is more performant to do
    #elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      encodedPixel = tex2Dfetch(SamplerBackBuffer, FetchPos).rgb;

      curPixelRgb = Csp::Trc::PqTo::Nits(encodedPixel);

    #endif

    waveformColour  = curPixelRgb - 100.f;
    waveformColour  = max(waveformColour, 0.f);
    waveformColour += 600.f;
    waveformColour /= 10500.f;

    waveformColour.r *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][0];
    waveformColour.b *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][2];

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    //this is more performant to do
    encodedPixel = tex2Dfetch(SamplerBackBuffer, FetchPos).rgb;

    waveformColour  = DECODE_SDR(encodedPixel) - 10.f;
    waveformColour  = max(waveformColour, 0.f);
    waveformColour += 60.f;
    waveformColour /= 150.f;

    waveformColour.r *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][0];
    waveformColour.b *= Csp::Mat::Bt709ToXYZ[1][1] / Csp::Mat::Bt709ToXYZ[1][2];

#endif

    waveformColour = sqrt(waveformColour);

    int xCoord0 = float(FetchPos.x)
                / float(TEXTURE_WAVEFORM_BUFFER_WIDTH_FACTOR)
                / 3.f;

    int xCoord1 = xCoord0 + (TEXTURE_WAVEFORM_WIDTH / 3);
    int xCoord2 = xCoord1 + (TEXTURE_WAVEFORM_WIDTH / 3);

    int3 yCoords = (float(TEXTURE_WAVEFORM_USED_HEIGHT)
                  - (encodedPixel * float(TEXTURE_WAVEFORM_USED_HEIGHT)))
                 + 0.5f;

    tex2Dstore(StorageWaveform,
               int2(xCoord0, yCoords[0]),
               float4(waveformColour.r, 0.f, 0.f, 1.f));

    tex2Dstore(StorageWaveform,
               int2(xCoord1, yCoords[1]),
               float4(0.f, waveformColour.g, 0.f, 1.f));

    tex2Dstore(StorageWaveform,
               int2(xCoord2, yCoords[2]),
               float4(0.f, 0.f, waveformColour.b, 1.f));

    return;
  }
}


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

    static const float maxFontSize =
      max(((TEXTURE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f) / 2.f) * 2.f / 32.f * FONT_SIZE_MULTIPLIER, 0.5f);

    const float fontSize =
      clamp(((TEXTURE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f) / 2.f * fontSizeFactor) * 2.f / 32.f * FONT_SIZE_MULTIPLIER, 0.5f, maxFontSize);

#ifndef IS_HDR_CSP
    waveDat.charDimensionXForPercent = WAVEFORM_CHAR_DIM_FLOAT.x * fontSize.x;
#endif

    waveDat.charDimensions = float2(WAVEFORM_CHAR_DIM_FLOAT.x - 1, WAVEFORM_CHAR_DIM_FLOAT.y) * fontSize;

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
    waveformScaleFactorXY.y += 3.f - (1.f - waveformScaleFactorXY.y) * 6.f;
    waveformScaleFactorXY.y  = clamp(waveformScaleFactorXY.y, WAVEFORM_SCALE_FACTOR_CLAMP_MIN, WAVEFORM_SCALE_FACTOR_CLAMP_MAX.y);
#endif

#ifdef IS_HDR_CSP
    static const int cutoffPoints[16] = {
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
      int(                                                                         float(TEXTURE_WAVEFORM_USED_HEIGHT)   * waveformScaleFactorXY.y + 0.5f) };
#else
    waveDat.tickPoints = {
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
      int(                                                              float(TEXTURE_WAVEFORM_USED_HEIGHT)   * waveformScaleFactorXY.y + 0.5f) };
#endif

    waveDat.waveformArea =
      int2(TEXTURE_WAVEFORM_WIDTH * waveformScaleFactorXY.x,
#ifdef IS_HDR_CSP
           cutoffPoints[15] - cutoffPoints[WAVEFORM_CUTOFF_POINT]
#else
           waveDat.tickPoints[13]
#endif
           );

#ifdef IS_HDR_CSP
    if (WAVEFORM_CUTOFF_POINT == 0)
    {
      waveDat.cutoffOffset = 0;

      waveDat.tickPoints = {
        int(0),
        int(cutoffPoints[ 1]),
        int(cutoffPoints[ 2]),
        int(cutoffPoints[ 3]),
        int(cutoffPoints[ 4]),
        int(cutoffPoints[ 5]),
        int(cutoffPoints[ 6]),
        int(cutoffPoints[ 7]),
        int(cutoffPoints[ 8]),
        int(cutoffPoints[ 9]),
        int(cutoffPoints[10]),
        int(cutoffPoints[11]),
        int(cutoffPoints[12]),
        int(cutoffPoints[13]),
        int(cutoffPoints[14]),
        int(cutoffPoints[15]) };
    }
    else if (WAVEFORM_CUTOFF_POINT == 1)
    {
      waveDat.cutoffOffset = cutoffPoints[1];

      waveDat.tickPoints = {
        int(-100),
        int(0),
        int(cutoffPoints[ 2] - waveDat.cutoffOffset),
        int(cutoffPoints[ 3] - waveDat.cutoffOffset),
        int(cutoffPoints[ 4] - waveDat.cutoffOffset),
        int(cutoffPoints[ 5] - waveDat.cutoffOffset),
        int(cutoffPoints[ 6] - waveDat.cutoffOffset),
        int(cutoffPoints[ 7] - waveDat.cutoffOffset),
        int(cutoffPoints[ 8] - waveDat.cutoffOffset),
        int(cutoffPoints[ 9] - waveDat.cutoffOffset),
        int(cutoffPoints[10] - waveDat.cutoffOffset),
        int(cutoffPoints[11] - waveDat.cutoffOffset),
        int(cutoffPoints[12] - waveDat.cutoffOffset),
        int(cutoffPoints[13] - waveDat.cutoffOffset),
        int(cutoffPoints[14] - waveDat.cutoffOffset),
        int(cutoffPoints[15] - waveDat.cutoffOffset) };
    }
    else if (WAVEFORM_CUTOFF_POINT == 2)
    {
      waveDat.cutoffOffset = cutoffPoints[2];

      waveDat.tickPoints = {
        int(-100),
        int(-100),
        int(0),
        int(cutoffPoints[ 3] - waveDat.cutoffOffset),
        int(cutoffPoints[ 4] - waveDat.cutoffOffset),
        int(cutoffPoints[ 5] - waveDat.cutoffOffset),
        int(cutoffPoints[ 6] - waveDat.cutoffOffset),
        int(cutoffPoints[ 7] - waveDat.cutoffOffset),
        int(cutoffPoints[ 8] - waveDat.cutoffOffset),
        int(cutoffPoints[ 9] - waveDat.cutoffOffset),
        int(cutoffPoints[10] - waveDat.cutoffOffset),
        int(cutoffPoints[11] - waveDat.cutoffOffset),
        int(cutoffPoints[12] - waveDat.cutoffOffset),
        int(cutoffPoints[13] - waveDat.cutoffOffset),
        int(cutoffPoints[14] - waveDat.cutoffOffset),
        int(cutoffPoints[15] - waveDat.cutoffOffset) };
    }
    else //if (WAVEFORM_CUTOFF_POINT == 3)
    {
      waveDat.cutoffOffset = cutoffPoints[3];

      waveDat.tickPoints = {
        int(-100),
        int(-100),
        int(-100),
        int(0),
        int(cutoffPoints[ 4] - waveDat.cutoffOffset),
        int(cutoffPoints[ 5] - waveDat.cutoffOffset),
        int(cutoffPoints[ 6] - waveDat.cutoffOffset),
        int(cutoffPoints[ 7] - waveDat.cutoffOffset),
        int(cutoffPoints[ 8] - waveDat.cutoffOffset),
        int(cutoffPoints[ 9] - waveDat.cutoffOffset),
        int(cutoffPoints[10] - waveDat.cutoffOffset),
        int(cutoffPoints[11] - waveDat.cutoffOffset),
        int(cutoffPoints[12] - waveDat.cutoffOffset),
        int(cutoffPoints[13] - waveDat.cutoffOffset),
        int(cutoffPoints[14] - waveDat.cutoffOffset),
        int(cutoffPoints[15] - waveDat.cutoffOffset) };
    }
#endif

    waveDat.textOffset = int2(0, int(float(waveDat.charDimensions.y) / 2.f + 0.5f));

    waveDat.tickXOffset = waveDat.borderSize
                        + textWidth
                        + tickSpacer;

    waveDat.lowerFrameStart = waveDat.frameSize
                            + waveDat.waveformArea.y;

    waveDat.endXY = waveDat.frameSize * 2
                  + waveDat.waveformArea;

    waveDat.endYminus1 = waveDat.endXY.y - 1;

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

  int2 GetNitsOffset(
    const int ActiveBorderSize,
    const int ActiveFrameSize,
    const int ActiveFontSpacer,
    const int YOffset)
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

  void DrawCharToScale(
    const uint   Char,
    const float2 CharDim,
    const int2   Pos,
    const int    CharCount)
  {
    const float2 charFetchPos = float2(WAVEFORM_ATLAS_OFFSET.x + (Char * WAVEFORM_CHAR_DIM_UINT.x),
                                       WAVEFORM_ATLAS_OFFSET.y);

    const int2 currentDrawPos = Pos + int2(CharCount * CharDim.x, 0);

    const int2 ceilCharDim = ceil(CharDim);

    int2 currentOffset = int2(0, 0);

    [loop]
    while (currentOffset.x < ceilCharDim.x)
    {
#ifndef NO_WORKAROUND_NEEDED
      currentOffset.x += floor(FRAMETIME / 100000.f + 0.1f);
#endif
      [loop]
      while (currentOffset.y < ceilCharDim.y)
      {
#ifndef NO_WORKAROUND_NEEDED
        currentOffset.y += floor(FRAMETIME / 100000.f);
#endif
        float2 currentSamplePos = charFetchPos
                                + float2(currentOffset) * (min(WAVEFORM_CHAR_DIM_FLOAT / CharDim, 2.f)) + 0.5f;

        float2 fract = frac(currentSamplePos);

        int2 currentFetchPos = floor(currentSamplePos);

        float4 a = tex2Dfetch(SamplerFontAtlasConsolidated, currentFetchPos);
        float4 b = tex2Dfetch(SamplerFontAtlasConsolidated, int2(currentFetchPos.x + 1, currentFetchPos.y));
        float4 c = tex2Dfetch(SamplerFontAtlasConsolidated, int2(currentFetchPos.x,     currentFetchPos.y + 1));
        float4 d = tex2Dfetch(SamplerFontAtlasConsolidated, currentFetchPos + 1);

        float4 mtsdf = lerp(lerp(a, b, fract.x),
                            lerp(c, d, fract.x),
                            fract.y);


        const float sd = GetMedian(mtsdf.rgb);

        const float screenPixelDistance = GetScreenPixelRange(CharDim.x / WAVEFORM_CHAR_DIM_FLOAT.x, WAVEFORM_RANGE)
                                        * (sd - 0.5f);

        const float opacity = saturate(screenPixelDistance + 0.5f);

        const float outline = smoothstep(0.f, 0.1f, (opacity + mtsdf.a) / 2.f);

        //float test = lerp(0.f, 0.5f, opacity);

        float test = lerp(1.f, 0.f, 1 - outline);
        test = lerp(test - 1, 0.5f, opacity);

        //float alpha = test;

        int2 currentDrawOffset = currentDrawPos + currentOffset;

        tex2Dstore(StorageWaveformScale, currentDrawOffset, float4(sqrt(test), test, test, test));

        currentOffset.y++;
      }
      currentOffset.x++;
      currentOffset.y = 0;
    }

    return;
  } //DrawCharToScale

}


void RenderWaveformScale()
{
  BRANCH(x)
  if (tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_X)       != _WAVEFORM_SIZE.x
   || tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_Y)       != _WAVEFORM_SIZE.y
#ifdef IS_HDR_CSP
   || tex1Dfetch(StorageConsolidated, COORDS_WAVEFORM_LAST_CUTOFF_POINT) != WAVEFORM_CUTOFF_POINT
#endif
  )
  {
    //make background all black
    for (int x = 0; x < TEXTURE_WAVEFORM_SCALE_WIDTH; x++)
    {
      for (int y = 0; y < TEXTURE_WAVEFORM_SCALE_HEIGHT; y++)
      {
        tex2Dstore(StorageWaveformScale, int2(x, y), float4(0.f, 0.f, 0.f, 0.f));
      }
    }

    Waveform::SWaveformData waveDat = Waveform::GetData();

#ifdef IS_HDR_CSP

    const int2 nits10000_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 0]);
    const int2 nits_4000_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 1]);
    const int2 nits_2000_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 2]);
    const int2 nits_1000_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 3]);
    const int2 nits__400_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 4]);
    const int2 nits__203_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 5]);
    const int2 nits__100_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 6]);
    const int2 nits___50_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 7]);
    const int2 nits___25_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 8]);
    const int2 nits___10_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 9]);
    const int2 nits____5_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[10]);
    const int2 nits____2_50Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[11]);
    const int2 nits____1_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[12]);
    const int2 nits____0_25Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[13]);
    const int2 nits____0_05Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[14]);
    const int2 nits____0_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[15]);


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

    if (WAVEFORM_CUTOFF_POINT == 0)
    {
      Waveform::DrawCharToScale(  _1_w, waveDat.charDimensions, text10000_00Offset, charOffsets[0]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[1]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[2]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[3]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[4]);
      Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text10000_00Offset, charOffsets[5]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[6]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text10000_00Offset, charOffsets[7]);
    }

    if (WAVEFORM_CUTOFF_POINT <= 1)
    {
      Waveform::DrawCharToScale(  _4_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[1]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[2]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[3]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[4]);
      Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[5]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[6]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_4000_00Offset, charOffsets[7]);
    }

    if (WAVEFORM_CUTOFF_POINT <= 2)
    {
      Waveform::DrawCharToScale(  _2_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[1]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[2]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[3]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[4]);
      Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[5]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[6]);
      Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_2000_00Offset, charOffsets[7]);
    }

    Waveform::DrawCharToScale(  _1_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[1]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[2]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text_1000_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _4_w, waveDat.charDimensions, text__400_00Offset, charOffsets[2]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__400_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__400_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text__400_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__400_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__400_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _2_w, waveDat.charDimensions, text__203_00Offset, charOffsets[2]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__203_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _3_w, waveDat.charDimensions, text__203_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text__203_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__203_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__203_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _1_w, waveDat.charDimensions, text__100_00Offset, charOffsets[2]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__100_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__100_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text__100_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__100_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text__100_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text___50_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___50_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text___50_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___50_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___50_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _2_w, waveDat.charDimensions, text___25_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text___25_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text___25_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___25_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___25_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _1_w, waveDat.charDimensions, text___10_00Offset, charOffsets[3]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___10_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text___10_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___10_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text___10_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text____5_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____5_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____5_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____5_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _2_w, waveDat.charDimensions, text____2_50Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____2_50Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text____2_50Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____2_50Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _1_w, waveDat.charDimensions, text____1_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____1_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____1_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____1_00Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_25Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____0_25Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _2_w, waveDat.charDimensions, text____0_25Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text____0_25Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_05Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____0_05Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_05Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _5_w, waveDat.charDimensions, text____0_05Offset, charOffsets[7]);

    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_00Offset, charOffsets[4]);
    Waveform::DrawCharToScale(_dot_w, waveDat.charDimensions, text____0_00Offset, charOffsets[5]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_00Offset, charOffsets[6]);
    Waveform::DrawCharToScale(  _0_w, waveDat.charDimensions, text____0_00Offset, charOffsets[7]);

#else

    const int2 nits100_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 0]);
    const int2 nits_87_50Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 1]);
    const int2 nits_75_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 2]);
    const int2 nits_60_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 3]);
    const int2 nits_50_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 4]);
    const int2 nits_35_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 5]);
    const int2 nits_25_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 6]);
    const int2 nits_18_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 7]);
    const int2 nits_10_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 8]);
    const int2 nits__5_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[ 9]);
    const int2 nits__2_50Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[10]);
    const int2 nits__1_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[11]);
#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)
    const int2 nits__0_25Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[12]);
#else
    const int2 nits__0_40Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[12]);
#endif
    const int2 nits__0_00Offset = Waveform::GetNitsOffset(waveDat.borderSize, waveDat.frameSize, waveDat.fontSpacer, waveDat.tickPoints[13]);

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

    const float2 charDimensionsForPercent = float2(waveDat.charDimensionXForPercent, waveDat.charDimensions.y);

    Waveform::DrawCharToScale(      _1_w, waveDat.charDimensions, text100_00Offset, 0);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text100_00Offset, 1);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text100_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text100_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text100_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text100_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text100_00Offset, 6);

    Waveform::DrawCharToScale(      _8_w, waveDat.charDimensions, text_87_50Offset, 1);
    Waveform::DrawCharToScale(      _7_w, waveDat.charDimensions, text_87_50Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_87_50Offset, 3);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text_87_50Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_87_50Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_87_50Offset, 6);

    Waveform::DrawCharToScale(      _7_w, waveDat.charDimensions, text_75_00Offset, 1);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text_75_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_75_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_75_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_75_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_75_00Offset, 6);

    Waveform::DrawCharToScale(      _6_w, waveDat.charDimensions, text_60_00Offset, 1);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_60_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_60_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_60_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_60_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_60_00Offset, 6);

    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text_50_00Offset, 1);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_50_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_50_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_50_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_50_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_50_00Offset, 6);

    Waveform::DrawCharToScale(      _3_w, waveDat.charDimensions, text_35_00Offset, 1);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text_35_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_35_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_35_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_35_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_35_00Offset, 6);

    Waveform::DrawCharToScale(      _2_w, waveDat.charDimensions, text_25_00Offset, 1);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text_25_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_25_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_25_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_25_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_25_00Offset, 6);

    Waveform::DrawCharToScale(      _1_w, waveDat.charDimensions, text_18_00Offset, 1);
    Waveform::DrawCharToScale(      _8_w, waveDat.charDimensions, text_18_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_18_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_18_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_18_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_18_00Offset, 6);

    Waveform::DrawCharToScale(      _1_w, waveDat.charDimensions, text_10_00Offset, 1);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_10_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text_10_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_10_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text_10_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text_10_00Offset, 6);

    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text__5_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__5_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__5_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__5_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__5_00Offset, 6);

    Waveform::DrawCharToScale(      _2_w, waveDat.charDimensions, text__2_50Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__2_50Offset, 3);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text__2_50Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__2_50Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__2_50Offset, 6);

    Waveform::DrawCharToScale(      _1_w, waveDat.charDimensions, text__1_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__1_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__1_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__1_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__1_00Offset, 6);

#if (OVERWRITE_SDR_GAMMA == GAMMA_UNSET \
  || OVERWRITE_SDR_GAMMA == GAMMA_22    \
  || OVERWRITE_SDR_GAMMA == GAMMA_24)

    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_25Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__0_25Offset, 3);
    Waveform::DrawCharToScale(      _2_w, waveDat.charDimensions, text__0_25Offset, 4);
    Waveform::DrawCharToScale(      _5_w, waveDat.charDimensions, text__0_25Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__0_25Offset, 6);
#else
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_40Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__0_40Offset, 3);
    Waveform::DrawCharToScale(      _4_w, waveDat.charDimensions, text__0_40Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_40Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__0_40Offset, 6);
#endif

    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_00Offset, 2);
    Waveform::DrawCharToScale(    _dot_w, waveDat.charDimensions, text__0_00Offset, 3);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_00Offset, 4);
    Waveform::DrawCharToScale(      _0_w, waveDat.charDimensions, text__0_00Offset, 5);
    Waveform::DrawCharToScale(_percent_w, charDimensionsForPercent, text__0_00Offset, 6);

#endif

    // draw the frame, ticks and horizontal lines
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
      if (y <  waveDat.frameSize
       || y >= waveDat.lowerFrameStart)
      {
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
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.frameSize); x++)
        {
          int2 curTickPos = int2(x,
                                 curPos.y);
          tex2Dstore(StorageWaveformScale, curTickPos, curColour);
        }
      }

      // draw ticks + draw horizontal lines
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
        for (int x = waveDat.tickXOffset; x < (waveDat.tickXOffset + waveDat.endXY.x); x++)
        {
          int2 curTickPos = int2(x,
                                 curPos.y);
          tex2Dstore(StorageWaveformScale, curTickPos, curColour);
        }
      }
    }

    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_X,       _WAVEFORM_SIZE.x);
    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_SIZE_Y,       _WAVEFORM_SIZE.y);
#ifdef IS_HDR_CSP
    tex1Dstore(StorageConsolidated, COORDS_WAVEFORM_LAST_CUTOFF_POINT, WAVEFORM_CUTOFF_POINT);
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
void VS_PrepareRenderWaveformToScale(
  in                  uint   VertexID : SV_VertexID,
  out                 float4 Position : SV_Position,
  out                 float2 TexCoord : TEXCOORD0,
  out nointerpolation int4   WaveDat0 : WaveDat0,
  out nointerpolation int4   WaveDat1 : WaveDat1,
  out nointerpolation int4   WaveDat2 : WaveDat2
#ifdef IS_HDR_CSP
                                                ,
  out nointerpolation int    WaveDat3 : WaveDat3
#endif
  )
{
  TexCoord.x = (VertexID == 2) ? 2.f
                               : 0.f;
  TexCoord.y = (VertexID == 1) ? 2.f
                               : 0.f;
  Position = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

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

#ifdef IS_HDR_CSP
  #define WaveformCutoffOffset WaveDat3

  WaveformCutoffOffset = 0;
#else
  #define WaveformCutoffOffset 0
#endif

  BRANCH(x)
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

    BRANCH(x)
    if (_WAVEFORM_SHOW_MIN_NITS_LINE)
    {
      BRANCH(x)
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
      {
        const float minNits = tex1Dfetch(SamplerConsolidated, COORDS_MIN_NITS_VALUE);

        [branch]
        if (minNits > 0.f
         && minNits < MAX_NITS_LINE_CUTOFF)
        {
          MinNitsLineY = GetNitsLine(minNits, waveformScaleFactorY
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
          MinRLineY = GetNitsLine(minRgb.r, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMinRgbLine.g)
        {
          MinGLineY = GetNitsLine(minRgb.g, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMinRgbLine.b)
        {
          MinBLineY = GetNitsLine(minRgb.b, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
      }
    }

    BRANCH(x)
    if (_WAVEFORM_SHOW_MAX_NITS_LINE)
    {
      BRANCH(x)
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
      {
        const float maxNits = tex1Dfetch(SamplerConsolidated, COORDS_MAX_NITS_VALUE);

        [branch]
        if (maxNits > 0.f
         && maxNits < MAX_NITS_LINE_CUTOFF)
        {
          MaxNitsLineY = GetNitsLine(maxNits, waveformScaleFactorY
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
          MaxRLineY = GetNitsLine(maxRgb.r, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMaxRgbLine.g)
        {
          MaxGLineY = GetNitsLine(maxRgb.g, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
        [branch]
        if (drawMaxRgbLine.b)
        {
          MaxBLineY = GetNitsLine(maxRgb.b, waveformScaleFactorY
#ifdef IS_HDR_CSP
                                                                , waveDat.cutoffOffset
#endif
                                 );
        }
      }
    }

    BRANCH(x)
    if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
    {
      GLinePartX = uint(waveDat.waveformArea.x) / 3u;
      BLinePartX = GLinePartX + GLinePartX;
    }
  }
}

void PS_RenderWaveformToScale(
  in                  float4 Position : SV_Position,
  in                  float2 TexCoord : TEXCOORD0,
  in  nointerpolation int4   WaveDat0 : WaveDat0,
  in  nointerpolation int4   WaveDat1 : WaveDat1,
  in  nointerpolation int4   WaveDat2 : WaveDat2,
#ifdef IS_HDR_CSP
  in  nointerpolation int    WaveDat3 : WaveDat3,
#endif
  out                 float4 Out      : SV_Target0)
{
  Out = 0.f;

  BRANCH(x)
  if (_SHOW_WAVEFORM)
  {
    const int2 pureCoordAsInt = int2(Position.xy);

    const int2 scaleCoords = pureCoordAsInt;

    const int2 waveformCoords = pureCoordAsInt - OffsetToWaveformArea;

    if (all(waveformCoords >= 0)
     && all(waveformCoords < WaveformActiveArea))
    {
      static const bool isRPart = waveformCoords.x  < GLinePartX;
      static const bool isGPart = waveformCoords.x  < BLinePartX;
      static const bool isBPart = waveformCoords.x >= BLinePartX;

      int minLineY;
      int maxLineY;

      BRANCH(x)
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
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
        else [flatten] if (isGPart)
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

      bool waveformCoordsGTEMaxLine;
      bool waveformCoordsSTEMinLine;

      BRANCH(x)
      if (_WAVEFORM_MODE == WAVEFORM_MODE_LUMINANCE)
      {
        waveformCoordsGTEMaxLine = waveformCoords.y >= MaxNitsLineY;
        waveformCoordsSTEMinLine = waveformCoords.y <= MinNitsLineY;
      }
      else //if (_WAVEFORM_MODE == WAVEFORM_MODE_RGB_INDIVIDUALLY)
      {
        if (isRPart)
        {
          waveformCoordsGTEMaxLine = waveformCoords.y >= MaxRLineY;
          waveformCoordsSTEMinLine = waveformCoords.y <= MinRLineY;
        }
        else if (isGPart)
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

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
  #define WAVEFORM_SAMPLER_CLAMP_MIN 50.f
  #define WAVEFORM_SAMPLER_CLAMP_MAX float2(100.f, 400.f)
#else
  #define WAVEFORM_SAMPLER_CLAMP_MIN  50.f
  #define WAVEFORM_SAMPLER_CLAMP_MAX 100.f
#endif

      BRANCH(x)
      if (( showMaxLineActive            &&  showMinLineActive)
       || (!_WAVEFORM_SHOW_MAX_NITS_LINE &&  showMinLineActive)
       || ( showMaxLineActive            && !_WAVEFORM_SHOW_MIN_NITS_LINE)
       || (!_WAVEFORM_SHOW_MAX_NITS_LINE && !_WAVEFORM_SHOW_MIN_NITS_LINE))
      {
        float2 waveformSize = _WAVEFORM_SIZE;

#if (!defined(IS_HDR_CSP) \
  && BUFFER_COLOR_BIT_DEPTH != 10)
        waveformSize.y += 300.f - (100.f - waveformSize.y) * 6.f;
#endif

        waveformSize = clamp(waveformSize, WAVEFORM_SAMPLER_CLAMP_MIN, WAVEFORM_SAMPLER_CLAMP_MAX);

        float2 waveformSamplerCoords = (float2(waveformCoords + int2(0, WaveformCutoffOffset)) + 0.5f)
                                     * (100.f / waveformSize)
                                     / float2(TEXTURE_WAVEFORM_WIDTH, TEXTURE_WAVEFORM_HEIGHT);

        float2 scaleColour = tex2Dfetch(SamplerWaveformScale, scaleCoords).rg;
        // using gamma 2 as intermediate gamma space
        scaleColour.r *= scaleColour.r;

        float4 waveformColour = tex2D(SamplerWaveform, waveformSamplerCoords);
        // using gamma 2 as intermediate gamma space
        waveformColour.rgb *= waveformColour.rgb;

        Out = scaleColour.rrrg
            + waveformColour;

        // using gamma 2 as intermediate gamma space
        Out.rgb = sqrt(Out.rgb);
        return;
      }
    }
    //else
    Out = tex2Dfetch(SamplerWaveformScale, scaleCoords).rrrg;
    return;
  }
  discard;
}
