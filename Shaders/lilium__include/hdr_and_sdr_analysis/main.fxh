#pragma once


#if defined(IS_HDR_COMPATIBLE_API)


#include "../draw_font.fxh"


#define WAVE64_THREAD_SIZE_X 8
#define WAVE64_THREAD_SIZE_Y 8

#if (BUFFER_WIDTH % 8 == 0)
  #define WAVE64_DISPATCH_X (BUFFER_WIDTH / 8)
#else
  #define WAVE64_FETCH_X_NEEDS_CLAMPING
  #define WAVE64_DISPATCH_X (BUFFER_WIDTH / 8 + 1)
#endif

#if (BUFFER_HEIGHT % 8 == 0)
  #define WAVE64_DISPATCH_Y (BUFFER_HEIGHT / 8)
#else
  #define WAVE64_FETCH_Y_NEEDS_CLAMPING
  #define WAVE64_DISPATCH_Y (BUFFER_HEIGHT / 8 + 1)
#endif


// 0.0000000894069671630859375 = ((ieee754_half_decode(0x0002)
//                               - ieee754_half_decode(0x0001))
//                              / 2)
//                             + ieee754_half_decode(0x0001)
#define SMALLEST_FP16   asfloat(0x33C00000)
// 0.0014662756584584712982177734375 = 1.5 / 1023
#define SMALLEST_UINT10 asfloat(0x3AC0300C)


static const uint  PixelCountInUint  = BUFFER_WIDTH_UINT
                                     * BUFFER_HEIGHT_UINT;
static const float PixelCountInFloat = PixelCountInUint;


uniform float FRAMETIME
<
  source = "frametime";
>;


#if defined(IS_FLOAT_HDR_CSP)

  #define NITS_AND_CSP_ENTRIES_AMOUNT (4 * 11 + 5 * 6)

  #define TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_WIDTH (3 + 5 + NITS_AND_CSP_ENTRIES_AMOUNT)

  #define   BT709_PERCENTAGE_POS 3
  #define   DCIP3_PERCENTAGE_POS 4
  #define  BT2020_PERCENTAGE_POS 5
  #define     AP0_PERCENTAGE_POS 6
  #define INVALID_PERCENTAGE_POS 7

  #define NITS_NUMBERS_START_POS 8

  #define CSP_PERCENTAGES_START_POS (8 + 11 * 4)

#elif defined(IS_HDR10_LIKE_CSP)

  #define NITS_AND_CSP_ENTRIES_AMOUNT (4 * 11 + 3 * 6)

  #define TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_WIDTH (3 + 3 + NITS_AND_CSP_ENTRIES_AMOUNT)

  #define  BT709_PERCENTAGE_POS 3
  #define  DCIP3_PERCENTAGE_POS 4
  #define BT2020_PERCENTAGE_POS 5

  #define NITS_NUMBERS_START_POS 6

  #define CSP_PERCENTAGES_START_POS (6 + 11 * 4)

#else

  #define NITS_AND_CSP_ENTRIES_AMOUNT (4 * 9)

  #define TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_WIDTH (3     + NITS_AND_CSP_ENTRIES_AMOUNT)

  #define NITS_NUMBERS_START_POS 3

#endif

#define MAX_NITS_POS 0
#define AVG_NITS_POS 1
#define MIN_NITS_POS 2


texture1D TextureMaxAvgMinNitsAndCspCounterAndShowNumbers
{
  Width  = TEXTURE_MAX_AVG_MIN_NITS_AND_CSP_COUNTER_AND_SHOW_NUMBERS_WIDTH;
  Format = R32U;
};

storage1D<uint> StorageMaxAvgMinNitsAndCspCounterAndShowNumbers
{
  Texture = TextureMaxAvgMinNitsAndCspCounterAndShowNumbers;
};

sampler1D<uint> SamplerMaxAvgMinNitsAndCspCounterAndShowNumbers
{
  Texture = TextureMaxAvgMinNitsAndCspCounterAndShowNumbers;
};


#define SHOW_NITS_VALUES_LINE_COUNT      3
#define SHOW_NITS_FROM_CURSOR_LINE_COUNT 1

#if defined(IS_HDR_CSP)
  #define SHOW_CSP_FROM_CURSOR_LINE_COUNT 1
#else
  #define SHOW_CSP_FROM_CURSOR_LINE_COUNT 0
#endif

#if defined(IS_HDR10_LIKE_CSP)

  #define SHOW_CSPS_LINE_COUNT 3

#elif defined(IS_HDR_CSP)

  #define SHOW_CSPS_LINE_COUNT 5

#else

  #define SHOW_CSPS_LINE_COUNT 0

#endif //IS_HDR10_LIKE_CSP


static const float TEXTURE_LUMINANCE_WAVEFORM_BUFFER_WIDTH_FACTOR  = BUFFER_WIDTH_FLOAT
                                                                   / float(TEXTURE_LUMINANCE_WAVEFORM_WIDTH);

static const float TEXTURE_LUMINANCE_WAVEFORM_BUFFER_FACTOR = (BUFFER_WIDTH_FLOAT  / 3840.f
                                                             + BUFFER_HEIGHT_FLOAT / 2160.f)
                                                            / 2.f;

static const uint TEXTURE_LUMINANCE_WAVEFORM_SCALE_BORDER = TEXTURE_LUMINANCE_WAVEFORM_BUFFER_FACTOR * 35.f + 0.5f;
static const uint TEXTURE_LUMINANCE_WAVEFORM_SCALE_FRAME  = TEXTURE_LUMINANCE_WAVEFORM_BUFFER_FACTOR *  7.f + 0.5f;

//static const uint TEXTURE_LUMINANCE_WAVEFORM_FONT_SIZE =
//  clamp(uint(round(TEXTURE_LUMINANCE_WAVEFORM_BUFFER_FACTOR * 27.f + 5.f)), 14, 32);

static const uint TEXTURE_LUMINANCE_WAVEFORM_SCALE_WIDTH = TEXTURE_LUMINANCE_WAVEFORM_WIDTH
                                                         + (CHAR_DIM_FLOAT.x * 8) //8 chars for 10000.00
                                                         + uint(CHAR_DIM_FLOAT.x / 2.f + 0.5f)
                                                         + (TEXTURE_LUMINANCE_WAVEFORM_SCALE_BORDER * 2)
                                                         + (TEXTURE_LUMINANCE_WAVEFORM_SCALE_FRAME  * 3);

#ifdef IS_HDR_CSP
  #define MAX_WAVEFORM_HEIGHT_FACTOR 1
#else
  #define MAX_WAVEFORM_HEIGHT_FACTOR 2
#endif

static const uint TEXTURE_LUMINANCE_WAVEFORM_SCALE_HEIGHT = TEXTURE_LUMINANCE_WAVEFORM_USED_HEIGHT * MAX_WAVEFORM_HEIGHT_FACTOR
                                                          + uint(CHAR_DIM_FLOAT.y / 2.f - TEXTURE_LUMINANCE_WAVEFORM_SCALE_FRAME + 0.5f)
                                                          + (TEXTURE_LUMINANCE_WAVEFORM_SCALE_BORDER * 2)
                                                          + (TEXTURE_LUMINANCE_WAVEFORM_SCALE_FRAME  * 2);


texture2D TextureLuminanceWaveformScale
<
  pooled = true;
>
{
  Width  = TEXTURE_LUMINANCE_WAVEFORM_SCALE_WIDTH;
  Height = TEXTURE_LUMINANCE_WAVEFORM_SCALE_HEIGHT;
  Format = RG8;
};

sampler2D<float4> SamplerLuminanceWaveformScale
{
  Texture = TextureLuminanceWaveformScale;
};

storage2D<float4> StorageLuminanceWaveformScale
{
  Texture = TextureLuminanceWaveformScale;
};


// consolidated texture start

// max, avg and min Nits
#define MAX_AVG_MIN_NITS_VALUES_COUNT 3
#define MAX_AVG_MIN_NITS_VALUES_X_OFFSET 0
#define MAX_AVG_MIN_NITS_VALUES_Y_OFFSET 0
static const int COORDS_MAX_NITS_VALUE = int(    MAX_AVG_MIN_NITS_VALUES_X_OFFSET);
static const int COORDS_AVG_NITS_VALUE = int(1 + MAX_AVG_MIN_NITS_VALUES_X_OFFSET);
static const int COORDS_MIN_NITS_VALUE = int(2 + MAX_AVG_MIN_NITS_VALUES_X_OFFSET);


// CSP percentages
#if defined(IS_FLOAT_HDR_CSP)
  #define CSP_PERCENTAGES_COUNT 5
#elif defined(IS_HDR10_LIKE_CSP)
  #define CSP_PERCENTAGES_COUNT 3
#else
  #define CSP_PERCENTAGES_COUNT 0
#endif
#define CSP_PERCENTAGES_X_OFFSET (MAX_AVG_MIN_NITS_VALUES_COUNT + MAX_AVG_MIN_NITS_VALUES_X_OFFSET)
#define CSP_PERCENTAGES_Y_OFFSET 0
static const int COORDS_PERCENTAGE_BT709   = int(    CSP_PERCENTAGES_X_OFFSET);
static const int COORDS_PERCENTAGE_DCI_P3  = int(1 + CSP_PERCENTAGES_X_OFFSET);
static const int COORDS_PERCENTAGE_BT2020  = int(2 + CSP_PERCENTAGES_X_OFFSET);
static const int COORDS_PERCENTAGE_AP0     = int(3 + CSP_PERCENTAGES_X_OFFSET);
static const int COORDS_PERCENTAGE_INVALID = int(4 + CSP_PERCENTAGES_X_OFFSET);


// show values for max, avg and min Nits plus CSP % for BT.709, DCI-P3, BT.2020, AP0 and invalid
#if defined(IS_FLOAT_HDR_CSP)
  #define SHOW_VALUES_COUNT 8
#elif defined(IS_HDR10_LIKE_CSP)
  #define SHOW_VALUES_COUNT 6
#else
  #define SHOW_VALUES_COUNT 3
#endif
#define SHOW_VALUES_X_OFFSET (CSP_PERCENTAGES_COUNT + CSP_PERCENTAGES_X_OFFSET)
#define SHOW_VALUES_Y_OFFSET 0
static const int COORDS_SHOW_MAX_NITS           = int(    SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_AVG_NITS           = int(1 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_MIN_NITS           = int(2 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_PERCENTAGE_BT709   = int(3 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_PERCENTAGE_DCI_P3  = int(4 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_PERCENTAGE_BT2020  = int(5 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_PERCENTAGE_AP0     = int(6 + SHOW_VALUES_X_OFFSET);
static const int COORDS_SHOW_PERCENTAGE_INVALID = int(7 + SHOW_VALUES_X_OFFSET);


// offsets for overlay text blocks
#define OVERLAY_TEXT_Y_OFFSETS_COUNT 3
#define OVERLAY_TEXT_Y_OFFSETS_X_OFFSET (SHOW_VALUES_COUNT + SHOW_VALUES_X_OFFSET)
#define OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET 0
static const int COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS = int(    OVERLAY_TEXT_Y_OFFSETS_X_OFFSET);
static const int COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS        = int(1 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET);
static const int COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP  = int(2 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET);


// update Nits values and CSP percentages for the overlay
#define UPDATE_OVERLAY_PERCENTAGES_COUNT 1
#define UPDATE_OVERLAY_PERCENTAGES_X_OFFSET (OVERLAY_TEXT_Y_OFFSETS_COUNT + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET)
#define UPDATE_OVERLAY_PERCENTAGES_Y_OFFSET 0
static const int COORDS_UPDATE_OVERLAY_PERCENTAGES = int(UPDATE_OVERLAY_PERCENTAGES_X_OFFSET);


// luminance waveform variables
#define LUMINANCE_WAVEFORM_VARIABLES_COUNT 3
#define LUMINANCE_WAVEFORM_VARIABLES_X_OFFSET (UPDATE_OVERLAY_PERCENTAGES_COUNT + UPDATE_OVERLAY_PERCENTAGES_X_OFFSET)
#define LUMINANCE_WAVEFORM_VARIABLES_Y_OFFSET 0
static const int COORDS_LUMINANCE_WAVEFORM_LAST_SIZE_X       = int(    LUMINANCE_WAVEFORM_VARIABLES_X_OFFSET);
static const int COORDS_LUMINANCE_WAVEFORM_LAST_SIZE_Y       = int(1 + LUMINANCE_WAVEFORM_VARIABLES_X_OFFSET);
static const int COORDS_LUMINANCE_WAVEFORM_LAST_CUTOFF_POINT = int(2 + LUMINANCE_WAVEFORM_VARIABLES_X_OFFSET);


#define CONSOLIDATED_TEXTURE_SIZE_WIDTH  (LUMINANCE_WAVEFORM_VARIABLES_COUNT + LUMINANCE_WAVEFORM_VARIABLES_X_OFFSET)
#define CONSOLIDATED_TEXTURE_SIZE_HEIGHT 0


texture1D TextureConsolidated
<
  pooled = true;
>
{
  Width  = CONSOLIDATED_TEXTURE_SIZE_WIDTH;
//  Height = CONSOLIDATED_TEXTURE_SIZE_HEIGHT;
  Format = R32F;
};

sampler1D<float> SamplerConsolidated
{
  Texture = TextureConsolidated;
};

storage1D<float> StorageConsolidated
{
  Texture = TextureConsolidated;
};

// consolidated texture end


void VS_Clear(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position)
{
  Position = float4(-2.f, -2.f, 0.f, 1.f);
}

void PS_Clear(
  in  float4 Position : SV_Position,
  out float4 Out      : SV_Target0)
{
  Out = 0.f;
  discard;
}


void ExtendedReinhardTmo(
  inout float3 Colour,
  in    float  WhitePoint)
{
#ifdef IS_HDR_CSP
  float maxWhite = 10000.f / WhitePoint;
#else
  float maxWhite = 100.f / WhitePoint;
#endif

  Colour = (Colour * (1.f + (Colour / (maxWhite * maxWhite))))
         / (1.f + Colour);
}

float3 MergeOverlay(
  float3 Output,
  float3 Overlay,
  float  OverlayBrightness,
  float  Alpha)
{
  // tone map pixels below the overlay area
  //
  // first set 1.0 to be equal to OverlayBrightness
  float adjustFactor;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  adjustFactor = OverlayBrightness / 80.f;

  Output = Csp::Mat::Bt709To::Bt2020(Output / adjustFactor);

  // safety clamp colours outside of BT.2020
  Output = max(Output, 0.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  adjustFactor = OverlayBrightness / 10000.f;

  Output = Csp::Trc::PqTo::Linear(Output);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  adjustFactor = OverlayBrightness / 100.f;

  Output = DECODE_SDR(Output);

#endif

#if (ACTUAL_COLOUR_SPACE != CSP_SCRGB)

  Output /= adjustFactor;

#endif

  // then tone map to 1.0 at max
  ExtendedReinhardTmo(Output, OverlayBrightness);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  // safety clamp for the case that there are values that represent above 10000 nits
  Output.rgb = min(Output.rgb, 1.f);

#endif

  // apply the overlay
  Output = lerp(Output, Overlay, Alpha);

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


#include "luminance.fxh"
#ifdef IS_HDR_CSP
  #include "csp.fxh"
#endif
#include "cie.fxh"
#include "waveform.fxh"
#include "draw_text.fxh"
#include "active_area.fxh"


float3 MapBt709IntoCurrentCsp(
  float3 Colour,
  float  Brightness)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Map::Bt709Into::Scrgb(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Map::Bt709Into::Hdr10(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  return Csp::Map::Bt709Into::Hlg(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  return Csp::Map::Bt709Into::Ps5(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  return ENCODE_SDR(Colour * (Brightness / 100.f));

#else

  return 0.f;

#endif
}


void CS_RenderLuminanceWaveformAndGenerateCieDiagram(uint3 DTID : SV_DispatchThreadID)
{

  if (_SHOW_LUMINANCE_WAVEFORM || _SHOW_CIE)
  {

#ifndef WAVE64_FETCH_X_NEEDS_CLAMPING
    const int fetchPosX = DTID.x;
#else
    const int fetchPosX = min(DTID.x, BUFFER_WIDTH_MINUS_1_UINT);
#endif

#ifndef WAVE64_FETCH_Y_NEEDS_CLAMPING
    const int fetchPosY = DTID.y;
#else
    const int fetchPosY = min(DTID.y, BUFFER_HEIGHT_MINUS_1_UINT);
#endif

    const int2 fetchPos = int2(fetchPosX, fetchPosY);

    precise const float3 pixel = tex2Dfetch(SamplerBackBuffer, fetchPos).rgb;

    // get XYZ
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    precise const float3 XYZ = Csp::Mat::Bt709To::XYZ(pixel);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::PqTo::Linear(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::HlgTo::Linear(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(pixel);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    precise const float3 XYZ  = Csp::Mat::Bt709To::XYZ(DECODE_SDR(pixel));

#else

    precise const float3 XYZ = float3(0.f, 0.f, 0.f);

#endif

//ignore negative luminance and luminance being 0

    BRANCH(x)
    if (XYZ.y <= 0.f)
    {
      return;
    }

    BRANCH(x)
    if (_SHOW_CIE)
    {
      GenerateCieDiagram(XYZ);
    }

    BRANCH(x)
    if (_SHOW_LUMINANCE_WAVEFORM)
    {
      RenderLuminanceWaveform(fetchPos);
    }
  }
}


void CopyShowValues()
{
  float frametimeCounter = tex1Dfetch(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES);
  frametimeCounter += FRAMETIME;

  // only update every 1/2 of a second
  if (frametimeCounter >= _VALUES_UPDATE_RATE)
  {
    tex1Dstore(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES, 0.f);

    float maxNits = tex1Dfetch(StorageConsolidated, COORDS_MAX_NITS_VALUE);
    float avgNits = tex1Dfetch(StorageConsolidated, COORDS_AVG_NITS_VALUE);
    float minNits = tex1Dfetch(StorageConsolidated, COORDS_MIN_NITS_VALUE);

    // avoid average nits being higher than max nits in and lower than min extreme edge cases
    avgNits = clamp(avgNits, minNits, maxNits);

#ifdef IS_HDR_CSP

    float percentageBt709   = tex1Dfetch(StorageConsolidated, COORDS_PERCENTAGE_BT709);
    float percentageDciP3   = tex1Dfetch(StorageConsolidated, COORDS_PERCENTAGE_DCI_P3);
    float percentageBt2020  = tex1Dfetch(StorageConsolidated, COORDS_PERCENTAGE_BT2020);
#if defined(IS_FLOAT_HDR_CSP)
    float percentageAp0     = tex1Dfetch(StorageConsolidated, COORDS_PERCENTAGE_AP0);
    float percentageInvalid = tex1Dfetch(StorageConsolidated, COORDS_PERCENTAGE_INVALID);
#endif //IS_FLOAT_HDR_CSP
#endif //IS_HDR_CSP

    tex1Dstore(StorageConsolidated, COORDS_SHOW_MAX_NITS, maxNits);
    tex1Dstore(StorageConsolidated, COORDS_SHOW_AVG_NITS, avgNits);
    tex1Dstore(StorageConsolidated, COORDS_SHOW_MIN_NITS, minNits);

#ifdef IS_HDR_CSP

    tex1Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709,  percentageBt709);
    tex1Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3, percentageDciP3);
    tex1Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020, percentageBt2020);

#if defined(IS_FLOAT_HDR_CSP)

    tex1Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0,     percentageAp0);
    tex1Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID, percentageInvalid);

#endif //IS_FLOAT_HDR_CSP
#endif //IS_HDR_CSP

  }
  else
  {
    tex1Dstore(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES, frametimeCounter);
  }

  return;
}


void CS_Finalise()
{

  FinaliseMaxAvgMinNits();

#ifdef IS_HDR_CSP
  FinaliseCspCounter();
#endif

  RenderLuminanceWaveformScale();

  groupMemoryBarrier();

  CopyShowValues();

  return;
}

#endif //is hdr API and hdr colour space
