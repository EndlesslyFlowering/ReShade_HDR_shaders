#pragma once

#include "colour_space.fxh"


#if (defined(IS_HDR_COMPATIBLE_API) \
  && defined(IS_POSSIBLE_HDR_CSP))

//max is 32
//#ifndef THREAD_SIZE0
  #define THREAD_SIZE0 8
//#endif

//max is 1024
//#ifndef THREAD_SIZE1
  #define THREAD_SIZE1 8
//#endif

//#if (BUFFER_WIDTH % THREAD_SIZE0 == 0)
#if (BUFFER_WIDTH % 8 == 0)
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0
  #define WIDTH0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0 + 1
#endif

//#if (BUFFER_HEIGHT % THREAD_SIZE0 == 0)
#if (BUFFER_HEIGHT % 8 == 0)
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0
  #define HEIGHT0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0 + 1
#endif

//#if (BUFFER_WIDTH % THREAD_SIZE1 == 0)
#if (BUFFER_WIDTH % 8 == 0)
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1
  #define WIDTH1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1 + 1
#endif

//#if (BUFFER_HEIGHT % THREAD_SIZE1 == 0)
#if (BUFFER_HEIGHT % 8 == 0)
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1
  #define HEIGHT1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1 + 1
#endif

static const uint WIDTH0 = BUFFER_WIDTH / 2;
static const uint WIDTH1 = BUFFER_WIDTH - WIDTH0;

static const uint HEIGHT0 = BUFFER_HEIGHT / 2;
static const uint HEIGHT1 = BUFFER_HEIGHT - HEIGHT0;


#if defined(HDR_ANALYSIS_ENABLE)


#include "draw_font.fxh"

// 0.0000000894069671630859375 = ((ieee754_half_decode(0x0002)
//                               - ieee754_half_decode(0x0001))
//                              / 2)
//                             + ieee754_half_decode(0x0001)
#define SMALLEST_FP16   asfloat(0x33C00000)
// 0.0014662756584584712982177734375 = 1.5 / 1023
#define SMALLEST_UINT10 asfloat(0x3AC0300C)


//#ifndef IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION
  #define IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION NO
//#endif


precise static const float PIXELS = uint(BUFFER_WIDTH) * uint(BUFFER_HEIGHT);


#define IS_CSP_BT709   0
#define IS_CSP_DCI_P3  1
#define IS_CSP_BT2020  2
#define IS_CSP_AP0     3
#define IS_CSP_INVALID 4


uniform float FRAMETIME
<
  source = "frametime";
>;


#define TEXTURE_OVERLAY_WIDTH FONT_ATLAS_SIZE_56_CHAR_DIM.x * 29
#if defined(IS_FLOAT_HDR_CSP)
  #define TEXTURE_OVERLAY_HEIGHT FONT_ATLAS_SIZE_56_CHAR_DIM.y * 16
#else
  #define TEXTURE_OVERLAY_HEIGHT FONT_ATLAS_SIZE_56_CHAR_DIM.y * 13
#endif //IS_FLOAT_HDR_CSP

texture2D TextureTextOverlay
<
  pooled = true;
>
{
  Width  = TEXTURE_OVERLAY_WIDTH;
  Height = TEXTURE_OVERLAY_HEIGHT;
  Format = RGBA8;
};

sampler2D SamplerTextOverlay
{
  Texture = TextureTextOverlay;
};

storage2D StorageTextOverlay
{
  Texture = TextureTextOverlay;
};

#endif //HDR_ANALYSIS_ENABLE

texture2D TextureNitsValues
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = R32F;
};

sampler2D<float> SamplerNitsValues
{
  Texture = TextureNitsValues;
};

storage2D<float> StorageNitsValues
{
  Texture = TextureNitsValues;
};

#if defined(HDR_ANALYSIS_ENABLE)

#if 0
static const uint _0_Dot_01_Percent_Pixels = BUFFER_WIDTH * BUFFER_HEIGHT * 0.01f;
static const uint _0_Dot_01_Percent_Texture_Width = _0_Dot_01_Percent_Pixels / 16;

texture2D TextureMaxNits0Dot01Percent
<
  pooled = true;
>
{
  Width  = _0_Dot_01_Percent_Texture_Width;
  Height = 16;

  Format = R32F;
};

sampler2D<float> SamplerMaxNits0Dot01Percent
{
  Texture = TextureMaxNits0Dot01Percent;
};

storage2D<float> StorageMaxNits0Dot01Percent
{
  Texture = TextureMaxNits0Dot01Percent;
};
#endif


#define CIE_TEXTURE_ENTRY_DIAGRAM_COLOUR   0
#define CIE_TEXTURE_ENTRY_DIAGRAM_BLACK_BG 1
#define CIE_TEXTURE_ENTRY_BT709_OUTLINE    2
#define CIE_TEXTURE_ENTRY_DCI_P3_OUTLINE   3
#define CIE_TEXTURE_ENTRY_BT2020_OUTLINE   4
#define CIE_TEXTURE_ENTRY_AP0_OUTLINE      5

//width and height description are in lilium__hdr_analysis.fx

texture2D TextureCieConsolidated
<
  source = CIE_TEXTURE_FILE_NAME;
  pooled = true;
>
{
  Width  = CIE_TEXTURE_WIDTH;
  Height = CIE_TEXTURE_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieConsolidated
{
  Texture = TextureCieConsolidated;
};

storage2D<float4> StorageCieConsolidated
{
  Texture = TextureCieConsolidated;
};

texture2D TextureCieCurrent
<
  pooled = true;
>
{
  Width  = CIE_1931_BG_WIDTH;
  Height = CIE_1931_BG_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieCurrent
{
  Texture = TextureCieCurrent;
};

storage2D<float4> StorageCieCurrent
{
  Texture = TextureCieCurrent;
};


texture2D TextureCsps
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = R8;
};

sampler2D SamplerCsps
{
  Texture = TextureCsps;
};


static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH  = 1820;
static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT = 1024;

static const float TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_WIDTH_FACTOR  =
  (BUFFER_WIDTH  - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH - 1.f);
static const float TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_HEIGHT_FACTOR =
  (BUFFER_HEIGHT - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - 1.f);

static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH  = 2130;
static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT = 1150;

static const float TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_FACTOR_X =
  (TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH  - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH  - 1.f);
static const float TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_FACTOR_Y =
  (TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - 1.f);

texture2D TextureBrightnessHistogram
<
  pooled = true;
>
{
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT;
  Format = RGBA16;
};

sampler2D SamplerBrightnessHistogram
{
  Texture = TextureBrightnessHistogram;
};

storage2D StorageBrightnessHistogram
{
  Texture = TextureBrightnessHistogram;
};

texture2D TextureBrightnessHistogramScale
<
  source = "lilium__brightness_histogram_scale.png";
  pooled = true;
>
{
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT;
  Format = RGBA8;
};

sampler2D SamplerBrightnessHistogramScale
{
  Texture = TextureBrightnessHistogramScale;
};

texture2D TextureBrightnessHistogramFinal
<
  pooled = true;
>
{
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT;
  Format = RGBA16;
};

sampler2D SamplerBrightnessHistogramFinal
{
  Texture   = TextureBrightnessHistogramFinal;
  MagFilter = POINT;
};

#endif //HDR_ANALYSIS_ENABLE

// consolidated texture start

#define INTERMEDIATE_NITS_VALUES_X_OFFSET 0
#define INTERMEDIATE_NITS_VALUES_Y_OFFSET 0


#define CSP_COUNTER_X_OFFSET 0
#define CSP_COUNTER_Y_OFFSET 6


// (12) 4x max, avg and min Nits
#define FINAL_4_NITS_VALUES_X_OFFSET 0
#define FINAL_4_NITS_VALUES_Y_OFFSET 12
static const int2 COORDS_FINAL_4_MAX_NITS_VALUE0 = int2(     FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVG_NITS_VALUE0 = int2( 1 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MIN_NITS_VALUE0 = int2( 2 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAX_NITS_VALUE1 = int2( 3 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVG_NITS_VALUE1 = int2( 4 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MIN_NITS_VALUE1 = int2( 5 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAX_NITS_VALUE2 = int2( 6 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVG_NITS_VALUE2 = int2( 7 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MIN_NITS_VALUE2 = int2( 8 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAX_NITS_VALUE3 = int2( 9 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVG_NITS_VALUE3 = int2(10 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MIN_NITS_VALUE3 = int2(11 + FINAL_4_NITS_VALUES_X_OFFSET, FINAL_4_NITS_VALUES_Y_OFFSET);


// (4) max, max 99.99%, avg and min Nits
#define MAX_AVG_MIN_NITS_VALUES_X_OFFSET 12
#define MAX_AVG_MIN_NITS_VALUES_Y_OFFSET 12
static const int2 COORDS_MAX_NITS_VALUE   = int2(    MAX_AVG_MIN_NITS_VALUES_X_OFFSET, MAX_AVG_MIN_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_MAX_NITS99_VALUE = int2(1 + MAX_AVG_MIN_NITS_VALUES_X_OFFSET, MAX_AVG_MIN_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_AVG_NITS_VALUE   = int2(2 + MAX_AVG_MIN_NITS_VALUES_X_OFFSET, MAX_AVG_MIN_NITS_VALUES_Y_OFFSET);
static const int2 COORDS_MIN_NITS_VALUE   = int2(3 + MAX_AVG_MIN_NITS_VALUES_X_OFFSET, MAX_AVG_MIN_NITS_VALUES_Y_OFFSET);


// (6) CSP counter for BT.709, DCI-P3, BT.2020, AP0 and invalid
#define CSP_COUNTER_FINAL_X_OFFSET 16
#define CSP_COUNTER_FINAL_Y_OFFSET 12
static const int2 COORDS_CSP_PERCENTAGE_BT709   = int2(    CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_DCI_P3  = int2(1 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_BT2020  = int2(2 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_AP0     = int2(3 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_INVALID = int2(4 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);


// (9) show values for max, avg and min Nits plus CSP % for BT.709, DCI-P3, BT.2020, AP0 and invalid
#define SHOW_VALUES_X_OFFSET 22
#define SHOW_VALUES_Y_OFFSET 12
static const int2 COORDS_SHOW_MAX_NITS           = int2(    SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_AVG_NITS           = int2(1 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_MIN_NITS           = int2(2 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_BT709   = int2(3 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_DCI_P3  = int2(4 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_BT2020  = int2(5 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_AP0     = int2(6 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_INVALID = int2(7 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);


// (1) adaptive Nits for tone mapping
#define ADAPTIVE_NITS_X_OFFSET 31
#define ADAPTIVE_NITS_Y_OFFSET 12
static const int2 COORDS_ADAPTIVE_NITS = int2(ADAPTIVE_NITS_X_OFFSET, ADAPTIVE_NITS_Y_OFFSET);


// (12) averaged Nits over the last 10 frames for adaptive Nits
#define AVERAGE_MAX_NITS_X_OFFSET 32
#define AVERAGE_MAX_NITS_Y_OFFSET 12
static const int2 COORDS_AVERAGE_MAX_NITS_CUR = int2(     AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_0   = int2( 1 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_1   = int2( 2 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_2   = int2( 3 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_3   = int2( 4 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_4   = int2( 5 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_5   = int2( 6 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_6   = int2( 7 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_7   = int2( 8 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_8   = int2( 9 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAX_NITS_9   = int2(10 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);
static const int2 COORDS_AVERAGED_MAX_NITS    = int2(11 + AVERAGE_MAX_NITS_X_OFFSET, AVERAGE_MAX_NITS_Y_OFFSET);


// (6) check if redraw of text is needed for overlay
#define CHECK_OVERLAY_REDRAW_X_OFFSET 44
#define CHECK_OVERLAY_REDRAW_Y_OFFSET 12
static const int2 COORDS_CHECK_OVERLAY_REDRAW  = int2(    CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);
static const int2 COORDS_CHECK_OVERLAY_REDRAW0 = int2(1 + CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);
static const int2 COORDS_CHECK_OVERLAY_REDRAW1 = int2(2 + CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);
static const int2 COORDS_CHECK_OVERLAY_REDRAW2 = int2(3 + CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);
static const int2 COORDS_CHECK_OVERLAY_REDRAW3 = int2(4 + CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);
static const int2 COORDS_CHECK_OVERLAY_REDRAW4 = int2(5 + CHECK_OVERLAY_REDRAW_X_OFFSET, CHECK_OVERLAY_REDRAW_Y_OFFSET);


// (3) offsets for overlay text blocks
#define OVERLAY_TEXT_Y_OFFSETS_X_OFFSET 50
#define OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET 12
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_NITS = int2(    OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS        = int2(1 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP  = int2(2 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);


// (1) update Nits values and CSP percentages for the overlay
#define UPDATE_OVERLAY_PERCENTAGES_X_OFFSET 53
#define UPDATE_OVERLAY_PERCENTAGES_Y_OFFSET 12
static const int2 COORDS_UPDATE_OVERLAY_PERCENTAGES = int2(UPDATE_OVERLAY_PERCENTAGES_X_OFFSET, UPDATE_OVERLAY_PERCENTAGES_Y_OFFSET);


#define CONSOLIDATED_TEXTURE_SIZE_WIDTH  BUFFER_WIDTH
#define CONSOLIDATED_TEXTURE_SIZE_HEIGHT 13


texture2D TextureConsolidated
<
  pooled = true;
>
{
  Width  = CONSOLIDATED_TEXTURE_SIZE_WIDTH;
  Height = CONSOLIDATED_TEXTURE_SIZE_HEIGHT;
  Format = R32F;
};

sampler2D<float> SamplerConsolidated
{
  Texture = TextureConsolidated;
};

storage2D<float> StorageConsolidated
{
  Texture = TextureConsolidated;
};

// consolidated texture end

#if defined(HDR_ANALYSIS_ENABLE)


float3 MapBt709IntoCurrentCsp(
  const float3 Colour,
  const float  Brightness)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Map::Bt709Into::Scrgb(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Map::Bt709Into::Hdr10(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  return Csp::Map::Bt709Into::Hlg(Colour, Brightness);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  return Csp::Map::Bt709Into::Ps5(Colour, Brightness);

#else

  return 0.f;

#endif
}


static const float4x3 HeatmapSteps0 = float4x3(
  100.f, 203.f, 400.f,
  100.f, 203.f, 400.f,
  100.f, 203.f, 400.f,
  100.f, 203.f, 400.f);

static const float4x3 HeatmapSteps1 = float4x3(
  1000.f, 4000.f, 10000.f,
  1000.f, 2000.f,  4000.f,
  1000.f, 1500.f,  2000.f,
   600.f,  800.f,  1000.f);

float HeatmapFadeIn(float Y, float CurrentStep, float NormaliseTo)
{
  return (Y - CurrentStep)
       / (NormaliseTo - CurrentStep);
}

float HeatmapFadeOut(float Y, float CurrentStep, float NormaliseTo)
{
  return 1.f - HeatmapFadeIn(Y, CurrentStep, NormaliseTo);
}

#define HEATMAP_MODE_10000 0
#define HEATMAP_MODE_4000  1
#define HEATMAP_MODE_2000  2
#define HEATMAP_MODE_1000  3

float3 HeatmapRgbValues(
  float Y,
  uint  Mode,
  bool  HistogramOutput)
{
  float3 output;

  if (IsNAN(Y))
  {
    output.r = 0.f;
    output.g = 0.f;
    output.b = 0.f;
  }
  else if (Y < 0.f)
  {
    output.r = 0.f;
    output.g = 0.f;
    output.b = 6.25f;
  }
  else if (Y <= HeatmapSteps0[Mode][0]) // <= 100nits
  {
    //shades of grey
    float clamped = !HistogramOutput ? Y / HeatmapSteps0[Mode][0] * 0.25f
                                     : 0.666f;
    output.rgb = clamped;
  }
  else if (Y <= HeatmapSteps0[Mode][1]) // <= 203nits
  {
    //(blue+green) to green
    output.r = 0.f;
    output.g = 1.f;
    output.b = HeatmapFadeOut(Y, HeatmapSteps0[Mode][0], HeatmapSteps0[Mode][1]);
  }
  else if (Y <= HeatmapSteps0[Mode][2]) // <= 400nits
  {
    //green to yellow
    output.r = HeatmapFadeIn(Y, HeatmapSteps0[Mode][1], HeatmapSteps0[Mode][2]);
    output.g = 1.f;
    output.b = 0.f;
  }
  else if (Y <= HeatmapSteps1[Mode][0]) // <= 1000nits
  {
    //yellow to red
    output.r = 1.f;
    output.g = HeatmapFadeOut(Y, HeatmapSteps0[Mode][2], HeatmapSteps1[Mode][0]);
    output.b = 0.f;
  }
  else if (Y <= HeatmapSteps1[Mode][1]) // <= 4000nits
  {
    //red to pink
    output.r = 1.f;
    output.g = 0.f;
    output.b = HeatmapFadeIn(Y, HeatmapSteps1[Mode][0], HeatmapSteps1[Mode][1]);
  }
  else if(Y <= HeatmapSteps1[Mode][2]) // <= 10000nits
  {
    //pink to blue
    output.r = HeatmapFadeOut(Y, HeatmapSteps1[Mode][1], HeatmapSteps1[Mode][2]);
    output.g = 0.f;
    output.b = 1.f;
  }
  else // > 10000nits
  {
    output.r = 6.25f;
    output.g = 0.f;
    output.b = 0.f;
  }

  return output;
}


void PS_ClearBrightnessHistogramTexture(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD0,
  out float4 Out      : SV_TARGET)
{
  discard;
}

void CS_RenderBrightnessHistogram(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      float curPixelCLL = tex2Dfetch(SamplerNitsValues, int2(ID.x, y)).x;

      int yCoord =
        round(TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - (Csp::Trc::NitsTo::Pq(curPixelCLL) * TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT));

      tex2Dstore(StorageBrightnessHistogram,
                 int2(round(ID.x / TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_WIDTH_FACTOR), yCoord),
                 float4(HeatmapRgbValues(curPixelCLL, HEATMAP_MODE_10000, true), 1.f));
    }
  }
}

// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareRenderBrightnessHistogramToScale(
  in                  uint   Id             : SV_VertexID,
  out                 float4 VPos           : SV_Position,
  out                 float2 TexCoord       : TEXCOORD0,
  out nointerpolation int2   CllWhiteLinesY : CllWhiteLinesY)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#define minCllWhiteLineY CllWhiteLinesY.x
#define maxCllWhiteLineY CllWhiteLinesY.y

  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    if (BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE)
    {
      float minCll = tex2Dfetch(SamplerConsolidated, COORDS_MIN_NITS_VALUE);

      int yPos =
        min(
          int(round(TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - (Csp::Trc::NitsTo::Pq(minCll) * TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT)))
        , 1023);

      minCllWhiteLineY = minCll > 0.f ? yPos
                                      : -1;
    }

    if (BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE)
    {
      float maxCll = tex2Dfetch(SamplerConsolidated, COORDS_MAX_NITS_VALUE);

      int yPos =
        max(
          int(round(TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - (Csp::Trc::NitsTo::Pq(maxCll) * TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT)))
        , 0);

      maxCllWhiteLineY = maxCll < 10000.f ? yPos
                                          : -1;
    }
  }
}

void PS_RenderBrightnessHistogramToScale(
  in                  float4 VPos           : SV_Position,
  in                  float2 TexCoord       : TEXCOORD0,
  in  nointerpolation int2   CllWhiteLinesY : CllWhiteLinesY,
  out                 float4 Out            : SV_TARGET0)
{
  if (SHOW_BRIGHTNESS_HISTOGRAM)
  {
    int2 histogramCoords = int2(round(TexCoord.x * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH  - 0.5f - 250.f),
                                round(TexCoord.y * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT - 0.5f -  64.f));

    if (histogramCoords.x >= 0 && histogramCoords.x < TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH
     && histogramCoords.y >= 0 && histogramCoords.y < TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT)
    {
      if (BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE && (histogramCoords.y == minCllWhiteLineY
                                                 || histogramCoords.y == minCllWhiteLineY - 1))
      {
        Out = float4(1.f, 1.f, 1.f, 1.f);
        return;
      }
      if (BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE && (histogramCoords.y == maxCllWhiteLineY
                                                 || histogramCoords.y == maxCllWhiteLineY + 1
                                                 || histogramCoords.y == maxCllWhiteLineY + 2))
      {
        Out = float4(1.f, 1.f, 0.f, 1.f);
        return;
      }
      Out = float4(tex2D(SamplerBrightnessHistogramScale, TexCoord).rgb
                 + tex2Dfetch(SamplerBrightnessHistogram, histogramCoords).rgb
            , 1.f);
      return;
    }
    //else
    Out = tex2D(SamplerBrightnessHistogramScale, TexCoord);
    return;
  }
  discard;
}

#endif //HDR_ANALYSIS_ENABLE

void PS_CalcNitsPerPixel(
              float4 VPos     : SV_Position,
              float2 TexCoord : TEXCOORD,
  out precise float  CurNits   : SV_TARGET)
{
#if defined(HDR_ANALYSIS_ENABLE)
  if (SHOW_NITS_VALUES
   || SHOW_NITS_FROM_CURSOR
   || SHOW_HEATMAP
   || SHOW_BRIGHTNESS_HISTOGRAM
   || HIGHLIGHT_NIT_RANGE
   || DRAW_ABOVE_NITS_AS_BLACK
   || DRAW_BELOW_NITS_AS_BLACK
   || SHOW_CSP_MAP)
  {
#endif //HDR_ANALYSIS_ENABLE

    precise const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    precise float curPixelNits = dot(Csp::Mat::Bt709ToXYZ[1], pixel) * 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::PqTo::Nits(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::HlgTo::Nits(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], pixel) * 100.f;

#else

    float curPixelNits = 0.f;

#endif //ACTUAL_COLOUR_SPACE ==

    CurNits = curPixelNits >= 0.f ? curPixelNits
                                  : 0.f;

    return;

#if defined(HDR_ANALYSIS_ENABLE)
  }
  discard;
#endif //HDR_ANALYSIS_ENABLE
}

#if defined(HDR_ANALYSIS_ENABLE)

#define COORDS_INTERMEDIATE_MAX_NITS(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 0 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
//#define COORDS_INTERMEDIATE_AVG_NITS(X) \
//  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 1 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
//#define COORDS_INTERMEDIATE_MIN_NITS(X) \
//  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 2 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
//
// per column first
//void CS_GetMaxAvgMinCll0(uint3 ID : SV_DispatchThreadID)
//{
//  if (SHOW_NITS_VALUES)
//  {
//#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
//
//  if (ID.x < BUFFER_WIDTH)
//  {
//
//#endif
//
//    float maxNits = 0.f;
//    float avgNits = 0.f;
//    float minNits = 65504.f;
//
//    for (uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;
//
//      if (curNits > maxNits)
//        maxNits = curNits;
//
//      avgNits += curNits;
//
//      if (curNits < minNits)
//        minNits = curNits;
//    }
//
//    avgNits /= BUFFER_HEIGHT;
//
//    tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS(ID.x), maxNits);
//    tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS(ID.x), avgNits);
//    tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS(ID.x), minNits);
//
//#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
//
//  }
//
//#endif
//  }
//}
//
//void CS_GetMaxAvgMinCll1(uint3 ID : SV_DispatchThreadID)
//{
//  if (SHOW_NITS_VALUES)
//  {
//  float maxNits = 0.f;
//  float avgNits = 0.f;
//  float minNits = 65504.f;
//
//  for (uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    float curMaxNits = tex2Dfetch(StorageConsolidated, int2(COORDS_INTERMEDIATE_MAX_NITS(x)));
//    float curAvgNits = tex2Dfetch(StorageConsolidated, int2(COORDS_INTERMEDIATE_AVG_NITS(x)));
//    float curMinNits = tex2Dfetch(StorageConsolidated, int2(COORDS_INTERMEDIATE_MIN_NITS(x)));
//
//    if (curMaxNits > maxNits)
//      maxNits = curMaxNits;
//
//    avgNits += curAvgNits;
//
//    if (curMinNits < minNits)
//      minNits = curMinNits;
//  }
//
//  avgNits /= BUFFER_WIDTH;
//
//  barrier();
//
//  tex2Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxNits);
//  tex2Dstore(StorageConsolidated, COORDS_AVG_NITS_VALUE, avgNits);
//  tex2Dstore(StorageConsolidated, COORDS_MIN_NITS_VALUE, minNits);
//  }
//}
//
//
// per column first
//void CS_GetMaxCll0(uint3 ID : SV_DispatchThreadID)
//{
//#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
//
//  if (ID.x < BUFFER_WIDTH)
//  {
//
//#endif
//
//    float maxNits = 0.f;
//
//    for (uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;
//
//      if (curNits > maxNits)
//        maxNits = curNits;
//    }
//
//    tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS(ID.x), maxNits);
//
//#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
//
//  }
//
//#endif
//}
//
//void CS_GetMaxCll1(uint3 ID : SV_DispatchThreadID)
//{
//  float maxNits = 0.f;
//
//  for (uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    float curNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS(x));
//
//    if (curNits > maxNits)
//      maxNits = curNits;
//  }
//
//  barrier();
//
//  tex2Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxNits);
//}
//
//#undef COORDS_INTERMEDIATE_MAX_NITS
//#undef COORDS_INTERMEDIATE_AVG_NITS
//#undef COORDS_INTERMEDIATE_MIN_NITS

#endif //HDR_ANALYSIS_ENABLE

#define COORDS_INTERMEDIATE_MAX_NITS0(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 0 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_AVG_NITS0(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 1 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MIN_NITS0(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 2 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MAX_NITS1(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 3 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_AVG_NITS1(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 4 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MIN_NITS1(X) \
  int2(X + INTERMEDIATE_NITS_VALUES_X_OFFSET, 5 + INTERMEDIATE_NITS_VALUES_Y_OFFSET)

#if defined(HDR_ANALYSIS_ENABLE)

void CS_GetMaxAvgMinNits0_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_NITS_VALUES
   || (SHOW_BRIGHTNESS_HISTOGRAM
    && (BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE || BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE)))
  {
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

    if (ID.x < BUFFER_WIDTH)
    {

#endif

      if(ID.y == 0)
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for (uint y = 0; y < HEIGHT0; y++)
        {
          const float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;

          maxNits = max(maxNits, curNits);

          avgNits += curNits;

          minNits = min(minNits, curNits);
        }

        avgNits /= HEIGHT0;

        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(ID.x), maxNits);
        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS0(ID.x), avgNits);
        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS0(ID.x), minNits);

        barrier();
      }
      else
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
        {
          const float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;

          maxNits = max(maxNits, curNits);

          avgNits += curNits;

          minNits = min(minNits, curNits);
        }

        avgNits /= HEIGHT1;

        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(ID.x), maxNits);
        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS1(ID.x), avgNits);
        tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS1(ID.x), minNits);

        barrier();
      }

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

    }

#endif
  }
}

void CS_GetMaxAvgMinNits1_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_NITS_VALUES
   || (SHOW_BRIGHTNESS_HISTOGRAM
    && (BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE || BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE)))
  {
    if (ID.x == 0)
    {
      if (ID.y == 0)
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for(uint x = 0; x < WIDTH0; x++)
        {
          const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(x));
          const float curAvgNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS0(x));
          const float curMinNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS0(x));

          maxNits = max(maxNits, curMaxNits);

          avgNits += curAvgNits;

          minNits = min(minNits, curMinNits);
        }

        avgNits /= WIDTH0;

        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE0, maxNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE0, avgNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE0, minNits);

        barrier();

        return;
      }
      else
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for(uint x = 0; x < WIDTH0; x++)
        {
          const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(x));
          const float curAvgNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS1(x));
          const float curMinNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS1(x));

          maxNits = max(maxNits, curMaxNits);

          avgNits += curAvgNits;

          minNits = min(minNits, curMinNits);
        }

        avgNits /= WIDTH0;

        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE1, maxNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE1, avgNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE1, minNits);

        barrier();

        return;
      }
    }
    else
    {
      if (ID.y == 0)
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
        {
          const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(x));
          const float curAvgNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS0(x));
          const float curMinNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS0(x));

          maxNits = max(maxNits, curMaxNits);

          avgNits += curAvgNits;

          minNits = min(minNits, curMinNits);
        }

        avgNits /= WIDTH1;

        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE2, maxNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE2, avgNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE2, minNits);

        barrier();

        return;
      }
      else
      {
        float maxNits = 0.f;
        float avgNits = 0.f;
        float minNits = FP32_MAX;

        for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
        {
          const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(x));
          const float curAvgNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_AVG_NITS1(x));
          const float curMinNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MIN_NITS1(x));

          maxNits = max(maxNits, curMaxNits);

          avgNits += curAvgNits;

          minNits = min(minNits, curMinNits);
        }

        avgNits /= WIDTH1;

        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE3, maxNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE3, avgNits);
        tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE3, minNits);

        barrier();

        return;
      }
    }
  }
}

void CS_GetFinalMaxAvgMinNits_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_NITS_VALUES
   || (SHOW_BRIGHTNESS_HISTOGRAM
    && (BRIGHTNESS_HISTOGRAM_SHOW_MINCLL_LINE || BRIGHTNESS_HISTOGRAM_SHOW_MAXCLL_LINE)))
  {
    const float maxNits0 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE0);
    const float maxNits1 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE1);
    const float maxNits2 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE2);
    const float maxNits3 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE3);

    const float maxNits = max(max(max(maxNits0, maxNits1), maxNits2), maxNits3);


    const float avgNits0 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE0);
    const float avgNits1 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE1);
    const float avgNits2 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE2);
    const float avgNits3 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_AVG_NITS_VALUE3);

    const float avgNits = (avgNits0 + avgNits1 + avgNits2 + avgNits3) / 4.f;


    const float minNits0 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE0);
    const float minNits1 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE1);
    const float minNits2 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE2);
    const float minNits3 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MIN_NITS_VALUE3);

    const float minNits = min(min(min(minNits0, minNits1), minNits2), minNits3);

    tex2Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxNits);
    tex2Dstore(StorageConsolidated, COORDS_AVG_NITS_VALUE, avgNits);
    tex2Dstore(StorageConsolidated, COORDS_MIN_NITS_VALUE, minNits);

    barrier();
  }
}

#endif //HDR_ANALYSIS_ENABLE

void CS_GetMaxNits0_NEW(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

    if(ID.y == 0)
    {
      float maxNits = 0.f;

      for (uint y = 0; y < HEIGHT0; y++)
      {
        const float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;

        maxNits = max(maxNits, curNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(ID.x), maxNits);

      barrier();
    }
    else
    {
      float maxNits = 0.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;

        maxNits = max(maxNits, curNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(ID.x), maxNits);

      barrier();
    }

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void CS_GetMaxNits1_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (ID.x == 0)
  {
    if (ID.y == 0)
    {
      float maxNits = 0.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(x));

        maxNits = max(maxNits, curMaxNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE0, maxNits);

      barrier();
    }
    else
    {
      float maxNits = 0.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(x));

        maxNits = max(maxNits, curMaxNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE1, maxNits);

      barrier();
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxNits = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS0(x));

        maxNits = max(maxNits, curMaxNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE2, maxNits);

      barrier();
    }
    else
    {
      float maxNits = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxNits = tex2Dfetch(StorageConsolidated, COORDS_INTERMEDIATE_MAX_NITS1(x));

        maxNits = max(maxNits, curMaxNits);
      }

      tex2Dstore(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE3, maxNits);

      barrier();
    }
  }
}

void CS_GetFinalMaxNits_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxNits0 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE0);
  const float maxNits1 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE1);
  const float maxNits2 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE2);
  const float maxNits3 = tex2Dfetch(StorageConsolidated, COORDS_FINAL_4_MAX_NITS_VALUE3);

  const float maxNits = max(max(max(maxNits0, maxNits1), maxNits2), maxNits3);

  tex2Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxNits);

  barrier();
}


#undef COORDS_INTERMEDIATE_MAX_NITS0
#undef COORDS_INTERMEDIATE_AVG_NITS0
#undef COORDS_INTERMEDIATE_MIN_NITS0
#undef COORDS_INTERMEDIATE_MAX_NITS1
#undef COORDS_INTERMEDIATE_AVG_NITS1
#undef COORDS_INTERMEDIATE_MIN_NITS1

#if defined(HDR_ANALYSIS_ENABLE)

// per column first
//void CS_GetAvgCll0(uint3 ID : SV_DispatchThreadID)
//{
//  if (ID.x < BUFFER_WIDTH)
//  {
//    float avgNits = 0.f;
//
//    for(uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;
//
//      avgNits += curNits;
//    }
//
//    avgNits /= BUFFER_HEIGHT;
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 1), avgNits);
//  }
//}
//
//void CS_GetAvgCll1(uint3 ID : SV_DispatchThreadID)
//{
//  float avgNits = 0.f;
//
//  for(uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    float curNits = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;
//
//    avgNits += curNits;
//  }
//
//  avgNits /= BUFFER_WIDTH;
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(1, 0), avgNits);
//}
//
//
//// per column first
//void CS_GetMinCll0(uint3 ID : SV_DispatchThreadID)
//{
//  if (ID.x < BUFFER_WIDTH)
//  {
//    float minNits = 65504.f;
//
//    for(uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      float curNits = tex2Dfetch(StorageNitsValues, int2(ID.x, y)).r;
//
//      if (curNits < minNits)
//        minNits = curNits;
//    }
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 2), minNits);
//  }
//}
//
//void CS_GetMinCll1(uint3 ID : SV_DispatchThreadID)
//{
//  float minNits = 65504.f;
//
//  for(uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    float curNits = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 2)).r;
//
//    if (curNits < minNits)
//      minNits = curNits;
//  }
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(2, 0), minNits);
//}


// copy over clean bg first every time
void PS_CopyCieBg(
  in  float4 VPos     : SV_Position,
  in  float2 TexCoord : TEXCOORD0,
  out float4 Out      : SV_Target0)
{
  float2 samplerPos = float2(VPos.x + CIE_BG_WIDTH[CIE_DIAGRAM_TYPE] * float(CIE_TEXTURE_ENTRY_DIAGRAM_BLACK_BG),
                             VPos.y + float(CIE_1931_BG_HEIGHT)      * float(CIE_DIAGRAM_TYPE));

  samplerPos /= CIE_CONSOLIDATED_TEXTURE_SIZE;

  Out = tex2D(SamplerCieConsolidated, samplerPos);
  return;
}

void CS_GenerateCieDiagram(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_CIE)
  {

#if !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
 && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)

    if (ID.x < BUFFER_WIDTH && ID.y < BUFFER_HEIGHT)
    {

#elif !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
    && defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)

    if (ID.y < BUFFER_HEIGHT)
    {

#elif !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW) \
    && defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)

    if (ID.y < BUFFER_WIDTH)
    {

#endif

      precise const float3 pixel = tex2Dfetch(ReShade::BackBuffer, ID.xy).rgb;

      if (all(pixel == 0.f))
      {
        return;
      }

    // get XYZ
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      precise const float3 XYZ = Csp::Mat::Bt709To::XYZ(pixel);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::PqTo::Linear(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

      precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::HlgTo::Linear(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

      precise const float3 XYZ = Csp::Mat::Bt2020To::XYZ(pixel);

#else

      precise const float3 XYZ = float3(0.f, 0.f, 0.f);

#endif

//ignore negative luminance in float based colour spaces
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

      if (XYZ.y <= 0.f)
      {
        return;
      }

#endif

      if (CIE_DIAGRAM_TYPE == CIE_1931)
      {
        // get xy
        precise const float xyz = XYZ.x + XYZ.y + XYZ.z;

        precise int2 xy = int2(round(XYZ.x / xyz * float(CIE_ORIGINAL_DIM)),
         CIE_1931_HEIGHT - 1 - round(XYZ.y / xyz * float(CIE_ORIGINAL_DIM)));

        // adjust for the added borders
        xy += CIE_BG_BORDER;

        // clamp to to borders
        xy = clamp(xy, CIE_BG_BORDER, CIE_1931_SIZE + CIE_BG_BORDER);

        const float4 xyColour = tex2Dfetch(StorageCieConsolidated, xy);

        tex2Dstore(StorageCieCurrent,
                   xy,
                   xyColour);
      }
      else if (CIE_DIAGRAM_TYPE == CIE_1976)
      {
        // get u'v'
        precise const float X15Y3Z = XYZ.x
                                   + 15.f * XYZ.y
                                   +  3.f * XYZ.z;

        precise int2 uv = int2(round(4.f * XYZ.x / X15Y3Z * float(CIE_ORIGINAL_DIM)),
         CIE_1976_HEIGHT - 1 - round(9.f * XYZ.y / X15Y3Z * float(CIE_ORIGINAL_DIM)));

        // adjust for the added borders
        uv += CIE_BG_BORDER;

        // clamp to to borders
        uv = clamp(uv, CIE_BG_BORDER, CIE_1976_SIZE + CIE_BG_BORDER);

        const int2 uvFetchPos = int2(uv.x, uv.y + CIE_1931_BG_HEIGHT);

        const float4 uvColour = tex2Dfetch(StorageCieConsolidated, uvFetchPos);

        tex2Dstore(StorageCieCurrent,
                   uv,
                   uvColour);
      }

#if (!defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)) \
 || (!defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  &&  defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)) \
 || ( defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW))

    }

#endif
  }
}

bool IsCsp(precise float3 Rgb)
{
  if (all(Rgb >= 0.f))
  {
    return true;
  }
  return false;
}

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  #define _IS_CSP_BT709(Rgb)  Rgb
  #define _IS_CSP_DCI_P3(Rgb) Csp::Mat::Bt709To::DciP3(Rgb)
  #define _IS_CSP_BT2020(Rgb) Csp::Mat::Bt709To::Bt2020(Rgb)
  #define _IS_CSP_AP0(Rgb)    Csp::Mat::Bt709To::Ap0D65(Rgb)

#elif (defined(IS_HDR10_LIKE_CSP) \
    || ACTUAL_COLOUR_SPACE == CSP_PS5)

  #define _IS_CSP_BT709(Rgb)  Csp::Mat::Bt2020To::Bt709(Rgb)
  #define _IS_CSP_DCI_P3(Rgb) Csp::Mat::Bt2020To::DciP3(Rgb)
  #define _IS_CSP_BT2020(Rgb) Rgb
  #define _IS_CSP_AP0(Rgb)    Csp::Mat::Bt2020To::Ap0D65(Rgb)

#endif

float GetCsp(precise float3 Rgb)
{
  if (IsCsp(_IS_CSP_BT709(Rgb)))
  {
    return IS_CSP_BT709;
  }
  else if (IsCsp(_IS_CSP_DCI_P3(Rgb)))
  {
    return IS_CSP_DCI_P3 / 255.f;
  }

#if defined(IS_HDR10_LIKE_CSP)

  else
  {
    return IS_CSP_BT2020 / 255.f;
  }

#else

  else if (IsCsp(_IS_CSP_BT2020(Rgb)))
  {
    return IS_CSP_BT2020 / 255.f;
  }
  else if (IsCsp(_IS_CSP_AP0(Rgb)))
  {
    return IS_CSP_AP0 / 255.f;
  }
  else
  {
    return IS_CSP_INVALID / 255.f;
  }

#endif //IS_HDR10_LIKE_CSP

  return IS_CSP_INVALID / 255.f;
}

void PS_CalcCsps(
              float4 VPos     : SV_Position,
              float2 TexCoord : TEXCOORD,
  out precise float  curCsp   : SV_TARGET)
{
  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR
   || SHOW_CSP_MAP)
  {
    precise const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if defined(IS_FLOAT_HDR_CSP)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

    const float3 absPixel = abs(pixel);
    if (absPixel.r > SMALLEST_FP16
     && absPixel.g > SMALLEST_FP16
     && absPixel.b > SMALLEST_FP16)
    {
      curCsp = GetCsp(pixel);
    }
    else
    {
      curCsp = IS_CSP_BT709 / 255.f;
    }
    return;

#else

    curCsp = GetCsp(pixel);

    return;

#endif

#elif defined(IS_HDR10_LIKE_CSP)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

    if (pixel.r > SMALLEST_UINT10
     && pixel.g > SMALLEST_UINT10
     && pixel.b > SMALLEST_UINT10)
    {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      precise const float3 curPixel = Csp::Trc::PqTo::Linear(pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
      precise const float3 curPixel = Csp::Trc::HlgTo::Linear(pixel);
#endif
      curCsp = GetCsp(curPixel);
    }
    else
    {
      curCsp = IS_CSP_BT709 / 255.f;
    }
    return;

#else

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    precise const float3 curPixel = Csp::Trc::PqTo::Linear(pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    precise const float3 curPixel = Csp::Trc::HlgTo::Linear(pixel);
#endif
    curCsp = GetCsp(curPixel);

    return;

#endif

#else

    curCsp = IS_CSP_INVALID / 255.f;

    return;

#endif
  }
  discard;
}


#define COORDS_CSP_COUNTER_BT709(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_BT709   + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_DCI_P3(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_DCI_P3  + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_BT2020(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_BT2020  + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_AP0(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_AP0     + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_INVALID(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_INVALID + CSP_COUNTER_Y_OFFSET)


void CS_CountCspsY(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR
   || SHOW_CSP_MAP)
  {
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

    if (ID.x < BUFFER_WIDTH)
    {

#endif

      precise uint counter_BT709  = 0;
      precise uint counter_DCI_P3 = 0;

#if defined(IS_FLOAT_HDR_CSP)

      precise uint counter_BT2020 = 0;
      precise uint counter_AP0    = 0;

#endif //IS_FLOAT_HDR_CSP

      for (int y = 0; y < BUFFER_HEIGHT; y++)
      {
        uint curCsp = uint(tex2Dfetch(SamplerCsps, int2(ID.x, y)).r * 255.f);
        if (curCsp == IS_CSP_BT709)
        {
          counter_BT709++;
        }
        else if (curCsp == IS_CSP_DCI_P3)
        {
          counter_DCI_P3++;
        }

#if defined(IS_FLOAT_HDR_CSP)

        else if (curCsp == IS_CSP_BT2020)
        {
          counter_BT2020++;
        }
        else if (curCsp == IS_CSP_AP0)
        {
          counter_AP0++;
        }

#endif //IS_FLOAT_HDR_CSP
      }

      tex2Dstore(StorageConsolidated, COORDS_CSP_COUNTER_BT709(ID.x),  counter_BT709);
      tex2Dstore(StorageConsolidated, COORDS_CSP_COUNTER_DCI_P3(ID.x), counter_DCI_P3);

#if defined(IS_FLOAT_HDR_CSP)

      tex2Dstore(StorageConsolidated, COORDS_CSP_COUNTER_BT2020(ID.x), counter_BT2020);
      tex2Dstore(StorageConsolidated, COORDS_CSP_COUNTER_AP0(ID.x),    counter_AP0);

#endif //IS_FLOAT_HDR_CSP

#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

    }

#endif
  }
}

void CS_CountCspsX(uint3 ID : SV_DispatchThreadID)
{
  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR
   || SHOW_CSP_MAP)
  {
    precise uint counter_BT709  = 0;
    precise uint counter_DCI_P3 = 0;

#if defined(IS_FLOAT_HDR_CSP)

    precise uint counter_BT2020 = 0;
    precise uint counter_AP0    = 0;

#endif //IS_FLOAT_HDR_CSP

    for (int x = 0; x < BUFFER_WIDTH; x++)
    {
      counter_BT709  += uint(tex2Dfetch(StorageConsolidated, COORDS_CSP_COUNTER_BT709(x)));
      counter_DCI_P3 += uint(tex2Dfetch(StorageConsolidated, COORDS_CSP_COUNTER_DCI_P3(x)));

#if defined(IS_FLOAT_HDR_CSP)

      counter_BT2020 += uint(tex2Dfetch(StorageConsolidated, COORDS_CSP_COUNTER_BT2020(x)));
      counter_AP0    += uint(tex2Dfetch(StorageConsolidated, COORDS_CSP_COUNTER_AP0(x)));

#endif //IS_FLOAT_HDR_CSP
    }

    barrier();

    precise float percentageBt709 = counter_BT709  / PIXELS;
    precise float percentageDciP3 = counter_DCI_P3 / PIXELS;
    tex2Dstore(StorageConsolidated, COORDS_CSP_PERCENTAGE_BT709,  percentageBt709);
    tex2Dstore(StorageConsolidated, COORDS_CSP_PERCENTAGE_DCI_P3, percentageDciP3);

#if defined(IS_FLOAT_HDR_CSP)

    precise float percentageBt2020 = counter_BT2020 / PIXELS;
    precise float percentageAp0    = counter_AP0    / PIXELS;
    tex2Dstore(StorageConsolidated, COORDS_CSP_PERCENTAGE_BT2020, percentageBt2020);
    tex2Dstore(StorageConsolidated, COORDS_CSP_PERCENTAGE_AP0,    percentageAp0);

#endif //IS_FLOAT_HDR_CSP
  }
}

float3 CreateCspMap(
  uint  Csp,
  float Y)
//  float WhitePoint)
{
  if (SHOW_CSP_MAP)
  {
    float3 output;

    if (Csp != IS_CSP_BT709)
    {
      Y += 20.f;
    }

    switch(Csp)
    {
      case IS_CSP_BT709:
      {
        // shades of grey
        float clamped = Y * 0.25f;
        output = float3(clamped,
                        clamped,
                        clamped);
      } break;
      case IS_CSP_DCI_P3:
      {
        // yellow
        output = float3(Y,
                        Y,
                        0.f);
      } break;
      case IS_CSP_BT2020:
      {
        // blue
        output = float3(0.f,
                        0.f,
                        Y);
      } break;
      case IS_CSP_AP0:
      {
        // red
        output = float3(Y,
                        0.f,
                        0.f);
      } break;
      default: // invalid
      {
        // pink
        output = float3(Y,
                        0.f,
                        Y);
      } break;
    }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    output /= 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    output = Csp::Trc::NitsTo::Pq(Csp::Mat::Bt709To::Bt2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    output = Csp::Trc::NitsTo::Hlg(Csp::Mat::Bt709To::Bt2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    output = Csp::Mat::Bt709To::Bt2020(output / 100.f);

#endif

    return output;
  }
}

void ShowValuesCopy(uint3 ID : SV_DispatchThreadID)
{
  float frametimeCounter = tex2Dfetch(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES);
  frametimeCounter += FRAMETIME;

  // only update every 1/2 of a second
  if (frametimeCounter >= 500.f)
  {
    tex2Dstore(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES, 0.f);

    float maxNits = tex2Dfetch(StorageConsolidated, COORDS_MAX_NITS_VALUE);
    float avgNits = tex2Dfetch(StorageConsolidated, COORDS_AVG_NITS_VALUE);
    float minNits = tex2Dfetch(StorageConsolidated, COORDS_MIN_NITS_VALUE);

    precise float counter_BT709  = tex2Dfetch(StorageConsolidated, COORDS_CSP_PERCENTAGE_BT709)
#if (__VENDOR__ == 0x1002)
                                 * 100.0001f;
#else
                                 * 100.f;
#endif
    precise float counter_DCI_P3 = tex2Dfetch(StorageConsolidated, COORDS_CSP_PERCENTAGE_DCI_P3)
#if (__VENDOR__ == 0x1002)
                                 * 100.0001f;
#else
                                 * 100.f;
#endif

#if defined(IS_FLOAT_HDR_CSP)

    precise float counter_BT2020 = tex2Dfetch(StorageConsolidated, COORDS_CSP_PERCENTAGE_BT2020)
#if (__VENDOR__ == 0x1002)
                                 * 100.0001f;
#else
                                 * 100.f;
#endif

#else

    precise float counter_BT2020 = 100.f
                                 - counter_DCI_P3
                                 - counter_BT709;

#endif //IS_FLOAT_HDR_CSP

#if defined(IS_FLOAT_HDR_CSP)

    precise float counter_AP0     = tex2Dfetch(StorageConsolidated, COORDS_CSP_PERCENTAGE_AP0)
#if (__VENDOR__ == 0x1002)
                                 * 100.0001f;
#else
                                 * 100.f;
#endif
    precise float counter_invalid = 100.f
                                  - counter_AP0
                                  - counter_BT2020
                                  - counter_DCI_P3
                                  - counter_BT709;

#endif //IS_FLOAT_HDR_CSP

    barrier();

    tex2Dstore(StorageConsolidated, COORDS_SHOW_MAX_NITS, maxNits);
    tex2Dstore(StorageConsolidated, COORDS_SHOW_AVG_NITS, avgNits);
    tex2Dstore(StorageConsolidated, COORDS_SHOW_MIN_NITS, minNits);

    tex2Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT709,  counter_BT709);
    tex2Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_DCI_P3, counter_DCI_P3);
    tex2Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_BT2020, counter_BT2020);

#if defined(IS_FLOAT_HDR_CSP)

    tex2Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_AP0,     counter_AP0);
    tex2Dstore(StorageConsolidated, COORDS_SHOW_PERCENTAGE_INVALID, counter_invalid);

#endif //IS_FLOAT_HDR_CSP

  }
  else
  {
    tex2Dstore(StorageConsolidated, COORDS_UPDATE_OVERLAY_PERCENTAGES, frametimeCounter);
  }
  return;
}

#endif //HDR_ANALYSIS_ENABLE

#endif //is hdr API and hdr colour space
