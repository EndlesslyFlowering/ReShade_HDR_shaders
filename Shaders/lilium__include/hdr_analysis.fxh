#pragma once

#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "colour_space.fxh"
#include "lilium__include\draw_font.fxh"

// TODO:
// - do 100.0001 multiplactions only on AMD
// - use precise for CLL calculations
// - rework "DISPATCH_DOESNT_OVERFLOW"
// - fix CIE diagram texture offset

#define SMALLEST_FP16   0.00000009
#define SMALLEST_UINT10 0.00013


#ifndef IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION
  #define IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION YES
#endif

//max is 32
//#ifndef THREAD_SIZE0
  #define THREAD_SIZE0 8
//#endif

//max is 1024
//#ifndef THREAD_SIZE1
  #define THREAD_SIZE1 8
//#endif

#if (BUFFER_WIDTH % THREAD_SIZE0 == 0)
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0
  #define WIDTH0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0 + 1
#endif

#if (BUFFER_HEIGHT % THREAD_SIZE0 == 0)
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0
  #define HEIGHT0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0 + 1
#endif

#if (BUFFER_WIDTH % THREAD_SIZE1 == 0)
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1
  #define WIDTH1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1 + 1
#endif

#if (BUFFER_HEIGHT % THREAD_SIZE1 == 0)
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1
  #define HEIGHT1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1 + 1
#endif

static const uint WIDTH0 = BUFFER_WIDTH / 2;
static const uint WIDTH1 = BUFFER_WIDTH - WIDTH0;

static const uint HEIGHT0 = BUFFER_HEIGHT / 2;
static const uint HEIGHT1 = BUFFER_HEIGHT - HEIGHT0;


static const float PIXELS = BUFFER_WIDTH * BUFFER_HEIGHT;


#define IS_CSP_BT709   0
#define IS_CSP_DCI_P3  1
#define IS_CSP_BT2020  2
#define IS_CSP_AP1     3
#define IS_CSP_AP0     4
#define IS_CSP_INVALID 5


uniform float2 PINGPONG
<
  source    = "pingpong";
  min       = 0;
  max       = 1;
  step      = 1;
  smoothing = 0.0;
>;


#define TEXTURE_OVERLAY_WIDTH FONT_ATLAS_SIZE_48_CHAR_DIM.x * 29
#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
  #define TEXTURE_OVERLAY_HEIGHT FONT_ATLAS_SIZE_48_CHAR_DIM.y * 16
#else
  #define TEXTURE_OVERLAY_HEIGHT FONT_ATLAS_SIZE_48_CHAR_DIM.y * 13
#endif

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

texture2D Texture_CLL_Values
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = R32F;
};

sampler2D Sampler_CLL_Values
{
  Texture = Texture_CLL_Values;
};

#if 0
static const uint _0_Dot_01_Percent_Pixels = BUFFER_WIDTH * BUFFER_HEIGHT * 0.01f;
static const uint _0_Dot_01_Percent_Texture_Width = _0_Dot_01_Percent_Pixels / 16;

texture2D Texture_Max_CLL_0_Dot_01_Percent
<
  pooled = true;
>
{
  Width  = _0_Dot_01_Percent_Texture_Width;
  Height = 16;

  Format = R32F;
};

sampler2D Sampler_Max_CLL_0_Dot_01_Percent
{
  Texture = Texture_Max_CLL_0_Dot_01_Percent;
};

storage2D Storage_Max_CLL_0_Dot_01_Percent
{
  Texture = Texture_Max_CLL_0_Dot_01_Percent;
};
#endif


#define CIE_1931 0
#define CIE_1976 1

#ifndef CIE_DIAGRAM
  #define CIE_DIAGRAM CIE_1931
#endif

#define CIE_BG_BORDER  50

#define CIE_1931_X    735
#define CIE_1931_Y    835
#define CIE_1931_BG_X 835
#define CIE_1931_BG_Y 935

#if (CIE_DIAGRAM == CIE_1931)
texture2D Texture_CIE_1931
<
  source = "lilium__cie_1931.png";
  pooled = true;
>
{
  Width  = CIE_1931_X;
  Height = CIE_1931_Y;
};

sampler2D Sampler_CIE_1931
{
  Texture = Texture_CIE_1931;
};

texture2D Texture_CIE_1931_Black_BG
<
  source = "lilium__cie_1931_black_bg.png";
  pooled = true;
>
{
  Width  = CIE_1931_BG_X;
  Height = CIE_1931_BG_Y;
};

sampler2D Sampler_CIE_1931_Black_BG
{
  Texture = Texture_CIE_1931_Black_BG;
};

texture2D Texture_CIE_1931_Current
<
  pooled = true;
>
{
  Width  = CIE_1931_BG_X;
  Height = CIE_1931_BG_Y;

  Format = RGBA8;
};

sampler2D Sampler_CIE_1931_Current
{
  Texture = Texture_CIE_1931_Current;
};

storage2D Storage_CIE_1931_Current
{
  Texture  = Texture_CIE_1931_Current;
};
#endif

#define CIE_1976_X    623
#define CIE_1976_Y    587
#define CIE_1976_BG_X 723
#define CIE_1976_BG_Y 687

#if (CIE_DIAGRAM == CIE_1976)
texture2D Texture_CIE_1976
<
  source = "lilium__cie_1976_ucs.png";
  pooled = true;
>
{
  Width  = CIE_1976_X;
  Height = CIE_1976_Y;
};

sampler2D Sampler_CIE_1976
{
  Texture = Texture_CIE_1976;
};

texture2D Texture_CIE_1976_Black_BG
<
  source = "lilium__cie_1976_ucs_black_bg.png";
  pooled = true;
>
{
  Width  = CIE_1976_BG_X;
  Height = CIE_1976_BG_Y;
};

sampler2D Sampler_CIE_1976_Black_BG
{
  Texture = Texture_CIE_1976_Black_BG;
};

texture2D Texture_CIE_1976_Current
<
  pooled = true;
>
{
  Width  = CIE_1976_BG_X;
  Height = CIE_1976_BG_Y;

  Format = RGBA8;
};

sampler2D Sampler_CIE_1976_Current
{
  Texture = Texture_CIE_1976_Current;
};

storage2D Storage_CIE_1976_Current
{
  Texture  = Texture_CIE_1976_Current;
};
#endif

texture2D Texture_CSPs
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = R8;
};

sampler2D Sampler_CSPs
{
  Texture    = Texture_CSPs;
  MipLODBias = 0;
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

texture2D Texture_Brightness_Histogram
<
  pooled = true;
>
{
  Width     = TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH;
  Height    = TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT;
  Format    = RGBA16;
};

sampler2D Sampler_Brightness_Histogram
{
  Texture = Texture_Brightness_Histogram;
};

storage2D Storage_Brightness_Histogram
{
  Texture = Texture_Brightness_Histogram;
};

texture2D Texture_Brightness_Histogram_Scale
<
  source = "lilium__brightness_histogram_scale.png";
  pooled = true;
>
{
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT;
};

sampler2D Sampler_Brightness_Histogram_Scale
{
  Texture = Texture_Brightness_Histogram_Scale;
};

texture2D Texture_Brightness_Histogram_Final
<
  pooled = true;
>
{
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT;
  Format = RGBA16;
};

sampler2D Sampler_Brightness_Histogram_Final
{
  Texture = Texture_Brightness_Histogram_Final;
};


// consolidated texture start

#define INTERMEDIATE_CLL_VALUES_X_OFFSET 0
#define INTERMEDIATE_CLL_VALUES_Y_OFFSET 0


#define CSP_COUNTER_X_OFFSET 0
#define CSP_COUNTER_Y_OFFSET 6


// (12) 4x max, avg and min CLL
#define FINAL_4_CLL_VALUES_X_OFFSET 0
#define FINAL_4_CLL_VALUES_Y_OFFSET 12
static const int2 COORDS_FINAL_4_MAXCLL_VALUE0 = int2(     FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVGCLL_VALUE0 = int2( 1 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MINCLL_VALUE0 = int2( 2 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAXCLL_VALUE1 = int2( 3 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVGCLL_VALUE1 = int2( 4 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MINCLL_VALUE1 = int2( 5 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAXCLL_VALUE2 = int2( 6 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVGCLL_VALUE2 = int2( 7 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MINCLL_VALUE2 = int2( 8 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MAXCLL_VALUE3 = int2( 9 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_AVGCLL_VALUE3 = int2(10 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_FINAL_4_MINCLL_VALUE3 = int2(11 + FINAL_4_CLL_VALUES_X_OFFSET, FINAL_4_CLL_VALUES_Y_OFFSET);


// (4) max, max 99.99%, avg and min CLL
#define MAX_AVG_MIN_CLL_VALUES_X_OFFSET 12
#define MAX_AVG_MIN_CLL_VALUES_Y_OFFSET 12
static const int2 COORDS_MAXCLL_VALUE   = int2(    MAX_AVG_MIN_CLL_VALUES_X_OFFSET, MAX_AVG_MIN_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_MAXCLL99_VALUE = int2(1 + MAX_AVG_MIN_CLL_VALUES_X_OFFSET, MAX_AVG_MIN_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_AVGCLL_VALUE   = int2(2 + MAX_AVG_MIN_CLL_VALUES_X_OFFSET, MAX_AVG_MIN_CLL_VALUES_Y_OFFSET);
static const int2 COORDS_MINCLL_VALUE   = int2(3 + MAX_AVG_MIN_CLL_VALUES_X_OFFSET, MAX_AVG_MIN_CLL_VALUES_Y_OFFSET);


// (6) CSP counter for BT.709, DCI-P3, BT.2020, AP1, AP0 and invalid
#define CSP_COUNTER_FINAL_X_OFFSET 16
#define CSP_COUNTER_FINAL_Y_OFFSET 12
static const int2 COORDS_CSP_PERCENTAGE_BT709   = int2(    CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_DCI_P3  = int2(1 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_BT2020  = int2(2 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_AP1     = int2(3 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_AP0     = int2(4 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);
static const int2 COORDS_CSP_PERCENTAGE_INVALID = int2(5 + CSP_COUNTER_FINAL_X_OFFSET, CSP_COUNTER_FINAL_Y_OFFSET);


// (9) show values for max, avg and min CLL plus CSP % for BT.709, DCI-P3, BT.2020, AP1, AP0 and invalid
#define SHOW_VALUES_X_OFFSET 22
#define SHOW_VALUES_Y_OFFSET 12
static const int2 COORDS_SHOW_MAXCLL             = int2(    SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_AVGCLL             = int2(1 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_MINCLL             = int2(2 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_BT709   = int2(3 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_DCI_P3  = int2(4 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_BT2020  = int2(5 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_AP1     = int2(6 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_AP0     = int2(7 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);
static const int2 COORDS_SHOW_PERCENTAGE_INVALID = int2(8 + SHOW_VALUES_X_OFFSET, SHOW_VALUES_Y_OFFSET);


// (1) adaptive CLL for tone mapping
#define ADAPTIVE_CLL_X_OFFSET 31
#define ADAPTIVE_CLL_Y_OFFSET 12
static const int2 COORDS_ADAPTIVE_CLL = int2(ADAPTIVE_CLL_X_OFFSET, ADAPTIVE_CLL_Y_OFFSET);


// (12) averaged CLL over the last 10 frames for adaptive CLL
#define AVERAGE_MAXCLL_X_OFFSET 32
#define AVERAGE_MAXCLL_Y_OFFSET 12
static const int2 COORDS_AVERAGE_MAXCLL_CUR  = int2(     AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL0 = int2( 1 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL1 = int2( 2 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL2 = int2( 3 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL3 = int2( 4 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL4 = int2( 5 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL5 = int2( 6 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL6 = int2( 7 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL7 = int2( 8 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL8 = int2( 9 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGE_MAXCLL_CLL9 = int2(10 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);
static const int2 COORDS_AVERAGED_MAXCLL     = int2(11 + AVERAGE_MAXCLL_X_OFFSET, AVERAGE_MAXCLL_Y_OFFSET);


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
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CLL = int2(    OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CSPS       = int2(1 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);
static const int2 COORDS_OVERLAY_TEXT_Y_OFFSET_CURSOR_CSP = int2(2 + OVERLAY_TEXT_Y_OFFSETS_X_OFFSET, OVERLAY_TEXT_Y_OFFSETS_Y_OFFSET);


#define CONSOLIDATED_TEXTURE_SIZE_WIDTH  BUFFER_WIDTH
#define CONSOLIDATED_TEXTURE_SIZE_HEIGHT 13


texture2D Texture_Consolidated
<
  pooled = true;
>
{
  Width     = CONSOLIDATED_TEXTURE_SIZE_WIDTH;
  Height    = CONSOLIDATED_TEXTURE_SIZE_HEIGHT;
  Format    = R32F;
};

sampler2D Sampler_Consolidated
{
  Texture = Texture_Consolidated;
};

storage2D Storage_Consolidated
{
  Texture = Texture_Consolidated;
};

// consolidated texture end


#define HEATMAP_MODE_10000 0
#define HEATMAP_MODE_1000  1

float3 Heatmap_RGB_Values(
  const float Y,
  const uint  Mode,
  const float WhitePoint,
  const bool  HistogramOutput)
{
  float3 output;

  float r0,
        r1,
        r2,
        r3,
        r4,
        r5;

  switch (Mode)
  {
    case HEATMAP_MODE_10000:
    {
      r0 =   100.f;
      r1 =   203.f;
      r2 =   400.f;
      r3 =  1000.f;
      r4 =  4000.f;
      r5 = 10000.f;
    } break;

    case HEATMAP_MODE_1000:
    {
      r0 =  100.f;
      r1 =  203.f;
      r2 =  400.f;
      r3 =  600.f;
      r4 =  800.f;
      r5 = 1000.f;
    } break;
  }

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
  else if (Y <= r0) // <= 100nits
  {
    //shades of grey
    const float clamped = !HistogramOutput ? Y / r0 * 0.25f
                                           : 0.666f;
    output.r = clamped;
    output.g = clamped;
    output.b = clamped;
  }
  else if (Y <= r1) // <= 203nits
  {
    //(blue+green) to green
    output.r = 0.f;
    output.g = 1.f;
    output.b = 1.f - ((Y - r0) / (r1 - r0));
  }
  else if (Y <= r2) // <= 400nits
  {
    //green to yellow
    output.r = (Y - r1) / (r2 - r1);
    output.g = 1.f;
    output.b = 0.f;
  }
  else if (Y <= r3) // <= 1000nits
  {
    //yellow to red
    output.r = 1.f;
    output.g = 1.f - ((Y - r2) / (r3 - r2));
    output.b = 0.f;
  }
  else if (Y <= r4) // <= 4000nits
  {
    //red to pink
    output.r = 1.f;
    output.g = 0.f;
    output.b = (Y - r3) / (r4 - r3);
  }
  else if(Y <= r5) // <= 10000nits
  {
    //pink to blue
    output.r = max(1.f - ((Y - r4) / (r5 - r4)), 0.f);
    output.g = 0.f;
    output.b = 1.f;
  }
  else // > 10000nits
  {
    output.r = 6.25f;
    output.g = 0.f;
    output.b = 0.f;
  }

  if (HistogramOutput == false)
  {
    output *= WhitePoint;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    output /= 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    output = Csp::Mat::Bt709To::Bt2020(output);
    output = Csp::Trc::ToPqFromNits(output);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    output = Csp::Mat::Bt709To::Bt2020(output);
    output = Csp::Trc::ToHlgFromNits(output);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    output /= 100.f;
    output =  Csp::Mat::Bt709To::Bt2020(output);

#endif

  }

  return output;
}


void ClearBrightnessHistogramTexture(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 Out      : SV_TARGET)
{
  return;
}

void ComputeBrightnessHistogram(uint3 ID : SV_DispatchThreadID)
{
  for (uint y = 0; y < BUFFER_HEIGHT; y++)
  {
    const float curPixelCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).x;

    const int yCoord =
     round(TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - (Csp::Trc::ToPqFromNits(curPixelCLL) * TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT));

    tex2Dstore(Storage_Brightness_Histogram,
               int2(round(ID.x / TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_WIDTH_FACTOR), yCoord),
               float4(
               Heatmap_RGB_Values(curPixelCLL, HEATMAP_MODE_10000, 1.f, true), 1.f));
  }
}

void RenderBrightnessHistogramToScale(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 Out      : SV_TARGET)
{
  const int2 histogramCoords = int2(round(TexCoord.x * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_WIDTH  - 0.5f - 250.f),
                                    round(TexCoord.y * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_HEIGHT - 0.5f -  64.f));

  if (histogramCoords.x >= 0 && histogramCoords.x < TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH
   && histogramCoords.y >= 0 && histogramCoords.y < TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT)
  {
    Out = float4(tex2D(Sampler_Brightness_Histogram_Scale, TexCoord).rgb
                  + tex2Dfetch(Sampler_Brightness_Histogram, histogramCoords).rgb
             , 1.f);
    return;
  }
  else
  {
    Out = tex2D(Sampler_Brightness_Histogram_Scale, TexCoord);
    return;
  }
}


void CalcCLL(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float  CurCLL   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  float curPixel = dot(Csp::Mat::Bt709ToXYZ[1], pixel) * 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  float curPixel = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::FromPq(pixel)) * 10000.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  float curPixel = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::FromHlg(pixel)) * 1000.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  float curPixel = dot(Csp::Mat::Bt2020ToXYZ[1], pixel) * 100.f;

#else

  float curPixel = 0.f;

#endif

  CurCLL = curPixel >= 0.f ? curPixel
                           : 0.f;
}


#define COORDS_INTERMEDIATE_MAXCLL(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 0 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_AVGCLL(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 1 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MINCLL(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 2 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)

// per column first
void GetMaxAvgMinCLL0(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

    float maxCLL = 0.f;
    float avgCLL = 0.f;
    float minCLL = 65504.f;

    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;

      avgCLL += curCLL;

      if (curCLL < minCLL)
        minCLL = curCLL;
    }

    avgCLL /= BUFFER_HEIGHT;

    tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL(ID.x), maxCLL);
    tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL(ID.x), avgCLL);
    tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL(ID.x), minCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void GetMaxAvgMinCLL1(uint3 ID : SV_DispatchThreadID)
{
  float maxCLL = 0.f;
  float avgCLL = 0.f;
  float minCLL = 65504.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curMaxCLL = tex2Dfetch(Storage_Consolidated, int2(COORDS_INTERMEDIATE_MAXCLL(x))).r;
    const float curAvgCLL = tex2Dfetch(Storage_Consolidated, int2(COORDS_INTERMEDIATE_AVGCLL(x))).r;
    const float curMinCLL = tex2Dfetch(Storage_Consolidated, int2(COORDS_INTERMEDIATE_MINCLL(x))).r;

    if (curMaxCLL > maxCLL)
      maxCLL = curMaxCLL;

    avgCLL += curAvgCLL;

    if (curMinCLL < minCLL)
      minCLL = curMinCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_MAXCLL_VALUE, maxCLL);
  tex2Dstore(Storage_Consolidated, COORDS_AVGCLL_VALUE, avgCLL);
  tex2Dstore(Storage_Consolidated, COORDS_MINCLL_VALUE, minCLL);
}


// per column first
void GetMaxCLL0(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

    float maxCLL = 0.f;

    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;
    }

    tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL(ID.x), maxCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void GetMaxCLL1(uint3 ID : SV_DispatchThreadID)
{
  float maxCLL = 0.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL(x)).r;

    if (curCLL > maxCLL)
      maxCLL = curCLL;
  }

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_MAXCLL_VALUE, maxCLL);
}

#undef COORDS_INTERMEDIATE_MAXCLL
#undef COORDS_INTERMEDIATE_AVGCLL
#undef COORDS_INTERMEDIATE_MINCLL


#define COORDS_INTERMEDIATE_MAXCLL0(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 0 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_AVGCLL0(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 1 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MINCLL0(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 2 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MAXCLL1(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 3 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_AVGCLL1(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 4 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)
#define COORDS_INTERMEDIATE_MINCLL1(X) \
  int2(X + INTERMEDIATE_CLL_VALUES_X_OFFSET, 5 + INTERMEDIATE_CLL_VALUES_Y_OFFSET)

void GetMaxAvgMinCLL0_NEW(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

    if(ID.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (uint y = 0; y < HEIGHT0; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT0;

      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(ID.x), maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL0(ID.x), avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL0(ID.x), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT1;

      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(ID.x), maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL1(ID.x), avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL1(ID.x), minCLL);
    }

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void GetMaxAvgMinCLL1_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (ID.x == 0)
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(x)).r;
        const float curAvgCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL0(x)).r;
        const float curMinCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL0(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE0, maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE0, avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE0, minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(x)).r;
        const float curAvgCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL1(x)).r;
        const float curMinCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL1(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE1, maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE1, avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE1, minCLL);
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(x)).r;
        const float curAvgCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL0(x)).r;
        const float curMinCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL0(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE2, maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE2, avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE2, minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(x)).r;
        const float curAvgCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_AVGCLL1(x)).r;
        const float curMinCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MINCLL1(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE3, maxCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE3, avgCLL);
      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE3, minCLL);
    }
  }
}

void GetFinalMaxAvgMinCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE0).r;
  const float maxCLL1 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE1).r;
  const float maxCLL2 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE2).r;
  const float maxCLL3 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE3).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);


  const float avgCLL0 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE0).r;
  const float avgCLL1 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE1).r;
  const float avgCLL2 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE2).r;
  const float avgCLL3 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_AVGCLL_VALUE3).r;

  const float avgCLL = (avgCLL0 + avgCLL1 + avgCLL2 + avgCLL3) / 4.f;


  const float minCLL0 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE0).r;
  const float minCLL1 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE1).r;
  const float minCLL2 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE2).r;
  const float minCLL3 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MINCLL_VALUE3).r;

  const float minCLL = min(min(min(minCLL0, minCLL1), minCLL2), minCLL3);

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_MAXCLL_VALUE, maxCLL);
  tex2Dstore(Storage_Consolidated, COORDS_AVGCLL_VALUE, avgCLL);
  tex2Dstore(Storage_Consolidated, COORDS_MINCLL_VALUE, minCLL);
}


void GetMaxCLL0_NEW(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

    if(ID.y == 0)
    {
      float maxCLL = 0.f;

      for (uint y = 0; y < HEIGHT0; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(ID.x), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(ID.x), maxCLL);
    }

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void GetMaxCLL1_NEW(uint3 ID : SV_DispatchThreadID)
{
  if (ID.x == 0)
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;
      }

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE0, maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;
      }

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE1, maxCLL);
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL0(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;
      }

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE2, maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Storage_Consolidated, COORDS_INTERMEDIATE_MAXCLL1(x)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;
      }

      barrier();

      tex2Dstore(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE3, maxCLL);
    }
  }
}

void GetFinalMaxCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE0).r;
  const float maxCLL1 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE1).r;
  const float maxCLL2 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE2).r;
  const float maxCLL3 = tex2Dfetch(Storage_Consolidated, COORDS_FINAL_4_MAXCLL_VALUE3).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_MAXCLL_VALUE, maxCLL);
}


#undef COORDS_INTERMEDIATE_MAXCLL0
#undef COORDS_INTERMEDIATE_AVGCLL0
#undef COORDS_INTERMEDIATE_MINCLL0
#undef COORDS_INTERMEDIATE_MAXCLL1
#undef COORDS_INTERMEDIATE_AVGCLL1
#undef COORDS_INTERMEDIATE_MINCLL1


// per column first
//void getAvgCLL0(uint3 ID : SV_DispatchThreadID)
//{
//  if (ID.x < BUFFER_WIDTH)
//  {
//    float avgCLL = 0.f;
//
//    for(uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float CurCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;
//
//      avgCLL += CurCLL;
//    }
//
//    avgCLL /= BUFFER_HEIGHT;
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 1), avgCLL);
//  }
//}
//
//void getAvgCLL1(uint3 ID : SV_DispatchThreadID)
//{
//  float avgCLL = 0.f;
//
//  for(uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float CurCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;
//
//    avgCLL += CurCLL;
//  }
//
//  avgCLL /= BUFFER_WIDTH;
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(1, 0), avgCLL);
//}
//
//
//// per column first
//void getMinCLL0(uint3 ID : SV_DispatchThreadID)
//{
//  if (ID.x < BUFFER_WIDTH)
//  {
//    float minCLL = 65504.f;
//
//    for(uint y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float CurCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, y)).r;
//
//      if (CurCLL < minCLL)
//        minCLL = CurCLL;
//    }
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 2), minCLL);
//  }
//}
//
//void getMinCLL1(uint3 ID : SV_DispatchThreadID)
//{
//  float minCLL = 65504.f;
//
//  for(uint x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float CurCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 2)).r;
//
//    if (CurCLL < minCLL)
//      minCLL = CurCLL;
//  }
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(2, 0), minCLL);
//}


// copy over clean bg first every time
#if (CIE_DIAGRAM == CIE_1931)
void Copy_CIE_1931_BG(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 CIE_BG   : SV_TARGET)
{
  CIE_BG = tex2D(Sampler_CIE_1931_Black_BG, TexCoord).rgba;
}
#endif

#if (CIE_DIAGRAM == CIE_1976)
void Copy_CIE_1976_BG(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 CIE_BG   : SV_TARGET)
{
  CIE_BG = tex2D(Sampler_CIE_1976_Black_BG, TexCoord).rgba;
}
#endif

void Generate_CIE_Diagram(uint3 ID : SV_DispatchThreadID)
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

    const float3 pixel = tex2Dfetch(ReShade::BackBuffer, ID.xy).rgb;

    // get XYZ
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    precise float3 XYZ = Csp::Mat::Bt709To::XYZ(pixel);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    precise float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::FromPq(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    precise float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::FromHlg(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise float3 XYZ = Csp::Mat::Bt2020To::XYZ(pixel);

#else

    precise float3 XYZ = float3(0.f, 0.f, 0.f);

#endif

#if (CIE_DIAGRAM == CIE_1931)
    // get xy
    precise float xyz = XYZ.x + XYZ.y + XYZ.z;
    precise int2  xy  = int2(round(XYZ.x / xyz * 1000.f),  // 1000 is the original texture size
            CIE_1931_Y - 1 - round(XYZ.y / xyz * 1000.f)); // clamp to texture size
                                                                                           // ^you should be able to do this
                                                                                           // via the sampler. set to clamping?

    precise int2 xyDiagramPos = int2(xy.x + CIE_BG_BORDER, xy.y + CIE_BG_BORDER); // adjust for the added borders

    tex2Dstore(Storage_CIE_1931_Current,
               xyDiagramPos,
               tex2Dfetch(Sampler_CIE_1931, xy).rgba);
#endif

#if (CIE_DIAGRAM == CIE_1976)
    // get u'v'
    precise float X15Y3Z = XYZ.x
                         + 15.f * XYZ.y
                         +  3.f * XYZ.z;
    precise int2 uv      = int2(round(4.f * XYZ.x / X15Y3Z * 1000.f),
               CIE_1976_Y - 1 - round(9.f * XYZ.y / X15Y3Z * 1000.f));

    precise int2 uvDiagramPos = int2(uv.x + CIE_BG_BORDER, uv.y + CIE_BG_BORDER); // adjust for the added borders

    tex2Dstore(Storage_CIE_1976_Current,
               uvDiagramPos,
               tex2Dfetch(Sampler_CIE_1976, uv).rgba);
#endif

#if (!defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)) \
 || (!defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  &&  defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)) \
 || ( defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW))

  }

#endif

}

bool IsCSP(precise const float3 RGB)
{
  if (RGB.r >= 0.f
   && RGB.g >= 0.f
   && RGB.b >= 0.f) {
    return true;
  }
  else {
    return false;
  }
}

float GetCSP(precise const float3 XYZ)
{
  if (IsCSP(Csp::Mat::XYZTo::Bt709(XYZ)))
  {
    return IS_CSP_BT709;
  }
  else if (IsCSP(Csp::Mat::XYZTo::DciP3(XYZ)))
  {
    return IS_CSP_DCI_P3 / 255.f;
  }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  else
  {
    return IS_CSP_BT2020 / 255.f;
  }

#else

  else if (IsCSP(Csp::Mat::XYZTo::Bt2020(XYZ)))
  {
    return IS_CSP_BT2020 / 255.f;
  }
  else if (IsCSP(Csp::Mat::XYZTo::AP1(XYZ)))
  {
    return IS_CSP_AP1 / 255.f;
  }
  else if (IsCSP(Csp::Mat::XYZTo::AP0(XYZ)))
  {
    return IS_CSP_AP0 / 255.f;
  }
  else
  {
    return IS_CSP_INVALID / 255.f;
  }

#endif
}

void CalcCSPs(
              float4 VPos     : SV_Position,
              float2 TexCoord : TEXCOORD,
  out precise float  curCSP   : SV_TARGET)
{
  precise const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  if (!(pixel.r > -SMALLEST_FP16 && pixel.r < SMALLEST_FP16
     || pixel.g > -SMALLEST_FP16 && pixel.g < SMALLEST_FP16
     || pixel.b > -SMALLEST_FP16 && pixel.b < SMALLEST_FP16))
  {
    curCSP = GetCSP(Csp::Mat::Bt709To::XYZ(pixel));
  }
  else
  {
    curCSP = IS_CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(Csp::Mat::Bt709To::XYZ(pixel));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  const float3 curPixel = Csp::Trc::FromPq(pixel);

  if (!(curPixel.r < SMALLEST_UINT10
     && curPixel.g < SMALLEST_UINT10
     && curPixel.b < SMALLEST_UINT10))
  {
    curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(curPixel));
  }
  else
  {
    curCSP = IS_CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(Csp::Trc::FromPq(pixel)));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  const float3 curPixel = Csp::Trc::FromPq(pixel);

  if (!(curPixel.r < SMALLEST_UINT10
     && curPixel.g < SMALLEST_UINT10
     && curPixel.b < SMALLEST_UINT10))
  {
    curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(Csp::Trc::FromHlg(curPixel)));
  }
  else
  {
    curCSP = IS_CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(Csp::Trc::FromHlg(pixel)));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  if (!(pixel.r > -SMALLEST_FP16 && pixel.r < SMALLEST_FP16
     && pixel.g > -SMALLEST_FP16 && pixel.g < SMALLEST_FP16
     && pixel.b > -SMALLEST_FP16 && pixel.b < SMALLEST_FP16))
  {
    curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(pixel));
  }
  else
  {
    curCSP = IS_CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(Csp::Mat::Bt2020To::XYZ(pixel));

#endif

#else

  curCSP = IS_CSP_INVALID / 255.f;

#endif
}


#define COORDS_CSP_COUNTER_BT709(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_BT709   + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_DCI_P3(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_DCI_P3  + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_BT2020(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_BT2020  + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_AP1(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_AP1     + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_AP0(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_AP0     + CSP_COUNTER_Y_OFFSET)
#define COORDS_CSP_COUNTER_INVALID(X) \
  int2(X + CSP_COUNTER_X_OFFSET, IS_CSP_INVALID + CSP_COUNTER_Y_OFFSET)


void CountCSPs_y(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

      uint counter_BT709   = 0;
      uint counter_DCI_P3  = 0;

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

      uint counter_BT2020  = 0;
      uint counter_AP1     = 0;
      uint counter_AP0     = 0;
      uint counter_invalid = 0;

#endif

      for (int y = 0; y < BUFFER_HEIGHT; y++)
      {
        const uint curCSP = uint(tex2Dfetch(Sampler_CSPs, int2(ID.x, y)).r * 255.f);
        if (curCSP == IS_CSP_BT709) {
          counter_BT709++;
        }
        else if (curCSP == IS_CSP_DCI_P3) {
          counter_DCI_P3++;
        }

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

        else if (curCSP == IS_CSP_BT2020) {
          counter_BT2020++;
        }
        else if (curCSP == IS_CSP_AP1) {
          counter_AP1++;
        }
        else if (curCSP == IS_CSP_AP0) {
          counter_AP0++;
        }
        else {
          counter_invalid++;
        }

#endif
      }

      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_BT709(ID.x),   counter_BT709);
      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_DCI_P3(ID.x),  counter_DCI_P3);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_BT2020(ID.x),  counter_BT2020);
      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_AP1(ID.x),     counter_AP1);
      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_AP0(ID.x),     counter_AP0);
      tex2Dstore(Storage_Consolidated, COORDS_CSP_COUNTER_INVALID(ID.x), counter_invalid);

#endif

#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void CountCSPs_x(uint3 ID : SV_DispatchThreadID)
{
  uint counter_BT709   = 0;
  uint counter_DCI_P3  = 0;

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

  uint counter_BT2020  = 0;
  uint counter_AP1     = 0;
  uint counter_AP0     = 0;
  uint counter_invalid = 0;

#endif

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    counter_BT709   += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_BT709(x)).r);
    counter_DCI_P3  += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_DCI_P3(x)).r);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

    counter_BT2020  += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_BT2020(x)).r);
    counter_AP1     += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_AP1(x)).r);
    counter_AP0     += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_AP0(x)).r);
    counter_invalid += uint(tex2Dfetch(Storage_Consolidated, COORDS_CSP_COUNTER_INVALID(x)).r);

#endif
  }

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_BT709,   counter_BT709   / PIXELS);
  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_DCI_P3,  counter_DCI_P3  / PIXELS);

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_BT2020,  counter_BT2020  / PIXELS);
  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_AP1,     counter_AP1     / PIXELS);
  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_AP0,     counter_AP0     / PIXELS);
  tex2Dstore(Storage_Consolidated, COORDS_CSP_PERCENTAGE_INVALID, counter_invalid / PIXELS);

#endif
}

float3 Create_CSP_Map(
  const uint  CSP,
        float Y)
//  const float WhitePoint)
{
  float3 output;

  if (CSP != IS_CSP_BT709)
  {
    Y += 20.f;
  }

  switch(CSP)
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
      // teal
      output = float3(0.f,
                      Y,
                      Y);
    } break;
    case IS_CSP_BT2020:
    {
      // yellow
      output = float3(Y,
                      Y,
                      0.f);
    } break;
    case IS_CSP_AP1:
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
    default: // else
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

  output = Csp::Trc::ToPqFromNits(Csp::Mat::Bt709To::Bt2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  output = Csp::Trc::ToHlgFromNits(Csp::Mat::Bt709To::Bt2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  output = Csp::Mat::Bt709To::Bt2020(output / 100.f);

#endif

  return output;
}

void ShowValuesCopy(uint3 ID : SV_DispatchThreadID)
{
  const float pingpong = PINGPONG.x;
  if (pingpong < 0.01f
  || (pingpong > 0.32f && pingpong < 0.34f)
  || (pingpong > 0.65f && pingpong < 0.67f)
  ||  pingpong > 0.99f)
  {
    float maxCLL = tex2Dfetch(Storage_Consolidated, COORDS_MAXCLL_VALUE).r;
    float avgCLL = tex2Dfetch(Storage_Consolidated, COORDS_AVGCLL_VALUE).r;
    float minCLL = tex2Dfetch(Storage_Consolidated, COORDS_MINCLL_VALUE).r;

    precise float counter_BT709  = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_BT709).r
                                 * 100.0001f;
    precise float counter_DCI_P3 = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_DCI_P3).r
                                 * 100.0001f;

#if (ACTUAL_COLOUR_SPACE != CSP_HDR10 \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

    precise float counter_BT2020 = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_BT2020).r
                                 * 100.0001f;

#else

            float counter_BT2020 = 100.f - counter_DCI_P3 - counter_BT709;

#endif

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise float counter_AP1     = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_AP1).r
                                  * 100.0001f;
    precise float counter_AP0     = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_AP0).r
                                  * 100.0001f;
    precise float counter_invalid = tex2Dfetch(Storage_Consolidated, COORDS_CSP_PERCENTAGE_INVALID).r
                                  * 100.0001f;

#endif

    barrier();

    tex2Dstore(Storage_Consolidated, COORDS_SHOW_MAXCLL, maxCLL);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_AVGCLL, avgCLL);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_MINCLL, minCLL);

    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT709,  counter_BT709);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_DCI_P3, counter_DCI_P3);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_BT2020, counter_BT2020);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_PS5)

    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP1,     counter_AP1);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_AP0,     counter_AP0);
    tex2Dstore(Storage_Consolidated, COORDS_SHOW_PERCENTAGE_INVALID, counter_invalid);

#endif

  }
  return;
}

#endif
