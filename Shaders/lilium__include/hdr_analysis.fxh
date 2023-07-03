#pragma once

#include "colour_space.fxh"


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


#define CSP_BT709   0
#define CSP_DCI_P3  1
#define CSP_BT2020  2
#define CSP_AP1     3
#define CSP_AP0     4
#define CSP_INVALID 5


uniform float2 PINGPONG
<
  source    = "pingpong";
  min       = 0;
  max       = 1;
  step      = 1;
  smoothing = 0.0;
>;


texture2D CLL_Values
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
  Texture = CLL_Values;
};

//  max = (0,0); max_99.99% = (1,0); avg = (2,0); min = (3,0)
texture2D Max_Avg_Min_CLL_Values
<
  pooled = true;
>
{
  Width  = 4;
  Height = 1;

  Format = R32F;
};

sampler2D Sampler_Max_Avg_Min_CLL_Values
{
  Texture = Max_Avg_Min_CLL_Values;
};

storage2D Storage_Max_Avg_Min_CLL_Values
{
  Texture = Max_Avg_Min_CLL_Values;
};

texture2D Intermediate_CLL_Values
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = 6;

  Format = R32F;
};

sampler2D Sampler_Intermediate_CLL_Values
{
  Texture = Intermediate_CLL_Values;
};

storage2D Storage_Intermediate_CLL_Values
{
  Texture = Intermediate_CLL_Values;
};

#if 0
static const uint _0_Dot_01_Percent_Pixels = BUFFER_WIDTH * BUFFER_HEIGHT * 0.01f;
static const uint _0_Dot_01_Percent_Texture_Width = _0_Dot_01_Percent_Pixels / 16;

texture2D Max_CLL_0_Dot_01_Percent
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
  Texture = Max_CLL_0_Dot_01_Percent;
};

storage2D Storage_Max_CLL_0_Dot_01_Percent
{
  Texture = Max_CLL_0_Dot_01_Percent;
};
#endif

texture2D Final_4
<
  pooled = true;
>
{
  Width  = 6;
  Height = 2;

  Format = R32F;
};

sampler2D Sampler_Final_4
{
  Texture = Final_4;
};

storage2D Storage_Final_4
{
  Texture = Final_4;
};

#define CIE_1931 0
#define CIE_1976 1

#ifndef CIE_DIAGRAM
  #define CIE_DIAGRAM CIE_1931
#endif

#define CIE_1931_X    735
#define CIE_1931_Y    835
#define CIE_1931_BG_X 835
#define CIE_1931_BG_Y 935
#define CIE_BG_BORDER  50

#if (CIE_DIAGRAM == CIE_1931)
texture2D Texture_CIE_1931
<
  source = "lilium__cie_1931_linear.png";
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
  source = "lilium__cie_1931_black_bg_linear.png";
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

#if (CIE_DIAGRAM == CIE_1976)
#define CIE_1976_X    623
#define CIE_1976_Y    587
#define CIE_1976_BG_X 723
#define CIE_1976_BG_Y 687
texture2D Texture_CIE_1976
<
  source = "lilium__cie_1976_ucs_linear.png";
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
  source = "lilium__cie_1976_ucs_black_bg_linear.png";
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

texture2D CSPs
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
  Texture    = CSPs;
  MipLODBias = 0;
};

texture2D CSP_Counter
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = 6;

  Format = R16;
};

sampler2D Sampler_CSP_Counter
{
  Texture    = CSP_Counter;
  MipLODBias = 0;
};

storage2D Storage_CSP_Counter
{
  Texture  = CSP_Counter;
};

texture2D CSP_Counter_Final
<
  pooled = true;
>
{
  Width  = 1;
  Height = 6;

  Format = R32F;
};

sampler2D Sampler_CSP_Counter_Final
{
  Texture    = CSP_Counter_Final;
  MipLODBias = 0;
};

storage2D Storage_CSP_Counter_Final
{
  Texture = CSP_Counter_Final;
};

//  max = (0, 0); avg = (1, 0); min = (2, 0)
// (0, 1) to (5, 1) = CSPs
texture2D Show_Values
<
  pooled = true;
>
{
  Width  = 6;
  Height = 2;

  Format = R32F;
};

sampler2D Sampler_Show_Values
{
  Texture = Show_Values;
};

storage2D Storage_Show_Values
{
  Texture = Show_Values;
};

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
    const float clamped = !HistogramOutput
                        ? Y / r0 * 0.25f
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

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

    output = CSP::Mat::BT709To::BT2020(output);
    output = CSP::TRC::ToPqFromNits(output);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    output = CSP::Mat::BT709To::BT2020(output);
    output = CSP::TRC::ToHlgFromNits(output);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    output /= 100.f;
    output =  CSP::Mat::BT709To::BT2020(output);

#endif

  }

  return output;
}


static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH  = 1820;
static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT = 1024;

static const float TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_WIDTH_FACTOR  = (BUFFER_WIDTH  - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH - 1.f);
static const float TEXTURE_BRIGHTNESS_HISTOGRAM_BUFFER_HEIGHT_FACTOR = (BUFFER_HEIGHT - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - 1.f);

static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_X = 2067;
static const uint TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_Y = 1149;

static const float TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_FACTOR_X = (TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_X - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_WIDTH  - 1.f);
static const float TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_FACTOR_Y = (TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_Y - 1.f) / (TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - 1.f);

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
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_X;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_Y;
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
  Width  = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_X;
  Height = TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_Y;
  Format = RGBA16;
};

sampler2D Sampler_Brightness_Histogram_Final
{
  Texture = Texture_Brightness_Histogram_Final;
};

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
     round(TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT - (CSP::TRC::ToPqFromNits(curPixelCLL) * TEXTURE_BRIGHTNESS_HISTOGRAM_HEIGHT));

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
  const int2 histogramCoords = int2(round(TexCoord.x * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_X - 0.5f - 187.f),
                                    round(TexCoord.y * TEXTURE_BRIGHTNESS_HISTOGRAM_SCALE_Y - 0.5f -  64.f));

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

  float curPixel = dot(CSP::Mat::BT709_To_XYZ[1], pixel) * 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

  float curPixel = dot(CSP::Mat::BT2020_To_XYZ[1], CSP::TRC::FromPq(pixel)) * 10000.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  float curPixel = dot(CSP::Mat::BT2020_To_XYZ[1], CSP::TRC::FromHlg(pixel)) * 1000.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  float curPixel = dot(CSP::Mat::BT2020_To_XYZ[1], pixel) * 100.f;

#else

  float curPixel = 0.f;

#endif

  CurCLL = curPixel >= 0.f
         ? curPixel
         : 0.f;
}

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

    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 0), maxCLL);
    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 1), avgCLL);
    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 2), minCLL);

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
    const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;
    const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;
    const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 2)).r;

    if (curMaxCLL > maxCLL)
      maxCLL = curMaxCLL;

    avgCLL += curAvgCLL;

    if (curMinCLL < minCLL)
      minCLL = curMinCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(0, 0), maxCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(1, 0), avgCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(2, 0), minCLL);
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

    tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 0), maxCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void GetMaxCLL1(uint3 ID : SV_DispatchThreadID)
{
  float maxCLL = 0.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;

    if (curCLL > maxCLL)
      maxCLL = curCLL;
  }

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(0, 0), maxCLL);
}


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

      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 0), maxCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 1), avgCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 2), minCLL);
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

      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 3), maxCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 4), avgCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, 5), minCLL);
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
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(Storage_Final_4, int2(0, 0), maxCLL);
      tex2Dstore(Storage_Final_4, int2(2, 0), avgCLL);
      tex2Dstore(Storage_Final_4, int2(4, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(Storage_Final_4, int2(0, 1), maxCLL);
      tex2Dstore(Storage_Final_4, int2(2, 1), avgCLL);
      tex2Dstore(Storage_Final_4, int2(4, 1), minCLL);
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
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(Storage_Final_4, int2(1, 0), maxCLL);
      tex2Dstore(Storage_Final_4, int2(3, 0), avgCLL);
      tex2Dstore(Storage_Final_4, int2(5, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(Storage_Final_4, int2(1, 1), maxCLL);
      tex2Dstore(Storage_Final_4, int2(3, 1), avgCLL);
      tex2Dstore(Storage_Final_4, int2(5, 1), minCLL);
    }
  }
}

void GetFinalMaxAvgMinCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Final_4, int2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(Storage_Final_4, int2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(Storage_Final_4, int2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(Storage_Final_4, int2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);


  const float avgCLL0 = tex2Dfetch(Storage_Final_4, int2(2, 0)).r;
  const float avgCLL1 = tex2Dfetch(Storage_Final_4, int2(3, 0)).r;
  const float avgCLL2 = tex2Dfetch(Storage_Final_4, int2(2, 1)).r;
  const float avgCLL3 = tex2Dfetch(Storage_Final_4, int2(3, 1)).r;

  const float avgCLL = (avgCLL0 + avgCLL1 + avgCLL2 + avgCLL3) / 4.f;


  const float minCLL0 = tex2Dfetch(Storage_Final_4, int2(4, 0)).r;
  const float minCLL1 = tex2Dfetch(Storage_Final_4, int2(5, 0)).r;
  const float minCLL2 = tex2Dfetch(Storage_Final_4, int2(4, 1)).r;
  const float minCLL3 = tex2Dfetch(Storage_Final_4, int2(5, 1)).r;

  const float minCLL = min(min(min(minCLL0, minCLL1), minCLL2), minCLL3);


  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(0, 0), maxCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(1, 0), avgCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(2, 0), minCLL);
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
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, int2(ID.x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Intermediate_CLL_Values, int2(ID.x, ID.y), maxCLL);
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
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, int2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for(uint x = 0; x < WIDTH0; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, int2(ID.x, ID.y), maxCLL);
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, int2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for(uint x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, int2(x, 1)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, int2(ID.x, ID.y), maxCLL);
    }
  }
}

void GetFinalMaxCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Final_4, int2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(Storage_Final_4, int2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(Storage_Final_4, int2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(Storage_Final_4, int2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, int2(0, 0), maxCLL);
}


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

    precise float3 XYZ = CSP::Mat::BT709To::XYZ(pixel);

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

    precise float3 XYZ = CSP::Mat::BT2020To::XYZ(CSP::TRC::FromPq(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    precise float3 XYZ = CSP::Mat::BT2020To::XYZ(CSP::TRC::FromHlg(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise float3 XYZ = CSP::Mat::BT2020To::XYZ(pixel);

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
  if (IsCSP(CSP::Mat::XYZTo::BT709(XYZ)))
  {
    return CSP_BT709;
  }
  else if (IsCSP(CSP::Mat::XYZTo::DCI_P3(XYZ)))
  {
    return CSP_DCI_P3 / 255.f;
  }

#if (ACTUAL_COLOUR_SPACE == CSP_PQ \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)

  else
  {
    return CSP_BT2020 / 255.f;
  }

#else

  else if (IsCSP(CSP::Mat::XYZTo::BT2020(XYZ)))
  {
    return CSP_BT2020 / 255.f;
  }
  else if (IsCSP(CSP::Mat::XYZTo::AP1(XYZ)))
  {
    return CSP_AP1 / 255.f;
  }
  else if (IsCSP(CSP::Mat::XYZTo::AP0(XYZ)))
  {
    return CSP_AP0 / 255.f;
  }
  else
  {
    return CSP_INVALID / 255.f;
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
    curCSP = GetCSP(CSP::Mat::BT709To::XYZ(pixel));
  }
  else
  {
    curCSP = CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(CSP::Mat::BT709To::XYZ(pixel));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  const float3 curPixel = CSP::TRC::FromPq(pixel);

  if (!(curPixel.r < SMALLEST_UINT10
     && curPixel.g < SMALLEST_UINT10
     && curPixel.b < SMALLEST_UINT10))
  {
    curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(curPixel));
  }
  else
  {
    curCSP = CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(CSP::TRC::FromPq(pixel)));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  const float3 curPixel = CSP::TRC::FromPq(pixel);

  if (!(curPixel.r < SMALLEST_UINT10
     && curPixel.g < SMALLEST_UINT10
     && curPixel.b < SMALLEST_UINT10))
  {
    curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(CSP::TRC::FromHlg(curPixel)));
  }
  else
  {
    curCSP = CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(CSP::TRC::FromHlg(pixel)));

#endif

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

  if (!(pixel.r > -SMALLEST_FP16 && pixel.r < SMALLEST_FP16
     && pixel.g > -SMALLEST_FP16 && pixel.g < SMALLEST_FP16
     && pixel.b > -SMALLEST_FP16 && pixel.b < SMALLEST_FP16))
  {
    curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(pixel));
  }
  else
  {
    curCSP = CSP_BT709 / 255.f;
  }

#else

  curCSP = GetCSP(CSP::Mat::BT2020To::XYZ(pixel));

#endif

#else

  curCSP = CSP_INVALID / 255.f;

#endif
}

void CountCSPs_y(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

  if (ID.x < BUFFER_WIDTH)
  {

#endif

      uint counter_BT709   = 0;
      uint counter_DCI_P3  = 0;

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

      uint counter_BT2020  = 0;
      uint counter_AP1     = 0;
      uint counter_AP0     = 0;
      uint counter_invalid = 0;

#endif

      for (int y = 0; y < BUFFER_HEIGHT; y++)
      {
        const uint curCSP = uint(tex2Dfetch(Sampler_CSPs, int2(ID.x, y)).r * 255.f);
        if (curCSP == CSP_BT709) {
          counter_BT709++;
        }
        else if (curCSP == CSP_DCI_P3) {
          counter_DCI_P3++;
        }

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

        else if (curCSP == CSP_BT2020) {
          counter_BT2020++;
        }
        else if (curCSP == CSP_AP1) {
          counter_AP1++;
        }
        else if (curCSP == CSP_AP0) {
          counter_AP0++;
        }
        else {
          counter_invalid++;
        }

#endif
      }

      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_BT709),   counter_BT709   / 65535.f);
      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_DCI_P3),  counter_DCI_P3  / 65535.f);

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_BT2020),  counter_BT2020  / 65535.f);
      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_AP1),     counter_AP1     / 65535.f);
      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_AP0),     counter_AP0     / 65535.f);
      tex2Dstore(Storage_CSP_Counter, int2(ID.x, CSP_INVALID), counter_invalid / 65535.f);

#endif

#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW

  }

#endif
}

void CountCSPs_x(uint3 ID : SV_DispatchThreadID)
{
  uint counter_BT709   = 0;
  uint counter_DCI_P3  = 0;

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

  uint counter_BT2020  = 0;
  uint counter_AP1     = 0;
  uint counter_AP0     = 0;
  uint counter_invalid = 0;

#endif

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    counter_BT709   += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_BT709)).r   * 65535.f);
    counter_DCI_P3  += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_DCI_P3)).r  * 65535.f);

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

    counter_BT2020  += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_BT2020)).r  * 65535.f);
    counter_AP1     += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_AP1)).r     * 65535.f);
    counter_AP0     += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_AP0)).r     * 65535.f);
    counter_invalid += uint(tex2Dfetch(Sampler_CSP_Counter, int2(x, CSP_INVALID)).r * 65535.f);

#endif
  }

  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_BT709),   counter_BT709   / PIXELS);
  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_DCI_P3),  counter_DCI_P3  / PIXELS);

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_BT2020),  counter_BT2020  / PIXELS);
  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_AP1),     counter_AP1     / PIXELS);
  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_AP0),     counter_AP0     / PIXELS);
  tex2Dstore(Storage_CSP_Counter_Final, int2(0, CSP_INVALID), counter_invalid / PIXELS);

#endif
}

float3 Create_CSP_Map(
  const uint  CSP,
        float Y)
//  const float WhitePoint)
{
  float3 output;

  if (CSP != CSP_BT709)
  {
    Y += 20.f;
  }

  switch(CSP)
  {
    case CSP_BT709:
    {
      // shades of grey
      float clamped = Y * 0.25f;
      output = float3(clamped,
                      clamped,
                      clamped);
    } break;
    case CSP_DCI_P3:
    {
      // teal
      output = float3(0.f,
                      Y,
                      Y);
    } break;
    case CSP_BT2020:
    {
      // yellow
      output = float3(Y,
                      Y,
                      0.f);
    } break;
    case CSP_AP1:
    {
      // blue
      output = float3(0.f,
                      0.f,
                      Y);
    } break;
    case CSP_AP0:
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

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

  output = CSP::TRC::ToPqFromNits(CSP::Mat::BT709To::BT2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  output = CSP::TRC::ToHlgFromNits(CSP::Mat::BT709To::BT2020(output));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  output = CSP::Mat::BT709To::BT2020(output / 100.f);

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
    switch(ID.x)
    {
      case 0:
      {
        float maxCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(0, 0)).r;
        float avgCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(1, 0)).r;
        float minCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(2, 0)).r;

        tex2Dstore(Storage_Show_Values, int2(0, 0), maxCLL);
        tex2Dstore(Storage_Show_Values, int2(1, 0), avgCLL);
        tex2Dstore(Storage_Show_Values, int2(2, 0), minCLL);
      } break;

      case 1:
      {
        precise float counter_BT709  = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_BT709)).r  * 100.0001f;
        precise float counter_DCI_P3 = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_DCI_P3)).r * 100.0001f;

        tex2Dstore(Storage_Show_Values, int2(CSP_BT709,  1), counter_BT709);
        tex2Dstore(Storage_Show_Values, int2(CSP_DCI_P3, 1), counter_DCI_P3);

#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)

        precise float counter_BT2020 = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_BT2020)).r * 100.0001f;

        tex2Dstore(Storage_Show_Values, int2(CSP_BT2020, 1), counter_BT2020);

#else

        tex2Dstore(Storage_Show_Values, int2(CSP_BT2020, 1), 100.f - counter_DCI_P3 - counter_BT709);

#endif
      } break;

      case 2:
      {
        precise float counter_AP1     = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_AP1)).r  * 100.0001f;
        precise float counter_AP0     = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_AP0)).r  * 100.0001f;
        precise float counter_invalid = tex2Dfetch(Sampler_CSP_Counter_Final, int2(0, CSP_INVALID)).r * 100.0001f;

        tex2Dstore(Storage_Show_Values, int2(CSP_AP1,     1), counter_AP1);
        tex2Dstore(Storage_Show_Values, int2(CSP_AP0,     1), counter_AP0);
        tex2Dstore(Storage_Show_Values, int2(CSP_INVALID, 1), counter_invalid);
      } break;
    }
  }
  return;
}
