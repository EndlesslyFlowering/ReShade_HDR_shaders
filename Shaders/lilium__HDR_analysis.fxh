#pragma once

#include "lilium__colour_space.fxh"

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

#define CSP_BT709  0
#define CSP_DCI_P3 1
#define CSP_BT2020 2
#define CSP_AP1    3
#define CSP_AP0    4
#define CSP_ELSE   5


uniform float2 PINGPONG
<
  source    = "pingpong";
  min       = 0;
  max       = 1;
  step      = 1;
  smoothing = 0.0;
>;


texture2D CLL_Values
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_CLL_Values
{
  Texture = CLL_Values;

  SRGBTexture = false;
};

storage2D Storage_CLL_Values
{
  Texture = CLL_Values;

  MipLevel = 0;
};

//  max = (0,0); max_99.99% = (1,0); avg = (2,0); min = (3,0)
texture2D Max_Avg_Min_CLL_Values
{
  Width  = 4;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_Max_Avg_Min_CLL_Values
{
  Texture = Max_Avg_Min_CLL_Values;

  SRGBTexture = false;
};

storage2D Storage_Max_Avg_Min_CLL_Values
{
  Texture = Max_Avg_Min_CLL_Values;

  MipLevel = 0;
};

texture2D Intermediate_CLL_Values
{
  Width  = BUFFER_WIDTH;
  Height = 6;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_Intermediate_CLL_Values
{
  Texture = Intermediate_CLL_Values;

  SRGBTexture = false;
};

storage2D Storage_Intermediate_CLL_Values
{
  Texture = Intermediate_CLL_Values;

  MipLevel = 0;
};

#if 0
static const uint _0_Dot_01_Percent_Pixels = BUFFER_WIDTH * BUFFER_HEIGHT * 0.01f;
static const uint _0_Dot_01_Percent_Texture_Width = _0_Dot_01_Percent_Pixels / 16;

texture2D Max_CLL_0_Dot_01_Percent
{
  Width  = _0_Dot_01_Percent_Texture_Width;
  Height = 16;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_Max_CLL_0_Dot_01_Percent
{
  Texture = Max_CLL_0_Dot_01_Percent;

  SRGBTexture = false;
};

storage2D Storage_Max_CLL_0_Dot_01_Percent
{
  Texture = Max_CLL_0_Dot_01_Percent;

  MipLevel = 0;
};
#endif

texture2D Final_4
{
  Width  = 6;
  Height = 2;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_Final_4
{
  Texture = Final_4;

  SRGBTexture = false;
};

storage2D Storage_Final_4
{
  Texture = Final_4;

  MipLevel = 0;
};

#define CIE_1931_X    735
#define CIE_1931_Y    835
#define CIE_1931_BG_X 835
#define CIE_1931_BG_Y 935
#define CIE_BG_BORDER  50
texture2D CIE_1931 <source = "lilium__CIE_1931_linear.png";>
{
  Width     = CIE_1931_X;
  Height    = CIE_1931_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1931
{
  Texture = CIE_1931;
};

texture2D CIE_1931_Black_BG <source = "lilium__CIE_1931_Black_BG_Linear.png";>
{
  Width     = CIE_1931_BG_X;
  Height    = CIE_1931_BG_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1931_Black_BG
{
  Texture = CIE_1931_Black_BG;
};

texture2D CIE_1931_Current
{
  Width     = CIE_1931_BG_X;
  Height    = CIE_1931_BG_Y;
  MipLevels = 0;
  Format    = RGBA8;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1931_Current
{
  Texture = CIE_1931_Current;
};

storage2D Storage_CIE_1931_Current
{
  Texture  = CIE_1931_Current;
  MipLevel = 0;
};

#define CIE_1976_X    623
#define CIE_1976_Y    587
#define CIE_1976_BG_X 723
#define CIE_1976_BG_Y 687
texture2D CIE_1976 <source = "lilium__CIE_1976_UCS_Linear.png";>
{
  Width     = CIE_1976_X;
  Height    = CIE_1976_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1976
{
  Texture = CIE_1976;
};

texture2D CIE_1976_Black_BG <source = "lilium__CIE_1976_UCS_Black_BG_linear.png";>
{
  Width     = CIE_1976_BG_X;
  Height    = CIE_1976_BG_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1976_Black_BG
{
  Texture = CIE_1976_Black_BG;
};

texture2D CIE_1976_Current
{
  Width     = CIE_1976_BG_X;
  Height    = CIE_1976_BG_Y;
  MipLevels = 0;
  Format    = RGBA8;

  SRGBTexture = false;
};

sampler2D Sampler_CIE_1976_Current
{
  Texture = CIE_1976_Current;
};

storage2D Storage_CIE_1976_Current
{
  Texture  = CIE_1976_Current;
  MipLevel = 0;
};

texture2D CSPs
{
  Width     = BUFFER_WIDTH;
  Height    = BUFFER_HEIGHT;
  MipLevels = 0;
  Format    = R8;
};

sampler2D Sampler_CSPs
{
  Texture    = CSPs;
  MipLODBias = 0;
};

texture2D CSP_Counter
{
  Width     = BUFFER_WIDTH;
  Height    = 6;
  MipLevels = 0;
  Format    = R16;
};

sampler2D Sampler_CSP_Counter
{
  Texture    = CSP_Counter;
  MipLODBias = 0;
};

storage2D Storage_CSP_Counter
{
  Texture  = CSP_Counter;
  MipLevel = 0;
};

texture2D CSP_Counter_Final
{
  Width     = 1;
  Height    = 6;
  MipLevels = 0;
  Format    = R32F;
};

sampler2D Sampler_CSP_Counter_Final
{
  Texture    = CSP_Counter_Final;
  MipLODBias = 0;
};

storage2D Storage_CSP_Counter_Final
{
  Texture  = CSP_Counter_Final;
  MipLevel = 0;
};

//  max = (0, 0); avg = (1, 0); min = (2, 0)
// (0, 1) to (5, 1) = CSPs
texture2D Show_Values
{
  Width  = 6;
  Height = 2;

  MipLevels = 0;

  Format = R32F;
};

sampler2D Sampler_Show_Values
{
  Texture = Show_Values;

  SRGBTexture = false;
};

storage2D Storage_Show_Values
{
  Texture = Show_Values;

  MipLevel = 0;
};


float3 Heatmap_RGB_Values(
  const float Y,
  const uint  Mode,
  const float WhitePoint,
  const bool  SDR_Output)
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
    case 0:
    {
      r0 =   100.f;
      r1 =   203.f;
      r2 =   400.f;
      r3 =  1000.f;
      r4 =  4000.f;
      r5 = 10000.f;
    } break;

    case 1:
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
    const float clamped = Y / r0 * 0.25f;
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

  if (SDR_Output == false)
  {
    output *= WhitePoint;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    output /= 80.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
    output  = mul(BT709_To_BT2020, output);
    output  = PQ_OETF(output);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    output  = mul(BT709_To_BT2020, output);
    output  = HLG_OETF(output);
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
    output /= 100.f;
    output  = mul(BT709_To_BT2020, output);
#endif

  }
  else
  {
    output = pow(output, 1.f / 2.2f);
  }

  return output;
}

void CalcCLL(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float  CurCLL   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  float curPixel = dot(BT709_To_XYZ[1].rgb, pixel) * 80.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
  float curPixel = dot(BT2020_To_XYZ[1].rgb, PQ_EOTF(pixel)) * 10000.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
  float curPixel = dot(BT2020_To_XYZ[1].rgb, HLG_EOTF(pixel)) * 1000.f;
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
  float curPixel = dot(BT2020_To_XYZ[1].rgb, pixel) * 100.f;
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
      const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;

      avgCLL += curCLL;

      if (curCLL < minCLL)
        minCLL = curCLL;
    }

    avgCLL /= BUFFER_HEIGHT;

    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 0), maxCLL);
    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 1), avgCLL);
    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 2), minCLL);

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
    const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;
    const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 1)).r;
    const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 2)).r;

    if (curMaxCLL > maxCLL)
      maxCLL = curMaxCLL;

    avgCLL += curAvgCLL;

    if (curMinCLL < minCLL)
      minCLL = curMinCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(0, 0), maxCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(1, 0), avgCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(2, 0), minCLL);
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
      const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;
    }

    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 0), maxCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void GetMaxCLL1(uint3 ID : SV_DispatchThreadID)
{
  float maxCLL = 0.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;

    if (curCLL > maxCLL)
      maxCLL = curCLL;
  }

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(0, 0), maxCLL);
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
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT0;

      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 0), maxCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 1), avgCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 2), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT1;

      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 3), maxCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 4), avgCLL);
      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 5), minCLL);
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

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(Storage_Final_4, uint2(0, 0), maxCLL);
      tex2Dstore(Storage_Final_4, uint2(2, 0), avgCLL);
      tex2Dstore(Storage_Final_4, uint2(4, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(Storage_Final_4, uint2(0, 1), maxCLL);
      tex2Dstore(Storage_Final_4, uint2(2, 1), avgCLL);
      tex2Dstore(Storage_Final_4, uint2(4, 1), minCLL);
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(Storage_Final_4, uint2(1, 0), maxCLL);
      tex2Dstore(Storage_Final_4, uint2(3, 0), avgCLL);
      tex2Dstore(Storage_Final_4, uint2(5, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(Storage_Final_4, uint2(1, 1), maxCLL);
      tex2Dstore(Storage_Final_4, uint2(3, 1), avgCLL);
      tex2Dstore(Storage_Final_4, uint2(5, 1), minCLL);
    }
  }
}

void GetFinalMaxAvgMinCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Final_4, uint2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(Storage_Final_4, uint2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(Storage_Final_4, uint2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(Storage_Final_4, uint2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);


  const float avgCLL0 = tex2Dfetch(Storage_Final_4, uint2(2, 0)).r;
  const float avgCLL1 = tex2Dfetch(Storage_Final_4, uint2(3, 0)).r;
  const float avgCLL2 = tex2Dfetch(Storage_Final_4, uint2(2, 1)).r;
  const float avgCLL3 = tex2Dfetch(Storage_Final_4, uint2(3, 1)).r;

  const float avgCLL = (avgCLL0 + avgCLL1 + avgCLL2 + avgCLL3) / 4.f;


  const float minCLL0 = tex2Dfetch(Storage_Final_4, uint2(4, 0)).r;
  const float minCLL1 = tex2Dfetch(Storage_Final_4, uint2(5, 0)).r;
  const float minCLL2 = tex2Dfetch(Storage_Final_4, uint2(4, 1)).r;
  const float minCLL3 = tex2Dfetch(Storage_Final_4, uint2(5, 1)).r;

  const float minCLL = min(min(min(minCLL0, minCLL1), minCLL2), minCLL3);


  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(0, 0), maxCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(1, 0), avgCLL);
  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(2, 0), minCLL);
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
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, ID.y), maxCLL);
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

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, uint2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, uint2(ID.x, ID.y), maxCLL);
    }
  }
  else
  {
    if (ID.y == 0)
    {
      float maxCLL = 0.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, uint2(ID.x, ID.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 1)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(Storage_Final_4, uint2(ID.x, ID.y), maxCLL);
    }
  }
}

void GetFinalMaxCLL_NEW(uint3 ID : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(Storage_Final_4, uint2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(Storage_Final_4, uint2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(Storage_Final_4, uint2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(Storage_Final_4, uint2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(0, 0), maxCLL);
}


// per column first
//void getAvgCLL0(uint3 ID : SV_DispatchThreadID)
//{
//  if (ID.x < BUFFER_WIDTH)
//  {
//    float avgCLL = 0.f;
//
//    for (int y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float CurCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;
//
//      avgCLL += CurCLL;
//    }
//
//    avgCLL /= BUFFER_HEIGHT;
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 1), avgCLL);
//  }
//}
//
//void getAvgCLL1(uint3 ID : SV_DispatchThreadID)
//{
//  float avgCLL = 0.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float CurCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 1)).r;
//
//    avgCLL += CurCLL;
//  }
//
//  avgCLL /= BUFFER_WIDTH;
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(1, 0), avgCLL);
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
//    for (int y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float CurCLL = tex2Dfetch(Sampler_CLL_Values, uint2(ID.x, y)).r;
//
//      if (CurCLL < minCLL)
//        minCLL = CurCLL;
//    }
//
//    tex2Dstore(Storage_Intermediate_CLL_Values, uint2(ID.x, 2), minCLL);
//  }
//}
//
//void getMinCLL1(uint3 ID : SV_DispatchThreadID)
//{
//  float minCLL = 65504.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float CurCLL = tex2Dfetch(Sampler_Intermediate_CLL_Values, uint2(x, 2)).r;
//
//    if (CurCLL < minCLL)
//      minCLL = CurCLL;
//  }
//
//  tex2Dstore(Storage_Max_Avg_Min_CLL_Values, uint2(2, 0), minCLL);
//}


// copy over clean bg first every time
void Copy_CIE_1931_BG(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 CIE_BG   : SV_TARGET)
{
  CIE_BG = tex2D(Sampler_CIE_1931_Black_BG, TexCoord).rgba;
}

void Copy_CIE_1976_BG(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 CIE_BG   : SV_TARGET)
{
  CIE_BG = tex2D(Sampler_CIE_1976_Black_BG, TexCoord).rgba;
}

void Generate_CIE_Diagram(uint3 ID : SV_DispatchThreadID)
{
#if !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
 && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)
  if (ID.x < BUFFER_WIDTH && ID.y < BUFFER_HEIGHT)
  {
#endif

#if !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
  && defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)
  if (ID.y < BUFFER_HEIGHT)
  {
#endif

#if !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW) \
  && defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)
  if (ID.y < BUFFER_WIDTH)
  {
#endif

    const float3 pixel = tex2Dfetch(ReShade::BackBuffer, ID.xy).rgb;

    // get XYZ
#if (ACTUAL_COLOUR_SPACE == CSP_SRGB || ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    float3 XYZ = mul(BT709_To_XYZ, pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
    float3 XYZ = mul(BT2020_To_XYZ, PQ_EOTF(pixel));
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    float3 XYZ = mul(BT2020_To_XYZ, HLG_EOTF(pixel));
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
    float3 XYZ = mul(BT2020_To_XYZ, pixel);
#else
    float3 XYZ = float3(0.f, 0.f, 0.f);
#endif

    // get xy
    float xyz = XYZ.x + XYZ.y + XYZ.z;
    uint2 xy  = uint2(clamp(uint(round(XYZ.x / xyz * 1000.f)), 0, CIE_1931_X - 1),  // 1000 is the original texture size
     CIE_1931_Y - 1 - clamp(uint(round(XYZ.y / xyz * 1000.f)), 0, CIE_1931_Y - 1)); // clamp to texture size
                                                                                    // ^you should be able to do this
                                                                                    // via the sampler. set to clamping?

    // get u'v'
    float X15Y3Z = XYZ.x
                 + 15.f * XYZ.y
                 +  3.f * XYZ.z;
    uint2 uv     = uint2(clamp(uint(round(4.f * XYZ.x / X15Y3Z * 1000.f)), 0, CIE_1976_X - 1),
        CIE_1976_Y - 1 - clamp(uint(round(9.f * XYZ.y / X15Y3Z * 1000.f)), 0, CIE_1976_Y - 1));

  tex2Dstore(Storage_CIE_1931_Current,
             uint2(xy.x + CIE_BG_BORDER, xy.y + CIE_BG_BORDER), // adjust for the added borders of 100 pixels
             tex2Dfetch(Sampler_CIE_1931, xy).rgba);

  tex2Dstore(Storage_CIE_1976_Current,
             uint2(uv.x + CIE_BG_BORDER, uv.y + CIE_BG_BORDER), // adjust for the added borders of 100 pixels
             tex2Dfetch(Sampler_CIE_1976, uv).rgba);

#if !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
 && !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)
  }
#endif

#if !defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)  \
  && defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW)
  }
#endif

#if !defined(HEIGHT1_DISPATCH_DOESNT_OVERFLOW) \
  && defined(WIDTH1_DISPATCH_DOESNT_OVERFLOW)
  }
#endif

}


bool IsCSP(const float3 RGB)
{
  if (RGB.r >= 0.f
   && RGB.g >= 0.f
   && RGB.b >= 0.f)
    return true;
  else
    return false;
}

float GetCSP(const float3 XYZ)
{
  if (IsCSP(mul(XYZ_To_BT709, XYZ)))
  {
    return CSP_BT709;
  }
  else if (IsCSP(mul(XYZ_To_DCI_P3, XYZ)))
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
  else if (IsCSP(mul(XYZ_To_BT2020, XYZ)))
  {
    return CSP_BT2020 / 255.f;
  }
  else if (IsCSP(mul(XYZ_To_AP1, XYZ)))
  {
    return CSP_AP1 / 255.f;
  }
  else if (IsCSP(mul(XYZ_To_AP0, XYZ)))
  {
    return CSP_AP0 / 255.f;
  }
  else
  {
    return CSP_ELSE / 255.f;
  }
#endif
}

void CalcCSPs(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float  curCSP   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  curCSP = GetCSP(mul(BT709_To_XYZ, pixel));
#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)
  curCSP = GetCSP(mul(BT2020_To_XYZ, PQ_EOTF(pixel)));
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
  curCSP = GetCSP(mul(BT2020_To_XYZ, HLG_EOTF(pixel)));
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
  curCSP = GetCSP(mul(BT2020_To_XYZ, pixel));
#else
  curCSP = CSP_ELSE / 255.f;
#endif
}

void CountCSPs_y(uint3 ID : SV_DispatchThreadID)
{
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW
  if (ID.x < BUFFER_WIDTH)
  {
#endif
      uint counter_BT709  = 0;
      uint counter_DCI_P3 = 0;
      uint counter_BT2020 = 0;
#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
      uint counter_AP1    = 0;
      uint counter_AP0    = 0;
      uint counter_else   = 0;
#endif

      for (uint y = 0; y < BUFFER_HEIGHT; y++)
      {
        const uint curCSP = uint(tex2Dfetch(Sampler_CSPs, uint2(ID.x, y)).r * 255.f);
        if (curCSP == CSP_BT709)
          counter_BT709++;
        else if (curCSP == CSP_DCI_P3)
          counter_DCI_P3++;
#if (ACTUAL_COLOUR_SPACE == CSP_PQ \
  || ACTUAL_COLOUR_SPACE == CSP_HLG)
        else
          counter_BT2020++;
#else
        else if (curCSP == CSP_BT2020)
          counter_BT2020++;
        else if (curCSP == CSP_AP1)
          counter_AP1++;
        else if (curCSP == CSP_AP0)
          counter_AP0++;
        else
          counter_else++;
#endif
      }
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_BT709),  counter_BT709  / 65535.f);
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_DCI_P3), counter_DCI_P3 / 65535.f);
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_BT2020), counter_BT2020 / 65535.f);
#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_AP1),    counter_AP1    / 65535.f);
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_AP0),    counter_AP0    / 65535.f);
      tex2Dstore(Storage_CSP_Counter, uint2(ID.x, CSP_ELSE),   counter_else   / 65535.f);
#endif
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void CountCSPs_x(uint3 ID : SV_DispatchThreadID)
{
  uint counter_BT709  = 0;
  uint counter_DCI_P3 = 0;
  uint counter_BT2020 = 0;
#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
  uint counter_AP1    = 0;
  uint counter_AP0    = 0;
  uint counter_else   = 0;
#endif

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    counter_BT709  += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_BT709)).r  * 65535.f);
    counter_DCI_P3 += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_DCI_P3)).r * 65535.f);
    counter_BT2020 += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_BT2020)).r * 65535.f);
#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
    counter_AP1    += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_AP1)).r    * 65535.f);
    counter_AP0    += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_AP0)).r    * 65535.f);
    counter_else   += uint(tex2Dfetch(Sampler_CSP_Counter, uint2(x, CSP_ELSE)).r   * 65535.f);
#endif
  }

  const float pixels = BUFFER_WIDTH * BUFFER_HEIGHT;
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_BT709),  counter_BT709  / pixels);
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_DCI_P3), counter_DCI_P3 / pixels);
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_BT2020), counter_BT2020 / pixels);
#if (ACTUAL_COLOUR_SPACE != CSP_PQ \
  && ACTUAL_COLOUR_SPACE != CSP_HLG)
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_AP1),    counter_AP1    / pixels);
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_AP0),    counter_AP0    / pixels);
  tex2Dstore(Storage_CSP_Counter_Final, uint2(0, CSP_ELSE),   counter_else   / pixels);
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
  output = PQ_OETF(mul(BT709_To_BT2020, output));
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
  output = HLG_OETF(mul(BT709_To_BT2020, output));
#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)
  output = mul(BT709_To_BT2020, output / 100.f);
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
        float maxCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, uint2(0, 0)).r;
        float avgCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, uint2(1, 0)).r;
        float minCLL = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, uint2(2, 0)).r;
        tex2Dstore(Storage_Show_Values, uint2(0, 0), maxCLL);
        tex2Dstore(Storage_Show_Values, uint2(1, 0), avgCLL);
        tex2Dstore(Storage_Show_Values, uint2(2, 0), minCLL);
      } break;
      case 1:
      {
        float counter_BT709  = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_BT709)).r  * 100.0001f;
        float counter_DCI_P3 = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_DCI_P3)).r * 100.0001f;
        float counter_BT2020 = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_BT2020)).r * 100.0001f;
        tex2Dstore(Storage_Show_Values, uint2(CSP_BT709,  1), counter_BT709);
        tex2Dstore(Storage_Show_Values, uint2(CSP_DCI_P3, 1), counter_DCI_P3);
        tex2Dstore(Storage_Show_Values, uint2(CSP_BT2020, 1), counter_BT2020);
      } break;
      case 2:
      {
        float counter_AP1  = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_AP1)).r  * 100.0001f;
        float counter_AP0  = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_AP0)).r  * 100.0001f;
        float counter_else = tex2Dfetch(Sampler_CSP_Counter_Final, uint2(0, CSP_ELSE)).r * 100.0001f;
        tex2Dstore(Storage_Show_Values, uint2(CSP_AP1,  1), counter_AP1);
        tex2Dstore(Storage_Show_Values, uint2(CSP_AP0,  1), counter_AP0);
        tex2Dstore(Storage_Show_Values, uint2(CSP_ELSE, 1), counter_else);
      } break;
    }
  }
  return;
}
