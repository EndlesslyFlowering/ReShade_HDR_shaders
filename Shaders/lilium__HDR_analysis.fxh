#include "lilium__colorspace.fxh"

//max is 32
//#ifndef THREAD_SIZE0
  #define THREAD_SIZE0 8
//#endif

//max is 1024
//#ifndef THREAD_SIZE1
  #define THREAD_SIZE1 8
//#endif

#if BUFFER_WIDTH % THREAD_SIZE0 == 0
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0
  #define WIDTH0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X0 BUFFER_WIDTH / THREAD_SIZE0 + 1
#endif

#if BUFFER_HEIGHT % THREAD_SIZE0 == 0
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0
  #define HEIGHT0_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y0 BUFFER_HEIGHT / THREAD_SIZE0 + 1
#endif

#if BUFFER_WIDTH % THREAD_SIZE1 == 0
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1
  #define WIDTH1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_X1 BUFFER_WIDTH / THREAD_SIZE1 + 1
#endif

#if BUFFER_HEIGHT % THREAD_SIZE1 == 0
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1
  #define HEIGHT1_DISPATCH_DOESNT_OVERFLOW
#else
  #define DISPATCH_Y1 BUFFER_HEIGHT / THREAD_SIZE1 + 1
#endif

static const uint WIDTH0 = BUFFER_WIDTH / 2;
static const uint WIDTH1 = BUFFER_WIDTH - WIDTH0;

static const uint HEIGHT0 = BUFFER_HEIGHT / 2;
static const uint HEIGHT1 = BUFFER_HEIGHT - HEIGHT0;

//matches CSP_* defines in lilium__colorspace.fxh
//uniform uint CSP_OVERRIDE
//<
//  ui_label   = "override current colourspace";
//  ui_tooltip = "only scRGB, PQ and PS5 work";
//  ui_type    = "combo";
//  ui_items   = "no\0"
//               "sRGB\0"
//               "scRGB\0"
//               "PQ\0"
//               "HLG\0"
//               "PS5\0";
//> = 0;

#ifndef CSP_OVERRIDE
  #define CSP_OVERRIDE CSP_UNKNOWN
#endif

#if (BUFFER_COLOR_SPACE == CSP_SRGB && CSP_OVERRIDE == CSP_UNKNOWN) \
 || (BUFFER_COLOR_SPACE == CSP_SRGB && CSP_OVERRIDE == CSP_SRGB)    \
 || (BUFFER_COLOR_SPACE != CSP_SRGB && CSP_OVERRIDE == CSP_SRGB)
  #define ACTUAL_COLOR_SPACE CSP_SRGB
  #define FONT_BRIGHTNESS 1

#elif (BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_UNKNOWN) \
   || (BUFFER_COLOR_SPACE == CSP_SCRGB && CSP_OVERRIDE == CSP_SCRGB)   \
   || (BUFFER_COLOR_SPACE != CSP_SCRGB && CSP_OVERRIDE == CSP_SCRGB)
  #define ACTUAL_COLOR_SPACE CSP_SCRGB
  #define FONT_BRIGHTNESS 2.5375f // 203.f / 80.f

#elif (BUFFER_COLOR_SPACE == CSP_PQ && CSP_OVERRIDE == CSP_UNKNOWN) \
   || (BUFFER_COLOR_SPACE == CSP_PQ && CSP_OVERRIDE == CSP_PQ)      \
   || (BUFFER_COLOR_SPACE != CSP_PQ && CSP_OVERRIDE == CSP_PQ)
  #define ACTUAL_COLOR_SPACE CSP_PQ
  #define FONT_BRIGHTNESS 0.58068888104160783796

#elif (BUFFER_COLOR_SPACE == CSP_UNKNOWN && CSP_OVERRIDE == CSP_PS5) \
   || (BUFFER_COLOR_SPACE != CSP_UNKNOWN && CSP_OVERRIDE == CSP_PS5) 
  #define ACTUAL_COLOR_SPACE CSP_PS5
  #define FONT_BRIGHTNESS 2.03f

#else
  #define ACTUAL_COLOR_SPACE CSP_UNKNOWN
  #define FONT_BRIGHTNESS 1.f
#endif


uniform float2 PINGPONG
<
  source    = "pingpong";
  min       = 0;
  max       = 1;
  step      = 1;
  smoothing = 0.0;
>;


texture2D CLL_values
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  MipLevels = 0;

  Format = R32F;
};

sampler2D sampler_CLL_values
{
  Texture = CLL_values;

  SRGBTexture = false;
};

storage2D storage_CLL_values
{
  Texture = CLL_values;

  MipLevel = 0;
};

//  max = (0,0); avg = (1,0); min = (2,0)
texture2D max_avg_min_CLL_values
{
  Width  = 3;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D sampler_max_avg_min_CLL_values
{
  Texture = max_avg_min_CLL_values;

  SRGBTexture = false;
};

storage2D storage_max_avg_min_CLL_values
{
  Texture = max_avg_min_CLL_values;

  MipLevel = 0;
};

texture2D intermediate_CLL_values
{
  Width  = BUFFER_WIDTH;
  Height = 6;

  MipLevels = 0;

  Format = R32F;
};

sampler2D sampler_intermediate_CLL_values
{
  Texture = intermediate_CLL_values;

  SRGBTexture = false;
};

storage2D storage_intermediate_CLL_values
{
  Texture = intermediate_CLL_values;

  MipLevel = 0;
};

texture2D final_4
{
  Width  = 6;
  Height = 2;

  MipLevels = 0;

  Format = R32F;
};

sampler2D sampler_final_4
{
  Texture = final_4;

  SRGBTexture = false;
};

storage2D storage_final_4
{
  Texture = final_4;

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

sampler2D sampler_CIE_1931
{
  Texture = CIE_1931;
};

texture2D CIE_1931_black_bg <source = "lilium__CIE_1931_black_bg_linear.png";>
{
  Width     = CIE_1931_BG_X;
  Height    = CIE_1931_BG_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1931_black_bg
{
  Texture = CIE_1931_black_bg;
};

texture2D CIE_1931_cur
{
  Width     = CIE_1931_BG_X;
  Height    = CIE_1931_BG_Y;
  MipLevels = 0;
  Format    = RGBA8;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1931_cur
{
  Texture = CIE_1931_cur;
};

storage2D storage_CIE_1931_cur
{
  Texture  = CIE_1931_cur;
  MipLevel = 0;
};

#define CIE_1976_X    623
#define CIE_1976_Y    587
#define CIE_1976_BG_X 723
#define CIE_1976_BG_Y 687
texture2D CIE_1976 <source = "lilium__CIE_1976_UCS_linear.png";>
{
  Width     = CIE_1976_X;
  Height    = CIE_1976_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1976
{
  Texture = CIE_1976;
};

texture2D CIE_1976_black_bg <source = "lilium__CIE_1976_UCS_black_bg_linear.png";>
{
  Width     = CIE_1976_BG_X;
  Height    = CIE_1976_BG_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1976_black_bg
{
  Texture = CIE_1976_black_bg;
};

texture2D CIE_1976_cur
{
  Width     = CIE_1976_BG_X;
  Height    = CIE_1976_BG_Y;
  MipLevels = 0;
  Format    = RGBA8;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1976_cur
{
  Texture = CIE_1976_cur;
};

storage2D storage_CIE_1976_cur
{
  Texture  = CIE_1976_cur;
  MipLevel = 0;
};

texture2D CSPs
{
  Width     = BUFFER_WIDTH;
  Height    = BUFFER_HEIGHT;
  MipLevels = 0;
  Format    = R8;
};

sampler2D sampler_CSPs
{
  Texture    = CSPs;
  MipLODBias = 0;
};

texture2D CSP_counter
{
  Width     = BUFFER_WIDTH;
  Height    = 6;
  MipLevels = 0;
  Format    = R16;
};

sampler2D sampler_CSP_counter
{
  Texture    = CSP_counter;
  MipLODBias = 0;
};

storage2D storage_CSP_counter
{
  Texture  = CSP_counter;
  MipLevel = 0;
};

texture2D CSP_counter_final
{
  Width     = 1;
  Height    = 6;
  MipLevels = 0;
  Format    = R32F;
};

sampler2D sampler_CSP_counter_final
{
  Texture    = CSP_counter_final;
  MipLODBias = 0;
};

storage2D storage_CSP_counter_final
{
  Texture  = CSP_counter_final;
  MipLevel = 0;
};

//  max = (0, 0); avg = (1, 0); min = (2, 0)
// (0, 1) to (5, 1) = CSPs
texture2D show_values
{
  Width  = 6;
  Height = 2;

  MipLevels = 0;

  Format = R32F;
};

sampler2D sampler_show_values
{
  Texture = show_values;

  SRGBTexture = false;
};

storage2D storage_show_values
{
  Texture = show_values;

  MipLevel = 0;
};


float3 heatmapRGBvalues(
  const float Y,
  const uint  mode,
  const uint  overrideCSP,
  const float whitepoint,
  const bool  sdrOutput)
{
  float3 output;

  float r0,
      r1,
      r2,
      r3,
      r4,
      r5;

  switch (mode)
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
    const float normalizeTo = r0;
    const float clampTo = 0.25f;
    output.r = Y / normalizeTo * clampTo;
    output.g = Y / normalizeTo * clampTo;
    output.b = Y / normalizeTo * clampTo;
  }
  else if (Y <= r1) // <= 203nits
  {
    //(blue+green) to green
    const float normalizeTo = r1;
    output.r = 0.f;
    output.g = 1.f;
    output.b = 1.f - ((Y - r0) / (normalizeTo - r0));
  }
  else if (Y <= r2) // <= 400nits
  {
    //green to yellow
    const float normalizeTo = r2;
    output.r = (Y - r1) / (normalizeTo - r1);
    output.g = 1.f;
    output.b = 0.f;
  }
  else if (Y <= r3) // <= 1000nits
  {
    //yellow to red
    const float normalizeTo = r3;
    output.r = 1.f;
    output.g = 1.f - ((Y - r2) / (normalizeTo - r2));
    output.b = 0.f;
  }
  else if (Y <= r4) // <= 4000nits
  {
    //red to pink
    const float normalizeTo = r4;
    output.r = 1.f;
    output.g = 0.f;
    output.b = (Y - r3) / (normalizeTo - r3);
  }
  else if(Y <= r5) // > 4000nits
  {
    //pink to blue
    const float normalizeTo = r5;
    output.r = max(1.f - ((Y - r4) / (normalizeTo - r4)), 0.f);
    output.g = 0.f;
    output.b = 1.f;
  }
  else // > 10000nits
  {
    output.r = 6.25f;
    output.g = 0.f;
    output.b = 0.f;
  }

  if (sdrOutput == false)
  {
    if (ACTUAL_COLOR_SPACE == CSP_PQ)
    {
      output  = mul(BT709_to_BT2020, output);
      output *= whitepoint;
      output  = PQ_OETF(output);
    }
    else if (ACTUAL_COLOR_SPACE == CSP_SCRGB)
    {
      output *= whitepoint / 80.f;
    }
  }
  else
  {
    output = pow(output, 1.f / 2.2f);
  }

  return output;
}

void calcCLL(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float  curCLL   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float curPixel;

  if (ACTUAL_COLOR_SPACE == CSP_PQ)
    curPixel = PQ_EOTF(dot(BT2020_to_XYZ[1].rgb, pixel)) * 10000.f;
  else if (ACTUAL_COLOR_SPACE == CSP_SCRGB)
    curPixel = dot(BT709_to_XYZ[1].rgb, pixel) * 80.f;
  else if (ACTUAL_COLOR_SPACE == CSP_PS5)
    curPixel = dot(BT2020_to_XYZ[1].rgb, pixel) * 100.f;
  else
    curPixel = 0.f;

  curCLL = curPixel < 0.f
         ? 0.f
         : curPixel;
}

// per column first
void getMaxAvgMinCLL0(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH)
  {
#endif
    float maxCLL = 0.f;
    float avgCLL = 0.f;
    float minCLL = 65504.f;

    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;

      avgCLL += curCLL;

      if (curCLL < minCLL)
        minCLL = curCLL;
    }

    avgCLL /= BUFFER_HEIGHT;

    tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 0), maxCLL);
    tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 1), avgCLL);
    tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 2), minCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void getMaxAvgMinCLL1(uint3 id : SV_DispatchThreadID)
{
  float maxCLL = 0.f;
  float avgCLL = 0.f;
  float minCLL = 65504.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curMaxCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;
    const float curAvgCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 1)).r;
    const float curMinCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 2)).r;

    if (curMaxCLL > maxCLL)
      maxCLL = curMaxCLL;

    avgCLL += curAvgCLL;

    if (curMinCLL < minCLL)
      minCLL = curMinCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  tex2Dstore(storage_max_avg_min_CLL_values, uint2(0, 0), maxCLL);
  tex2Dstore(storage_max_avg_min_CLL_values, uint2(1, 0), avgCLL);
  tex2Dstore(storage_max_avg_min_CLL_values, uint2(2, 0), minCLL);
}


// per column first
void getMaxCLL0(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH)
  {
#endif
    float maxCLL = 0.f;

    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;
    }

    tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 0), maxCLL);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void getMaxCLL1(uint3 id : SV_DispatchThreadID)
{
  float maxCLL = 0.f;

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;

    if (curCLL > maxCLL)
      maxCLL = curCLL;
  }

  tex2Dstore(storage_max_avg_min_CLL_values, uint2(0, 0), maxCLL);
}


void getMaxAvgMinCLL0_NEW(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH)
  {
#endif
    if(id.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (uint y = 0; y < HEIGHT0; y++)
      {
        const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT0;

      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 0), maxCLL);
      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 1), avgCLL);
      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 2), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, y)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;

        avgCLL += curCLL;

        if (curCLL < minCLL)
          minCLL = curCLL;
      }

      avgCLL /= HEIGHT1;

      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 3), maxCLL);
      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 4), avgCLL);
      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, 5), minCLL);
    }
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void getMaxAvgMinCLL1_NEW(uint3 id : SV_DispatchThreadID)
{
  if (id.x == 0)
  {
    if (id.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(storage_final_4, uint2(0, 0), maxCLL);
      tex2Dstore(storage_final_4, uint2(2, 0), avgCLL);
      tex2Dstore(storage_final_4, uint2(4, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curMaxCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH0;

      tex2Dstore(storage_final_4, uint2(0, 1), maxCLL);
      tex2Dstore(storage_final_4, uint2(2, 1), avgCLL);
      tex2Dstore(storage_final_4, uint2(4, 1), minCLL);
    }
  }
  else
  {
    if (id.y == 0)
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;
        const float curAvgCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 1)).r;
        const float curMinCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 2)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(storage_final_4, uint2(1, 0), maxCLL);
      tex2Dstore(storage_final_4, uint2(3, 0), avgCLL);
      tex2Dstore(storage_final_4, uint2(5, 0), minCLL);
    }
    else
    {
      float maxCLL = 0.f;
      float avgCLL = 0.f;
      float minCLL = 65504.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curMaxCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 3)).r;
        const float curAvgCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 4)).r;
        const float curMinCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 5)).r;

        if (curMaxCLL > maxCLL)
          maxCLL = curMaxCLL;

        avgCLL += curAvgCLL;

        if (curMinCLL < minCLL)
          minCLL = curMinCLL;
      }

      avgCLL /= WIDTH1;

      tex2Dstore(storage_final_4, uint2(1, 1), maxCLL);
      tex2Dstore(storage_final_4, uint2(3, 1), avgCLL);
      tex2Dstore(storage_final_4, uint2(5, 1), minCLL);
    }
  }
}

void getFinalMaxAvgMinCLL_NEW(uint3 id : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(storage_final_4, uint2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(storage_final_4, uint2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(storage_final_4, uint2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(storage_final_4, uint2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);


  const float avgCLL0 = tex2Dfetch(storage_final_4, uint2(2, 0)).r;
  const float avgCLL1 = tex2Dfetch(storage_final_4, uint2(3, 0)).r;
  const float avgCLL2 = tex2Dfetch(storage_final_4, uint2(2, 1)).r;
  const float avgCLL3 = tex2Dfetch(storage_final_4, uint2(3, 1)).r;

  const float avgCLL = (avgCLL0 + avgCLL1 + avgCLL2 + avgCLL3) / 4.f;


  const float minCLL0 = tex2Dfetch(storage_final_4, uint2(4, 0)).r;
  const float minCLL1 = tex2Dfetch(storage_final_4, uint2(5, 0)).r;
  const float minCLL2 = tex2Dfetch(storage_final_4, uint2(4, 1)).r;
  const float minCLL3 = tex2Dfetch(storage_final_4, uint2(5, 1)).r;

  const float minCLL = min(min(min(minCLL0, minCLL1), minCLL2), minCLL3);


  tex2Dstore(storage_max_avg_min_CLL_values, uint2(0, 0), maxCLL);
  tex2Dstore(storage_max_avg_min_CLL_values, uint2(1, 0), avgCLL);
  tex2Dstore(storage_max_avg_min_CLL_values, uint2(2, 0), minCLL);
}


void getMaxCLL0_NEW(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH)
  {
#endif
    if(id.y == 0)
    {
      float maxCLL = 0.f;

      for (uint y = 0; y < HEIGHT0; y++)
      {
        const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, id.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (uint y = HEIGHT0; y < BUFFER_HEIGHT; y++)
      {
        const float curCLL = tex2Dfetch(sampler_CLL_values, uint2(id.x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_intermediate_CLL_values, uint2(id.x, id.y), maxCLL);
    }
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void getMaxCLL1_NEW(uint3 id : SV_DispatchThreadID)
{
  if (id.x == 0)
  {
    if (id.y == 0)
    {
      float maxCLL = 0.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_final_4, uint2(id.x, id.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (int x = 0; x < WIDTH0; x++)
      {
        const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 3)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_final_4, uint2(id.x, id.y), maxCLL);
    }
  }
  else
  {
    if (id.y == 0)
    {
      float maxCLL = 0.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 0)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_final_4, uint2(id.x, id.y), maxCLL);
    }
    else
    {
      float maxCLL = 0.f;

      for (int x = WIDTH0; x < BUFFER_WIDTH; x++)
      {
        const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, uint2(x, 1)).r;

        if (curCLL > maxCLL)
          maxCLL = curCLL;
      }

      tex2Dstore(storage_final_4, uint2(id.x, id.y), maxCLL);
    }
  }
}

void getFinalMaxCLL_NEW(uint3 id : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(storage_final_4, uint2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(storage_final_4, uint2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(storage_final_4, uint2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(storage_final_4, uint2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  tex2Dstore(storage_max_avg_min_CLL_values, uint2(0, 0), maxCLL);
}


// per column first
//void getAvgCLL0(uint3 id : SV_DispatchThreadID)
//{
//  if (id.x < BUFFER_WIDTH)
//  {
//    float avgCLL = 0.f;
//
//    for (int y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float curCLL = tex2Dfetch(sampler_CLL_values, int2(id.x, y)).r;
//
//      avgCLL += curCLL;
//    }
//
//    avgCLL /= BUFFER_HEIGHT;
//
//    tex2Dstore(storage_intermediate_CLL_values, int2(id.x, 1), avgCLL);
//  }
//}
//
//void getAvgCLL1(uint3 id : SV_DispatchThreadID)
//{
//  float avgCLL = 0.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, int2(x, 1)).r;
//
//    avgCLL += curCLL;
//  }
//
//  avgCLL /= BUFFER_WIDTH;
//
//  tex2Dstore(storage_max_avg_min_CLL_values, int2(1, 0), avgCLL);
//}
//
//
//// per column first
//void getMinCLL0(uint3 id : SV_DispatchThreadID)
//{
//  if (id.x < BUFFER_WIDTH)
//  {
//    float minCLL = 65504.f;
//
//    for (int y = 0; y < BUFFER_HEIGHT; y++)
//    {
//      const float curCLL = tex2Dfetch(sampler_CLL_values, int2(id.x, y)).r;
//
//      if (curCLL < minCLL)
//        minCLL = curCLL;
//    }
//
//    tex2Dstore(storage_intermediate_CLL_values, int2(id.x, 2), minCLL);
//  }
//}
//
//void getMinCLL1(uint3 id : SV_DispatchThreadID)
//{
//  float minCLL = 65504.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float curCLL = tex2Dfetch(sampler_intermediate_CLL_values, int2(x, 2)).r;
//
//    if (curCLL < minCLL)
//      minCLL = curCLL;
//  }
//
//  tex2Dstore(storage_max_avg_min_CLL_values, int2(2, 0), minCLL);
//}


// copy over clean bg first every time
void copy_CIE_1931_bg(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 CIE_bg   : SV_TARGET)
{
  CIE_bg = tex2D(sampler_CIE_1931_black_bg, texcoord).rgba;
}

void copy_CIE_1976_bg(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 CIE_bg   : SV_TARGET)
{
  CIE_bg = tex2D(sampler_CIE_1976_black_bg, texcoord).rgba;
}

void generate_CIE_diagram(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  #ifndef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH && id.y < BUFFER_HEIGHT)
  {
  #endif
#endif

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  #ifdef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  if (id.y < BUFFER_HEIGHT)
  {
  #endif
#endif

#ifndef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  #ifdef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  if (id.y < BUFFER_WIDTH)
  {
  #endif
#endif

    const float3 pixel = tex2Dfetch(ReShade::BackBuffer, id.xy).rgb;

    // get XYZ
    const float3 XYZ = ACTUAL_COLOR_SPACE == CSP_PQ
                     ? mul(BT2020_to_XYZ, PQ_EOTF(pixel))
                     : ACTUAL_COLOR_SPACE == CSP_SCRGB
                     ? mul(BT709_to_XYZ, pixel / 125.f)
                     : ACTUAL_COLOR_SPACE == CSP_PS5
                     ? mul(BT2020_to_XYZ, pixel / 100.f)
                     : float3(0.f, 0.f, 0.f);
    // get xy
    const float xyz = XYZ.x + XYZ.y + XYZ.z;
    const uint2 xy  = uint2(clamp(uint(round(XYZ.x / xyz * 1000.f)), 0, CIE_1931_X - 1),  // 1000 is the original texture size
           CIE_1931_Y - 1 - clamp(uint(round(XYZ.y / xyz * 1000.f)), 0, CIE_1931_Y - 1)); // clamp to texture size
                                                                                          // ^you should be able to do this
                                                                                          // via the sampler. set to clamping?

    // get u'v'
    const float X15Y3Z = XYZ.x
                       + 15.f * XYZ.y
                       +  3.f * XYZ.z;
    const uint2 uv     = uint2(clamp(uint(round(4.f * XYZ.x / X15Y3Z * 1000.f)), 0, CIE_1976_X - 1),
              CIE_1976_Y - 1 - clamp(uint(round(9.f * XYZ.y / X15Y3Z * 1000.f)), 0, CIE_1976_Y - 1));

  tex2Dstore(storage_CIE_1931_cur,
             uint2(xy.x + CIE_BG_BORDER, xy.y + CIE_BG_BORDER), // adjust for the added borders of 100 pixels
             tex2Dfetch(sampler_CIE_1931, xy).rgba);

  tex2Dstore(storage_CIE_1976_cur,
             uint2(uv.x + CIE_BG_BORDER, uv.y + CIE_BG_BORDER), // adjust for the added borders of 100 pixels
             tex2Dfetch(sampler_CIE_1976, uv).rgba);

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  #ifndef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  }
  #endif
#endif

#ifndef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  #ifdef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  }
  #endif
#endif

#ifndef HEIGHT1_DISPATCH_DOESNT_OVERFLOW
  #ifdef WIDTH1_DISPATCH_DOESNT_OVERFLOW
  }
  #endif
#endif

}


#if ACTUAL_COLOR_SPACE != CSP_SRGB
bool is_CSP(const float3 RGB)
{
  if (RGB.r >= 0.f
   && RGB.g >= 0.f
   && RGB.b >= 0.f)
    return true;
  else
    return false;
}

float get_CSP(const float3 XYZ)
{
#if ACTUAL_COLOR_SPACE != CSP_PQ
  if (is_CSP(mul(XYZ_to_BT709, XYZ)))
    return 0.f;
  else if (is_CSP(mul(XYZ_to_DCI_P3, XYZ)))
    return 1.f / 255.f;
  else if (is_CSP(mul(XYZ_to_BT2020, XYZ)))
    return 2.f / 255.f;
  else if (is_CSP(mul(XYZ_to_AP1, XYZ)))
    return 3.f / 255.f;
  else if (is_CSP(mul(XYZ_to_AP0, XYZ)))
    return 4.f / 255.f;
  else
    return 5.f / 255.f;
#else
  if (is_CSP(mul(XYZ_to_BT709, XYZ)))
    return 0.f;
  else if (is_CSP(mul(XYZ_to_DCI_P3, XYZ)))
    return 1.f / 255.f;
  else
    return 2.f / 255.f;
#endif
}

void calc_CSPs(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float  curCSP   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, texcoord).rgb;

  if (ACTUAL_COLOR_SPACE == CSP_PQ)
  {
    curCSP = get_CSP(mul(BT2020_to_XYZ, PQ_EOTF(pixel)));
  }
  else if (ACTUAL_COLOR_SPACE == CSP_SCRGB)
  {
    curCSP = get_CSP(mul(BT709_to_XYZ, pixel));
  }
  else if (ACTUAL_COLOR_SPACE == CSP_PS5)
  {
    curCSP = get_CSP(mul(BT2020_to_XYZ, pixel));
  }
  else
  {
    curCSP = 5.f / 255.f;
  }
}

void count_CSPs_y(uint3 id : SV_DispatchThreadID)
{
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW
  if (id.x < BUFFER_WIDTH)
  {
#endif
      uint counter_BT709  = 0;
      uint counter_DCI_P3 = 0;
      uint counter_BT2020 = 0;
#if ACTUAL_COLOR_SPACE != CSP_PQ
      uint counter_AP1    = 0;
      uint counter_AP0    = 0;
      uint counter_else   = 0;
#endif

      for (uint y = 0; y < BUFFER_HEIGHT; y++)
      {
        const uint curCSP = uint(tex2Dfetch(sampler_CSPs, uint2(id.x, y)).r * 255.f);
        if (curCSP == 0)
          counter_BT709++;
        else if (curCSP == 1)
          counter_DCI_P3++;
#if ACTUAL_COLOR_SPACE != CSP_PQ
        else if (curCSP == 2)
          counter_BT2020++;
#else
        else
          counter_BT2020++;
#endif
#if ACTUAL_COLOR_SPACE != CSP_PQ
        else if (curCSP == 3)
          counter_AP1++;
        else if (curCSP == 4)
          counter_AP0++;
        else
          counter_else++;
#endif
      }
      tex2Dstore(storage_CSP_counter, uint2(id.x, 0), counter_BT709  / 65535.f);
      tex2Dstore(storage_CSP_counter, uint2(id.x, 1), counter_DCI_P3 / 65535.f);
      tex2Dstore(storage_CSP_counter, uint2(id.x, 2), counter_BT2020 / 65535.f);
#if ACTUAL_COLOR_SPACE != CSP_PQ
      tex2Dstore(storage_CSP_counter, uint2(id.x, 3), counter_AP1    / 65535.f);
      tex2Dstore(storage_CSP_counter, uint2(id.x, 4), counter_AP0    / 65535.f);
      tex2Dstore(storage_CSP_counter, uint2(id.x, 5), counter_else   / 65535.f);
#endif
#ifndef WIDTH0_DISPATCH_DOESNT_OVERFLOW
  }
#endif
}

void count_CSPs_x(uint3 id : SV_DispatchThreadID)
{
  uint counter_BT709  = 0;
  uint counter_DCI_P3 = 0;
  uint counter_BT2020 = 0;
#if ACTUAL_COLOR_SPACE != CSP_PQ
  uint counter_AP1    = 0;
  uint counter_AP0    = 0;
  uint counter_else   = 0;
#endif

  for (uint x = 0; x < BUFFER_WIDTH; x++)
  {
    counter_BT709  += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  0)).r * 65535.f);
    counter_DCI_P3 += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  1)).r * 65535.f);
    counter_BT2020 += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  2)).r * 65535.f);
#if ACTUAL_COLOR_SPACE != CSP_PQ
    counter_AP1    += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  3)).r * 65535.f);
    counter_AP0    += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  4)).r * 65535.f);
    counter_else   += uint(tex2Dfetch(sampler_CSP_counter, uint2(x,  5)).r * 65535.f);
#endif
  }

  const float pixels = BUFFER_WIDTH * BUFFER_HEIGHT;
  tex2Dstore(storage_CSP_counter_final, uint2(0, 0), counter_BT709  / pixels);
  tex2Dstore(storage_CSP_counter_final, uint2(0, 1), counter_DCI_P3 / pixels);
  tex2Dstore(storage_CSP_counter_final, uint2(0, 2), counter_BT2020 / pixels);
#if ACTUAL_COLOR_SPACE != CSP_PQ
  tex2Dstore(storage_CSP_counter_final, uint2(0, 3), counter_AP1    / pixels);
  tex2Dstore(storage_CSP_counter_final, uint2(0, 4), counter_AP0    / pixels);
  tex2Dstore(storage_CSP_counter_final, uint2(0, 5), counter_else   / pixels);
#endif
}
#endif

void show_values_copy(uint3 id : SV_DispatchThreadID)
{
  const float pingpong = PINGPONG.x;
  if (pingpong < 0.01f
  || (pingpong > 0.32f && pingpong < 0.34f)
  || (pingpong > 0.65f && pingpong < 0.67f)
  ||  pingpong > 0.99f)
  {
    const float maxCLL = tex2Dfetch(sampler_max_avg_min_CLL_values, uint2(0, 0)).r;
    const float avgCLL = tex2Dfetch(sampler_max_avg_min_CLL_values, uint2(1, 0)).r;
    const float minCLL = tex2Dfetch(sampler_max_avg_min_CLL_values, uint2(2, 0)).r;

    tex2Dstore(storage_show_values, uint2(0, 0), maxCLL);
    tex2Dstore(storage_show_values, uint2(1, 0), avgCLL);
    tex2Dstore(storage_show_values, uint2(2, 0), minCLL);


#if ACTUAL_COLOR_SPACE != CSP_SRGB
    const float percentage_BT709  = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 0)).r * 100.0001f;
    const float percentage_DCI_P3 = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 1)).r * 100.0001f;
    const float percentage_BT2020 = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 2)).r * 100.0001f;
#if ACTUAL_COLOR_SPACE != CSP_PQ
    const float percentage_AP1    = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 3)).r * 100.0001f;
    const float percentage_AP0    = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 4)).r * 100.0001f;
    const float percentage_else   = tex2Dfetch(sampler_CSP_counter_final, uint2(0, 5)).r * 100.0001f;
#endif

    tex2Dstore(storage_show_values, uint2(0, 1), percentage_BT709);
    tex2Dstore(storage_show_values, uint2(1, 1), percentage_DCI_P3);
    tex2Dstore(storage_show_values, uint2(2, 1), percentage_BT2020);
#if ACTUAL_COLOR_SPACE != CSP_PQ
    tex2Dstore(storage_show_values, uint2(3, 1), percentage_AP1);
    tex2Dstore(storage_show_values, uint2(4, 1), percentage_AP0);
    tex2Dstore(storage_show_values, uint2(5, 1), percentage_else);
#endif

#endif
  }
}
