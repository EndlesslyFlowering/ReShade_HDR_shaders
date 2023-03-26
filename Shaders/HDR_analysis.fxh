#include "colorspace.fxh"

//max is 32
//#ifndef THREAD_SIZE0
  #define THREAD_SIZE0 8
//#endif

//max is 1024
//#ifndef THREAD_SIZE1
  #define THREAD_SIZE1 8
//#endif

#define DISPATCH_X0 uint(float(BUFFER_WIDTH)  / THREAD_SIZE0) + 1
#define DISPATCH_Y0 uint(float(BUFFER_HEIGHT) / THREAD_SIZE0) + 1

#define DISPATCH_X1 uint(float(BUFFER_WIDTH)  / THREAD_SIZE1) + 1
#define DISPATCH_Y1 uint(float(BUFFER_HEIGHT) / THREAD_SIZE1) + 1

#define WIDTH0 uint(BUFFER_WIDTH) / 2
#define WIDTH1 uint(BUFFER_WIDTH) - WIDTH0

#define HEIGHT0 uint(BUFFER_HEIGHT) / 2
#define HEIGHT1 uint(BUFFER_HEIGHT) - HEIGHT0


uniform float2 PINGPONG
<
  source    = "pingpong";
  min       = 0;
  max       = 1;
  step      = 1;
  smoothing = 0.0;
>;


texture2D CLLvalues
{
   Width = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerCLLvalues
{
  Texture = CLLvalues;

  SRGBTexture = false;
};

storage2D storageTargetCLLvalues
{
  Texture = CLLvalues;

  MipLevel = 0;
};

//  max = (0,0); avg = (1,0); min = (2,0)
texture2D maxAvgMinCLLvalues
{
   Width = 3;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerMaxAvgMinCLLvalues
{
  Texture = maxAvgMinCLLvalues;

  SRGBTexture = false;
};

storage2D storageTargetMaxAvgMinCLLvalues
{
  Texture = maxAvgMinCLLvalues;

  MipLevel = 0;
};

//  max = (0,0); avg = (1,0); min = (2,0)
texture2D maxAvgMinCLLSHOWvalues
{
  Width = 3;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerMaxAvgMinCLLSHOWvalues
{
  Texture = maxAvgMinCLLSHOWvalues;

  SRGBTexture = false;
};

storage2D storageTargetMaxAvgMinCLLSHOWvalues
{
  Texture = maxAvgMinCLLSHOWvalues;

  MipLevel = 0;
};

texture2D intermediateCLLvalues
{
   Width = BUFFER_WIDTH;
  Height = 3;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerIntermediateCLLvalues
{
  Texture = intermediateCLLvalues;

  SRGBTexture = false;
};

storage2D storageTargetIntermediateCLLvalues
{
  Texture = intermediateCLLvalues;

  MipLevel = 0;
};

texture2D final4
{
  Width = 2;
  Height = 2;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerFinal4
{
  Texture = final4;

  SRGBTexture = false;
};

storage2D storageFinal4
{
  Texture = final4;

  MipLevel = 0;
};

#define CIE_X          735
#define CIE_Y          835
#define CIE_BG_X       935
#define CIE_BG_Y      1035
#define CIE_BG_BORDER  100
texture2D CIE_1931 <source = "CIE_1931_linear.png";>
{
  Width     = CIE_X;
  Height    = CIE_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1931
{
  Texture = CIE_1931;
};

texture2D CIE_1931_black_bg <source = "CIE_1931_black_bg_linear.png";>
{
  Width     = CIE_BG_X;
  Height    = CIE_BG_Y;
  MipLevels = 0;

  SRGBTexture = false;
};

sampler2D sampler_CIE_1931_black_bg
{
  Texture = CIE_1931_black_bg;
};

texture2D CIE_1931_cur
{
  Width     = CIE_BG_X;
  Height    = CIE_BG_Y;
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
    if (BUFFER_COLOR_SPACE == CSP_PQ || overrideCSP == 1)
    {
      output  = mul(BT709_to_BT2020_matrix, output);
      output *= whitepoint;
      output  = PQ_OETF(output);
    }
    else if (BUFFER_COLOR_SPACE == CSP_SCRGB || overrideCSP == 2)
      output *= whitepoint / 80.f;
  }
  else
    output = pow(output, 1.f / 2.2f);

  return output;
}

void calcCLL(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float  curCLL   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float curPixel;

  if (BUFFER_COLOR_SPACE == CSP_PQ)
    curPixel = PQ_EOTF(dot(BT2020_to_XYZ[1].rgb, pixel), true);
  else if (BUFFER_COLOR_SPACE == CSP_SCRGB)
    curPixel = dot(BT709_to_XYZ[1].rgb, pixel) * 80.f;
  else if (BUFFER_COLOR_SPACE == CSP_UNKNOWN)
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
  if (id.x < BUFFER_WIDTH)
  {
    float maxCLL = 0.f;
    float avgCLL = 0.f;
    float minCLL = 65504.f;

    for (int y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;

      avgCLL += curCLL;

      if (curCLL < minCLL)
        minCLL = curCLL;
    }

    avgCLL /= BUFFER_HEIGHT;

    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 0), maxCLL);
    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 1), avgCLL);
    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 2), minCLL);
  }
}

void getMaxAvgMinCLL1(uint3 id : SV_DispatchThreadID)
{
  float maxCLL = 0.f;
  float avgCLL = 0.f;
  float minCLL = 65504.f;

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curMaxCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;
    const float curAvgCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 1)).r;
    const float curMinCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 2)).r;

    if (curMaxCLL > maxCLL)
      maxCLL = curMaxCLL;

    avgCLL += curAvgCLL;

    if (curMinCLL < minCLL)
      minCLL = curMinCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(0, 0), maxCLL);
  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(1, 0), avgCLL);
  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(2, 0), minCLL);
}


// per column first
void getMaxCLL0(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH)
  {
    float maxCLL = 0.f;

    for (int y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

      if (curCLL > maxCLL)
        maxCLL = curCLL;
    }

    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 0), maxCLL);
  }
}

void getMaxCLL1(uint3 id : SV_DispatchThreadID)
{
  float maxCLL = 0.f;

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;

    if (curCLL > maxCLL)
      maxCLL = curCLL;
  }

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(0, 0), maxCLL);
}


void getMaxCLL0_NEW(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH)
  {
    switch (id.y)
    {
      case 0:
        {
          float maxCLL = 0.f;

          for (int y = 0; y < HEIGHT0; y++)
          {
            const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

            if (curCLL > maxCLL)
              maxCLL = curCLL;
          }

          tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 0), maxCLL);
        }
        break;
      case 1:
        {
          float maxCLL = 0.f;

          for (int y = HEIGHT0; y < HEIGHT1; y++)
          {
            const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

            if (curCLL > maxCLL)
              maxCLL = curCLL;
          }

          tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 1), maxCLL);
        }
        break;
    }
  }
}

void getMaxCLL1_NEW(uint3 id : SV_DispatchThreadID)
{
  switch (id.x)
  {
    case 0:
      switch (id.y)
      {
        case 0:
          {
            float maxCLL = 0.f;

            for (int x = 0; x < WIDTH0; x++)
            {
              const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;

              if (curCLL > maxCLL)
                maxCLL = curCLL;
            }

            tex2Dstore(storageFinal4, int2(0, id.y), maxCLL);
          }
          break;

        case 1:
          {
            float maxCLL = 0.f;

            for (int x = 0; x < WIDTH0; x++)
            {
              const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 1)).r;

              if (curCLL > maxCLL)
                maxCLL = curCLL;
            }

            tex2Dstore(storageFinal4, int2(0, id.y), maxCLL);
          }
          break;
      }
      break;

    case 1:
      switch (id.y)
      {
        case 0:
          {
            float maxCLL = 0.f;

            for (int x = WIDTH0; x < WIDTH1; x++)
            {
              const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;

              if (curCLL > maxCLL)
                maxCLL = curCLL;
            }

            tex2Dstore(storageFinal4, int2(1, id.y), maxCLL);
          }
          break;

        case 1:
          {
            float maxCLL = 0.f;

            for (int x = WIDTH0; x < WIDTH1; x++)
            {
              const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 1)).r;

              if (curCLL > maxCLL)
                maxCLL = curCLL;
            }

            tex2Dstore(storageFinal4, int2(1, id.y), maxCLL);
          }
          break;
      }
      break;

  }
}

void getFinalMaxCLL_NEW(uint3 id : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(storageFinal4, int2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(storageFinal4, int2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(storageFinal4, int2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(storageFinal4, int2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(0, 0), maxCLL);
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
//      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;
//
//      avgCLL += curCLL;
//    }
//
//    avgCLL /= BUFFER_HEIGHT;
//
//    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 1), avgCLL);
//  }
//}
//
//void getAvgCLL1(uint3 id : SV_DispatchThreadID)
//{
//  float avgCLL = 0.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 1)).r;
//
//    avgCLL += curCLL;
//  }
//
//  avgCLL /= BUFFER_WIDTH;
//
//  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(1, 0), avgCLL);
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
//      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;
//
//      if (curCLL < minCLL)
//        minCLL = curCLL;
//    }
//
//    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 2), minCLL);
//  }
//}
//
//void getMinCLL1(uint3 id : SV_DispatchThreadID)
//{
//  float minCLL = 65504.f;
//
//  for (int x = 0; x < BUFFER_WIDTH; x++)
//  {
//    const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 2)).r;
//
//    if (curCLL < minCLL)
//      minCLL = curCLL;
//  }
//
//  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(2, 0), minCLL);
//}


void showCLLvaluesCopy(uint3 id : SV_DispatchThreadID)
{
  if (PINGPONG.x < 0.01f
  || (PINGPONG.x > 0.24f && PINGPONG.x < 0.26f)
  || (PINGPONG.x > 0.49f && PINGPONG.x < 0.51f)
  || (PINGPONG.x > 0.74f && PINGPONG.x < 0.76f)
  ||  PINGPONG.x > 0.99f)
  {
    const float maxCLL = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(0, 0)).r;
    const float avgCLL = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(1, 0)).r;
    const float minCLL = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(2, 0)).r;
    tex2Dstore(storageTargetMaxAvgMinCLLSHOWvalues, int2(0, 0), maxCLL);
    tex2Dstore(storageTargetMaxAvgMinCLLSHOWvalues, int2(1, 0), avgCLL);
    tex2Dstore(storageTargetMaxAvgMinCLLSHOWvalues, int2(2, 0), minCLL);
  }
}


// copy over clean bg first every time
void copy_CIE_bg(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 CIE_bg   : SV_TARGET)
{
  CIE_bg = tex2D(sampler_CIE_1931_black_bg, texcoord).rgba;
}

void generate_CIE_diagram(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH && id.y < BUFFER_HEIGHT)
  {
    const float3 pixel = tex2Dfetch(ReShade::BackBuffer, id.xy).rgb;

    // get XYZ
    const float3 XYZ = BUFFER_COLOR_SPACE == CSP_PQ
                     ? mul(BT2020_to_XYZ, PQ_EOTF(pixel, false))
                     : BUFFER_COLOR_SPACE == CSP_SCRGB
                     ? mul(BT709_to_XYZ, pixel / 125.f)
                     : BUFFER_COLOR_SPACE == CSP_UNKNOWN
                     ? mul(BT2020_to_XYZ, pixel / 100.f)
                     : float3(0.f, 0.f, 0.f);
    // get xy
    const float  xyz = XYZ.x + XYZ.y + XYZ.z;
    const uint2  xy  = uint2(clamp(uint(round(XYZ.x / xyz * 1000.f)), 0, CIE_X -1),  // 1000 is the original texture size
                 CIE_Y - 1 - clamp(uint(round(XYZ.y / xyz * 1000.f)), 0, CIE_Y -1)); // clamp to texture size
                                                                                     // ^you should be able to do this
                                                                                     // via the sampler. set to clamping?

  tex2Dstore(storage_CIE_1931_cur,
             uint2(xy.x + CIE_BG_BORDER, xy.y + CIE_BG_BORDER), // adjust for the added borders of 100 pixels
             tex2Dfetch(sampler_CIE_1931, xy).rgba);
  }
}

bool isCSP(const float3 RGB)
{
  if (RGB.r >= 0.f
   && RGB.g >= 0.f
   && RGB.b >= 0.f)
    return true;
  else
    return false;
}

float getCSP(const float3 XYZ)
{
  if (isCSP(mul(XYZ_to_BT709, XYZ)))
    return 0.f;
  else if (isCSP(mul(XYZ_to_P3Display, XYZ)))
    return 1.f / 255.f;
  else if (isCSP(mul(XYZ_to_BT2020, XYZ)))
    return 2.f / 255.f;
  else if (isCSP(mul(XYZ_to_AP1, XYZ)))
    return 3.f / 255.f;
  else if (isCSP(mul(XYZ_to_AP0, XYZ)))
    return 4.f / 255.f;
  else
    return 5.f / 255.f;
}

void calcCSPs(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float  curCSP   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, texcoord).rgb;

  float3   curPixel;
  float3x3 curMatrix;

  if (BUFFER_COLOR_SPACE == CSP_PQ)
  {
    curCSP = getCSP(mul(BT2020_to_XYZ, PQ_EOTF(pixel, false)));
  }
  else if (BUFFER_COLOR_SPACE == CSP_SCRGB)
  {
    curCSP = getCSP(mul(BT709_to_XYZ, pixel));
  }
  else if (BUFFER_COLOR_SPACE == CSP_UNKNOWN)
  {
    curCSP = getCSP(mul(BT2020_to_XYZ, pixel));
  }
  else
    curCSP = 5.f / 255.f;
}

void count_CSPs_0(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH)
  {
    uint counter_BT709     = 0;
    uint counter_P3Display = 0;
    uint counter_BT2020    = 0;
    uint counter_AP1       = 0;
    uint counter_AP0       = 0;
    uint counter_else      = 0;

    for (uint y = 0; y < BUFFER_HEIGHT; y++)
    {
      const uint curCSP = uint(tex2Dfetch(sampler_CSPs, int2(id.x, y)).r * 255.f);
      if (curCSP == 0)
        counter_BT709++;
      else if (curCSP == 1)
        counter_P3Display++;
      else if (curCSP == 2)
        counter_BT2020++;
      else if (curCSP == 3)
        counter_AP1++;
      else if (curCSP == 4)
        counter_AP0++;
      else if (curCSP == 5)
        counter_else++;
    }
    tex2Dstore(storage_CSP_counter, uint2(id.x, 0), counter_BT709     / 65536.f);
    tex2Dstore(storage_CSP_counter, uint2(id.x, 1), counter_P3Display / 65536.f);
    tex2Dstore(storage_CSP_counter, uint2(id.x, 2), counter_BT2020    / 65536.f);
    tex2Dstore(storage_CSP_counter, uint2(id.x, 3), counter_AP1       / 65536.f);
    tex2Dstore(storage_CSP_counter, uint2(id.x, 4), counter_AP0       / 65536.f);
    tex2Dstore(storage_CSP_counter, uint2(id.x, 5), counter_else      / 65536.f);
  }
}

void count_CSPs_1(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_HEIGHT)
  {
    uint counter_BT709     = 0;
    uint counter_P3Display = 0;
    uint counter_BT2020    = 0;
    uint counter_AP1       = 0;
    uint counter_AP0       = 0;
    uint counter_else      = 0;

    for (uint x = 0; x < BUFFER_WIDTH; x++)
    {
      counter_BT709     += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 0)).r * 65536.f);
      counter_P3Display += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 1)).r * 65536.f);
      counter_BT2020    += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 2)).r * 65536.f);
      counter_AP1       += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 3)).r * 65536.f);
      counter_AP0       += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 4)).r * 65536.f);
      counter_else      += uint(tex2Dfetch(sampler_CSP_counter, int2(x, 5)).r * 65536.f);
    }
    tex2Dstore(storage_CSP_counter_final, uint2(0, 0), counter_BT709     / float(BUFFER_WIDTH * BUFFER_HEIGHT));
    tex2Dstore(storage_CSP_counter_final, uint2(0, 1), counter_P3Display / float(BUFFER_WIDTH * BUFFER_HEIGHT));
    tex2Dstore(storage_CSP_counter_final, uint2(0, 2), counter_BT2020    / float(BUFFER_WIDTH * BUFFER_HEIGHT));
    tex2Dstore(storage_CSP_counter_final, uint2(0, 3), counter_AP1       / float(BUFFER_WIDTH * BUFFER_HEIGHT));
    tex2Dstore(storage_CSP_counter_final, uint2(0, 4), counter_AP0       / float(BUFFER_WIDTH * BUFFER_HEIGHT));
    tex2Dstore(storage_CSP_counter_final, uint2(0, 5), counter_else      / float(BUFFER_WIDTH * BUFFER_HEIGHT));
  }  
}
