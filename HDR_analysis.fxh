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
  Height = 2;

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

  if (BUFFER_COLOR_SPACE == CSP_PQ)
    curCLL = PQ_EOTF(dot(BT2020_to_XYZ[1].rgb, pixel));
  else if (BUFFER_COLOR_SPACE == CSP_SCRGB)
    curCLL = dot(BT709_to_XYZ[1].rgb, pixel) * 80.f;
  else
    curCLL = 0.f;
}

// calculate per column first
void calcMaxCLL0(uint3 id : SV_DispatchThreadID)
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

void calcMaxCLL0_NEW(uint3 id : SV_DispatchThreadID)
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


void calcMaxCLL1(uint3 id : SV_DispatchThreadID)
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

void calcMaxCLL1_NEW(uint3 id : SV_DispatchThreadID)
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

void calcFinalMaxCLL_NEW(uint3 id : SV_DispatchThreadID)
{
  const float maxCLL0 = tex2Dfetch(storageFinal4, int2(0, 0)).r;
  const float maxCLL1 = tex2Dfetch(storageFinal4, int2(1, 0)).r;
  const float maxCLL2 = tex2Dfetch(storageFinal4, int2(0, 1)).r;
  const float maxCLL3 = tex2Dfetch(storageFinal4, int2(1, 1)).r;

  const float maxCLL = max(max(max(maxCLL0, maxCLL1), maxCLL2), maxCLL3);

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(0, 0), maxCLL);
}

// calculate per column first
void calcAvgCLL0(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH)
  {
    float avgCLL = 0.f;

    for (int y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

      avgCLL = avgCLL + curCLL;
    }

    avgCLL /= BUFFER_HEIGHT;

    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 0), avgCLL);
  }
}


void calcAvgCLL1(uint3 id : SV_DispatchThreadID)
{
  float avgCLL = 0.f;

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;

    avgCLL += curCLL;
  }

  avgCLL /= BUFFER_WIDTH;

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(1, 0), avgCLL);
}

// calculate per column first
void calcMinCLL0(uint3 id : SV_DispatchThreadID)
{
  if (id.x < BUFFER_WIDTH)
  {
    float minCLL = 10001.f;

    for (int y = 0; y < BUFFER_HEIGHT; y++)
    {
      const float curCLL = tex2Dfetch(samplerCLLvalues, int2(id.x, y)).r;

      if (curCLL < minCLL)
        minCLL = curCLL;
    }

    tex2Dstore(storageTargetIntermediateCLLvalues, int2(id.x, 0), minCLL);
  }
}


void calcMinCLL1(uint3 id : SV_DispatchThreadID)
{
  float minCLL = 10001.f;

  for (int x = 0; x < BUFFER_WIDTH; x++)
  {
    const float curCLL = tex2Dfetch(samplerIntermediateCLLvalues, int2(x, 0)).r;

    if (curCLL < minCLL)
      minCLL = curCLL;
  }

  tex2Dstore(storageTargetMaxAvgMinCLLvalues, int2(2, 0), minCLL);
}

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
