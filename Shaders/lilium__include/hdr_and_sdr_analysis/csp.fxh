#pragma once


//#ifndef IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION
  #define IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION NO
//#endif


#ifdef IS_HDR_CSP
texture2D TextureCsps
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = R8;
};

sampler2D<float> SamplerCsps
{
  Texture = TextureCsps;
};


void FinaliseCspCounter()
{

#ifdef IS_HDR_CSP

  precise uint counterBt709   = atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 3, 0);
  precise uint counterDciP3   = atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 4, 0);
  precise uint counterBt2020  = atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 5, 0);

#if defined(IS_FLOAT_HDR_CSP)

  precise uint counterAp0     = atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 6, 0);
  precise uint counterInvalid = atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 7, 0);

#endif //IS_FLOAT_HDR_CSP

#if (__VENDOR__ == 0x1002)
  #define TIMES_100 100.0001f
#else
  #define TIMES_100 100.f
#endif

    precise float percentageBt709   = float(counterBt709)   / PixelCountInFloat * TIMES_100;
    precise float percentageDciP3   = float(counterDciP3)   / PixelCountInFloat * TIMES_100;
    precise float percentageBt2020  = float(counterBt2020)  / PixelCountInFloat * TIMES_100;

#if defined(IS_FLOAT_HDR_CSP)

    precise float percentageAp0     = float(counterAp0)     / PixelCountInFloat * TIMES_100;
    precise float percentageInvalid = float(counterInvalid) / PixelCountInFloat * TIMES_100;

#endif //IS_FLOAT_HDR_CSP

  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_BT709,   percentageBt709);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_DCI_P3,  percentageDciP3);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_BT2020,  percentageBt2020);

#if defined(IS_FLOAT_HDR_CSP)

  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_AP0,     percentageAp0);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_INVALID, percentageInvalid);

#endif //IS_FLOAT_HDR_CSP
#endif //IS_HDR_CSP

}


bool IsCsp(precise float3 Rgb)
{
  if (all(Rgb >= 0.f))
  {
    return true;
  }
  return false;
}

#define IS_CSP_BT709   0
#define IS_CSP_DCI_P3  1
#define IS_CSP_BT2020  2
#define IS_CSP_AP0     3
#define IS_CSP_INVALID 4


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
              float2 TexCoord : TEXCOORD0,
  out precise float  CurCsp   : SV_Target0)
{
  CurCsp = 0.f;

  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR
   || SHOW_CSP_MAP)
  {
    precise const float3 pixel = tex2Dfetch(ReShade::BackBuffer, int2(VPos.xy)).rgb;

#if defined(IS_FLOAT_HDR_CSP)

#if (IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION == YES)

    const float3 absPixel = abs(pixel);
    if (absPixel.r > SMALLEST_FP16
     && absPixel.g > SMALLEST_FP16
     && absPixel.b > SMALLEST_FP16)
    {
      CurCsp = GetCsp(pixel);
    }
    else
    {
      CurCsp = IS_CSP_BT709;
    }
    return;

#else

    CurCsp = GetCsp(pixel);

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
      CurCsp = GetCsp(curPixel);
    }
    else
    {
      CurCsp = IS_CSP_BT709;
    }
    return;

#else

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
    precise const float3 curPixel = Csp::Trc::PqTo::Linear(pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    precise const float3 curPixel = Csp::Trc::HlgTo::Linear(pixel);
#endif
    CurCsp = GetCsp(curPixel);

    return;

#endif

#else

    CurCsp = IS_CSP_INVALID / 255.f;

    return;

#endif
  }
  discard;
}


// 8 * 4
#if (BUFFER_WIDTH % 32 == 0)
  #define CSP_COUNTER_DISPATCH_X (BUFFER_WIDTH / 32)
#else
  #define CSP_COUNTER_FETCH_X_NEEDS_CLAMPING
  #define CSP_COUNTER_DISPATCH_X (BUFFER_WIDTH / 32 + 1)
#endif

#if (BUFFER_HEIGHT % 32 == 0)
  #define CSP_COUNTER_DISPATCH_Y (BUFFER_HEIGHT / 32)
#else
  #define CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING
  #define CSP_COUNTER_DISPATCH_Y (BUFFER_HEIGHT / 32 + 1)
#endif

#if defined(IS_FLOAT_HDR_CSP)
  #define COUNTED_CSPS 5
#elif defined(IS_HDR10_LIKE_CSP)
  #define COUNTED_CSPS 3
#endif

#define GROUP_CSP_COUNTER_SHARED_VARIABLES (64 * COUNTED_CSPS)


groupshared uint GroupCspCounter[GROUP_CSP_COUNTER_SHARED_VARIABLES];
void CS_CountCsps(uint3 GID  : SV_GroupID,
                  uint3 GTID : SV_GroupThreadID,
                  uint3 DTID : SV_DispatchThreadID)
{
  if (SHOW_CSPS)
  {

#if defined(IS_FLOAT_HDR_CSP)
    uint counter[5] = {0,0,0,0,0};
#else
    uint counter[3] = {0,0,0};
#endif

    const int xStart = DTID.x * 4;
#ifndef CSP_COUNTER_FETCH_X_NEEDS_CLAMPING
    const int xStop  = xStart + 4;
#else
    const int xStop  = min(xStart + 4, BUFFER_WIDTH);
#endif

    const int yStart = DTID.y * 4;
#ifndef CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING
    const int yStop  = yStart + 4;
#else
    const int yStop  = min(yStart + 4, BUFFER_HEIGHT);
#endif

    for (int x = xStart; x < xStop; x++)
    {
      for (int y = yStart; y < yStop; y++)
      {
        uint curCsp = uint(tex2Dfetch(SamplerCsps, int2(x, y)) * 255.f);
        counter[curCsp]++;
      }
    }

    const uint groupCspCounterId = (DTID.x - (GID.x * 8)) | ((DTID.y - (GID.y * 8)) << 3);
    GroupCspCounter[groupCspCounterId]         = counter[0];
    GroupCspCounter[groupCspCounterId | 0x40]  = counter[1];
    GroupCspCounter[groupCspCounterId | 0x80]  = counter[2];
#if defined(IS_FLOAT_HDR_CSP)
    GroupCspCounter[groupCspCounterId | 0xC0]  = counter[3];
    GroupCspCounter[groupCspCounterId | 0x100] = counter[4];
#endif

    barrier();

    if (all(GTID.xy == 0))
    {
      uint counterBt709   = 0;
      uint counterDciP3   = 0;
      uint counterBt2020  = 0;
#if defined(IS_FLOAT_HDR_CSP)
      uint counterAp0     = 0;
      uint counterInvalid = 0;
#endif
      for (uint i = 0; i < 64; i++)
      {
        counterBt709   += GroupCspCounter[i];
        counterDciP3   += GroupCspCounter[i | 0x40];
        counterBt2020  += GroupCspCounter[i | 0x80];
#if defined(IS_FLOAT_HDR_CSP)
        counterAp0     += GroupCspCounter[i | 0xC0];
        counterInvalid += GroupCspCounter[i | 0x100];
#endif
      }
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 3, counterBt709);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 4, counterDciP3);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 5, counterBt2020);
#if defined(IS_FLOAT_HDR_CSP)
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 6, counterAp0);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 7, counterInvalid);
#endif
    }
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
#endif //IS_HDR_CSP
