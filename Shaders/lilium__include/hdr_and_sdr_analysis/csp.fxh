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

  uint counterBt709   = atomicExchange(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_BT709_PERCENTAGE,   0);
  uint counterDciP3   = atomicExchange(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_DCIP3_PERCENTAGE,   0);
  uint counterBt2020  = atomicExchange(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_BT2020_PERCENTAGE,  0);

#if defined(IS_FLOAT_HDR_CSP)

  uint counterAp0     = atomicExchange(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_AP0_PERCENTAGE,     0);
  uint counterInvalid = atomicExchange(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_INVALID_PERCENTAGE, 0);

#endif //IS_FLOAT_HDR_CSP

#if (__VENDOR__ == 0x1002)
  #define TIMES_100 100.0001f
#else
  #define TIMES_100 100.f
#endif

    float percentageBt709   = float(counterBt709)   / PixelCountInFloat * TIMES_100;
    float percentageDciP3   = float(counterDciP3)   / PixelCountInFloat * TIMES_100;
    float percentageBt2020  = float(counterBt2020)  / PixelCountInFloat * TIMES_100;

#if defined(IS_FLOAT_HDR_CSP)

    float percentageAp0     = float(counterAp0)     / PixelCountInFloat * TIMES_100;
    float percentageInvalid = float(counterInvalid) / PixelCountInFloat * TIMES_100;

#endif //IS_FLOAT_HDR_CSP

  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_BT709,   percentageBt709);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_DCI_P3,  percentageDciP3);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_BT2020,  percentageBt2020);

#if defined(IS_FLOAT_HDR_CSP)

  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_AP0,     percentageAp0);
  tex1Dstore(StorageConsolidated, COORDS_PERCENTAGE_INVALID, percentageInvalid);

#endif //IS_FLOAT_HDR_CSP
#endif //IS_HDR_CSP

  return;
}


bool IsCsp(float3 Rgb)
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


float GetCsp(float3 Rgb)
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
      float4 Position : SV_Position,
  out float  CurCsp   : SV_Target0)
{
  CurCsp = 0.f;

  if (SHOW_CSPS
   || SHOW_CSP_FROM_CURSOR
   || SHOW_CSP_MAP)
  {
    const float3 pixel = tex2Dfetch(SamplerBackBuffer, int2(Position.xy)).rgb;

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
      const float3 curPixel = Csp::Trc::PqTo::Linear(pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
      const float3 curPixel = Csp::Trc::HlgTo::Linear(pixel);
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
    const float3 curPixel = Csp::Trc::PqTo::Linear(pixel);
#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)
    const float3 curPixel = Csp::Trc::HlgTo::Linear(pixel);
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


#if (BUFFER_WIDTH  % WAVE_SIZE_6_X == 0  \
  && BUFFER_HEIGHT % WAVE_SIZE_6_Y == 0)
  #define CSP_COUNTER_THREAD 6
#elif (BUFFER_WIDTH  % WAVE_SIZE_4_X == 0  \
    && BUFFER_HEIGHT % WAVE_SIZE_4_Y == 0)
  #define CSP_COUNTER_THREAD 4
#else
  #define CSP_COUNTER_THREAD 2
#endif

#define CSP_COUNTER_THREAD_SIZE (CSP_COUNTER_THREAD * CSP_COUNTER_THREAD)

#define CSP_COUNTER_GROUP_PIXELS_X (CSP_COUNTER_THREAD * WAVE64_THREAD_SIZE_X)
#define CSP_COUNTER_GROUP_PIXELS_Y (CSP_COUNTER_THREAD * WAVE64_THREAD_SIZE_Y)

#if (BUFFER_WIDTH % CSP_COUNTER_GROUP_PIXELS_X == 0)
  #define CSP_COUNTER_DISPATCH_X (BUFFER_WIDTH / CSP_COUNTER_GROUP_PIXELS_X)
#else
  #define CSP_COUNTER_FETCH_X_NEEDS_CLAMPING
  #define CSP_COUNTER_DISPATCH_X (BUFFER_WIDTH / CSP_COUNTER_GROUP_PIXELS_X + 1)
#endif

#if (BUFFER_HEIGHT % CSP_COUNTER_GROUP_PIXELS_Y == 0)
  #define CSP_COUNTER_DISPATCH_Y (BUFFER_HEIGHT / CSP_COUNTER_GROUP_PIXELS_Y)
#else
  #define CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING
  #define CSP_COUNTER_DISPATCH_Y (BUFFER_HEIGHT / CSP_COUNTER_GROUP_PIXELS_Y + 1)
#endif


groupshared uint GroupBt709;
groupshared uint GroupDciP3;
groupshared uint GroupBt2020;
#if defined(IS_FLOAT_HDR_CSP)
groupshared uint GroupAp0;
groupshared uint GroupInvalid;
#endif
void CS_CountCsps(uint3 GTID : SV_GroupThreadID,
                  uint3 DTID : SV_DispatchThreadID)
{
  if (SHOW_CSPS)
  {

    if (all(GTID.xy == 0))
    {
      GroupBt709   = 0;
      GroupDciP3   = 0;
      GroupBt2020  = 0;
#if defined(IS_FLOAT_HDR_CSP)
      GroupAp0     = 0;
      GroupInvalid = 0;
#endif
    }
    barrier();

#if defined(IS_FLOAT_HDR_CSP)
    uint counter[5] = {0,0,0,0,0};
#else
    uint counter[3] = {0,0,0};
#endif

    int2 curThreadPos = DTID.xy * CSP_COUNTER_THREAD;

    [unroll]
    for (int x = 0; x < CSP_COUNTER_THREAD; x++)
    {
      [unroll]
      for (int y = 0; y < CSP_COUNTER_THREAD; y++)
      {
        int2 curFetchPos = curThreadPos + int2(x, y);

        uint curCsp = uint(tex2Dfetch(SamplerCsps, curFetchPos) * 255.f);

        #if (defined(CSP_COUNTER_FETCH_X_NEEDS_CLAMPING)  \
  && defined(CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING))

        [branch]
        if (curFetchPos.x < BUFFER_WIDTH_INT
         && curFetchPos.y < BUFFER_HEIGHT_INT)

#elif (defined(CSP_COUNTER_FETCH_X_NEEDS_CLAMPING)  \
    || defined(CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING))

  #if defined(CSP_COUNTER_FETCH_X_NEEDS_CLAMPING)

        [branch]
        if (curFetchPos.x < BUFFER_WIDTH_INT)

  #else //defined(CSP_COUNTER_FETCH_Y_NEEDS_CLAMPING)

        [branch]
        if (curFetchPos.y < BUFFER_HEIGHT_INT)

  #endif

#endif
        {
          counter[curCsp]++;
        }
      }
    }

//    const uint groupCspCounterId = (DTID.x - (GID.x * 8)) | ((DTID.y - (GID.y * 8)) << 3);
    atomicAdd(GroupBt709,   counter[0]);
    atomicAdd(GroupDciP3,   counter[1]);
    atomicAdd(GroupBt2020,  counter[2]);
#if defined(IS_FLOAT_HDR_CSP)
    atomicAdd(GroupAp0,     counter[3]);
    atomicAdd(GroupInvalid, counter[4]);
#endif

    barrier();

    if (all(GTID.xy == 0))
    {
      atomicAdd(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_BT709_PERCENTAGE,   GroupBt709);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_DCIP3_PERCENTAGE,   GroupDciP3);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_BT2020_PERCENTAGE,  GroupBt2020);
#if defined(IS_FLOAT_HDR_CSP)
      atomicAdd(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_AP0_PERCENTAGE,     GroupAp0);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounterAndShowNumbers, POS_INVALID_PERCENTAGE, GroupInvalid);
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
