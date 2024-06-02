#pragma once


texture2D TextureNitsValues
<
  pooled = true;
>
{
  Width  = BUFFER_WIDTH;
  Height = BUFFER_HEIGHT;

  Format = RGBA32F;
};

sampler2D<float4> SamplerNitsValues
{
  Texture = TextureNitsValues;
};

storage2D<float4> StorageNitsValues
{
  Texture = TextureNitsValues;
};


//#if 0
//static const uint _0_Dot_01_Percent_Pixels = BUFFER_WIDTH * BUFFER_HEIGHT * 0.01f;
//static const uint _0_Dot_01_Percent_Texture_Width = _0_Dot_01_Percent_Pixels / 16;
//
//texture2D TextureMaxNits0Dot01Percent
//<
//  pooled = true;
//>
//{
//  Width  = _0_Dot_01_Percent_Texture_Width;
//  Height = 16;
//
//  Format = R32F;
//};
//
//sampler2D<float> SamplerMaxNits0Dot01Percent
//{
//  Texture = TextureMaxNits0Dot01Percent;
//};
//
//storage2D<float> StorageMaxNits0Dot01Percent
//{
//  Texture = TextureMaxNits0Dot01Percent;
//};
//#endif


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
#ifdef IS_HDR_CSP
  uint  Mode,
#endif
  bool  WaveformOutput)
{
  float3 output;


#ifdef IS_HDR_CSP
  #define HEATMAP_STEP_0 HeatmapSteps0[Mode][0]
  #define HEATMAP_STEP_1 HeatmapSteps0[Mode][1]
  #define HEATMAP_STEP_2 HeatmapSteps0[Mode][2]
  #define HEATMAP_STEP_3 HeatmapSteps1[Mode][0]
  #define HEATMAP_STEP_4 HeatmapSteps1[Mode][1]
  #define HEATMAP_STEP_5 HeatmapSteps1[Mode][2]
#else
  #define HEATMAP_STEP_0   1.f
  #define HEATMAP_STEP_1  18.f
  #define HEATMAP_STEP_2  50.f
  #define HEATMAP_STEP_3  75.f
  #define HEATMAP_STEP_4  87.5f
  #define HEATMAP_STEP_5 100.f
#endif


  if (Y < 0.f)
  {
    output.r = 0.f;
    output.g = 0.f;
    output.b = 6.25f;
  }
  else if (Y <= HEATMAP_STEP_0) // <= 100nits
  {
    //shades of grey
    float clamped = !WaveformOutput ? Y / HEATMAP_STEP_0 * 0.25f
                                    : 0.666f;
    output.rgb = clamped;
  }
  else if (Y <= HEATMAP_STEP_1) // <= 203nits
  {
    //(blue+green) to green
    output.r = 0.f;
    output.g = 1.f;
    output.b = HeatmapFadeOut(Y, HEATMAP_STEP_0, HEATMAP_STEP_1);
  }
  else if (Y <= HEATMAP_STEP_2) // <= 400nits
  {
    //green to yellow
    output.r = HeatmapFadeIn(Y, HEATMAP_STEP_1, HEATMAP_STEP_2);
    output.g = 1.f;
    output.b = 0.f;
  }
  else if (Y <= HEATMAP_STEP_3) // <= 1000nits
  {
    //yellow to red
    output.r = 1.f;
    output.g = HeatmapFadeOut(Y, HEATMAP_STEP_2, HEATMAP_STEP_3);
    output.b = 0.f;
  }
  else if (Y <= HEATMAP_STEP_4) // <= 4000nits
  {
    //red to pink
    output.r = 1.f;
    output.g = 0.f;
    output.b = HeatmapFadeIn(Y, HEATMAP_STEP_3, HEATMAP_STEP_4);
  }
  else if(Y <= HEATMAP_STEP_5) // <= 10000nits
  {
    //pink to blue
    output.r = HeatmapFadeOut(Y, HEATMAP_STEP_4, HEATMAP_STEP_5);
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


#ifdef IS_COMPUTE_CAPABLE_API
// calls HeatmapRgbValues with predefined parameters
float3 WaveformRgbValues(
  const float Y)
{
#ifdef IS_HDR_CSP
  // WAVEFORM_CUTOFF_POINT values match heatmap modes 1:1
  return HeatmapRgbValues(Y, WAVEFORM_CUTOFF_POINT, true);
#else
  return HeatmapRgbValues(Y, true);
#endif
}
#endif


void PS_CalcNitsPerPixel(
      float4 Position : SV_Position,
  out float4 CurNits  : SV_Target0)
{
  CurNits = 0.f;

  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR
   || _SHOW_HEATMAP
#ifdef IS_COMPUTE_CAPABLE_API
   || _SHOW_WAVEFORM
#endif
   || _HIGHLIGHT_NIT_RANGE
   || _DRAW_ABOVE_NITS_AS_BLACK
   || _DRAW_BELOW_NITS_AS_BLACK
#ifdef IS_HDR_CSP
   || SHOW_GAMUT_MAP
#endif //IS_HDR_CSP
  )
  {

    const float3 pixel = tex2Dfetch(SamplerBackBuffer, int2(Position.xy)).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    float3 curRgb = pixel * 80.f;

    float curPixelNits = dot(Csp::Mat::Bt709ToXYZ[1], curRgb);

    curRgb = Csp::Mat::Bt709To::Bt2020(curRgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    float3 curRgb = Csp::Trc::PqTo::Nits(pixel);

    float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], curRgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    float3 curRgb = Csp::Trc::HlgTo::Nits(pixel);

    float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], curRgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    float3 curRgb = pixel * 100.f;

    float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], curRgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    float3 curRgb = DECODE_SDR(pixel) * 100.f;

    float curPixelNits = dot(Csp::Mat::Bt709ToXYZ[1], curRgb);

#else

    float3 curRgb = 0.f;

    float curPixelNits = 0.f;

#endif //ACTUAL_COLOUR_SPACE ==


    CurNits = float4(curRgb, curPixelNits);

    return;
  }
  discard;
}


#ifdef IS_COMPUTE_CAPABLE_API


void CS_ResetMinNits(uint3 DTID : SV_DispatchThreadID)
{
  const int2 storePos = int2(POS_MIN_NITS.x, POS_MIN_NITS.y + DTID.y);

  tex2Dstore(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, storePos, INT_MAX);

  return;
}


#if (BUFFER_WIDTH  % WAVE_SIZE_6_X == 0  \
  && BUFFER_HEIGHT % WAVE_SIZE_6_Y == 0)
  #define GET_MAX_AVG_MIN_NITS_THREAD 6
#elif (BUFFER_WIDTH  % WAVE_SIZE_5_X == 0  \
    && BUFFER_HEIGHT % WAVE_SIZE_5_Y == 0)
  #define GET_MAX_AVG_MIN_NITS_THREAD 5
#elif (BUFFER_WIDTH  % WAVE_SIZE_4_X == 0  \
    && BUFFER_HEIGHT % WAVE_SIZE_4_Y == 0)
  #define GET_MAX_AVG_MIN_NITS_THREAD 4
#elif (BUFFER_WIDTH  % WAVE_SIZE_3_X == 0  \
    && BUFFER_HEIGHT % WAVE_SIZE_3_Y == 0)
  #define GET_MAX_AVG_MIN_NITS_THREAD 3
#else
  #define GET_MAX_AVG_MIN_NITS_THREAD 2
#endif

#define GET_MAX_AVG_MIN_NITS_THREAD_SIZE (GET_MAX_AVG_MIN_NITS_THREAD * GET_MAX_AVG_MIN_NITS_THREAD)

#define GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_X (GET_MAX_AVG_MIN_NITS_THREAD * WAVE64_THREAD_SIZE_X)
#define GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_Y (GET_MAX_AVG_MIN_NITS_THREAD * WAVE64_THREAD_SIZE_Y)

#if (BUFFER_WIDTH % GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_X == 0)
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_X (BUFFER_WIDTH / GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_X)
#else
  #define GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_X (BUFFER_WIDTH / GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_X + 1)
#endif

#if (BUFFER_HEIGHT % GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_Y == 0)
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_Y (BUFFER_HEIGHT / GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_Y)
#else
  #define GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_Y (BUFFER_HEIGHT / GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_Y + 1)
#endif

#ifdef IS_FLOAT_HDR_CSP
  groupshared  int4 GroupMax;
  groupshared  int4 GroupAvg;
  groupshared  int4 GroupMin;
#else
  groupshared uint4 GroupMax;
  groupshared uint4 GroupAvg;
  groupshared uint4 GroupMin;
#endif
void CS_GetMaxAvgMinNits(uint3 GID  : SV_GroupID,
                         uint3 GTID : SV_GroupThreadID,
                         uint3 DTID : SV_DispatchThreadID)
{
  BRANCH(x)
  if (_SHOW_NITS_VALUES
   || (_SHOW_WAVEFORM
    && (_WAVEFORM_SHOW_MIN_NITS_LINE || _WAVEFORM_SHOW_MAX_NITS_LINE)))
  {

    if (all(GTID.xy == 0))
    {
      GroupMax = 0;
      GroupAvg = 0;
      GroupMin = INT_MAX;
    }
    barrier();

    float4 threadMaxNits = 0.f;
    float4 threadAvgNits = 0.f;
    float4 threadMinNits = FP32_MAX;

#if (defined(GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING)  \
  || defined(GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING))

    float avgDiv = 0.f;

#else

    static const float avgDiv = GET_MAX_AVG_MIN_NITS_THREAD_SIZE;

#endif

    int2 curThreadPos = DTID.xy * GET_MAX_AVG_MIN_NITS_THREAD;

    [unroll]
    for (int x = 0; x < GET_MAX_AVG_MIN_NITS_THREAD; x++)
    {
      [unroll]
      for (int y = 0; y < GET_MAX_AVG_MIN_NITS_THREAD; y++)
      {
        int2 curFetchPos = curThreadPos + int2(x, y);

        const float4 curNits = tex2Dfetch(SamplerNitsValues, curFetchPos);

#if (defined(GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING)  \
  && defined(GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING))

        [branch]
        if (curFetchPos.x < BUFFER_WIDTH_INT
         && curFetchPos.y < BUFFER_HEIGHT_INT)

#elif (defined(GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING)  \
    || defined(GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING))

  #if defined(GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING)

        [branch]
        if (curFetchPos.x < BUFFER_WIDTH_INT)

  #else //defined(GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING)

        [branch]
        if (curFetchPos.y < BUFFER_HEIGHT_INT)

  #endif

#endif
        {
          threadMaxNits = max(curNits, threadMaxNits);
          threadMinNits = min(curNits, threadMinNits);

          threadAvgNits += curNits;

#if (defined(GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING)  \
  || defined(GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING))

          avgDiv += 1.f;

#endif
        }
      }
    }

    threadAvgNits /= avgDiv;

#ifdef IS_FLOAT_HDR_CSP
    int4 threadMaxNitsAsInt = asint(threadMaxNits);

    static const bool4 threadMaxNitsIsNegative = threadMaxNits < 0.f;

    [flatten]
    if (threadMaxNitsIsNegative.r)
    {
      threadMaxNitsAsInt.r ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMaxNitsIsNegative.g)
    {
      threadMaxNitsAsInt.g ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMaxNitsIsNegative.b)
    {
      threadMaxNitsAsInt.b ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMaxNitsIsNegative.w)
    {
      threadMaxNitsAsInt.w ^= 0x7FFFFFFF;
    }

    const int4 threadAvgNitsAsInt = int4((threadAvgNits + (sign(threadAvgNits) * 0.0005f)) * 1000.f);

    int4 threadMinNitsAsInt = asint(threadMinNits);

    static const bool4 threadMinNitsIsNegative = threadMinNits < 0.f;

    [flatten]
    if (threadMinNitsIsNegative.r)
    {
      threadMinNitsAsInt.r ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMinNitsIsNegative.g)
    {
      threadMinNitsAsInt.g ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMinNitsIsNegative.b)
    {
      threadMinNitsAsInt.b ^= 0x7FFFFFFF;
    }
    [flatten]
    if (threadMinNitsIsNegative.w)
    {
      threadMinNitsAsInt.w ^= 0x7FFFFFFF;
    }
#else
    const uint4 threadMaxNitsAsUint = asuint(threadMaxNits);
    const uint4 threadAvgNitsAsUint = uint4((threadAvgNits + 0.0005f) * 1000.f);
    const uint4 threadMinNitsAsUint = asuint(threadMinNits);
#endif

#ifdef IS_FLOAT_HDR_CSP
    atomicMax(GroupMax.w, threadMaxNitsAsInt.w);
    atomicMax(GroupMax.r, threadMaxNitsAsInt.r);
    atomicMax(GroupMax.g, threadMaxNitsAsInt.g);
    atomicMax(GroupMax.b, threadMaxNitsAsInt.b);
    atomicAdd(GroupAvg.w, threadAvgNitsAsInt.w);
    atomicAdd(GroupAvg.r, threadAvgNitsAsInt.r);
    atomicAdd(GroupAvg.g, threadAvgNitsAsInt.g);
    atomicAdd(GroupAvg.b, threadAvgNitsAsInt.b);
    atomicMin(GroupMin.w, threadMinNitsAsInt.w);
    atomicMin(GroupMin.r, threadMinNitsAsInt.r);
    atomicMin(GroupMin.g, threadMinNitsAsInt.g);
    atomicMin(GroupMin.b, threadMinNitsAsInt.b);
#else
    atomicMax(GroupMax.w, threadMaxNitsAsUint.w);
    atomicMax(GroupMax.r, threadMaxNitsAsUint.r);
    atomicMax(GroupMax.g, threadMaxNitsAsUint.g);
    atomicMax(GroupMax.b, threadMaxNitsAsUint.b);
    atomicAdd(GroupAvg.w, threadAvgNitsAsUint.w);
    atomicAdd(GroupAvg.r, threadAvgNitsAsUint.r);
    atomicAdd(GroupAvg.g, threadAvgNitsAsUint.g);
    atomicAdd(GroupAvg.b, threadAvgNitsAsUint.b);
    atomicMin(GroupMin.w, threadMinNitsAsUint.w);
    atomicMin(GroupMin.r, threadMinNitsAsUint.r);
    atomicMin(GroupMin.g, threadMinNitsAsUint.g);
    atomicMin(GroupMin.b, threadMinNitsAsUint.b);
#endif

    barrier();

    if (all(GTID.xy == 0))
    {
      const float4 groupAvgNits = float4(GroupAvg)
                                / 1000.f
                                / float(WAVE64_THREAD_SIZE);

#ifdef IS_FLOAT_HDR_CSP
      const  int4 groupAvgNitsInt  =  int4((groupAvgNits + (sign(groupAvgNits) * 0.0005f)) * 1000.f);
#else
      const uint4 groupAvgNitsUint = uint4((groupAvgNits + 0.0005f) * 1000.f);
#endif

      const int2 averageStorePosW = int2(GID.xy % uint2(AVG_NITS_WIDTH, AVG_NITS_HEIGHT));
      const int2 averageStorePosR = averageStorePosW + int2(AVG_NITS_WIDTH,     0);
      const int2 averageStorePosG = averageStorePosW + int2(AVG_NITS_WIDTH * 2, 0);
      const int2 averageStorePosB = averageStorePosW + int2(AVG_NITS_WIDTH * 3, 0);

      atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_NITS, GroupMax.w);
      atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_R,    GroupMax.r);
      atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_G,    GroupMax.g);
      atomicMax(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_B,    GroupMax.b);

#ifdef IS_FLOAT_HDR_CSP
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosW, groupAvgNitsInt.w);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosR, groupAvgNitsInt.r);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosG, groupAvgNitsInt.g);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosB, groupAvgNitsInt.b);
#else
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosW, groupAvgNitsUint.w);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosR, groupAvgNitsUint.r);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosG, groupAvgNitsUint.g);
      atomicAdd(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, averageStorePosB, groupAvgNitsUint.b);
#endif

      atomicMin(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_NITS, GroupMin.w);
      atomicMin(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_R,    GroupMin.r);
      atomicMin(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_G,    GroupMin.g);
      atomicMin(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_B,    GroupMin.b);
    }
    return;
  }
  return;
}


void FinaliseMaxAvgMinNits()
{

#ifdef IS_FLOAT_HDR_CSP

  //max
  int4 maxRgbNitsInt = int4(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_R),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_G),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_B),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_NITS));

  const bool4 maxRgbNitsIsNegative = maxRgbNitsInt < 0;

  [flatten]
  if (maxRgbNitsIsNegative.r)
  {
    maxRgbNitsInt.r ^= 0x7FFFFFFF;
  }
  [flatten]
  if (maxRgbNitsIsNegative.g)
  {
    maxRgbNitsInt.g ^= 0x7FFFFFFF;
  }
  [flatten]
  if (maxRgbNitsIsNegative.b)
  {
    maxRgbNitsInt.b ^= 0x7FFFFFFF;
  }
  [flatten]
  if (maxRgbNitsIsNegative.w)
  {
    maxRgbNitsInt.w ^= 0x7FFFFFFF;
  }

  const float4 maxRgbNits = asfloat(maxRgbNitsInt);

  //min
  int4 minRgbNitsInt = int4(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_R),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_G),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_B),
                            tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_NITS));

  const bool4 minRgbNitsIsNegative = minRgbNitsInt < 0;

  [flatten]
  if (minRgbNitsIsNegative.r)
  {
    minRgbNitsInt.r ^= 0x7FFFFFFF;
  }
  [flatten]
  if (minRgbNitsIsNegative.g)
  {
    minRgbNitsInt.g ^= 0x7FFFFFFF;
  }
  [flatten]
  if (minRgbNitsIsNegative.b)
  {
    minRgbNitsInt.b ^= 0x7FFFFFFF;
  }
  [flatten]
  if (minRgbNitsIsNegative.w)
  {
    minRgbNitsInt.w ^= 0x7FFFFFFF;
  }

  const float4 minRgbNits = asfloat(minRgbNitsInt);

#else

  const float4 maxRgbNits = asfloat(uint4(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_R),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_G),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_B),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MAX_NITS)));

  const float4 minRgbNits = asfloat(uint4(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_R),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_G),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_B),
                                          tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, POS_MIN_NITS)));

#endif

  float4 avgRgbNits = 0.f;

  for (int x = 0; x < AVG_NITS_WIDTH; x++)
  {
    for (int y = 0; y < AVG_NITS_HEIGHT; y++)
    {
      int2 fetchPosW = int2(x, y);
      int2 fetchPosR = fetchPosW + int2(AVG_NITS_WIDTH,     0);
      int2 fetchPosG = fetchPosW + int2(AVG_NITS_WIDTH * 2, 0);
      int2 fetchPosB = fetchPosW + int2(AVG_NITS_WIDTH * 3, 0);

      avgRgbNits.w += float(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, fetchPosW)) / 1000.f;
      avgRgbNits.r += float(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, fetchPosR)) / 1000.f;
      avgRgbNits.g += float(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, fetchPosG)) / 1000.f;
      avgRgbNits.b += float(tex2Dfetch(StorageMaxAvgMinNitsAndGamutCounterAndShowNumbers, fetchPosB)) / 1000.f;
    }
  }

  static const float2 dispatchWxH = BUFFER_SIZE_FLOAT
                                  / float2(GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_X,
                                           GET_MAX_AVG_MIN_NITS_GROUP_PIXELS_Y);

  static const float dispatchArea = dispatchWxH.x
                                  * dispatchWxH.y;

  avgRgbNits /= dispatchArea;

  tex1Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxRgbNits.w);
  tex1Dstore(StorageConsolidated, COORDS_MAX_R_VALUE,    maxRgbNits.r);
  tex1Dstore(StorageConsolidated, COORDS_MAX_G_VALUE,    maxRgbNits.g);
  tex1Dstore(StorageConsolidated, COORDS_MAX_B_VALUE,    maxRgbNits.b);
  tex1Dstore(StorageConsolidated, COORDS_AVG_NITS_VALUE, avgRgbNits.w);
  tex1Dstore(StorageConsolidated, COORDS_AVG_R_VALUE,    avgRgbNits.r);
  tex1Dstore(StorageConsolidated, COORDS_AVG_G_VALUE,    avgRgbNits.g);
  tex1Dstore(StorageConsolidated, COORDS_AVG_B_VALUE,    avgRgbNits.b);
  tex1Dstore(StorageConsolidated, COORDS_MIN_NITS_VALUE, minRgbNits.w);
  tex1Dstore(StorageConsolidated, COORDS_MIN_R_VALUE,    minRgbNits.r);
  tex1Dstore(StorageConsolidated, COORDS_MIN_G_VALUE,    minRgbNits.g);
  tex1Dstore(StorageConsolidated, COORDS_MIN_B_VALUE,    minRgbNits.b);

  return;
}

#else //IS_COMPUTE_CAPABLE_API

void PS_GetMaxAvgMinNits(
  in  float4 Position : SV_Position,
  out float4 Output   : SV_Target0)
{
  const uint2 id = uint2(Position.xy);

  const uint2 arrayId = max(id - 2, 0);

  float maxNits = 0.f;
  float avgNits = 0.f;
  float minNits = FP32_MAX;

  const int intermediateXStop = INTERMEDIATE_X[arrayId.x];
  const int intermediateYStop = INTERMEDIATE_Y[arrayId.y];

  [loop]
  for (int x = 0; x < intermediateXStop; x++)
  {
    [loop]
    for (int y = 0; y < intermediateYStop; y++)
    {
      int2 xy = int2(x + INTERMEDIATE_X_0 * id.x,
                     y + INTERMEDIATE_Y_0 * id.y);

      const float curNits = tex2Dfetch(SamplerNitsValues, xy);

      maxNits  = max(curNits, maxNits);
      avgNits += curNits;
      minNits  = min(curNits, minNits);
    }
  }

  avgNits /= float(intermediateXStop * intermediateYStop);

  Output = float4(maxNits,
                  avgNits,
                  minNits,
                  1.f);
}


void VS_PrepareFinaliseGetMaxAvgMinNits(
  in  uint   VertexID : SV_VertexID,
  out float4 Position : SV_Position)
{
  static const float positions[2] =
  {
    GetPositonXCoordFromRegularXCoord(COORDS_MAX_NITS_VALUE),
    GetPositonXCoordFromRegularXCoord(COORDS_MIN_NITS_VALUE + 1)
  };

  Position = float4(positions[VertexID], 0.f, 0.f, 1.f);

  return;
}

void PS_FinaliseGetMaxAvgMinNits(
  in  float4 Position : SV_Position,
  out float  Output   : SV_Target0)
{
  const uint id = uint(Position.x - COORDS_MAX_NITS_VALUE);

  float3 maxAvgMinNits = float3(0.f, 0.f, FP32_MAX);

  [loop]
  for (int x = 0; x < TEXTURE_INTERMEDIATE_WIDTH; x++)
  {
    [loop]
    for (int y = 0; y < TEXTURE_INTERMEDIATE_HEIGHT; y++)
    {
      float3 curMaxAvgMinNits = tex2Dfetch(SamplerIntermediate, int2(x, y)).xyz;

      maxAvgMinNits[0]  = max(curMaxAvgMinNits[0], maxAvgMinNits[0]);
      maxAvgMinNits[1] += curMaxAvgMinNits[1];
      maxAvgMinNits[2]  = min(curMaxAvgMinNits[2], maxAvgMinNits[2]);
    }
  }

  maxAvgMinNits[1] /= (TEXTURE_INTERMEDIATE_WIDTH * TEXTURE_INTERMEDIATE_HEIGHT);

  Output = maxAvgMinNits[id];
}

#endif //IS_COMPUTE_CAPABLE_API
