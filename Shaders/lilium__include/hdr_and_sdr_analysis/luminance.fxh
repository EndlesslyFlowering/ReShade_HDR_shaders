#pragma once


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


void FinaliseMaxAvgMinNits()
{
  const float maxNits = asfloat(atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 0, 0));
  const float minNits = asfloat(atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 2, UINT_MAX));

  float avgNits = float(atomicExchange(StorageMaxAvgMinNitsAndCspCounter, 1, 0)) / 1000.f;

  static const float2 dispatchWxH = float2(BUFFER_WIDTH, BUFFER_HEIGHT)
                                  / 16.f;

  static const float dispatchArea = dispatchWxH.x
                                  * dispatchWxH.y;

  avgNits /= dispatchArea;

  tex1Dstore(StorageConsolidated, COORDS_MAX_NITS_VALUE, maxNits);
  tex1Dstore(StorageConsolidated, COORDS_AVG_NITS_VALUE, avgNits);
  tex1Dstore(StorageConsolidated, COORDS_MIN_NITS_VALUE, minNits);
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


// calls HeatmapRgbValues with predefined parameters
float3 WaveformRgbValues(
  const float Y)
{
#ifdef IS_HDR_CSP
  // LUMINANCE_WAVEFORM_CUTOFF_POINT values match heatmap modes 1:1
  return HeatmapRgbValues(Y, LUMINANCE_WAVEFORM_CUTOFF_POINT, true);
#else
  return HeatmapRgbValues(Y, true);
#endif
}


void PS_CalcNitsPerPixel(
              float4 VPos     : SV_Position,
              float2 TexCoord : TEXCOORD0,
  out precise float  CurNits  : SV_Target0)
{
  CurNits = 0.f;

  if (_SHOW_NITS_VALUES
   || _SHOW_NITS_FROM_CURSOR
   || _SHOW_HEATMAP
   || _HIGHLIGHT_NIT_RANGE
   || _DRAW_ABOVE_NITS_AS_BLACK
   || _DRAW_BELOW_NITS_AS_BLACK
#ifdef IS_HDR_CSP
   || SHOW_CSP_MAP
#endif //IS_HDR_CSP
  )
  {

    precise const float3 pixel = tex2Dfetch(ReShade::BackBuffer, int2(VPos.xy)).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    precise float curPixelNits = dot(Csp::Mat::Bt709ToXYZ[1], pixel) * 80.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::PqTo::Nits(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], Csp::Trc::HlgTo::Nits(pixel));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

    precise float curPixelNits = dot(Csp::Mat::Bt2020ToXYZ[1], pixel) * 100.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

    precise float curPixelNits = dot(Csp::Mat::Bt709ToXYZ[1], DECODE_SDR(pixel)) * 100.f;

#else

    float curPixelNits = 0.f;

#endif //ACTUAL_COLOUR_SPACE ==

    if (curPixelNits > 0.f)
    {
      CurNits = curPixelNits;
    }
    return;

  }
  discard;
}


// 8 * 2
#if (BUFFER_WIDTH % 16 == 0)
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_X (BUFFER_WIDTH / 16)
#else
  #define GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_X (BUFFER_WIDTH / 16 + 1)
#endif

#if (BUFFER_HEIGHT % 16 == 0)
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_Y (BUFFER_HEIGHT / 16)
#else
  #define GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING
  #define GET_MAX_AVG_MIN_NITS_DISPATCH_Y (BUFFER_HEIGHT / 16 + 1)
#endif

#define GROUP_MAX_AVG_MIN_SHARED_VARIABLES (64 * 3)

groupshared float GroupMaxAvgMin[GROUP_MAX_AVG_MIN_SHARED_VARIABLES];
void CS_GetMaxAvgMinNits(uint3 GID  : SV_GroupID,
                         uint3 GTID : SV_GroupThreadID,
                         uint3 DTID : SV_DispatchThreadID)
{
  if (_SHOW_NITS_VALUES
   || (_SHOW_LUMINANCE_WAVEFORM
    && (_LUMINANCE_WAVEFORM_SHOW_MIN_NITS_LINE || _LUMINANCE_WAVEFORM_SHOW_MAX_NITS_LINE)))
  {

    float threadMaxNits = 0.f;
    float threadAvgNits = 0.f;
    float threadMinNits = FP32_MAX;

    const int xStart = DTID.x * 2;
#ifndef GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING
    const int xStop  = xStart + 2;
#else
    const int xStop  = min(xStart + 2, BUFFER_WIDTH);
#endif

    const int yStart = DTID.y * 2;
#ifndef GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING
    const int yStop  = yStart + 2;
#else
    const int yStop  = min(yStart + 2, BUFFER_HEIGHT);
#endif

    for (int x = xStart; x < xStop; x++)
    {
      for (int y = yStart; y < yStop; y++)
      {
        const float curNits = tex2Dfetch(SamplerNitsValues, int2(x, y));

        threadMaxNits = max(curNits, threadMaxNits);
        threadMinNits = min(curNits, threadMinNits);

        threadAvgNits += curNits;
      }
    }

    static const float avgXDiv =
#ifdef GET_MAX_AVG_MIN_NITS_FETCH_X_NEEDS_CLAMPING
                                 (xStop == BUFFER_WIDTH)
                               ? (uint(BUFFER_WIDTH) - uint(xStart))
                               : 2.f;
#else
                                 2.f;
#endif

    static const float avgYDiv =
#ifdef GET_MAX_AVG_MIN_NITS_FETCH_Y_NEEDS_CLAMPING
                                 (yStop == BUFFER_HEIGHT)
                               ? (uint(BUFFER_HEIGHT) - uint(yStart))
                               : 2.f;
#else
                                 2.f;
#endif

    static const float avgDiv = avgXDiv * avgYDiv;

    threadAvgNits /= avgDiv;

    const uint groupMaxAvgMinId = (DTID.x - (GID.x * 8)) | ((DTID.y - (GID.y * 8)) << 3);
    GroupMaxAvgMin[groupMaxAvgMinId]        = threadMaxNits;
    GroupMaxAvgMin[groupMaxAvgMinId | 0x40] = threadAvgNits;
    GroupMaxAvgMin[groupMaxAvgMinId | 0x80] = threadMinNits;

    barrier();

    if (all(GTID.xy == 0))
    {
      float groupMaxNits = 0.f;
      float groupAvgNits = 0.f;
      float groupMinNits = FP32_MAX;

      for (uint i = 0; i < 64; i++)
      {
        groupMaxNits = max(GroupMaxAvgMin[i],        groupMaxNits);
        groupMinNits = min(GroupMaxAvgMin[i | 0x80], groupMinNits);

        groupAvgNits += GroupMaxAvgMin[i | 0x40];
      }

      groupAvgNits /= 64.f;

      const uint groupMaxNitsAsUintRounded = asuint(groupMaxNits);
      const uint groupAvgNitsAsUintRounded = uint((groupAvgNits + 0.0005f) * 1000.f);
      const uint groupMinNitsAsUintRounded = asuint(groupMinNits);

      atomicMax(StorageMaxAvgMinNitsAndCspCounter, 0, groupMaxNitsAsUintRounded);
      atomicAdd(StorageMaxAvgMinNitsAndCspCounter, 1, groupAvgNitsAsUintRounded);
      atomicMin(StorageMaxAvgMinNitsAndCspCounter, 2, groupMinNitsAsUintRounded);
    }
    return;
  }
  return;
}
