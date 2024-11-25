#include "colour_space.fxh"


HDR10_TO_LINEAR_LUT()


// for pixel shader
float2 GetEfhiCoords
(
  float2 Coords
)
{
  return Coords + 0.5f * PIXEL_SIZE;
}


float3 PrepareForProcessing
(
  float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour = Csp::Mat::ScRgbTo::Bt2020Normalised(Colour);
  return saturate(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return FetchFromHdr10ToLinearLUT(Colour);

#else

  return DECODE_SDR(Colour);

#endif
}


float3 PrepareForOutput
(
  float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Mat::Bt2020NormalisedTo::ScRgb(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::LinearTo::Pq(Colour);

#else

  return ENCODE_SDR(Colour);

#endif
}


struct SPixelsToProcess
{
  float3 a;
  float3 b;
  float3 c;
  float3 d;
  float3 e;
  float3 f;
  float3 g;
  float3 h;
  float3 i;
};


// for compute shader
void CSGetPixels
(
  in  const int2             Coords,
  in  const float2           EfhiCoords,
  out       SPixelsToProcess Ptp
)
{
  float4 efhiR = tex2DgatherR(SamplerBackBuffer, EfhiCoords);
  float4 efhiG = tex2DgatherG(SamplerBackBuffer, EfhiCoords);
  float4 efhiB = tex2DgatherB(SamplerBackBuffer, EfhiCoords);

  Ptp.e = float3(efhiR.w, efhiG.w, efhiB.w);
  Ptp.f = float3(efhiR.z, efhiG.z, efhiB.z);
  Ptp.h = float3(efhiR.x, efhiG.x, efhiB.x);
  Ptp.i = float3(efhiR.y, efhiG.y, efhiB.y);

  Ptp.a = tex2Dfetch(SamplerBackBuffer, Coords + int2(-1, -1)).rgb;
  Ptp.b = tex2Dfetch(SamplerBackBuffer, Coords + int2( 0, -1)).rgb;
  Ptp.c = tex2Dfetch(SamplerBackBuffer, Coords + int2( 1, -1)).rgb;
  Ptp.d = tex2Dfetch(SamplerBackBuffer, Coords + int2(-1,  0)).rgb;
  Ptp.g = tex2Dfetch(SamplerBackBuffer, Coords + int2(-1,  1)).rgb;

  return;
}


// for pixel shader
void PSGetPixels
(
  in  const int2             Position,
  out       SPixelsToProcess Ptp
)
{
  Ptp.e = tex2Dfetch(SamplerBackBuffer,   Position   ).rgb;
  Ptp.e = PrepareForProcessing(Ptp.e);

  const int4 abPosition = Position.xyxy + int4(-1, -1,  0, -1);

  Ptp.a = tex2Dfetch(SamplerBackBuffer, abPosition.xy).rgb;
  Ptp.a = PrepareForProcessing(Ptp.a);

  Ptp.b = tex2Dfetch(SamplerBackBuffer, abPosition.zw).rgb;
  Ptp.b = PrepareForProcessing(Ptp.b);

  const int4 cdPosition = Position.xyxy + int4( 1, -1, -1,  0);

  Ptp.c = tex2Dfetch(SamplerBackBuffer, cdPosition.xy).rgb;
  Ptp.c = PrepareForProcessing(Ptp.c);

  Ptp.d = tex2Dfetch(SamplerBackBuffer, cdPosition.zw).rgb;
  Ptp.d = PrepareForProcessing(Ptp.d);

  const int4 fgPosition = Position.xyxy + int4( 1,  0, -1,  1);

  Ptp.f = tex2Dfetch(SamplerBackBuffer, fgPosition.xy).rgb;
  Ptp.f = PrepareForProcessing(Ptp.f);

  Ptp.g = tex2Dfetch(SamplerBackBuffer, fgPosition.zw).rgb;
  Ptp.g = PrepareForProcessing(Ptp.g);

  const int4 hiPosition = Position.xyxy + int4( 0,  1,  1,  1);

  Ptp.h = tex2Dfetch(SamplerBackBuffer, hiPosition.xy).rgb;
  Ptp.h = PrepareForProcessing(Ptp.h);

  Ptp.i = tex2Dfetch(SamplerBackBuffer, hiPosition.zw).rgb;
  Ptp.i = PrepareForProcessing(Ptp.i);

  return;
}
