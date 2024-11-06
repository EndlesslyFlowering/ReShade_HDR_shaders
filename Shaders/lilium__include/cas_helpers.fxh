#include "colour_space.fxh"


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

  return Csp::Trc::PqTo::Linear(Colour);

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
  in  const float2           Coords,
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

  Ptp.a = tex2D(SamplerBackBuffer, Coords, int2(-1, -1)).rgb;
  Ptp.b = tex2D(SamplerBackBuffer, Coords, int2( 0, -1)).rgb;
  Ptp.c = tex2D(SamplerBackBuffer, Coords, int2( 1, -1)).rgb;
  Ptp.d = tex2D(SamplerBackBuffer, Coords, int2(-1,  0)).rgb;
  Ptp.g = tex2D(SamplerBackBuffer, Coords, int2(-1,  1)).rgb;

  return;
}
