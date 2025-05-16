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


float3 RgbModePrepareForProcessing
(
  const float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Mat::ScRgbTo::Bt2020Normalised(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return FetchFromHdr10ToLinearLUT(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  return DECODE_SDR(Colour);

#else // fallback for shader permutations

  return 0.f;

#endif
}

float3 RgbModePrepareForOutput
(
  const float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  return Csp::Mat::Bt2020NormalisedTo::ScRgb(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::LinearTo::Pq(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  return ENCODE_SDR(Colour);

#else // fallback for shader permutations

  return 0.f;

#endif
}

float3 LuminanceModePrepareForOutput
(
  const float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::LinearTo::Pq(Colour);

#else

  return Colour;

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

struct SLumianceOfPixels
{
  float aLum;
  float bLum;
  float cLum;
  float dLum;
  float eLum;
  float fLum;
  float gLum;
  float hLum;
  float iLum;
};


void GetLuminance
(
  inout SPixelsToProcess  Ptp,
  out   SLumianceOfPixels Lop
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Lop.aLum = dot(Ptp.a, Csp::Mat::ScRgbToXYZ[1]);
  Lop.bLum = dot(Ptp.b, Csp::Mat::ScRgbToXYZ[1]);
  Lop.cLum = dot(Ptp.c, Csp::Mat::ScRgbToXYZ[1]);
  Lop.dLum = dot(Ptp.d, Csp::Mat::ScRgbToXYZ[1]);
  Lop.eLum = dot(Ptp.e, Csp::Mat::ScRgbToXYZ[1]);
  Lop.fLum = dot(Ptp.f, Csp::Mat::ScRgbToXYZ[1]);
  Lop.gLum = dot(Ptp.g, Csp::Mat::ScRgbToXYZ[1]);
  Lop.hLum = dot(Ptp.h, Csp::Mat::ScRgbToXYZ[1]);
  Lop.iLum = dot(Ptp.i, Csp::Mat::ScRgbToXYZ[1]);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  float3 aRgb  = FetchFromHdr10ToLinearLUT(Ptp.a);
  float3 bRgb  = FetchFromHdr10ToLinearLUT(Ptp.b);
  float3 cRgb  = FetchFromHdr10ToLinearLUT(Ptp.c);
  float3 dRgb  = FetchFromHdr10ToLinearLUT(Ptp.d);
         Ptp.e = FetchFromHdr10ToLinearLUT(Ptp.e);
  float3 fRgb  = FetchFromHdr10ToLinearLUT(Ptp.f);
  float3 gRgb  = FetchFromHdr10ToLinearLUT(Ptp.g);
  float3 hRgb  = FetchFromHdr10ToLinearLUT(Ptp.h);
  float3 iRgb  = FetchFromHdr10ToLinearLUT(Ptp.i);

  Lop.aLum = dot(aRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.bLum = dot(bRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.cLum = dot(cRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.dLum = dot(dRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.eLum = dot(Ptp.e, Csp::Mat::Bt2020ToXYZ[1]);
  Lop.fLum = dot(fRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.gLum = dot(gRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.hLum = dot(hRgb,  Csp::Mat::Bt2020ToXYZ[1]);
  Lop.iLum = dot(iRgb,  Csp::Mat::Bt2020ToXYZ[1]);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  Lop.aLum = dot(Ptp.a, Csp::Mat::Bt709ToXYZ[1]);
  Lop.bLum = dot(Ptp.b, Csp::Mat::Bt709ToXYZ[1]);
  Lop.cLum = dot(Ptp.c, Csp::Mat::Bt709ToXYZ[1]);
  Lop.dLum = dot(Ptp.d, Csp::Mat::Bt709ToXYZ[1]);
  Lop.eLum = dot(Ptp.e, Csp::Mat::Bt709ToXYZ[1]);
  Lop.fLum = dot(Ptp.f, Csp::Mat::Bt709ToXYZ[1]);
  Lop.gLum = dot(Ptp.g, Csp::Mat::Bt709ToXYZ[1]);
  Lop.hLum = dot(Ptp.h, Csp::Mat::Bt709ToXYZ[1]);
  Lop.iLum = dot(Ptp.i, Csp::Mat::Bt709ToXYZ[1]);

#else // fallback for shader permutations

  Lop.aLum = 0.f;
  Lop.bLum = 0.f;
  Lop.cLum = 0.f;
  Lop.dLum = 0.f;
  Lop.eLum = 0.f;
  Lop.fLum = 0.f;
  Lop.gLum = 0.f;
  Lop.hLum = 0.f;
  Lop.iLum = 0.f;

#endif

  return;
}


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
  // Load a collection of samples in a 3x3 neighorhood, where e is the current pixel.
  // a b c
  // d e f
  // g h i

  Ptp.e = tex2Dfetch(SamplerBackBuffer,   Position   ).rgb;

  const int4 abPosition = Position.xyxy + int4(-1, -1,  0, -1);

  Ptp.a = tex2Dfetch(SamplerBackBuffer, abPosition.xy).rgb;
  Ptp.b = tex2Dfetch(SamplerBackBuffer, abPosition.zw).rgb;

  const int4 cdPosition = Position.xyxy + int4( 1, -1, -1,  0);

  Ptp.c = tex2Dfetch(SamplerBackBuffer, cdPosition.xy).rgb;
  Ptp.d = tex2Dfetch(SamplerBackBuffer, cdPosition.zw).rgb;

  const int4 fgPosition = Position.xyxy + int4( 1,  0, -1,  1);

  Ptp.f = tex2Dfetch(SamplerBackBuffer, fgPosition.xy).rgb;
  Ptp.g = tex2Dfetch(SamplerBackBuffer, fgPosition.zw).rgb;

  const int4 hiPosition = Position.xyxy + int4( 0,  1,  1,  1);

  Ptp.h = tex2Dfetch(SamplerBackBuffer, hiPosition.xy).rgb;
  Ptp.i = tex2Dfetch(SamplerBackBuffer, hiPosition.zw).rgb;

  return;
}
