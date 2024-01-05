// Original functions are part of the FidelityFX SDK.
//
// Copyright (C)2023 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the “Software”), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


#include "colour_space.fxh"


// for pixel shader
float2 GetEfhiCoords(float2 Coords)
{
  return Coords + 0.5f * ReShade::PixelSize;
}


float3 PrepareForProcessing(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour /= 125.f;
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
  return saturate(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::PqTo::Linear(Colour);

#else

  return pow(Colour, 2.2f);

#endif
}

float3 PrepareForOutput(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  Colour *= 125.f;
  return Colour;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  return Csp::Trc::LinearTo::Pq(Colour);

#else

  return pow(Colour, 1.f / 2.2f);

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
void CSGetPixels(
  in  const int2             Coords,
  in  const float2           EfhiCoords,
  out       SPixelsToProcess Ptp)
{
  float4 efhiR = tex2DgatherR(ReShade::BackBuffer, EfhiCoords);
  float4 efhiG = tex2DgatherG(ReShade::BackBuffer, EfhiCoords);
  float4 efhiB = tex2DgatherB(ReShade::BackBuffer, EfhiCoords);

  Ptp.e = float3(efhiR.w, efhiG.w, efhiB.w);
  Ptp.f = float3(efhiR.z, efhiG.z, efhiB.z);
  Ptp.h = float3(efhiR.x, efhiG.x, efhiB.x);
  Ptp.i = float3(efhiR.y, efhiG.y, efhiB.y);

  Ptp.a = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1, -1)).rgb;
  Ptp.b = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 0, -1)).rgb;
  Ptp.c = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 1, -1)).rgb;
  Ptp.d = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  0)).rgb;
  Ptp.g = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  1)).rgb;

  return;
}

// for pixel shader
void PSGetPixels(
  in  const float2           Coords,
  in  const float2           EfhiCoords,
  out       SPixelsToProcess Ptp)
{
  float4 efhiR = tex2DgatherR(ReShade::BackBuffer, EfhiCoords);
  float4 efhiG = tex2DgatherG(ReShade::BackBuffer, EfhiCoords);
  float4 efhiB = tex2DgatherB(ReShade::BackBuffer, EfhiCoords);

  Ptp.e = float3(efhiR.w, efhiG.w, efhiB.w);
  Ptp.f = float3(efhiR.z, efhiG.z, efhiB.z);
  Ptp.h = float3(efhiR.x, efhiG.x, efhiB.x);
  Ptp.i = float3(efhiR.y, efhiG.y, efhiB.y);

  Ptp.a = tex2D(ReShade::BackBuffer, Coords, int2(-1, -1)).rgb;
  Ptp.b = tex2D(ReShade::BackBuffer, Coords, int2( 0, -1)).rgb;
  Ptp.c = tex2D(ReShade::BackBuffer, Coords, int2( 1, -1)).rgb;
  Ptp.d = tex2D(ReShade::BackBuffer, Coords, int2(-1,  0)).rgb;
  Ptp.g = tex2D(ReShade::BackBuffer, Coords, int2(-1,  1)).rgb;

  return;
}
