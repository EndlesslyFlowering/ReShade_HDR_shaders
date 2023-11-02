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
#include "cas_helpers.fxh"


float3 CasSharpenOnly(
  const int2   Coords,
  const float2 EfhiCoords)
{
  // Load a collection of samples in a 3x3 neighorhood, where e is the current pixel.
  // a b c
  // d e f
  // g h i

  float4 efhiR = tex2DgatherR(ReShade::BackBuffer, EfhiCoords);
  float4 efhiG = tex2DgatherG(ReShade::BackBuffer, EfhiCoords);
  float4 efhiB = tex2DgatherB(ReShade::BackBuffer, EfhiCoords);

  float3 e = float3(efhiR.w, efhiG.w, efhiB.w);
  float3 f = float3(efhiR.z, efhiG.z, efhiB.z);
  float3 h = float3(efhiR.x, efhiG.x, efhiB.x);
  float3 i = float3(efhiR.y, efhiG.y, efhiB.y);

  float3 a = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1, -1)).rgb;
  float3 b = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 0, -1)).rgb;
  float3 c = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 1, -1)).rgb;
  float3 d = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  0)).rgb;
  float3 g = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  1)).rgb;


  PrepareForProcessing(a);
  PrepareForProcessing(b);
  PrepareForProcessing(c);
  PrepareForProcessing(d);
  PrepareForProcessing(e);
  PrepareForProcessing(f);
  PrepareForProcessing(g);
  PrepareForProcessing(h);
  PrepareForProcessing(i);


  // Soft min and max.
  //  a b c             b
  //  d e f * 0.5  +  d e f * 0.5
  //  g h i             h
  // These are 2.0x bigger (factored out the extra multiply).
  float3 minRgb = MIN3(MIN3(d, e, f), b, h);
  float3 maxRgb = MAX3(MAX3(d, e, f), b, h);

  if (BETTER_DIAGONALS)
  {
    float3 minRgb2 = MIN3(MIN3(minRgb, a, c), g, i);
    minRgb += + minRgb2;

    float3 maxRgb2 = MAX3(MAX3(maxRgb, a, c), g, i);
    maxRgb = maxRgb + maxRgb2;
  }

  float3 rcpMaxRgb;

  // Smooth minimum distance to signal limit divided by smooth max.
  if (USE_PRECISE_MATH)
  {
    rcpMaxRgb = rcp(maxRgb);
  }
  else
  {
    rcpMaxRgb = ApproximateReciprocal(maxRgb);
  }

  float3 amplifyRgb;

  if (USE_PRECISE_MATH)
  {
    amplifyRgb = saturate(min(minRgb, 2.f - maxRgb) * rcpMaxRgb);
  }
  else
  {
    amplifyRgb = saturate(min(minRgb, 1.f - maxRgb) * rcpMaxRgb);
  }

  // Shaping amount of sharpening.
  if (USE_PRECISE_MATH)
  {
    amplifyRgb = sqrt(amplifyRgb);
  }
  else
  {
    amplifyRgb = ApproximateSqrt(amplifyRgb);
  }

  // Filter shape.
  //  0 w 0
  //  w 1 w
  //  0 w 0
  float peak = -rcp(lerp(8.f, 5.f, SHARPEN_AMOUNT));

  float3 output;

  if (WEIGH_BY_ALL_CHANNELS)
  {
    float3 weight = amplifyRgb * peak;

    float3 rcpWeight = 1.f + 4.f * weight;

    if (USE_PRECISE_MATH)
    {
      rcpWeight = rcp(rcpWeight);
    }
    else
    {
      rcpWeight = ApproximateReciprocalMedium(rcpWeight);
    }

    output = (b * weight
            + d * weight
            + f * weight
            + h * weight
            + e)
           * rcpWeight;
  }
  else
  {
    float weight = amplifyRgb.g * peak;

    float rcpWeight = 1.f + 4.f * weight;

    // Filter using green coef only, depending on dead code removal to strip out the extra overhead.
    if (USE_PRECISE_MATH)
    {
      rcpWeight = rcp(rcpWeight);
    }
    else
    {
      rcpWeight = ApproximateReciprocalMedium(rcpWeight);
    }

    output = (b * weight
            + d * weight
            + f * weight
            + h * weight
            + e)
           * rcpWeight;
  }

  output = lerp(e, output, APPLY_AMOUNT);

  PrepareForOutput(output);

  return output;
}


float3 CasSharpenAndUpscale(
  const int2   Coords,
  const float2 FgjkCoords)
{
  //  a b c d
  //  e f g h
  //  i j k l
  //  m n o p
  // Working these 4 results.
  //  +-----+-----+
  //  |     |     |
  //  |  f..|..g  |
  //  |  .  |  .  |
  //  +-----+-----+
  //  |  .  |  .  |
  //  |  j..|..k  |
  //  |     |     |
  //  +-----+-----+

  float4 fgjkR = tex2DgatherR(ReShade::BackBuffer, FgjkCoords);
  float4 fgjkG = tex2DgatherG(ReShade::BackBuffer, FgjkCoords);
  float4 fgjkB = tex2DgatherB(ReShade::BackBuffer, FgjkCoords);

  float3 f = float3(fgjkR.w, fgjkG.w, fgjkB.w);
  float3 g = float3(fgjkR.z, fgjkG.z, fgjkB.z);
  float3 j = float3(fgjkR.x, fgjkG.x, fgjkB.x);
  float3 k = float3(fgjkR.y, fgjkG.y, fgjkB.y);

  float3 a = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1, -1)).rgb;
  float3 b = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 0, -1)).rgb;
  float3 c = tex2Dfetch(ReShade::BackBuffer, Coords + int2( 1, -1)).rgb;
  float3 e = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  0)).rgb;
  float3 i = tex2Dfetch(ReShade::BackBuffer, Coords + int2(-1,  1)).rgb;


  PrepareForProcessing(a);
  PrepareForProcessing(b);
  PrepareForProcessing(c);
  PrepareForProcessing(e);
  PrepareForProcessing(f);
  PrepareForProcessing(g);
  PrepareForProcessing(i);
  PrepareForProcessing(j);
  PrepareForProcessing(k);


  // Soft min and max.
  // These are 2.0x bigger (factored out the extra multiply).
  //  a b c             b
  //  e f g * 0.5  +  e f g * 0.5  [F]
  //  i j k             j
  float3 minRgb = MIN3(MIN3(b, e, f), g, j);
  float3 mxfRgb = MAX3(MAX3(b, e, f), g, j);

  if (BETTER_DIAGONALS)
  {
    float3 mnfRgb2 = MIN3(MIN3(minRgb, a, c), i, k);
    minRgb += mnfRgb2;

    float3 mxfRgb2 = MAX3(MAX3(mxfRgb, a, c), i, k);
    mxfRgb += mxfRgb2;
  }

  float3 rcpMfRgb;

  // Smooth minimum distance to signal limit divided by smooth max.
  if (USE_PRECISE_MATH)
  {
    rcpMfRgb = rcp(mxfRgb);
  }
  else
  {
    rcpMfRgb = ApproximateReciprocal(mxfRgb);
  }

  float3 ampfRgb;

  if (BETTER_DIAGONALS)
  {
    ampfRgb = saturate(min(minRgb, 2.f - mxfRgb) * rcpMfRgb);
  }
  else
  {
    ampfRgb = saturate(min(minRgb, 1.f - mxfRgb) * rcpMfRgb);
  }

  // Shaping amount of sharpening.
  if (USE_PRECISE_MATH)
  {
    ampfRgb = sqrt(ampfRgb);
  }
  else
  {
    ampfRgb = ApproximateSqrt(ampfRgb);
  }

  // Filter shape.
  //  0 w 0
  //  w 1 w
  //  0 w 0
  float peak = -rcp(lerp(8.0, 5.0, SHARPEN_AMOUNT));

  float3 wfRgb = ampfRgb * peak;

  // Thin edges to hide bilinear interpolation (helps diagonals).
  static const float thinB = 1.f / 32.f;

  float s = thinB + mxfRgb.g - minRgb.g;

  if (USE_PRECISE_MATH)
  {
    s = rcp(s);
  }
  else
  {
    s = ApproximateReciprocal(s);
  }

  // Final weighting.
  //    b c
  //  e f g h
  //  i j k l
  //    n o
  //  _____  _____  _____  _____
  //         fs        gt
  //
  //  _____  _____  _____  _____
  //  fs      s gt  fs  t     gt
  //         ju        kv
  //  _____  _____  _____  _____
  //         fs        gt
  //  ju      u kv  ju  v     kv
  //  _____  _____  _____  _____
  //
  //         ju        kv

  float3 wfRgb_x_s = wfRgb * s;


  float3 output;

  if (WEIGH_BY_ALL_CHANNELS)
  {
    float3 rcpWeight = 4.f * wfRgb_x_s + s;

    if (USE_PRECISE_MATH)
    {
      rcpWeight = rcp(rcpWeight);
    }
    else
    {
      rcpWeight = ApproximateReciprocalMedium(rcpWeight);
    }

    output = saturate((b * wfRgb_x_s
                     + e * wfRgb_x_s
                     + f * s
                     + g * wfRgb_x_s
                     + j * wfRgb_x_s)
                    * rcpWeight);
  }
  else
  {
    float rcpWeight = 4.f * wfRgb_x_s.g + s;

    // Using green coef only, depending on dead code removal to strip out the extra overhead.
    if (USE_PRECISE_MATH)
    {
      rcpWeight = rcp(rcpWeight);
    }
    else
    {
      rcpWeight = ApproximateReciprocalMedium(rcpWeight);
    }

    output = saturate((b * wfRgb_x_s.g
                     + e * wfRgb_x_s.g
                     + f * s
                     + g * wfRgb_x_s.g
                     + j * wfRgb_x_s.g)
                    * rcpWeight);
  }

  output = lerp(f, output, APPLY_AMOUNT);

  PrepareForOutput(output);

  return output;
}


float2 GetEfhiCoords(uint2 Coords)
{
  return (float2(Coords) + 0.5f)
       / ReShade::ScreenSize;
}
