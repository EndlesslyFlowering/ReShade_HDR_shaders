// Original functions are part of the FidelityFX SDK.
//
// Copyright (C) 2024 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________/\_______________________________________________________________
//==============================================================================================================================
//
//                                      FSR - [RCAS] ROBUST CONTRAST ADAPTIVE SHARPENING
//
//------------------------------------------------------------------------------------------------------------------------------
// CAS uses a simplified mechanism to convert local contrast into a variable amount of sharpness.
// RCAS uses a more exact mechanism, solving for the maximum local sharpness possible before clipping.
// RCAS also has a built in process to limit sharpening of what it detects as possible noise.
// RCAS sharper does not support scaling, as it should be applied after EASU scaling.
// Pass EASU output straight into RCAS, no color conversions necessary.
//------------------------------------------------------------------------------------------------------------------------------
// RCAS is based on the following logic.
// RCAS uses a 5 tap filter in a cross pattern (same as CAS),
//    w                n
//  w 1 w  for taps  w m e
//    w                s
// Where 'w' is the negative lobe weight.
//  output = (w*(n+e+w+s)+m)/(4*w+1)
// RCAS solves for 'w' by seeing where the signal might clip out of the {0 to 1} input range,
//  0 == (w*(n+e+w+s)+m)/(4*w+1) -> w = -m/(n+e+w+s)
//  1 == (w*(n+e+w+s)+m)/(4*w+1) -> w = (1-m)/(n+e+w+s-4*1)
// Then chooses the 'w' which results in no clipping, limits 'w', and multiplies by the 'sharp' amount.
// This solution above has issues with MSAA input as the steps along the gradient cause edge detection issues.
// So RCAS uses 4x the maximum and 4x the minimum (depending on equation) in place of the individual taps.
// As well as switching from 'm' to either the minimum or maximum (depending on side), to help in energy conservation.
// This stabilizes RCAS.
// RCAS does a simple highpass which is normalized against the local contrast then shaped,
//       0.25
//  0.25  -1  0.25
//       0.25
// This is used as a noise detection filter, to reduce the effect of RCAS on grain, and focus on real edges.
//
//  GLSL example for the required callbacks :
//
//  FfxFloat16x4 FsrRcasLoadH(FfxInt16x2 p){return FfxFloat16x4(imageLoad(imgSrc,FfxInt32x2(p)));}
//  void FsrRcasInputH(inout FfxFloat16 r,inout FfxFloat16 g,inout FfxFloat16 b)
//  {
//    //do any simple input color conversions here or leave empty if none needed
//  }
//
//  FsrRcasCon need to be called from the CPU or GPU to set up constants.
//  Including a GPU example here, the 'con' value would be stored out to a constant buffer.
//
//  FfxUInt32x4 con;
//  FsrRcasCon(con,
//   0.0); // The scale is {0.0 := maximum sharpness, to N>0, where N is the number of stops (halving) of the reduction of sharpness}.
// ---------------
// RCAS sharpening supports a CAS-like pass-through alpha via,
//  #define FSR_RCAS_PASSTHROUGH_ALPHA 1
// RCAS also supports a define to enable a more expensive path to avoid some sharpening of noise.
// Would suggest it is better to apply film grain after RCAS sharpening (and after scaling) instead of using this define,
//  #define FSR_RCAS_DENOISE 1


#include "colour_space.fxh"


HDR10_TO_LINEAR_LUT()


uniform float SHARPEN_AMOUNT
<
  ui_label   = "sharpness amount";
  ui_tooltip = "Even a value of 0 applies a bit of sharpness!";
  ui_type    = "drag";
  ui_min     = 0.f;
  ui_max     = 1.f;
  ui_step    = 0.001f;
> = 0.5f;

uniform uint RCAS_MODE
<
  ui_label   = "mode";
  ui_tooltip = "to do the sharpening:"
          "\n" " - linear uses linearised RGB values"
          "\n" " - gamma and original use in-gamma RGB values"
          "\n"
          "\n" "for the noise removal calculation:"
          "\n" " - linear uses the relative luminance"
          "\n" " - gamma uses luma"
          "\n" " - original uses an approximated luma";
  ui_type    = "combo";
  ui_items   = "linear\0"
               "gamma\0"
               "original\0";
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
> = 0;
#else
> = 1;
#endif

#define RCAS_MODE_LINEAR   0
#define RCAS_MODE_GAMMA    1
#define RCAS_MODE_ORIGINAL 2

uniform bool ENABLE_NOISE_REMOVAL
<
  ui_label   = "enable noise removal";
  ui_tooltip = "Suppresses ringing and halos a bit."
          "\n" "Can also remove film grain and other noise!";
  ui_type    = "radio";
> = true;


void PrepareForProcessing
(
  inout float3 B,
  inout float3 D,
  inout float3 E,
  inout float3 F,
  inout float3 H
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  B = Csp::Mat::ScRgbTo::Bt2020Normalised(B);
  D = Csp::Mat::ScRgbTo::Bt2020Normalised(D);
  E = Csp::Mat::ScRgbTo::Bt2020Normalised(E);
  F = Csp::Mat::ScRgbTo::Bt2020Normalised(F);
  H = Csp::Mat::ScRgbTo::Bt2020Normalised(H);

  B = saturate(B);
  D = saturate(D);
  E = saturate(E);
  F = saturate(F);
  H = saturate(H);

  BRANCH()
  if (RCAS_MODE != RCAS_MODE_LINEAR)
  {
    B = Csp::Trc::LinearTo::Pq(B);
    D = Csp::Trc::LinearTo::Pq(D);
    E = Csp::Trc::LinearTo::Pq(E);
    F = Csp::Trc::LinearTo::Pq(F);
    H = Csp::Trc::LinearTo::Pq(H);
  }

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  BRANCH()
  if (RCAS_MODE == RCAS_MODE_LINEAR)
  {
    B = FetchFromHdr10ToLinearLUT(B);
    D = FetchFromHdr10ToLinearLUT(D);
    E = FetchFromHdr10ToLinearLUT(E);
    F = FetchFromHdr10ToLinearLUT(F);
    H = FetchFromHdr10ToLinearLUT(H);
  }

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  BRANCH()
  if (RCAS_MODE == RCAS_MODE_LINEAR)
  {
    B = DECODE_SDR(B);
    D = DECODE_SDR(D);
    E = DECODE_SDR(E);
    F = DECODE_SDR(F);
    H = DECODE_SDR(H);
  }

#else

  B = 0.f;
  D = 0.f;
  E = 0.f;
  F = 0.f;
  H = 0.f;

#endif

  return;
}


float3 PrepareForOutput
(
  float3 Colour
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  BRANCH()
  if (RCAS_MODE != RCAS_MODE_LINEAR)
  {
    Colour = Csp::Trc::PqTo::Linear(Colour);
  }

  Colour = Csp::Mat::Bt2020NormalisedTo::ScRgb(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  BRANCH()
  if (RCAS_MODE == RCAS_MODE_LINEAR)
  {
    Colour = Csp::Trc::LinearTo::Pq(Colour);
  }

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  BRANCH()
  if (RCAS_MODE == RCAS_MODE_LINEAR)
  {
    Colour = ENCODE_SDR(Colour);
  }

#else

  Colour = 0.f;

#endif

  return Colour;
}


void GetLuma
(
  const float3 B,
  const float3 D,
  const float3 E,
  const float3 F,
  const float3 H,
  inout float  LB,
  inout float  LD,
  inout float  LE,
  inout float  LF,
  inout float  LH
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
  || ACTUAL_COLOUR_SPACE == CSP_HDR10)

  LB = dot(B, Csp::Mat::Bt2020ToXYZ[1] * 2.f);
  LD = dot(D, Csp::Mat::Bt2020ToXYZ[1] * 2.f);
  LE = dot(E, Csp::Mat::Bt2020ToXYZ[1] * 2.f);
  LF = dot(F, Csp::Mat::Bt2020ToXYZ[1] * 2.f);
  LH = dot(H, Csp::Mat::Bt2020ToXYZ[1] * 2.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  LB = dot(B, Csp::Mat::Bt709ToXYZ[1] * 2.f);
  LD = dot(D, Csp::Mat::Bt709ToXYZ[1] * 2.f);
  LE = dot(E, Csp::Mat::Bt709ToXYZ[1] * 2.f);
  LF = dot(F, Csp::Mat::Bt709ToXYZ[1] * 2.f);
  LH = dot(H, Csp::Mat::Bt709ToXYZ[1] * 2.f);

#else

  LB = 0.f;
  LD = 0.f;
  LE = 0.f;
  LF = 0.f;
  LH = 0.f;

#endif

  return;
}


float3 RCas
(
  const int2  Position,
  const float Sharpness
)
{
  // Algorithm uses minimal 3x3 pixel neighborhood.
  //    b
  //  d e f
  //    h
  int4 bdPosition = Position.xyxy + int4(0, -1, -1,  0);
  int4 fhPosition = Position.xyxy + int4(1,  0,  0,  1);

  bdPosition = clamp(bdPosition, int4(0, 0, 0, 0), int4(BUFFER_SIZE_MINUS_1_INT, BUFFER_SIZE_MINUS_1_INT));
  fhPosition = clamp(fhPosition, int4(0, 0, 0, 0), int4(BUFFER_SIZE_MINUS_1_INT, BUFFER_SIZE_MINUS_1_INT));

  float3 b = tex2Dfetch(SamplerBackBuffer, bdPosition.xy).rgb;
  float3 d = tex2Dfetch(SamplerBackBuffer, bdPosition.zw).rgb;
  float3 e = tex2Dfetch(SamplerBackBuffer,   Position).rgb;
  float3 f = tex2Dfetch(SamplerBackBuffer, fhPosition.xy).rgb;
  float3 h = tex2Dfetch(SamplerBackBuffer, fhPosition.zw).rgb;

  PrepareForProcessing(b, d, e, f, h);

  // Min and max of ring.
  float3 min4Rgb = MIN4(b, d, f, h);
  float3 max4Rgb = MAX4(b, d, f, h);

  // Immediate constants for peak range.
  static const float2 peakC = float2(1.f, -4.f);

  // Limiters, these need to be high precision RCPs.
  float3 hitMinRgb = min4Rgb
                   * rcp(4.f * max4Rgb);

  float3 hitMaxRgb = (peakC.x - max4Rgb)
                   * rcp(4.f * min4Rgb + peakC.y);

  float3 lobeRgb = max(-hitMinRgb, hitMaxRgb);

// This is set at the limit of providing unnatural results for sharpening.
//0.25f - (1.f / 16.f)
#define FSR_RCAS_LIMIT 0.1875f

  float lobe = max(float(-FSR_RCAS_LIMIT),
                   min(MAXRGB(lobeRgb), 0.f)) * Sharpness;

  // Apply noise removal.
  BRANCH()
  if (ENABLE_NOISE_REMOVAL)
  {
    // Luma times 2.
    float bL;
    float dL;
    float eL;
    float fL;
    float hL;

    BRANCH()
    if (RCAS_MODE != RCAS_MODE_ORIGINAL) //RCAS_MODE == RCAS_MODE_LINEAR || RCAS_MODE == RCAS_MODE_GAMMA
    {
      GetLuma(b,  d,  e,  f,  h,
              bL, dL, eL, fL, hL);
    }
    else //if (RCAS_MODE == RCAS_MODE_ORIGINAL)
    {
      bL = b.b * 0.5f + (b.r * 0.5f + b.g);
      dL = d.b * 0.5f + (d.r * 0.5f + d.g);
      eL = e.b * 0.5f + (e.r * 0.5f + e.g);
      fL = f.b * 0.5f + (f.r * 0.5f + f.g);
      hL = h.b * 0.5f + (h.r * 0.5f + h.g);
    }

    // Noise detection.
    float nz = 0.25f * bL
             + 0.25f * dL
             + 0.25f * fL
             + 0.25f * hL
             - eL;

    float maxL = MAX3(MAX3(bL, dL, eL), fL, hL);
    float minL = MIN3(MIN3(bL, dL, eL), fL, hL);

    nz = saturate(abs(nz) * rcp(maxL - minL));
    nz = -0.5f * nz + 1.f;

    lobe *= nz;
  }

  // Resolve, which needs the medium precision rcp approximation to avoid visible tonality changes.
  float  rcpL = rcp(4.f * lobe + 1.f);
  float3 pix  = ((b + d + h + f) * lobe + e) * rcpL;

  pix = PrepareForOutput(pix);

  return pix;
}
