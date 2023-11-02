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


#define MIN3(A, B, C) min(min(A, B), C)

#define MAX3(A, B, C) max(max(A, B), C)

uint BitfieldInsertMask(uint src, uint ins)
{
  static const uint mask = (1 << 1) - 1;
  return (ins & mask) | (src & (~mask));
}

uint BitfieldExtract(uint src, uint off)
{
  static const uint mask = (1 << 3) - 1;
  return (src >> off) & mask;
}

uint2 RemapForQuad(uint a)
{
  return uint2(BitfieldExtract(a, 1),
               BitfieldInsertMask(BitfieldExtract(a, 3), a));
}


// For additional information on the approximation family of functions, you can refer to Michal Drobot's excellent
// presentation materials:
//
// - https://michaldrobot.files.wordpress.com/2014/05/gcn_alu_opt_digitaldragons2014.pdf
// - https://github.com/michaldrobot/ShaderFastLibs/blob/master/ShaderFastMathLib.h

float ApproximateReciprocal(float value)
{
  return asfloat(uint(0x7ef07ebb) - asuint(value));
}

float3 ApproximateReciprocal(float3 value)
{
  return asfloat(uint(0x7ef07ebb) - asuint(value));
}

float ApproximateReciprocalMedium(float value)
{
  float b = asfloat(uint(0x7ef19fff) - asuint(value));
  return b * (-b * value + 2.f);
}

float3 ApproximateReciprocalMedium(float3 value)
{
  float3 b = asfloat(uint(0x7ef19fff) - asuint(value));
  return b * (-b * value + 2.f);
}

float ApproximateSqrt(float value)
{
  return asfloat((asuint(value) >> 1) + uint(0x1fbc4639));
}

float3 ApproximateSqrt(float3 value)
{
  return asfloat((asuint(value) >> 1) + uint(0x1fbc4639));
}


void PrepareForProcessing(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour /= 125.f;
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
  Colour = saturate(Colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Colour = Csp::Trc::FromPq(Colour);

#else

  Colour = pow(Colour, 2.2f);

#endif
}

void PrepareForOutput(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
  Colour *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  Colour = Csp::Trc::ToPq(Colour);

#else

  Colour = pow(Colour, 1.f / 2.2f);

#endif
}
