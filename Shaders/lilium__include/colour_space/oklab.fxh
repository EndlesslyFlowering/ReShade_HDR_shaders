
namespace Csp
{

  namespace OkLab
  {

//  Copyright (c) 2021 Björn Ottosson
//
//  Permission is hereby granted, free of charge, to any person obtaining a copy of
//  this software and associated documentation files (the "Software"), to deal in
//  the Software without restriction, including without limitation the rights to
//  use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
//  of the Software, and to permit persons to whom the Software is furnished to do
//  so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included in all
//  copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
//  SOFTWARE.

    //RGB BT.709->OKLMS
    static const float3x3 Bt709ToOkLms =
      float3x3
      (
        0.412176460f,  0.536273956f, 0.0514403730f,
        0.211909204f,  0.680717885f, 0.107399843f,
        0.0883448123f, 0.281853973f, 0.630280852f
      );

    //RGB BT.2020->OKLMS
    static const float3x3 Bt2020ToOkLms =
      float3x3
      (
        0.616688430f, 0.360159069f, 0.0230432935f,
        0.265140205f, 0.635856509f, 0.0990302339f,
        0.100150644f, 0.204004317f, 0.696324706f
      );

    //OKL'M'S'->OKLab
    static const float3x3 G3OkLmsToOkLab =
      float3x3
      (
        0.2104542553f,  0.7936177850f, -0.0040720468f,
        1.9779984951f, -2.4285922050f,  0.4505937099f,
        0.0259040371f,  0.7827717662f, -0.8086757660f
      );

    //OKLab->OKL'M'S'
    static const float3x3 OkLabToG3OkLms =
      float3x3
      (
        1.f,  0.3963377774f,  0.2158037573f,
        1.f, -0.1055613458f, -0.0638541728f,
        1.f, -0.0894841775f, -1.2914855480f
      );

    //OKLMS->RGB BT.709
    static const float3x3 OkLmsToBt709 =
      float3x3
      (
         4.07718706f,    -3.30762243f,   0.230859190f,
        -1.26857650f,     2.60968708f,  -0.341155737f,
        -0.00419654231f, -0.703399658f,  1.70679605f
      );

    //OKLMS->RGB BT.2020
    static const float3x3 OkLmsToBt2020 =
      float3x3
      (
         2.14014029f,   -1.24635589f,   0.106431722f,
        -0.884832441f,   2.16317272f,  -0.278361588f,
        -0.0485790595f, -0.454490900f,  1.50235629f
      );


// provided matrices
//
//    //RGB BT.709->OKLMS
//    #define  Bt709ToOkLms = float3x3(
//      0.4122214708f, 0.5363325363f, 0.0514459929f, \
//      0.2119034982f, 0.6806995451f, 0.1073969566f, \
//      0.0883024619f, 0.2817188376f, 0.6299787005f) \
//
//    //OKLMS->RGB BT.709
//    #define OkLmsToBt709 float3x3 ( \
//       4.0767416621f, -3.3077115913f,  0.2309699292f, \
//      -1.2684380046f,  2.6097574011f, -0.3413193965f, \
//      -0.0041960863f, -0.7034186147f,  1.7076147010f) \

    namespace OkLabTo
    {

      //OKLab->OKLCh°
      float3 OkLch(const float3 OkLab)
      {
        float C = sqrt(OkLab.y * OkLab.y
                     + OkLab.z * OkLab.z);

        float h = atan2(OkLab.z, OkLab.y);

        return float3(OkLab.x, C, h);
      }

    } //OkLabTo

    namespace OkLchTo
    {

      //OKLCh°->OKLab
      float3 OkLab(const float3 OkLch)
      {
        float2 ab;
        sincos(OkLch[2], ab[1], ab[0]);

        ab *= OkLch.y;

        return float3(OkLch.x, ab);
      }

    } //OkLchTo

    namespace OkLmsTo
    {

      //OKLMS->OKL'M'S'
      float3 G3OkLms(float3 Lms)
      {
        //apply gamma 3
        return sign(Lms) * pow(abs(Lms), 1.f / 3.f);
      }

    } //OkLmsTo

    namespace G3OkLmsTo
    {

      //OKL'M'S'->OKLab
      float3 OkLab(float3 G3Lms)
      {
        return mul(G3OkLmsToOkLab, G3Lms);
      }

      //OKL'M'S'->OKLMS
      float3 OkLms(float3 G3Lms)
      {
        //remove gamma 3
        return G3Lms * G3Lms * G3Lms;
      }

    } //G3OkLmsTo

    namespace OkLabTo
    {

      //OKLab->OKL'M'S'
      float3 G3OkLms(float3 Lab)
      {
        return mul(OkLabToG3OkLms, Lab);
      }

    } //OkLabTo

    namespace Bt709To
    {

      //RGB BT.709->OKLab
      float3 OkLab(float3 Rgb)
      {
        //to OKLMS
        float3 OkLms = mul(Bt709ToOkLms, Rgb);

        //to OKL'M'S'
        //apply gamma 3
        float3 g3OkLms = Csp::OkLab::OkLmsTo::G3OkLms(OkLms);

        //to OKLab
        return Csp::OkLab::G3OkLmsTo::OkLab(g3OkLms);
      }

    } //Bt709To

    namespace Bt2020To
    {

      //RGB BT.2020->OKLab
      float3 OkLab(float3 Rgb)
      {
        //to OKLMS
        float3 OkLms = mul(Bt2020ToOkLms, Rgb);

        //to OKL'M'S'
        //apply gamma 3
        float3 g3OkLms = Csp::OkLab::OkLmsTo::G3OkLms(OkLms);

        //to OKLab
        return Csp::OkLab::G3OkLmsTo::OkLab(g3OkLms);
      }

    } //Bt2020To

    namespace OkLabTo
    {

      //OKLab->RGB BT.709
      float3 Bt709(float3 Lab)
      {
        //to OKL'M'S'
        float3 g3OkLms = mul(OkLabToG3OkLms, Lab);

        //to OKLMS
        //remove gamma 3
        float3 okLms = Csp::OkLab::G3OkLmsTo::OkLms(g3OkLms);

        //to RGB BT.709
        return mul(OkLmsToBt709, okLms);
      }

      //OKLab->RGB BT.2020
      float3 Bt2020(float3 Lab)
      {
        //to OKL'M'S'
        float3 g3OkLms = mul(OkLabToG3OkLms, Lab);

        //to OKLMS
        //remove gamma 3
        float3 okLms = Csp::OkLab::G3OkLmsTo::OkLms(g3OkLms);

        //to RGB BT.2020
        return mul(OkLmsToBt2020, okLms);
      }

    } //OkLabTo


    // Finds the maximum saturation possible for a given hue that fits in sRGB
    // Saturation here is defined as S = C/L
    // a and b must be normalized so a^2 + b^2 == 1
    float ComputeMaxSaturation(float2 ab)
    {
      // Max saturation will be when one of r, g or b goes below zero.

      // Select different coefficients depending on which component goes below zero first
      float k0, k1, k2, k3, k4;

      float3 wLms;

      if (-1.88170328f * ab.x - 0.80936493f * ab.y > 1)
      {
        // Red component
        k0 = 1.19086277f;
        k1 = 1.76576728f;
        k2 = 0.59662641f;
        k3 = 0.75515197f;
        k4 = 0.56771245f;

        wLms.rgb = OkLmsToBt709[0].rgb;
      }
      else if (1.81444104f * ab.x - 1.19445276f * ab.y > 1)
      {
        // Green component
        k0 =  0.73956515f;
        k1 = -0.45954404f;
        k2 =  0.08285427f;
        k3 =  0.12541070f;
        k4 =  0.14503204f;

        wLms.rgb = OkLmsToBt709[1].rgb;
      }
      else
      {
        // Blue component
        k0 =  1.35733652f;
        k1 = -0.00915799f;
        k2 = -1.15130210f;
        k3 = -0.50559606f;
        k4 =  0.00692167f;

        wLms.rgb = OkLmsToBt709[2].rgb;
      }

      // Approximate max saturation using a polynomial:
      float s = k0
              + k1 * ab.x
              + k2 * ab.y
              + k3 * ab.x * ab.x
              + k4 * ab.x * ab.y;

      // Do one step Halley's method to get closer
      // this gives an error less than 10e6, except for some blue hues where the dS/dh is close to infinite
      // this should be sufficient for most applications, otherwise do two/three steps

      float3 kLms = mul(OkLabToG3OkLms, float3(0.f, ab));

      {
        float3 g3Lms = 1.f + s * kLms;

        float3 intermediateLms = g3Lms * g3Lms;

        float3 lms = intermediateLms * g3Lms;

        float3 g3LmsdS  = 3.f * kLms * intermediateLms;
        float3 g3LmsdS2 = 6.f * kLms * kLms * g3Lms;

        float3 f = mul(float3x3(lms,
                                g3LmsdS,
                                g3LmsdS2), wLms);

        s = s
          - f.x * f.y
          / (f.y * f.y - 0.5f * f.x * f.z);
      }

      return s;
    }

    // finds L_cusp and C_cusp for a given hue
    // a and b must be normalized so a^2 + b^2 == 1
    float2 FindCusp(float2 ab)
    {
      // First, find the maximum saturation (saturation S = C/L)
      float sCusp = ComputeMaxSaturation(ab);

      float2 lcCusp;

      // Convert to linear sRGB to find the first point where at least one of r, g or b >= 1:
      float3 rgbAtMax = Csp::OkLab::OkLabTo::Bt709(float3(1.f, sCusp * ab));

      lcCusp.x = pow(1.f / max(max(rgbAtMax.r, rgbAtMax.g), rgbAtMax.b), 1.f / 3.f);
      lcCusp.y = lcCusp.x * sCusp;

      return lcCusp;
    }

    float2 ToSt(float2 LC)
    {
      return LC.y / float2(LC.x,
                           1.f - LC.x);
    }

    static const float ToeK1 = 0.206f;
    static const float ToeK2 = 0.03f;
    static const float ToeK3 = (1.f + ToeK1) / (1.f + ToeK2);

    // toe function for L_r
    float Toe(float X)
    {
      float i0 = ToeK3 * X - ToeK1;
      return 0.5f * (i0 + sqrt(i0 * i0 + 4.f * ToeK2 * ToeK3 * X));
    }

    // inverse toe function for L_r
    float ToeInv(float X)
    {
      return (X * X     + ToeK1 * X)
           / (X * ToeK3 + ToeK2 * ToeK3);
    }

    float2 GetStMid(float2 ab)
    {

      float s = 0.11516993f
              + 1.f / (7.44778970f
                     + 4.15901240f * ab.y
                     + ab.x * (-2.19557347f
                              + 1.75198401f * ab.y
                              + ab.x * (-2.13704948f
                                      - 10.02301043f * ab.y
                                      + ab.x * (-4.24894561f
                                               + 5.38770819f * ab.y
                                               + 4.69891013f * ab.x))));

      float t = 0.11239642f
              + 1.f / (1.61320320f
                     - 0.68124379f * ab.y
                     + ab.x * (0.40370612f
                             + 0.90148123f * ab.y
                             + ab.x * (-0.27087943f
                                      + 0.61223990f * ab.y
                                      + ab.x * (0.00299215f
                                              - 0.45399568f * ab.y
                                              - 0.14661872f * ab.x))));

      return float2(s, t);
    }

    float FindGamutIntersection(
      float2 ab,
      float  L1,
      float  C1,
      float  L0,
      float2 cusp)
    {
      // Find the intersection for upper and lower half seprately
      float t;

      if (((L1 - L0) * cusp.y - (cusp.x - L0) * C1) <= 0.f)
      {
        // Lower half
        t = cusp.y * L0 / (C1 * cusp.x + cusp.y * (L0 - L1));
      }
      else
      {
        // Upper half
        // First intersect with triangle
        t = cusp.y * (L0 - 1.f) / (C1 * (cusp.x - 1.f) + cusp.y * (L0 - L1));

        // Then one step Halley's method
        {
          float dL = L1 - L0;
          float dC = C1;

          float3 kLms = mul(OkLabToG3OkLms, float3(0.f, ab));

          float3 g3LmsDt = dL + dC * kLms;

          // If higher accuracy is required, 2 or 3 iterations of the following block can be used:
          {
            float L = L0 * (1.f - t) + t * L1;
            float C = t * C1;

            float3 g3Lms = L + C * kLms;

            float3 intermediateLms = g3Lms * g3Lms;

            float3 lms = g3Lms * intermediateLms;

            float3 lmsDt  = 3.f * g3LmsDt * intermediateLms;
            float3 lmsDt2 = 6.f * g3LmsDt * g3LmsDt * g3Lms;

            static const float3x3 iLms = float3x3(lms.xyz,
                                                  lmsDt.xyz,
                                                  lmsDt2.xyz);

            static const float3 ir = OkLmsToBt709[0].rgb;

            static const float3 ig = OkLmsToBt709[1].rgb;

            static const float3 ib = OkLmsToBt709[2].rgb;

            float3 r = mul(iLms, ir);

            r.x -= 1.f;

            float u_r = r.y / (r.y * r.y - 0.5f * r.x * r.z);
            float t_r = -r.x * u_r;

            float3 g = mul(iLms, ig);

            g.x -= 1.f;

            float u_g = g.y / (g.y * g.y - 0.5f * g.x * g.z);
            float t_g = -g.x * u_g;

            float3 b = mul(iLms, ib);

            b.x -= 1.f;

            float u_b = b.y / (b.y * b.y - 0.5f * b.x * b.z);
            float t_b = -b.x * u_b;

            t_r = u_r >= 0.f ? t_r : FP32_MAX;
            t_g = u_g >= 0.f ? t_g : FP32_MAX;
            t_b = u_b >= 0.f ? t_b : FP32_MAX;

            t += min(t_r, min(t_g, t_b));
          }
        }
      }

      return t;
    }

    float3 GetCs(float3 Lab)
    {
      float2 cusp = FindCusp(Lab.yz);

      float   C_max = FindGamutIntersection(Lab.yz, Lab.x, 1.f, Lab.x, cusp);
      float2 ST_max = ToSt(cusp);

      // Scale factor to compensate for the curved part of gamut shape:
      float k = C_max / min((Lab.x * ST_max.x),
                            (1.f - Lab.x) * ST_max.y);

      float C_mid;
      {
        float2 ST_mid = GetStMid(Lab.yz);

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        float2 C_ab = Lab.x * ST_mid;

        C_ab.y = ST_mid.y - C_ab.y;

        C_mid = 0.9f * k * sqrt(sqrt(1.f
                                   / (1.f / (C_ab.x * C_ab.x * C_ab.x * C_ab.x)
                                    + 1.f / (C_ab.y * C_ab.y * C_ab.y * C_ab.y))));
      }

      float C_0;
      {
        // for C_0, the shape is independent of hue, so ST are constant. Values picked to roughly be the average values of ST.
        float C_a = Lab.x * 0.4f;
        float C_b = (1.f - Lab.x) * 0.8f;

        // Use a soft minimum function, instead of a sharp triangle shape to get a smooth value for chroma.
        C_0 = sqrt(1.f
                 / (1.f / (C_a * C_a)
                  + 1.f / (C_b * C_b)));
      }

      return float3(C_0, C_mid, C_max);
    }

    namespace OkLabTo
    {

      //OKLab->OKHSV
      float3 OkHsv(float3 Lab)
      {
        float2 LC;

        LC.x = Lab.x;
        LC.y = sqrt(Lab.y * Lab.y
                  + Lab.z * Lab.z);

        float2 ab = Lab.yz / LC.y;

        float3 hsv;

        hsv.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) * _1_DIV_PI;

        float2 cusp = Csp::OkLab::FindCusp(ab);

        float2 stMax = Csp::OkLab::ToSt(cusp);

        float s0 = 0.5f;
        float k = 1.f - s0 / stMax.x;

        // first we find L_v, C_v, L_vt and C_vt

        float t = stMax.y / (LC.y + LC.x * stMax.y);
        float2 LC_v = LC * t;

        float L_vt = ToeInv(LC_v.x);
        float C_vt = LC_v.y * L_vt / LC_v.x;

        // we can then use these to invert the step that compensates for the toe and the curved top part of the triangle:
        float3 rgbScale = Csp::OkLab::OkLabTo::Bt709(float3(L_vt,
                                                            ab * C_vt));

        float scaleL = pow(1.f / max(max(rgbScale.r, rgbScale.g), max(rgbScale.b, 0.f)), 1.f / 3.f);

        LC /= scaleL;

        float toeL = Toe(LC.x);

        LC.y = LC.y * toeL / LC.x;
        LC.x = toeL;

        // we can now compute v and s:

        hsv.z = LC.x / LC_v.x;
        hsv.y = (s0 + stMax.y) * LC_v.y / ((stMax.y * s0) + stMax.y * k * LC_v.y);

        return hsv;
      }

      //OKLab->OKHSL
      float3 OkHsl(float3 Lab)
      {
        float C;

        float2 ab = Lab.yz;

        C = sqrt(Lab.y * Lab.y
               + Lab.z * Lab.z);

        ab /= C;

        float3 hsl;

        hsl.x = 0.5f + 0.5f * atan2(-Lab.z, -Lab.y) * _1_DIV_PI;

        float3 cs = GetCs(float3(Lab.x, ab));

        float C_0   = cs.x;
        float C_mid = cs.y;
        float C_max = cs.z;

        // Inverse of the interpolation in okhsl_to_srgb:
        float mid    = 0.8f;
        float midInv = 1.25f;

        float s;
        if (C < C_mid)
        {
          float k1 = mid * C_0;

          float k2 = 1.f
                   - k1 / C_mid;

          float t = C
                  / (k1 + k2 * C);

          hsl.y = t * mid;
        }
        else
        {
          float k0 = C_mid;

          float k1 = (1.f - mid)
                   * C_mid * C_mid
                   * midInv * midInv / C_0;

          float k2 = 1.f
                   - k1 / (C_max - C_mid);

          float t = (C - k0) / (k1 + k2 * (C - k0));

          hsl.y = mid + (1.f - mid) * t;
        }

        hsl.z = Csp::OkLab::Toe(Lab.x);

        return hsl;
      }

    }

    namespace Bt709To
    {

      //RGB BT.709->OKHSV
      float3 OkHsv(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt709To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsv(lab);
      }

      //RGB BT.709->OKHSL
      float3 OkHsl(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt709To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsl(lab);
      }

    } //Bt709To

    namespace Bt2020To
    {

      //RGB BT.2020->OKHSV
      float3 OkHsv(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt2020To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsv(lab);
      }

      //RGB BT.2020->OKHSL
      float3 OkHsl(float3 Rgb)
      {
        float3 lab = Csp::OkLab::Bt2020To::OkLab(Rgb);

        return Csp::OkLab::OkLabTo::OkHsl(lab);
      }

    } //Bt2020To

    namespace OkHsvTo
    {
      //OKHSV->OKLab
      float3 OkLab(float3 Hsv)
      {
        float i_ab = PI_2 * Hsv.x;

        float2 ab;
        sincos(i_ab, ab[1], ab[0]);

        float2 cusp = Csp::OkLab::FindCusp(ab);

        float2 stMax = Csp::OkLab::ToSt(cusp);

        float s0 = 0.5f;
        float k = 1.f - s0 / stMax.x;

        // first we compute L and V as if the gamut is a perfect triangle:

        // L, C when v==1:
        float i_num = Hsv.y * s0;
        float i_den = s0 + stMax.y - stMax.y * k * Hsv.y;

        float i_div = i_num / i_den;

        float L_v =     1.f - i_div;
        float C_v = stMax.y * i_div;

        float2 LC = float2(L_v, C_v) * Hsv.z;

        // then we compensate for both toe and the curved top part of the triangle:
        float L_vt = Csp::OkLab::ToeInv(L_v);
        float C_vt = C_v * L_vt / L_v;

        float L_new = Csp::OkLab::ToeInv(LC.x);

        LC.y = LC.y * L_new / LC.x;
        LC.x = L_new;

        float3 rgbScale = Csp::OkLab::OkLabTo::Bt709(float3(L_vt,
                                                            ab * C_vt));

        float scaleL = pow(1.f / max(max(rgbScale.r, rgbScale.g), max(rgbScale.b, 0.f)), 1.f / 3.f);

        LC *= scaleL;

        return float3(LC.x,
                      ab * LC.y);
      }

      //OKHSV->BT.709
      float3 Bt709(float3 Hsv)
      {
        float3 lab = Csp::OkLab::OkHsvTo::OkLab(Hsv);

        return Csp::OkLab::OkLabTo::Bt709(lab);
      }

      //OKHSV->BT.2020
      float3 Bt2020(float3 Hsv)
      {
        float3 lab = Csp::OkLab::OkHsvTo::OkLab(Hsv);

        return Csp::OkLab::OkLabTo::Bt2020(lab);
      }
    } //OkHsvTo

    namespace OkHslTo
    {
      //OKHSL->OKLab
      float3 OkLab(float3 Hsl)
      {
        if (Hsl.z == 1.f)
        {
          return 1.f;
        }
        else if (Hsl.z == 0.f)
        {
          return 0.f;
        }

        float  L;
        float2 ab;

        float i_ab = PI_2 * Hsl.x;

        L = Csp::OkLab::ToeInv(Hsl.z);

        sincos(i_ab, ab[1], ab[0]);

        float3 cs = Csp::OkLab::GetCs(float3(L, ab));

        float C_0   = cs.x;
        float C_mid = cs.y;
        float C_max = cs.z;

        // Interpolate the three values for C so that:
        // At s=0: dC/ds = C_0, C=0
        // At s=0.8: C=C_mid
        // At s=1.0: C=C_max

        float mid    = 0.8f;
        float midInv = 1.25f;

        float C,
              t,
              k0,
              k1,
              k2;

        if (Hsl.y < mid)
        {
          t = midInv * Hsl.y;

          k1 = mid * C_0;
          k2 = 1.f
             - k1 / C_mid;

          C = t * k1 / (1.f - k2 * t);
        }
        else
        {
          float i0 = 1.f - mid;

          t = (Hsl.y - mid) / i0;

          k0 = C_mid;

          k1 = i0
             * C_mid * C_mid
             * midInv * midInv
             / C_0;

          k2 = 1.f
             - k1 / (C_max - C_mid);

          C = k0
            + t * k1
            / (1.f - k2 * t);
        }

        return float3(L,
                      ab * C);
      }

      //OKHSL->BT.709
      float3 Bt709(float3 Hsl)
      {
        float3 lab = Csp::OkLab::OkHslTo::OkLab(Hsl);

        return Csp::OkLab::OkLabTo::Bt709(lab);
      }

      //OKHSL->BT.2020
      float3 Bt2020(float3 Hsl)
      {
        float3 lab = Csp::OkLab::OkHslTo::OkLab(Hsl);

        return Csp::OkLab::OkLabTo::Bt2020(lab);
      }
    } //OkHslTo

  } //OkLab

} //Csp
