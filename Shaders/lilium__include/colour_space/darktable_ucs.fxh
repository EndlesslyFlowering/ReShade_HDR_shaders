
namespace Csp
{

  namespace DarktableUcs
  {

    // -  UV is pure colour coordinates
    // - JCH is lightness (J), chroma     (C) and hue        (H)  [Helmholtz-Kohlrausch effect is corrected]
    // - HSB is hue       (H), saturation (S) and brightness (B)  [derived from JCH so HKE is corrected]
    // - HCB is hue       (H), chroma     (C) and brightness (B)  [derived from JCH so HKE is corrected]
    // - HPW is hue       (H), purity     (P) and whiteness  (W)  [derived from HSB/HCB so HKE is corrected]
    //
    // -  UV is a Lab/Luv like colour space
    // - JCH is a LCh like colour space
    //
    // -  UV is for perceptually uniform gamut mapping
    // - JCH gets you chroma amount and hue angles and is adjusted for the Helmholtz-Kohlrausch effect
    // - HSB and HCB are intermediate colour spaces to get to specifc HPW versions
    // - HPW is for purity artistic grading (painter's saturation)
    //
    // there are 3 implementations of HPW, each with different numerical issues:
    // - method 1: when (W^2 - S^2) is smaller than 0 it needs to be clipped
    // - method 2: slightly improves the issues of method 1
    // - method 3: adjusting Purity or Whiteness can have issues when converting back if I understand it correctly?


    namespace YTo
    {
      float LStar(const float Y)
      {
        float YHat = pow(Y, 0.631651345306265);

        float LStar = 2.098883786377
                    * YHat
                    / (YHat + 1.12426773749357);

        return LStar;
      }
    } //YTo

    namespace LStarTo
    {
      float Y(const float LStar)
      {
        float Y = pow(-1.12426773749357
                    * LStar
                    / (LStar - 2.098883786377)
                  , 1.5831518565279648);

        return Y;
      }
    } //LStarTo

//#define TTTTTT

    namespace xyTo
    {
      float2 UV(const float2 xy)
      {
#ifndef TTTTTT
        static const float3x3 xyToUVD =
          float3x3
          (
            -0.783941002840055,  0.277512987809202,  0.153836578598858,
             0.745273540913283, -0.205375866083878, -0.165478376301988,
             0.318707282433486,  2.16743692732158,   0.291320554395942
          );

        float3 UVD = mul(xyToUVD, float3(xy, 1.f));
#else
        static const float3x2 xyToUVD =
          float3x2
          (
            -0.783941002840055,  0.277512987809202,
             0.745273540913283, -0.205375866083878,
             0.318707282433486,  2.16743692732158
          );

        float3 UVD = mul(xyToUVD, xy) + float3(0.153836578598858, -0.165478376301988, 0.291320554395942);
#endif

        float2 UV = UVD.xy / UVD.z;

        float2 UVStar = float2(1.39656225667, 1.4513954287)
                      * UV
                      / (abs(UV) + float2(1.49217352929, 1.52488637914));

        static const float2x2 UVStarToUVStarPrime =
          float2x2
          (
            -1.124983854323892, -0.980483721769325,
             1.86323315098672,   1.971853092390862
          );

        float2 UVStarPrime = mul(UVStarToUVStarPrime, UVStar);

        return UVStarPrime;
      }
    } //xyTo

    namespace UVTo
    {
      float2 xy
      (
        const float2 UVStarPrime
      )
      {
        static const float2x2 UVStarPrimeToUVStar =
          float2x2
          (
            -5.037522385190711, -2.504856328185843,
             4.760029407436461,  2.874012963239247
          );

        float2 UVStar = mul(UVStarPrimeToUVStar, UVStarPrime);

        float2 UV = float2(-1.49217352929, -1.52488637914)
                  * UVStar
                  / (abs(UVStar) - float2(1.39656225667, 1.4513954287));

#ifndef TTTTTT
        static const float3x3 UVToxyD =
          float3x3
          (
             0.167171472114775,  0.141299802443708, -0.00801531300850582,
            -0.150959086409163, -0.155185060382272, -0.00843312433578007,
             0.940254742367256,  1.0,               -0.0256325967652889
          );

        float3 xyD = mul(UVToxyD, float3(UV, 1.f));
#else
        static const float2x2 UVToxyD =
          float2x2
          (
             0.167171472114775,  0.141299802443708,
            -0.150959086409163, -0.155185060382272
          );

        float2 xy = mul(UVToxyD, UV);

        float3 xyD = float3(xy, UV[0] * 0.940254742367256 + UV[1])
                   + float3(-0.00801531300850582, -0.00843312433578007, -0.0256325967652889);
#endif

        float2 xy = xyD.xy / xyD.z;

        return xy;
      }
    } //UVTo

    namespace xyYTo
    {
      float3 LUV
      (
        const s_xyY xyY
      )
      {
        float LStar = YTo::LStar(xyY.Y);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        return float3(LStar, UVStarPrime);
      }

      float3 LCH
      (
        const s_xyY xyY
      )
      {
        float LStar = YTo::LStar(xyY.Y);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        float C = sqrt(UVStarPrime.x * UVStarPrime.x
                     + UVStarPrime.y * UVStarPrime.y);

        float H = atan2(UVStarPrime.y, UVStarPrime.x);

        return float3(LStar, C, H);
      }

      float3 JCH
      (
        const s_xyY xyY,
        const float YWhite,
        const float cz
      )
      {
        //input:
        //  * xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
        //  * LWhite the lightness of white as dt UCS L* lightness. [is this the max white you want to display? like 10000 nits?]
        //  * cz: c * z
        //    * n = ratio of background luminance and the luminance of white,
        //    * z = 1 + sqrt(n)
        //    * c = 0.69 for average surround lighting
        //          0.59 for dim surround lighting (sRGB standard)
        //          0.525 for dark surround lighting
        //    * cz = 1 for standard pre-print proofing conditions with average surround and n = 20 %
        //          (background = middle grey, white = perfect diffuse white)
        //range:
        //  * xy in [0; 1]
        //  * Y normalized for perfect diffuse white = 1


        float LStar  = YTo::LStar(xyY.Y);
        float LWhite = YTo::LStar(YWhite);

        float2 UVStarPrime = xyTo::UV(xyY.xy);

        float M2 = UVStarPrime.x * UVStarPrime.x
                 + UVStarPrime.y * UVStarPrime.y;

        float C = 15.932993652962535
                * pow(LStar, 0.6523997524738018)
                * pow(M2,    0.6007557017508491)
                / LWhite;

        float J = pow(LStar / LWhite, cz);

        float H = atan2(UVStarPrime.y, UVStarPrime.x);

        return float3(J, C, H);
      }
    } //xyYTo

    namespace LUVTo
    {
      s_xyY xyY
      (
        const float3 LUV
      )
      {
        s_xyY xyY;

        xyY.xy = UVTo::xy(LUV.yz);

        xyY.Y = LStarTo::Y(LUV[0]);

        return xyY;
      }
    } //LUVTo

    namespace LCHTo
    {
      s_xyY xyY
      (
        const float3 LCH
      )
      {
        float2 UVStarPrime;
        sincos(LCH[2], UVStarPrime[1], UVStarPrime[0]);

        UVStarPrime *= LCH[1];

        s_xyY xyY;

        xyY.xy = UVTo::xy(UVStarPrime);

        xyY.Y = LStarTo::Y(LCH[0]);

        return xyY;
      }
    } //LCHTo

    namespace JCHTo
    {
      s_xyY xyY
      (
        const float3 JCH,
        const float  YWhite,
        const float  cz
      )
      {
        //output: xyY in normalized CIE XYZ for the 2° 1931 observer adapted for D65
        //range:
        //  * xy in [0; 1]
        //  * Y normalized for perfect diffuse white = 1

        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float LWhite = YTo::LStar(YWhite);

        float LStar = pow(J, (1.f / cz)) * LWhite;

        float M = pow((C
                     * LWhite
                     / (15.932993652962535 * pow(LStar,
                                                 0.6523997524738018)))
                  , 0.8322850678616855);

        float2 UVStarPrime;
        sincos(H, UVStarPrime[1], UVStarPrime[0]);

        UVStarPrime *= M;

        s_xyY xyY;

        xyY.xy = UVTo::xy(UVStarPrime);

        xyY.Y = LStarTo::Y(LStar);

        return xyY;
      }

      float3 HCB(const float3 JCH)
      {
        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float B = J
                * (pow(C, 1.33654221029386) + 1.f);

        return float3(H, C, B);
      }

      float3 HSB(const float3 JCH)
      {
        float J = JCH[0];
        float C = JCH[1];
        float H = JCH[2];

        float B = J
                * (pow(C, 1.33654221029386) + 1.f);

        float S = C / B;

        return float3(H, S, B);
      }
    } //JCHTo

    namespace HCBTo
    {
      float3 JCH(const float3 HCB)
      {
        float H = HCB[0];
        float C = HCB[1];
        float B = HCB[2];

        float J = B
                / (pow(C, 1.33654221029386) + 1.f);

        return float3(J, C, H);
      }
    } //HCBTo

    namespace HSBTo
    {
      float3 JCH(const float3 HSB)
      {
        float H = HSB[0];
        float S = HSB[1];
        float B = HSB[2];

        float C = S * B;
        float J = B
                / (pow(C, 1.33654221029386) + 1.f);

        return float3(J, C, H);
      }
    } //HSBTo

    // has numerical issues:
    // when (W^2 - S^2) is smaller than 0 it needs to be clipped
    namespace Method1
    {

      namespace HSBTo
      {
        float3 HPW(const float3 HSB)
        {
          float H = HSB[0];
          float S = HSB[1];
          float B = HSB[2];

          float W = sqrt(S * S + B * B);
          float P = S / W;

          return float3(H, P, W);
        }
      } //HSBTo

      namespace HPWTo
      {
        float3 HSB(const float3 HPW)
        {
          float H = HPW[0];
          float P = HPW[1];
          float W = HPW[2];

          float S = P;
          float B = sqrt(W * W - S * S);

          return float3(H, S, B);
        }
      } //HPWTo

    } //Method1

    namespace Method2
    {

      namespace HSBTo
      {
        float3 HPW(const float3 HSB)
        {
          float H = HSB[0];
          float S = HSB[1];
          float B = HSB[2];

          float W = B;
          float P = W / B;

          return float3(H, P, W);
        }
      } //HSBTo

      namespace HPWTo
      {
        float3 HSB(const float3 HPW)
        {
          float H = HPW[0];
          float P = HPW[1];
          float W = HPW[2];

          float S = W * P;
          float B = W;

          return float3(H, S, B);
        }
      } //HPWTo

    } //Method2

    namespace Method3
    {

      namespace HCBTo
      {
        float3 HPWcc(const float3 HCB)
        {
          float H = HCB[0];
          float C = HCB[1];
          float B = HCB[2];

          float W = sqrt(2.f * (C * C) + (B * B));
          float P = 2.f * C / W;

          return float3(H, P, W);
        }
      } //HCBTo

      namespace HPWccTo
      {
        float3 HCB(const float3 HPWcc)
        {
          float H = HPWcc[0];
          float P = HPWcc[1];
          float W = HPWcc[2];

          float C = P * W / 2.f;
          float B = sqrt((W * W) - 2.f * (C * C));

          return float3(H, C, B);
        }
      } //HPWccTo

    } //Method3

  } //DarktableUcs

} //Csp
