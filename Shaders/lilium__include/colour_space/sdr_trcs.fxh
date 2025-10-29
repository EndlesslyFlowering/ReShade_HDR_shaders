
namespace Csp
{

  namespace Trc
  {

    //linear->gamma compressed = inverse EOTF -> ^(1 / 2.2)
    //
    //gamma compressed->display (also linear) = EOTF -> ^(2.2)

    namespace SrgbTo
    {
      // IEC 61966-2-1
      #define SRGB_TO_LINEAR(T)                                      \
        T Linear(T C)                                                \
        {                                                            \
          return C <= 0.04045f ? C / 12.92f                          \
                               : pow(((C + 0.055f) / 1.055f), 2.4f); \
        }

      SRGB_TO_LINEAR(float)
      SRGB_TO_LINEAR(float3)
    } //SrgbTo


    namespace LinearTo
    {
      #define LINEAR_TO_SRGB(T)                                          \
        T Srgb(T C)                                                      \
        {                                                                \
          return C <= 0.0031308f ? C * 12.92f                            \
                                 : 1.055f * pow(C, 1.f / 2.4f) - 0.055f; \
        }

      LINEAR_TO_SRGB(float)
      LINEAR_TO_SRGB(float3)
    } //LinearTo


    namespace ExtendedSrgbRollOffToLinearTo
    {
      // extended sRGB gamma including above 1 and below -1
      #define EXTENDED_SRGB_ROLL_OFF_TO_LINEAR_TO_LINEAR(T)                                                          \
        T Linear(const T C)                                                                                          \
        {                                                                                                            \
          const T absC  = abs(C);                                                                                    \
          const T signC = sign(C);                                                                                   \
                                                                                                                     \
          return                                                                                                     \
            absC > 1.24438285f ? C + (signC * 0.0720067024f)                                                         \
          : absC > 1.f         ? signC * ((1.055f * pow(absC - 0.940277040f, (1.f / 2.4f)) - 0.055f) + 0.728929638f) \
          : absC > 0.04045f    ? signC * pow((absC + 0.055f) / 1.055f, 2.4f)                                         \
          :                      C / 12.92f;                                                                         \
        }

      EXTENDED_SRGB_ROLL_OFF_TO_LINEAR_TO_LINEAR(float)
      EXTENDED_SRGB_ROLL_OFF_TO_LINEAR_TO_LINEAR(float3)
    } //ExtendedSrgbRollOffToLinearTo


    namespace ExtendedSrgbSCurveTo
    {
      // extended sRGB gamma including above 1 and below -1
      #define EXTENDED_SRGB_S_CURVE_TO_LINEAR(T)                                                                  \
        T Linear(T C)                                                                                             \
        {                                                                                                         \
          const T absC  = abs(C);                                                                                 \
          const T signC = sign(C);                                                                                \
                                                                                                                  \
          return                                                                                                  \
            absC > 1.f      ? signC * ((1.055f * pow(absC - 0.940277040f, (1.f / 2.4f)) - 0.055f) + 0.728929638f) \
          : absC > 0.04045f ? signC * pow((absC + 0.055f) / 1.055f, 2.4f)                                         \
          :                   C / 12.92f;                                                                         \
        }

      EXTENDED_SRGB_S_CURVE_TO_LINEAR(float)
      EXTENDED_SRGB_S_CURVE_TO_LINEAR(float3)
    } //ExtendedSrgbSCurveTo


    namespace ExtendedSrgbLinearTo
    {
      // extended sRGB gamma including above 1 and below -1
      #define EXTENDED_SRGB_LINEAR_TO_LINEAR(T)                           \
        T Linear(const T C)                                               \
        {                                                                 \
          const T absC  = abs(C);                                         \
          const T signC = sign(C);                                        \
                                                                          \
          return                                                          \
            absC > 1.f      ? C                                           \
          : absC > 0.04045f ? signC * pow((absC + 0.055f) / 1.055f, 2.4f) \
          :                   C / 12.92f;                                 \
        }

      EXTENDED_SRGB_LINEAR_TO_LINEAR(float)
      EXTENDED_SRGB_LINEAR_TO_LINEAR(float3)
    } //ExtendedSrgbLinearTo


// DO NOT USE!!!
// it does not match the ExtendedSrgbSCurveToLinear version!
//
//    namespace LinearTo
//    {
//      float ExtendedSrgbSCurve(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * pow((absC - 0.728929579257965087890625f + 0.055f) / 1.055f, 2.4f) + 0.940277040004730224609375f;
//        }
//        else if (absC > 0.0031308f)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else
//        {
//          return C * 12.92f;
//        }
//      }
//
//      float3 ExtendedSrgbSCurve(float3 Colour)
//      {
//        return float3(Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.r),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.g),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurve(Colour.b));
//      }
//    }


    namespace SrgbAccurateTo
    {
      // accurate sRGB with no slope discontinuity
      #define SrgbX       asfloat(0x3D20EA0B) //  0.0392857
      #define SrgbPhi     asfloat(0x414EC578) // 12.92321
      #define SrgbXDivPhi asfloat(0x3B4739A5) //  0.003039935

      #define SRGB_ACCURATE_TO_LINEAR(T)                          \
        T Linear(const T C)                                       \
        {                                                         \
          return C <= SrgbX ? C / SrgbPhi                         \
                            : pow(((C + 0.055f) / 1.055f), 2.4f); \
        }

      SRGB_ACCURATE_TO_LINEAR(float)
      SRGB_ACCURATE_TO_LINEAR(float3)
    } //SrgbAccurateTo


    namespace LinearTo
    {
      #define LINEAR_TO_SRGB_ACCURATE(T)                                    \
        T SrgbAccurate(const T C)                                           \
        {                                                                   \
          return C <= SrgbXDivPhi ? C * SrgbPhi                             \
                                  : 1.055f * pow(C, (1.f / 2.4f)) - 0.055f; \
        }

      LINEAR_TO_SRGB_ACCURATE(float)
      LINEAR_TO_SRGB_ACCURATE(float3)
    } //LinearTo


//    namespace ExtendedSrgbSCurveAccurateTo
//    {
//      float Linear(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else if (absC > SrgbX)
//        {
//          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
//        }
//        else
//        {
//          return C / SrgbPhi;
//        }
//      }
//
//      float3 Linear(float3 Colour)
//      {
//        return float3(Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.r),
//                      Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.g),
//                      Csp::Trc::ExtendedSrgbSCurveAccurateTo::Linear(Colour.b));
//      }
//    } //ExtendedSrgbSCurveAccurateTo
//
//
//    namespace LinearTo
//    {
//      float ExtendedSrgbSCurveAccurate(float C)
//      {
//        static const float absC  = abs(C);
//        static const float signC = sign(C);
//
//        if (absC > 1.f)
//        {
//          return signC * pow((absC + 0.055f) / 1.055f, 2.4f);
//        }
//        else if (absC > SrgbXDivPhi)
//        {
//          return signC * (1.055f * pow(absC, (1.f / 2.4f)) - 0.055f);
//        }
//        else
//        {
//          return C * SrgbPhi;
//        }
//      }
//
//      float3 ExtendedSrgbSCurveAccurate(float3 Colour)
//      {
//        return float3(Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.r),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.g),
//                      Csp::Trc::LinearTo::ExtendedSrgbSCurveAccurate(Colour.b));
//      }
//    } //LinearTo


    namespace ExtendedGamma22RollOffToLinearTo
    {
      // extended gamma 2.2 including above 1 and below -1
      #define EXTENDED_GAMMA_22_ROLL_OFF_TO_LINEAR_TO_LINEAR(T)                                \
        T Linear(const T C)                                                                    \
        {                                                                                      \
          const T absC  = abs(C);                                                              \
          const T signC = sign(C);                                                             \
                                                                                               \
          return                                                                               \
            absC > 1.23562705f ? C + (signC * 0.0663648843)                                    \
          : absC > 1.f         ? signC * (pow(absC - 0.944479882f, 1.f / 2.2f) + 0.731282770f) \
          :                      signC * pow(absC, 2.2f);                                      \
        }

      EXTENDED_GAMMA_22_ROLL_OFF_TO_LINEAR_TO_LINEAR(float)
      EXTENDED_GAMMA_22_ROLL_OFF_TO_LINEAR_TO_LINEAR(float3)
    } //ExtendedGamma22RollOffToLinearTo


    namespace ExtendedGamma22SCurveTo
    {
      // extended gamma 2.2 including above 1 and below 0
      #define EXTENDED_GAMMA_22_S_CURVE_TO_LINEAR(T)                                     \
        T Linear(const T C)                                                              \
        {                                                                                \
          const T absC  = abs(C);                                                        \
          const T signC = sign(C);                                                       \
                                                                                         \
          return                                                                         \
            absC <= 1.f ? signC * pow(absC, 2.2f)                                        \
                        : signC * (pow(absC - 0.944479882f, 1.f / 2.2f) + 0.731282770f); \
        }

        EXTENDED_GAMMA_22_S_CURVE_TO_LINEAR(float)
        EXTENDED_GAMMA_22_S_CURVE_TO_LINEAR(float3)
    } //ExtendedGamma22SCurveTo


    namespace ExtendedGamma22LinearTo
    {
      // extended gamma 2.2 including above 1 and below 0
      #define EXTENDED_GAMMA_22_LINEAR_TO_LINEAR(T)   \
        T Linear(const T C)                           \
        {                                             \
          const T absC  = abs(C);                     \
          const T signC = sign(C);                    \
                                                      \
          return absC < 1.f ? signC * pow(absC, 2.2f) \
                            : C;                      \
        }

      EXTENDED_GAMMA_22_LINEAR_TO_LINEAR(float)
      EXTENDED_GAMMA_22_LINEAR_TO_LINEAR(float3)
    } //ExtendedGamma22LinearTo


    namespace ExtendedGamma24RollOffToLinearTo
    {
      // extended sRGB gamma including above 1 and below -1
      #define EXTENDED_GAMMA_24_ROLL_OFF_TO_LINEAR_TO_LINEAR(T)                                \
        T Linear(const T C)                                                                    \
        {                                                                                      \
          const T absC  = abs(C);                                                              \
          const T signC = sign(C);                                                             \
                                                                                               \
          return                                                                               \
            absC > 1.22295093f ? C + (signC * 0.0726261138)                                    \
          : absC > 1.f         ? signC * (pow(absC - 0.950292885f, 1.f / 2.4f) + 0.713687002f) \
          :                      signC * pow(absC, 2.4f);                                      \
        }

      EXTENDED_GAMMA_24_ROLL_OFF_TO_LINEAR_TO_LINEAR(float)
      EXTENDED_GAMMA_24_ROLL_OFF_TO_LINEAR_TO_LINEAR(float3)
    } //ExtendedGamma24RollOffToLinearTo


    namespace ExtendedGamma24SCurveTo
    {
      // extended gamma 2.4 including above 1 and below 0
      #define EXTENDED_GAMMA_24_S_CURVE_TO_LINEAR(T)                                     \
        T Linear(const T C)                                                              \
        {                                                                                \
          const T absC  = abs(C);                                                        \
          const T signC = sign(C);                                                       \
                                                                                         \
          return                                                                         \
            absC <= 1.f ? signC * pow(absC, 2.4f)                                        \
                        : signC * (pow(absC - 0.950292885f, 1.f / 2.4f) + 0.713687002f); \
        }

      EXTENDED_GAMMA_24_S_CURVE_TO_LINEAR(float)
      EXTENDED_GAMMA_24_S_CURVE_TO_LINEAR(float3)
    } //ExtendedGamma24SCurveTo


    namespace ExtendedGamma24LinearTo
    {
      // extended gamma 2.4 including above 1 and below 0
      #define EXTENDED_GAMMA_24_LINEAR_TO_LINEAR(T)   \
        T Linear(const T C)                           \
        {                                             \
          const T absC  = abs(C);                     \
          const T signC = sign(C);                    \
                                                      \
          return absC < 1.f ? signC * pow(absC, 2.4f) \
                            : C;                      \
        }

      EXTENDED_GAMMA_24_LINEAR_TO_LINEAR(float)
      EXTENDED_GAMMA_24_LINEAR_TO_LINEAR(float3)
    } //ExtendedGamma24LinearTo


    //float X_power_TRC(float C, float pow_gamma)
    //{
    //  float pow_Inverse_gamma = 1.f / pow_gamma;
    //
    //  if (C < -1)
    //    return
    //      -pow(-C, pow_Inverse_gamma);
    //  else if (C < 0)
    //    return
    //      -pow(-C, pow_gamma);
    //  else if (C <= 1)
    //    return
    //      pow(C, pow_gamma);
    //  else
    //    return
    //      pow(C, pow_Inverse_gamma);
    //}
    //
    //float3 X_power_TRC(float3 Colour, float pow_gamma)
    //{
    //  return float3(
    //    X_power_TRC(Colour.r, pow_gamma),
    //    X_power_TRC(Colour.g, pow_gamma),
    //    X_power_TRC(Colour.b, pow_gamma));
    //}


    // gamma adjust including values above 1 and below 0
    float ExtendedGammaAdjust(float C, float Adjust)
    {
      float inverseAdjust = 1.f / Adjust;

      static const float absC = abs(C);
      static const float signC = sign(C);

      [branch]
      if (absC > 1.f)
      {
        return signC * pow(absC, inverseAdjust);
      }
      else
      {
        return signC * pow(absC, Adjust);
      }
    }

    float3 ExtendedGammaAdjust(float3 Colour, float Adjust)
    {
      return float3(ExtendedGammaAdjust(Colour.r, Adjust),
                    ExtendedGammaAdjust(Colour.g, Adjust),
                    ExtendedGammaAdjust(Colour.b, Adjust));
    }

  } //Trc

} //Csp
