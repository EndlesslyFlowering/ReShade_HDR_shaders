
namespace Csp
{

  namespace Jzazbz
  {

    // https://doi.org/10.1364/OE.25.015131

    static const float Jzazbz_b  =   1.15f;
    static const float Jzazbz_g  =   0.66f;
    static const float Jzazbz_p  = 134.034375f; // 1.7 * 2523 / 2^5 (2523 / 2^5 = 2523 / 4096 * 128, which is the PQ constant m2)
    static const float Jzazbz_d  =  -0.56f;
    static const float Jzazbz_d0 =   0.0000000000162954996f; // 1.6295499532821566 * 10^-11

    static const float Jzazbz_b_minus_1 =  0.15f;
    static const float Jzazbz_g_minus_1 = -0.34f;

    static const float Jzazbz_d_plus_1 = 0.44f;

    static const float Jzazbz_rcp_p = 0.00746077252f;

    //X'Y'Z -> Jzazbz LMS
    static const float3x3 XY_Prime_Z_To_Jzazbz_LMS =
      float3x3
      (
         0.41478972f, 0.579999f, 0.014648f,
        -0.20151f,    1.120649f, 0.0531008f,
        -0.0166008f,  0.2648f,   0.6684799f
      );

    //Jzazbz LMS -> X'Y'Z
    static const float3x3 Jzazbz_LMS_To_XY_Prime_Z =
      float3x3
      (
         1.92422640f,   -1.00479233f,   0.0376514047f,
         0.350316762f,   0.726481199f, -0.0653844252f,
        -0.0909828096f, -0.312728285f,  1.52276659f
      );

    //Jzazbz L'M'S' -> Izazbz
    static const float3x3 PQ_Jzazbz_LMS_To_Izazbz =
      float3x3
      (
        0.5f,       0.5f,       0.f,
        3.524f,    -4.066708f,  0.542708f,
        0.199076f,  1.096799f, -1.295875f
      );

    //Izazbz -> Jzazbz L'M'S'
    static const float3x3 Izazbz_To_PQ_Jzazbz_LMS =
      float3x3
      (
        1.f,  0.138605043f,   0.0580473169f,
        1.f, -0.138605043f,  -0.0580473169f,
        1.f, -0.0960192456f, -0.811891913f
      );

    namespace XYZ_To
    {
      //XYZ -> X'Y'Z
      float3 XY_Prime_Z(const float3 XYZ)
      {
        float2 val_0 = float2(Jzazbz_b * XYZ.x,
                              Jzazbz_g * XYZ.y);

        float2 val_1 = float2(Jzazbz_b_minus_1 * XYZ.z,
                              Jzazbz_g_minus_1 * XYZ.x);

        return float3(val_0 - val_1, XYZ.z);
      }
    } //XYZ_To

    namespace XY_Prime_Z_To
    {
      //X'Y'Z -> XYZ
      float3 XYZ(const float3 XY_Prime_Z)
      {
        float X = (XY_Prime_Z.x + (Jzazbz_b_minus_1 * XY_Prime_Z.z))
                / Jzazbz_b;

        float Y = (XY_Prime_Z.y + (Jzazbz_g_minus_1 * X))
                / Jzazbz_g;

        return float3(X, Y, XY_Prime_Z.z);
      }

      //X'Y'Z -> LMS (Jzazbz variant)
      float3 Jzazbz_LMS(const float3 XY_Prime_Z)
      {
        return mul(XY_Prime_Z_To_Jzazbz_LMS, XY_Prime_Z);
      }
    } //XY_Prime_Z_To

    namespace Jzazbz_LMS_To
    {
      //LMS (Jzazbz variant) -> X'Y'Z
      float3 XY_Prime_Z(const float3 Jzazbz_LMS)
      {
        return mul(Jzazbz_LMS_To_XY_Prime_Z, Jzazbz_LMS);
      }

      //LMS (Jzazbz variant) -> L'M'S' (Jzazbz variant)
      float3 PQ_Jzazbz_LMS(const float3 Jzazbz_LMS)
      {
        float3 lms_pow = pow(Jzazbz_LMS, PQ_m1);

        float3 num = PQ_c1 + PQ_c2 * lms_pow;

        float3 den = 1.f   + PQ_c3 * lms_pow;

        return pow(num / den, Jzazbz_p);
      }
    } //Jzazbz_LMS_To

    namespace PQ_Jzazbz_LMS_To
    {
      //L'M'S' (Jzazbz variant) -> Izazbz
      float3 Izazbz(const float3 PQ_Jzazbz_LMS)
      {
        return mul(PQ_Jzazbz_LMS_To_Izazbz, PQ_Jzazbz_LMS);
      }

      //L'M'S' (Jzazbz variant) -> LMS (Jzazbz variant)
      float3 Jzazbz_LMS(const float3 PQ_Jzazbz_LMS)
      {
        float3 pq_lms_pow = pow(PQ_Jzazbz_LMS, Jzazbz_rcp_p);

        float3 num = PQ_c1 - pq_lms_pow;

        float3 den = PQ_c3 * pq_lms_pow - PQ_c2;

        return pow(num / den, PQ_rcp_m1);
      }
    } //PQ_Jzazbz_LMS_To

    namespace Izazbz_To
    {
      //Izazbz -> Jzazbz
      float3 Jzazbz(const float3 Izazbz)
      {
        float num = Jzazbz_d_plus_1 * Izazbz.x;

        float den = 1.f + (Jzazbz_d * Izazbz.x);

        //Jz
        float jz = (num / den) - Jzazbz_d0;

        return float3(jz, Izazbz.yz);
      }

      //Izazbz -> L'M'S' (Jzazbz variant)
      float3 PQ_Jzazbz_LMS(const float3 Izazbz)
      {
        return mul(Izazbz_To_PQ_Jzazbz_LMS, Izazbz);
      }
    } //Izazbz_To

    namespace Jzazbz_To
    {
      //Jzazbz -> Izazbz
      float3 Izazbz(const float3 Jzazbz)
      {
        float num = Jzazbz.x + Jzazbz_d0;

        float den = Jzazbz_d_plus_1 - (Jzazbz_d * (Jzazbz.x + Jzazbz_d0));

        //Iz
        float iz = num / den;

        return float3(iz, Jzazbz.yz);
      }
    } //Jzazbz_To

    namespace XYZ_To
    {
      //XYZ -> Jzazbz
      float3 Jzazbz(const float3 XYZ)
      {
        //X'Y'Z
        float3 xy_prime_z = XYZ_To::XY_Prime_Z(XYZ);

        float3 lms        = XY_Prime_Z_To::Jzazbz_LMS(xy_prime_z);

        float3 pq_lms     = Jzazbz_LMS_To::PQ_Jzazbz_LMS(lms);

        float3 izazbz     = PQ_Jzazbz_LMS_To::Izazbz(pq_lms);

        return Izazbz_To::Jzazbz(izazbz);
      }
    } //XYZ_To

    namespace Jzazbz_To
    {
      //Jzazbz -> XYZ
      float3 XYZ(const float3 Jzazbz)
      {
        float3 izazbz     = Jzazbz_To::Izazbz(Jzazbz);

        float3 pq_lms     = Izazbz_To::PQ_Jzazbz_LMS(izazbz);

        float3 lms        = PQ_Jzazbz_LMS_To::Jzazbz_LMS(pq_lms);

        //X'Y'Z
        float3 xy_prime_z = Jzazbz_LMS_To::XY_Prime_Z(lms);

        return XY_Prime_Z_To::XYZ(xy_prime_z);
      }
    } //Jzazbz_To

    namespace BT709_To
    {
      //RGB BT.709 -> Jzazbz
      float3 Jzazbz(const float3 RGB)
      {
        float3 XYZ = Csp::Mat::BT709_To::XYZ(RGB);

        return XYZ_To::Jzazbz(XYZ);
      }
    } //BT709_To

    namespace DCIP3_To
    {
      //RGB DCI-P3 -> Jzazbz
      float3 Jzazbz(const float3 RGB)
      {
        float3 XYZ = Csp::Mat::DCIP3_To::XYZ(RGB);

        return XYZ_To::Jzazbz(XYZ);
      }
    } //DCIP3_To

    namespace BT2020_To
    {
      //RGB BT.2020 -> Jzazbz
      float3 Jzazbz(const float3 RGB)
      {
        float3 XYZ = Csp::Mat::BT2020_To::XYZ(RGB);

        return XYZ_To::Jzazbz(XYZ);
      }
    } //BT2020_To

    namespace Jzazbz_To
    {
      //Jzazbz -> RGB BT.709
      float3 BT709(const float3 Jzazbz)
      {
        float3 xyz = Jzazbz_To::XYZ(Jzazbz);

        return Csp::Mat::XYZ_To::BT709(xyz);
      }

      //Jzazbz -> RGB DCI-P3
      float3 DCIP3(const float3 Jzazbz)
      {
        float3 xyz = Jzazbz_To::XYZ(Jzazbz);

        return Csp::Mat::XYZ_To::DCIP3(xyz);
      }

      //Jzazbz -> RGB BT.2020
      float3 BT2020(const float3 Jzazbz)
      {
        float3 xyz = Jzazbz_To::XYZ(Jzazbz);

        return Csp::Mat::XYZ_To::BT2020(xyz);
      }
    } //Jzazbz_To

  } //Jzazbz

} //Csp
