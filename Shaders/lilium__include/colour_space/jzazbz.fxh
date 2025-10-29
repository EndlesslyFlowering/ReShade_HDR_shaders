
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

    static const float3x3 NonLinearXYAndLinearZToJzazbzLms =
      float3x3
      (
         0.41478972f, 0.579999f, 0.014648f,
        -0.20151f,    1.120649f, 0.0531008f,
        -0.0166008f,  0.2648f,   0.6684799f
      );

    static const float3x3 JzazbzLmsToNonLinearXYAndLinearZ =
      float3x3
      (
         1.92422640f,   -1.00479233f,   0.0376514047f,
         0.350316762f,   0.726481199f, -0.0653844252f,
        -0.0909828096f, -0.312728285f,  1.52276659f
      );

    static const float3x3 PqJzazbzLmsToIzazbz =
      float3x3
      (
        0.5f,       0.5f,       0.f,
        3.524f,    -4.066708f,  0.542708f,
        0.199076f,  1.096799f, -1.295875f
      );

    static const float3x3 IzazbzToPqJzazbzLms =
      float3x3
      (
        1.f,  0.138605043f,   0.0580473169f,
        1.f, -0.138605043f,  -0.0580473169f,
        1.f, -0.0960192456f, -0.811891913f
      );

    namespace XYZTo
    {
      //XYZ->X'Y'Z
      float3 NonLinearXYAndLinearZ(float3 XYZ)
      {
        float2 val0 = float2(Jzazbz_b * XYZ.x,
                             Jzazbz_g * XYZ.y);

        float2 val1 = float2(Jzazbz_b_minus_1 * XYZ.z,
                             Jzazbz_g_minus_1 * XYZ.x);

        return float3(val0 - val1, XYZ.z);
      }
    } //XYZTo

    namespace NonLinearXYAndLinearZTo
    {
      //X'Y'Z->XYZ
      float3 XYZ(float3 NonLinearXYAndLinearZ)
      {
        float X = (NonLinearXYAndLinearZ.x + (Jzazbz_b_minus_1 * NonLinearXYAndLinearZ.z))
                / Jzazbz_b;

        float Y = (NonLinearXYAndLinearZ.y + (Jzazbz_g_minus_1 * X))
                / Jzazbz_g;

        return float3(X, Y, NonLinearXYAndLinearZ.z);
      }

      //X'Y'Z->LMS (Jzazbz variant)
      float3 JzazbzLms(float3 NonLinearXYAndLinearZ)
      {
        return mul(NonLinearXYAndLinearZToJzazbzLms, NonLinearXYAndLinearZ);
      }
    } //NonLinearXYAndLinearZTo

    namespace JzazbzLmsTo
    {
      //LMS (Jzazbz variant)->X'Y'Z
      float3 NonLinearXYAndLinearZ(float3 JzazbzLms)
      {
        return mul(JzazbzLmsToNonLinearXYAndLinearZ, JzazbzLms);
      }

      //LMS (Jzazbz variant)->L'M'S' (Jzazbz variant)
      float3 PqJzazbzLms(float3 JzazbzLms)
      {
        float3 powJzazbzLms = pow(JzazbzLms, PQ_m1);

        float3 numerator    = PQ_c1 + PQ_c2 * powJzazbzLms;

        float3 denominator  = 1.f   + PQ_c3 * powJzazbzLms;

        return pow(numerator / denominator, Jzazbz_p);
      }
    } //JzazbzLmsTo

    namespace PqJzazbzLmsTo
    {
      //L'M'S' (Jzazbz variant)->Izazbz
      float3 Izazbz(float3 PqJzazbzLms)
      {
        return mul(PqJzazbzLmsToIzazbz, PqJzazbzLms);
      }

      //L'M'S' (Jzazbz variant)->LMS (Jzazbz variant)
      float3 JzazbzLms(float3 PqJzazbzLms)
      {
        float3 powPqJzazbzLms = pow(PqJzazbzLms, Jzazbz_rcp_p);

        float3 numerator      = PQ_c1 - powPqJzazbzLms;

        float3 denominator    = PQ_c3 * powPqJzazbzLms - PQ_c2;

        return pow(numerator / denominator, PQ_rcp_m1);
      }
    } //PqJzazbzLmsTo

    namespace IzazbzTo
    {
      //Izazbz->Jzazbz
      float3 Jzazbz(float3 Izazbz)
      {
        float numerator   = Jzazbz_d_plus_1 * Izazbz.x;

        float denominator = 1.f + (Jzazbz_d * Izazbz.x);

        float Jz = (numerator / denominator) - Jzazbz_d0;

        return float3(Jz, Izazbz.yz);
      }

      //Izazbz->L'M'S' (Jzazbz variant)
      float3 PqJzazbzLms(float3 Izazbz)
      {
        return mul(IzazbzToPqJzazbzLms, Izazbz);
      }
    } //IzazbzTo

    namespace JzazbzTo
    {
      //Jzazbz->Izazbz
      float3 Izazbz(float3 Jzazbz)
      {
        float numerator   = Jzazbz.x + Jzazbz_d0;

        float denominator = Jzazbz_d_plus_1 - (Jzazbz_d * (Jzazbz.x + Jzazbz_d0));

        float Iz = numerator / denominator;

        return float3(Iz, Jzazbz.yz);
      }
    } //JzazbzTo

    namespace XYZTo
    {
      //XYZ->Jzazbz
      float3 Jzazbz(float3 XYZ)
      {
        float3 NonLinearXYAndLinearZ = XYZTo::NonLinearXYAndLinearZ(XYZ);

        float3 JzazbzLms             = NonLinearXYAndLinearZTo::JzazbzLms(NonLinearXYAndLinearZ);

        float3 PqJzazbzLms           = JzazbzLmsTo::PqJzazbzLms(JzazbzLms);

        float3 Izazbz                = PqJzazbzLmsTo::Izazbz(PqJzazbzLms);

        //Jzazbz
        return IzazbzTo::Jzazbz(Izazbz);
      }
    } //XYZTo

    namespace JzazbzTo
    {
      //Jzazbz->XYZ
      float3 XYZ(float3 Jzazbz)
      {
        float3 Izazbz                = JzazbzTo::Izazbz(Jzazbz);

        float3 PqJzazbzLms           = IzazbzTo::PqJzazbzLms(Izazbz);

        float3 JzazbzLms             = PqJzazbzLmsTo::JzazbzLms(PqJzazbzLms);

        float3 NonLinearXYAndLinearZ = JzazbzLmsTo::NonLinearXYAndLinearZ(JzazbzLms);

        //XYZ
        return NonLinearXYAndLinearZTo::XYZ(NonLinearXYAndLinearZ);
      }
    } //JzazbzTo

    namespace Bt709To
    {
      //RGB BT.709->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::Bt709To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //Bt709To

    namespace DciP3To
    {
      //RGB DCI-P3->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::DciP3To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //DciP3To

    namespace Bt2020To
    {
      //RGB BT.2020->Jzazbz
      float3 Jzazbz(float3 Rgb)
      {
        float3 XYZ = Csp::Mat::Bt2020To::XYZ(Rgb);

        //Jzazbz
        return XYZTo::Jzazbz(XYZ);
      }
    } //Bt2020To

    namespace JzazbzTo
    {
      //Jzazbz->RGB BT.709
      float3 Bt709(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB BT.709
        return Csp::Mat::XYZTo::Bt709(XYZ);
      }

      //Jzazbz->RGB DCI-P3
      float3 DciP3(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB DCI-P3
        return Csp::Mat::XYZTo::DciP3(XYZ);
      }

      //Jzazbz->RGB BT.2020
      float3 Bt2020(float3 Jzazbz)
      {
        float3 XYZ = JzazbzTo::XYZ(Jzazbz);

        //RGB BT.2020
        return Csp::Mat::XYZTo::Bt2020(XYZ);
      }
    } //JzazbzTo

  } //Jzazbz

} //Csp
