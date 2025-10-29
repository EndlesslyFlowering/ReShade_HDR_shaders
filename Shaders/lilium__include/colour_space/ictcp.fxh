
namespace Csp
{

  namespace Ictcp
  {

    // The matrices use higher precision rather than being rounded to 12 bit values like the Dolby spec describes.
    // This is because I only use these for internal processing.

    //L'M'S'->ICtCp
    static const float3x3 PqLmsToIctcp =
      float3x3
      (
        0.5f,         0.5f,         0.f,
        1.61370003f, -3.32339620f,  1.70969617f,
        4.37806224f, -4.24553966f, -0.132522642f
      );

    //ICtCp->L'M'S'
    static const float3x3 IctcpToPqLms =
      float3x3
      (
        1.f,  0.00860647484f,  0.111033529f,
        1.f, -0.00860647484f, -0.111033529f,
        1.f,  0.560046315f,   -0.320631951f
      );


    //RGB BT.709->LMS
    static const float3x3 Bt709ToLms =
      float3x3
      (
        0.295764088f,  0.623072445f, 0.0811667516f,
        0.156191974f,  0.727251648f, 0.116557933f,
        0.0351022854f, 0.156589955f, 0.808302998f
      );

    //scRGB->LMS
    static const float3x3 ScRgbToLms =
      float3x3
      (
        0.00236611254f,  0.00498457951f, 0.000649333989f,
        0.00124953582f,  0.00581801310f, 0.000932463502f,
        0.000280818290f, 0.00125271955f, 0.00646642409f
      );

    //RGB DCI-P3->LMS
    static const float3x3 DciP3ToLms =
      float3x3
      (
        0.334494858f,  0.576365113f,  0.0891432985f,
        0.158450931f,  0.713538110f,  0.128012508f,
        0.0205394085f, 0.0917179808f, 0.887737870f
      );

    //RGB BT.2020->LMS
    static const float3x3 Bt2020ToLms =
      float3x3
      (
        0.412036389f,  0.523911893f,  0.0640549808f,
        0.166660219f,  0.720395207f,  0.112946122f,
        0.0241123586f, 0.0754749625f, 0.900407910f
      );


    //LMS->RGB BT.709
    static const float3x3 LmsToBt709 =
      float3x3
      (
         6.17353248f,   -5.32089900f,   0.147354885f,
        -1.32403194f,    2.56026983f,  -0.236238613f,
        -0.0115983877f, -0.264921456f,  1.27652633f
      );

    //LMS->scRGB
    static const float3x3 LmsToScRgb =
      float3x3
      (
         771.691589f,   -665.112365f,    18.4193611f,
        -165.503982f,    320.033721f,   -29.5298271f,
          -1.44979846f,  -33.1151809f,  159.565795f
      );

    //LMS->RGB DCI-P3
    static const float3x3 LmsToDciP3 =
      float3x3
      (
         4.84242963f,     -3.92169165f,   0.0792524516f,
        -1.07515621f,      2.29866075f,  -0.223505541f,
        -0.000956906354f, -0.146754235f,  1.14771676f
      );

    //LMS->RGB BT.2020
    static const float3x3 LmsToBt2020 =
      float3x3
      (
         3.43681478f,   -2.50677371f,    0.0699519291f,
        -0.791058242f,   1.98360168f,   -0.192544832f,
        -0.0257268063f, -0.0991417691f,  1.12487411f
      );

    namespace IctcpTo
    {
      //ICtCp->L'M'S'
      float3 PqLms(float3 Ictcp)
      {
        return mul(IctcpToPqLms, Ictcp);
      }
    } //IctcpTo

    namespace PqLmsTo
    {
      //L'M'S'->ICtCp
      float3 Ictcp(float3 PqLms)
      {
        return mul(PqLmsToIctcp, PqLms);
      }

      //L'M'S'->LMS
      float3 Lms(float3 PqLms)
      {
        return Csp::Trc::PqTo::Linear(PqLms);
      }
    } //PqLmsTo

    namespace IctcpTo
    {
      //ICtCp->LMS
      float3 Lms(float3 Ictcp)
      {
        float3 pqLms = IctcpTo::PqLms(Ictcp);

        //LMS
        return PqLmsTo::Lms(pqLms);
      }
    } //IctcpTo

    namespace LmsTo
    {
      //LMS->L'M'S'
      float3 PqLms(float3 Lms)
      {
        return Csp::Trc::LinearTo::Pq(Lms);
      }

      //LMS->ICtCp
      float3 Ictcp(float3 Lms)
      {
        float3 pqLms = LmsTo::PqLms(Lms);

        //ICtCp
        return PqLmsTo::Ictcp(pqLms);
      }

      //LMS->RGB BT.709
      float3 Bt709(float3 Colour)
      {
        return mul(LmsToBt709, Colour);
      }

      //LMS->scRGB
      float3 ScRgb(float3 Colour)
      {
        return mul(LmsToScRgb, Colour);
      }

      //LMS->RGB DCI-P3
      float3 DciP3(float3 Colour)
      {
        return mul(LmsToDciP3, Colour);
      }

      //LMS->RGB BT.2020
      float3 Bt2020(float3 Colour)
      {
        return mul(LmsToBt2020, Colour);
      }
    } //LmsTo

    namespace PqLmsTo
    {
      //L'M'S'->RGB BT.709
      float3 Bt709(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //BT.709
        return LmsTo::Bt709(lms);
      }

      //L'M'S'->scRGB
      float3 ScRgb(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //scRGB
        return LmsTo::ScRgb(lms);
      }

      //L'M'S'->RGB DCI-P3
      float3 DciP3(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //DCI-P3
        return LmsTo::DciP3(lms);
      }

      //L'M'S'->RGB BT.2020
      float3 Bt2020(float3 PqLms)
      {
        float3 lms = PqLmsTo::Lms(PqLms);

        //BT.2020
        return LmsTo::Bt2020(lms);
      }
    } //PqLmsTo

    namespace IctcpTo
    {
      //ICtCp->RGB BT.709
      float3 Bt709(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.709
        return LmsTo::Bt709(lms);
      }

      //ICtCp->scRGB
      float3 ScRgb(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //scRGB
        return LmsTo::ScRgb(lms);
      }

      //ICtCp->RGB DCI-P3
      float3 DciP3(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.709
        return LmsTo::DciP3(lms);
      }

      //ICtCp->RGB BT.2020
      float3 Bt2020(float3 Ictcp)
      {
        float3 lms = IctcpTo::Lms(Ictcp);

        //BT.2020
        return LmsTo::Bt2020(lms);
      }
    } //IctcpTo

    namespace Bt709To
    {
      //RGB BT.709->LMS
      float3 Lms(float3 Colour)
      {
        return mul(Bt709ToLms, Colour);
      }

      //RGB BT.709->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = Bt709To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB BT.709->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = Bt709To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //Bt709To

    namespace ScRgbTo
    {
      //scRGB->LMS
      float3 Lms(float3 Colour)
      {
        return mul(ScRgbToLms, Colour);
      }

      //scRGB->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = ScRgbTo::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //scRGB->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = ScRgbTo::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //ScRgbTo

    namespace DciP3To
    {
      //RGB DCI-P3->LMS
      float3 Lms(float3 Colour)
      {
        return mul(DciP3ToLms, Colour);
      }

      //RGB DCI-P3->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = DciP3To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB DCI-P3->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = DciP3To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //DciP3To

    namespace Bt2020To
    {
      //RGB BT.2020->LMS
      float3 Lms(float3 Colour)
      {
        return mul(Bt2020ToLms, Colour);
      }

      //RGB BT.2020->L'M'S'
      float3 PqLms(float3 Rgb)
      {
        float3 lms = Bt2020To::Lms(Rgb);

        //L'M'S'
        return LmsTo::PqLms(lms);
      }

      //RGB BT.2020->ICtCp
      float3 Ictcp(float3 Rgb)
      {
        float3 lms = Bt2020To::Lms(Rgb);

        //ICtCp
        return LmsTo::Ictcp(lms);
      }
    } //Bt2020To

  } //ICtCp

} //Csp
