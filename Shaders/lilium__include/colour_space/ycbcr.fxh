
namespace Csp
{

  namespace Ycbcr
  {

    //#define K_BT709  float3(0.2126f, 0.7152f, 0.0722f)
    //#define K_BT2020 float3(0.2627f, 0.6780f, 0.0593f)

    //#define KB_BT709_HELPER 1.8556f //2 - 2 * 0.0722
    //#define KR_BT709_HELPER 1.5748f //2 - 2 * 0.2126
    //#define KG_BT709_HELPER float2(0.187324272930648, 0.468124272930648)
    //(0.0722/0.7152)*(2-2*0.0722), (0.2126/0.7152)*(2-2*0.2126)

    //#define KB_BT2020_HELPER 1.8814f //2 - 2 * 0.0593
    //#define KR_BT2020_HELPER 1.4746f //2 - 2 * 0.2627
    //#define KG_BT2020_HELPER float2(0.164553126843658, 0.571353126843658)
    //(0.0593/0.6780)*(2-2*0.0593), (0.2627/0.6780)*(2-2*0.2627)

      static const float3 K_Bt709         = float3(0.212639003f,
                                                   0.715168654f,
                                                   0.0721923187f);
      static const float  KB_enc_Bt709    = 0.538904786f;
      static const float  KR_enc_Bt709    = 0.635032713f;
      static const float  KB_dec_Bt709    = 1.85561537f;
      static const float  KR_dec_Bt709    = 1.57472193f;
      static const float2 KG_dec_Bt709    = float2(0.187314093f,
                                                   0.468207478f);
      static const float2 PB_NB_Bt709_g2_enc = float2(0.683701097f,
                                                      0.519088029f);
      static const float2 PR_NR_Bt709_g2_enc = float2(0.927864074f,
                                                      0.563485920f);
      static const float2 PB_NB_Bt709_g2_dec = float2(1.46262741f,
                                                      1.92645549f);
      static const float2 PR_NR_Bt709_g2_dec = float2(1.07774400f,
                                                      1.77466726f);
      static const float  K_Bt709G_inverse   = 1.39827156f;

      static const float3 K_Bt2020            = float3(0.262700200f,
                                                       0.677998065f,
                                                       0.0593017153f);
      static const float  KB_enc_Bt2020       = 0.531520068f;
      static const float  KR_enc_Bt2020       = 0.678150177f;
      static const float  KB_dec_Bt2020       = 1.88139653f;
      static const float  KR_dec_Bt2020       = 1.47459959f;
      static const float2 KG_dec_Bt2020       = float2(0.164558053f,
                                                       0.571355044f);
      static const float2 PB_NB_Bt2020_g2_enc = float2(0.660955488f,
                                                       0.515519201f);
      static const float2 PR_NR_Bt2020_g2_enc = float2(1.02573120f,
                                                       0.582301557f);
      static const float2 PB_NB_Bt2020_g2_dec = float2(1.51296114f,
                                                       1.93979203f);
      static const float2 PR_NR_Bt2020_g2_dec = float2(0.974914252f,
                                                       1.71732330f);
      static const float  K_Bt2020G_inverse   = 1.47493040f;

      static const float3 KAp0D65  = float3( 0.343172907f,
                                             0.734696388f,
                                            -0.0778692960f);
      static const float  KbAp0D65 = 2.15573859f;
      static const float  KrAp0D65 = 1.31365418f;
      static const float2 KgAp0D65 = float2(-0.228483289f,
                                             0.613601088f);


    namespace YcbcrTo
    {

      float3 RgbBt709(float3 Colour)
      {
        return float3(
          Colour.x + KR_dec_Bt709    * Colour.z,
          Colour.x - KG_dec_Bt709[0] * Colour.y - KG_dec_Bt709[1] * Colour.z,
          Colour.x + KB_dec_Bt709    * Colour.y);
      }

      float3 RgbBt2020(float3 Colour)
      {
        return float3(
          Colour.x + KR_dec_Bt2020    * Colour.z,
          Colour.x - KG_dec_Bt2020[0] * Colour.y - KG_dec_Bt2020[1] * Colour.z,
          Colour.x + KB_dec_Bt2020    * Colour.y);
      }

    } //YcbcrTo


    namespace RgbTo
    {

      float3 YcbcrBt709(float3 Colour)
      {
        float Y = dot(Colour, K_Bt709);
        return float3(Y,
                      (Colour.b - Y) * KB_enc_Bt709,
                      (Colour.r - Y) * KR_enc_Bt709);
      }

      float3 YcbcrBt2020(float3 Colour)
      {
        float Y = dot(Colour, K_Bt2020);
        return float3(Y,
                      (Colour.b - Y) * KB_enc_Bt2020,
                      (Colour.r - Y) * KR_enc_Bt2020);
      }

    } //RgbTo

  } //Ycbcr

} //Csp
