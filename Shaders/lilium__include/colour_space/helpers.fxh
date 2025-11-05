
namespace Csp
{

  namespace Map
  {

    namespace Bt709Into
    {

      float3 ScRgb
      (
        const float3 Colour,
        const float  Brightness
      )
      {
        return Colour / 80.f * Brightness;
      }

      float3 Hdr10
      (
        const float3 Colour,
        const float  Brightness
      )
      {
        return Csp::Trc::Nits_To::PQ(Csp::Mat::Bt709To::Bt2020(Colour) * Brightness);
      }

      float3 Ps5
      (
        const float3 Colour,
        const float  Brightness
      )
      {
        return Csp::Mat::Bt709To::Bt2020(Colour / 100.f) * Brightness;
      }

    } //Bt709Into

  } //Map

} //Csp


#if (OVERWRITE_SDR_GAMMA == GAMMA_24)

  #define ENCODE_SDR(COLOUR) \
            pow(COLOUR, 1.f / 2.4f)

  #define DECODE_SDR(COLOUR) \
            pow(COLOUR, 2.4f)

#elif (OVERWRITE_SDR_GAMMA == GAMMA_SRGB)

  #define ENCODE_SDR(COLOUR) \
            Csp::Trc::Linear_To::sRGB(COLOUR)

  #define DECODE_SDR(COLOUR) \
            Csp::Trc::sRGB_To::Linear(COLOUR)

#else

  #define ENCODE_SDR(COLOUR) \
            pow(COLOUR, 1.f / 2.2f)

  #define DECODE_SDR(COLOUR) \
            pow(COLOUR, 2.2f)

#endif


// convert BT.709 to BT.2020
float3 ConditionallyConvertBt709ToBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
  || ACTUAL_COLOUR_SPACE == CSP_BT2020_EXTENDED)
  Colour = Csp::Mat::Bt709To::Bt2020(Colour);
#endif
  return Colour;
}

// convert DCI-P3 to BT.709
float3 ConditionallyConvertDciP3ToBt709(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::DciP3To::Bt709(Colour);
#endif
  return Colour;
}

// convert DCI-P3 to BT.2020
float3 ConditionallyConvertDciP3ToBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Mat::DciP3To::Bt2020(Colour);
#endif
  return Colour;
}

// convert BT.2020 to BT.709
float3 ConditionallyConvertBt2020ToBt709(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020To::Bt709(Colour);
#endif
  return Colour;
}

// normalise so that 10000 = 1
float3 ConditionallyNormaliseScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour /= 125.f;
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
float3 ConditionallyConvertScRgbToNormalisedBt2020(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::ScRgbTo::Bt2020Normalised(Colour);
#endif
  return Colour;
}

// - normalise so that 10000 = 1
// - convert BT.709 to BT.2020
// - convert normalised to PQ
float3 ConditionallyConvertScRgbToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = ConditionallyConvertScRgbToNormalisedBt2020(Colour);
  Colour = Csp::Trc::Linear_To::PQ(Colour);
#endif
  return Colour;
}

// convert HDR10 to normalised BT.2020
float3 ConditionallyLineariseHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::PQ_To::Linear(Colour);
#endif
  return Colour;
}

// convert normalised BT.2020 to HDR10
float3 ConditionallyConvertNormalisedBt2020ToHdr10(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  Colour = Csp::Trc::Linear_To::PQ(Colour);
#endif
  return Colour;
}

// convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt709ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour *= 125.f;
#endif
  return Colour;
}

// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertNormalisedBt2020ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Mat::Bt2020NormalisedTo::ScRgb(Colour);
#endif
  return Colour;
}

// - convert HDR10 to normalised BT.2020
// - convert BT.2020 to BT.709
// - convert normalised BT.709 to scRGB
float3 ConditionallyConvertHdr10ToScRgb(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  Colour = Csp::Trc::PQ_To::Linear(Colour);
  Colour = ConditionallyConvertNormalisedBt2020ToScRgb(Colour);
#endif
  return Colour;
}

// get luminance for the current colour space
float GetLuminance(float3 Colour)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
  return dot(Colour, Csp::Mat::Bt709ToXYZ[1]);
#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)
  return dot(Colour, Csp::Mat::Bt2020ToXYZ[1]);
#else
  return 0;
#endif
}
