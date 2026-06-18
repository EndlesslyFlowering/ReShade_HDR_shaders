
#if (defined(IS_COMPUTE_CAPABLE_API)   \
  && (ACTUAL_COLOUR_SPACE == CSP_HDR10 \
   || defined(MANUAL_OVERRIDE_MODE_ENABLE_INTERNAL)))

  #define HDR10_TO_LINEAR_LUT()                                     \
    texture1D TextureHdr10ToLinearLUT                               \
    {                                                               \
      Width  = 1024;                                                \
      Format = R32F;                                                \
    };                                                              \
                                                                    \
    sampler1D<float> SamplerHdr10ToLinearLUT                        \
    {                                                               \
      Texture = TextureHdr10ToLinearLUT;                            \
                                                                    \
      MagFilter = POINT;                                            \
      MinFilter = POINT;                                            \
      MipFilter = POINT;                                            \
    };                                                              \
                                                                    \
    storage1D<float> StorageHdr10ToLinearLUT                        \
    {                                                               \
      Texture = TextureHdr10ToLinearLUT;                            \
    };                                                              \
                                                                    \
    void CS_CreateHdr10ToLinearLUT                                  \
    (                                                               \
      uint3 DTID : SV_DispatchThreadID                              \
    )                                                               \
    {                                                               \
      float curr = Csp::Trc::PQ_To::Linear(float(DTID.x) / 1023.f); \
                                                                    \
      tex1Dstore(StorageHdr10ToLinearLUT, DTID.x, curr);            \
                                                                    \
      return;                                                       \
    }                                                               \
                                                                    \
    float SampleFromHdr10ToLinearLUT                                \
    (                                                               \
      float Channel                                                 \
    )                                                               \
    {                                                               \
      Channel = tex1D(SamplerHdr10ToLinearLUT, Channel);            \
                                                                    \
      return Channel;                                               \
    }                                                               \
                                                                    \
    float2 SampleFromHdr10ToLinearLUT                               \
    (                                                               \
      float2 Channels                                               \
    )                                                               \
    {                                                               \
      Channels.x = tex1D(SamplerHdr10ToLinearLUT, Channels.x);      \
      Channels.y = tex1D(SamplerHdr10ToLinearLUT, Channels.y);      \
                                                                    \
      return Channels;                                              \
    }                                                               \
                                                                    \
    float3 SampleFromHdr10ToLinearLUT                               \
    (                                                               \
      float3 Colour                                                 \
    )                                                               \
    {                                                               \
      Colour.r = tex1D(SamplerHdr10ToLinearLUT, Colour.r);          \
      Colour.g = tex1D(SamplerHdr10ToLinearLUT, Colour.g);          \
      Colour.b = tex1D(SamplerHdr10ToLinearLUT, Colour.b);          \
                                                                    \
      return Colour;                                                \
    }                                                               \
                                                                    \
    float FetchFromHdr10ToLinearLUT                                 \
    (                                                               \
      float Channel                                                 \
    )                                                               \
    {                                                               \
      const int lookup = int(Channel * 1023.f + 0.5f);              \
                                                                    \
      Channel = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup);        \
                                                                    \
      return Channel;                                               \
    }                                                               \
                                                                    \
    float2 FetchFromHdr10ToLinearLUT                                \
    (                                                               \
      float2 Channels                                               \
    )                                                               \
    {                                                               \
      const int2 lookup = int2(Channels * 1023.f + 0.5f);           \
                                                                    \
      Channels.x = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.x);   \
      Channels.y = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.y);   \
                                                                    \
      return Channels;                                              \
    }                                                               \
                                                                    \
    float3 FetchFromHdr10ToLinearLUT                                \
    (                                                               \
      float3 Colour                                                 \
    )                                                               \
    {                                                               \
      const int3 lookup = int3(Colour * 1023.f + 0.5f);             \
                                                                    \
      Colour.r = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.r);     \
      Colour.g = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.g);     \
      Colour.b = tex1Dfetch(SamplerHdr10ToLinearLUT, lookup.b);     \
                                                                    \
      return Colour;                                                \
    }                                                               \
                                                                    \
    technique lilium__create_hdr10_to_linear_lut                    \
    <                                                               \
      enabled = true;                                               \
      hidden  = true;                                               \
      timeout = 1;                                                  \
    >                                                               \
    {                                                               \
      pass CreateHdr10ToLinearLUT                                   \
      {                                                             \
        ComputeShader = CS_CreateHdr10ToLinearLUT <16, 1>;          \
        DispatchSizeX = 64; /* (1024 / 16) */                       \
        DispatchSizeY = 1;                                          \
      }                                                             \
    }

#else

  #define HDR10_TO_LINEAR_LUT()

#endif
