
//#define ATOMIC

uniform int CAS_ABOUT
<
  ui_category = "About";
  ui_label    = " ";
  ui_type     = "radio";
  ui_text     = "AMD FidelityFX Contrast Adaptive Sharpening 1.1"
           "\n" "FidelityFX Contrast Adaptive Sharpening (CAS) is a low overhead adaptive sharpening algorithm with optional up-sampling."
                "The technique is developed by Timothy Lottes (creator of FXAA) and was created to provide natural sharpness without artifacts.";
>;

uniform bool SHARPEN_ONLY
<
  ui_label   = "sharpen only path";
  ui_tooltip = "If unchecked will use the upscale path of CAS."
          "\n" "Which does processing a little different."
          "\n" "But does not do any upscaling at all!";
> = true;

uniform float SHARPEN_AMOUNT
<
  ui_label   = "sharpness amount";
  ui_tooltip = "Even a value of 0 applies a bit of sharpness!";
  ui_type    = "drag";
  ui_min     = 0.f;
  ui_max     = 1.f;
  ui_step    = 0.001f;
> = 0.f;

uniform float APPLY_AMOUNT
<
  ui_label   = "amount of sharpness to apply";
  ui_tooltip = "How much of the sharpness to apply to the final image.";
  ui_type    = "drag";
  ui_min     = 0.f;
  ui_max     = 1.f;
  ui_step    = 0.001f;
> = 1.f;

uniform bool WEIGH_BY_ALL_CHANNELS
<
  ui_label = "apply sharpen weight by all channels";
> = false;


//#ifndef USE_32_BIT_FLOAT_BUFFER
//  #define USE_32_BIT_FLOAT_BUFFER NO
//#endif


texture2D TextureCas
{
	Width  = BUFFER_WIDTH;
	Height = BUFFER_HEIGHT;

#if (USE_32_BIT_FLOAT_BUFFER == YES)
  Format = RGBA32F;
#else
  Format = RGBA16F;
#endif
};

sampler2D<float4> SamplerCas
{
  Texture = TextureCas;

  AddressU = BORDER;
  AddressV = BORDER;
  AddressW = BORDER;
};

storage2D<float4> StorageCas
{
  Texture = TextureCas;
};


#include "lilium__include/cas.fxh"


void CS_Cas(
  uint3 LocalThreadId : SV_GroupThreadID,
  uint3 WorkGroupId   : SV_GroupID)
{
  static const int2 coords = RemapForQuad(LocalThreadId.x)
                           + uint2(WorkGroupId.x << 4u, WorkGroupId.y << 4u);

  static const int2 coords8 = coords + 8;

  static const float2 coordsEfhi  = GetEfhiCoords(coords);
  static const float2 coordsEfhi8 = GetEfhiCoords(coords8);

  if (SHARPEN_ONLY)
  {
    float4 sharpened = float4(CasSharpenOnly(coords, coordsEfhi), 1.f);
    tex2Dstore(StorageCas, coords, sharpened);

    int2 curCoords = int2(coords8.x, coords.y);
    float2 curEfhiCoords = float2(coordsEfhi8.x, coordsEfhi.y);

    sharpened.rgb = CasSharpenOnly(curCoords, curEfhiCoords);
    tex2Dstore(StorageCas, curCoords, sharpened);

    sharpened.rgb = CasSharpenOnly(coords8, coordsEfhi8);
    tex2Dstore(StorageCas, coords8, sharpened);

    curCoords = int2(coords.x, coords8.y);
    curEfhiCoords = float2(coordsEfhi.x, coordsEfhi8.y);

    sharpened.rgb = CasSharpenOnly(curCoords, curEfhiCoords);
    tex2Dstore(StorageCas, curCoords, sharpened);
  }
  else
  {
    float4 sharpened = float4(CasSharpenAndUpscale(coords, coordsEfhi), 1.f);
    tex2Dstore(StorageCas, coords, sharpened);

    int2 curCoords = int2(coords8.x, coords.y);
    float2 curEfhiCoords = float2(coordsEfhi8.x, coordsEfhi.y);

    sharpened.rgb = CasSharpenAndUpscale(curCoords, curEfhiCoords);
    tex2Dstore(StorageCas, curCoords, sharpened);

    sharpened.rgb = CasSharpenAndUpscale(coords8, coordsEfhi8);
    tex2Dstore(StorageCas, coords8, sharpened);

    curCoords = int2(coords.x, coords8.y);
    curEfhiCoords = float2(coordsEfhi.x, coordsEfhi8.y);

    sharpened.rgb = CasSharpenAndUpscale(curCoords, curEfhiCoords);
    tex2Dstore(StorageCas, curCoords, sharpened);
  }

}


void PS_OutputCas(
  in  float4 VPos     : SV_Position,
  in  float2 TexCoord : TEXCOORD0,
  out float4 Output   : SV_Target0)
{
  Output = tex2D(SamplerCas, TexCoord);
}

technique lilium__cas_hdr
<
#if defined(IS_POSSIBLE_HDR_CSP)
  ui_label = "Lilium's HDR Contrast Adaptive Sharpening (AMD FidelityFX CAS)";
#else
  ui_label = "Lilium's Contrast Adaptive Sharpening (AMD FidelityFX CAS)";
#endif
>
{
#if (BUFFER_WIDTH % 16 != 0)
  #define CAS_DISPATCH_X (BUFFER_WIDTH / 16) + 1
#else
  #define CAS_DISPATCH_X BUFFER_WIDTH / 16
#endif

#if (BUFFER_HEIGHT % 16 != 0)
  #define CAS_DISPATCH_Y (BUFFER_HEIGHT / 16) + 1
#else
  #define CAS_DISPATCH_Y BUFFER_HEIGHT / 16
#endif

  pass CS_Cas
  {
    ComputeShader = CS_Cas <64, 1, 1>;
    DispatchSizeX = CAS_DISPATCH_X;
    DispatchSizeY = CAS_DISPATCH_Y;
    DispatchSizeZ = 1;
  }

  pass PS_OutputCas
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_OutputCas;
  }
}
