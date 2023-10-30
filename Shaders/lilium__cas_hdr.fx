
#include "lilium__include/colour_space.fxh"

//#define ATOMIC


uniform bool SharpenOnly
<
  ui_label   = "sharpen only path";
  ui_tooltip = "If unchecked will use the upscale path of CAS."
          "\n" "Which uses more samples and may lead to higher quality."
          "\n" "While also costing a bit more performance."
          "\n" "But does not do any upscaling at all!";
> = true;

uniform float CasSharpness
<
  ui_label   = "sharpness";
  ui_tooltip = "Even a value of 0 applies a bit of sharpness!";
  ui_type    = "drag";
  ui_min     = 0.f;
  ui_max     = 1.f;
  ui_step    = 0.001f;
> = 0.f;

uniform bool FFX_CAS_USE_PRECISE_MATH
<
  ui_label   = "use precise math";
  ui_tooltip = "Leads to slightly better quality."
          "\n" "At the cost of a bit of performance.";
> = false;

uniform bool FFX_CAS_BETTER_DIAGONALS
<
  ui_label   = "better diagonals";
  ui_tooltip = "Leads to slightly better quality."
          "\n" "At the cost of a bit of performance.";
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


#define FFX_GPU  1
#define FFX_HLSL 1

#include "lilium__include/cas/gpu/cas/ffx_cas_callbacks_hlsl.h"
#include "lilium__include/cas/gpu/cas/ffx_cas_sharpen.h"


#ifdef ATOMIC
groupshared int casPeak;
#endif
void CS_Cas(
  uint3 LocalThreadId    : SV_GroupThreadID,
  uint3 WorkGroupId      : SV_GroupID,
  uint3 DispatchThreadID : SV_DispatchThreadID)
{
#ifdef ATOMIC
  if (all(LocalThreadId == uint3(0, 0, 0)))
  {
    casPeak = asint(-ffxReciprocal(ffxLerp(8.0, 5.0, CasSharpness)));
  }

  barrier();
  memoryBarrier();
#endif

#ifdef ATOMIC
  Sharpen(LocalThreadId, WorkGroupId, DispatchThreadID, asfloat(casPeak));
#else
  Sharpen(LocalThreadId, WorkGroupId, DispatchThreadID);
#endif
  return;
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
  ui_label = "Lilium's HDR Contrast Adaptive Sharpening (CAS)";
#else
  ui_label = "Lilium's Contrast Adaptive Sharpening (CAS)";
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
