
#include "lilium__include/cas.fxh"


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareCas(
  in                  uint   Id       : SV_VertexID,
  out                 float4 VPos     : SV_Position,
  out                 float2 TexCoord : TEXCOORD0,
  out nointerpolation float  Peak     : Peak)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

  Peak = -1.f / (-3.f * SHARPEN_AMOUNT + 8.f);
}

void PS_Cas(
  in                 float4 VPos     : SV_Position,
  in                 float2 TexCoord : TEXCOORD0,
  in nointerpolation float  Peak     : Peak,
  out                float4 Output   : SV_Target0)
{
  static const float2 coordsEfhi = GetEfhiCoords(TexCoord);

  SPixelsToProcess ptp;

  PSGetPixels(TexCoord, coordsEfhi, ptp);

  if (SHARPEN_ONLY)
  {
    Output = float4(CasSharpenOnly(ptp, Peak), 1.f);

    return;
  }
  else
  {
    Output = float4(CasSharpenAndUpscale(ptp, Peak), 1.f);

    return;
  }
}


technique lilium__cas_hdr_ps
<
#if defined(IS_POSSIBLE_HDR_CSP)
  ui_label = "Lilium's HDR Contrast Adaptive Sharpening (AMD FidelityFX CAS)";
#else
  ui_label = "Lilium's Contrast Adaptive Sharpening (AMD FidelityFX CAS)";
#endif
>
{
  pass PS_Cas
  {
    VertexShader = VS_PrepareCas;
     PixelShader = PS_Cas;
  }
}
