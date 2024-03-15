#pragma once


#define CIE_TEXTURE_ENTRY_DIAGRAM_COLOUR   0
#define CIE_TEXTURE_ENTRY_DIAGRAM_BLACK_BG 1
#define CIE_TEXTURE_ENTRY_BT709_OUTLINE    2
#define CIE_TEXTURE_ENTRY_DCI_P3_OUTLINE   3
#define CIE_TEXTURE_ENTRY_BT2020_OUTLINE   4
#define CIE_TEXTURE_ENTRY_AP0_OUTLINE      5

//width and height description are in lilium__hdr_and_sdr_analysis.fx

texture2D TextureCieConsolidated
<
  source = CIE_TEXTURE_FILE_NAME;
  pooled = true;
>
{
  Width  = CIE_TEXTURE_WIDTH;
  Height = CIE_TEXTURE_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieConsolidated
{
  Texture = TextureCieConsolidated;
};

storage2D<float4> StorageCieConsolidated
{
  Texture = TextureCieConsolidated;
};

texture2D TextureCieCurrent
<
  pooled = true;
>
{
  Width  = CIE_1931_BG_WIDTH;
  Height = CIE_1931_BG_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieCurrent
{
  Texture = TextureCieCurrent;
};

storage2D<float4> StorageCieCurrent
{
  Texture = TextureCieCurrent;
};


void GenerateCieDiagram(
  const float3 XYZ)
{
  if (_CIE_DIAGRAM_TYPE == CIE_1931)
  {
    // get xy
    precise const float xyz = XYZ.x + XYZ.y + XYZ.z;

    precise int2 xy = int2(round(XYZ.x / xyz * float(CIE_ORIGINAL_DIM)),
     CIE_1931_HEIGHT - 1 - round(XYZ.y / xyz * float(CIE_ORIGINAL_DIM)));

    // adjust for the added borders
    xy += CIE_BG_BORDER;

    // clamp to borders
    xy = clamp(xy, CIE_BG_BORDER, CIE_1931_SIZE_INT + CIE_BG_BORDER);

    // leave this as sampler and not storage
    // otherwise d3d complains about the resource still being bound on input
    // D3D11 WARNING: ID3D11DeviceContext::CSSetUnorderedAccessViews:
    // Resource being set to CS UnorderedAccessView slot 3 is still bound on input!
    // [ STATE_SETTING WARNING #2097354: DEVICE_CSSETUNORDEREDACCESSVIEWS_HAZARD]
    const float4 xyColour = tex2Dfetch(SamplerCieConsolidated, xy);

    tex2Dstore(StorageCieCurrent,
               xy,
               xyColour);
  }
  else //if (_CIE_DIAGRAM_TYPE == CIE_1976)
  {
    // get u'v'
    precise const float X15Y3Z = XYZ.x
                               + 15.f * XYZ.y
                               +  3.f * XYZ.z;

    precise int2 uv = int2(round(4.f * XYZ.x / X15Y3Z * float(CIE_ORIGINAL_DIM)),
     CIE_1976_HEIGHT - 1 - round(9.f * XYZ.y / X15Y3Z * float(CIE_ORIGINAL_DIM)));

    // adjust for the added borders
    uv += CIE_BG_BORDER;

    // clamp to borders
    uv = clamp(uv, CIE_BG_BORDER, CIE_1976_SIZE_INT + CIE_BG_BORDER);

    const int2 uvFetchPos = int2(uv.x, uv.y + CIE_1931_BG_HEIGHT);

    // leave this as sampler and not storage
    // otherwise d3d complains about the resource still being bound on input
    // D3D11 WARNING: ID3D11DeviceContext::CSSetUnorderedAccessViews:
    // Resource being set to CS UnorderedAccessView slot 3 is still bound on input!
    // [ STATE_SETTING WARNING #2097354: DEVICE_CSSETUNORDEREDACCESSVIEWS_HAZARD]
    const float4 uvColour = tex2Dfetch(SamplerCieConsolidated, uvFetchPos);

    tex2Dstore(StorageCieCurrent,
               uv,
               uvColour);
  }
}
