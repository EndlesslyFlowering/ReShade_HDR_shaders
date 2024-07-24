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

texture2D TextureCieIntermediate
<
  pooled = true;
>
{
  Width  = CIE_1931_BG_WIDTH;
  Height = CIE_1931_BG_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieIntermediate
{
  Texture = TextureCieIntermediate;
};

storage2D<float4> StorageCieIntermediate
{
  Texture = TextureCieIntermediate;
};

texture2D TextureCieFinal
<
  pooled = true;
>
{
  Width  = CIE_1931_BG_WIDTH;
  Height = CIE_1931_BG_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerCieFinal
{
  Texture = TextureCieFinal;
};

storage2D<float4> StorageCieFinal
{
  Texture = TextureCieFinal;
};


float4 FetchGamutOutline
(
  const int OutlineTextureOffset,
  const int CieBgWidth,
  const int PositionXAsInt,
  const int FetchPosY
)
{
  int2 fetchPos =
    int2(PositionXAsInt.x + (CieBgWidth * OutlineTextureOffset),
         FetchPosY);

  float4 fetchedPixel = tex2Dfetch(SamplerCieConsolidated, fetchPos);

  // using gamma 2 as intermediate gamma space
  fetchedPixel.rgb *= fetchedPixel.rgb;

  return fetchedPixel;
}

// draw the gamut outlines on the CIE diagram
void PS_DrawCieGamutOutlines
(
  in  float4 Position : SV_Position,
  out float4 Out      : SV_Target0
)
{
  const int2 positionAsInt2 = int2(Position.xy);

  const int fetchPosY = positionAsInt2.y + int(_CIE_DIAGRAM_TYPE) * int(CIE_1931_BG_HEIGHT);

  const int2 bgFetchPos = int2(positionAsInt2.x + CIE_BG_WIDTH_INT[_CIE_DIAGRAM_TYPE],
                               fetchPosY);

  Out = tex2Dfetch(SamplerCieConsolidated, bgFetchPos);

  // using gamma 2 as intermediate gamma space
  Out.rgb *= Out.rgb;

  float4 cieCurrent = tex2Dfetch(SamplerCieIntermediate, positionAsInt2);

  cieCurrent.rgb *= cieCurrent.rgb;

  Out += cieCurrent;

  BRANCH(x)
  if (_CIE_SHOW_GAMUT_BT709_OUTLINE)
  {
    float4 fetchedPixel = FetchGamutOutline(int(CIE_TEXTURE_ENTRY_BT709_OUTLINE),
                                            CIE_BG_WIDTH_INT[_CIE_DIAGRAM_TYPE],
                                            positionAsInt2.x,
                                            fetchPosY);

    Out += fetchedPixel;
  }
#ifdef IS_HDR_CSP
  BRANCH(x)
  if (CIE_SHOW_GAMUT_DCI_P3_OUTLINE)
  {
    float4 fetchedPixel = FetchGamutOutline(int(CIE_TEXTURE_ENTRY_DCI_P3_OUTLINE),
                                            CIE_BG_WIDTH_INT[_CIE_DIAGRAM_TYPE],
                                            positionAsInt2.x,
                                            fetchPosY);

    Out += fetchedPixel;
  }
  BRANCH(x)
  if (CIE_SHOW_GAMUT_BT2020_OUTLINE)
  {
    float4 fetchedPixel = FetchGamutOutline(int(CIE_TEXTURE_ENTRY_BT2020_OUTLINE),
                                            CIE_BG_WIDTH_INT[_CIE_DIAGRAM_TYPE],
                                            positionAsInt2.x,
                                            fetchPosY);

    Out += fetchedPixel;
  }
#ifdef IS_FLOAT_HDR_CSP
  BRANCH(x)
  if (CIE_SHOW_GAMUT_AP0_OUTLINE)
  {
    float4 fetchedPixel = FetchGamutOutline(int(CIE_TEXTURE_ENTRY_AP0_OUTLINE),
                                            CIE_BG_WIDTH_INT[_CIE_DIAGRAM_TYPE],
                                            positionAsInt2.x,
                                            fetchPosY);

    Out += fetchedPixel;
  }
#endif
#endif

  // using gamma 2 as intermediate gamma space
  Out.rgb = sqrt(Out.rgb);

  return;
}


float3 GetXYZFromRgb
(
  const float3 Rgb
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  const float3 XYZ = Csp::Mat::Bt709To::XYZ(Rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::PqTo::Linear(Rgb));

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Csp::Trc::HlgTo::Linear(Rgb));

#elif (ACTUAL_COLOUR_SPACE == CSP_PS5)

  const float3 XYZ = Csp::Mat::Bt2020To::XYZ(Rgb);

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  const float3 XYZ  = Csp::Mat::Bt709To::XYZ(DECODE_SDR(Rgb));

#else

  const float3 XYZ = float3(0.f, 0.f, 0.f);

#endif

  return XYZ;
}

int2 GetxyFromXYZForDiagram
(
  const float3 XYZ
)
{
  const float xyz = XYZ.x + XYZ.y + XYZ.z;

           int2 xy = int2(round(XYZ.x / xyz * float(CIE_ORIGINAL_DIM)),
    CIE_1931_HEIGHT - 1 - round(XYZ.y / xyz * float(CIE_ORIGINAL_DIM)));

  // adjust for the added borders
  xy += CIE_BG_BORDER;

  // clamp to borders
  xy = clamp(xy, CIE_BG_BORDER, CIE_1931_SIZE_INT + CIE_BG_BORDER);

  return xy;
}

int2 GetuvFromXYZForDiagram
(
  const float3 XYZ
)
{
  const float X15Y3Z = XYZ.x
                     + 15.f * XYZ.y
                     +  3.f * XYZ.z;

           int2 uv = int2(round(4.f * XYZ.x / X15Y3Z * float(CIE_ORIGINAL_DIM)),
    CIE_1976_HEIGHT - 1 - round(9.f * XYZ.y / X15Y3Z * float(CIE_ORIGINAL_DIM)));

  // adjust for the added borders
  uv += CIE_BG_BORDER;

  // clamp to borders
  uv = clamp(uv, CIE_BG_BORDER, CIE_1976_SIZE_INT + CIE_BG_BORDER);

  return uv;
}


void GenerateCieDiagram
(
  const float3 XYZ
)
{
  BRANCH(x)
  if (_CIE_DIAGRAM_TYPE == CIE_1931)
  {
    // get xy
    const int2 xy = GetxyFromXYZForDiagram(XYZ);

    // leave this as sampler and not storage
    // otherwise d3d complains about the resource still being bound on input
    // D3D11 WARNING: ID3D11DeviceContext::CSSetUnorderedAccessViews:
    // Resource being set to CS UnorderedAccessView slot 3 is still bound on input!
    // [ STATE_SETTING WARNING #2097354: DEVICE_CSSETUNORDEREDACCESSVIEWS_HAZARD]
    const float4 xyColour = tex2Dfetch(SamplerCieConsolidated, xy);

    tex2Dstore(StorageCieIntermediate,
               xy,
               xyColour);
  }
  else //if (_CIE_DIAGRAM_TYPE == CIE_1976)
  {
    // get u'v'
    const int2 uv = GetuvFromXYZForDiagram(XYZ);

    const int2 uvFetchPos = int2(uv.x, uv.y + CIE_1931_BG_HEIGHT);

    // leave this as sampler and not storage
    // otherwise d3d complains about the resource still being bound on input
    // D3D11 WARNING: ID3D11DeviceContext::CSSetUnorderedAccessViews:
    // Resource being set to CS UnorderedAccessView slot 3 is still bound on input!
    // [ STATE_SETTING WARNING #2097354: DEVICE_CSSETUNORDEREDACCESSVIEWS_HAZARD]
    const float4 uvColour = tex2Dfetch(SamplerCieConsolidated, uvFetchPos);

    tex2Dstore(StorageCieIntermediate,
               uv,
               uvColour);
  }
}

groupshared int2 StorePos;
void CS_RenderCrosshairToCieDiagram
(
  uint3 DTID : SV_DispatchThreadID
)
{
  static const float4 storeColourBlack = float4(0.f, 0.f, 0.f, 1.f);
  static const float4 storeColourWhite = float4(1.f, 1.f, 1.f, 1.f);

  BRANCH(x)
  if (_SHOW_CROSSHAIR_ON_CIE_DIAGRAM)
  {

    if (all(DTID.xy == 0))
    {
      const float3 cursorRgb = tex2Dfetch(SamplerBackBuffer, MOUSE_POSITION).rgb;

      const float3 cursorXYZ = GetXYZFromRgb(cursorRgb);

      [branch]
      if (cursorXYZ.y > 0.f)
      {
        BRANCH(x)
        if (_CIE_DIAGRAM_TYPE == CIE_1931)
        {
          // get xy
          StorePos = GetxyFromXYZForDiagram(cursorXYZ);
        }
        else //if (_CIE_DIAGRAM_TYPE == CIE_1976)
        {
          // get u'v'
          StorePos = GetuvFromXYZForDiagram(cursorXYZ);
        }
      }
      else
      {
        StorePos = int2(0, 0);
      }
    }
    barrier();

    [branch]
    if (all(StorePos != 0))
    {
      static const int2 DTIDInt2 = int2(DTID.xy);

      int2 posX;

      posX.x = DTIDInt2.x + 4;
      posX.y = DTIDInt2.y - 5;

      int2 negX = int2(-posX.x,
                        posX.y);

      int2 posY = int2(posX.y,
                       posX.x);

      int2 negY = int2(posX.y,
                       negX.x);

      float4 storeColour;

      const uint absPosXy = abs(posX.y);

      [branch]
      if ((absPosXy == 5 && posX.x == 16)
       || (absPosXy == 3 && posX.x ==  4)
       || ((absPosXy == 4 || absPosXy == 5) && (posX.x == 4 || posX.x == 5 || posX.x == 17))
       || (absPosXy == 5 && posX.x ==  6))
      {
        return;
      }
      else if (absPosXy >   3
            || posX.x   ==  4
            || posX.x   ==  5
            || posX.x   >  15
            || (absPosXy ==  2 && posX.x == 6)
            || (absPosXy ==  3 && (posX.x ==  6
                                || posX.x ==  7
                                || posX.x == 15)))
      {
        storeColour = float4(1.f, 1.f, 1.f, 1.f);
      }
      else
      {
        storeColour = float4(0.f, 0.f, 0.f, 1.f);
      }

      tex2Dstore(StorageCieFinal,
                 StorePos + posX,
                 storeColour);

      tex2Dstore(StorageCieFinal,
                 StorePos + negX,
                 storeColour);

      tex2Dstore(StorageCieFinal,
                 StorePos + posY,
                 storeColour);

      tex2Dstore(StorageCieFinal,
                 StorePos + negY,
                 storeColour);
    }
    else
    {
      return;
    }
  }
  else
  {
    return;
  }
}
