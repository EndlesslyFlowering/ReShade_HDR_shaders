#pragma once


#define TEXTURE_FILENAME "lilium__font_atlas_mtsdf.png"

#define CHAR_WIDTH  25
#define CHAR_HEIGHT 39

#define FONT_TEXTURE_WIDTH  (54 * CHAR_WIDTH)
#define FONT_TEXTURE_HEIGHT (13 * CHAR_HEIGHT)

#define CHAR_DIM_UINT   uint2(CHAR_WIDTH, CHAR_HEIGHT)
#define CHAR_DIM_INT     int2(CHAR_WIDTH, CHAR_HEIGHT)
#define CHAR_DIM_FLOAT float2(CHAR_WIDTH, CHAR_HEIGHT)

#define PX_RANGE 6.f

#define FONT_SIZE_MULTIPLIER (((21.f / float(CHAR_WIDTH)) + (27.f / float(CHAR_HEIGHT))) * 0.6f)


#define FONT_TEXTURE_SIZE_FLOAT float2(FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT)


#define TEXT_BLOCK_FETCH_OFFSET_ANALYIS_HEADER float2(0, 2)

#ifdef IS_HDR_CSP

  #define TEXT_BLOCK_SIZE_ANALYIS_HEADER   float2(20, 1)

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  #if (OVERWRITE_SDR_GAMMA == GAMMA_24)

    #define TEXT_BLOCK_SIZE_ANALYIS_HEADER float2(23, 1)

  #elif (OVERWRITE_SDR_GAMMA == CSP_SRGB)

    #define TEXT_BLOCK_SIZE_ANALYIS_HEADER float2(19, 1)

  #else

    #define TEXT_BLOCK_SIZE_ANALYIS_HEADER float2(23, 1)

  #endif

#endif

#define TEXT_BLOCK_FETCH_OFFSET_NITS_RGB_DESCRIPTION float2( 0, 3)
#define TEXT_BLOCK_FETCH_OFFSET_NITS_MAX_AVG_MIN     float2( 0, 4)
#define TEXT_BLOCK_FETCH_OFFSET_NITS_CURSOR          float2( 0, 7)

#ifdef IS_HDR_CSP
  #define TEXT_BLOCK_SIZE_NITS_RGB_DESCRIPTION       float2(50, 1) //       "|nits        |red (CLL)   |green (CLL) |blue (CLL)"
  #define TEXT_BLOCK_SIZE_NITS_RGB_MAX_AVG_MIN       float2(49, 3) //    "max|     .      |     .      |     .      |     ."
  #define TEXT_BLOCK_SIZE_NITS_RGB_CURSOR            float2(52, 1) // "cursor|     .      |     .      |     .      |     ."
#else
  #define TEXT_BLOCK_SIZE_NITS_RGB_DESCRIPTION       float2(47, 1) //       "|nits       |red (CLL)  |green (CLL)|blue (CLL)"
  #define TEXT_BLOCK_SIZE_NITS_RGB_MAX_AVG_MIN       float2(51, 3) //    "max|   .      %|   .      %|   .      %|   .      %"
  #define TEXT_BLOCK_SIZE_NITS_RGB_CURSOR            float2(54, 1) // "cursor|   .      %|   .      %|   .      %|   .      %"
#endif

#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_PERCENTAGES    float2( 0,  8)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR \
          float2(TEXT_BLOCK_SIZE_ANALYIS_HEADER.x + 1, TEXT_BLOCK_FETCH_OFFSET_ANALYIS_HEADER.y)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_BT709   float2(17,  8)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_DCI_P3  float2(17,  9)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_BT2020  float2(17, 10)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_AP0     float2(17, 11)
#define TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR_INVALID float2(17, 12)

#define TEXT_BLOCK_SIZE_GAMUT_PERCENTAGES            float2(17,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR_BT709           float2( 7,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR_DCI_P3          float2( 7,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR_BT2020          float2( 7,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR_AP0             float2( 7,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR_INVALID         float2( 7,  1)
#define TEXT_BLOCK_SIZE_GAMUT_CURSOR                 float2( 7,  1)


#ifdef IS_FLOAT_HDR_CSP

  #define GAMUT_PERCENTAGES_LINES 5

#else

  #define GAMUT_PERCENTAGES_LINES 3

#endif

static const float2 TEXT_BLOCK_SIZES[] =
{
  TEXT_BLOCK_SIZE_ANALYIS_HEADER,
  TEXT_BLOCK_SIZE_NITS_RGB_DESCRIPTION,
  TEXT_BLOCK_SIZE_NITS_RGB_MAX_AVG_MIN,
  TEXT_BLOCK_SIZE_NITS_RGB_CURSOR,
#ifdef IS_HDR_CSP
  float2(TEXT_BLOCK_SIZE_GAMUT_PERCENTAGES.x, GAMUT_PERCENTAGES_LINES),
  TEXT_BLOCK_SIZE_GAMUT_CURSOR
#endif
};

static const float TEXT_BLOCK_DRAW_X_OFFSET[] =
{
  0,
  6,
  3,
  0,
#ifdef IS_HDR_CSP
  0,
  1
#endif
};

static const float2 TEXT_BLOCK_FETCH_OFFSETS[] =
{
  TEXT_BLOCK_FETCH_OFFSET_ANALYIS_HEADER,
  TEXT_BLOCK_FETCH_OFFSET_NITS_RGB_DESCRIPTION,
  TEXT_BLOCK_FETCH_OFFSET_NITS_MAX_AVG_MIN,
  TEXT_BLOCK_FETCH_OFFSET_NITS_CURSOR,
#ifdef IS_HDR_CSP
  TEXT_BLOCK_FETCH_OFFSET_GAMUT_PERCENTAGES,
  TEXT_BLOCK_FETCH_OFFSET_GAMUT_CURSOR
#endif
};


namespace Msdf
{
  float GetScreenPixelRange
  (
    const float Factor
  )
  {
    return Factor * PX_RANGE;
  }

  float GetMedian
  (
    const float3 Rgb
  )
  {
    return max(min(Rgb.r, Rgb.g), min(max(Rgb.r, Rgb.g), Rgb.b));
  }

  float2 GetTextOpacities
  (
    const float4 Mtsdf,
    const float  ScreenPixelRange
  )
  {
    const float signedDistance = max(GetMedian(Mtsdf.rgb), Mtsdf.a);

    [branch]
    if (signedDistance > 0.f)
    {
      static const float innerBias = 0.05f;
      static const float outerBias = 0.4f;
      static const float outline   = 0.4f;
      static const float threshold = 0.5f;

      const float2 screenPixelDistances = ScreenPixelRange
                                        * (signedDistance - threshold + float2(innerBias, outline));

      float2 opacities = saturate(screenPixelDistances + 0.5f + float2(0.f, outerBias));
      opacities *= opacities;

      return opacities;
    }
    else
    {
      return (float2)0;
    }
  }
}



#define _0                  uint2( 0, 0)
#define _1                  uint2( 1, 0)
#define _2                  uint2( 2, 0)
#define _3                  uint2( 3, 0)
#define _4                  uint2( 4, 0)
#define _5                  uint2( 5, 0)
#define _6                  uint2( 6, 0)
#define _7                  uint2( 7, 0)
#define _8                  uint2( 8, 0)
#define _9                  uint2( 9, 0)
#define _minus              uint2(10, 0)
#define _A                  uint2(11, 0)
#define _B                  uint2(12, 0)
#define _C                  uint2(13, 0)
#define _D                  uint2(14, 0)
#define _E                  uint2(15, 0)
#define _F                  uint2(16, 0)
#define _G                  uint2(17, 0)
#define _H                  uint2(18, 0)
#define _I                  uint2(19, 0)
#define _J                  uint2(20, 0)
#define _K                  uint2(21, 0)
#define _L                  uint2(22, 0)
#define _M                  uint2(23, 0)
#define _N                  uint2(24, 0)
#define _O                  uint2(25, 0)
#define _P                  uint2(26, 0)
#define _Q                  uint2(27, 0)
#define _R                  uint2(28, 0)
#define _S                  uint2(29, 0)
#define _T                  uint2(30, 0)
#define _U                  uint2(31, 0)
#define _V                  uint2(32, 0)
#define _W                  uint2(33, 0)
#define _X                  uint2(34, 0)
#define _Y                  uint2(35, 0)
#define _Z                  uint2(36, 0)
#define _space              uint2( 0, 1)
#define _percent            uint2( 1, 1)
#define _roundBracketOpen   uint2( 2, 1)
#define _roundBracketClose  uint2( 3, 1)
#define _squareBracketOpen  uint2( 4, 1)
#define _squareBracketClose uint2( 5, 1)
#define _dot                uint2( 6, 1)
#define _colon              uint2( 7, 1)
#define _comma              uint2( 8, 1)
#define _semicolon          uint2( 9, 1)
#define _verticalLine       uint2(10, 1)
#define _a                  uint2(11, 1)
#define _b                  uint2(12, 1)
#define _c                  uint2(13, 1)
#define _d                  uint2(14, 1)
#define _e                  uint2(15, 1)
#define _f                  uint2(16, 1)
#define _g                  uint2(17, 1)
#define _h                  uint2(18, 1)
#define _i                  uint2(19, 1)
#define _j                  uint2(20, 1)
#define _k                  uint2(21, 1)
#define _l                  uint2(22, 1)
#define _m                  uint2(23, 1)
#define _n                  uint2(24, 1)
#define _o                  uint2(25, 1)
#define _p                  uint2(26, 1)
#define _q                  uint2(27, 1)
#define _r                  uint2(28, 1)
#define _s                  uint2(29, 1)
#define _t                  uint2(30, 1)
#define _u                  uint2(31, 1)
#define _v                  uint2(32, 1)
#define _w                  uint2(33, 1)
#define _x                  uint2(34, 1)
#define _y                  uint2(35, 1)
#define _z                  uint2(36, 1)


texture2D TextureFontAtlasConsolidated
<
  source = TEXTURE_FILENAME;
>
{
  Width  = FONT_TEXTURE_WIDTH;
  Height = FONT_TEXTURE_HEIGHT;
  Format = RGBA8;
};

sampler2D<float4> SamplerFontAtlasConsolidated
{
  Texture  = TextureFontAtlasConsolidated;
  AddressU = BORDER;
  AddressV = BORDER;
  AddressW = BORDER;
};

storage2D<float4> StorageFontAtlasConsolidated
{
  Texture = TextureFontAtlasConsolidated;
};
