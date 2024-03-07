#pragma once


#if (BUFFER_WIDTH  > 2560) \
 && (BUFFER_HEIGHT > 1440)

  #define CHAR_DIM_UINT  uint2(29, 49)
  #define CHAR_DIM_FLOAT float2(CHAR_DIM_UINT)

  #define RANGE 1.5f

#ifdef IS_HDR_CSP

  #define TEXTURE_FILENAME "lilium__font_atlas_1.333_hdr.png"

  #define FONT_TEXTURE_WIDTH  580
  #define FONT_TEXTURE_HEIGHT 881

  #define WAVEFORM_ATLAS_OFFSET float2(328, 854)

#else //IS_HDR_CSP

  #define TEXTURE_FILENAME "lilium__font_atlas_1.333_sdr.png"

  #define FONT_TEXTURE_WIDTH  696
  #define FONT_TEXTURE_HEIGHT 391

  #define WAVEFORM_ATLAS_OFFSET float2(444, 0)

#endif //IS_HDR_CSP

  #define IS_BIG_FONT_ATLAS

#else

  #define CHAR_DIM_UINT  uint2(22, 38)
  #define CHAR_DIM_FLOAT float2(CHAR_DIM_UINT)

  #define RANGE 2.f

#ifdef IS_HDR_CSP

  #define TEXTURE_FILENAME "lilium__font_atlas_1.000_hdr.png"

  #define FONT_TEXTURE_WIDTH  440
  #define FONT_TEXTURE_HEIGHT 683

  #define WAVEFORM_ATLAS_OFFSET float2(188, 656)

#else //IS_HDR_CSP

  #define TEXTURE_FILENAME "lilium__font_atlas_1.000_sdr.png"

  #define FONT_TEXTURE_WIDTH  528
  #define FONT_TEXTURE_HEIGHT 303

  #define WAVEFORM_ATLAS_OFFSET float2(276, 0)

#endif //IS_HDR_CSP

#endif


#define FONT_TEXTURE_SIZE_FLOAT float2(FONT_TEXTURE_WIDTH, FONT_TEXTURE_HEIGHT)

#define WAVEFORM_CHAR_DIM_UINT  uint2(21, 27)

#define WAVEFORM_CHAR_DIM_FLOAT float2(WAVEFORM_CHAR_DIM_UINT)


#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  #define TEXT_OFFSET_ANALYIS_HEADER float2(20,  1)

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  #define TEXT_OFFSET_ANALYIS_HEADER float2(20,  2)

#elif (ACTUAL_COLOUR_SPACE == CSP_SRGB)

  #if (OVERWRITE_SDR_GAMMA == GAMMA_24)

    #define TEXT_OFFSET_ANALYIS_HEADER float2(24,  2)

  #elif (OVERWRITE_SDR_GAMMA == CSP_SRGB)

    #define TEXT_OFFSET_ANALYIS_HEADER float2(19,  3)

  #else

    #define TEXT_OFFSET_ANALYIS_HEADER float2(24,  1)

  #endif

#endif


#ifdef IS_HDR_CSP
  #define TEXT_OFFSET_NITS_MAX_AVG_MIN     float2(15,  3)
  #define TEXT_OFFSET_NITS_CURSOR          float2(18,  6)
#else
  #define TEXT_OFFSET_NITS_MAX_AVG_MIN     float2(20,  4)
  #define TEXT_OFFSET_NITS_CURSOR          float2(23,  7)
#endif

#define TEXT_OFFSET_GAMUT_PERCENTAGES    float2(19,  7)
#define TEXT_OFFSET_GAMUT_CURSOR         float2( 7, 12)
#define TEXT_OFFSET_GAMUT_CURSOR_BT709   float2( 7, 13)
#define TEXT_OFFSET_GAMUT_CURSOR_DCI_P3  float2( 7, 14)
#define TEXT_OFFSET_GAMUT_CURSOR_BT2020  float2( 7, 15)
#define TEXT_OFFSET_GAMUT_CURSOR_AP0     float2( 7, 16)
#define TEXT_OFFSET_GAMUT_CURSOR_INVALID float2( 7, 17)


#ifdef IS_FLOAT_HDR_CSP

  #define GAMUT_PERCENTAGES_LINES 5

#else

  #define GAMUT_PERCENTAGES_LINES 3

#endif

#ifdef IS_HDR_CSP
  #define TEXT_BLOCK_ARRAY_SIZE 5
#else
  #define TEXT_BLOCK_ARRAY_SIZE 3
#endif

static const float2 TEXT_BLOCK_SIZES[TEXT_BLOCK_ARRAY_SIZE] =
{
  float2(TEXT_OFFSET_ANALYIS_HEADER.x,    1),
  float2(TEXT_OFFSET_NITS_MAX_AVG_MIN.x,  3),
  float2(TEXT_OFFSET_NITS_CURSOR.x,       1),
#ifdef IS_HDR_CSP
  float2(TEXT_OFFSET_GAMUT_PERCENTAGES.x, GAMUT_PERCENTAGES_LINES),
  float2(TEXT_OFFSET_GAMUT_CURSOR.x,      1)
#endif
};

static const float TEXT_BLOCK_DRAW_X_OFFSET[TEXT_BLOCK_ARRAY_SIZE] =
{
  0,
  3,
  0,
#ifdef IS_HDR_CSP
  3,
  4
#endif
};

static const float TEXT_BLOCK_FETCH_Y_OFFSET[TEXT_BLOCK_ARRAY_SIZE] =
{
  TEXT_OFFSET_ANALYIS_HEADER.y,
  TEXT_OFFSET_NITS_MAX_AVG_MIN.y,
  TEXT_OFFSET_NITS_CURSOR.y,
#ifdef IS_HDR_CSP
  TEXT_OFFSET_GAMUT_PERCENTAGES.y,
  TEXT_OFFSET_GAMUT_CURSOR.y
#endif
};


float GetScreenPixelRange(const float Factor)
{
  float unitRange = Factor * RANGE;

//  return max(unitRange, 1.f);
  return unitRange;
}

float GetMedian(
  const float3 Rgb)
{
  return max(min(Rgb.r, Rgb.g), min(max(Rgb.r, Rgb.g), Rgb.b));
}


#define _0         uint( 0)
#define _1         uint( 1)
#define _2         uint( 2)
#define _3         uint( 3)
#define _4         uint( 4)
#define _5         uint( 5)
#define _6         uint( 6)
#define _7         uint( 7)
#define _8         uint( 8)
#define _9         uint( 9)
#define _dot       uint(10)
#define _percent   uint(11)

#define _0_w       uint( 0)
#define _1_w       uint( 1)
#define _2_w       uint( 2)
#define _3_w       uint( 3)
#define _4_w       uint( 4)
#define _5_w       uint( 5)
#define _6_w       uint( 6)
#define _7_w       uint( 7)
#define _8_w       uint( 8)
#define _9_w       uint( 9)
#define _dot_w     uint(10)
#define _percent_w uint(11)


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
