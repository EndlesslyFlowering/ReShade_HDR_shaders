#pragma once


#define FONT_SIZE_58_CHAR_DIM uint2(45, 72)
#define FONT_SIZE_56_CHAR_DIM uint2(44, 69)
#define FONT_SIZE_54_CHAR_DIM uint2(43, 67)
#define FONT_SIZE_52_CHAR_DIM uint2(42, 64)
#define FONT_SIZE_50_CHAR_DIM uint2(40, 62)
#define FONT_SIZE_48_CHAR_DIM uint2(39, 60)
#define FONT_SIZE_46_CHAR_DIM uint2(36, 57)
#define FONT_SIZE_44_CHAR_DIM uint2(35, 55)
#define FONT_SIZE_42_CHAR_DIM uint2(34, 52)
#define FONT_SIZE_40_CHAR_DIM uint2(30, 49)
#define FONT_SIZE_38_CHAR_DIM uint2(29, 47)
#define FONT_SIZE_36_CHAR_DIM uint2(28, 45)
#define FONT_SIZE_34_CHAR_DIM uint2(27, 42)
#define FONT_SIZE_32_CHAR_DIM uint2(26, 40)
#define FONT_SIZE_30_CHAR_DIM uint2(22, 37)
#define FONT_SIZE_28_CHAR_DIM uint2(21, 34)
#define FONT_SIZE_26_CHAR_DIM uint2(20, 32)
#define FONT_SIZE_24_CHAR_DIM uint2(19, 30)
#define FONT_SIZE_22_CHAR_DIM uint2(18, 28)
#define FONT_SIZE_20_CHAR_DIM uint2(14, 24)
#define FONT_SIZE_18_CHAR_DIM uint2(13, 22)
#define FONT_SIZE_16_CHAR_DIM uint2(12, 20)
#define FONT_SIZE_14_CHAR_DIM uint2(11, 18)
#define FONT_SIZE_13_CHAR_DIM uint2(10, 16)

#define NUMBER_OF_CHARS_X uint(1)
#define NUMBER_OF_CHARS_Y uint(46)

static const uint CharSize[48] = {
  FONT_SIZE_58_CHAR_DIM.x, FONT_SIZE_58_CHAR_DIM.y,
  FONT_SIZE_56_CHAR_DIM.x, FONT_SIZE_56_CHAR_DIM.y,
  FONT_SIZE_54_CHAR_DIM.x, FONT_SIZE_54_CHAR_DIM.y,
  FONT_SIZE_52_CHAR_DIM.x, FONT_SIZE_52_CHAR_DIM.y,
  FONT_SIZE_50_CHAR_DIM.x, FONT_SIZE_50_CHAR_DIM.y,
  FONT_SIZE_48_CHAR_DIM.x, FONT_SIZE_48_CHAR_DIM.y,
  FONT_SIZE_46_CHAR_DIM.x, FONT_SIZE_46_CHAR_DIM.y,
  FONT_SIZE_44_CHAR_DIM.x, FONT_SIZE_44_CHAR_DIM.y,
  FONT_SIZE_42_CHAR_DIM.x, FONT_SIZE_42_CHAR_DIM.y,
  FONT_SIZE_40_CHAR_DIM.x, FONT_SIZE_40_CHAR_DIM.y,
  FONT_SIZE_38_CHAR_DIM.x, FONT_SIZE_38_CHAR_DIM.y,
  FONT_SIZE_36_CHAR_DIM.x, FONT_SIZE_36_CHAR_DIM.y,
  FONT_SIZE_34_CHAR_DIM.x, FONT_SIZE_34_CHAR_DIM.y,
  FONT_SIZE_32_CHAR_DIM.x, FONT_SIZE_32_CHAR_DIM.y,
  FONT_SIZE_30_CHAR_DIM.x, FONT_SIZE_30_CHAR_DIM.y,
  FONT_SIZE_28_CHAR_DIM.x, FONT_SIZE_28_CHAR_DIM.y,
  FONT_SIZE_26_CHAR_DIM.x, FONT_SIZE_26_CHAR_DIM.y,
  FONT_SIZE_24_CHAR_DIM.x, FONT_SIZE_24_CHAR_DIM.y,
  FONT_SIZE_22_CHAR_DIM.x, FONT_SIZE_22_CHAR_DIM.y,
  FONT_SIZE_20_CHAR_DIM.x, FONT_SIZE_20_CHAR_DIM.y,
  FONT_SIZE_18_CHAR_DIM.x, FONT_SIZE_18_CHAR_DIM.y,
  FONT_SIZE_16_CHAR_DIM.x, FONT_SIZE_16_CHAR_DIM.y,
  FONT_SIZE_14_CHAR_DIM.x, FONT_SIZE_14_CHAR_DIM.y,
  FONT_SIZE_13_CHAR_DIM.x, FONT_SIZE_13_CHAR_DIM.y };

uint2 GetCharSize(const uint CharArrayEntry)
{
  return uint2(CharSize[CharArrayEntry], CharSize[CharArrayEntry + 1]);
}

#define ATLAS_X_OFFSET_58 0
#define ATLAS_X_OFFSET_56 (FONT_SIZE_58_CHAR_DIM.x)
#define ATLAS_X_OFFSET_54 (ATLAS_X_OFFSET_56 + FONT_SIZE_56_CHAR_DIM.x)
#define ATLAS_X_OFFSET_52 (ATLAS_X_OFFSET_54 + FONT_SIZE_54_CHAR_DIM.x)
#define ATLAS_X_OFFSET_50 (ATLAS_X_OFFSET_52 + FONT_SIZE_52_CHAR_DIM.x)
#define ATLAS_X_OFFSET_48 (ATLAS_X_OFFSET_50 + FONT_SIZE_50_CHAR_DIM.x)
#define ATLAS_X_OFFSET_46 (ATLAS_X_OFFSET_48 + FONT_SIZE_48_CHAR_DIM.x)
#define ATLAS_X_OFFSET_44 (ATLAS_X_OFFSET_46 + FONT_SIZE_46_CHAR_DIM.x)
#define ATLAS_X_OFFSET_42 (ATLAS_X_OFFSET_44 + FONT_SIZE_44_CHAR_DIM.x)
#define ATLAS_X_OFFSET_40 (ATLAS_X_OFFSET_42 + FONT_SIZE_42_CHAR_DIM.x)
#define ATLAS_X_OFFSET_38 (ATLAS_X_OFFSET_40 + FONT_SIZE_40_CHAR_DIM.x)
#define ATLAS_X_OFFSET_36 (ATLAS_X_OFFSET_38 + FONT_SIZE_38_CHAR_DIM.x)
#define ATLAS_X_OFFSET_34 (ATLAS_X_OFFSET_36 + FONT_SIZE_36_CHAR_DIM.x)
#define ATLAS_X_OFFSET_32 (ATLAS_X_OFFSET_34 + FONT_SIZE_34_CHAR_DIM.x)
#define ATLAS_X_OFFSET_30 (ATLAS_X_OFFSET_32 + FONT_SIZE_32_CHAR_DIM.x)
#define ATLAS_X_OFFSET_28 (ATLAS_X_OFFSET_30 + FONT_SIZE_30_CHAR_DIM.x)
#define ATLAS_X_OFFSET_26 (ATLAS_X_OFFSET_28 + FONT_SIZE_28_CHAR_DIM.x)
#define ATLAS_X_OFFSET_24 (ATLAS_X_OFFSET_26 + FONT_SIZE_26_CHAR_DIM.x)
#define ATLAS_X_OFFSET_22 (ATLAS_X_OFFSET_24 + FONT_SIZE_24_CHAR_DIM.x)
#define ATLAS_X_OFFSET_20 (ATLAS_X_OFFSET_22 + FONT_SIZE_22_CHAR_DIM.x)
#define ATLAS_X_OFFSET_18 (ATLAS_X_OFFSET_20 + FONT_SIZE_20_CHAR_DIM.x)
#define ATLAS_X_OFFSET_16 (ATLAS_X_OFFSET_18 + FONT_SIZE_18_CHAR_DIM.x)
#define ATLAS_X_OFFSET_14 (ATLAS_X_OFFSET_16 + FONT_SIZE_16_CHAR_DIM.x)
#define ATLAS_X_OFFSET_13 (ATLAS_X_OFFSET_14 + FONT_SIZE_14_CHAR_DIM.x)

static const uint AtlasXOffset[24] = {
  ATLAS_X_OFFSET_58,
  ATLAS_X_OFFSET_56,
  ATLAS_X_OFFSET_54,
  ATLAS_X_OFFSET_52,
  ATLAS_X_OFFSET_50,
  ATLAS_X_OFFSET_48,
  ATLAS_X_OFFSET_46,
  ATLAS_X_OFFSET_44,
  ATLAS_X_OFFSET_42,
  ATLAS_X_OFFSET_40,
  ATLAS_X_OFFSET_38,
  ATLAS_X_OFFSET_36,
  ATLAS_X_OFFSET_34,
  ATLAS_X_OFFSET_32,
  ATLAS_X_OFFSET_30,
  ATLAS_X_OFFSET_28,
  ATLAS_X_OFFSET_26,
  ATLAS_X_OFFSET_24,
  ATLAS_X_OFFSET_22,
  ATLAS_X_OFFSET_20,
  ATLAS_X_OFFSET_18,
  ATLAS_X_OFFSET_16,
  ATLAS_X_OFFSET_14,
  ATLAS_X_OFFSET_13 };


//waveform font atlas
#define WAVE_TEXTURE_OFFSET uint2(480, 2964)

#define WAVE_NUMBER_OF_CHARS_X uint(1)
#define WAVE_NUMBER_OF_CHARS_Y uint(11)

#define WAVE_FONT_SIZE_32_CHAR_DIM uint2(24, 29)
#define WAVE_FONT_SIZE_30_CHAR_DIM uint2(22, 28)
#define WAVE_FONT_SIZE_28_CHAR_DIM uint2(21, 26)
#define WAVE_FONT_SIZE_26_CHAR_DIM uint2(18, 22)
#define WAVE_FONT_SIZE_24_CHAR_DIM uint2(17, 21)
#define WAVE_FONT_SIZE_22_CHAR_DIM uint2(16, 19)
#define WAVE_FONT_SIZE_20_CHAR_DIM uint2(14, 17)
#define WAVE_FONT_SIZE_18_CHAR_DIM uint2(13, 16)
#define WAVE_FONT_SIZE_16_CHAR_DIM uint2(12, 15)
#define WAVE_FONT_SIZE_14_CHAR_DIM uint2(11, 14)
#define WAVE_FONT_SIZE_13_CHAR_DIM uint2(10, 13)

static const uint WaveCharSize[22] = {
  WAVE_FONT_SIZE_32_CHAR_DIM.x, WAVE_FONT_SIZE_32_CHAR_DIM.y,
  WAVE_FONT_SIZE_30_CHAR_DIM.x, WAVE_FONT_SIZE_30_CHAR_DIM.y,
  WAVE_FONT_SIZE_28_CHAR_DIM.x, WAVE_FONT_SIZE_28_CHAR_DIM.y,
  WAVE_FONT_SIZE_26_CHAR_DIM.x, WAVE_FONT_SIZE_26_CHAR_DIM.y,
  WAVE_FONT_SIZE_24_CHAR_DIM.x, WAVE_FONT_SIZE_24_CHAR_DIM.y,
  WAVE_FONT_SIZE_22_CHAR_DIM.x, WAVE_FONT_SIZE_22_CHAR_DIM.y,
  WAVE_FONT_SIZE_20_CHAR_DIM.x, WAVE_FONT_SIZE_20_CHAR_DIM.y,
  WAVE_FONT_SIZE_18_CHAR_DIM.x, WAVE_FONT_SIZE_18_CHAR_DIM.y,
  WAVE_FONT_SIZE_16_CHAR_DIM.x, WAVE_FONT_SIZE_16_CHAR_DIM.y,
  WAVE_FONT_SIZE_14_CHAR_DIM.x, WAVE_FONT_SIZE_14_CHAR_DIM.y,
  WAVE_FONT_SIZE_13_CHAR_DIM.x, WAVE_FONT_SIZE_13_CHAR_DIM.y };

#define WAVE_ATLAS_X_OFFSET_32 (WAVE_TEXTURE_OFFSET.x)
#define WAVE_ATLAS_X_OFFSET_30 (WAVE_ATLAS_X_OFFSET_32 + WAVE_FONT_SIZE_32_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_28 (WAVE_ATLAS_X_OFFSET_30 + WAVE_FONT_SIZE_30_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_26 (WAVE_ATLAS_X_OFFSET_28 + WAVE_FONT_SIZE_28_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_24 (WAVE_ATLAS_X_OFFSET_26 + WAVE_FONT_SIZE_26_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_22 (WAVE_ATLAS_X_OFFSET_24 + WAVE_FONT_SIZE_24_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_20 (WAVE_ATLAS_X_OFFSET_22 + WAVE_FONT_SIZE_22_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_18 (WAVE_ATLAS_X_OFFSET_20 + WAVE_FONT_SIZE_20_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_16 (WAVE_ATLAS_X_OFFSET_18 + WAVE_FONT_SIZE_18_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_14 (WAVE_ATLAS_X_OFFSET_16 + WAVE_FONT_SIZE_16_CHAR_DIM.x)
#define WAVE_ATLAS_X_OFFSET_13 (WAVE_ATLAS_X_OFFSET_14 + WAVE_FONT_SIZE_14_CHAR_DIM.x)

static const uint WaveAtlasXOffset[11] = {
  WAVE_ATLAS_X_OFFSET_32,
  WAVE_ATLAS_X_OFFSET_30,
  WAVE_ATLAS_X_OFFSET_28,
  WAVE_ATLAS_X_OFFSET_26,
  WAVE_ATLAS_X_OFFSET_24,
  WAVE_ATLAS_X_OFFSET_22,
  WAVE_ATLAS_X_OFFSET_20,
  WAVE_ATLAS_X_OFFSET_18,
  WAVE_ATLAS_X_OFFSET_16,
  WAVE_ATLAS_X_OFFSET_14,
  WAVE_ATLAS_X_OFFSET_13 };


#define _0                 uint( 0)
#define _1                 uint( 1)
#define _2                 uint( 2)
#define _3                 uint( 3)
#define _4                 uint( 4)
#define _5                 uint( 5)
#define _6                 uint( 6)
#define _7                 uint( 7)
#define _8                 uint( 8)
#define _9                 uint( 9)
#define _A                 uint(10)
#define _B                 uint(11)
#define _C                 uint(12)
#define _D                 uint(13)
#define _G                 uint(14)
#define _H                 uint(15)
#define _I                 uint(16)
#define _N                 uint(17)
#define _P                 uint(18)
#define _R                 uint(19)
#define _S                 uint(20)
#define _T                 uint(21)
#define _a                 uint(22)
#define _c                 uint(23)
#define _d                 uint(24)
#define _g                 uint(25)
#define _i                 uint(26)
#define _k                 uint(27)
#define _l                 uint(28)
#define _m                 uint(29)
#define _n                 uint(30)
#define _o                 uint(31)
#define _r                 uint(32)
#define _s                 uint(33)
#define _t                 uint(34)
#define _u                 uint(35)
#define _v                 uint(36)
#define _w                 uint(37)
#define _x                 uint(38)
#define _y                 uint(39)
#define _percent           uint(40)
#define _roundBracketOpen  uint(41)
#define _roundBracketClose uint(42)
#define _minus             uint(43)
#define _dot               uint(44)
#define _colon             uint(45)

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
  source = "lilium__font_atlas_consolidated.png";
>
{
  Width  =  658;
  Height = 3312;
  Format = RG8;
};

sampler2D<float4> SamplerFontAtlasConsolidated
{
  Texture = TextureFontAtlasConsolidated;
};
