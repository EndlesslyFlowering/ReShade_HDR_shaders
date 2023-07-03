#include "lilium__include\tone_mappers.fxh"
#include "lilium__include\draw_text_fix.fxh"

#if 0
#include "lilium__include\HDR_black_floor_fix.fxh"
#endif


//#define _DEBUG


//ideas:
// - average maxCLL over last 60? frames -> save last 100/1000 CLL values and their frametime and average over that
// - maxCLL adapation: fade to actual maxCLL over time (should read time to render per frame for this); make adjustable


#ifndef SHOW_ADAPTIVE_MAXCLL
  #define SHOW_ADAPTIVE_MAXCLL NO
#endif


uniform float FRAMETIME
<
  source = "frametime";
>;

#ifdef _DEBUG
uniform float TEST_H
<
  ui_category = "mod1";
  ui_label    = "testH";
  ui_type     = "drag";
  ui_min      = -10000.f;
  ui_max      =  10000.f;
  ui_step     =  10.f;
> = 0.f;

uniform float TEST_S
<
  ui_category = "mod1";
  ui_label    = "testS";
  ui_type     = "drag";
  ui_min      = -10000.f;
  ui_max      =  10000.f;
  ui_step     =  10.f;
> = 0.f;
#endif

uniform uint TONE_MAPPING_METHOD
<
  hidden      = false;
  ui_category = "global";
  ui_label    = "tone mapping method";
  ui_type     = "combo";
  ui_items    = "BT.2446 Method A\0"
                "BT.2390\0"
                "Dice\0"
#ifdef _DEBUG
                "BT.2446A mod1\0"
#endif
                ;
//                "BT.2446A mod2\0"
//                "BT.2446A mod1+mod2\0";
> = 0;

#define TM_METHOD_BT2446A      0
#define TM_METHOD_BT2390       1
#define TM_METHOD_DICE         2
#define TM_METHOD_BT2446A_MOD1 3

//global
uniform uint CLL_MODE
<
  ui_category = "global";
  ui_label    = "tone mapping mode";
  ui_tooltip  = "adaptive: maxCLL will adapat to actual maxCLL over time\n"
                "          DON'T FORGET TO TURN ON THE \"adaptive_maxCLL\" technique!\n"
                "  static: tone map only according to the specified maxCLL";
  ui_type     = "combo";
  ui_items    = "static\0"
                "adaptive\0";
> = 0;

uniform float TARGET_CLL
<
  ui_category = "global";
  ui_label    = "target peak luminance (nits)";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 1000.f;

//static tone mapping
uniform float MAX_CLL
<
  ui_category = "static tone mapping";
  ui_label    = "maxCLL for static tone mapping (nits)";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 10000.f;

uniform float BT2446A_GAMUT_COMPRESSION
<
  ui_category = "BT.2446 Method A";
  ui_label    = "gamut compression";
  ui_tooltip  = "1.10 is the default of the spec\n"
                "1.05 about matches the input colour space\n"
                "1.00 slightly expands the colour space";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 2.f;
  ui_step     = 0.005f;
> = 1.1f;

uniform uint BT2390_PROCESSING_MODE
<
  ui_category = "BT.2390";
  ui_label    = "processing mode";
  ui_tooltip  = "YCbCr: process in YCbCr space\n"
                "YRGB:  process RGB according to luma\n"
                "ICtCp: process in ICtCp space\n"
                "RGB:   process each channel individually";
  ui_type     = "combo";
  ui_items    = "YCbCr\0"
                "YRGB\0"
                "ICtCp\0"
                "RGB\0";
> = 0;

//uniform float BT2390_SRC_BLACK_POINT
//<
//  ui_category = "BT.2390";
//  ui_label    = "black point (in nits)";
//  ui_tooltip  = "";
//  ui_type     = "drag";
//  ui_min      = 0.f;
//  ui_max      = 1.f;
//  ui_step     = 0.0001f;
//> = 0.f;

uniform float BT2390_TARGET_BLACK_POINT
<
  ui_category = "BT.2390";
  ui_label    = "target black point (in nits)";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.001f;
> = 0.f;

uniform float BT2390_KNEE_FACTOR
<
  ui_category = "BT.2390";
  ui_label    = "knee factor";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 3.f;
  ui_step     = 0.01f;
> = 1.5f;

uniform float BT2390_KNEE_MINUS
<
  ui_category = "BT.2390";
  ui_label    = "knee minus";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 3.f;
  ui_step     = 0.01f;
> = 0.5f;

uniform float DICE_SHOULDER_START
<
  ui_category = "Dice";
  ui_label    = "shoulder start (in %)";
  ui_tooltip  = "set this to where the luminance compression starts";
  ui_type     = "drag";
  ui_min      = 0.1f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 50.f;

uniform uint DICE_PROCESSING_MODE
<
  ui_category = "Dice";
  ui_label    = "processing mode";
  ui_tooltip  = "ICtCp: process in ICtCp space\n"
                "YCbCr: process in YCbCr space";
  ui_type     = "combo";
  ui_items    = "ICtCp\0"
                "YCbCr\0";
> = 0;

uniform uint DICE_WORKING_COLOR_SPACE
<
  ui_category = "Dice";
  ui_label    = "processing mode";
  ui_tooltip  = "AP0_D65: AP0 primaries with D65 white point";
  ui_type     = "combo";
  ui_items    = "BT.2020\0"
                "AP0_D65\0";
> = 0;

//adaptive tone mapping
uniform float MAX_CLL_CAP
<
  ui_category = "adaptive tone mapping";
  ui_label    = "maxCLL cap";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 10000.f;

uniform float TIME_TO_ADAPT
<
  ui_category = "adaptive tone mapping";
  ui_label    = "adaption to maxCLL (seconds)";
  ui_tooltip  = "time it takes to adapt to current maxCLL";
  ui_type     = "drag";
  ui_min      = 3.f;
  ui_max      = 30.f;
  ui_step     = 0.1f;
> = 2.2f;

uniform float FINAL_ADAPT_START
<
  ui_category = "adaptive tone mapping";
  ui_label    = "final adaption starting point (nits)";
  ui_tooltip  = "if the difference between maxCLL and the adaptive maxCLL is smaller than this\n"
                "use the \"final adaption steps\"";
  ui_type     = "drag";
  ui_min      = 10.f;
  ui_max      = 100.f;
  ui_step     = 1.f;
> = 50.f;

uniform float FINAL_ADAPT
<
  ui_category = "adaptive tone mapping";
  ui_label    = "final adaption steps (nits)";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 0.01f;
  ui_max      = 0.1f;
  ui_step     = 0.01f;
> = 0.05f;

uniform float FINAL_ADAPT_SPEED
<
  ui_category = "adaptive tone mapping";
  ui_label    = "final adaption speed";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_min      = 10.f;
  ui_max      = 100.f;
  ui_step     = 1.f;
> = 50.f;

#ifdef _DEBUG
uniform float DEBUG_MAX_CLL
<
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 10000.f;
  ui_step  = 10.0;
  ui_label = "debug: maxCLL"
;> = 10000.f;
#endif


texture2D Texture_Adaptive_CLL_Value0
<
  pooled = true;
>
{
  Width  = 1;
  Height = 1;

  Format = R32F;
};

sampler2D Sampler_Adaptive_CLL_Value0
{
  Texture = Texture_Adaptive_CLL_Value0;
};

storage2D Storage_Adaptive_CLL_Value0
{
  Texture = Texture_Adaptive_CLL_Value0;

  MipLevel = 0;
};

texture2D Texture_Adaptive_CLL_Value1
<
  pooled = true;
>
{
  Width  = 1;
  Height = 1;

  Format = R32F;
};

sampler2D Sampler_Adaptive_CLL_Value1
{
  Texture = Texture_Adaptive_CLL_Value1;
};

storage2D Storage_Adaptive_CLL_Value1
{
  Texture = Texture_Adaptive_CLL_Value1;
};


//static const uint numberOfAdaptiveValues = 1000;
//texture2D Texture_Adaptive_CLL_Values
//{
//   Width = numberOfAdaptiveValues;
//  Height = 2;
//
//  MipLevels = 0;
//
//  Format = R32F;
//};
//
//sampler2D Sampler_Adaptive_CLL_Values
//{
//  Texture = Texture_Adaptive_CLL_Values;
//
//  SRGBTexture = false;
//};
//
//storage2D Storage_Adaptive_CLL_Values
//{
//  Texture = Texture_Adaptive_CLL_Values;
//
//  MipLevel = 0;
//};


void ToneMapping(
      float4 VPos     : SV_Position,
      float2 TexCoord : TEXCOORD,
  out float4 Output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, TexCoord).rgb;

  //float maxCLL = tex2Dfetch(sampler_max_avg_min_CLL_values, int2(0, 0)).r;

  float maxCLL;
  switch (CLL_MODE)
  {
    case 0: {
      maxCLL = MAX_CLL;
    }
    break;
    case 1: {
      maxCLL = tex2Dfetch(Sampler_Adaptive_CLL_Value0, int2(0, 0)).r;
    }
    break;
  }

  float3 hdr = input.rgb;

//  if (maxCLL > TARGET_CLL)
//  {

#if (ACTUAL_COLOUR_SPACE == CSP_PQ)

    if ((TONE_MAPPING_METHOD != TM_METHOD_BT2390)
     || (TONE_MAPPING_METHOD == TM_METHOD_BT2390
      && (BT2390_PROCESSING_MODE != BT2390_PRO_MODE_RGB || BT2390_PROCESSING_MODE != BT2390_PRO_MODE_YCBCR)))
    {
      hdr = CSP::TRC::FromPq(hdr);
    }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    //if (TONE_MAPPING_METHOD != TM_METHOD_DICE)
    hdr /= 125.f;

    if (TONE_MAPPING_METHOD == TM_METHOD_BT2446A
     || TONE_MAPPING_METHOD == TM_METHOD_BT2446A_MOD1)
    {
      hdr = saturate(CSP::Mat::BT709To::BT2020(hdr));
    }
    else if (TONE_MAPPING_METHOD == TM_METHOD_BT2390)
    {
      hdr = clamp(CSP::Mat::BT709To::BT2020(hdr), 0.f, 65504.f);
    }
    else if (TONE_MAPPING_METHOD == TM_METHOD_DICE)
    {
      hdr = clamp(CSP::Mat::BT709To::BT2020(hdr), 0.f, 65504.f);
    }

#else

    hdr = float3(0.f, 0.f, 0.f);

#endif

    switch (TONE_MAPPING_METHOD)
    {
      case TM_METHOD_BT2446A:
      {
        hdr = BT2446A_ToneMapping(hdr,
                                  TARGET_CLL,
                                  maxCLL,
                                  BT2446A_GAMUT_COMPRESSION);
      }
      break;
      case TM_METHOD_BT2390:
      {
        //const float srcMinPQ = CSP::TRC::ToPq(0.f / 10000.f); // Lb in PQ
        //const float tgtMinPQ = CSP::TRC::ToPq(BT2390_TARGET_BLACK_POINT / 10000.f); // Lmin in PQ
        //const float tgtMaxPQ = CSP::TRC::ToPq(BT2390_TARGET_WHITE_POINT / 10000.f); // Lmax in PQ
        //const float minLum = (tgtMinPQ - srcMinPQ) / (srcMaxPQ - srcMinPQ);
        //const float maxLum = (tgtMaxPQ - srcMinPQ) / (srcMaxPQ - srcMinPQ);

        // this assumes the source black point is always 0 nits
        const float srcMaxPQ  = CSP::TRC::ToPq(maxCLL / 10000.f); // Lw in PQ
        const float tgtMinPQ  = BT2390_TARGET_BLACK_POINT == 0.f  // Lmin in PQ
                              ? 0.f
                              : CSP::TRC::ToPq(BT2390_TARGET_BLACK_POINT / 10000.f);
        const float tgtMaxPQ  = CSP::TRC::ToPq(TARGET_CLL                / 10000.f); // Lmax in PQ
        const float minLum    = tgtMinPQ / srcMaxPQ;
        const float maxLum    = tgtMaxPQ / srcMaxPQ;
        const float KneeStart = BT2390_KNEE_FACTOR * maxLum - BT2390_KNEE_MINUS;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        if (BT2390_PROCESSING_MODE == BT2390_PRO_MODE_RGB
         || BT2390_PROCESSING_MODE == BT2390_PRO_MODE_YCBCR)
        {
          hdr = CSP::TRC::ToPq(hdr);
        }

#endif

        hdr = BT2390_ToneMapping(hdr,
                                 BT2390_PROCESSING_MODE,
                                 srcMaxPQ,
                                 minLum,
                                 maxLum,
                                 KneeStart);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        if (BT2390_PROCESSING_MODE == BT2390_PRO_MODE_RGB
         || BT2390_PROCESSING_MODE == BT2390_PRO_MODE_YCBCR)
        {
          hdr = CSP::TRC::FromPq(hdr);
        }

#endif

      }
      break;
      case TM_METHOD_DICE:
      {
        const float target_CLL_normalized = TARGET_CLL / 10000.f;
        hdr = dice(
          hdr,
          CSP::ICtCp::NormalisedToIntensity::AP0_D65(target_CLL_normalized),
          CSP::ICtCp::NormalisedToIntensity::AP0_D65(DICE_SHOULDER_START / 100.f * target_CLL_normalized),
          DICE_PROCESSING_MODE,
          DICE_WORKING_COLOR_SPACE);
        //hdr = saturate(hdr);
      }
      break;

#ifdef _DEBUG

      case TM_METHOD_BT2446A_MOD1:
      {
        const float testH = clamp(TEST_H + maxCLL,     0.f, 10000.f);
        const float testS = clamp(TEST_S + TARGET_CLL, 0.f, 10000.f);
        hdr = BT2446A_ToneMapping_mod1(hdr,
                                       TARGET_CLL,
                                       maxCLL,
                                       BT2446A_GAMUT_COMPRESSION,
                                       testH,
                                       testS);
      } break;

#endif
    }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    if (TONE_MAPPING_METHOD == TM_METHOD_BT2446A
     || TONE_MAPPING_METHOD == TM_METHOD_BT2446A_MOD1
     || TONE_MAPPING_METHOD == TM_METHOD_BT2390)
    {
      hdr = CSP::Mat::BT2020To::BT709(hdr);
    }
    else if (TONE_MAPPING_METHOD == TM_METHOD_DICE)
    {
      hdr = CSP::Mat::AP0_D65To::BT709(hdr);
    }
    hdr *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_PQ)

    hdr = CSP::TRC::ToPq(hdr);

#endif
//  }

  Output = float4(hdr, 1.f);

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  float actualMaxCLL    = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(0, 0)).r;
  float adaptiveMaxCLL0 = tex2Dfetch(Sampler_Adaptive_CLL_Value0,    int2(0, 0)).r;
  float adaptiveMaxCLL1 = tex2Dfetch(Sampler_Adaptive_CLL_Value1,    int2(0, 0)).r;
  DrawTextDigit(float2(30.f * 4.f, 20.f * 40.f),        30, 1, TexCoord, 6, actualMaxCLL,    Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(30.f * 4.f, 20.f * 40.f + 30.f), 30, 1, TexCoord, 6, adaptiveMaxCLL0, Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(30.f * 4.f, 20.f * 40.f + 60.f), 30, 1, TexCoord, 6, adaptiveMaxCLL1, Output, FONT_BRIGHTNESS);
  //DrawTextDigit(float2(100.f, 590.f), 30, 1, TexCoord, 0, CLL_MODE, Output, FONT_BRIGHTNESS);

#endif

}


void AdaptiveCLL(uint3 ID : SV_DispatchThreadID)
{
  const float currentMaxCLL      = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(0, 0)).r;
  const float currentAdaptiveCLL = tex2Dfetch(Sampler_Adaptive_CLL_Value0,    int2(0, 0)).r;

  const float absFrametime = abs(FRAMETIME);

  const float curDiff = currentMaxCLL - currentAdaptiveCLL;
  float adapt = curDiff * (absFrametime / (TIME_TO_ADAPT * 1000.f));
  if (abs(curDiff) < FINAL_ADAPT_START)
  {
    const float actualFinalAdapt = absFrametime * FINAL_ADAPT * (FINAL_ADAPT_SPEED / 1000.f);
    adapt = adapt > 0.f
          ? actualFinalAdapt
          : -actualFinalAdapt;
  }
  //else
  //  adapt = adapt > 0.f ? adapt + ADAPT_OFFSET : adapt - ADAPT_OFFSET;
  const float AdaptiveCLL = currentAdaptiveCLL + adapt;

  tex2Dstore(Storage_Adaptive_CLL_Value1, int2(0, 0), AdaptiveCLL);
}

void CopyAdaptiveCLL(uint3 ID : SV_DispatchThreadID)
{
  float currentAdaptiveCLL = tex2Dfetch(Sampler_Adaptive_CLL_Value1, int2(0, 0)).r;
  currentAdaptiveCLL = currentAdaptiveCLL < MAX_CLL_CAP
                     ? currentAdaptiveCLL
                     : MAX_CLL_CAP;

  tex2Dstore(Storage_Adaptive_CLL_Value0, int2(0, 0), currentAdaptiveCLL);
}


//technique lilium__tone_mapping_adaptive_maxCLL_OLD
//<
//  enabled = false;
//>
//{
//  pass CalcCLL
//  {
//    VertexShader = PostProcessVS;
//     PixelShader = CalcCLL;
//    RenderTarget = Texture_CLL_Values;
//  }
//
//  pass GetMaxCLL0
//  {
//    ComputeShader = GetMaxCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass GetMaxCLL1
//  {
//    ComputeShader = GetMaxCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass AdaptiveCLL
//  {
//    ComputeShader = AdaptiveCLL <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass CopyAdaptiveCLL
//  {
//    ComputeShader = CopyAdaptiveCLL <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//}

technique lilium__tone_mapping_adaptive_maxCLL
<
  enabled = false;
>
{
  pass CalcCLL
  {
    VertexShader = PostProcessVS;
     PixelShader = CalcCLL;
    RenderTarget = Texture_CLL_Values;
  }

  pass GetMaxCLL0_NEW
  {
    ComputeShader = GetMaxCLL0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass GetMaxCLL1_NEW
  {
    ComputeShader = GetMaxCLL1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass GetFinalMaxCLL_NEW
  {
    ComputeShader = GetFinalMaxCLL_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass AdaptiveCLL
  {
    ComputeShader = AdaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass CopyAdaptiveCLL
  {
    ComputeShader = CopyAdaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique lilium__tone_mapping
{
  pass ToneMapping
  {
    VertexShader = PostProcessVS;
     PixelShader = ToneMapping;
  }
}
