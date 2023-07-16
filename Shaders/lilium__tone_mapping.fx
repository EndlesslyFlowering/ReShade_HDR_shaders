#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "lilium__include\tone_mappers.fxh"
#include "lilium__include\draw_text_fix.fxh"

#if 0
#include "lilium__include\HDR_black_floor_fix.fxh"
#endif


//#define _DEBUG


//ideas:
// - average maxCLL over last 60? frames -> save last 100/1000 CLL values and their frametime and average over that
// - maxCLL adapation: fade to actual maxCLL over time (should read time to render per frame for this); make adjustable

#undef CIE_DIAGRAM
#undef IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION

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

uniform uint TM_METHOD
<
  ui_category = "global";
  ui_label    = "tone mapping method";
  ui_type     = "combo";
  ui_items    = "BT.2446 Method A\0"
                "BT.2390 EETF\0"
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
uniform uint TM_MODE
<
  ui_category = "global";
  ui_label    = "tone mapping mode";
  ui_tooltip  = "  static: tone map only according to the specified maximum brightness\n"
                "adaptive: the maximum brightness will adapat to the actual maximum brightness over time\n"
                "          DON'T FORGET TO TURN ON THE \"adaptive mode\" TECHNIQUE!";
  ui_type     = "combo";
  ui_items    = "static\0"
                "adaptive\0";
> = 0;

uniform float TARGET_BRIGHTNESS
<
  ui_category = "global";
  ui_label    = "target brightness (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 5.f;
> = 1000.f;

//static tone mapping
uniform float MAX_CLL
<
  ui_category = "static tone mapping";
  ui_label    = "maximum brightness that will be tone mapped (in nits)";
  ui_tooltip  = "everything above this will be clipped!";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 5.f;
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
  ui_category = "BT.2390 EETF";
  ui_label    = "processing mode";
  ui_tooltip  = "ICtCp: process in ICtCp space (best quality)\n"
                "YCbCr: process in YCbCr space\n"
                "YRGB:  process RGB according to brightness\n"
                "RGB:   process each channel individually";
  ui_type     = "combo";
  ui_items    = "ICtCp\0"
                "YCbCr\0"
                "YRGB\0"
                "RGB\0";
> = 0;

uniform float BT2390_SRC_BLACK_POINT
<
  ui_category = "BT.2390 EETF";
  ui_label    = "source black point (in nits)";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.001f;
> = 0.f;

uniform float BT2390_TARGET_BLACK_POINT
<
  ui_category = "BT.2390 EETF";
  ui_label    = "target black point (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 1.f;
  ui_step     = 0.001f;
> = 0.f;

uniform float BT2390_KNEE_FACTOR
<
  ui_category = "BT.2390 EETF";
  ui_label    = "knee factor";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 3.f;
  ui_step     = 0.01f;
> = 1.5f;

uniform float BT2390_KNEE_MINUS
<
  ui_category = "BT.2390 EETF";
  ui_label    = "knee minus";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 3.f;
  ui_step     = 0.01f;
> = 0.5f;

uniform float DICE_SHOULDER_START
<
  ui_category = "Dice";
  ui_label    = "shoulder start (in %)";
  ui_tooltip  = "Set this to where the brightness compression starts.\n"
                "In % of the maximum brightness.\n"
                "example:\n"
                "With \"maximum brightness\" set to \"1000 nits\" and \"shoulder start\" to \"50%\".\n"
                "The brightness compression will start at 500 nits.";
  ui_type     = "drag";
  ui_units    = "%%";
  ui_min      = 0.1f;
  ui_max      = 100.f;
  ui_step     = 0.1f;
> = 50.f;

uniform uint DICE_PROCESSING_MODE
<
  ui_category = "Dice";
  ui_label    = "processing mode";
  ui_tooltip  = "ICtCp: process in ICtCp space (best quality)\n"
                "YCbCr: process in YCbCr space";
  ui_type     = "combo";
  ui_items    = "ICtCp\0"
                "YCbCr\0";
> = 0;

uniform uint DICE_WORKING_COLOUR_SPACE
<
  ui_category = "Dice";
  ui_label    = "processing colour space";
  ui_tooltip  = "AP0_D65: AP0 primaries with D65 white point";
  ui_type     = "combo";
  ui_items    = "BT.2020\0"
                "AP0_D65\0";
> = 0;

//adaptive tone mapping
uniform float MAX_CLL_CAP
<
  ui_category = "adaptive tone mapping";
  ui_label    = "cap maximum brightness (in nits)";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 10.f;
> = 10000.f;

uniform float TIME_TO_ADAPT
<
  ui_category = "adaptive tone mapping";
  ui_label    = "adaption to maximum brightness (in seconds)";
  ui_tooltip  = "time it takes to adapt to the current maximum brightness";
  ui_type     = "drag";
  ui_units    = " seconds";
  ui_min      = 3.f;
  ui_max      = 30.f;
  ui_step     = 0.1f;
> = 2.2f;

uniform float FINAL_ADAPT_START
<
  ui_category = "adaptive tone mapping";
  ui_label    = "final adaption starting point (in nits)";
  ui_tooltip  = "If the difference between actual \"maximum brightness\"\n"
                "and the \"adaptive maximum brightness\" is smaller than this\n"
                "use the \"final adaption steps\".";
  ui_type     = "drag";
  ui_units    = " nits";
  ui_min      = 10.f;
  ui_max      = 100.f;
  ui_step     = 1.f;
> = 50.f;

uniform float FINAL_ADAPT
<
  ui_category = "adaptive tone mapping";
  ui_label    = "final adaption steps (in nits)";
  ui_tooltip  = "";
  ui_type     = "drag";
  ui_units    = " nits";
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
  switch (TM_MODE)
  {
    case 0: {
      maxCLL = MAX_CLL;
    }
    break;
    case 1: {
      maxCLL = tex2Dfetch(Sampler_Consolidated, COORDS_ADAPTIVE_CLL).r;
    }
    break;
  }

  float3 hdr = input.rgb;

//  if (maxCLL > TARGET_BRIGHTNESS)
//  {

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    if ((TM_METHOD != TM_METHOD_BT2390)
     || (TM_METHOD == TM_METHOD_BT2390
      && (BT2390_PROCESSING_MODE != BT2390_PRO_MODE_RGB || BT2390_PROCESSING_MODE != BT2390_PRO_MODE_YCBCR)))
    {
      hdr = Csp::Trc::FromPq(hdr);
    }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    //if (TM_METHOD != TM_METHOD_DICE)
    hdr /= 125.f;

    if (TM_METHOD == TM_METHOD_BT2446A
     || TM_METHOD == TM_METHOD_BT2446A_MOD1)
    {
      hdr = saturate(Csp::Mat::Bt709To::Bt2020(hdr));
    }
    else if (TM_METHOD == TM_METHOD_BT2390)
    {
      hdr = clamp(Csp::Mat::Bt709To::Bt2020(hdr), 0.f, 65504.f);
    }
    else if (TM_METHOD == TM_METHOD_DICE)
    {
      hdr = clamp(Csp::Mat::Bt709To::Bt2020(hdr), 0.f, 65504.f);
    }

#else

    hdr = float3(0.f, 0.f, 0.f);

#endif

    switch (TM_METHOD)
    {
      case TM_METHOD_BT2446A:
      {
        hdr = ToneMapping::Bt2446a(hdr,
                                   TARGET_BRIGHTNESS,
                                   maxCLL,
                                   BT2446A_GAMUT_COMPRESSION);
      }
      break;
      case TM_METHOD_BT2390:
      {
        //const float minLum = (tgtMinPQ - srcMinPQ) / (srcMaxPQ - srcMinPQ);
        //const float maxLum = (tgtMaxPQ - srcMinPQ) / (srcMaxPQ - srcMinPQ);

        // this assumes the source black point is always 0 nits
        const float srcMinPQ  = Csp::Trc::ToPq(BT2390_SRC_BLACK_POINT / 10000.f); // Lb in PQ
        const float srcMaxPQ  = Csp::Trc::ToPq(maxCLL / 10000.f); // Lw in PQ
        const float tgtMinPQ  = BT2390_TARGET_BLACK_POINT == 0.f  // Lmin in PQ
                              ? 0.f
                              : Csp::Trc::ToPq(BT2390_TARGET_BLACK_POINT / 10000.f);
        const float tgtMaxPQ  = Csp::Trc::ToPq(TARGET_BRIGHTNESS         / 10000.f); // Lmax in PQ
        const float minLum    = tgtMinPQ / srcMaxPQ;
        const float maxLum    = tgtMaxPQ / srcMaxPQ;
        const float kneeStart = BT2390_KNEE_FACTOR * maxLum - BT2390_KNEE_MINUS;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        if (BT2390_PROCESSING_MODE == BT2390_PRO_MODE_RGB
         || BT2390_PROCESSING_MODE == BT2390_PRO_MODE_YCBCR)
        {
          hdr = Csp::Trc::ToPq(hdr);
        }

#endif

        hdr = ToneMapping::Bt2390::Eetf(hdr,
                                        BT2390_PROCESSING_MODE,
                                        srcMinPQ,
                                        srcMaxPQ,
                                        minLum,
                                        maxLum,
                                        kneeStart);

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

        if (BT2390_PROCESSING_MODE == BT2390_PRO_MODE_RGB
         || BT2390_PROCESSING_MODE == BT2390_PRO_MODE_YCBCR)
        {
          hdr = Csp::Trc::FromPq(hdr);
        }

#endif

      }
      break;
      case TM_METHOD_DICE:
      {
        const float targetCllnormalised = TARGET_BRIGHTNESS / 10000.f;
        hdr = ToneMapping::Dice::ToneMapper(
          hdr,
          Csp::Ictcp::NormalisedToIntensity::Ap0D65(targetCllnormalised),
          Csp::Ictcp::NormalisedToIntensity::Ap0D65(DICE_SHOULDER_START / 100.f * targetCllnormalised),
          DICE_PROCESSING_MODE,
          DICE_WORKING_COLOUR_SPACE);
        //hdr = saturate(hdr);
      }
      break;

#ifdef _DEBUG

      case TM_METHOD_BT2446A_MOD1:
      {
        const float testH = clamp(TEST_H + maxCLL,            0.f, 10000.f);
        const float testS = clamp(TEST_S + TARGET_BRIGHTNESS, 0.f, 10000.f);
        hdr = ToneMapping::Bt2446a_MOD1(hdr,
                                        TARGET_BRIGHTNESS,
                                        maxCLL,
                                        BT2446A_GAMUT_COMPRESSION,
                                        testH,
                                        testS);
      } break;

#endif
    }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    if (TM_METHOD == TM_METHOD_BT2446A
     || TM_METHOD == TM_METHOD_BT2446A_MOD1
     || TM_METHOD == TM_METHOD_BT2390)
    {
      hdr = Csp::Mat::Bt2020To::Bt709(hdr);
    }
    else if (TM_METHOD == TM_METHOD_DICE)
    {
      hdr = Csp::Mat::Ap0D65To::Bt709(hdr);
    }
    hdr *= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    hdr = Csp::Trc::ToPq(hdr);

#endif
//  }

  Output = float4(hdr, 1.f);

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  float actualMaxCLL   = tex2Dfetch(Sampler_Max_Avg_Min_CLL_Values, int2(0, 0)).r;
  float adaptiveMaxCLL = tex2Dfetch(Sampler_Adaptive_CLL_Value,     int2(0, 0)).r;
  DrawTextDigit(float2(30.f * 4.f, 20.f * 40.f),        30, 1, TexCoord, 6, actualMaxCLL,   Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(30.f * 4.f, 20.f * 40.f + 30.f), 30, 1, TexCoord, 6, adaptiveMaxCLL, Output, FONT_BRIGHTNESS);
  //DrawTextDigit(float2(100.f, 590.f), 30, 1, TexCoord, 0, TM_MODE, Output, FONT_BRIGHTNESS);

#endif

}


void AdaptiveCLL(uint3 ID : SV_DispatchThreadID)
{
  const float currentMaxCLL      = tex2Dfetch(Sampler_Consolidated, COORDS_MAXCLL_VALUE).r;
  const float currentAdaptiveCLL = tex2Dfetch(Sampler_Consolidated, COORDS_ADAPTIVE_CLL).r;

  const float absFrametime = abs(FRAMETIME);

  const float curDiff = currentMaxCLL - currentAdaptiveCLL;
  float adapt = curDiff * (absFrametime / (TIME_TO_ADAPT * 1000.f));
  if (abs(curDiff) < FINAL_ADAPT_START)
  {
    const float actualFinalAdapt = absFrametime * FINAL_ADAPT * (FINAL_ADAPT_SPEED / 1000.f);
    adapt = adapt > 0.f ?  actualFinalAdapt
                        : -actualFinalAdapt;
  }
  //else
  //  adapt = adapt > 0.f ? adapt + ADAPT_OFFSET : adapt - ADAPT_OFFSET;
  const float AdaptiveCLL = currentAdaptiveCLL + adapt;

  barrier();

  tex2Dstore(Storage_Consolidated, COORDS_ADAPTIVE_CLL, AdaptiveCLL);
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
//}

technique lilium__tone_mapping_adaptive_maximum_brightness
<
  ui_label   = "Lilium's tone mapping adaptive mode";
  ui_tooltip = "enable this to use the adaptive tone mapping mode";
  enabled    = false;
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
}

technique lilium__tone_mapping
<
  ui_label = "Lilium's tone mapping";
>
{
  pass ToneMapping
  {
    VertexShader = PostProcessVS;
     PixelShader = ToneMapping;
  }
}

#else

uniform int GLOBAL_INFO
<
  ui_category = "info";
  ui_label    = " ";
  ui_type     = "radio";
  ui_text     = "Only DirectX 11, 12 and Vulkan are supported!";
>;

#endif
