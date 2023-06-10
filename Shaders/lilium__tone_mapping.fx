#include "ReShade.fxh"
#include "lilium__tone_mappers.fxh"
#include "lilium__DrawText_fix.fxh" //for testing! remove afterwards

//#define _DEBUG


//ideas:
// - average maxCLL over last 60? frames -> save last 100/1000 CLL values and their frametime and average over that
// - maxCLL adapation: fade to actual maxCLL over time (should read time to render per frame for this); make adjustable


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
                "1.05 about matches the input color space\n"
                "1.00 slightly expands the color space";
  ui_type     = "drag";
  ui_min      = 1.f;
  ui_max      = 2.f;
  ui_step     = 0.005f;
> = 1.1f;

uniform uint BT2390_PROCESSING_MODE
<
  ui_category = "BT.2390";
  ui_label    = "processing mode";
  ui_tooltip  = "RGB:   process each channel individually\n"
                "YRGB:  process RGB according to luma\n"
                "YCbCr: process in YCbCr space\n"
                "ICtCp: process in ICtCp space";
  ui_type     = "combo";
  ui_items    = "RGB\0"
                "YRGB\0"
                "YCbCr\0"
                "ICtCp\0";
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
  ui_items    = "AP0_D65\0"
                "BT.2020\0";
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
> = 10.f;

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

uniform bool SHOW_ADAPTIVE_MAXCLL
<
  ui_category = "adaptive tone mapping";
  ui_label    = "show maxCLL and adaptive maxCLL";
> = false;

//uniform uint COLORSPACE
//<
//  ui_type  = "combo";
//  ui_items = "HDR10\0"
//             "scRGB\0";
//  ui_label = "colorspace";
//> = 0;

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


texture2D adaptiveCLLvalue0
{
   Width = 1;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerAdaptiveCLLvalue0
{
  Texture = adaptiveCLLvalue0;

  SRGBTexture = false;
};

storage2D storageTargetAdaptiveCLLvalue0
{
  Texture = adaptiveCLLvalue0;

  MipLevel = 0;
};

texture2D adaptiveCLLvalue1
{
  Width = 1;
  Height = 1;

  MipLevels = 0;

  Format = R32F;
};

sampler2D samplerAdaptiveCLLvalue1
{
  Texture = adaptiveCLLvalue1;

  SRGBTexture = false;
};

storage2D storageTargetAdaptiveCLLvalue1
{
  Texture = adaptiveCLLvalue1;

  MipLevel = 0;
};


//static const uint numberOfAdaptiveValues = 1000;
//texture2D adaptiveCLLvalues
//{
//   Width = numberOfAdaptiveValues;
//  Height = 2;
//
//  MipLevels = 0;
//
//  Format = R32F;
//};
//
//sampler2D samplerAdaptiveCLLvalues
//{
//  Texture = adaptiveCLLvalues;
//
//  SRGBTexture = false;
//};
//
//storage2D storageTargetAdaptiveCLLvalues
//{
//  Texture = adaptiveCLLvalues;
//
//  MipLevel = 0;
//};

//#if BUFFER_COLOR_SPACE == CSP_PQ || CSP_OVERRIDE == CSP_PQ
//  #define FONT_BRIGHTNESS 0.58068888104160783796
//#elif BUFFER_COLOR_SPACE == CSP_SCRGB || CSP_OVERRIDE == CSP_SCRGB
//  #define FONT_BRIGHTNESS 203.f / 80.f
//#elif BUFFER_COLOR_SPACE == CSP_UNKNOWN && CSP_OVERRIDE == CSP_PS5
//  #define FONT_BRIGHTNESS 2.03f
//#else
//  #define FONT_BRIGHTNESS 1.f
//#endif

void BT2446A_tm(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  //float maxCLL = tex2Dfetch(sampler_max_avg_min_CLL_values, int2(0, 0)).r;

  float maxCLL;
  switch (CLL_MODE)
  {
    case 0:
      maxCLL = MAX_CLL;
      break;
    case 1:
      maxCLL = tex2Dfetch(samplerAdaptiveCLLvalue0, int2(0, 0)).r;
      break;
  }

  float3 hdr = input.rgb;

//  if (maxCLL > TARGET_CLL)
//  {
    if ((BUFFER_COLOR_SPACE == CSP_PQ && TONE_MAPPING_METHOD != TM_METHOD_BT2390)
     || (BUFFER_COLOR_SPACE == CSP_PQ && TONE_MAPPING_METHOD == TM_METHOD_BT2390
      && (BT2390_PROCESSING_MODE != PRO_MODE_RGB || BT2390_PROCESSING_MODE != PRO_MODE_YCBCR)))
    {
      hdr = PQ_EOTF(hdr);
    }
    else if (BUFFER_COLOR_SPACE == CSP_SCRGB || CSP_OVERRIDE == CSP_SCRGB)
    {
      //if (TONE_MAPPING_METHOD != TM_METHOD_DICE)
      hdr = hdr * 0.0080f;

      if (TONE_MAPPING_METHOD == TM_METHOD_BT2446A
       || TONE_MAPPING_METHOD == TM_METHOD_BT2446A_MOD1)
      {
        hdr = saturate(mul(BT709_to_BT2020, hdr));
      }
      else if (TONE_MAPPING_METHOD == TM_METHOD_BT2390)
      {
        hdr = clamp(mul(BT709_to_BT2020, hdr), 0.f, 65504.f);
      }
      else if (TONE_MAPPING_METHOD == TM_METHOD_DICE)
      {
        hdr = clamp(mul(BT709_to_AP0_D65, hdr), 0.f, 65504.f);
      }
    }
    else
      hdr = float3(0.f, 0.f, 0.f);

    switch (TONE_MAPPING_METHOD)
    {
      case TM_METHOD_BT2446A:
        hdr = BT2446A_tone_mapping(hdr, TARGET_CLL, maxCLL, BT2446A_GAMUT_COMPRESSION);
        break;
      case TM_METHOD_BT2390:
        //const float src_min_PQ = PQ_inverse_EOTF(0.f / 10000.f); // Lb in PQ
        //const float tgt_min_PQ = PQ_inverse_EOTF(BT2390_TARGET_BLACK_POINT / 10000.f); // Lmin in PQ
        //const float tgt_max_PQ = PQ_inverse_EOTF(BT2390_TARGET_WHITE_POINT / 10000.f); // Lmax in PQ
        //const float min_lum = (tgt_min_PQ - src_min_PQ) / (src_max_PQ - src_min_PQ);
        //const float max_lum = (tgt_max_PQ - src_min_PQ) / (src_max_PQ - src_min_PQ);

        // this assumes the source black point is always 0 nits
        const float src_max_PQ = PQ_inverse_EOTF(maxCLL / 10000.f); // Lw in PQ
        const float tgt_min_PQ = BT2390_TARGET_BLACK_POINT == 0.f   // Lmin in PQ
                               ? 0.f
                               : PQ_inverse_EOTF(BT2390_TARGET_BLACK_POINT / 10000.f);
        const float tgt_max_PQ = PQ_inverse_EOTF(TARGET_CLL                / 10000.f); // Lmax in PQ
        const float min_lum    = tgt_min_PQ / src_max_PQ;
        const float max_lum    = tgt_max_PQ / src_max_PQ;
        const float knee_start = BT2390_KNEE_FACTOR * max_lum - BT2390_KNEE_MINUS;

        if (BUFFER_COLOR_SPACE == CSP_SCRGB && (BT2390_PROCESSING_MODE == PRO_MODE_RGB
                                             || BT2390_PROCESSING_MODE == PRO_MODE_YCBCR))
        {
          hdr = PQ_inverse_EOTF(hdr);
        }
        hdr = BT2390_tone_mapping(hdr,
                                  BT2390_PROCESSING_MODE,
                                  src_max_PQ,
                                  min_lum,
                                  max_lum,
                                  knee_start);
        if (BUFFER_COLOR_SPACE == CSP_SCRGB && (BT2390_PROCESSING_MODE == PRO_MODE_RGB
                                             || BT2390_PROCESSING_MODE == PRO_MODE_YCBCR))
        {
          hdr = PQ_EOTF(hdr);
        }
        break;
      case TM_METHOD_DICE:
      {
        const float target_CLL_normalized = TARGET_CLL / 10000.f;
        hdr = dice(hdr,
                   normalised_to_I(target_CLL_normalized),
                   normalised_to_I(DICE_SHOULDER_START / 100.f * target_CLL_normalized),
                   DICE_PROCESSING_MODE,
                   DICE_WORKING_COLOR_SPACE);
        //hdr = saturate(hdr);
      }
      break;
#ifdef _DEBUG
      case TM_METHOD_BT2446A_MOD1:
        const float testH = clamp(TEST_H + maxCLL,     0.f, 10000.f);
        const float testS = clamp(TEST_S + TARGET_CLL, 0.f, 10000.f);
        hdr = BT2446A_tone_mapping_mod1(hdr, TARGET_CLL, maxCLL, BT2446A_GAMUT_COMPRESSION, testH, testS);
        break;
#endif
    }
    if (BUFFER_COLOR_SPACE == CSP_SCRGB)
    {
      if (TONE_MAPPING_METHOD == TM_METHOD_BT2446A
       || TONE_MAPPING_METHOD == TM_METHOD_BT2446A_MOD1
       || TONE_MAPPING_METHOD == TM_METHOD_BT2390)
      {
        hdr = mul(BT2020_to_BT709, hdr);
      }
      else if (TONE_MAPPING_METHOD == TM_METHOD_DICE)
      {
        hdr = mul(AP0_D65_to_BT709, hdr);
      }
    }

    if (BUFFER_COLOR_SPACE == CSP_PQ)
      hdr = PQ_inverse_EOTF(hdr);
    else if (BUFFER_COLOR_SPACE  == CSP_SCRGB)// && TONE_MAPPING_METHOD != TM_METHOD_DICE)
    {
      hdr = hdr * 125.f;
    }
//  }

  output = float4(hdr, 1.f);

  if (SHOW_ADAPTIVE_MAXCLL)
  {
    float actualMaxCLL    = tex2Dfetch(sampler_max_avg_min_CLL_values, int2(0, 0)).r;
    float adaptiveMaxCLL0 = tex2Dfetch(samplerAdaptiveCLLvalue0,  int2(0, 0)).r;
    float adaptiveMaxCLL1 = tex2Dfetch(samplerAdaptiveCLLvalue1,  int2(0, 0)).r;
    DrawText_Digit(float2(100.f, 500.f), 30, 1, texcoord, 2, actualMaxCLL    + 0.01f, output, FONT_BRIGHTNESS);
    DrawText_Digit(float2(100.f, 530.f), 30, 1, texcoord, 2, adaptiveMaxCLL0 + 0.01f, output, FONT_BRIGHTNESS);
    DrawText_Digit(float2(100.f, 560.f), 30, 1, texcoord, 2, adaptiveMaxCLL1 + 0.01f, output, FONT_BRIGHTNESS);
    //DrawText_Digit(float2(100.f, 590.f), 30, 1, texcoord, 0, CLL_MODE, output, FONT_BRIGHTNESS);
  }
}


void adaptiveCLL(uint3 id : SV_DispatchThreadID)
{
  const float currentMaxCLL      = tex2Dfetch(sampler_max_avg_min_CLL_values, int2(0, 0)).r;
  const float currentAdaptiveCLL = tex2Dfetch(samplerAdaptiveCLLvalue0,  int2(0, 0)).r;

  const float abs_frametime = abs(FRAMETIME);

  const float curDiff = currentMaxCLL - currentAdaptiveCLL;
  float adapt = curDiff * (abs_frametime / (TIME_TO_ADAPT * 1000.f));
  if (abs(curDiff) < FINAL_ADAPT_START)
  {
    const float actualFinalAdapt = abs_frametime * FINAL_ADAPT * (FINAL_ADAPT_SPEED / 1000.f);
    adapt = adapt > 0.f
          ? actualFinalAdapt
          : -actualFinalAdapt;
  }
  //else
  //  adapt = adapt > 0.f ? adapt + ADAPT_OFFSET : adapt - ADAPT_OFFSET;
  const float adaptiveCLL = currentAdaptiveCLL + adapt;

  tex2Dstore(storageTargetAdaptiveCLLvalue1, int2(0, 0), adaptiveCLL);
}

void copyAdaptiveCLL(uint3 id : SV_DispatchThreadID)
{
  float currentAdaptiveCLL = tex2Dfetch(samplerAdaptiveCLLvalue1, int2(0, 0)).r;
  currentAdaptiveCLL = currentAdaptiveCLL < MAX_CLL_CAP
                     ? currentAdaptiveCLL
                     : MAX_CLL_CAP;

  tex2Dstore(storageTargetAdaptiveCLLvalue0, int2(0, 0), currentAdaptiveCLL);
}


technique lilium__tone_mapping_adaptive_maxCLL_OLD
<
  enabled = false;
>
{
  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLL;
    RenderTarget = CLL_values;
  }

  pass getMaxCLLvalue0
  {
    ComputeShader = getMaxCLL0 <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass getMaxCLLvalue1
  {
    ComputeShader = getMaxCLL1 <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass adaptiveCLL
  {
    ComputeShader = adaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass copyAdaptiveCLL
  {
    ComputeShader = copyAdaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique lilium__tone_mapping_adaptive_maxCLL
<
  enabled = false;
>
{
  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLL;
    RenderTarget = CLL_values;
  }

  pass getMaxCLLvalue0_NEW
  {
    ComputeShader = getMaxCLL0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass getMaxCLLvalue1_NEW
  {
    ComputeShader = getMaxCLL1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass getFinalMaxCLLvalue1_NEW
  {
    ComputeShader = getFinalMaxCLL_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass adaptiveCLL
  {
    ComputeShader = adaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass copyAdaptiveCLL
  {
    ComputeShader = copyAdaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique lilium__tone_mapping
{
  pass tone_mapping
  {
    VertexShader = PostProcessVS;
     PixelShader = BT2446A_tm;
  }
}
