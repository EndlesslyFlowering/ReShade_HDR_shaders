#include "ReShade.fxh"
#include "tone_mappers.fxh"
#include "DrawText_fix.fxh" //for testing! remove afterwards

//#define _DEBUG

//missing:
// - HDR10/scRGB detection -> documentation: current render format


//ideas:
// - average maxCLL over last 60? frames -> save last 100/1000 CLL values and their frametime and average over that
// - maxCLL adapation: fade to actual maxCLL over time (should read time to render per frame for this); make adjustable


uniform float FRAMETIME
<
  source = "frametime";
>;

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

uniform uint TONE_MAPPING_METHOD
<
  hidden      = false;
  ui_category = "global";
  ui_label    = "tone mapping method";
  ui_type     = "combo";
  ui_items    = "BT.2446 Method A\0BT.2446A mod1\0BT.2446A mod2\0BT.2446A mod1+mod2\0";
> = 0;

//global
uniform uint CLL_MODE
<
  ui_category = "global";
  ui_label    = "tone mapping mode";
  ui_tooltip  = "adaptive: maxCLL will adapat to actual maxCLL over time\n"
                "          DON'T FORGET TO TURN ON THE \"adaptive_maxCLL\" technique!\n"
                "  static: tone map only according to the specified maxCLL";
  ui_type     = "combo";
  ui_items    = "static\0adaptive\0";
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
  ui_tooltip  = "if the difference between maxCLL and the adaptive maxCLL is smaller than this\nuse the \"final adaption steps\"";
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
//  ui_items = "HDR10\0scRGB\0";
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

void BT2446A_tm(
      float4     vpos : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4   output : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  //float maxCLL = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(0, 0)).r;

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
    if (CSP_PQ)
      hdr = PQ_EOTF(hdr);
    else if (CSP_SCRGB)
    {
      hdr = hdr * 80.f / 10000.f;
      hdr = mul(bt709_to_bt2020_matrix, hdr);
    }
    else
      hdr = float3(0.f, 0.f, 0.f);

    switch (TONE_MAPPING_METHOD)
    {
      case 0:
        hdr = BT2446A_toneMapping(hdr, TARGET_CLL, maxCLL);
        break;
      case 1:
        const float testH = clamp(TEST_H + maxCLL, 0.f, 10000.f);
        const float testS = clamp(TEST_S + TARGET_CLL, 0.f, 10000.f);
        hdr = BT2446A_toneMapping_mod1(hdr, TARGET_CLL, maxCLL, testH, testS);
        break;
    }

    if (CSP_PQ)
      hdr = PQ_inverse_EOTF(hdr);
    else if (CSP_SCRGB)
    {
      hdr = mul(bt2020_to_bt709_matrix, hdr);
      hdr = hdr * 10000.f / 80.f;
    }
//  }

  output = float4(hdr, 1.f);

  if (SHOW_ADAPTIVE_MAXCLL)
  {
    const float bright = CSP_PQ
                       ? 0.58068888f
                       : CSP_SCRGB
                       ? 203.f / 80.f
                       : 1.f;
    float actualMaxCLL    = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(0, 0)).r;
    float adaptiveMaxCLL0 = tex2Dfetch(samplerAdaptiveCLLvalue0,  int2(0, 0)).r;
    float adaptiveMaxCLL1 = tex2Dfetch(samplerAdaptiveCLLvalue1,  int2(0, 0)).r;
    DrawText_Digit(float2(100.f, 500.f), 30, 1, texcoord, 2, actualMaxCLL    + 0.01f, output, bright);
    DrawText_Digit(float2(100.f, 530.f), 30, 1, texcoord, 2, adaptiveMaxCLL0 + 0.01f, output, bright);
    DrawText_Digit(float2(100.f, 560.f), 30, 1, texcoord, 2, adaptiveMaxCLL1 + 0.01f, output, bright);
    //DrawText_Digit(float2(100.f, 590.f), 30, 1, texcoord, 0, CLL_MODE, output, bright);
  }
}


void adaptiveCLL(uint3 id : SV_DispatchThreadID)
{
  const float currentMaxCLL      = tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(0, 0)).r;
  const float currentAdaptiveCLL = tex2Dfetch(samplerAdaptiveCLLvalue0,  int2(0, 0)).r;

  const float curDiff = currentMaxCLL - currentAdaptiveCLL;
  float adapt = curDiff * (FRAMETIME / (TIME_TO_ADAPT * 1000.f));
  if (abs(curDiff) < FINAL_ADAPT_START)
  {
    const float actualFinalAdapt = FRAMETIME * FINAL_ADAPT * (FINAL_ADAPT_SPEED / 1000.f);
    adapt = adapt > 0.f ? actualFinalAdapt : -actualFinalAdapt;
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


technique adaptive_maxCLL
<
  enabled = false;
>
{
  //pass calcCLLvalues
  //{
  //  ComputeShader = calcCLL < THREAD_SIZE0, THREAD_SIZE0>;
  //  DispatchSizeX = DISPATCH_X0;
  //  DispatchSizeY = DISPATCH_Y0;
  //}

  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLL;
    RenderTarget = CLLvalues;
  }

  pass calcMaxCLLvalue0
  {
    ComputeShader = calcMaxCLL0<THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass calcMaxCLLvalue1
  {
    ComputeShader = calcMaxCLL1<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass adaptiveCLL
  {
    ComputeShader = adaptiveCLL<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass copyAdaptiveCLL
  {
    ComputeShader = copyAdaptiveCLL<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique adaptive_maxCLL_NEW
<
  enabled = false;
>
{
  //pass calcCLLvalues
  //{
  //  ComputeShader = calcCLL < THREAD_SIZE0, THREAD_SIZE0>;
  //  DispatchSizeX = DISPATCH_X0;
  //  DispatchSizeY = DISPATCH_Y0;
  //}

  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLL;
    RenderTarget = CLLvalues;
  }

  pass calcMaxCLLvalue0_NEW
  {
    ComputeShader = calcMaxCLL0_NEW<THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass calcMaxCLLvalue1_NEW
  {
    ComputeShader = calcMaxCLL1_NEW<4, 4>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass calcFinalMaxCLLvalue1_NEW
  {
    ComputeShader = calcFinalMaxCLL_NEW<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass adaptiveCLL
  {
    ComputeShader = adaptiveCLL<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass copyAdaptiveCLL
  {
    ComputeShader = copyAdaptiveCLL<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique tone_mapping
{
  pass tone_mapping
  {
    VertexShader = PostProcessVS;
     PixelShader = BT2446A_tm;
  }
}