#include "lilium__include/tone_mappers.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


#include "lilium__include/draw_text_fix.fxh"

#if 0
#include "lilium__include/HDR_black_floor_fix.fxh"
#endif


//#define _DEBUG

// TODO
// - add black point adaption for every tone mapper

//ideas:
// - average maxCLL over last 60? frames -> save last 100/1000 CLL values and their frametime and average over that
// - maxCLL adapation: fade to actual maxCLL over time (should read time to render per frame for this); make adjustable

#undef CIE_DIAGRAM
#undef IGNORE_NEAR_BLACK_VALUES_FOR_CSP_DETECTION

#ifndef SHOW_ADAPTIVE_MAXCLL
  #define SHOW_ADAPTIVE_MAXCLL NO
#endif


namespace RuntimeValues
{
  uniform float Frametime
  <
    source = "frametime";
  >;
}

namespace Ui
{
  namespace ToneMapping
  {
    namespace Global
    {
      uniform uint Method
      <
        ui_category = "global";
        ui_label    = "tone mapping method";
        ui_tooltip  = "BT.2390 EETF:"
                 "\n" "  Is a highlight compressor."
                 "\n" "  It only compresses the highlights according to \"knee start\"."
                 "\n" "  Which dictate where the highlight compression starts."
                 "\n" "BT.2446 Method A:"
                 "\n" "  Compared to other tone mappers this one tries to compress the whole brightness range"
                 "\n" "  the image has rather than just compressing the highlights."
                 "\n" "  It is designed for tone mapping 1000 nits to 100 nits originally"
                 "\n" "  and therefore can't handle compressing brightness differences that are too high."
                 "\n" "  Like 10000 nits to 800 nits. As it's trying to compress the whole brightness range."
                 "\n" "Dice:"
                 "\n" "  Works similarly to BT.2390."
                 "\n" "  The compression curve is slightly different and it's a bit faster.";
        ui_type     = "combo";
        ui_items    = "BT.2390 EETF\0"
                      "BT.2446 Method A\0"
                      "Dice\0"
      #ifdef _DEBUG
                      "BT.2446A mod1\0"
      #endif
                      ;
      //                "BT.2446A mod2\0"
      //                "BT.2446A mod1+mod2\0";
      > = 0;

#define TM_METHOD_BT2390       0
#define TM_METHOD_BT2446A      1
#define TM_METHOD_DICE         2
#define TM_METHOD_BT2446A_MOD1 3

      uniform uint Mode
      <
        ui_category = "global";
        ui_label    = "tone mapping mode";
        ui_tooltip  = "  static: tone map only according to the specified maximum brightness"
                 "\n" "adaptive: the maximum brightness will adapat to the actual maximum brightness over time"
                 "\n" "          DON'T FORGET TO TURN ON THE \"adaptive mode\" TECHNIQUE!";
        ui_type     = "combo";
        ui_items    = "static\0"
                      "adaptive\0";
      > = 0;

#define TM_MODE_STATIC   0
#define TM_MODE_ADAPTIVE 1

      uniform float TargetBrightness
      <
        ui_category = "global";
        ui_label    = "target brightness";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 0.f;
        ui_max      = 10000.f;
        ui_step     = 5.f;
      > = 1000.f;
    } //Global

    namespace StaticMode
    {
      uniform float MaxCll
      <
        ui_category = "static tone mapping";
        ui_label    = "maximum tone mapping brightness";
        ui_tooltip  = "Everything above this will be clipped!";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 0.f;
        ui_max      = 10000.f;
        ui_step     = 5.f;
      > = 10000.f;
    } //StaticMode

    namespace Bt2446A
    {
      uniform float GamutCompression
      <
        ui_category = "BT.2446 Method A";
        ui_label    = "gamut compression";
        ui_tooltip  = "1.10 is the default of the BT.2446 specification"
                 "\n" "1.05 about matches the input colour space"
                 "\n" "1.00 slightly expands the colour space";
        ui_type     = "drag";
        ui_min      = 1.f;
        ui_max      = 2.f;
        ui_step     = 0.005f;
      > = 1.1f;
    } //Bt2446A

    namespace Bt2390
    {
      uniform uint ProcessingModeBt2390
      <
        ui_category = "BT.2390 EETF";
        ui_label    = "processing mode";
        ui_tooltip  = "ICtCp: process in ICtCp space (best quality)"
                 "\n" "YCbCr: process in YCbCr space"
                 "\n" "YRGB:  process RGB according to brightness"
                 "\n" "RGB:   process each channel individually";
        ui_type     = "combo";
        ui_items    = "ICtCp\0"
                      "YCbCr\0"
                      "YRGB\0"
                      "RGB\0";
      > = 0;

      uniform float OldBlackPoint
      <
        ui_category = "BT.2390 EETF";
        ui_label    = "old black point";
        ui_type     = "slider";
        ui_units    = " nits";
        ui_min      = 0.f;
        ui_max      = 0.5f;
        ui_step     = 0.0000001f;
      > = 0.f;

      uniform float NewBlackPoint
      <
        ui_category = "BT.2390 EETF";
        ui_label    = "new black point";
        ui_type     = "slider";
        ui_units    = " nits";
        ui_min      = -0.1f;
        ui_max      = 0.1f;
        ui_step     = 0.0001f;
      > = 0.f;

      uniform float KneeStart
      <
        ui_category = "BT.2390 EETF";
        ui_label    = "knee start";
        ui_tooltip  = "Set this to where the brightness compression curve starts."
                 "\n" "In % of the target brightness."
                 "\n" "example:"
                 "\n" "With \"target brightness\" set to \"1000 nits\" and \"shoulder start\" to \"50%\"."
                 "\n" "The brightness compression will start at 500 nits.";
        ui_type     = "drag";
        ui_units    = "%%";
        ui_min      = 0.1f;
        ui_max      = 100.f;
        ui_step     = 0.1f;
      > = 50.f;
    } //Bt2390

    namespace Dice
    {
      uniform float ShoulderStart
      <
        ui_category = "Dice";
        ui_label    = "shoulder start";
        ui_tooltip  = "Set this to where the brightness compression curve starts."
                 "\n" "In % of the target brightness."
                 "\n" "example:"
                 "\n" "With \"target brightness\" set to \"1000 nits\" and \"shoulder start\" to \"50%\"."
                 "\n" "The brightness compression will start at 500 nits.";
        ui_type     = "drag";
        ui_units    = "%%";
        ui_min      = 0.1f;
        ui_max      = 100.f;
        ui_step     = 0.1f;
      > = 50.f;

      uniform uint ProcessingModeDice
      <
        ui_category = "Dice";
        ui_label    = "processing mode";
        ui_tooltip  = "ICtCp: process in ICtCp space (best quality)"
                 "\n" "YCbCr: process in YCbCr space";
        ui_type     = "combo";
        ui_items    = "ICtCp\0"
                      "YCbCr\0";
      > = 0;

      uniform uint WorkingColourSpace
      <
        ui_category = "Dice";
        ui_label    = "processing colour space";
        ui_tooltip  = "AP0_D65: AP0 primaries with D65 white point"
                 "\n" "AP0_D65 technically covers every humanly perceivable colour."
                 "\n" "It's meant for future use if we ever move past the BT.2020 colour space."
                 "\n" "Just use BT.2020 for now.";
        ui_type     = "combo";
        ui_items    = "BT.2020\0"
                      "AP0_D65\0";
      > = 0;
    } //Dice

    namespace AdaptiveMode
    {
      uniform float MaxCllCap
      <
        ui_category = "adaptive tone mapping";
        ui_label    = "cap maximum brightness";
        ui_tooltip  = "Caps maximum brightness that the adaptive maximum brightness can reach."
                 "\n" "Everything above this value will be clipped!";
        ui_type     = "drag";
        ui_units    = " nits";
        ui_min      = 0.f;
        ui_max      = 10000.f;
        ui_step     = 10.f;
      > = 10000.f;

      uniform float TimeToAdapt
      <
        ui_category = "adaptive tone mapping";
        ui_label    = "adaption to maximum brightness";
        ui_tooltip  = "Time it takes to adapt to the current maximum brightness";
        ui_type     = "drag";
        ui_min      = 0.5f;
        ui_max      = 3.f;
        ui_step     = 0.1f;
      > = 1.0f;

      uniform float FinalAdaptStart
      <
        ui_category = "adaptive tone mapping";
        ui_label    = "final adaption starting point";
        ui_tooltip  = "If the difference between the \"maximum brightness\""
                 "\n" "and the \"adaptive maximum brightness\" is smaller than this percentage"
                 "\n" "use the \"final adaption steps\"."
                 "\n" "For flickery games use 90% or lower.";
        ui_type     = "drag";
        ui_units    = "%%";
        ui_min      = 80.f;
        ui_max      = 99.f;
        ui_step     = 0.1f;
      > = 95.f;

      uniform float FinalAdaptSteps
      <
        ui_category = "adaptive tone mapping";
        ui_label    = "final adaption steps";
        ui_tooltip  = "For flickery games use 7.00 or lower.";
        ui_type     = "drag";
        ui_min      = 1.f;
        ui_max      = 10.f;
        ui_step     = 0.05f;
      > = 7.5f;

      uniform float FinalAdaptSpeed
      <
        ui_category = "adaptive tone mapping";
        ui_label    = "final adaption speed";
        ui_tooltip  = "For flickery games use 5.00 or lower.";
        ui_type     = "drag";
        ui_min      = 1.f;
        ui_max      = 10.f;
        ui_step     = 0.05f;
      > = 7.5f;
    } //AdaptiveMode
  }
}


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


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareToneMapping(
  in  uint   Id         : SV_VertexID,
  out float4 VPos       : SV_Position,
  out float2 TexCoord   : TEXCOORD0,
  out float  TmParms[7] : TmParms)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#define usedMaxCll TmParms[0]

  if (Ui::ToneMapping::Global::Mode == TM_MODE_STATIC)
  {
    usedMaxCll = Ui::ToneMapping::StaticMode::MaxCll;
  }
  else
  {
    usedMaxCll = tex2Dfetch(Sampler_Consolidated, COORDS_ADAPTIVE_CLL).r;
  }

  usedMaxCll = usedMaxCll > Ui::ToneMapping::Global::TargetBrightness
             ? usedMaxCll
             : Ui::ToneMapping::Global::TargetBrightness;

  if (Ui::ToneMapping::Global::Method == TM_METHOD_BT2390)
  {

#define bt2390SrcMinPq              TmParms[1]
#define bt2390SrcMaxPq              TmParms[2]
#define bt2390SrcMaxPqMinusSrcMinPq TmParms[3]
#define bt2390MinLum                TmParms[4]
#define bt2390MaxLum                TmParms[5]
#define bt2390KneeStart             TmParms[6]

    // source min brightness (Lb) in PQ
    bt2390SrcMinPq = Csp::Trc::ToPqFromNits(Ui::ToneMapping::Bt2390::OldBlackPoint);
    // source max brightness (Lw) in PQ
    bt2390SrcMaxPq = Csp::Trc::ToPqFromNits(usedMaxCll);

    // target min brightness (Lmin) in PQ
    float tgtMinPQ = Csp::Trc::ToPqFromNits(Ui::ToneMapping::Bt2390::NewBlackPoint);
    // target max brightness (Lmin) in PQ
    float tgtMaxPQ = Csp::Trc::ToPqFromNits(Ui::ToneMapping::Global::TargetBrightness);

    // this is needed often so precalculate it
    bt2390SrcMaxPqMinusSrcMinPq = bt2390SrcMaxPq - bt2390SrcMinPq;

    bt2390MinLum = (tgtMinPQ - bt2390SrcMinPq) / bt2390SrcMaxPqMinusSrcMinPq;
    bt2390MaxLum = (tgtMaxPQ - bt2390SrcMinPq) / bt2390SrcMaxPqMinusSrcMinPq;

    // knee start (KS)
    bt2390KneeStart =
      Csp::Trc::ToPqFromNits(Ui::ToneMapping::Bt2390::KneeStart / 100.f
                           * Ui::ToneMapping::Global::TargetBrightness);

  }
  else if (Ui::ToneMapping::Global::Method == TM_METHOD_DICE)
  {

#define diceTargetCllInPq     TmParms[1]
#define diceShoulderStartInPq TmParms[2]

    diceTargetCllInPq = Csp::Trc::ToPqFromNits(Ui::ToneMapping::Global::TargetBrightness);
    diceShoulderStartInPq =
      Csp::Trc::ToPqFromNits(Ui::ToneMapping::Dice::ShoulderStart / 100.f
                           * Ui::ToneMapping::Global::TargetBrightness);
  }

}


void PS_ToneMapping(
  in  float4 VPos       : SV_Position,
  in  float2 TexCoord   : TEXCOORD0,
  in  float  TmParms[7] : TmParms,
  out float4 Output     : SV_Target0)
{
  float3 hdr = tex2D(ReShade::BackBuffer, TexCoord).rgb;

//  if (usedMaxCll > Ui::ToneMapping::Global::TargetBrightness)
//  {

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    if (Ui::ToneMapping::Global::Method == TM_METHOD_DICE)
    {
      if (Ui::ToneMapping::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
       || Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_ICTCP)
      {
        hdr = Csp::Trc::FromPq(hdr);
      }

      if (Ui::ToneMapping::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
      {
        hdr = clamp(Csp::Mat::Bt2020To::Ap0D65(hdr), 0.f, 65504.f);

        if (Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
        {
          hdr = Csp::Trc::ToPq(hdr);
        }
      }
    }
    else if(Ui::ToneMapping::Global::Method != TM_METHOD_BT2390
         || (Ui::ToneMapping::Global::Method == TM_METHOD_BT2390
          && Ui::ToneMapping::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
          && Ui::ToneMapping::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR))
    {
      hdr = Csp::Trc::FromPq(hdr);
    }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    //if (Ui::ToneMapping::Global::Method != TM_METHOD_DICE)
    hdr /= 125.f;

    if (Ui::ToneMapping::Global::Method == TM_METHOD_BT2446A
     || Ui::ToneMapping::Global::Method == TM_METHOD_BT2446A_MOD1)
    {
      hdr = saturate(Csp::Mat::Bt709To::Bt2020(hdr));
    }
    else if (Ui::ToneMapping::Global::Method == TM_METHOD_BT2390)
    {
      hdr = clamp(Csp::Mat::Bt709To::Bt2020(hdr), 0.f, 65504.f);

      if (Ui::ToneMapping::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
       || Ui::ToneMapping::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR)
      {
        hdr = Csp::Trc::ToPq(hdr);
      }
    }
    else if (Ui::ToneMapping::Global::Method == TM_METHOD_DICE)
    {
      if (Ui::ToneMapping::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_BT2020)
      {
        hdr = clamp(Csp::Mat::Bt709To::Bt2020(hdr), 0.f, 65504.f);
      }
      else
      {
        hdr = clamp(Csp::Mat::Bt709To::Ap0D65(hdr), 0.f, 65504.f);
      }

      if (Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
      {
        hdr = Csp::Trc::ToPq(hdr);
      }
    }

#else

    hdr = float3(0.f, 0.f, 0.f);

#endif

    switch (Ui::ToneMapping::Global::Method)
    {
      case TM_METHOD_BT2446A:
      {
        hdr = ToneMappers::Bt2446A(hdr,
                                   Ui::ToneMapping::Global::TargetBrightness,
                                   usedMaxCll,
                                   Ui::ToneMapping::Bt2446A::GamutCompression);
      }
      break;
      case TM_METHOD_BT2390:
      {

        hdr = ToneMappers::Bt2390::Eetf(hdr,
                                        Ui::ToneMapping::Bt2390::ProcessingModeBt2390,
                                        bt2390SrcMinPq,  // source min brightness (Lb) in PQ
                                        bt2390SrcMaxPq,  // source max brightness (Lw) in PQ
                                        bt2390SrcMaxPqMinusSrcMinPq,
                                        bt2390MinLum,
                                        bt2390MaxLum,
                                        bt2390KneeStart);
      }
      break;
      case TM_METHOD_DICE:
      {
        hdr = ToneMappers::Dice::ToneMapper(hdr,
                                            diceTargetCllInPq,
                                            diceShoulderStartInPq,
                                            Ui::ToneMapping::Dice::ProcessingModeDice,
                                            Ui::ToneMapping::Dice::WorkingColourSpace);
        //hdr = saturate(hdr);
      }
      break;

#ifdef _DEBUG

      case TM_METHOD_BT2446A_MOD1:
      {
        const float testH = clamp(TEST_H + usedMaxCll,                                0.f, 10000.f);
        const float testS = clamp(TEST_S + Ui::ToneMapping::Global::TargetBrightness, 0.f, 10000.f);
        hdr = ToneMappers::Bt2446A_MOD1(hdr,
                                        Ui::ToneMapping::Global::TargetBrightness,
                                        usedMaxCll,
                                        Ui::ToneMapping::Bt2446A::GamutCompression,
                                        testH,
                                        testS);
      } break;

#endif
    }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

    if (Ui::ToneMapping::Global::Method == TM_METHOD_DICE)
    {
      if (Ui::ToneMapping::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
      {
        if (Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
        {
          hdr = Csp::Trc::FromPq(hdr);
          hdr = Csp::Mat::Ap0D65To::Bt2020(hdr);
        }
        else
        {
          hdr = Csp::Mat::Ap0D65To::Bt2020(hdr);
        }
        hdr = Csp::Trc::ToPq(hdr);
      }
      else if (Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_ICTCP)
      {
        hdr = Csp::Trc::ToPq(hdr);
      }
    }
    else if(Ui::ToneMapping::Global::Method != TM_METHOD_BT2390
         || (Ui::ToneMapping::Global::Method == TM_METHOD_BT2390
          && Ui::ToneMapping::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
          && Ui::ToneMapping::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR))
    {
      hdr = Csp::Trc::ToPq(hdr);
    }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

    if (Ui::ToneMapping::Global::Method == TM_METHOD_BT2446A
     || Ui::ToneMapping::Global::Method == TM_METHOD_BT2446A_MOD1
     || Ui::ToneMapping::Global::Method == TM_METHOD_BT2390)
    {
      hdr = Csp::Mat::Bt2020To::Bt709(hdr);

      if (Ui::ToneMapping::Global::Method == TM_METHOD_BT2390
       && (Ui::ToneMapping::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
        || Ui::ToneMapping::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR))
      {
        hdr = Csp::Trc::FromPq(hdr);
      }
    }
    else if (Ui::ToneMapping::Global::Method == TM_METHOD_DICE)
    {
      if (Ui::ToneMapping::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
      {
        hdr = Csp::Trc::FromPq(hdr);
      }

      if (Ui::ToneMapping::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_BT2020)
      {
        hdr = Csp::Mat::Bt2020To::Bt709(hdr);
      }
      else
      {
        hdr = Csp::Mat::Ap0D65To::Bt709(hdr);
      }
    }
    hdr *= 125.f;

#endif
//  }

  Output = float4(hdr, 1.f);

#define FINAL_ADAPT_STOP 0.9979f

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  const uint text_maxCLL[26] = { __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  const uint text_avgdCLL[35] = { __a, __v, __e, __r, __a, __g, __e, __d, __Space,
                                  __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  const uint text_adaptiveMaxCLL[35] = { __a, __d, __a, __p, __t, __i, __v, __e, __Space,
                                         __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  const uint text_finalAdaptMode[17] = { __f, __i, __n, __a, __l, __Space, __a, __d, __a, __p, __t, __Space, __m, __o, __d, __e, __Colon };

  const float actualMaxCll = tex2Dfetch(Sampler_Consolidated, COORDS_MAXCLL_VALUE).r;

  const float avgMaxCllInPq = tex2Dfetch(Sampler_Consolidated, COORDS_AVERAGED_MAXCLL).r;
  const float avgMaxCll     = Csp::Trc::FromPqToNits(avgMaxCllInPq);

  const float adaptiveMaxCll     = tex2Dfetch(Sampler_Consolidated, COORDS_ADAPTIVE_CLL).r;
  const float adaptiveMaxCllInPQ = Csp::Trc::ToPqFromNits(adaptiveMaxCll);

  const float absDiff = abs(avgMaxCllInPq - adaptiveMaxCllInPQ);

  const float finalAdaptMode = 
    absDiff < abs((avgMaxCllInPq - FINAL_ADAPT_STOP * avgMaxCllInPq))
  ? 2.f
  : absDiff < abs((avgMaxCllInPq - Ui::ToneMapping::AdaptiveMode::FinalAdaptStart / 100.f * avgMaxCllInPq))
  ? 1.f
  : 0.f;

  DrawTextString(float2(10.f * 15.f, 20.f * 40.f),        30, 1, TexCoord, text_maxCLL,         26, Output, FONT_BRIGHTNESS);
  DrawTextString(float2(       15.f, 20.f * 40.f + 30.f), 30, 1, TexCoord, text_avgdCLL,        35, Output, FONT_BRIGHTNESS);
  DrawTextString(float2(       15.f, 20.f * 40.f + 60.f), 30, 1, TexCoord, text_adaptiveMaxCLL, 35, Output, FONT_BRIGHTNESS);
  DrawTextString(float2(        0.f, 20.f * 40.f + 90.f), 30, 1, TexCoord, text_finalAdaptMode, 17, Output, FONT_BRIGHTNESS);

  DrawTextDigit(float2(24.f * 15.f, 20.f * 40.f),        30, 1, TexCoord,  6, actualMaxCll,   Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(24.f * 15.f, 20.f * 40.f + 30.f), 30, 1, TexCoord,  6, avgMaxCll,      Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(24.f * 15.f, 20.f * 40.f + 60.f), 30, 1, TexCoord,  6, adaptiveMaxCll, Output, FONT_BRIGHTNESS);
  DrawTextDigit(float2(19.f * 15.f, 20.f * 40.f + 90.f), 30, 1, TexCoord, -1, finalAdaptMode, Output, FONT_BRIGHTNESS);

#endif

}


void AdaptiveCLL(uint3 ID : SV_DispatchThreadID)
{
  const float currentMaxCllinPqAveraged = (tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL0).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL1).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL2).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL3).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL4).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL5).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL6).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL7).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL8).r
                                         + tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL9).r) / 10.f;

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  tex2Dstore(Storage_Consolidated, COORDS_AVERAGED_MAXCLL, currentMaxCllinPqAveraged);

#endif

  const float currentMaxCllInPQ =
    Csp::Trc::ToPqFromNits(tex2Dfetch(Storage_Consolidated, COORDS_MAXCLL_VALUE).r);
  const float currentAdaptiveMaxCllInPQ =
    Csp::Trc::ToPqFromNits(tex2Dfetch(Storage_Consolidated, COORDS_ADAPTIVE_CLL).r);

  const int curSlot = tex2Dfetch(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CUR).r;
  const int newSlot = curSlot > 10 ? 1
                                   : curSlot + 1;
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CUR, newSlot);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CUR
                                 + int2(newSlot, 0), currentMaxCllInPQ);

  const float absFrametime = abs(RuntimeValues::Frametime);

  const float curDiff = currentMaxCllinPqAveraged * 1.0005f - currentAdaptiveMaxCllInPQ;
  const float absCurDiff = abs(curDiff);
  const float finalAdaptPointInPq = abs(currentMaxCllinPqAveraged
                                      - Ui::ToneMapping::AdaptiveMode::FinalAdaptStart / 100.f * currentMaxCllinPqAveraged);
  const float stopAdaptPointInPq  = abs(currentMaxCllinPqAveraged
                                      - FINAL_ADAPT_STOP * currentMaxCllinPqAveraged);
  float adapt = 0.f;

  if (absCurDiff < stopAdaptPointInPq)
  {
    adapt = 0.000015f;
  }
  else if (absCurDiff < finalAdaptPointInPq)
  {
    const float actualFinalAdapt = absFrametime
                                 * (Ui::ToneMapping::AdaptiveMode::FinalAdaptSteps / 10000.f)
                                 * (Ui::ToneMapping::AdaptiveMode::FinalAdaptSpeed / 1000.f);
    adapt = curDiff > 0.f ?  actualFinalAdapt
                          : -actualFinalAdapt;
  }
  else
  {
    adapt = curDiff * (absFrametime / (Ui::ToneMapping::AdaptiveMode::TimeToAdapt * 1000.f));
  }
  //else
  //  adapt = adapt > 0.f ? adapt + ADAPT_OFFSET : adapt - ADAPT_OFFSET;

  barrier();

  tex2Dstore(Storage_Consolidated,
             COORDS_ADAPTIVE_CLL,
             min(Csp::Trc::FromPqToNits(currentAdaptiveMaxCllInPQ + adapt),
                 Ui::ToneMapping::AdaptiveMode::MaxCllCap));

}


void ResetAveragedMaxCll(uint3 ID : SV_DispatchThreadID)
{
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL0, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL1, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL2, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL3, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL4, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL5, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL6, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL7, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL8, 10000.f);
  tex2Dstore(Storage_Consolidated, COORDS_AVERAGE_MAXCLL_CLL9, 10000.f);
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

technique lilium__reset_averaged_max_cll_values
<
  enabled = true;
  hidden  = true;
  timeout = 1;
>
{
  pass ResetAveragedMaxCll
  {
    ComputeShader = ResetAveragedMaxCll <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

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
    VertexShader = VS_PrepareToneMapping;
     PixelShader = PS_ToneMapping;
  }
}

#else

ERROR_STUFF

technique lilium__tone_mapping
<
  ui_label = "Lilium's tone mapping (ERROR)";
>
{
  pass CS_Error
  {
    ComputeShader = CS_Error<1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

#endif
