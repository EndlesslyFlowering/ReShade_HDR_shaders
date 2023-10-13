#include "lilium__include/tone_mappers.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


#include "lilium__include/draw_text_fix.fxh"

#if 0
#include "lilium__include/HDR_black_floor_fix.fxh"
#endif


//#define BT2446A_MOD1_ENABLE


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
  namespace Tm
  {
    namespace Global
    {
      uniform uint TmMethod
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
      #ifdef BT2446A_MOD1_ENABLE
                      "BT.2446A mod1\0"
      #endif
                      ;
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
        ui_step     = 0.000001f;
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

      uniform float KneeOffset
      <
        ui_category = "BT.2390 EETF";
        ui_label    = "knee offset";
        ui_tooltip  = "This adjusts where the brightness compression curve starts."
                 "\n" "The higher the value the earlier the ealier the compression starts."
                 "\n" "0.5 is the spec default.";
        ui_type     = "drag";
        ui_min      = 0.5f;
        ui_max      = 1.0f;
        ui_step     = 0.005f;
      > = 0.5f;
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
        ui_min      = 10.f;
        ui_max      = 90.f;
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

//      uniform uint WorkingColourSpace
//      <
//        ui_category = "Dice";
//        ui_label    = "processing colour space";
//        ui_tooltip  = "AP0_D65: AP0 primaries with D65 white point"
//                 "\n" "AP0_D65 technically covers every humanly perceivable colour."
//                 "\n" "It's meant for future use if we ever move past the BT.2020 colour space."
//                 "\n" "Just use BT.2020 for now.";
//        ui_type     = "combo";
//        ui_items    = "BT.2020\0"
//                      "AP0_D65\0";
//      > = 0;
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

#ifdef BT2446A_MOD1_ENABLE
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
//texture2D TextureAdaptiveCllValues
//{
//   Width = numberOfAdaptiveValues;
//  Height = 2;
//
//  MipLevels = 0;
//
//  Format = R32F;
//};
//
//sampler2D SamplerAdaptiveCllValues
//{
//  Texture = TextureAdaptiveCllValues;
//
//  SRGBTexture = false;
//};
//
//storage2D StorageAdaptiveCllValues
//{
//  Texture = TextureAdaptiveCllValues;
//
//  MipLevel = 0;
//};


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareToneMapping(
  in  uint   Id       : SV_VertexID,
  out float4 VPos     : SV_Position,
  out float2 TexCoord : TEXCOORD0,
  out float4 TmParms0 : TmParms0,
  out float3 TmParms1 : TmParms1)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#define usedMaxCll TmParms0.x

  if (Ui::Tm::Global::Mode == TM_MODE_STATIC)
  {
    usedMaxCll = Ui::Tm::StaticMode::MaxCll;
  }
  else
  {
    usedMaxCll = tex2Dfetch(SamplerConsolidated, COORDS_ADAPTIVE_CLL);
  }

  usedMaxCll = usedMaxCll > Ui::Tm::Global::TargetBrightness
             ? usedMaxCll
             : Ui::Tm::Global::TargetBrightness;

  if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390)
  {

#define bt2390SrcMinPq              TmParms0.y
#define bt2390SrcMaxPq              TmParms0.z
#define bt2390SrcMaxPqMinusSrcMinPq TmParms0.w
#define bt2390MinLum                TmParms1.x
#define bt2390MaxLum                TmParms1.y
#define bt2390KneeStart             TmParms1.z

    // source min brightness (Lb) in PQ
    bt2390SrcMinPq = Csp::Trc::ToPqFromNits(Ui::Tm::Bt2390::OldBlackPoint);
    // source max brightness (Lw) in PQ
    bt2390SrcMaxPq = Csp::Trc::ToPqFromNits(usedMaxCll);

    // target min brightness (Lmin) in PQ
    float tgtMinPQ = Csp::Trc::ToPqFromNits(Ui::Tm::Bt2390::NewBlackPoint);
    // target max brightness (Lmax) in PQ
    float tgtMaxPQ = Csp::Trc::ToPqFromNits(Ui::Tm::Global::TargetBrightness);

    // this is needed often so precalculate it
    bt2390SrcMaxPqMinusSrcMinPq = bt2390SrcMaxPq - bt2390SrcMinPq;

    bt2390MinLum = (tgtMinPQ - bt2390SrcMinPq) / bt2390SrcMaxPqMinusSrcMinPq;
    bt2390MaxLum = (tgtMaxPQ - bt2390SrcMinPq) / bt2390SrcMaxPqMinusSrcMinPq;

    // knee start (KS)
    bt2390KneeStart = 1.5f
                    * bt2390MaxLum
                    - Ui::Tm::Bt2390::KneeOffset;

  }
  else if (Ui::Tm::Global::TmMethod == TM_METHOD_DICE)
  {

#define diceTargetCllInPq     TmParms0.y
#define diceShoulderStartInPq TmParms0.z
#define diceUnused0           TmParms0.w
#define diceUnused1           TmParms1 //.xyz

    diceTargetCllInPq = Csp::Trc::ToPqFromNits(Ui::Tm::Global::TargetBrightness);
    diceShoulderStartInPq =
      Csp::Trc::ToPqFromNits(Ui::Tm::Dice::ShoulderStart
                           / 100.f
                           * Ui::Tm::Global::TargetBrightness);

    diceUnused0 = 0.f;
    diceUnused1 = float3(0.f, 0.f, 0.f);
  }

}


void PS_ToneMapping(
  in  float4 VPos     : SV_Position,
  in  float2 TexCoord : TEXCOORD0,
  in  float4 TmParms0 : TmParms0,
  in  float3 TmParms1 : TmParms1,
  out float4 Output   : SV_Target0)
{
  float3 hdr = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  if ((Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
    && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
    && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR)

   || (Ui::Tm::Global::TmMethod == TM_METHOD_DICE
    && Ui::Tm::Dice::ProcessingModeDice != DICE_PRO_MODE_YCBCR)

   || Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A)
  {
    hdr = Csp::Trc::FromPq(hdr);
  }

//  if (Ui::Tm::Global::TmMethod == TM_METHOD_DICE)
//  {
//    if (Ui::Tm::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65
//     || Ui::Tm::Dice::ProcessingModeDice != DICE_PRO_MODE_YCBCR)
//    {
//      hdr = Csp::Trc::FromPq(hdr);
//    }
//
//    if (Ui::Tm::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
//    {
//      hdr = max(Csp::Mat::Bt2020To::Ap0D65(hdr), 0.f);
//
//      if (Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
//      {
//        hdr = Csp::Trc::ToPq(hdr);
//      }
//    }
//  }
//  else if(Ui::Tm::Global::TmMethod != TM_METHOD_BT2390
//       || (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
//        && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
//        && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR))
//  {
//    hdr = Csp::Trc::FromPq(hdr);
//  }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  hdr /= 125.f;

  hdr = max(Csp::Mat::Bt709To::Bt2020(hdr), 0.f);

  if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
   && (Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
    || Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR)

   || (Ui::Tm::Global::TmMethod == TM_METHOD_DICE
    && Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR))
  {
    hdr = Csp::Trc::ToPq(hdr);
  }

//  if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A
//   || Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A_MOD1)
//  {
//    hdr = max(Csp::Mat::Bt709To::Bt2020(hdr), 0.f);
//  }
//  else if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390)
//  {
//    hdr = max(Csp::Mat::Bt709To::Bt2020(hdr), 0.f);
//
//    if (Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
//     || Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR)
//    {
//      hdr = Csp::Trc::ToPq(hdr);
//    }
//  }
//  else if (Ui::Tm::Global::TmMethod == TM_METHOD_DICE)
//  {
//    if (Ui::Tm::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_BT2020)
//    {
//      hdr = max(Csp::Mat::Bt709To::Bt2020(hdr), 0.f);
//    }
//    else
//    {
//      hdr = max(Csp::Mat::Bt709To::Ap0D65(hdr), 0.f);
//    }
//
//    if (Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
//    {
//      hdr = Csp::Trc::ToPq(hdr);
//    }
//  }

#else

  hdr = float3(0.f, 0.f, 0.f);

#endif

  switch (Ui::Tm::Global::TmMethod)
  {
    case TM_METHOD_BT2446A:
    {
      Tmos::Bt2446A(hdr,
                    usedMaxCll,
                    Ui::Tm::Global::TargetBrightness,
                    Ui::Tm::Bt2446A::GamutCompression);
    }
    break;
    case TM_METHOD_BT2390:
    {
      Tmos::Bt2390::Eetf(hdr,
                         Ui::Tm::Bt2390::ProcessingModeBt2390,
                         bt2390SrcMinPq,
                         bt2390SrcMaxPq,
                         bt2390SrcMaxPqMinusSrcMinPq,
                         bt2390MinLum,
                         bt2390MaxLum,
                         bt2390KneeStart);
    }
    break;
    case TM_METHOD_DICE:
    {
      Tmos::Dice::ToneMapper(hdr,
                             Ui::Tm::Dice::ProcessingModeDice,
                             diceTargetCllInPq,
                             diceShoulderStartInPq);
    }
    break;

#ifdef BT2446A_MOD1_ENABLE

    case TM_METHOD_BT2446A_MOD1:
    {
      //move test parameters to vertex shader if this ever gets released
      float testH = clamp(TEST_H + usedMaxCll,                                0.f, 10000.f);
      float testS = clamp(TEST_S + Ui::Tm::Global::TargetBrightness, 0.f, 10000.f);
      Tmos::Bt2446A_MOD1(hdr,
                         usedMaxCll,
                         Ui::Tm::Global::TargetBrightness,
                         Ui::Tm::Bt2446A::GamutCompression,
                         testH,
                         testS);
    } break;

#endif
  }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  if ((Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
    && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
    && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR)

   || (Ui::Tm::Global::TmMethod == TM_METHOD_DICE
    && Ui::Tm::Dice::ProcessingModeDice != DICE_PRO_MODE_YCBCR)

   || Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A)
  {
    hdr = Csp::Trc::ToPq(hdr);
  }

//  if (Ui::Tm::Global::TmMethod == TM_METHOD_DICE)
//  {
//    if (Ui::Tm::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_AP0_D65)
//    {
//      if (Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
//      {
//        hdr = Csp::Trc::FromPq(hdr);
//        hdr = Csp::Mat::Ap0D65To::Bt2020(hdr);
//      }
//      else
//      {
//        hdr = Csp::Mat::Ap0D65To::Bt2020(hdr);
//      }
//      hdr = Csp::Trc::ToPq(hdr);
//    }
//    else if (Ui::Tm::Dice::ProcessingModeDice != DICE_PRO_MODE_YCBCR)
//    {
//      hdr = Csp::Trc::ToPq(hdr);
//    }
//  }
//  else if(Ui::Tm::Global::TmMethod != TM_METHOD_BT2390
//       || (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
//        && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_RGB
//        && Ui::Tm::Bt2390::ProcessingModeBt2390 != BT2390_PRO_MODE_YCBCR))
//  {
//    hdr = Csp::Trc::ToPq(hdr);
//  }

#elif (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
   && (Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
    || Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR)

   || (Ui::Tm::Global::TmMethod == TM_METHOD_DICE
    && Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR))
  {
    hdr = Csp::Trc::FromPq(hdr);
  }

  hdr = Csp::Mat::Bt2020To::Bt709(hdr);

  hdr *= 125.f;

//  if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A
//   || Ui::Tm::Global::TmMethod == TM_METHOD_BT2446A_MOD1
//   || Ui::Tm::Global::TmMethod == TM_METHOD_BT2390)
//  {
//    if (Ui::Tm::Global::TmMethod == TM_METHOD_BT2390
//     && (Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_RGB
//      || Ui::Tm::Bt2390::ProcessingModeBt2390 == BT2390_PRO_MODE_YCBCR))
//    {
//      hdr = Csp::Trc::FromPq(hdr);
//    }
//    hdr = Csp::Mat::Bt2020To::Bt709(hdr);
//  }
//  else if (Ui::Tm::Global::TmMethod == TM_METHOD_DICE)
//  {
//    if (Ui::Tm::Dice::ProcessingModeDice == DICE_PRO_MODE_YCBCR)
//    {
//      hdr = Csp::Trc::FromPq(hdr);
//    }
//
//    if (Ui::Tm::Dice::WorkingColourSpace == DICE_WORKING_COLOUR_SPACE_BT2020)
//    {
//      hdr = Csp::Mat::Bt2020To::Bt709(hdr);
//    }
//    else
//    {
//      hdr = Csp::Mat::Ap0D65To::Bt709(hdr);
//    }
//  }
//
//  hdr *= 125.f;

#endif

  Output = float4(hdr, 1.f);

#define FINAL_ADAPT_STOP 0.9979f

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  uint text_maxCLL[26] = { __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  uint text_avgdCLL[35] = { __a, __v, __e, __r, __a, __g, __e, __d, __Space,
                                  __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  uint text_adaptiveMaxCLL[35] = { __a, __d, __a, __p, __t, __i, __v, __e, __Space,
                                         __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
  uint text_finalAdaptMode[17] = { __f, __i, __n, __a, __l, __Space, __a, __d, __a, __p, __t, __Space, __m, __o, __d, __e, __Colon };

  float actualMaxCll = tex2Dfetch(SamplerConsolidated, COORDS_MAXCLL_VALUE);

  float avgMaxCllInPq = tex2Dfetch(SamplerConsolidated, COORDS_AVERAGED_MAXCLL);
  float avgMaxCll     = Csp::Trc::FromPqToNits(avgMaxCllInPq);

  float adaptiveMaxCll     = tex2Dfetch(SamplerConsolidated, COORDS_ADAPTIVE_CLL);
  float adaptiveMaxCllInPQ = Csp::Trc::ToPqFromNits(adaptiveMaxCll);

  float absDiff = abs(avgMaxCllInPq - adaptiveMaxCllInPQ);

  float finalAdaptMode = 
    absDiff < abs((avgMaxCllInPq - FINAL_ADAPT_STOP * avgMaxCllInPq))
  ? 2.f
  : absDiff < abs((avgMaxCllInPq - Ui::Tm::AdaptiveMode::FinalAdaptStart / 100.f * avgMaxCllInPq))
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


void CS_AdaptiveCLL(uint3 ID : SV_DispatchThreadID)
{
  float currentMaxCllinPqAveraged = (tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL0)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL1)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL2)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL3)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL4)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL5)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL6)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL7)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL8)
                                   + tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL9)) / 10.f;

#if (SHOW_ADAPTIVE_MAXCLL == YES)

  tex2Dstore(StorageConsolidated, COORDS_AVERAGED_MAXCLL, currentMaxCllinPqAveraged);

#endif

  float currentMaxCllInPQ =
    Csp::Trc::ToPqFromNits(tex2Dfetch(StorageConsolidated, COORDS_MAXCLL_VALUE));
  float currentAdaptiveMaxCllInPQ =
    Csp::Trc::ToPqFromNits(tex2Dfetch(StorageConsolidated, COORDS_ADAPTIVE_CLL));

  int curSlot = tex2Dfetch(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CUR);
  int newSlot = curSlot > 10 ? 1
                                   : curSlot + 1;
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CUR, newSlot);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CUR
                                 + int2(newSlot, 0), currentMaxCllInPQ);

  float absFrametime = abs(RuntimeValues::Frametime);

  float curDiff = currentMaxCllinPqAveraged * 1.0005f - currentAdaptiveMaxCllInPQ;
  float absCurDiff = abs(curDiff);
  float finalAdaptPointInPq = abs(currentMaxCllinPqAveraged
                            - Ui::Tm::AdaptiveMode::FinalAdaptStart / 100.f * currentMaxCllinPqAveraged);
  float stopAdaptPointInPq  = abs(currentMaxCllinPqAveraged
                            - FINAL_ADAPT_STOP * currentMaxCllinPqAveraged);
  float adapt = 0.f;

  if (absCurDiff < stopAdaptPointInPq)
  {
    adapt = 0.000015f;
  }
  else if (absCurDiff < finalAdaptPointInPq)
  {
    float actualFinalAdapt = absFrametime
                           * (Ui::Tm::AdaptiveMode::FinalAdaptSteps / 10000.f)
                           * (Ui::Tm::AdaptiveMode::FinalAdaptSpeed / 1000.f);
    adapt = curDiff > 0.f ?  actualFinalAdapt
                          : -actualFinalAdapt;
  }
  else
  {
    adapt = curDiff * (absFrametime / (Ui::Tm::AdaptiveMode::TimeToAdapt * 1000.f));
  }
  //else
  //  adapt = adapt > 0.f ? adapt + ADAPT_OFFSET : adapt - ADAPT_OFFSET;

  barrier();

  tex2Dstore(StorageConsolidated,
             COORDS_ADAPTIVE_CLL,
             min(Csp::Trc::FromPqToNits(currentAdaptiveMaxCllInPQ + adapt),
                 Ui::Tm::AdaptiveMode::MaxCllCap));

}


void CS_ResetAveragedMaxCll(uint3 ID : SV_DispatchThreadID)
{
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL0, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL1, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL2, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL3, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL4, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL5, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL6, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL7, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL8, 10000.f);
  tex2Dstore(StorageConsolidated, COORDS_AVERAGE_MAXCLL_CLL9, 10000.f);
}


//technique lilium__tone_mapping_adaptive_maxCLL_OLD
//<
//  enabled = false;
//>
//{
//  pass PS_CalcCllPerPixel
//  {
//    VertexShader = VS_PostProcess;
//     PixelShader = PS_CalcCllPerPixel;
//    RenderTarget = TextureCllValues;
//  }
//
//  pass CS_GetMaxCll0
//  {
//    ComputeShader = CS_GetMaxCll0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_GetMaxCll1
//  {
//    ComputeShader = CS_GetMaxCll1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass CS_AdaptiveCLL
//  {
//    ComputeShader = CS_AdaptiveCLL <1, 1>;
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
  pass CS_ResetAveragedMaxCll
  {
    ComputeShader = CS_ResetAveragedMaxCll <1, 1>;
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
  pass PS_CalcCllPerPixel
  {
    VertexShader = VS_PostProcess;
     PixelShader = PS_CalcCllPerPixel;
    RenderTarget = TextureCllValues;
  }

  pass CS_GetMaxCll0_NEW
  {
    ComputeShader = CS_GetMaxCll0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass CS_GetMaxCll1_NEW
  {
    ComputeShader = CS_GetMaxCll1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass CS_GetFinalMaxCll_NEW
  {
    ComputeShader = CS_GetFinalMaxCll_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass CS_AdaptiveCLL
  {
    ComputeShader = CS_AdaptiveCLL <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique lilium__tone_mapping
<
  ui_label = "Lilium's tone mapping";
>
{
  pass PS_ToneMapping
  {
    VertexShader = VS_PrepareToneMapping;
     PixelShader = PS_ToneMapping;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__tone_mapping
<
  ui_label = "Lilium's tone mapping (ERROR)";
>
CS_ERROR

#endif //is hdr API and hdr colour space
