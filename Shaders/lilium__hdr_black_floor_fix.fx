#include "lilium__include/hdr_black_floor_fix.fxh"


#if (defined(IS_ANALYSIS_CAPABLE_API) \
  && defined(IS_HDR_CSP))


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrBlackFloorFix(
  in                  uint   VertexID   : SV_VertexID,
  out                 float4 Position   : SV_Position,
  out nointerpolation float4 FuncParms0 : FuncParms0,
  out nointerpolation float  FuncParms1 : FuncParms1)
{
  float2 texCoord;
  texCoord.x = (VertexID == 2) ? 2.f
                               : 0.f;
  texCoord.y = (VertexID == 1) ? 2.f
                               : 0.f;
  Position = float4(texCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);


  FuncParms0 = 0.f;
  FuncParms1 = 0.f;


// black flower lowering
#define rollOffStoppingPoint      FuncParms0.x
#define oldBlackPoint             FuncParms0.y
#define rollOffMinusOldBlackPoint FuncParms0.z
#define minLum                    FuncParms0.w

// gamma 2.2 emulation
#define whitePointNormalised FuncParms1

  float rollOffStoppingPoint0;
  float oldBlackPoint0;
  float newBlackPoint0;
  float rollOffMinusOldBlackPoint0;

  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode != PRO_MODE_RGB)
  {
    rollOffStoppingPoint0 = Csp::Trc::NitsTo::Pq(Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint);

    oldBlackPoint0 = Csp::Trc::NitsTo::Pq(Ui::HdrBlackFloorFix::Lowering::OldBlackPoint);

    newBlackPoint0 = sign(newBlackPoint0)
                   * Csp::Trc::NitsTo::Pq(abs(newBlackPoint0));

    newBlackPoint0 = Csp::Trc::NitsTo::Pq(Ui::HdrBlackFloorFix::Lowering::NewBlackPoint);

    rollOffMinusOldBlackPoint0 = Csp::Trc::LinearTo::Pq(rollOffStoppingPoint0) - oldBlackPoint0;
  }
  else
  {
    rollOffStoppingPoint0      = Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint / 10000.f;
    oldBlackPoint0             = Ui::HdrBlackFloorFix::Lowering::OldBlackPoint        / 10000.f;
    newBlackPoint0             = Ui::HdrBlackFloorFix::Lowering::NewBlackPoint        / 10000.f;
    rollOffMinusOldBlackPoint0 = rollOffStoppingPoint0 - oldBlackPoint0;
  }

  minLum = (newBlackPoint0 - oldBlackPoint0) / rollOffMinusOldBlackPoint0;

  rollOffStoppingPoint      = rollOffStoppingPoint0;
  oldBlackPoint             = oldBlackPoint0;
  rollOffMinusOldBlackPoint = rollOffMinusOldBlackPoint0;


  // gamma 2.2 emulation
  whitePointNormalised = Ui::HdrBlackFloorFix::Gamma22Emu::WhitePoint / 10000.f;
}


void PS_HdrBlackFloorFix(
  in                  float4 Position   : SV_Position,
  out                 float4 Output     : SV_Target0,
  in  nointerpolation float4 FuncParms0 : FuncParms0,
  in  nointerpolation float  FuncParms1 : FuncParms1)
{
  BRANCH(x)
  if (!Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu
   && !Ui::HdrBlackFloorFix::Lowering::EnableLowering)
  {
    discard;
  }

  const float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  float3 colour = inputColour.rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  colour /= 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  colour = Csp::Trc::PqTo::Linear(colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  colour = Csp::Trc::HlgTo::Linear(colour);

#else //ACTUAL_COLOUR_SPACE ==

  colour = float3(0.f, 0.f, 0.f);

#endif //ACTUAL_COLOUR_SPACE ==


  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    colour = Gamma22Emulation(colour,
                              whitePointNormalised);

    BRANCH(x)
    if (Ui::HdrBlackFloorFix::Lowering::EnableLowering)
    {
      colour = LowerBlackFloor(colour,
                               rollOffStoppingPoint,
                               oldBlackPoint,
                               rollOffMinusOldBlackPoint,
                               minLum);
    }
    else
    {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

      BRANCH(x)
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
      {
        colour = Csp::Mat::DciP3To::Bt709(colour);
      }
      else
      BRANCH(x)
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT2020)
      {
        colour = Csp::Mat::Bt2020To::Bt709(colour);
      }

      colour *= 125.f;

#elif defined(IS_HDR10_LIKE_CSP)

      BRANCH(x)
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_BT709)
      {
        colour = Csp::Mat::Bt709To::Bt2020(colour);
      }
      else
      BRANCH(x)
      if (Ui::HdrBlackFloorFix::Gamma22Emu::ProcessingColourSpace == HDR_BF_FIX_CSP_DCI_P3)
      {
        colour = Csp::Mat::DciP3To::Bt2020(colour);
      }

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)

      colour = Csp::Trc::LinearTo::Pq(colour);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

      colour = Csp::Trc::LinearTo::Hlg(colour);

#endif //ACTUAL_COLOUR_SPACE ==

#endif //IS_XXX_LIKE_CSP

    }
  }
  else
  {
    colour = LowerBlackFloor(colour,
                             rollOffStoppingPoint,
                             oldBlackPoint,
                             rollOffMinusOldBlackPoint,
                             minLum);
  }

  Output = float4(colour, inputColour.a);
}


technique lilium__hdr_black_floor_fix
<
  ui_label = "Lilium's HDR black floor fix";
>
{
  pass PS_HdrBlackFloorFix
  {
    VertexShader = VS_PrepareHdrBlackFloorFix;
     PixelShader = PS_HdrBlackFloorFix;
  }
}

#else //is hdr API and hdr colour space

ERROR_STUFF

technique lilium__hdr_black_floor_fix
<
  ui_label = "Lilium's HDR black floor fix (ERROR)";
>
VS_ERROR

#endif //is hdr API and hdr colour space
