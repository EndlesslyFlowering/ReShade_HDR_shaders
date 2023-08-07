#include "lilium__include/hdr_black_floor_fix.fxh"


#if (((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
   || __RENDERER__ >= 0x20000)                           \
  && defined(IS_POSSIBLE_HDR_CSP))


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrBlackFloorFix(
  in  uint   Id           : SV_VertexID,
  out float4 VPos         : SV_Position,
  out float2 TexCoord     : TEXCOORD0,
  out float  FuncParms[5] : FuncParms)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

// black flower lowering
#define rollOffStoppingPoint      FuncParms[0]
#define oldBlackPoint             FuncParms[1]
#define rollOffMinusOldBlackPoint FuncParms[2]
#define minLum                    FuncParms[3]

// gamma 2.2 emulation
#define whitePointNormalised FuncParms[4]


  // black floor lowering
  float newBlackPoint;

  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode != PRO_MODE_RGB)
  {
    rollOffStoppingPoint = Csp::Trc::ToPqFromNits(Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint);
    oldBlackPoint        = Csp::Trc::ToPqFromNits(Ui::HdrBlackFloorFix::Lowering::OldBlackPoint);

    newBlackPoint = Ui::HdrBlackFloorFix::Lowering::NewBlackPoint < 0.f
                  ? -Csp::Trc::ToPqFromNits(abs(Ui::HdrBlackFloorFix::Lowering::NewBlackPoint))
                  :  Csp::Trc::ToPqFromNits(    Ui::HdrBlackFloorFix::Lowering::NewBlackPoint);
  }
  else
  {
    rollOffStoppingPoint = Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint / 10000.f;
    oldBlackPoint        = Ui::HdrBlackFloorFix::Lowering::OldBlackPoint        / 10000.f;

    newBlackPoint = Ui::HdrBlackFloorFix::Lowering::NewBlackPoint / 10000.f;
  }

  rollOffMinusOldBlackPoint = rollOffStoppingPoint - oldBlackPoint;

  minLum = (newBlackPoint - oldBlackPoint) / rollOffMinusOldBlackPoint;


  // gamma 2.2 emulation
  whitePointNormalised = Ui::HdrBlackFloorFix::Gamma22Emu::WhitePoint / 10000.f;

}


void PS_HdrBlackFloorFix(
      float4 VPos         : SV_Position,
      float2 TexCoord     : TEXCOORD0,
  out float4 Output       : SV_Target0,
  in  float  FuncParms[5] : FuncParms)
{
  float3 hdr = tex2D(ReShade::BackBuffer, TexCoord).rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  hdr = Csp::Mat::Bt709To::Bt2020(hdr / 125.f);

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  hdr = Csp::Trc::FromPq(hdr);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  hdr = Csp::Trc::FromHlg(hdr);

#else

  hdr = float3(0.f, 0.f, 0.f);

#endif

  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    Gamma22Emulation(hdr,
                     whitePointNormalised);
  }
  if (Ui::HdrBlackFloorFix::Lowering::EnableLowering)
  {
    hdr = LowerBlackFloor(hdr,
                          rollOffStoppingPoint,
                          oldBlackPoint,
                          rollOffMinusOldBlackPoint,
                          minLum);
  }

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  hdr = Csp::Mat::Bt2020To::Bt709(hdr) * 125.f;

#elif (ACTUAL_COLOUR_SPACE == CSP_HDR10)

  hdr = Csp::Trc::ToPq(hdr);

#elif (ACTUAL_COLOUR_SPACE == CSP_HLG)

  hdr = Csp::Trc::ToHlg(hdr);

#endif

  Output = float4(hdr, 1.f);
}


technique lilium__hdr_black_floor_fix
<
  ui_label = "Lilium's HDR black floor fix";
>
{
  pass HdrBlackFloorFix
  {
    VertexShader = VS_PrepareHdrBlackFloorFix;
     PixelShader = PS_HdrBlackFloorFix;
  }
}

#else

ERROR_STUFF

technique lilium__hdr_black_floor_fix
<
  ui_label = "Lilium's HDR black floor fix (ERROR)";
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
