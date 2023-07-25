#if ((__RENDERER__ >= 0xB000 && __RENDERER__ < 0x10000) \
  || __RENDERER__ >= 0x20000)


#include "lilium__include\hdr_black_floor_fix.fxh"


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void PrepareFuncParameters(
  in  uint   Id              : SV_VertexID,
  out float4 VPos            : SV_Position,
  out float2 TexCoord        : TEXCOORD0,
  out float4 FuncParameters0 : FuncParameters0,
  out float  FuncParameters1 : FuncParameters1)
{
  TexCoord.x = (Id == 2) ? 2.f
                         : 0.f;
  TexCoord.y = (Id == 1) ? 2.f
                         : 0.f;
  VPos = float4(TexCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

// black flower lowering
#define rollOffStoppingPoint      FuncParameters0.x
#define oldBlackPoint             FuncParameters0.y
#define rollOffMinusOldBlackPoint FuncParameters0.z
#define minLum                    FuncParameters0.w

// gamma 2.2 emulation
#define whitePointNormalised FuncParameters1


  // black floor lowering
  float newBlackPoint;

  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode != PRO_MODE_RGB)
  {
    rollOffStoppingPoint = Csp::Trc::ToPqFromNits(Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint);
    oldBlackPoint        = Csp::Trc::ToPqFromNits(Ui::HdrBlackFloorFix::Lowering::SourceBlackPoint);

    newBlackPoint = Ui::HdrBlackFloorFix::Lowering::TargetBlackPoint < 0.f
                  ? -Csp::Trc::ToPqFromNits(abs(Ui::HdrBlackFloorFix::Lowering::TargetBlackPoint))
                  :  Csp::Trc::ToPqFromNits(    Ui::HdrBlackFloorFix::Lowering::TargetBlackPoint);
  }
  else
  {
    rollOffStoppingPoint = Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint / 10000.f;
    oldBlackPoint        = Ui::HdrBlackFloorFix::Lowering::SourceBlackPoint        / 10000.f;

    newBlackPoint = Ui::HdrBlackFloorFix::Lowering::TargetBlackPoint / 10000.f;
  }

  rollOffMinusOldBlackPoint = rollOffStoppingPoint - oldBlackPoint;

  minLum = (newBlackPoint - oldBlackPoint) / rollOffMinusOldBlackPoint;


  // gamma 2.2 emulation
  whitePointNormalised = Ui::HdrBlackFloorFix::Gamma22Emu::WhitePoint / 10000.f;

}


void HdrBlackFloorFix(
      float4 VPos            : SV_Position,
      float2 TexCoord        : TEXCOORD0,
  out float4 Output          : SV_Target0,
  in  float4 FuncParameters0 : FuncParameters0,
  in  float  FuncParameters1 : FuncParameters1)
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
    VertexShader = PrepareFuncParameters;
     PixelShader = HdrBlackFloorFix;
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
