#include "lilium__include/hdr_black_floor_fix.fxh"


#if (defined(IS_ANALYSIS_CAPABLE_API) \
  && (ACTUAL_COLOUR_SPACE == CSP_SCRGB \
   || ACTUAL_COLOUR_SPACE == CSP_HDR10))


void GetParams
(
  out float OutRollOffStoppingPoint,
  out float OutOldBlackPoint,
  out float OutRollOffMinusOldBlackPoint,
  out float OutMinLum,
  out float OutWhitePointNormalised
)
{
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    OutWhitePointNormalised = Ui::HdrBlackFloorFix::Gamma22Emu::WhitePoint / 80.f;
#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)
    OutWhitePointNormalised = Ui::HdrBlackFloorFix::Gamma22Emu::WhitePoint / 10000.f;
#endif


  OutRollOffStoppingPoint = Ui::HdrBlackFloorFix::Lowering::RollOffStoppingPoint / 10000.f;
  OutOldBlackPoint        = Ui::HdrBlackFloorFix::Lowering::OldBlackPoint        / 10000.f;
  float newBlackPoint     = Ui::HdrBlackFloorFix::Lowering::NewBlackPoint        / 10000.f;

  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Lowering::ProcessingMode != PRO_MODE_RGB)
  {
    OutRollOffStoppingPoint = Csp::Trc::LinearTo::Pq(OutRollOffStoppingPoint);

    OutOldBlackPoint = Csp::Trc::LinearTo::Pq(OutOldBlackPoint);

    newBlackPoint = sign(newBlackPoint)
                  * Csp::Trc::LinearTo::Pq(abs(newBlackPoint));
  }

  OutRollOffMinusOldBlackPoint = OutRollOffStoppingPoint
                               - OutOldBlackPoint;

  OutMinLum = (newBlackPoint - OutOldBlackPoint)
            / OutRollOffMinusOldBlackPoint;

  return;
}


// Vertex shader generating a triangle covering the entire screen.
// Calculate values only "once" (3 times because it's 3 vertices)
// for the pixel shader.
void VS_PrepareHdrBlackFloorFix
(
  in                  uint   VertexID   : SV_VertexID,
  out                 float4 Position   : SV_Position
#if (__RESHADE_PERFORMANCE_MODE__ == 0)
                                                     ,
  out nointerpolation float4 FuncParms0 : FuncParms0,
  out nointerpolation float  FuncParms1 : FuncParms1
#endif
)
{
  float2 texCoord;
  texCoord.x = (VertexID == 2) ? 2.f
                               : 0.f;
  texCoord.y = (VertexID == 1) ? 2.f
                               : 0.f;
  Position = float4(texCoord * float2(2.f, -2.f) + float2(-1.f, 1.f), 0.f, 1.f);

#if (__RESHADE_PERFORMANCE_MODE__ == 0)

  FuncParms0 = 0.f;
  FuncParms1 = 0.f;


// black flower lowering
#define rollOffStoppingPoint      FuncParms0.x
#define oldBlackPoint             FuncParms0.y
#define rollOffMinusOldBlackPoint FuncParms0.z
#define minLum                    FuncParms0.w

// gamma 2.2 emulation
#define whitePointNormalised FuncParms1

  GetParams(rollOffStoppingPoint,
            oldBlackPoint,
            rollOffMinusOldBlackPoint,
            minLum,
            whitePointNormalised);

#endif
}


void PS_HdrBlackFloorFix
(
  in                  float4 Position   : SV_Position,
#if (__RESHADE_PERFORMANCE_MODE__ == 0)
  in  nointerpolation float4 FuncParms0 : FuncParms0,
  in  nointerpolation float  FuncParms1 : FuncParms1,
#endif
  out                 float4 Output     : SV_Target0
)
{
#if (__RESHADE_PERFORMANCE_MODE__ == 1)

  float rollOffStoppingPoint;
  float oldBlackPoint;
  float rollOffMinusOldBlackPoint;
  float minLum;
  float whitePointNormalised;

  GetParams(rollOffStoppingPoint,
            oldBlackPoint,
            rollOffMinusOldBlackPoint,
            minLum,
            whitePointNormalised);

#endif

  const float4 inputColour = tex2Dfetch(SamplerBackBuffer, int2(Position.xy));

  bool processingDone = false;

  CO::ColourObject co;

  co.RGB = inputColour.rgb;

#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)

  co.trc  = TRC_LINEAR_80;
  co.prim = PRIM_BT709;

#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)

  co.trc  = TRC_PQ;
  co.prim = PRIM_BT2020;

#endif

  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu)
  {
    co = ConvertColourForGamma22Emulation(co);

    Gamma22Emulation(co,
                     whitePointNormalised,
                     processingDone);
  }

  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Lowering::EnableLowering)
  {
    LowerBlackFloor(co,
                    rollOffStoppingPoint,
                    oldBlackPoint,
                    rollOffMinusOldBlackPoint,
                    minLum,
                    processingDone);
  }

  BRANCH(x)
  if (Ui::HdrBlackFloorFix::Gamma22Emu::EnableGamma22Emu
  && !Ui::HdrBlackFloorFix::Lowering::EnableLowering)
  {
#if (ACTUAL_COLOUR_SPACE == CSP_SCRGB)
    co = CO::ConvertCspTo::ScRgb(co);
#else //(ACTUAL_COLOUR_SPACE == CSP_HDR10)
    co = CO::ConvertCspTo::Hdr10(co);
#endif
  }

  Output = float4(co.RGB, inputColour.a);
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
