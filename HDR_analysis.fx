#include "ReShade.fxh"
#include "HDR_analysis.fxh"
#include "DrawText_fix.fxh"

//#define _DEBUG
//#define _TESTY

uniform float2 MOUSE_POSITION
<
  source = "mousepoint";
>;

uniform bool SHOW_CLL_VALUES
<
  ui_category = "CLL stuff";
  ui_label    = "show CLL values";
  ui_tooltip  = "shows max/avg/min Content Light Level";
> = false;

uniform bool SHOW_CLL_FROM_CURSOR
<
  ui_category = "CLL stuff";
  ui_label    = "show CLL value from cursor position";
> = false;

uniform uint CLL_FONT_SIZE
<
  ui_category = "CLL stuff";
  ui_label    = "font size for CLL values";
  ui_type     = "slider";
  ui_min      = 30;
  ui_max      = 40;
> = 30;

uniform bool ROUNDING_WORKAROUND
<
  ui_category = "CLL stuff";
  ui_label    = "work around rounding errors for displaying CLL values";
  ui_tooltip  = "a value of 0.01 is added to every CLL value\n"
                "0.00 may be shown as 0.01 instead";
> = false;

uniform bool SHOW_HEATMAP
<
  ui_category = "heatmap stuff";
  ui_label    = "show heatmap";
> = false;

uniform uint HEATMAP_CUTOFF_POINT
<
  ui_category = "heatmap stuff";
  ui_label    = "heatmap cutoff point";
  ui_tooltip  = "colours:            10000 nits:     1000 nits:\n"
                "black and white        0-  100         0- 100 \n"
                "teal   to green      100-  203       100- 203 \n"
                "green  to yellow     203-  400       203- 400 \n"
                "yellow to red        400- 1000       400- 600 \n"
                "red    to pink      1000- 4000       600- 800 \n"
                "pink   to blue      4000-10000       800-1000";
  ui_type     = "combo";
  ui_items    = " 10000nits\0 1000nits\0";
> = 0;

uniform uint OVERRIDE_CSP
<
  ui_category = "heatmap stuff";
  ui_label    = "override current colourspace";
  ui_type     = "combo";
  ui_items    = "no\0PQ\0scRGB\0";
> = 0;

uniform float HEATMAP_WHITEPOINT
<
  ui_category = "heatmap stuff";
  ui_label    = "heatmap whitepoint (nits)";
  ui_type     = "slider";
  ui_min      = 1.f;
  ui_max      = 203.f;
  ui_step     = 1.f;
> = 80.f;

#ifdef _DEBUG
uniform bool HEATMAP_SDR
<
  ui_category = "heatmap stuff";
  ui_label    = "[DEBUG] output heatmap in SDR gamma2.2";
> = false;
#endif

uniform bool DRAW_ABOVE_NITS_AS_BLACK
<
  ui_label = "draw above nits as black";
> = false;

uniform float ABOVE_NITS_AS_BLACK
<
  ui_label = "draw above nits as black";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 10000.f;
  ui_step  = 1.f;
> = 10000.f;

uniform bool DRAW_BELOW_NITS_AS_BLACK
<
  ui_label = "draw below nits as black";
> = false;

uniform float BELOW_NITS_AS_BLACK
<
  ui_label = "draw below nits as black";
  ui_type  = "drag";
  ui_min   = 0.f;
  ui_max   = 10000.f;
  ui_step  = 1.f;
> = 0.f;

#ifdef _TESTY
uniform bool ENABLE_TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "enable test thingy";
> = false;

uniform float TEST_THINGY
<
  ui_category = "TESTY";
  ui_label    = "test thingy";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 125.f;
  ui_step     = 0.1f;
> = 0.f;
#endif


//void draw_maxCLL(float4 position : POSITION, float2 txcoord : TEXCOORD) : COLOR
//void draw_maxCLL(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 fragment : SV_Target0)
//{
//  const uint int_maxCLL = int(round(maxCLL));
//  uint digit1;
//  uint digit2;
//  uint digit3;
//  uint digit4;
//  uint digit5;
//  
//  if (maxCLL < 10)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = 0;
//    digit4 = 0;
//    digit5 = int_maxCLL;
//  }
//  else if (maxCLL < 100)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = 0;
//    digit4 = int_maxCLL / 10;
//    digit5 = int_maxCLL % 10;
//  }
//  else if (maxCLL < 1000)
//  {
//    digit1 = 0;
//    digit2 = 0;
//    digit3 = int_maxCLL / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }
//  else if (maxCLL < 10000)
//  {
//    digit1 = 0;
//    digit2 = int_maxCLL / 1000;
//    digit3 = (int_maxCLL % 1000) / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }
//  else
//  {
//    digit1 = int_maxCLL / 10000;
//    digit2 = (int_maxCLL % 10000) / 1000;
//    digit3 = (int_maxCLL % 1000) / 100;
//    digit4 = (int_maxCLL % 100) / 10;
//    digit5 = (int_maxCLL % 10);
//  }

  //res += tex2D(samplerText, (frac(uv) + float2(index % 14.0, trunc(index / 14.0))) /
  //            float2(_DRAWTEXT_GRID_X, _DRAWTEXT_GRID_Y)).x;

//  float4 hud = tex2D(samplerNumbers, texcoord);
//  fragment = lerp(tex2Dfetch(ReShade::BackBuffer, texcoord), hud, 1.f);
//
//}

#ifdef _TESTY
void testy(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  if(ENABLE_TEST_THINGY == true)
  {
    const float xTest = TEST_THINGY;
    const float xxx = BUFFER_WIDTH  / 2.f - 100.f;
    const float xxe = (BUFFER_WIDTH  - xxx);
    const float yyy = BUFFER_HEIGHT / 2.f - 100.f;
    const float yye = (BUFFER_HEIGHT - yyy);
    if (texcoord.x > xxx / BUFFER_WIDTH
     && texcoord.x < xxe / BUFFER_WIDTH
     && texcoord.y > yyy / BUFFER_HEIGHT
     && texcoord.y < yye / BUFFER_HEIGHT)
      output = float4(0.f, xTest, 0.f, 1.f);
    else
      output = float4(0.f, 0.f, 0.f, 0.f);
  }
  else
    output = float4(tex2D(ReShade::BackBuffer, texcoord).rgb, 1.f);
}
#endif

void calcCLLown(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float  curCLL   : SV_TARGET)
{
  const float3 pixel = tex2D(ReShade::BackBuffer, texcoord).rgb;

  if ((CSP_PQ || OVERRIDE_CSP == 1) && OVERRIDE_CSP != 2)
    curCLL = 10000.f * PQ_EOTF_single(dot(bt2020_to_XYZ[1].rgb, pixel));
  else if ((CSP_SCRGB || OVERRIDE_CSP == 2) && OVERRIDE_CSP != 1)
    curCLL = dot(bt709_to_XYZ[1].rgb, pixel) * 80.f;
  else
    curCLL = 0.f;
}

void HDR_analysis(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;

  if (CSP_PQ || CSP_SCRGB || OVERRIDE_CSP > 0)
  {
    //float maxCLL = float(uint(tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(0, 0)).r*10000.f+0.5)/100)/100.f;
    //float avgCLL = float(uint(tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(1, 0)).r*10000.f+0.5)/100)/100.f;
    //float minCLL = float(uint(tex2Dfetch(samplerMaxAvgMinCLLvalues, int2(2, 0)).r*10000.f+0.5)/100)/100.f;
    float maxCLL_show = tex2Dfetch(samplerMaxAvgMinCLLSHOWvalues, int2(0, 0)).r;
    float avgCLL_show = tex2Dfetch(samplerMaxAvgMinCLLSHOWvalues, int2(1, 0)).r;
    float minCLL_show = tex2Dfetch(samplerMaxAvgMinCLLSHOWvalues, int2(2, 0)).r;

    if (ROUNDING_WORKAROUND)
    {
      maxCLL_show += 0.01f;
      avgCLL_show += 0.01f;
      minCLL_show += 0.01f;
    }

    if (SHOW_HEATMAP)
    {
#ifdef _DEBUG
      output = float4(heatmapRGBvalues(tex2D(samplerCLLvalues, texcoord).r, HEATMAP_CUTOFF_POINT, OVERRIDE_CSP, HEATMAP_WHITEPOINT, HEATMAP_SDR), 1.f);
#else
      output = float4(heatmapRGBvalues(tex2D(samplerCLLvalues, texcoord).r, HEATMAP_CUTOFF_POINT, OVERRIDE_CSP, HEATMAP_WHITEPOINT, false), 1.f);
#endif
    }
    else
      output = float4(input, 1.f);
    
    if (DRAW_ABOVE_NITS_AS_BLACK)
    {
      const float pixelCLL = tex2D(samplerCLLvalues, texcoord).r;
      if (pixelCLL > ABOVE_NITS_AS_BLACK)
        output = (0.f, 0.f, 0.f, 0.f);
    }
    else if (DRAW_BELOW_NITS_AS_BLACK)
    {
      const float pixelCLL = tex2D(samplerCLLvalues, texcoord).r;
      if (pixelCLL < BELOW_NITS_AS_BLACK)
        output = (0.f, 0.f, 0.f, 0.f);
    }

    const float bright = CSP_PQ || OVERRIDE_CSP == 1
                       ? 0.58068888f
                       : CSP_SCRGB || OVERRIDE_CSP == 2
                       ? 203.f / 80.f
                       : 1.f;

    if (SHOW_CLL_VALUES)
    {
      DrawText_Digit(float2(100.f, 0.f),                 CLL_FONT_SIZE, 1, texcoord, 2, maxCLL_show, output, bright);
      DrawText_Digit(float2(100.f, CLL_FONT_SIZE),       CLL_FONT_SIZE, 1, texcoord, 2, avgCLL_show, output, bright);
      DrawText_Digit(float2(100.f, CLL_FONT_SIZE * 2.f), CLL_FONT_SIZE, 1, texcoord, 2, minCLL_show, output, bright);
    }

    if (SHOW_CLL_FROM_CURSOR)
    {
      const float mousePositionX = clamp(MOUSE_POSITION.x, 0.f, BUFFER_WIDTH  - 1);
      const float mousePositionY = clamp(MOUSE_POSITION.y, 0.f, BUFFER_HEIGHT - 1);
      const int2 mouseXY = int2(mousePositionX, mousePositionY);
      const float cursorCLL = tex2Dfetch(samplerCLLvalues, mouseXY).r;
      const int textX[2] = { __x, __Colon };
      const int textY[2] = { __y, __Colon };
      const float yOffset0 =  CLL_FONT_SIZE / 2.f;
      const float yOffset1 =  CLL_FONT_SIZE - 30;
      DrawText_String(float2(0.f,              CLL_FONT_SIZE * 4), CLL_FONT_SIZE, 1, texcoord, textX, 2,              output, bright);
      DrawText_String(float2(0.f,              CLL_FONT_SIZE * 5), CLL_FONT_SIZE, 1, texcoord, textY, 2,              output, bright);
      DrawText_Digit (float2(100.f + yOffset0, CLL_FONT_SIZE * 4), CLL_FONT_SIZE, 1, texcoord, -1,    mousePositionX, output, bright);
      DrawText_Digit (float2(100.f + yOffset0, CLL_FONT_SIZE * 5), CLL_FONT_SIZE, 1, texcoord, -1,    mousePositionY, output, bright);
      DrawText_Digit (float2(100.f + yOffset0, CLL_FONT_SIZE * 6), CLL_FONT_SIZE, 1, texcoord,  2,    cursorCLL,      output, bright);
      const int textR[2] = { __R, __Colon };
      const int textG[2] = { __G, __Colon };
      const int textB[2] = { __B, __Colon };
      const float3 cursorRGB = tex2Dfetch(ReShade::BackBuffer, mouseXY).rgb;
      DrawText_String(float2( 0.f,            CLL_FONT_SIZE *  8), CLL_FONT_SIZE, 1, texcoord, textR, 2,           output, bright);
      DrawText_String(float2( 0.f,            CLL_FONT_SIZE *  9), CLL_FONT_SIZE, 1, texcoord, textG, 2,           output, bright);
      DrawText_String(float2( 0.f,            CLL_FONT_SIZE * 10), CLL_FONT_SIZE, 1, texcoord, textB, 2,           output, bright);
      DrawText_Digit (float2(96.f + yOffset1, CLL_FONT_SIZE *  8), CLL_FONT_SIZE, 1, texcoord, 6,     cursorRGB.r, output, bright);
      DrawText_Digit (float2(96.f + yOffset1, CLL_FONT_SIZE *  9), CLL_FONT_SIZE, 1, texcoord, 6,     cursorRGB.g, output, bright);
      DrawText_Digit (float2(96.f + yOffset1, CLL_FONT_SIZE * 10), CLL_FONT_SIZE, 1, texcoord, 6,     cursorRGB.b, output, bright);
    }
  }
  else
  {
    output = float4(input, 1.f);
    const int textError[24] = { __C, __O, __L, __O, __R, __S, __P, __A, __C, __E, __Space,
                                __N, __O, __T, __Space,
                                __S, __U, __P, __P, __O, __R, __T, __E, __D};
    DrawText_String(float2(0.f, 0.f), 100.f, 1, texcoord, textError, 24, output, 1.f);
  }
}

technique HDR_analysis
{
#ifdef _TESTY
  pass test_thing
  {
    VertexShader = PostProcessVS;
     PixelShader = testy;
  }
#endif

  //pass calcCLLvalues
  //{
  //  ComputeShader = calcCLL < THREAD_SIZE0, THREAD_SIZE0>;
  //  DispatchSizeX = DISPATCH_X0;
  //  DispatchSizeY = DISPATCH_Y0;
  //}

  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLLown;
    RenderTarget = CLLvalues;
  }

  pass calcMaxCLLvalue0
  {
    ComputeShader = calcMaxCLL0 < THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass calcMaxCLLvalue1
  {
    ComputeShader = calcMaxCLL1 < 1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass calcAvgCLLvalue0
  {
    ComputeShader = calcAvgCLL0 < THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass calcAvgCLLvalue1
  {
    ComputeShader = calcAvgCLL1 < 1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass calcMinCLLvalue0
  {
    ComputeShader = calcMinCLL0 < THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass calcMinCLLvalue1
  {
    ComputeShader = calcMinCLL1 < 1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass copySHOWvalues
  {
    ComputeShader = showCLLvaluesCopy < 1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass
  {
    VertexShader = PostProcessVS;
     PixelShader = HDR_analysis;
  }
}