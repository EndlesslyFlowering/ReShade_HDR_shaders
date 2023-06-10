#include "ReShade.fxh"
#include "HDR_analysis.fxh"
#include "DrawText_fix.fxh"

//#define _DEBUG
//#define _TESTY


uniform float2 MOUSE_POSITION
<
  source = "mousepoint";
>;


uniform bool SHOW_CIE
<
  ui_category = "CIE diagram";
  ui_label    = "show CIE diagram";
> = false;

uniform uint CIE_DIAGRAM
<
  ui_category = "CIE diagram";
  ui_label    = "CIE diagram";
  ui_type     = "combo";
  ui_items    = "1931 xy\0"
                "1976 u'v'\0";
> = 0;

#define CIE_1931 0
#define CIE_1976 1

uniform float CIE_DIAGRAM_BRIGHTNESS
<
  ui_category = "CIE diagram";
  ui_label    = "CIE diagram brightness";
  ui_type     = "slider";
  ui_min      = 10.f;
  ui_max      = 250.f;
  ui_step     = 1.f;
> = 80.f;

uniform bool SHOW_CSPS
<
  ui_category = "colour space %";
  ui_label    = "show colour spaces used";
  ui_tooltip  = "in %";
> = false;

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
  ui_label    = "work around rounding errors for displaying maxCLL";
  ui_tooltip  = "a value of 0.005 is added to the maxCLL value";
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
  ui_items    = "10000nits\0"
                " 1000nits\0";
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
  ui_label    = "[DEBUG] output heatmap in SDR gamma 2.2";
> = false;
#endif

uniform bool DRAW_ABOVE_NITS_AS_BLACK
<
  ui_category = "draw nits as black stuff";
  ui_label    = "draw above nits as black";
> = false;

uniform float ABOVE_NITS_AS_BLACK
<
  ui_category = "draw nits as black stuff";
  ui_label    = "draw above nits as black";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 0.001f;
> = 10000.f;

uniform bool DRAW_BELOW_NITS_AS_BLACK
<
  ui_category = "draw nits as black stuff";
  ui_label    = "draw below nits as black";
> = false;

uniform float BELOW_NITS_AS_BLACK
<
  ui_category = "draw nits as black stuff";
  ui_label    = "draw below nits as black";
  ui_type     = "drag";
  ui_min      = 0.f;
  ui_max      = 10000.f;
  ui_step     = 1.f;
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


  float curPixel = 0.f;

  if (ACTUAL_COLOR_SPACE == CSP_PQ)
  {
    curPixel = PQ_EOTF(dot(BT2020_to_XYZ[1].rgb, pixel)) * 10000.f;
  }
  else if (ACTUAL_COLOR_SPACE == CSP_SCRGB)
  {
    curPixel = dot(BT709_to_XYZ[1].rgb, pixel) * 80.f;
  }
  else if (ACTUAL_COLOR_SPACE == CSP_PS5)
  {
    curPixel = dot(BT2020_to_XYZ[1].rgb, pixel) * 100.f;
  }
  else
  {
    curPixel = 0.f;
  }

  curCLL = curPixel >= 0.f
         ? curPixel
         : 0.f;
}

///text stuff

#define yOffset0 CLL_FONT_SIZE / 2.f
#define yOffset1 CLL_FONT_SIZE - 30

// max/avg/min CLL
static const uint text_maxCLL[26] = { __m, __a, __x, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
static const uint text_avgCLL[26] = { __a, __v, __g, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };
static const uint text_minCLL[26] = { __m, __i, __n, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };

// CLL from cursor position
static const uint text_cursorCLL[28] = { __c, __u, __r, __s, __o, __r, __C, __L, __L, __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __n, __i, __t, __s };

static const uint text_X[2] = { __x, __Colon };
static const uint text_Y[2] = { __y, __Colon };

// RGB value from cursor position
static const uint text_R[2] = { __R, __Colon };
static const uint text_G[2] = { __G, __Colon };
static const uint text_B[2] = { __B, __Colon };

// colour space percentages
static const uint text_BT709[18]  = { __B, __T, __Dot, __7,     __0,     __9,     __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };
static const uint text_BT2020[18] = { __B, __T, __Dot, __2,     __0,     __2,     __0,     __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };
static const uint text_DCI_P3[18] = { __D, __C, __I,   __Minus, __P,     __3,     __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };
static const uint text_AP1[18]    = { __A, __P, __1,   __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };
static const uint text_AP0[18]    = { __A, __P, __0,   __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };
static const uint text_else[18]   = { __e, __l, __s,   __e,     __Colon, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Space, __Percent };

// colour space not supported
static const uint text_Error[24] = { __C, __O, __L, __O, __R, __S, __P, __A, __C, __E, __Space,
                                     __N, __O, __T, __Space,
                                     __S, __U, __P, __P, __O, __R, __T, __E, __D};


void HDR_analysis(
      float4 vpos     : SV_Position,
      float2 texcoord : TEXCOORD,
  out float4 output   : SV_Target0)
{
  const float3 input = tex2D(ReShade::BackBuffer, texcoord).rgb;


  if (ACTUAL_COLOR_SPACE == CSP_PQ
   || ACTUAL_COLOR_SPACE == CSP_SCRGB
   || ACTUAL_COLOR_SPACE == CSP_PS5)
  {
    //float maxCLL = float(uint(tex2Dfetch(sampler_max_avg_min_CLL_values, int2(0, 0)).r*10000.f+0.5)/100)/100.f;
    //float avgCLL = float(uint(tex2Dfetch(sampler_max_avg_min_CLL_values, int2(1, 0)).r*10000.f+0.5)/100)/100.f;
    //float minCLL = float(uint(tex2Dfetch(sampler_max_avg_min_CLL_values, int2(2, 0)).r*10000.f+0.5)/100)/100.f;

    if (SHOW_HEATMAP)
    {
#ifdef _DEBUG
      output = float4(heatmapRGBvalues(tex2D(sampler_CLL_values, texcoord).r, HEATMAP_CUTOFF_POINT, CSP_OVERRIDE, HEATMAP_WHITEPOINT, HEATMAP_SDR), 1.f);
#else
      output = float4(heatmapRGBvalues(tex2D(sampler_CLL_values, texcoord).r, HEATMAP_CUTOFF_POINT, CSP_OVERRIDE, HEATMAP_WHITEPOINT, false), 1.f);
#endif
    }
    else
      output = float4(input, 1.f);
    
    if (DRAW_ABOVE_NITS_AS_BLACK)
    {
      float pixelCLL = tex2D(sampler_CLL_values, texcoord).r;
      if (pixelCLL > ABOVE_NITS_AS_BLACK)
      {
        output = (0.f, 0.f, 0.f, 0.f);
      }
    }
    else if (DRAW_BELOW_NITS_AS_BLACK)
    {
      float pixelCLL = tex2D(sampler_CLL_values, texcoord).r;
      if (pixelCLL < BELOW_NITS_AS_BLACK)
      {
        output = (0.f, 0.f, 0.f, 0.f);
      }
    }

    if (SHOW_CIE)
    {
      uint CIE_BG_X = CIE_DIAGRAM == CIE_1931
                    ? CIE_1931_BG_X
                    : CIE_1976_BG_X;
      uint CIE_BG_Y = CIE_DIAGRAM == CIE_1931
                    ? CIE_1931_BG_Y
                    : CIE_1976_BG_Y;

      uint current_x_coord = texcoord.x * BUFFER_WIDTH;  // expand to actual pixel values
      uint current_y_coord = texcoord.y * BUFFER_HEIGHT; // ^

      // draw the diagram in the bottom left corner
      if (current_x_coord < CIE_BG_X && current_y_coord >= (BUFFER_HEIGHT - CIE_BG_Y))
      {
        // get coords for the sampler
        uint2 current_sampler_coords = uint2(current_x_coord,
                                             current_y_coord - (BUFFER_HEIGHT - CIE_BG_Y));

        float3 current_pixel_to_display = CIE_DIAGRAM == CIE_1931
                                        ? tex2Dfetch(sampler_CIE_1931_cur, current_sampler_coords).rgb
                                        : tex2Dfetch(sampler_CIE_1976_cur, current_sampler_coords).rgb;

        if (ACTUAL_COLOR_SPACE == CSP_PQ)
        {
          output = float4(
            PQ_inverse_EOTF(mul(BT709_to_BT2020, current_pixel_to_display) * (CIE_DIAGRAM_BRIGHTNESS / 10000.f)), 1.f);
        }
        else if (ACTUAL_COLOR_SPACE == CSP_SCRGB)
        {
          output = float4(
            current_pixel_to_display * (CIE_DIAGRAM_BRIGHTNESS / 80.f), 1.f);
        }
        else if (ACTUAL_COLOR_SPACE == CSP_PS5)
        {
          output = float4(
            mul(BT709_to_BT2020, current_pixel_to_display) * (CIE_DIAGRAM_BRIGHTNESS / 100.f), 1.f);
        }
      }
    }

    if (SHOW_CLL_VALUES)
    {
      float maxCLL_show = tex2Dfetch(sampler_show_values, uint2(0, 0)).r;
      float avgCLL_show = tex2Dfetch(sampler_show_values, uint2(1, 0)).r;
      float minCLL_show = tex2Dfetch(sampler_show_values, uint2(2, 0)).r;

      if (ROUNDING_WORKAROUND)
      {
        maxCLL_show += 0.005f;
        //avgCLL_show += 0.005f;
        //minCLL_show += 0.005f;
      }

      DrawText_String(float2(0.f, CLL_FONT_SIZE * 0), CLL_FONT_SIZE, 1, texcoord, text_maxCLL, 26, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 1), CLL_FONT_SIZE, 1, texcoord, text_avgCLL, 26, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 2), CLL_FONT_SIZE, 1, texcoord, text_minCLL, 26, output, FONT_BRIGHTNESS);

      DrawText_Digit(float2(CLL_FONT_SIZE * 7, CLL_FONT_SIZE * 0), CLL_FONT_SIZE, 1, texcoord, 6, maxCLL_show, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 7, CLL_FONT_SIZE * 1), CLL_FONT_SIZE, 1, texcoord, 6, avgCLL_show, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 7, CLL_FONT_SIZE * 2), CLL_FONT_SIZE, 1, texcoord, 6, minCLL_show, output, FONT_BRIGHTNESS);
    }

    if (SHOW_CLL_FROM_CURSOR)
    {
      float mousePositionX = clamp(MOUSE_POSITION.x, 0.f, BUFFER_WIDTH  - 1);
      float mousePositionY = clamp(MOUSE_POSITION.y, 0.f, BUFFER_HEIGHT - 1);
      uint2 mouseXY        = uint2(mousePositionX, mousePositionY);
      float cursorCLL      = tex2Dfetch(sampler_CLL_values, mouseXY).r;

      DrawText_String(float2(CLL_FONT_SIZE * 4, CLL_FONT_SIZE * 4), CLL_FONT_SIZE, 1, texcoord, text_X, 2, output, FONT_BRIGHTNESS);
      DrawText_String(float2(CLL_FONT_SIZE * 4, CLL_FONT_SIZE * 5), CLL_FONT_SIZE, 1, texcoord, text_Y, 2, output, FONT_BRIGHTNESS);

      DrawText_Digit(float2(CLL_FONT_SIZE * 8, CLL_FONT_SIZE * 4), CLL_FONT_SIZE, 1, texcoord, -1, mousePositionX, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 8, CLL_FONT_SIZE * 5), CLL_FONT_SIZE, 1, texcoord, -1, mousePositionY, output, FONT_BRIGHTNESS);

      DrawText_String(float2(0.f, CLL_FONT_SIZE * 6), CLL_FONT_SIZE, 1, texcoord, text_cursorCLL, 28, output, FONT_BRIGHTNESS);

      DrawText_Digit(float2(CLL_FONT_SIZE * 8, CLL_FONT_SIZE * 6), CLL_FONT_SIZE, 1, texcoord, 6, cursorCLL, output, FONT_BRIGHTNESS);

      float3 cursorRGB = tex2Dfetch(ReShade::BackBuffer, mouseXY).rgb;

      DrawText_String(float2(0.f, CLL_FONT_SIZE *  8), CLL_FONT_SIZE, 1, texcoord, text_R, 2, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE *  9), CLL_FONT_SIZE, 1, texcoord, text_G, 2, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 10), CLL_FONT_SIZE, 1, texcoord, text_B, 2, output, FONT_BRIGHTNESS);

      uint RGB_text_offset = 4;
      if (ACTUAL_COLOR_SPACE == CSP_PQ
       || ACTUAL_COLOR_SPACE == CSP_SRGB)
      {
        RGB_text_offset = 2;
      }

      DrawText_Digit(float2(CLL_FONT_SIZE * RGB_text_offset, CLL_FONT_SIZE *  8), CLL_FONT_SIZE, 1, texcoord, 8, cursorRGB.r, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * RGB_text_offset, CLL_FONT_SIZE *  9), CLL_FONT_SIZE, 1, texcoord, 8, cursorRGB.g, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * RGB_text_offset, CLL_FONT_SIZE * 10), CLL_FONT_SIZE, 1, texcoord, 8, cursorRGB.b, output, FONT_BRIGHTNESS);
    }

    if (SHOW_CSPS)
    {
#if ACTUAL_COLOR_SPACE == CSP_PQ
      float precentage_BT709  = tex2Dfetch(sampler_show_values, uint2(0, 1)).r;
      float precentage_DCI_P3 = tex2Dfetch(sampler_show_values, uint2(1, 1)).r;
      float precentage_BT2020 = tex2Dfetch(sampler_show_values, uint2(2, 1)).r;
#elif ACTUAL_COLOR_SPACE != CSP_SRGB
      float precentage_BT709  = tex2Dfetch(sampler_show_values, uint2(0, 1)).r;
      float precentage_DCI_P3 = tex2Dfetch(sampler_show_values, uint2(1, 1)).r;
      float precentage_BT2020 = tex2Dfetch(sampler_show_values, uint2(2, 1)).r;
      float precentage_AP1    = tex2Dfetch(sampler_show_values, uint2(3, 1)).r;
      float precentage_AP0    = tex2Dfetch(sampler_show_values, uint2(4, 1)).r;
      float precentage_else   = tex2Dfetch(sampler_show_values, uint2(5, 1)).r;
#else
      float precentage_BT709  = 100.f;
      float precentage_DCI_P3 =   0.f;
      float precentage_BT2020 =   0.f;
      float precentage_AP1    =   0.f;
      float precentage_AP0    =   0.f;
      float precentage_else   =   0.f;
#endif

      DrawText_String(float2(0.f, CLL_FONT_SIZE * 12), CLL_FONT_SIZE, 1, texcoord, text_BT709,  18, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 13), CLL_FONT_SIZE, 1, texcoord, text_DCI_P3, 18, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 14), CLL_FONT_SIZE, 1, texcoord, text_BT2020, 18, output, FONT_BRIGHTNESS);
#if ACTUAL_COLOR_SPACE != CSP_PQ
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 15), CLL_FONT_SIZE, 1, texcoord, text_AP1,    18, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 16), CLL_FONT_SIZE, 1, texcoord, text_AP0,    18, output, FONT_BRIGHTNESS);
      DrawText_String(float2(0.f, CLL_FONT_SIZE * 17), CLL_FONT_SIZE, 1, texcoord, text_else,   18, output, FONT_BRIGHTNESS);
#endif

      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 12), CLL_FONT_SIZE, 1, texcoord, 4, precentage_BT709,  output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 13), CLL_FONT_SIZE, 1, texcoord, 4, precentage_DCI_P3, output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 14), CLL_FONT_SIZE, 1, texcoord, 4, precentage_BT2020, output, FONT_BRIGHTNESS);
#if ACTUAL_COLOR_SPACE != CSP_PQ
      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 15), CLL_FONT_SIZE, 1, texcoord, 4, precentage_AP0,    output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 16), CLL_FONT_SIZE, 1, texcoord, 4, precentage_AP1,    output, FONT_BRIGHTNESS);
      DrawText_Digit(float2(CLL_FONT_SIZE * 6, CLL_FONT_SIZE * 17), CLL_FONT_SIZE, 1, texcoord, 4, precentage_else,   output, FONT_BRIGHTNESS);
#endif
    }

  }
  else
  {
    output = float4(input, 1.f);
    DrawText_String(float2(0.f, 0.f), 100.f, 1, texcoord, text_Error, 24, output, 1.f);
  }
}

technique CLL_OLD
<
  enabled = false;
>
{
  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLLown;
    RenderTarget = CLL_values;
  }

  pass getMaxAvgMinCLLvalue0
  {
    ComputeShader = getMaxAvgMinCLL0 <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 1;
  }

  pass getMaxAvgMinCLLvalue1
  {
    ComputeShader = getMaxAvgMinCLL1 <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

//  pass getMaxCLLvalue0
//  {
//    ComputeShader = getMaxCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass getMaxCLLvalue1
//  {
//    ComputeShader = getMaxCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass getAvgCLLvalue0
//  {
//    ComputeShader = getAvgCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass getAvgCLLvalue1
//  {
//    ComputeShader = getAvgCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
//
//  pass getMinCLLvalue0
//  {
//    ComputeShader = getMinCLL0 <THREAD_SIZE1, 1>;
//    DispatchSizeX = DISPATCH_X1;
//    DispatchSizeY = 1;
//  }
//
//  pass getMinCLLvalue1
//  {
//    ComputeShader = getMinCLL1 <1, 1>;
//    DispatchSizeX = 1;
//    DispatchSizeY = 1;
//  }
}

technique CLL
<
  enabled = false;
>
{
  pass calcCLLvalues
  {
    VertexShader = PostProcessVS;
     PixelShader = calcCLLown;
    RenderTarget = CLL_values;
  }

  pass getMaxAvgMinCLL0_NEW
  {
    ComputeShader = getMaxAvgMinCLL0_NEW <THREAD_SIZE1, 1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = 2;
  }

  pass getMaxAvgMinCLL1_NEW
  {
    ComputeShader = getMaxAvgMinCLL1_NEW <1, 1>;
    DispatchSizeX = 2;
    DispatchSizeY = 2;
  }

  pass getFinalMaxAvgMinCLL_NEW
  {
    ComputeShader = getFinalMaxAvgMinCLL_NEW <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
}

technique CIE
<
  enabled = false;
>
{
  pass copy_CIE_1931_bg
  {
    VertexShader = PostProcessVS;
     PixelShader = copy_CIE_1931_bg;
    RenderTarget = CIE_1931_cur;
  }

  pass copy_CIE_1976_bg
  {
    VertexShader = PostProcessVS;
     PixelShader = copy_CIE_1976_bg;
    RenderTarget = CIE_1976_cur;
  }

  pass generate_CIE_diagram
  {
    ComputeShader = generate_CIE_diagram <THREAD_SIZE1, THREAD_SIZE1>;
    DispatchSizeX = DISPATCH_X1;
    DispatchSizeY = DISPATCH_Y1;
  }
}

technique CSP
<
  enabled = false;
>
{
#if ACTUAL_COLOR_SPACE != CSP_SRGB
  pass calc_CSPs
  {
    VertexShader = PostProcessVS;
     PixelShader = calc_CSPs;
    RenderTarget = CSPs;
  }

  pass count_CSPs_y
  {
    ComputeShader = count_CSPs_y <THREAD_SIZE0, 1>;
    DispatchSizeX = DISPATCH_X0;
    DispatchSizeY = 1;
  }

  pass count_CSPs_x
  {
    ComputeShader = count_CSPs_x <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }
#endif
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

  pass copy_show_values
  {
    ComputeShader = show_values_copy <1, 1>;
    DispatchSizeX = 1;
    DispatchSizeY = 1;
  }

  pass
  {
    VertexShader = PostProcessVS;
     PixelShader = HDR_analysis;
  }
}
