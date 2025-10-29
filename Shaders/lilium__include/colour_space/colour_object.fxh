
#ifdef ENABLE_COLOUR_OBJECT

  HDR10_TO_LINEAR_LUT()

  namespace CO
  {
    #define PRIM_BT709  0
    #define PRIM_DCI_P3 1
    #define PRIM_BT2020 2

    #define TRC_LINEAR_NORMALISED 0
    #define TRC_LINEAR_80         1
    #define TRC_PQ                2

    struct ColourObject
    {
      float3 RGB;
      uint   prim;
      uint   trc;
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
      bool   isUntouched;
#endif
    };


    namespace GetLuminance
    {
      float LinearNormalised
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                return dot(CO.RGB, Csp::Mat::Bt709ToXYZ[1]);
              }
              case PRIM_DCI_P3:
              {
                return dot(CO.RGB, Csp::Mat::DciP3ToXYZ[1]);
              }
              case PRIM_BT2020:
              {
                return dot(CO.RGB, Csp::Mat::Bt2020ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                return dot(CO.RGB, Csp::Mat::ScRgbToXYZ[1]);
              }
              case PRIM_DCI_P3:
              {
                return dot(CO.RGB, Csp::Mat::DciP3_80ToXYZ[1]);
              }
              case PRIM_BT2020:
              {
                return dot(CO.RGB, Csp::Mat::Bt2020_80ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
                float3 linearColour;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  linearColour = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  linearColour = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                return dot(linearColour, Csp::Mat::Bt2020ToXYZ[1]);
              }
              default:
                return 0.f;
            }
          }
          break;
          default:
            return 0.f;
        }
      }

//      float Linear80
//      (
//        ColourObject CO
//      )
//      {
//        [forcecase]
//        switch(CO.trc)
//        {
//          case TRC_LINEAR_NORMALISED:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT709:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt709ToXYZ[1]);
//              }
//              case PRIM_DCI_P3:
//              {
//                return dot(CO.RGB, Csp::Mat::DciP3ToXYZ[1]);
//              }
//              case PRIM_BT2020:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt2020ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//          case TRC_LINEAR_80:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT709:
//              {
//                return dot(CO.RGB, Csp::Mat::ScRgbToXYZ[1]);
//              }
//              case PRIM_DCI_P3:
//              {
//                return dot(CO.RGB, Csp::Mat::DciP3_80ToXYZ[1]);
//              }
//              case PRIM_BT2020:
//              {
//                return dot(CO.RGB, Csp::Mat::Bt2020_80ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//          case TRC_PQ:
//          {
//            [forcecase]
//            switch(CO.prim)
//            {
//              case PRIM_BT2020:
//              {
//                return dot(Csp::Trc::PqTo::Linear(CO.RGB), Csp::Mat::Bt2020ToXYZ[1]);
//              }
//              default:
//                return 0.f;
//            }
//          }
//        }
//      }
    }


    namespace ConvertCspTo
    {
      ColourObject ScRgb
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
                break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB = Csp::Mat::Bt2020NormalisedTo::ScRgb(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT709;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Hdr10
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::ScRgbTo::Bt2020Normalised(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3_80To::Bt2020Normalised(CO.RGB);

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB /= 125.f;

                CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT2020;
        CO.trc  = TRC_PQ;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject DciP3_80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
                break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Bt2020_80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);

                CO.RGB *= 125.f;
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          case TRC_PQ:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT2020:
              {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
                if (CO.isUntouched)
                {
                  CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
                }
                else
#endif
                {
                  CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
                }

                CO.RGB *= 125.f;
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;
        CO.trc  = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }

    namespace ConvertPrimariesTo
    {
      ColourObject Bt709
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
                break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt709(CO.RGB);
              }
              break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::Bt709(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT709;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject DciP3
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::DciP3(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
                break;
              case PRIM_BT2020:
              {
                CO.RGB = Csp::Mat::Bt2020To::DciP3(CO.RGB);
              }
              break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_DCI_P3;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Bt2020
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          case TRC_LINEAR_80:
          {
            [forcecase]
            switch(CO.prim)
            {
              case PRIM_BT709:
              {
                CO.RGB = Csp::Mat::Bt709To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_DCI_P3:
              {
                CO.RGB = Csp::Mat::DciP3To::Bt2020(CO.RGB);
              }
              break;
              case PRIM_BT2020:
                break;
              default:
                CO.RGB = 0.f;
                break;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.prim = PRIM_BT2020;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }


    namespace ConvertTrcTo
    {
      ColourObject LinearNormalised
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
            break;
          case TRC_LINEAR_80:
          {
            CO.RGB /= 125.f;
          }
          break;
          case TRC_PQ:
          {
            [branch]
            if (CO.prim == PRIM_BT2020)
            {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
              if (CO.isUntouched)
              {
                CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
              }
              else
#endif
              {
                CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
              }
            }
            else
            {
              CO.RGB = 0.f;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.trc = TRC_LINEAR_NORMALISED;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Linear80
      (
        ColourObject CO
      )
      {
        [forcecase]
        switch(CO.trc)
        {
          case TRC_LINEAR_NORMALISED:
          {
            CO.RGB *= 125.f;
          }
          break;
          case TRC_LINEAR_80:
            break;
          case TRC_PQ:
          {
            [branch]
            if (CO.prim == PRIM_BT2020)
            {
#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
              if (CO.isUntouched)
              {
                CO.RGB = FetchFromHdr10ToLinearLUT(CO.RGB);
              }
              else
#endif
              {
                CO.RGB = Csp::Trc::PqTo::Linear(CO.RGB);
              }

              CO.RGB *= 125.f;
            }
            else
            {
              CO.RGB = 0.f;
            }
          }
          break;
          default:
            CO.RGB = 0.f;
            break;
        }

        CO.trc = TRC_LINEAR_80;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }

      ColourObject Pq
      (
        ColourObject CO
      )
      {
        [branch]
        if (CO.prim == PRIM_BT2020)
        {
          [forcecase]
          switch(CO.trc)
          {
            case TRC_LINEAR_NORMALISED:
            {
              CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB);
            }
            break;
            case TRC_LINEAR_80:
            {
              CO.RGB = Csp::Trc::LinearTo::Pq(CO.RGB / 125.f);
            }
            break;
            default:
              CO.RGB = 0.f;
              break;
          }
        }
        else
        {
          CO.RGB = 0.f;
        }

        CO.trc = TRC_PQ;

#if (ACTUAL_COLOUR_SPACE == CSP_HDR10)
        CO.isUntouched = false;
#endif

        return CO;
      }
    }

  }

#endif //ENABLE_COLOUR_OBJECT
