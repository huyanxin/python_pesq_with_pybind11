/*****************************************************************************

Perceptual Evaluation of Speech Quality (PESQ)
ITU-T Recommendation P.862.
Version 1.2 - 2 August 2002.

              ****************************************
              PESQ Intellectual Property Rights Notice
              ****************************************

DEFINITIONS:
------------
For the purposes of this Intellectual Property Rights Notice
the terms 'Perceptual Evaluation of Speech Quality Algorithm'
and 'PESQ Algorithm' refer to the objective speech quality
measurement algorithm defined in ITU-T Recommendation P.862;
the term 'PESQ Software' refers to the C-code component of P.862. 

NOTICE:
-------
All copyright, trade marks, trade names, patents, know-how and
all or any other intellectual rights subsisting in or used in
connection with including all algorithms, documents and manuals
relating to the PESQ Algorithm and or PESQ Software are and remain
the sole property in law, ownership, regulations, treaties and
patent rights of the Owners identified below. The user may not
dispute or question the ownership of the PESQ Algorithm and
or PESQ Software.

OWNERS ARE:
-----------

1.	British Telecommunications plc (BT), all rights assigned
      to Psytechnics Limited
2.	Royal KPN NV, all rights assigned to OPTICOM GmbH

RESTRICTIONS:
-------------

The user cannot:

1.	alter, duplicate, modify, adapt, or translate in whole or in
      part any aspect of the PESQ Algorithm and or PESQ Software
2.	sell, hire, loan, distribute, dispose or put to any commercial
      use other than those permitted below in whole or in part any
      aspect of the PESQ Algorithm and or PESQ Software

PERMITTED USE:
--------------

The user may:

1.	Use the PESQ Software to:
      i)   understand the PESQ Algorithm; or
      ii)  evaluate the ability of the PESQ Algorithm to perform
           its intended function of predicting the speech quality
           of a system; or
      iii) evaluate the computational complexity of the PESQ Algorithm,
           with the limitation that none of said evaluations or its
           results shall be used for external commercial use.

2.	Use the PESQ Software to test if an implementation of the PESQ
      Algorithm conforms to ITU-T Recommendation P.862.

3.	With the prior written permission of both Psytechnics Limited
      and OPTICOM GmbH, use the PESQ Software in accordance with the
      above Restrictions to perform work that meets all of the following
      criteria:
      i)    the work must contribute directly to the maintenance of an
            existing ITU recommendation or the development of a new ITU
            recommendation under an approved ITU Study Item; and
      ii)   the work and its results must be fully described in a
            written contribution to the ITU that is presented at a formal
            ITU meeting within one year of the start of the work; and
      iii)  neither the work nor its results shall be put to any
            commercial use other than making said contribution to the ITU.
            Said permission will be provided on a case-by-case basis.


ANY OTHER USE OR APPLICATION OF THE PESQ SOFTWARE AND/OR THE PESQ
ALGORITHM WILL REQUIRE A PESQ LICENCE AGREEMENT, WHICH MAY BE OBTAINED
FROM EITHER OPTICOM GMBH OR PSYTECHNICS LIMITED. 

EACH COMPANY OFFERS OEM LICENSE AGREEMENTS, WHICH COMBINE OEM
IMPLEMENTATIONS OF THE PESQ ALGORITHM TOGETHER WITH A PESQ PATENT LICENSE
AGREEMENT. PESQ PATENT-ONLY LICENSE AGREEMENTS MAY BE OBTAINED FROM OPTICOM.


***********************************************************************
*  OPTICOM GmbH                    *  Psytechnics Limited             *
*  Am Weichselgarten 7,            *  Fraser House, 23 Museum Street, *
*  D- 91058 Erlangen, Germany      *  Ipswich IP1 1HN, England        *
*  Phone: +49 (0) 9131 691 160     *  Phone: +44 (0) 1473 261 800     *
*  Fax:   +49 (0) 9131 691 325     *  Fax:   +44 (0) 1473 261 880     *
*  E-mail: info@opticom.de,        *  E-mail: info@psytechnics.com,   *
*  www.opticom.de                  *  www.psytechnics.com             *
***********************************************************************

Further information is also available from www.pesq.org

*****************************************************************************/

#include <stdio.h>
#include <math.h>
#include "pesq.h"
#include "dsp.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include <iostream>
int Nb; // define in pesq.h
#define max( x, y) ( (x) > (y) ? (x) :  (y) )

namespace py = pybind11;

extern float InIIR_Hsos_16k [];
extern float InIIR_Hsos_8k [];
extern long InIIR_Nsos;

void alloc_other( SIGNAL_INFO * ref_info, SIGNAL_INFO * deg_info, 
        long * Error_Flag, char ** Error_Type, float ** ftmp)
{
    *ftmp = (float *)safe_malloc(
       max( max(
            (*ref_info).Nsamples + DATAPADDING_MSECS  * (Fs / 1000),
            (*deg_info).Nsamples + DATAPADDING_MSECS  * (Fs / 1000) ),
           12 * Align_Nfft) * sizeof(float) );
    if( (*ftmp) == NULL )
    {
        *Error_Flag = 2;
        *Error_Type = "Failed to allocate memory for temporary storage.";
        printf ("%s!\n", *Error_Type);
        return;
    }
}

/* END OF FILE */

void select_rate( long sample_rate, long * Error_Flag, char ** Error_Type )
{
    if( Fs == sample_rate )
        return;
    if( Fs_16k == sample_rate )
    {
        Fs = Fs_16k;
        Downsample = Downsample_16k;
        InIIR_Hsos = InIIR_Hsos_16k;
        InIIR_Nsos = InIIR_Nsos_16k;
        Align_Nfft = Align_Nfft_16k;
        return;
    }
    if( Fs_8k == sample_rate )
    {
        Fs = Fs_8k;
        Downsample = Downsample_8k;
        InIIR_Hsos = InIIR_Hsos_8k;
        InIIR_Nsos = InIIR_Nsos_8k;
        Align_Nfft = Align_Nfft_8k;
        return;
    }

    (*Error_Flag) = -1;
    (*Error_Type) = "Invalid sample rate specified";    
}



void read_data_from_array(SIGNAL_INFO *sinfo, py::buffer_info & bfinfo)
{
    int size = bfinfo.size + 2*SEARCHBUFFER*Downsample;
    sinfo->Nsamples = size;

    sinfo->data = (float*) safe_malloc((size+DATAPADDING_MSECS*Fs/1000)*sizeof(float));
    
    if (sinfo->data == NULL)
       throw std::runtime_error("Cannot alloc memory for signal's data");
    // initial
    memset(sinfo->data, 0, sizeof(float)*(size+DATAPADDING_MSECS*Fs/1000));
    for(int i =0;i < bfinfo.size; i++)
    {
        *(sinfo->data + SEARCHBUFFER*Downsample + i) = *((float*)bfinfo.ptr+i)*32768;
    }
    sinfo-> VAD = (float*)safe_malloc( sinfo-> Nsamples * sizeof(float) / Downsample );
    sinfo-> logVAD = (float*)safe_malloc( sinfo-> Nsamples * sizeof(float) / Downsample );
    
    if( (sinfo-> VAD == NULL) || (sinfo-> logVAD == NULL))
    {
        printf("Failed to allocate memory for VAD\n");
        throw std::runtime_error("Cannot alloc memory for signal's vad");
    }
    return;
}


double align_filter_dB [26] [2] = {{0.,-500},
                                 {50., -500},
                                 {100., -500},
                                 {125., -500},
                                 {160., -500},
                                 {200., -500},
                                 {250., -500},
                                 {300., -500},
                                 {350.,  0},
                                 {400.,  0},
                                 {500.,  0},
                                 {600.,  0},
                                 {630.,  0},
                                 {800.,  0},
                                 {1000., 0},
                                 {1250., 0},
                                 {1600., 0},
                                 {2000., 0},
                                 {2500., 0},
                                 {3000., 0},
                                 {3250., 0},
                                 {3500., -500},
                                 {4000., -500},
                                 {5000., -500},
                                 {6300., -500},
                                 {8000., -500}}; 


double standard_IRS_filter_dB [26] [2] = {{  0., -200},
                                         { 50., -40}, 
                                         {100., -20},
                                         {125., -12},
                                         {160.,  -6},
                                         {200.,   0},
                                         {250.,   4},
                                         {300.,   6},
                                         {350.,   8},
                                         {400.,  10},
                                         {500.,  11},
                                         {600.,  12},
                                         {700.,  12},
                                         {800.,  12},
                                         {1000., 12},
                                         {1300., 12},
                                         {1600., 12},
                                         {2000., 12},
                                         {2500., 12},
                                         {3000., 12},
                                         {3250., 12},
                                         {3500., 4},
                                         {4000., -200},
                                         {5000., -200},
                                         {6300., -200},
                                         {8000., -200}}; 


#define TARGET_AVG_POWER    1E7

void fix_power_level (SIGNAL_INFO *info, char *name, long maxNsamples) 
{
    long   n = info-> Nsamples;
    long   i;
    float *align_filtered = (float *) safe_malloc ((n + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));    
    float  global_scale;
    float  power_above_300Hz;

    for (i = 0; i < n + DATAPADDING_MSECS  * (Fs / 1000); i++) {
        align_filtered [i] = info-> data [i];
    }
    apply_filter (align_filtered, info-> Nsamples, 26, align_filter_dB);
    power_above_300Hz = (float) pow_of (align_filtered, 
                                        SEARCHBUFFER * Downsample, 
                                        n - SEARCHBUFFER * Downsample + DATAPADDING_MSECS  * (Fs / 1000),
                                        maxNsamples - 2 * SEARCHBUFFER * Downsample + DATAPADDING_MSECS  * (Fs / 1000));

    global_scale = (float) sqrt (TARGET_AVG_POWER / power_above_300Hz); 
    for (i = 0; i < n; i++) {
        info-> data [i] *= global_scale;    
    }

    safe_free (align_filtered);
}

       
void pesq_measure (SIGNAL_INFO * ref_info, SIGNAL_INFO * deg_info,
    ERROR_INFO * err_info, long * Error_Flag, char ** Error_Type)
{
    float * ftmp = NULL;

        
    if (((ref_info-> Nsamples - 2 * SEARCHBUFFER * Downsample < Fs / 4) ||
         (deg_info-> Nsamples - 2 * SEARCHBUFFER * Downsample < Fs / 4)) &&
        ((*Error_Flag) == 0))
    {
        (*Error_Flag) = 2;
        (*Error_Type) = "Reference or Degraded below 1/4 second - processing stopped ";
    }

    if ((*Error_Flag) == 0)
    {
        alloc_other (ref_info, deg_info, Error_Flag, Error_Type, &ftmp);
    }
    if ((*Error_Flag) == 0)
    {   
        int     maxNsamples = max (ref_info-> Nsamples, deg_info-> Nsamples);
        float * model_ref; 
        float * model_deg; 
        long    i;

        //printf (" Level normalization ...\n");            
        fix_power_level (ref_info, "reference", maxNsamples);
        fix_power_level (deg_info,  "degraded", maxNsamples);
        
        //printf (" IRS filtering...\n"); 
        apply_filter (deg_info-> data, deg_info-> Nsamples, 26, standard_IRS_filter_dB);
        apply_filter (ref_info-> data, ref_info-> Nsamples, 26, standard_IRS_filter_dB);

        model_ref = (float *) safe_malloc ((ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));
        model_deg = (float *) safe_malloc ((deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof (float));
        for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            model_ref [i] = ref_info-> data [i];
        }
    
        for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            model_deg [i] = deg_info-> data [i];
        }
    
        input_filter( ref_info, deg_info, ftmp );

        //printf (" Variable delay compensation...\n");            
        calc_VAD (ref_info);
        calc_VAD (deg_info);
        
        crude_align (ref_info, deg_info, err_info, WHOLE_SIGNAL, ftmp);

        utterance_locate (ref_info, deg_info, err_info, ftmp);
    
        for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            ref_info-> data [i] = model_ref [i];
        }
    
        for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
            deg_info-> data [i] = model_deg [i];
        }

        safe_free (model_ref);
        safe_free (model_deg); 
        // padding 
        if ((*Error_Flag) == 0) {
            if (ref_info-> Nsamples < deg_info-> Nsamples) {
                float *new_ref = (float *) safe_malloc((deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof(float));
                long  i;
                for (i = 0; i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                    new_ref [i] = ref_info-> data [i];
                }
                for (i = ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); 
                     i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                    new_ref [i] = 0.0f;
                }
                safe_free (ref_info-> data);
                ref_info-> data = new_ref;
                new_ref = NULL;
            } else {
                if (ref_info-> Nsamples > deg_info-> Nsamples) {
                    float *new_deg = (float *) safe_malloc((ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000)) * sizeof(float));
                    long  i;
                    for (i = 0; i < deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                        new_deg [i] = deg_info-> data [i];
                    }
                    for (i = deg_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); 
                         i < ref_info-> Nsamples + DATAPADDING_MSECS  * (Fs / 1000); i++) {
                        new_deg [i] = 0.0f;
                    }
                    safe_free (deg_info-> data);
                    deg_info-> data = new_deg;
                    new_deg = NULL;
                }
            }
        }        
        //printf (" Acoustic model processing...\n");    
        pesq_psychoacoustic_model (ref_info, deg_info, err_info, ftmp);
        safe_free (ref_info-> data);
        safe_free (ref_info-> VAD);
        safe_free (ref_info-> logVAD);
        safe_free (deg_info-> data);
        safe_free (deg_info-> VAD);
        safe_free (deg_info-> logVAD);
        safe_free (ftmp);
        //fprintf (resultsFile, "%.3f\t ", err_info->pesq_mos);
    }

    return;
}

float pesq (py::array_t<float> &ref, py::array_t<float> &deg, long &sample_rate) {
    
    long Error_Flag = 0;
    char * Error_Type = "Unknown error type.";
//    if (sample_rate != 8000 || sample_rate != 16000)
//    {
//        std::cout<<sample_rate<<" AAAAAAAAAAAAAAAAA"<<std::endl;
//       throw std::runtime_error("Please send  8000 or 16000 as sample_rate !!!");
//   }
    
    select_rate (sample_rate, &Error_Flag, &Error_Type);
    if ( ref.ndim() != 1 || deg.ndim() != 1)
        throw std::runtime_error("Please send ref_sig or deg_sig like 1D array !!!");

    py::buffer_info ref_buffer = ref.request();
    py::buffer_info deg_buffer = deg.request();
   
    SIGNAL_INFO ref_info;
    SIGNAL_INFO deg_info;
    ERROR_INFO err_info;
    
    read_data_from_array(&deg_info, deg_buffer);
    read_data_from_array(&ref_info, ref_buffer);

    err_info. subj_mos = 0;
    err_info. cond_nr = 0;
    ref_info. apply_swap = 0;
    deg_info. apply_swap = 0;
    if (Error_Flag == -1)
        return -1;
    pesq_measure (&ref_info, &deg_info, &err_info, &Error_Flag, &Error_Type);
    return  err_info.pesq_mos;
}

PYBIND11_MODULE(python_pesq, m)
{
    m.doc() = "python pesq: a wrapper for itu.p862 ";
    m.def("pesq", &pesq, "caculate pesq: ref_wave_array(1D), deg_wave_arrary(1D), sample_rate(16000 or 8000)");
}

/* END OF FILE */
