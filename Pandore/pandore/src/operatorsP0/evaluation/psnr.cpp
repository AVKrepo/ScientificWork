/* -*- c-basic-offset: 3; mode: c++ -*-
 *
 * Copyright (c), GREYC.
 * All rights reserved
 *
 * You may use this file under the terms of the BSD license as follows:
 *
 * "Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are
 * met:
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice, this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of the GREYC, nor the name of its
 *     contributors may be used to endorse or promote products
 *     derived from this software without specific prior written
 *     permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, 
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 * 
 * For more information, refer to:
 * https://clouard.users.greyc.fr/Pandore
 */

#include <math.h>
#include <pandore.h>
using namespace pandore;

/**
 * @file psnr.cpp
 *
 * Calculates the Signal Noise Ration.
 *
 *     SNR = 10 * log10 ( xsb / ysb )
 *
 * where xsb is the measure of the energy in the signal, 
 * xsb = sum (ims1^2)
 * and ysb is the energy of the noise that was added to the signal.
 * ysb = sum ((ims1-ims2)^2)
 */
template <typename T>
Errc PSNR( const Imx3d<T> &ims1, const Imx3d<T> &ims2 ) {
   if (ims1.Size() != ims2.Size() ) {
      std::cerr << "Error psnr: Input images have different size." << std::endl;
      return FAILURE;
   }
   if (ims1.Bands() != ims2.Bands() ) {
      std::cerr << "Error psnr: Input images have different number of bands." << std::endl;
      return FAILURE;
   }

   const T *pend = ims1.Vector(0) + ims1.VectorSize();
   double xsb = 0.0;
   double ysb = 0.0;
   for (int b = 0; b < ims1.Bands(); b++) {
      T *p1 = ims1.Vector(b);
      T *p2 = ims2.Vector(b);
      
      for (; p1 < pend; ) {
	 xsb  += *p1 * *p1;
	 double val = *(p1++) - *(p2++);
	 ysb  += val * val;
      }
   }
   return (Float)(10.0 * log10( xsb / ysb));
}



#ifdef MAIN
#define	USAGE	"USAGE : %s [im_in1|-] [im_in2|-]"
#define	PARC	0
#define	FINC	2
#define	FOUTC	0
#define	MASK	0

int main( int argc, char *argv[] ) {
   Errc result;                // The result code of the execution.
   Pobject* mask;              // The region map.
   Pobject* objin[FINC + 1];   // The input objects.
   Pobject* objs[FINC + 1];    // The source objects masked.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK); 
if (objs[0]->Type() == Po_Img2duc && objs[1]->Type() == Po_Img2duc) {
      Img2duc* const ims1 = (Img2duc*)objs[0];
      Img2duc* const ims2 = (Img2duc*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
if (objs[0]->Type() == Po_Img2dsl && objs[1]->Type() == Po_Img2dsl) {
      Img2dsl* const ims1 = (Img2dsl*)objs[0];
      Img2dsl* const ims2 = (Img2dsl*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
if (objs[0]->Type() == Po_Img2dsf && objs[1]->Type() == Po_Img2dsf) {
      Img2dsf* const ims1 = (Img2dsf*)objs[0];
      Img2dsf* const ims2 = (Img2dsf*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
if (objs[0]->Type() == Po_Img3duc && objs[1]->Type() == Po_Img3duc) {
      Img3duc* const ims1 = (Img3duc*)objs[0];
      Img3duc* const ims2 = (Img3duc*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
if (objs[0]->Type() == Po_Img3dsl && objs[1]->Type() == Po_Img3dsl) {
      Img3dsl* const ims1 = (Img3dsl*)objs[0];
      Img3dsl* const ims2 = (Img3dsl*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
if (objs[0]->Type() == Po_Img3dsf && objs[1]->Type() == Po_Img3dsf) {
      Img3dsf* const ims1 = (Img3dsf*)objs[0];
      Img3dsf* const ims2 = (Img3dsf*)objs[1];
       
      result = PSNR(*ims1, *ims2);
      goto end;
   }
  {
     PrintErrorFormat(objin, FINC, argv); 
     result = FAILURE; 
  }	

end:
  if (result) {
	WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, MASK); 
  }
  Exit(result); 
  return 0; 
}

#endif