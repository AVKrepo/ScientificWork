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

/**
 * @author Regis Clouard - 2000-06-27
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file pimc2img.cpp
 * Converts 3 graylevels images to color image.
 */
template <typename T>
Errc PImgs2Imc( const Imx3d<T> &imr, const Imx3d<T> &img, const Imx3d<T> &imb, Imx3d<T> &imd ) {

   if ((imr.Size() != img.Size()) || (imr.Size() != imb.Size())) {
      std::cerr << "Error pimgs2imc: Input image with different size\n" << std::endl;
      return FAILURE;
   }
   T *psr = imr.Vector();
   T *psg = img.Vector();
   T *psb = imb.Vector();

   T *pdr = imd.Vector(0);
   T *pdg = imd.Vector(1);
   T *pdb = imd.Vector(2);

   for ( ; psr< imr.Vector() + imr.VectorSize(); ) {
      *pdr++ = *psr++;
      *pdg++ = *psg++;
      *pdb++ = *psb++;
   }

   imd.ColorSpace((PColorSpace)0);
   return SUCCESS;
}





#ifdef MAIN
#define	USAGE	"usage: %s [-m mask] [im_in1|-] [im_in2|-] [im_in3|-] [im_out|-]"
#define	PARC	0
#define	FINC	3
#define	FOUTC	1
#define	MASK	2

int main( int argc, char *argv[] ) {
   Errc result;                // The result code of the execution.
   Pobject* mask;              // The region map.
   Pobject* objin[FINC + 1];   // The input objects.
   Pobject* objs[FINC + 1];    // The source objects masked.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK); 
   if ((objs[0]->Type() == Po_Img2duc) && (objs[1]->Type() == Po_Img2duc) && (objs[2]->Type() == Po_Img2duc)) {
      Img2duc* const imr = (Img2duc*)objs[0];
      Img2duc* const img = (Img2duc*)objs[1];
      Img2duc* const imb = (Img2duc*)objs[2];
      objd[0] = new Imc2duc(imr->Size());
      Imc2duc* const imd = (Imc2duc*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
      goto end;
   }
   if ((objs[0]->Type() == Po_Img2dsl) && (objs[1]->Type() == Po_Img2dsl) && (objs[2]->Type() == Po_Img2dsl)) {
      Img2dsl* const imr = (Img2dsl*)objs[0];
      Img2dsl* const img = (Img2dsl*)objs[1];
      Img2dsl* const imb = (Img2dsl*)objs[2];
      objd[0] = new Imc2dsl(imr->Size());
      Imc2dsl* const imd = (Imc2dsl*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
      goto end;
   }
   if ((objs[0]->Type() == Po_Img2dsf) && (objs[1]->Type() == Po_Img2dsf) && (objs[2]->Type() == Po_Img2dsf)) {
      Img2dsf* const imr = (Img2dsf*)objs[0];
      Img2dsf* const img = (Img2dsf*)objs[1];
      Img2dsf* const imb = (Img2dsf*)objs[2];
      objd[0] = new Imc2dsf(imr->Size());
      Imc2dsf* const imd = (Imc2dsf*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
      goto end;
   }
   if ((objs[0]->Type() == Po_Img3duc) && (objs[1]->Type() == Po_Img3duc) && (objs[2]->Type() == Po_Img3duc)) {
      Img3duc* const imr = (Img3duc*)objs[0];
      Img3duc* const img = (Img3duc*)objs[1];
      Img3duc* const imb = (Img3duc*)objs[2];
      objd[0] = new Imc3duc(imr->Size());
      Imc3duc* const imd = (Imc3duc*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
      goto end;
   }
   if ((objs[0]->Type() == Po_Img3dsl) && (objs[1]->Type() == Po_Img3dsl) && (objs[2]->Type() == Po_Img3dsl)) {
      Img3dsl* const imr = (Img3dsl*)objs[0];
      Img3dsl* const img = (Img3dsl*)objs[1];
      Img3dsl* const imb = (Img3dsl*)objs[2];
      objd[0] = new Imc3dsl(imr->Size());
      Imc3dsl* const imd = (Imc3dsl*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
      goto end;
   }
   if ((objs[0]->Type() == Po_Img3dsf) && (objs[1]->Type() == Po_Img3dsf) && (objs[2]->Type() == Po_Img3dsf)) {
      Img3dsf* const imr = (Img3dsf*)objs[0];
      Img3dsf* const img = (Img3dsf*)objs[1];
      Img3dsf* const imb = (Img3dsf*)objs[2];
      objd[0] = new Imc3dsf(imr->Size());
      Imc3dsf* const imd = (Imc3dsf*)objd[0];
      
      result = PImgs2Imc(*imr,*img,*imb,*imd);
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