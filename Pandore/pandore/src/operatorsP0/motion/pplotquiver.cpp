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

//--------------------------------------------------------------------------
// IMPORTANT NOTICE :
//--------------------
// The source code of this Pandore operator is governed by a specific
// Free-Software License (the CeCiLL License), also applying to the
// CImg Library. Please read it carefully, if you want to use this module
// in your own project (file CImg.h).
// IN PARTICULAR, YOU ARE NOT ALLOWED TO USE THIS PANDORE MODULE IN A
// CLOSED-SOURCE PROPRIETARY PROJECT WITHOUT ASKING AN AUTHORIZATION
// TO THE CIMG LIBRARY AUTHOR ( http://www.greyc.ensicaen.fr/~dtschump/ )
//--------------------------------------------------------------------------

/**
 * @author David Tschumperlé - 2005-08-31
 */

#include <stdio.h>
#define cimg_OS 0
#define cimg_display_type 0
#include "CImg1-16.h"
using namespace cimg_library1_16;
#include <pandore.h>
using namespace pandore;

/**
 * @file pplotquiver
 * @brief Dessine un champ de vecteurs 2D.
 */

template<typename T>
Errc PPlotQuiver( const Imx2d<T> &ims, Img2duc &imd, Short sampling=20, Float factor=1 ) {
   CImg<> img(ims.Width(),ims.Height(),1,2);
   if (ims.Bands()<2) return FAILURE;
   cimg_mapXYV(img,x,y,k) img(x,y,k) = (float)ims[k][y][x];
   CImg<unsigned char> visu(imd.Width(),imd.Height());
   const unsigned char white = 255;
   visu.draw_quiver(img,&white,sampling,factor);
   cimg_mapXY(visu,x,y) imd[y][x] = visu(x,y);
   return SUCCESS;
}



#ifdef MAIN
#define USAGE   "usage: %s dimx dimy sampling factor [im_in|-] [im_out|-]"
#define	PARC	4
#define	FINC	1
#define	FOUTC	1
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
   if (objs[0]->Type()==Po_Imx2duc) {
      Imx2duc* const ims=(Imx2duc*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx2dsl) {
      Imx2dsl* const ims=(Imx2dsl*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx2dsf) {
      Imx2dsf* const ims=(Imx2dsf*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2duc) {
      Imc2duc* const ims=(Imc2duc*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2dsl) {
      Imc2dsl* const ims=(Imc2dsl*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2dsf) {
      Imc2dsf* const ims=(Imc2dsf*)objs[0];
      Short dimx = (Short)atoi(parv[0]), dimy = (Short)atoi(parv[1]);
      objd[0] = new Img2duc(dimy,dimx);
      Img2duc *imd = (Img2duc*)objd[0];
      result = PPlotQuiver(*ims,*imd,(Short)atoi(parv[2]),(Float)atof(parv[3]));
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
