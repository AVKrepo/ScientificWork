/* -*- c-basic-offset: 3; mode: c++ -*-
 *
 * Copyright (c) 2013,GREYC.
 * All rights reserved
 *
 * You may use this file under the terms of the BSD license as follows:
 *
 * "Redistribution and use in source and binary forms,with or without
 * modification,are permitted provided that the following conditions are
 * met:
 *   * Redistributions of source code must retain the above copyright
 *     notice,this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above copyright
 *     notice,this list of conditions and the following disclaimer in
 *     the documentation and/or other materials provided with the
 *     distribution.
 *   * Neither the name of the GREYC,nor the name of its
 *     contributors may be used to endorse or promote products
 *     derived from this software without specific prior written
 *     permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,INCLUDING,BUT NOT
 * LIMITED TO,THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,INDIRECT,INCIDENTAL, 
 * SPECIAL,EXEMPLARY,OR CONSEQUENTIAL DAMAGES (INCLUDING,BUT NOT
 * LIMITED TO,PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, 
 * DATA,OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY,WHETHER IN CONTRACT,STRICT LIABILITY,OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE,EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE."
 *
 * 
 * For more information,refer to:
 * https://clouard.users.greyc.fr/Pandore
 */

/**
 * @author R�gis Clouard - 1997-02-22
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file psubsampling.cpp
 * Sub-sampling of the image by the specified factor.
 *
 * @param factor along X axis.
 * @return SUCCESS or FAILURE
 * imd(i, j) = ims(i/samplingy, j/samplingx).
 */
template <typename T>
Errc PSubsampling( const Imx2d<T> &ims,Imx2d<T> &imd,const int factor ) {
   if ( factor <= 0 ) {
      fprintf(stderr, "Error psubsampling: Bad parameter values: %d\n", factor);
      return FAILURE;
   }

   for (int b = 0; b < imd.Bands(); b++) {
      Point2d p;

      for (p.y = 0 ; p.y < imd.Height(); p.y += factor) {
	 for (p.x = 0 ; p.x < imd.Width(); p.x += factor) {
	    float mean = 0.0F;
	    Point2d d;
	    for (d.y = 0; d.y < factor; d.y++) {
	       for (d.x = 0; d.x < factor; d.x++) {
		  mean += ims[b][p + d];
	       }
	    }
	    for (d.y = 0; d.y < factor; d.y++) {
	       for (d.x = 0; d.x < factor; d.x++) {
		  imd[b][p + d] = (T)(mean / (factor * factor));
	       }
	    }
	 }
      }
   }
   return SUCCESS;
}




#ifdef MAIN
#define	USAGE	"usage: %s factor [im_in|-] [im_out|-]"
#define	PARC	1
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
   if (objs[0]->Type() == Po_Img2duc) {
      const int factor = (int)atol(parv[0]);
      Img2duc* const ims = (Img2duc*)objs[0];
      objd[0] = new Img2duc(ims->Props());
      Img2duc* const imd = (Img2duc*)objd[0];
      result = PSubsampling(*ims, *imd, factor);
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsl) {
      const int factor = (int)atol(parv[0]);
      Img2dsl* const ims = (Img2dsl*)objs[0];
      objd[0] = new Img2dsl(ims->Props());
      Img2dsl* const imd = (Img2dsl*)objd[0];
      result = PSubsampling(*ims, *imd, factor);
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsf) {
      const int factor = (int)atol(parv[0]);
      Img2dsf* const ims = (Img2dsf*)objs[0];
      objd[0] = new Img2dsf(ims->Props());
      Img2dsf* const imd = (Img2dsf*)objd[0];
      result = PSubsampling(*ims, *imd, factor);
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
