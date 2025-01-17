/* -*- mode: c++; c-basic-offset: 3 -*-
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
 * @author Carlotti & Joguet (ENSEA) - 2000-05-15
 */

/**
 * @file pderichesmooting.cpp
 * Lissage causal et anticausal horizontal et
 * vertical d'une image par Deriche.
 */

#include <math.h>
#include <pandore.h>
using namespace pandore;

template <typename T>
Errc PDericheSmoothing( const Img2d<T>& ims, Img2d<T>& imd, Float alpha ) {
    // Constants
   const double e1a = exp(-alpha);
   const double e2a = exp(-2*alpha);
   const double C0 = ((1-e1a)*(1-e1a))/(1+2*alpha*e1a-e2a);
   const double b1 = -2*e1a;
   const double b2 = e2a;

   const double C1 = ((1-e1a)*(1-e1a))/e1a;
   const double a = -C1*e1a;
   const double a0 = C0;
   const double a1 = C0 * (alpha-1) * e1a;
   const double a2 = C0 * (alpha+1) * e1a;
   const double a3 = - C0 * e2a;
   
   int nbrows = ims.Height();
   int nbcols = ims.Width();
   int nbpix = nbrows * nbcols;

   const int nbrows_1 = nbrows - 1;
   const int nbrows_2 = nbrows - 2;
   const int nbrows_3 = nbrows - 3;
   const int nbcols_1 = nbcols - 1;
   const int nbcols_2 = nbcols - 2;
   const int nbcols_3 = nbcols - 3;

   register int x,y;
   int col_i = 0;
   int col_i1;
   int col_i2;
   
   float *nf_grx = new float[nbpix];
   float *nf_gry = new float[nbpix];

   float *inter1 = new float[nbpix];
   float *inter2 = new float[nbpix];
   float *inter3 = new float[nbpix];

   ////
   //// Y GRADIENT : L(x)*G(y).
   ////

   // 1. L(X): x smoothing

   // causal scanning
   for (y = 0; y < nbrows; ++y) {
      col_i = y*nbcols;
      col_i1 = col_i - 1;
      col_i2 = col_i - 2;

      inter1[col_i] = Float(a0*ims(y,0));
      inter1[col_i+1] = Float(a0*ims(y,1) + a1*ims(y,0) - b1*inter1[col_i]);
      for (x=2; x<nbcols; ++x) {
	 inter1[col_i + x] = Float(a0*ims(y,x) + a1*ims(y,x-1) - b1*inter1[col_i1+x] - b2*inter1[col_i2+x]);
      }
   }

   // anti-causal scanning
   for (y = 0; y < nbrows; ++y) {
      col_i = y*nbcols;
      col_i1 = col_i + 1;
      col_i2 = col_i + 2;
      // Initialization: no smoothing
      inter2[col_i + nbcols_1] = 0;
      inter2[col_i + nbcols_2] = Float(a2*ims(y,nbcols_1));
      for (x = nbcols_3; x >= 0; --x) {
	 inter2[col_i + x] = Float(a2*ims(y,x+1) + a3*ims(y,x+2) - b1*inter2[col_i1 + x] - b2*inter2[col_i2+x]);
      }
   }

   for (x=0; x<nbpix; ++x) {
      inter1[x] += inter2[x];
   }

   ////
   //// X GRADIENT : G(x).L(y)
   ////

   // top down
   for (x=0; x<nbcols; ++x) {
      inter2[x] = Float(a0*inter1[x]);
      inter2[nbcols + x] = Float(a*inter1[nbcols+x] + a1*inter1[x]);
      for (y=2; y<nbrows; ++y) {
	 inter2[y*nbcols + x] = Float(a0*inter1[y*nbcols + x] + a1*inter1[(y-1)*nbcols + x] - b1*inter2[(y-1)*nbcols + x] - b2*inter2[(y-2) * nbcols + x]);
      }
   }
   
   // bottom up
   for (x=0; x<nbcols; ++x) {
      // Initalization: no smoothing.
      inter3[nbrows_1 * nbcols + x] = 0;
      inter3[nbrows_2 * nbcols + x] = Float(a2*inter1[nbrows_1*nbcols+x]);
      for (y=nbrows_3; y>=0; --y) {
	 inter3[y*nbcols + x] = Float(a2*inter1[(y+1)*nbcols + x] + a3*inter1[(y+2)*nbcols + x] - b1*inter3[(y+1)*nbcols + x] - b2*inter3[(y+2)*nbcols + x]);
      }
   }

   T *pimd = imd.Vector();
   for (x = 0; x < nbpix; ++x) {
      *(pimd++) = (T)(inter2[x] + inter3[x]);
   }

   delete[] nf_grx;
   delete[] nf_gry;

   delete[] inter1;
   delete[] inter2;
   delete[] inter3;

   return SUCCESS;
}






#ifdef MAIN

#define	USAGE	"usage: %s alpha [-m mask] [im_in|-] [im_out|-]"
#define	PARC	1
#define	FINC	1
#define	FOUTC	1
#define	MASK	3

int main( int argc, char *argv[] ) {
   Errc result;                // The result code of the execution.
   Pobject* mask;              // The region map.
   Pobject* objin[FINC + 1];   // The input objects.
   Pobject* objs[FINC + 1];    // The source objects masked.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK); 
   if (objs[0]->Type()==Po_Img2duc) {
      Img2duc* const ims=(Img2duc*)objs[0];
      objd[0]=new Img2duc(ims->Size());		  
      Img2duc* const imd=(Img2duc*)objd[0];
      
      result=PDericheSmoothing(*ims, *imd, (Float)atof(parv[0]));
      goto end;
   }

   if (objs[0]->Type()==Po_Img2dsl) {
      Img2dsl* const ims=(Img2dsl*)objs[0];
      objd[0]=new Img2dsl(ims->Size());		  
      Img2dsl* const imd=(Img2dsl*)objd[0];
      
      result=PDericheSmoothing(*ims, *imd, (Float)atof(parv[0]));
      goto end;
   }

   if (objs[0]->Type()==Po_Img2dsf) {
      Img2dsf* const ims=(Img2dsf*)objs[0];
      objd[0]=new Img2dsf(ims->Size());		  
      Img2dsf* const imd=(Img2dsf*)objd[0];
      
      result=PDericheSmoothing(*ims, *imd, (Float)atof(parv[0]));
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
