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
 * @file padaptivemeanfiltering.cpp
 * Filtre de lissage adaptatif.
 * Affecter comme valeur du pixel courant 
 * la moyenne des voisins du pixel voisin
 * de valeur de gradient minimale.
 */

#include <pandore.h>
using namespace pandore;


/*
 * Modify only the following constants, and the function call.
 */
Errc PAdaptiveMeanFiltering( const Img2duc &ims, Img2duc &ima, Img2duc &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2duc::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2duc::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}

Errc PAdaptiveMeanFiltering( const Img2duc &ims, Img2dsl &ima, Img2duc &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2duc::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2duc::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}

Errc PAdaptiveMeanFiltering( const Img2dsl &ims, Img2duc &ima, Img2dsl &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2dsl::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2dsl::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}

Errc PAdaptiveMeanFiltering( const Img2dsl &ims, Img2dsl &ima, Img2dsl &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2dsl::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2dsl::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}

Errc PAdaptiveMeanFiltering( const Img2dsf &ims, Img2duc &ima, Img2dsf &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2dsf::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2dsf::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}

Errc PAdaptiveMeanFiltering( const Img2dsf &ims, Img2dsl &ima, Img2dsf &imd, Uchar connexity ) {
   Point2d p, p2;
   int v, w;
   Float m;
  
   // Recopie des bords de l'image source dans l'image destination
   imd.Frame(ims, 2, 2);
   
   if (connexity == 4) {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 4; v++) {
	    if (ima[p + v4[v]] < ima[p + v4[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v4[w];
	 m = 0;
	 for (v = 0; v<4; v++) {
	    m += ims[p2 + v4[v]];
	 }
	 imd[p] = (Img2dsf::ValueType)(m / 4);
      }
   } else {
   for (p.y = 2; p.y < ims.Height() - 2; p.y++)
   for (p.x = 2; p.x < ims.Width() - 2; p.x++)
      {
	 for (w = 0, v = 0; v < 8; v++) {
	    if (ima[p + v8[v]] < ima[p + v8[w]]) {
	       w = v;
	    }
	 }
	 p2 = p + v8[w];
	 m = 0;
	 for (v = 0; v < 8; v++) {
	    m += ims[p2 + v8[v]];
	 }
	 imd[p] = (Img2dsf::ValueType)(m / 8);
      }
   }
   
   return SUCCESS;
}


/*
 * Modify only the following constants, and the function call.
 */
#ifdef MAIN
#define	USAGE	"usage: %s connexity [-m mask] [im_in|-] [im_amp|-] [im_out|-]"
#define	PARC	1
#define	FINC	2
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
   if (objs[0]->Type() == Po_Img2duc && objs[1]->Type() == Po_Img2duc) {
      Img2duc* const ims1 = (Img2duc*)objs[0];
      Img2duc* const ims2 = (Img2duc*)objs[1];
      objd[0] = new Img2duc(ims1->Size());
      Img2duc* const imd = (Img2duc*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
      goto end;
   }
   if (objs[0]->Type() == Po_Img2duc && objs[1]->Type() == Po_Img2dsl) {
      Img2duc* const ims1 = (Img2duc*)objs[0];
      Img2dsl* const ims2 = (Img2dsl*)objs[1];
      objd[0] = new Img2duc(ims1->Size());
      Img2duc* const imd = (Img2duc*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsl && objs[1]->Type() == Po_Img2duc) {
      Img2dsl* const ims1 = (Img2dsl*)objs[0];
      Img2duc* const ims2 = (Img2duc*)objs[1];
      objd[0] = new Img2dsl(ims1->Size());
      Img2dsl* const imd = (Img2dsl*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsl && objs[1]->Type() == Po_Img2dsl) {
      Img2dsl* const ims1 = (Img2dsl*)objs[0];
      Img2dsl* const ims2 = (Img2dsl*)objs[1];
      objd[0] = new Img2dsl(ims1->Size());
      Img2dsl* const imd = (Img2dsl*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsf && objs[1]->Type() == Po_Img2duc) {
      Img2dsf* const ims1 = (Img2dsf*)objs[0];
      Img2duc* const ims2 = (Img2duc*)objs[1];
      objd[0] = new Img2dsf(ims1->Size());
      Img2dsf* const imd = (Img2dsf*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsf && objs[1]->Type() == Po_Img2dsl) {
      Img2dsf* const ims1 = (Img2dsf*)objs[0];
      Img2dsl* const ims2 = (Img2dsl*)objs[1];
      objd[0] = new Img2dsf(ims1->Size());
      Img2dsf* const imd = (Img2dsf*)objd[0];
      
      result = PAdaptiveMeanFiltering(*ims1, *ims2, *imd, (Uchar)atoi(parv[0]));
      
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