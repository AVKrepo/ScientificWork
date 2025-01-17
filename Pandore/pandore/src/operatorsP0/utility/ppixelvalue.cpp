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
 * @author Fran�ois Angot - 2000-07-04
 * @author R�gis Clouard - 2001-04-27 (version 3.00)
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file ppixelvalue.cpp
 * Gets the pixel value at the specified coordinate.
 */
template <typename T>
Float PPixelValue( const Imx2d<T> &ims, Collection &col_out, Long y, Long x ) {
   if (y>=ims.Height() || x>=ims.Width() || y< 0 || x <0) {
      return FAILURE;
   }
   Float *t = new Float[ims.Bands()];
   for ( int b=0; b<ims.Bands(); b++ ) {
      t[b] = (Float)ims[b][y][x];
   }
   col_out.SETARRAY("internal_array", Float, t, ims.Bands());

   return t[0];
}


template <typename T>
Float PPixelValue( const Imx3d<T> &ims, Collection &col_out, Long z, Long y, Long x ) {
   if (z >= ims.Depth() || y>=ims.Height() || x>=ims.Width() || y<0 || x<0 || z<0) {
      return FAILURE;
   }

   Float *t = new Float[ims.Bands()];
   for ( int b=0; b<ims.Bands(); b++ ) {
      t[b] = (Float)ims[b][z][y][x];
   }
   col_out.SETARRAY("internal_array", Float, t, ims.Bands());
   return t[0];
}



Float PPixelValue( const Graph2d &grs, Collection & col_out, Long i ) {
   if (i<0 || i>grs.Size()) {
      return FAILURE;
   }
   Float *t = new Float[1];
   t[0] = Float(grs[i]->value);
   col_out.SETARRAY("internal_array", Float, t, 1);
   
   return t[0];
}

Float PPixelValue( const Graph3d &grs, Collection & col_out, Long i ) {
   if (i<0 || i>grs.Size()) {
      return FAILURE;
   }
   Float *t = new Float[1];
   t[0] = Float(grs[i]->value);
   col_out.SETARRAY("internal_array", Float, t, 1);
   
   return t[0];
}


#ifdef MAIN
#define	USAGE	"usage: %s x y z [-m mask] [im_in|-] [col_out|-]"
#define	PARC	3
#define	FINC	1
#define	FOUTC	1
#define	MASK	1

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
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Img2dsl) {
      Img2dsl* const ims=(Img2dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Img2dsf) {
      Img2dsf* const ims=(Img2dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2duc) {
      Imc2duc* const ims=(Imc2duc*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2dsl) {
      Imc2dsl* const ims=(Imc2dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc2dsf) {
      Imc2dsf* const ims=(Imc2dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx2duc) {
      Imx2duc* const ims=(Imx2duc*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx2dsl) {
      Imx2dsl* const ims=(Imx2dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx2dsf) {
      Imx2dsf* const ims=(Imx2dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Reg2d) {
      Reg2d* const ims=(Reg2d*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];

      result=PPixelValue(*ims, *cold, (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Img3duc) {
      Img3duc* const ims=(Img3duc*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Img3dsl) {
      Img3dsl* const ims=(Img3dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Img3dsf) {
      Img3dsf* const ims=(Img3dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc3duc) {
      Imc3duc* const ims=(Imc3duc*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc3dsl) {
      Imc3dsl* const ims=(Imc3dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imc3dsf) {
      Imc3dsf* const ims=(Imc3dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx3duc) {
      Imx3duc* const ims=(Imx3duc*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx3dsl) {
      Imx3dsl* const ims=(Imx3dsl*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Imx3dsf) {
      Imx3dsf* const ims=(Imx3dsf*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Reg3d) {
      Reg3d* const ims=(Reg3d*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[2]), (Long)atoi(parv[1]), (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Graph2d) {
      Graph2d* const ims=(Graph2d*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type()==Po_Graph3d) {
      Graph3d* const ims=(Graph3d*)objs[0];
      objd[0]=new Collection;
      Collection* const cold=(Collection*)objd[0];
      
      result=PPixelValue(*ims, *cold, (Long)atoi(parv[0]));
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
