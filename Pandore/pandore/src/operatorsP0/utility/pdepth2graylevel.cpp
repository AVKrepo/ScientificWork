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
 * @author Regis Clouard - 2006-05-30
 */

#include <math.h>
#include <pandore.h>
using namespace pandore;

/**
 * @file pdepth2graylevel.cpp
 * Build 2D image from 3D image where depths of non nul pixel are
 * converted to gray levels.
 */

/**
 * Converts depth to gray level.
 */

Errc PDepth2Graylevel( const Img3duc &ims, Img2dsl &imd, Img3duc::ValueType threshold ) {
   imd=0;
   Point3d p;
   
   for (p.z=1; p.z<ims.Depth(); p.z++)
      for (p.y=0; p.y<ims.Height(); p.y++)
	 for (p.x=0; p.x<ims.Width(); p.x++) {
	    if (ims[p]>threshold && imd[p.y][p.x]==0) imd[p.y][p.x]=p.z;
	 }
   
   return SUCCESS;
}

Errc PDepth2Graylevel( const Img3dsl &ims, Img2dsl &imd, Img3dsl::ValueType threshold ) {
   imd=0;
   Point3d p;
   
   for (p.z=1; p.z<ims.Depth(); p.z++)
      for (p.y=0; p.y<ims.Height(); p.y++)
	 for (p.x=0; p.x<ims.Width(); p.x++) {
	    if (ims[p]>threshold && imd[p.y][p.x]==0) imd[p.y][p.x]=p.z;
	 }
   
   return SUCCESS;
}

Errc PDepth2Graylevel( const Img3dsf &ims, Img2dsl &imd, Img3dsf::ValueType threshold ) {
   imd=0;
   Point3d p;
   
   for (p.z=1; p.z<ims.Depth(); p.z++)
      for (p.y=0; p.y<ims.Height(); p.y++)
	 for (p.x=0; p.x<ims.Width(); p.x++) {
	    if (ims[p]>threshold && imd[p.y][p.x]==0) imd[p.y][p.x]=p.z;
	 }
   
   return SUCCESS;
}


#ifdef MAIN
#define	USAGE	"usage: %s threshold [im_in|-] [im_out|-]"
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
   if (objs[0]->Type() == Po_Img3duc) {
      Img3duc* const ims=(Img3duc*)objs[0];
      objd[0]=new Img2dsl(ims->Height(),ims->Width());
      Img2dsl* const imd=(Img2dsl*)objd[0];

      result=PDepth2Graylevel(*ims,*imd, (Img3duc::ValueType)atof(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img3dsl) {
      Img3dsl* const ims=(Img3dsl*)objs[0];
      objd[0]=new Img2dsl(ims->Height(),ims->Width());
      Img2dsl* const imd=(Img2dsl*)objd[0];

      result=PDepth2Graylevel(*ims,*imd, (Img3dsl::ValueType)atof(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img3dsf) {
      Img3dsf* const ims=(Img3dsf*)objs[0];
      objd[0]=new Img2dsl(ims->Height(),ims->Width());
      Img2dsl* const imd=(Img2dsl*)objd[0];

      result=PDepth2Graylevel(*ims,*imd, (Img3dsf::ValueType)atof(parv[0]));
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