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
 * @author Francois Angot - 1997-07-08
 * @author Regis Clouard - 2002-06-07
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file pextrude1d22d.cpp
 *
 */
#define AXEX 0
#define AXEY 1
#define AXEZ 2


Errc PExtrude1D22D( const Img1duc &ims, Img2d<Img1duc::ValueType> &imd, int axis, int length ) {
   Point2d p;
   switch (axis) {
       case AXEX:
	  imd.New(ims.Width(), length);
	  for (p.y = 0; p.y < imd.Height(); p.y++) {
	     for (p.x = 0; p.x < imd.Width(); p.x++) {
		imd(p.y, p.x) = ims(p.y);
	     }
	  }
	  break;
       case AXEY:
	  imd.New(length, ims.Width());
	  for (p.x = 0; p.x < imd.Width(); p.x++) {
	     for (p.y = 0; p.y < imd.Height(); p.y++) {
		imd(p.y, p.x) = ims(p.x);
	     }
	  }
	  break;
       default: {
	  std::cerr << "pextrude1d22d: bad parameter value for axis: "<< axis  << std::endl;
	  return FAILURE;
       }
   }
   
   return SUCCESS;
}

Errc PExtrude1D22D( const Img1dsl &ims, Img2d<Img1dsl::ValueType> &imd, int axis, int length ) {
   Point2d p;
   switch (axis) {
       case AXEX:
	  imd.New(ims.Width(), length);
	  for (p.y = 0; p.y < imd.Height(); p.y++) {
	     for (p.x = 0; p.x < imd.Width(); p.x++) {
		imd(p.y, p.x) = ims(p.y);
	     }
	  }
	  break;
       case AXEY:
	  imd.New(length, ims.Width());
	  for (p.x = 0; p.x < imd.Width(); p.x++) {
	     for (p.y = 0; p.y < imd.Height(); p.y++) {
		imd(p.y, p.x) = ims(p.x);
	     }
	  }
	  break;
       default: {
	  std::cerr << "pextrude1d22d: bad parameter value for axis: "<< axis  << std::endl;
	  return FAILURE;
       }
   }
   
   return SUCCESS;
}

Errc PExtrude1D22D( const Img1dsf &ims, Img2d<Img1dsf::ValueType> &imd, int axis, int length ) {
   Point2d p;
   switch (axis) {
       case AXEX:
	  imd.New(ims.Width(), length);
	  for (p.y = 0; p.y < imd.Height(); p.y++) {
	     for (p.x = 0; p.x < imd.Width(); p.x++) {
		imd(p.y, p.x) = ims(p.y);
	     }
	  }
	  break;
       case AXEY:
	  imd.New(length, ims.Width());
	  for (p.x = 0; p.x < imd.Width(); p.x++) {
	     for (p.y = 0; p.y < imd.Height(); p.y++) {
		imd(p.y, p.x) = ims(p.x);
	     }
	  }
	  break;
       default: {
	  std::cerr << "pextrude1d22d: bad parameter value for axis: "<< axis  << std::endl;
	  return FAILURE;
       }
   }
   
   return SUCCESS;
}


#ifdef MAIN
#define USAGE	"usage: %s axis length [im_in|- ] [im_out|-]"
#define PARC	2
#define FINC	1
#define FOUTC	1
#define MASK	0

int main( int argc, char *argv[] ) {
   Errc result;                // The result code of the execution.
   Pobject* mask;              // The region map.
   Pobject* objin[FINC + 1];   // The input objects.
   Pobject* objs[FINC + 1];    // The source objects masked.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK); 
   if (objs[0]->Type() == Po_Img1duc) {
      Img1duc* const ims = (Img1duc*)objs[0];
      objd[0] = new  Img2d<Img1duc::ValueType>;
      Img2d<Img1duc::ValueType>* const imd = (Img2d<Img1duc::ValueType>*)objd[0];
      result = PExtrude1D22D(*ims, *imd, atoi(parv[0]), atoi(parv[1]));
      goto end;
   }

   if (objs[0]->Type() == Po_Img1dsl) {
      Img1dsl* const ims = (Img1dsl*)objs[0];
      objd[0] = new  Img2d<Img1dsl::ValueType>;
      Img2d<Img1dsl::ValueType>* const imd = (Img2d<Img1dsl::ValueType>*)objd[0];
      result = PExtrude1D22D(*ims, *imd, atoi(parv[0]), atoi(parv[1]));
      goto end;
   }

   if (objs[0]->Type() == Po_Img1dsf) {
      Img1dsf* const ims = (Img1dsf*)objs[0];
      objd[0] = new  Img2d<Img1dsf::ValueType>;
      Img2d<Img1dsf::ValueType>* const imd = (Img2d<Img1dsf::ValueType>*)objd[0];
      result = PExtrude1D22D(*ims, *imd, atoi(parv[0]), atoi(parv[1]));
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
