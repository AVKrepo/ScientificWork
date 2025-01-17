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

#include <pandore.h>
using namespace pandore;

/**
 * @file pboundaryregularization.cpp
 */
Errc PBoundaryRegularization( const Reg2d &rgs, Reg2d &rgd, int halfsize ) {
   if (halfsize > rgs.Width() / 2) {
      halfsize = rgs.Width() / 2;
   }
   if (halfsize > rgs.Height() / 2) {
      halfsize = rgs.Height() / 2;
   }

   Ulong * labels = new Ulong[halfsize * halfsize];
   Long * counts = new Long[halfsize * halfsize + 1];

   for (int y = 0; y < rgs.Height(); y++) {
      for (int x = 0; x < rgs.Width(); x++) {
	 counts[0] = 0;
	 for (int j = -halfsize; j <= halfsize; j++) {
	    if (y + j < 0) {
	       continue;
	    }
	    if (y + j>= rgs.Height()) {
	       break;
	    }
	    for (int i = -halfsize; i <= halfsize; i++) {
	       if (x + i < 0) {
		  continue;
	       }
	       if (x + i >= rgs.Width()) {
		  break;
	       }
	       Ulong v = rgs(y + j, x + i);
	       for (int l = 0; l < halfsize * halfsize; l++ ) {
		  if (counts[l] == 0) {
		     counts[l] = 1;
		     labels[l] = v;
		     counts[l+1] = 0;
		     break;
		  }
		  if (labels[l] == v) {
		     counts[l]++;
		     break;
		  }
	       }
	    }
	 }
	 int value = 0;
	 int maxCounts = 0;
	 for (int l = 0; counts[l] > 0; l++ ) {
	    if (counts[l] > maxCounts) {
	       value = labels[l];
	       maxCounts = counts[l];
	    }
	 }
	 rgd(y, x) = value;
      }
   }

   delete[] labels;
   delete[] counts;
   return SUCCESS;
}


Errc PBoundaryRegularization( const Reg3d &rgs, Reg3d &rgd, int halfsize ) {
   if (halfsize > rgs.Width() / 2) {
      halfsize = rgs.Width() / 2;
   }
   if (halfsize > rgs.Height() / 2) {
      halfsize = rgs.Height() / 2;
   }
   if (halfsize > rgs.Depth() / 2) {
      halfsize = rgs.Depth() / 2;
   }
   Ulong * labels = new Ulong[halfsize * halfsize * halfsize];
   Long * counts = new Long[halfsize * halfsize * halfsize + 1];

   for (int z = 0; z < rgs.Depth(); z++) {
      for (int y = 0; y < rgs.Height(); y++) {
	 for (int x = 0; x < rgs.Width(); x++) {
 	    counts[0] = 0;
  	    for (int k = -halfsize; k <= halfsize; k++) {
 	       if (z + k < 0) {
		  continue;
	       }
 	       if (z + k >= rgs.Depth()) {
		  break;
	       }
  	       for (int j = -halfsize; j <= halfsize; j++) {
 		  if (y + j < 0) {
		     continue;
		  }
 		  if (y + j >= rgs.Height()) {
		     break;
		  }
 		  for (int i  = -halfsize; i <= halfsize; i++) {
 		     if (x + i < 0) {
			continue;
		     }
 		     if (x + i>= rgs.Width()) {
			break;
		     }
 		     Ulong v = rgs(z + k, y + j, x + i);
  		     for (int l = 0; l < halfsize * halfsize * halfsize; l++ ) {
 			if (counts[l] == 0) {
 			   counts[l] = 1;
 			   labels[l] = v;
 			   counts[l + 1] = 0;
 			   break;
 			}
 			if (labels[l] == v) {
 			   counts[l]++;
 			   break;
			}
 		     }
  		  }
  	       }
  	    }
	    int value = 0;
	    int maxCounts = 0;
 	    for (int l = 0; counts[l] > 0; l++ ) {
	       if (counts[l] > maxCounts) {
		  value = labels[l];
		  maxCounts = counts[l];
	       }
	    }
	    rgd(z, y, x) = value;
	 }
      }
   }

   delete[] labels;
   delete[] counts;
   return SUCCESS;
}



/*
 * Modify only the following constants, and the function call.
 */
#ifdef MAIN
#define	USAGE	"usage: %s halfsize [-m mask] [im_in|-] [im_out|-]"
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
   if (objs[0]->Type() == Po_Reg2d) {
      Reg2d* const rgs = (Reg2d*)objs[0];
      objd[0] = new Reg2d(rgs->Props());
      Reg2d* const rgd = (Reg2d*)objd[0];
      
      result = PBoundaryRegularization(*rgs, *rgd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Reg3d) {
      Reg3d* const rgs = (Reg3d*)objs[0];
      objd[0] = new Reg3d(rgs->Props());
      Reg3d* const rgd = (Reg3d*)objd[0];
      
      result = PBoundaryRegularization(*rgs, *rgd, atoi(parv[0]));
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
