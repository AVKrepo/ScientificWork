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
 * @author Abderrahim Elmoataz - 1996-09-09
 */

#include <math.h>
#include <pandore.h>
using namespace pandore;

/**
 * @file pmcmfitering.cpp
 * Mean Curvature Motion filtering.
 */

/**
 * @param t temperature.
 * iterations: iteration number.
 */
void Evolution( Img2dsf &ims, Img2dsf &imd, float deltaT, int iterations ) {
   Point2d p;
   float G, C;

   imd.Frame(ims, 1);
   for (int i = 0; i < iterations; i++) {
      for (p.y = 1; p.y < ims.Height() - 1; p.y++) {
	 for (p.x = 1; p.x < ims.Width() - 1; p.x++) {
	    float X = (ims[p + v8[4]] - ims[p + v8[0]]) / 2.0F;
	    float Y = (ims[p + v8[6]] - ims[p + v8[2]]) / 2.0F;
	    
	    float XX = ims[p + v8[4]] - 2 * ims[p] + ims[p + v8[0]];
	    float YY = ims[p + v8[6]] - 2 * ims[p] + ims[p + v8[2]];
	    float XY = (ims[p + v8[5]] - ims[p + v8[3]] - ims[p + v8[7]] + ims[p + v8[1]]) / 4.0F;

	    G = 1.0F + X*X + Y*Y; // Gradient.

	    if (G <= 0.00001) {
	       C = XX + YY;
	    } else {
	       C = ((1 + Y*Y) * XX -  2*X*Y*XY + (1 + X*X) * YY) / G;
	    }
	    imd[p] = ims[p] + deltaT * C;
	 }
      }
      ims = imd;
   }
}

void Evolution( Img3dsf &ims, Img3dsf &imd, float deltaT, int iterations ) {
   Point3d p;
   float G, C;
   imd.Frame(ims, 1);

   for (int i = 0; i < iterations; i++) {
      for (p.z = 1; p.z < ims.Depth() - 1; p.z++) {
	 for (p.y = 1; p.y < ims.Height() - 1; p.y++) {
	    for (p.x = 1; p.x < ims.Width() - 1 ;p.x++) {
	       float X = (ims[p.z][p.y][p.x+1]-ims[p.z][p.y][p.x-1])/2.0F;
	       float Y = (ims[p.z][p.y+1][p.x]-ims[p.z][p.y-1][p.x])/2.0F;
	       float Z = (ims[p.z+1][p.y][p.x]-ims[p.z-1][p.y][p.x])/(2.0F);
	       
	       float XX = ims[p.z][p.y][p.x+1]-2*ims[p.z][p.y][p.x]+ims[p.z][p.y][p.x-1];
	       float YY = ims[p.z][p.y+1][p.x]-2*ims[p.z][p.y][p.x]+ims[p.z][p.y-1][p.x];
	       float ZZ = (ims[p.z+1][p.y][p.x]-2*ims[p.z][p.y][p.x]+ims[p.z-1][p.y][p.x])/4.0F;
	       
	       float XY = (ims[p.z][p.y+1][p.x+1]-ims[p.z][p.y-1][p.x+1]-ims[p.z][p.y+1][p.x-1]+ims[p.z][p.y-1][p.x-1]) / 4.0F;
	       float XZ = (ims[p.z+1][p.y][p.x+1]-ims[p.z-1][p.y][p.x+1]-ims[p.z+1][p.y][p.x-1]+ims[p.z-1][p.y][p.x-1]) / (4.0F);
	       float YZ = (ims[p.z+1][p.y+1][p.x]-ims[p.z+1][p.y-1][p.x]-ims[p.z-1][p.y+1][p.x]+ims[p.z-1][p.y-1][p.x]) / (4.0F);

	       G = (float)(1 + (double)(X*X) + (double)(Y*Y) + (double)(Z*Z));
	       if (G <= 0.0000001) {
		  C = XX + YY + ZZ;
	       } else  {
		  C = (float)((1+(double)(Y*Y) + (double)(Z*Z)) * XX + (1 + (double)(X*X) + (double)(Z*Z)) * YY + (1 + (double)(X*X) + (double)(Y*Y)) * ZZ -  2*X*Y*XY -2*X*Z*XZ -2*Y*Z*YZ)/G;
	       }
	       imd[p] = (ims[p] + deltaT * C);
	    }
	 }
      }
      ims = imd;
   }
}

/**
 * iterations: iteration number.
 */
template  <typename T>
Errc PMcmFiltering2d( const Imx3d<T> &ims, Imx3d<T> &imd, int iterations ) {
   Img2dsf imsi(ims.Props());
   Img2dsf imdi(imd.Props());
   const float deltaT = 0.05f;
   iterations *= 5;

   for (int b = 0; b < ims.Bands(); b++) {
      T *ps = ims.Vector(b);
      T *pse = ims.Vector(b) + ims.VectorSize();
      Float *psi = imsi.Vector();
      for ( ; ps < pse; ps++, psi++) {
	 *psi = (Float)*ps;
      }
      Evolution(imsi, imdi, deltaT, iterations);
      T *pd = imd.Vector(b);
      T *pde = imd.Vector(b) + imd.VectorSize();
      Float *pdi = imdi.Vector();
      for ( ; pd < pde; pd++, pdi++) {
	 if (*pdi > Limits<T>::max()) {
	    *pd = Limits<T>::max();
	 } else {
	    *pd = (T)*pdi;
	 }
      }
   }
   return SUCCESS;
}

template  <typename T>
Errc PMcmFiltering3d( const Imx3d<T> &ims, Imx3d<T> &imd, int iterations ) {
   Img3dsf imsi(ims.Props());
   Img3dsf imdi(imd.Props());
   const float deltaT = 0.05f;
   iterations *= 5;
   
   for (int b = 0; b < ims.Bands(); b++) {
      T *ps = ims.Vector(b);
      T *pse = ims.Vector(b)+ims.VectorSize();
      Float *psi = imsi.Vector();
      for ( ; ps < pse; ps++, psi++) {
	 *psi = (Float)*ps;
      }
      Evolution(imsi, imdi, deltaT, iterations);

      T *pd = imd.Vector(b);
      T *pde = imd.Vector(b) + imd.VectorSize();
      Float *pdi=imdi.Vector();
      for ( ; pd < pde; pd++, pdi++) {
	 if (*pdi > Limits<T>::max()) {
	    *pd = Limits<T>::max();
	 } else {
	    *pd = (T)*pdi;
	 }
      }
   }
   return SUCCESS;
}



Errc PMcmFiltering( const Img2duc &ims, Img2duc &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Img2dsl &ims, Img2dsl &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Img2dsf &ims, Img2dsf &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc2duc &ims, Imc2duc &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc2dsl &ims, Imc2dsl &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc2dsf &ims, Imc2dsf &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering2d(ims, imd, iterations);
}
Errc PMcmFiltering( const Img3duc &ims, Img3duc &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}
Errc PMcmFiltering( const Img3dsl &ims, Img3dsl &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}
Errc PMcmFiltering( const Img3dsf &ims, Img3dsf &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc3duc &ims, Imc3duc &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc3dsl &ims, Imc3dsl &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}
Errc PMcmFiltering( const Imc3dsf &ims, Imc3dsf &imd, int iterations ) {
   if (iterations <= 0) {
      std::cerr << "Error pmcmfiltering: Bad iteration number: " << iterations<<std::endl;
      return FAILURE;
   }
   return PMcmFiltering3d(ims, imd, iterations);
}

#ifdef MAIN
#define	USAGE	"usage: %s iterations [-m mask] [im_in|-] [im_out|-]"
#define	PARC	1
#define	FINC	1
#define	FOUTC	1
#define	MASK	3 //  Unmasking only.

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
      Img2duc* const ims = (Img2duc*)objs[0];
      objd[0] = new Img2duc(ims->Props());
      Img2duc * const imd = (Img2duc*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsl) {
      Img2dsl* const ims = (Img2dsl*)objs[0];
      objd[0] = new Img2dsl(ims->Props());
      Img2dsl * const imd = (Img2dsl*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img2dsf) {
      Img2dsf* const ims = (Img2dsf*)objs[0];
      objd[0] = new Img2dsf(ims->Props());
      Img2dsf * const imd = (Img2dsf*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc2duc) {
      Imc2duc* const ims = (Imc2duc*)objs[0];
      objd[0] = new Imc2duc(ims->Props());
      Imc2duc * const imd = (Imc2duc*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc2dsl) {
      Imc2dsl* const ims = (Imc2dsl*)objs[0];
      objd[0] = new Imc2dsl(ims->Props());
      Imc2dsl * const imd = (Imc2dsl*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc2dsf) {
      Imc2dsf* const ims = (Imc2dsf*)objs[0];
      objd[0] = new Imc2dsf(ims->Props());
      Imc2dsf * const imd = (Imc2dsf*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img3duc) {
      Img3duc* const ims = (Img3duc*)objs[0];
      objd[0] = new Img3duc(ims->Props());
      Img3duc * const imd = (Img3duc*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img3dsl) {
      Img3dsl* const ims = (Img3dsl*)objs[0];
      objd[0] = new Img3dsl(ims->Props());
      Img3dsl * const imd = (Img3dsl*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Img3dsf) {
      Img3dsf* const ims = (Img3dsf*)objs[0];
      objd[0] = new Img3dsf(ims->Props());
      Img3dsf * const imd = (Img3dsf*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc3duc) {
      Imc3duc* const ims = (Imc3duc*)objs[0];
      objd[0] = new Imc3duc(ims->Props());
      Imc3duc * const imd = (Imc3duc*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc3dsl) {
      Imc3dsl* const ims = (Imc3dsl*)objs[0];
      objd[0] = new Imc3dsl(ims->Props());
      Imc3dsl * const imd = (Imc3dsl*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
      goto end;
   }
   if (objs[0]->Type() == Po_Imc3dsf) {
      Imc3dsf* const ims = (Imc3dsf*)objs[0];
      objd[0] = new Imc3dsf(ims->Props());
      Imc3dsf * const imd = (Imc3dsf*)objd[0];
      
      result = PMcmFiltering(*ims, *imd, atoi(parv[0]));
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
