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
 * @author R�gis Clouard - Jun 30, 2011
 */

#include <pandore.h>
using namespace pandore;

template <typename T>
bool Track( Img2d<T> &imd, Point2d pt, int w, int maxAngle, int maxLength ) {
   if (maxLength<0) { return false; }
   if (imd[pt]>0) {  return true; }

   for ( int angle = 0; angle <= maxAngle; angle ++ ) {
      int angle1 = (w+angle)%8;
      Point2d pt1 = pt+v8[angle1];
      if (Track(imd, pt1, angle1, maxAngle, maxLength-1)) {
	 imd[pt1]=255;
	 return true;
      }
      if (angle==0) {
	 continue;
      }
      int angle2 = (w-angle+8)%8;
      Point2d pt2 = pt+v8[angle2];
      if (Track(imd, pt2, angle2, maxAngle, maxLength-1)) {
	 imd[pt2]=255;
	 return true;
      }
   }
   return false;
}

/**
 * @file pedgeclosing.cpp
 * Fermeture de contours a partir de l'image d'amplitude du gradient.
 *
 * Parametres:	voisinage
 *		Longueur maximale de contours manquants.
 * Consult:	Par voisinage.
 * Fonction:	Marquer les bords de l'image comme contours.
 *		Suppression des points isoles.
 *		Pour chaque pixel terminal de contours
 *		Recherche des trois directions de poursuite.
 *		       Choix de la direction
 *			Maximum  du gradient si anticipation =1
 *			Critere de maximisation sur anticipation pixels.
 */

Errc PBlindEdgeClosing( const Img2duc &ims, Img2duc &imd, int maxAngle, int maxLength ) {
   if ((maxAngle > 2) || (maxAngle < 0)) {
      std::cerr << "Error pblindedgeclosing: Bad parameter value for angle: " << maxAngle << std::endl;
      return FAILURE;
   }

   imd=ims;
   // PEdgeClosing.
   Point2d pt;
   int nbpt=0;
   for (pt.y=0;  pt.y<imd.Height(); pt.y++) {
      for (pt.x=0; pt.x<imd.Width(); pt.x++) {

	 if ((imd[pt])) {
	    // Is it an end point
	    int nvois=0;
	    int w=0;
	    for (int v=0; v<8; v++) {
	       Point2d p1= pt+v8[v];
	       if (imd.Hold(p1) && imd[p1]) {
		  nvois++;
		  w = (v+4)%8; // keep the direction
	       }
	    }
	    
	    if (nvois == 1) {	// end point
	       imd[pt]=0;
	       if (Track(imd, pt, w, maxAngle, maxLength)) {
		  nbpt++;
	       }
	       imd[pt]=255;
	    }
	 }
      }
   }

   return nbpt;
}



#ifdef MAIN
#define USAGE	"usage: %s angle length [-m mask] [im_in|-] [im_out|-]"
#define PARC	2
#define FINC	1
#define FOUTC	1
#define MASK	1

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
      Img2duc* const ims=(Img2duc*)objs[0];
      objd[0]=new Img2duc(ims->Props());
      Img2duc* const imd=(Img2duc*)objd[0];
      
      result=PBlindEdgeClosing(*ims, *imd, atoi(parv[0]), atoi(parv[1]));
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
