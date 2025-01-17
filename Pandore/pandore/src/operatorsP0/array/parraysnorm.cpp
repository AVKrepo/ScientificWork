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
 * @author Alexandre Duret-Lutz - 1999-11-03
 */

#include <pandore.h>
using namespace pandore;

#if (defined _MSC_VER) && (!defined snprintf)
#define snprintf _snprintf
#endif

/**
 * @file parraysnorm.cpp
 * Convert the type of value to double, and
 * normalize values between 0,1.
 */

Errc PArraysNorm( Collection &col, const std::string &name ) {
   char tmp[256];
   Long nbrcomp;
   std::string type;
   Long nin;

   if (!col.NbOf(name, type, nbrcomp, nin)) {
      std::cerr << "Error parraysnorm: Invalid collection" << std::endl;
      return FAILURE;
   }

   if (type == "Array:Char") {
      Char** in = col.GETNARRAYS(name, Char, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Char>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Uchar") {
      Uchar** in = col.GETNARRAYS(name, Uchar, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Uchar>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Short") {
      Short** in = col.GETNARRAYS(name, Short, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Short>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Ushort") {
      Ushort** in = col.GETNARRAYS(name, Ushort, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Ushort>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Long") {
      Long** in = col.GETNARRAYS(name, Long, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Long>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Ulong") {
      Ulong** in = col.GETNARRAYS(name, Ulong, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Ulong>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Llong") {
      Llong** in = col.GETNARRAYS(name, Llong, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Llong>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Ullong") {
      Ullong** in = col.GETNARRAYS(name, Ullong, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Ullong>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Float") {
      Float** in = col.GETNARRAYS(name, Float, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Float>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   if (type == "Array:Double") {
      Double** in = col.GETNARRAYS(name, Double, nbrcomp, nin);
      Double** out = new Double*[nbrcomp];
      for (int j = 0; j < nbrcomp; ++j) {
	 out[j] = new Double[nin];
	 for (int i = 0; i < nin; ++i) {
	    out[j][i] = (Double)in[j][i] / (Double)Limits<Double>::max();
	 }
	 snprintf(tmp, sizeof(tmp), "%s.%d", name.c_str(), j + 1);
	 tmp[sizeof(tmp) - 1] = 0;
	 col.SETARRAY(tmp, Double, out[j], nin);
      }
   } else
   {
      std::cerr << "Error parrysnorm: Invalid collection type" << std::endl;     
      return FAILURE;
   }
   return SUCCESS;
}

#ifdef MAIN


/*
 * Modify only the following constants, and the function call.
 */
#define	USAGE	"usage: %s name [col_in|-] [col_out|-]"
#define	PARC	1
#define	FINC	1
#define	FOUTC	1
#define	MASK	0

int main( int argc, char *argv[] ) {
   Errc	 result;               // The result code of the execution.
   Pobject* mask;              // The region map as mask.
   Pobject* objin[FINC + 1];   // The input objects;
   Pobject* objs[FINC + 1];    // The source objects masked by the mask.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];	     // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK);

   if (objs[0]->Type() == Po_Collection) {
      Collection* ims1 = (Collection*)objs[0];
      objd[0] = ims1;
      result = PArraysNorm(*ims1, parv[0]);
   } else {
      PrintErrorFormat(objin, FINC);
      result = FAILURE;
   }
   
   if (result) {
      WriteArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, MASK);
   }
   Exit(result);
   return 0;
}

#endif
