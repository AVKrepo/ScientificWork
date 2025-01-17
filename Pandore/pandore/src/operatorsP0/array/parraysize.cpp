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
 * @author Alexandre Duret-Lutz - 1999-11-02
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file parraysize.cpp
 * Returns the size of an array in a collection.
 */

Errc PArraySize( const Collection &in, const std::string &name  ) {
   std::string type = in.GetType(name);
   std::string s = type;
   
   s.resize(6);
   if (s == "PArray") {
      return in.Get(name)->NbrElements();
   }
   if (s != "Array:") {
      std::cerr << "Error parraysize: " << name << " is not an array" <<std::endl;
      return FAILURE;
   }
   if (type == "Array:Char") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Uchar") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Short") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Ushort") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Long") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Ulong") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Llong") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Ullong") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Float") {
      return in.Get(name)->NbrElements();
   }
   if (type == "Array:Double") {
      return in.Get(name)->NbrElements();
   }
      {
	 std::cerr << "Error parraysize: Invalid collection type" << std::endl;
	 return FAILURE;
      }
}

#ifdef MAIN

/*
 * Modify only the following constants, and the function call.
 */
#define	USAGE	"usage: %s name [col_in|-]"
#define	PARC	1
#define	FINC	1
#define	FOUTC	0
#define	MASK	0

int main( int argc, char *argv[] ) {
   Errc  result;               // The result code of the execution.
   Pobject* mask;              // The region mask.
   Pobject* objin[FINC + 1];   // The input objects;
   Pobject* objs[FINC + 1];    // The source objects masked by the mask.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.

   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK);

   if (objs[0]->Type() == Po_Collection) {
      Collection* ims1 = (Collection*)objs[0];
      Collection* imd1 = new Collection;
      objd[0] = imd1;
      Exit(PArraySize(*ims1, parv[0]));
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
