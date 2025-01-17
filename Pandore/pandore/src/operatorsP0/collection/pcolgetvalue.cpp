/* -*- c-basic-offset: 3;mode: c++ -*-
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
 * @author R�gis Clouard - 2003-01-08
 */

#include <pandore.h>
using namespace pandore;

/**
 * @file pcolgetvalue.cpp
 * @brief Gets a specified value in a collection.
 *
 * Gets the value of the specified attribute.
 * @param name_in The name of attribute.
 * @param col_in_out The collection.
 * @return the attribute value.
 */
Errc PColGetValue( const Collection &col_in_out, const std::string &name_in ) { 
   Errc result;
   
   std::string type = col_in_out.GetType(name_in);
   if (type == "Char")
      result= col_in_out.GETVALUE(name_in, Char);
   else
   if (type == "Uchar")
      result= col_in_out.GETVALUE(name_in, Uchar);
   else
   if (type == "Short")
      result= col_in_out.GETVALUE(name_in, Short);
   else
   if (type == "Ushort")
      result= col_in_out.GETVALUE(name_in, Ushort);
   else
   if (type == "Long")
      result= col_in_out.GETVALUE(name_in, Long);
   else
   if (type == "Ulong")
      result= col_in_out.GETVALUE(name_in, Ulong);
   else
   if (type == "Llong")
      result= col_in_out.GETVALUE(name_in, Llong);
   else
   if (type == "Ullong")
      result= col_in_out.GETVALUE(name_in, Ullong);
   else
   if (type == "Float")
      result= col_in_out.GETVALUE(name_in, Float);
   else
   if (type == "Double")
      result= col_in_out.GETVALUE(name_in, Double);
   else
      return FAILURE;
   return result;
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
   Errc result;                // The result code of the execution.
   Pobject* mask;              // The region mask.
   Pobject* objin[FINC + 1];   // The input objects;
   Pobject* objs[FINC + 1];    // The source objects masked by the mask.
   Pobject* objout[FOUTC + 1]; // The output object.
   Pobject* objd[FOUTC + 1];   // The result object of the execution.
   char* parv[PARC + 1];       // The input parameters.
   
   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK);

   if (objs[0]->Type() == Po_Collection) {
      Collection* ims1 = (Collection*)objs[0];
      result = PColGetValue(*ims1, parv[0]);
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
