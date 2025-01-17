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
 * @author Regis Clouard - 2002-12-11
 * @author Regis Clouard - 2004-02-09 (fix bug )
 */

#include <math.h>
#include <time.h>
#include <pandore.h>
using namespace pandore;

/**
 * @file pkmeans.cpp
 * K-means classification.
 */

/*
 * K is the number of class.
 * max is the number of iteration
 * input is the vector of characteristic of each item.
 * Output is a label map.
 */
void kmeans_( Char const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Uchar const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Short const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Ushort const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Long const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Ulong const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Llong const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Ullong const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Float const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}
void kmeans_( Double const*const* featureVectors, Long numberOfVectors, Long numberOfElements, Long *output, const int K, int maxIteration ) {
   double** means = new double*[K];
   means[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      means[i] = means[i - 1] + numberOfVectors;
   }

   double** partial_sums = new double*[K];
   partial_sums[0] = new double[K * numberOfVectors];
   for (int i = 1; i < K; i++) {
      partial_sums[i] = partial_sums[i - 1] + numberOfVectors;
   }
   Ulong *partial_nbrs = new Ulong[K];
   
   // Select first centroids (randomly).
   for (int i = 0; i < K; ++i) {
      int pos = (int)(((double)rand() / RAND_MAX) * (double)(numberOfElements - 1));
      for (int j = 0; j < numberOfVectors; ++j) {
	 means[i][j] = (double)featureVectors[j][pos];
      }
   }
   
   bool idempotent = false;
   while (!idempotent && --maxIteration >= 0) {
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; ++j) {
	    partial_sums[k][j] = 0.0;
	 }
	 partial_nbrs[k] = 0;
      }
      for (int i = 0; i < numberOfElements; ++i) {
	 double dist_min = MAXDOUBLE;
	 Long K_min = 0;
	 for (int k = 0; k < K; ++k) { // Find the nearest class.
	    double dist = 0.0;
	    for (int d = 0; d < numberOfVectors; ++d) {
	       double t = means[k][d] - (double)featureVectors[d][i];
	       dist += t * t;
	    }
	    if (dist < dist_min) {
	       dist_min = dist;
	       K_min = k;
	    }
	 }
	 ++partial_nbrs[K_min];
	 for (int d = 0; d < numberOfVectors; ++d) {
	    partial_sums[K_min][d] += featureVectors[d][i];
	 }
	 output[i] = 1 + K_min;
      }
      idempotent = true;
      
      for (int k = 0; k < K; k++) {
	 for (int j = 0; j < numberOfVectors; j++) {
	    if (partial_nbrs[k]) {
	       double ps = partial_sums[k][j] / (float)partial_nbrs[k];
	       if (ps != means[k][j]) {
		  idempotent = false;
	       }
	       means[k][j] = ps;
	    }
	 }
      }
   }
   
   delete[] means[0];
   delete[] means;	
   delete[] partial_sums[0];
   delete[] partial_sums;	
   delete[] partial_nbrs;
}

Errc PKmeans( const std::string& a_in, const Collection& c_in, const std::string& a_out, Collection& c_out, int numberOfClasses, int maxIteration ) {
   Long numberOfVectors; // Number of arrays in the collection.
   Long numberOfElements;
   std::string type;
   
   if (!c_in.NbOf(a_in, type, numberOfVectors, numberOfElements)) {
      return FAILURE;
   }
   if (numberOfElements < numberOfVectors) {
      fprintf(stderr, "Error pkmeans: Less elements than dimsension.");
      Exit(FAILURE);
   }
   // Get an array of each arrays.
   if (type == "Array:Char") {
      Char** in = c_in.GETNARRAYS(a_in, Char, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Uchar") {
      Uchar** in = c_in.GETNARRAYS(a_in, Uchar, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Short") {
      Short** in = c_in.GETNARRAYS(a_in, Short, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Ushort") {
      Ushort** in = c_in.GETNARRAYS(a_in, Ushort, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Long") {
      Long** in = c_in.GETNARRAYS(a_in, Long, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Ulong") {
      Ulong** in = c_in.GETNARRAYS(a_in, Ulong, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Llong") {
      Llong** in = c_in.GETNARRAYS(a_in, Llong, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Ullong") {
      Ullong** in = c_in.GETNARRAYS(a_in, Ullong, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Float") {
      Float** in = c_in.GETNARRAYS(a_in, Float, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
   if (type == "Array:Double") {
      Double** in = c_in.GETNARRAYS(a_in, Double, numberOfVectors, numberOfElements);
      Long* out = new Long[numberOfElements];	
      
      kmeans_(in, numberOfVectors, numberOfElements, out, numberOfClasses, maxIteration);
      c_out.SETARRAY(a_out, Long, out, numberOfElements);
   } else
      return FAILURE;
   return SUCCESS;
}

#ifdef MAIN

/*
 * Modify only the following constants, and the function call.
 */
#define	USAGE	"usage: %s in_attr out_attr k max_iteration [col_in|-] [col_out|-]"
#define	PARC	4
#define	FINC	1
#define	FOUTC	1
#define	MASK	0

int main( int argc, char *argv[] ) {
   Errc	 result;                // The result code of the execution.
   Pobject* mask;               // The region mask
   Pobject* objin[FINC + 1];    // The featureVectors objects;
   Pobject* objs[FINC + 1];     // The source objects masked by the mask.
   Pobject* objout[FOUTC + 1];  // The output object.
   Pobject* objd[FOUTC + 1];    // The result object of the execution.
   char* parv[PARC + 1];        // The input parameters.
   
   ReadArgs(argc, argv, PARC, FINC, FOUTC, &mask, objin, objs, objout, objd, parv, USAGE, MASK);
   
   srand((unsigned int)time(0));
   if (objs[0]->Type() == Po_Collection) {
      Collection* cols = (Collection*)objs[0];
      Collection* cold = new Collection;
      objd[0] = cold;
      
      result = PKmeans(argv[1], *cols, argv[2], *cold, atoi(argv[3]), atoi(argv[4]));
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
