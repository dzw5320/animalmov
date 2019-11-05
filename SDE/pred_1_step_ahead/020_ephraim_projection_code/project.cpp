#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]


//////////////////////////////////////////////////////////
////
//// Function to project onto the line segment
////   https://en.wikipedia.org/wiki/Vector_projection#Definitions_in_terms_of_a_and_b
/////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::mat projpoly(arma::mat xy, arma::mat P, int index, float eps = 1.0e-6) {
  // xy = row of (x,y) locations - a 1x2 array
  // P = polygon with vertices in the rows
  // index = index of which edge of P to check 
  //
  float w;
  arma::mat projection;
  arma::mat l=P.row(index+1) - P.row(index);
  w = min(NumericVector::create(1,dot(xy - P.row(index), l) / dot(l,l)));
  w = max(NumericVector::create(0.0, w));
  projection = P.row(index) + w * l;  // Projection falls on the segment
  // now put the projection just barely inside the polygon
  // float eps = 1.0e-2;
  // projection = projection+(projection-xy)*eps*dot(projection-xy,projection-xy);
  projection = projection+(projection-xy)/sqrt(dot(projection-xy,projection-xy))*eps;
  return projection;
}

// [[Rcpp::export]]
int InPoly(arma::mat xy, arma::mat P) {
  // input: a 1x2 row vector of a locations
  xy=xy.t();
  int inpoly=0;
  int nvert=P.n_rows;
  for (int i = 0, j = nvert-1; i < nvert; j = i++) {
    if ( ((P(i,1)>xy(1,0)) != (P(j,1)>xy(1,0))) &&
	 (xy(0,0) < (P(j,0)-P(i,0)) * (xy(1,0)-P(i,1)) / (P(j,1)-P(i,1)) + P(i,0)) ){
      inpoly = !inpoly;
    }
  }
  return inpoly;
}


// [[Rcpp::export]]
arma::mat InP(arma::mat xy, arma::mat P) {
  // initialize T = number of time points
  int T=xy.n_rows;
  int nvert=P.n_rows;
  arma::mat inp(T,1);
  for (int i=0; i<T; i++){
    inp(i)=InPoly(xy.row(i),P);
  }
  return inp;
}


// [[Rcpp::export]]
arma::mat project(arma::mat xy, arma::mat P, float eps = 1.0e-6) {
  // initialize T = number of time points
  int T=xy.n_rows;
  int nvert=P.n_rows;
  int inp;
  double d;
  double dtmp;
  arma::mat xyProj(1,2);
  arma::mat xyOut(1,2);
  arma::mat xyTmp(1,2);
  for (int i=0; i<T; i++){
    d=std::numeric_limits<double>::infinity();
    // check if in or out of polygon
    inp=InPoly(xy.row(i),P);
    //if out of polygon, then project onto border
    if(inp==0){
      xyOut=xy.row(i);
      xyTmp=xy.row(i);
      for (int j=0; j < (nvert-1); j++){
	xyProj=projpoly(xyOut,P,j,eps);
	dtmp=pow(xyOut(0,0)-xyProj(0,0),2)+pow(xyOut(0,1)-xyProj(0,1),2);
	if(dtmp<d){
	  d=dtmp;
	  xyTmp=xyProj;
	}
      }
      xy.row(i)=xyTmp;
    }
  }
  return xy;
}
