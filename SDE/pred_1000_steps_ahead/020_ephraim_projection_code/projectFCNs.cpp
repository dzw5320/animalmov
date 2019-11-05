#include <RcppArmadillo.h>

using namespace Rcpp;

// [[Rcpp::depends(RcppArmadillo)]]


//////////////////////////////////////////////////////////
////
//// Function to check if two line segments overlap
////
////  public domain function by Darel Rex Finley, 2006
/////////////////////////////////////////////////////////


//  Determines the intersection point of the line defined by points A and B with the
//  line defined by points C and D.
//
//  Returns YES if the intersection point was found, and stores that point in X,Y.
//  Returns NO if there is no determinable intersection point, in which case X,Y will
//  be unmodified.


// [[Rcpp::export]]
int lineIntersection(
			 double Ax, double Ay,
			 double Bx, double By,
			 double Cx, double Cy,
			 double Dx, double Dy) {
  
  double  distAB, theCos, theSin, newX, ABpos, Xint, Yint ;

  //  Fail if either line segment is zero-length.
  //if ((Ax==Bx && Ay==By) || (Cx==Dx && Cy==Dy)) return 0;

  //  Fail if the segments share an end-point.
  //if ((Ax==Cx && Ay==Cy) || (Bx==Cx && By==Cy)
  //||  (Ax==Dx && Ay==Dy) || (Bx==Dx && By==Dy)) {
  //  return 0; }

  //  (1) Translate the system so that point A is on the origin.
  Bx-=Ax; By-=Ay;
  Cx-=Ax; Cy-=Ay;
  Dx-=Ax; Dy-=Ay;

  //  Discover the length of segment A-B.
  distAB=sqrt(Bx*Bx+By*By);

  //  (2) Rotate the system so that point B is on the positive X axis.
  theCos=Bx/distAB;
  theSin=By/distAB;
  newX=Cx*theCos+Cy*theSin;
  Cy  =Cy*theCos-Cx*theSin; Cx=newX;
  newX=Dx*theCos+Dy*theSin;
  Dy  =Dy*theCos-Dx*theSin; Dx=newX;

  //  Fail if segment C-D doesn't cross line A-B.
  if ((Cy<0. && Dy<0.) || (Cy>=0. && Dy>=0.)) return 0;

  //  (3) Discover the position of the intersection point along line A-B.
  ABpos=Dx+(Cx-Dx)*Dy/(Dy-Cy);

  //  Fail if segment C-D crosses line A-B outside of segment A-B.
  if (ABpos<0. || ABpos>distAB) return 0;

  //  (4) Apply the discovered position to line A-B in the original coordinate system.
  Xint=Ax+ABpos*theCos;
  Yint=Ay+ABpos*theSin;

  //  Success.
  return 1; }



//////////////////////////////////////////////////////////
////
//// Function to project onto the line segment
////   https://en.wikipedia.org/wiki/Vector_projection#Definitions_in_terms_of_a_and_b
/////////////////////////////////////////////////////////

// [[Rcpp::export]]
arma::mat project(arma::mat xy, arma::mat P, int index) {
  float w;
  arma::mat projection;
  arma::mat l=P.row(index+1) - P.row(index);
  w = min(NumericVector::create(1,dot(xy - P.row(index), l) / dot(l,l)));
  w = max(NumericVector::create(0.0, w));
  projection = P.row(index) + w * l;  // Projection falls on the segment
  // now put the projection just barely inside the polygon
  float eps = 1.0e-1;
  projection = projection+(projection-xy)*eps*dot(projection-xy,projection-xy);
  return projection;
}




