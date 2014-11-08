/* 
 * File:   casadi.hpp
 * Author: Abuenameh
 *
 * Created on 06 November 2014, 17:45
 */

#ifndef CASADI_HPP
#define	CASADI_HPP

#include <casadi/casadi.hpp>

using namespace casadi;

#include "gutzwiller.hpp"

class GroundStateProblem {
public:
    GroundStateProblem();
    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
    void setTheta(double theta);
    
    double solve(vector<double>& f);
    
    string& getStatus() { return status; }
    double getRuntime() { return runtime; }
    
//    double call(vector<double>& f);
    
private:
//    string frinName(int i, int n) { return "fr[" + to_string(i) + "][" + to_string(n) + "]"; }
//    string fiinName(int i, int n) { return "fi[" + to_string(i) + "][" + to_string(n) + "]"; }
//    string UName(int i) { return "U[" + to_string(i) + "]"; }
//    string dUName(int i) { return "dU[" + to_string(i) + "]"; }
//    string JName(int i) { return "J[" + to_string(i) + "]"; }
    
    SX energy();
    
    vector<SX> fin;
    SX U0;
    vector<SX> dU;
    vector<SX> J;
    SX mu;
    SX theta;
    
    SX x;
    SX p;
    
    vector<double> params;
    
//    DMatrix lb;
//    DMatrix ub;
//    DMatrix x0;
    
    SX E;
    SX Eparams;
    SX Etheta;
    SXFunction Ef;
    NlpSolver nlp;
    
    string status;
    double runtime;
};



#endif	/* CASADI_HPP */

