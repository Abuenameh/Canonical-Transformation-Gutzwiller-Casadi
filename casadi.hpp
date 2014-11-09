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

#include <boost/date_time.hpp>

using namespace boost::posix_time;

#include "gutzwiller.hpp"

class GroundStateProblem {
public:
    GroundStateProblem();
    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
    void setTheta(double theta);
    
    double solve(vector<double>& f);
    
    double E(const vector<double>& f, vector<double>& grad);
    
    string& getStatus() { return status; }
//    double getRuntime() { return runtime; }
    string getRuntime();
    
    void start() { start_time = microsec_clock::local_time(); }
    void stop() { stop_time = microsec_clock::local_time(); }
    
//    double call(vector<double>& f);
    
private:
//    string frinName(int i, int n) { return "fr[" + to_string(i) + "][" + to_string(n) + "]"; }
//    string fiinName(int i, int n) { return "fi[" + to_string(i) + "][" + to_string(n) + "]"; }
//    string UName(int i) { return "U[" + to_string(i) + "]"; }
//    string dUName(int i) { return "dU[" + to_string(i) + "]"; }
//    string JName(int i) { return "J[" + to_string(i) + "]"; }
    
    ptime start_time;
    ptime stop_time;
    
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
    vector<SX> paramsx;
    
//    SX E;
//    SX Eparams;
//    SX Etheta;
    SXFunction Ef;
    Function Egradf;
    NlpSolver nlp;
    
    string status;
    double runtime;
};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

