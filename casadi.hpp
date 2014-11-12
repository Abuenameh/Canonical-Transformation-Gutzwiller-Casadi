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

//#include <pagmo/src/pagmo.h>
//
//using namespace pagmo;
//using namespace pagmo::algorithm;
//using namespace pagmo::problem;

#include "gutzwiller.hpp"

class GroundStateProblem {
public:
    GroundStateProblem();
    
//    problem::base_ptr clone() const { return problem::base_ptr(new GroundStateProblem(*this));s }
//    
//    void objfun_impl(fitness_vector& f, const decision_vector& x) const {
//        vector<double> grad;
//        double En = const_cast<GroundStateProblem*>(this)->E(x, grad);
//        f[0] = En;
//    }
    
    void setParameters(double U0, vector<double>& dU, vector<double>& J, double mu);
    void setTheta(double theta);
    
    double solve(vector<double>& f);
    
    double E(const vector<double>& f, vector<double>& grad);
    
    string& getStatus() { return status; }
    string getRuntime();
    
    void start() { start_time = microsec_clock::local_time(); }
    void stop() { stop_time = microsec_clock::local_time(); }
    
private:
    
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
    
    SXFunction Ef;
    Function Egradf;
    NlpSolver nlp;
    
    string status;
    double runtime;
};

//class energyprob : public problem::base {
//public:
//    energyprob(int n/*, GroundStateProblem* prob_*/) : base(n)/*, prob(prob_)*/ {}
//    problem::base_ptr clone() const { return problem::base_ptr(new energyprob(*this)); }
//    
//    void setProblem(GroundStateProblem* prob_) { prob = prob_; }
//    
//    void objfun_impl(fitness_vector& f, const decision_vector& x) const {
//        vector<double> grad;
//        double En = prob->E(x, grad);
//        f[0] = En;
//    }
//    
//private:
//    GroundStateProblem* prob;
//};

double energyfunc(const vector<double>& x, vector<double>& grad, void *data);

#endif	/* CASADI_HPP */

