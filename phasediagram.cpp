#include "casadi.hpp"
#include "gutzwiller.hpp"

int main(int argc, char** argv) {
    GroundStateProblem prob;

//    cout << prob.getE() << endl;
//    cout << prob.subst() << endl;
    vector<double> dU(L, 0);
    vector<double> J(L, 0.01);
    prob.setParameters(1, dU, J, 0.5);
    prob.setTheta(0);
    prob.solve();
//    cout << prob.getEtheta() << endl;
    
    return 0;
}