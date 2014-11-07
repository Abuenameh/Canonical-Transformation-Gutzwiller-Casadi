#include <queue>

using namespace std;

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/progress.hpp>
#include <boost/random.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>
#include <boost/date_time.hpp>

using namespace boost;
using namespace boost::random;
using namespace boost::filesystem;
using namespace boost::posix_time;

#include "casadi.hpp"
#include "gutzwiller.hpp"
#include "mathematica.hpp"

typedef boost::array<double, L> Parameter;

double M = 1000;
double g13 = 2.5e9;
double g24 = 2.5e9;
double delta = 1.0e12;
double Delta = -2.0e10;
double alpha = 1.1e7;

double Ng = sqrt(M) * g13;

double JW(double W) {
    return alpha * (W * W) / (Ng * Ng + W * W);
}

double JWij(double Wi, double Wj) {
    return alpha * (Wi * Wj) / (sqrt(Ng * Ng + Wi * Wi) * sqrt(Ng * Ng + Wj * Wj));
}

Parameter JW(Parameter W) {
    Parameter v;
    for (int i = 0; i < L; i++) {
        v[i] = W[i] / sqrt(Ng * Ng + W[i] * W[i]);
    }
    Parameter J;
    for (int i = 0; i < L - 1; i++) {
        J[i] = alpha * v[i] * v[i + 1];
    }
    J[L - 1] = alpha * v[L - 1] * v[0];
    return J;
}

double UW(double W) {
    return -2 * (g24 * g24) / Delta * (Ng * Ng * W * W) / ((Ng * Ng + W * W) * (Ng * Ng + W * W));
}

Parameter UW(Parameter W) {
    Parameter U;
    for (int i = 0; i < L; i++) {
        U[i] = -2 * (g24 * g24) / Delta * (Ng * Ng * W[i] * W[i]) / ((Ng * Ng + W[i] * W[i]) * (Ng * Ng + W[i] * W[i]));
    }
    return U;
}

boost::mutex progress_mutex;
boost::mutex points_mutex;

struct Point {
    double x;
    double mu;
};

struct PointResults {
    double W;
    double mu;
    double E0;
    double Eth;
    double E2th;
    double fs;
    vector<double> J;
    vector<double> U;
    double fmin;
    vector<double> fn0;
    vector<double> fmax;
    vector<double> f0;
    vector<double> fth;
    vector<double> f2th;
};

vector<double> norm(vector<double>& x) {
    const complex<double> * f[L];
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<const doublecomplex*> (&x[2 * i * dim]);
    }

//    norms.resize(L);
    
    vector<double> norms(L);

    //    double norm = 1;
    for (int i = 0; i < L; i++) {
        double normi = 0;
        for (int n = 0; n <= nmax; n++) {
            normi += norm(f[i][n]);
        }
        //        norm *= normi;
        norms[i] = sqrt(normi);
    }
        return norms;
//    return 0;
}

void phasepoints(Parameter& xi, phase_parameters pparms, queue<Point>& points, vector<PointResults>& pres, progress_display& progress) {

    int ndim = 2 * L * dim;

    vector<double> x(ndim);
    doublecomplex * f[L];
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<doublecomplex*> (&x[2 * i * dim]);
    }

    vector<double> U(L), J(L), dU(L);

    vector<double> x0(ndim), xth(ndim), x2th(ndim);
    doublecomplex * f0[L];
    for (int i = 0; i < L; i++) {
        f0[i] = reinterpret_cast<doublecomplex*> (&x0[2 * i * dim]);
    }

    vector<double> xabs(ndim / 2);
    double* fabs[L];
    for (int i = 0; i < L; i++) {
        fabs[i] = &xabs[i * dim];
    }

    vector<double> fn0(L);
    vector<double> fmax(L);

    vector<double> norms(L);

    double theta = pparms.theta;

    double scale = 1;
    
    GroundStateProblem prob;

    for (;;) {
        Point point;
        {
            boost::mutex::scoped_lock lock(points_mutex);
            if (points.empty()) {
                break;
            }
            point = points.front();
            points.pop();
        }

        PointResults pointRes;
        pointRes.W = point.x;
        pointRes.mu = point.mu;

        double W[L];
        for (int i = 0; i < L; i++) {
            W[i] = xi[i] * point.x;
        }
        double U0 = 1 / scale;
        for (int i = 0; i < L; i++) {
            U[i] = UW(W[i]) / UW(point.x) / scale;
            dU[i] = U[i] - U0;
            J[i] = JWij(W[i], W[mod(i + 1)]) / UW(point.x) / scale;
        }
        pointRes.J = J;
        pointRes.U = U;

        prob.setParameters(U0, dU, J, point.mu / scale);
        
        prob.setTheta(0);
        
        double E0;
        try {
            E0 = prob.solve(x0);
        } catch (std::exception& e) {
            printf("Ipopt failed for E0 at %f, %f\n", point.x, point.mu);
            cout << e.what() << endl;
            E0 = numeric_limits<double>::quiet_NaN();
        }
        
        norms = norm(x0);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x0[2 * (i * dim + n)] /= norms[i];
                x0[2 * (i * dim + n) + 1] /= norms[i];
            }
            transform(f0[i], f0[i] + dim, fabs[i], std::ptr_fun<const doublecomplex&, double>(abs));
            fmax[i] = *max_element(fabs[i], fabs[i] + dim);
            fn0[i] = fabs[i][1];
        }
        
        pointRes.fmin = *min_element(fn0.begin(), fn0.end());
        pointRes.fn0 = fn0;
        pointRes.fmax = fmax;
        pointRes.f0 = x0;
        pointRes.E0 = E0;

        prob.setTheta(theta);
        
        double Eth;
        try {
            Eth = prob.solve(xth);
        } catch (std::exception& e) {
            printf("Ipopt failed for Eth at %f, %f\n", point.x, point.mu);
            cout << e.what() << endl;
            Eth = numeric_limits<double>::quiet_NaN();
        }
        
        norms = norm(xth);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                xth[2 * (i * dim + n)] /= norms[i];
                xth[2 * (i * dim + n) + 1] /= norms[i];
            }
        }
            
        pointRes.fth = xth;
        pointRes.Eth = Eth;

        prob.setTheta(2*theta);
        
        double E2th;
        try {
            E2th = prob.solve(x2th);
        } catch (std::exception& e) {
            printf("Ipopt failed for E2th at %f, %f\n", point.x, point.mu);
            cout << e.what() << endl;
            E2th = numeric_limits<double>::quiet_NaN();
        }
        
        norms = norm(x2th);
        for (int i = 0; i < L; i++) {
            for (int n = 0; n <= nmax; n++) {
                x2th[2 * (i * dim + n)] /= norms[i];
                x2th[2 * (i * dim + n) + 1] /= norms[i];
            }
        }
            
        pointRes.f2th = x2th;
        pointRes.E2th = E2th;

        pointRes.fs = (E2th - 2 * Eth + E0) / (L * theta * theta);

        {
            boost::mutex::scoped_lock lock(points_mutex);
            pres.push_back(pointRes);
        }

        {
            boost::mutex::scoped_lock lock(progress_mutex);
            ++progress;
        }
    }

}


int main(int argc, char** argv) {
    GroundStateProblem prob;
//
////    cout << prob.getE() << endl;
////    cout << prob.subst() << endl;
//    vector<double> dU(L, 0);
//    vector<double> J(L, 0.01);
//    prob.setParameters(1, dU, J, 0.5);
//    prob.setTheta(0);
//    vector<double> f_(2*L*dim, 1);
//    cout << ::math(prob.call(f_)) << endl;
//    return 0;
////    vector<double> f;
//    vector<double> f;
//    double E = prob.solve(f);
////    prob.solve();
//    cout << E << endl;
//    cout << str(f) << endl;
//    cout << prob.getEtheta() << endl;
    

    mt19937 rng;
    uniform_real_distribution<> uni(-1, 1);

    int seed = lexical_cast<int>(argv[1]);
    int nseed = lexical_cast<int>(argv[2]);

    double xmin = lexical_cast<double>(argv[3]);
    double xmax = lexical_cast<double>(argv[4]);
    int nx = lexical_cast<int>(argv[5]);

    deque<double> x(nx);
    if (nx == 1) {
        x[0] = xmin;
    } else {
        double dx = (xmax - xmin) / (nx - 1);
        for (int ix = 0; ix < nx; ix++) {
            x[ix] = xmin + ix * dx;
        }
    }

    double mumin = lexical_cast<double>(argv[6]);
    double mumax = lexical_cast<double>(argv[7]);
    int nmu = lexical_cast<int>(argv[8]);

    deque<double> mu(nmu);
    if (nmu == 1) {
        mu[0] = mumin;
    } else {
        double dmu = (mumax - mumin) / (nmu - 1);
        for (int imu = 0; imu < nmu; imu++) {
            mu[imu] = mumin + imu * dmu;
        }
    }

    double D = lexical_cast<double>(argv[9]);
    double theta = lexical_cast<double>(argv[10]);

    int numthreads = lexical_cast<int>(argv[11]);

    int resi = lexical_cast<int>(argv[12]);

    double Wthresh = lexical_cast<double>(argv[13]);

    bool canonical = lexical_cast<bool>(argv[14]);

#ifdef AMAZON
    //    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/home/ubuntu/Dropbox/Amazon EC2/Simulation Results/Canonical Transformation Gutzwiller");
#else
    //    path resdir("/Users/Abuenameh/Dropbox/Amazon EC2/Simulation Results/Gutzwiller Phase Diagram");
    path resdir("/Users/Abuenameh/Documents/Simulation Results/Canonical Transformation Gutzwiller");
#endif
    if (!exists(resdir)) {
        cerr << "Results directory " << resdir << " does not exist!" << endl;
        exit(1);
    }
    for (int iseed = 0; iseed < nseed; iseed++, seed++) {
        ptime begin = microsec_clock::local_time();


        ostringstream oss;
        oss << "res." << resi << ".txt";
        path resfile = resdir / oss.str();
        while (exists(resfile)) {
            resi++;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }
        if (seed < 0) {
            resi = seed;
            oss.str("");
            oss << "res." << resi << ".txt";
            resfile = resdir / oss.str();
        }

        Parameter xi;
        xi.fill(1);
        //        xi.assign(1);
        rng.seed(seed);

        int xiset = 0;
        double threshold = 0;

        while (true) {
            if (seed > -1) {
                for (int j = 0; j < L; j++) {
                    xi[j] = (1 + D * uni(rng));
                }
            }

            double W[L]; //, U[L], J[L];
            vector<double> U(L), J(L);
            for (int i = 0; i < L; i++) {
                W[i] = xi[i] * Wthresh;
            }
            for (int i = 0; i < L; i++) {
                U[i] = UW(W[i]) / UW(xmax);
                J[i] = JWij(W[i], W[mod(i + 1)]) / UW(xmax);
            }
            bool reject = false;
            //        for (int i = 0; i < L; i++) {
            //            double threshold = 1.2*(JWij(xmax,xmax)/UW(xmax));
            //                if (J[i]/U[i] > threshold || J[mod(i-1)]/U[i] > threshold) {
            //                    iseed--;
            //                    seed++;
            //                    reject = true;
            //                    break;
            //                }
            //        }
            for (int i = 0; i < L; i++) {
                U[i] = UW(W[i]) / UW(Wthresh);
            }
            for (int i = 0; i < L; i++) {
                int j1 = mod(i - 1);
                int j2 = mod(i + 1);
                for (int n = 0; n < nmax; n++) {
                    for (int m = 1; m <= nmax; m++) {
                        if (n != m - 1) {
                            if (fabs(eps(U, i, j1, n, m)) < threshold || fabs(eps(U, i, j2, n, m)) < threshold) {
                                reject = true;
                                break;
                            }
                        }
                    }
                    if (reject) {
                        break;
                    }
                }
                if (reject) {
                    break;
                }
            }
            //        if (reject) {
            //            iseed--;
            //            seed++;
            //            continue;
            //        }
            if (!reject) {
                break;
            }
            xiset++;
        }

//        int Lres = L;
//        int nmaxres = nmax;

        boost::filesystem::ofstream os(resfile);
        printMath(os, "canonical", resi, canonical);
        printMath(os, "Lres", resi, L);
        printMath(os, "nmaxres", resi, nmax);
        printMath(os, "seed", resi, seed);
        printMath(os, "theta", resi, theta);
        printMath(os, "Delta", resi, D);
        printMath(os, "xres", resi, x);
        printMath(os, "mures", resi, mu);
        printMath(os, "xires", resi, xi);
        printMath(os, "xiset", resi, xiset);
        printMath(os, "threshold", resi, threshold);
        os << flush;

        cout << "Res: " << resi << endl;

        double muwidth = 0.1;
        queue<Point> points;
        bool sample = true;
        if (sample) {
            for (int ix = 0; ix < nx; ix++) {
                //            double mu0 = x[ix] / 1e12 + 0.05;
                double mu0 = 7.142857142857143e-13 * x[ix] + 0.08571428571428572;
                double mui = max(mumin, mu0 - muwidth);
                double muf = min(mumax, mu0 + muwidth);
                deque<double> mu(nmu);
                if (nmu == 1) {
                    mu[0] = mui;
                } else {
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = x[ix];
                    point.mu = mu[imu];
                    points.push(point);
                }
            }
            for (int ix = 0; ix < nx; ix++) {
                //            double mu0 = -3*x[ix] / 1e12 + 0.96;
                double mu0 = -2.142857142857143e-12 * x[ix] + 0.942857142857143;
                double mui = max(mumin, mu0 - muwidth);
                double muf = min(mumax, mu0 + muwidth);
                deque<double> mu(nmu);
                if (nmu == 1) {
                    mu[0] = mui;
                } else {
                    double dmu = (muf - mui) / (nmu - 1);
                    for (int imu = 0; imu < nmu; imu++) {
                        mu[imu] = mui + imu * dmu;
                    }
                }
                for (int imu = 0; imu < nmu; imu++) {
                    Point point;
                    point.x = x[ix];
                    point.mu = mu[imu];
                    points.push(point);
                }
            }
        } else {
            for (int imu = 0; imu < nmu; imu++) {
                for (int ix = 0; ix < nx; ix++) {
                    Point point;
                    point.x = x[ix];
                    point.mu = mu[imu];
                    points.push(point);
                }
            }
        }
        progress_display progress(points.size());

        phase_parameters parms;
        parms.theta = theta;
        parms.canonical = canonical;

        vector<PointResults> pointRes;

        thread_group threads;
        for (int i = 0; i < numthreads; i++) {
            //                        threads.emplace_back(phasepoints, std::ref(xi), theta, std::ref(points), std::ref(f0res), std::ref(E0res), std::ref(Ethres), std::ref(fsres), std::ref(progress));
            threads.create_thread(bind(&phasepoints, boost::ref(xi), parms, boost::ref(points), boost::ref(pointRes), boost::ref(progress)));
        }
        threads.join_all();

        vector<pair<double, double> > Wmu;
        vector<vector<double> > Js;
        vector<vector<double> > Us;
        vector<double> fs;
        vector<double> fmin;
        vector<vector<double> > fn0;
        vector<vector<double> > fmax;
        vector<vector<double> > f0;
        vector<vector<double> > fth;
        vector<vector<double> > f2th;
        vector<double> E0;
        vector<double> Eth;
        vector<double> E2th;

//        for (vector<PointResults>::iterator iter = pointRes.begin(); iter != pointRes.end(); ++iter) {
//            PointResults pres = *iter;
            for (PointResults pres : pointRes) {
            Wmu.push_back(make_pair(pres.W, pres.mu));
            Js.push_back(pres.J);
            Us.push_back(pres.U);
            fs.push_back(pres.fs);
            fmin.push_back(pres.fmin);
            fn0.push_back(pres.fn0);
            fmax.push_back(pres.fmax);
            f0.push_back(pres.f0);
            fth.push_back(pres.fth);
            f2th.push_back(pres.f2th);
            E0.push_back(pres.E0);
            Eth.push_back(pres.Eth);
            E2th.push_back(pres.E2th);
        }

        printMath(os, "Wmu", resi, Wmu);
        printMath(os, "Js", resi, Js);
        printMath(os, "Us", resi, Us);
        printMath(os, "fs", resi, fs);
        printMath(os, "fn0", resi, fn0);
        printMath(os, "fmin", resi, fmin);
        printMath(os, "fmax", resi, fmax);
        printMath(os, "f0", resi, f0);
        printMath(os, "fth", resi, fth);
        printMath(os, "f2th", resi, f2th);
        printMath(os, "E0", resi, E0);
        printMath(os, "Eth", resi, Eth);
        printMath(os, "E2th", resi, E2th);

        ptime end = microsec_clock::local_time();
        time_period period(begin, end);
        cout << endl << period.length() << endl << endl;

        os << "runtime[" << resi << "]=\"" << period.length() << "\";" << endl;
    }

    return 0;
}