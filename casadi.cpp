#include "casadi.hpp"

namespace casadi {
    inline bool isnan(SX& sx) {
//        cout << "isnan: " << sx << endl;
        return sx.at(0).isNan();
    }

    inline bool isinf(SX sx) {
//        cout << "isinf: " << sx << endl;
        return sx.at(0).isInf();
    }
}

GroundStateProblem::GroundStateProblem() {
    for (int i = 0; i < L; i++) {
        for (int n = 0; n <= nmax; n++) {
            fin.push_back(SX::sym(frinName(i, n)));
            fin.push_back(SX::sym(fiinName(i, n)));
        }
        U.push_back(SX::sym(UName(i)));
        dU.push_back(SX::sym(dUName(i)));
        J.push_back(SX::sym(JName(i)));
    }
    U0 = SX::sym("U0");
    mu = SX::sym("mu");
    
    E = energy();
}

SX GroundStateProblem::energy() {

    vector<complex<SX>* > f(L);
        vector<SX> norm2(L, 0);
    for (int i = 0; i < L; i++) {
        f[i] = reinterpret_cast<complex<SX>*> (&fin[2 * i * (nmax + 1)]);
                for (int n = 0; n <= nmax; n++) {
                    norm2[i] += f[i][n].real() * f[i][n].real() + f[i][n].imag() * f[i][n].imag();
                }
    }

    complex<SX> E;
    E = 0; //SX::zeros();

    complex<SX> Ei, Ej1, Ej2, Ej1j2, Ej1k1, Ej2k2, E1j1, E1j2, E2j1, E2j2, E3j1, E3j2, E4j1j2, E5j1k1, E5j2k1;
    for (int i = 0; i < L; i++) {

        int k1 = mod(i - 2);
        int j1 = mod(i - 1);
        int j2 = mod(i + 1);
        int k2 = mod(i + 2);

//        Ei = SX::zeros();
        Ei = 0;
        Ej1 = 0;
        Ej2 = 0;
        Ej1j2 = 0;
        Ej1k1 = 0;
        Ej2k2 = 0;
        for (int n = 0; n <= nmax; n++) {
            Ei += (0.5 * U[i] * n * (n - 1) - mu * n) * ~f[i][n] * f[i][n];
            
            if (n < nmax) {
                E1j1 += -J[j1] * expth * g(n, n + 1) * ~f[i][n + 1] * ~f[j1][n]
                        * f[i][n] * f[j1][n + 1];
                E1j2 += -J[i] * expmth * g(n, n + 1) * ~f[i][n + 1] * ~f[j2][n] * f[i][n]
                        * f[j2][n + 1];

                if (n > 0) {
                    E2j1 += 0.5 * J[j1] * J[j1] * exp2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j1][n - 1] * f[i][n - 1] * f[j1][n + 1];
                    E2j2 += 0.5 * J[i] * J[i] * expm2th * g(n, n) * g(n - 1, n + 1) * (1 / eps(U0, n, n))
                            * ~f[i][n + 1] * ~f[j2][n - 1] * f[i][n - 1] * f[j2][n + 1];
                }
                if (n < nmax - 1) {
                    E2j1 -= 0.5 * J[j1] * J[j1] * exp2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j1][n] * f[i][n] * f[j1][n + 2];
                    E2j2 -= 0.5 * J[i] * J[i] * expm2th * g(n, n + 2) * g(n + 1, n + 1) * (1 / eps(U0, n, n + 2))
                            * ~f[i][n + 2] * ~f[j2][n] * f[i][n] * f[j2][n + 2];
                }

                for (int m = 1; m <= nmax; m++) {
                    if (n != m - 1) {
                        E3j1 += 0.5 * J[j1] * J[j1] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * f[i][n + 1] * f[j1][m - 1] -
                                ~f[i][n] * ~f[j1][m] * f[i][n] * f[j1][m]);
                        E3j2 += 0.5 * J[i] * J[i] * g(n, m) * g(m - 1, n + 1) * (1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * f[i][n + 1] * f[j2][m - 1] -
                                ~f[i][n] * ~f[j2][m] * f[i][n] * f[j2][m]);

                        for (int p = 0; p < nmax; p++) {

                            if (p != n - 1 && 2 * n - m == p && n > 0) {
                                E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n - 1] * f[j1][m] * f[j2][p + 1];
                                E4j1j2 += 0.5 * J[j1] * J[i] * g(n, m) * g(n - 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 1] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n - 1] * f[j2][m] * f[j1][p + 1];
                            }
                            if (p != n + 1 && 2 * n - m == p - 2 && n < nmax - 1) {
                                E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j1][m - 1] * ~f[j2][p]
                                        * f[i][n] * f[j1][m] * f[j2][p + 1];
                                E4j1j2 -= 0.5 * J[j1] * J[i] * g(n, m) * g(n + 1, p + 1) * (1 / eps(U0, n, m))
                                        * ~f[i][n + 2] * ~f[j2][m - 1] * ~f[j1][p]
                                        * f[i][n] * f[j2][m] * f[j1][p + 1];
                            }

                        }

                        E5j1k1 += 0.5 * J[j1] * J[k1] * exp2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j1][m - 1] * ~f[k1][n]
                                * f[i][n] * f[j1][m - 1] * f[k1][n + 1] -
                                ~f[i][n + 1] * ~f[j1][m] * ~f[k1][n]
                                * f[i][n] * f[j1][m] * f[k1][n + 1]);
                        E5j2k2 += 0.5 * J[j2] * J[i] * expm2th * g(n, m) * g(m - 1, n + 1)*(1 / eps(U0, n, m))
                                * (~f[i][n + 1] * ~f[j2][m - 1] * ~f[k2][n]
                                * f[i][n] * f[j2][m - 1] * f[k2][n + 1] -
                                ~f[i][n + 1] * ~f[j2][m] * ~f[k2][n]
                                * f[i][n] * f[j2][m] * f[k2][n + 1]);

                    }
                }

            }

        }
        
        Ei /= norm2[i];
        
        E += Ei;
    }

    return E.real();
}