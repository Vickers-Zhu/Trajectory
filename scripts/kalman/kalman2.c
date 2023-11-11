// Yoshinori Hayakawa
// 2021-05-08

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define POW2(x) ((x)*(x))

static double Q[3][3] = {{0.1, 0.0, 0.0},
                         {0.0, 0.1, 0.0},
                         {0.0, 0.0, 0.1}} ;
static double R[3][3] = {{1.0, 0.0, 0.0},
                         {0.0, 1.0, 0.0},
                         {0.0, 0.0, 1.0}} ;

static double px0, py0, pz0, vx0, vy0, vz0, ax0, ay0, az0 ;
static double px1, py1, pz1, vx1, vy1, vz1, ax1, ay1, az1 ;

static double S0[9][9],S1[9][9] ;

#define DT (1.0/60)
#define RDT (0.12909944487358056281) // sqrt(1/60)

static double F[9][9] = {{1, 0, 0,DT, 0, 0, 0, 0, 0},
                         {0, 1, 0, 0,DT, 0, 0, 0, 0},
                         {0, 0, 1, 0, 0,DT, 0, 0, 0},
                         {0, 0, 0, 1, 0, 0,DT, 0, 0},
                         {0, 0, 0, 0, 1, 0, 0,DT, 0},
                         {0, 0, 0, 0, 0, 1, 0, 0,DT},
                         {0, 0, 0, 0, 0, 0, 1, 0, 0},
                         {0, 0, 0, 0, 0, 0, 0, 1, 0},
                         {0, 0, 0, 0, 0, 0, 0, 0, 1}} ;
static double FT[9][9] ;

static double H[3][9] = {{1,0,0,0,0,0,0,0,0},
                         {0,1,0,0,0,0,0,0,0},
                         {0,0,1,0,0,0,0,0,0}} ;
static double HT[9][3] ;

static double G[9][3] = {{0,0,0},
                         {0,0,0},
                         {0,0,0},
                         {0,0,0},
                         {0,0,0},
                         {0,0,0},
                         {RDT,0,0},
                         {0,RDT,0},
                         {0,0,RDT}} ;                         
static double GT[3][9] ;

static double E[9][9] = {{1,0,0,0,0,0,0,0,0},
                         {0,1,0,0,0,0,0,0,0},
                         {0,0,1,0,0,0,0,0,0},
                         {0,0,0,1,0,0,0,0,0},
                         {0,0,0,0,1,0,0,0,0},
                         {0,0,0,0,0,1,0,0,0},
                         {0,0,0,0,0,0,1,0,0},
                         {0,0,0,0,0,0,0,1,0},
                         {0,0,0,0,0,0,0,0,1}} ;

static double K[9][3] ;


void transpose_9x9(double A[][9], double B[][9]) {
    int i,j ;
    for (i=0 ; i<9; i++) {
        for (j=0; j<9; j++) {
            B[i][j] = A[j][i] ;
        }
    }
}

void transpose_3x9(double A[][9], double B[][3])
{
    int i,j ;
    for (i=0 ; i<9; i++) {
        for (j=0; j<3; j++) {
            B[i][j] = A[j][i] ;
        }
    }
}

void transpose_9x3(double A[][3], double B[][9])
{
    int i,j ;
    for (i=0 ; i<3; i++) {
        for (j=0; j<9; j++) {
            B[i][j] = A[j][i] ;
        }
    }
}

void inv_3x3(double A[][3], double B[][3]) {
    double det =
        A[0][0]*A[1][1]*A[2][2] + A[1][0]*A[2][1]*A[0][2] + A[0][1]*A[1][2]*A[2][0]
        - A[0][2]*A[1][1]*A[2][0] - A[0][1]*A[1][0]*A[2][2] - A[1][2]*A[2][1]*A[0][0] ;

    double c00 = A[1][1] * A[2][2] - A[1][2]*A[2][1] ;
    double c01 = A[1][0] * A[2][2] - A[1][2]*A[2][0] ;
    double c02 = A[1][0] * A[2][1] - A[1][1]*A[2][0] ;
    double c10 = A[0][1] * A[2][2] - A[0][2]*A[2][1] ;
    double c11 = A[0][0] * A[2][2] - A[0][2]*A[2][0] ;
    double c12 = A[0][0] * A[2][1] - A[0][1]*A[2][0] ;
    double c20 = A[0][1] * A[1][2] - A[0][2]*A[1][1] ;            
    double c21 = A[0][0] * A[1][2] - A[0][2]*A[1][0] ;
    double c22 = A[0][0] * A[1][1] - A[0][1]*A[1][0] ;                

    B[0][0] = c00/det ;
    B[0][1] = -c10/det ;
    B[0][2] = c20/det ;
    B[1][0] = -c01/det ;
    B[1][1] = c11/det ;
    B[1][2] = -c21/det ;
    B[2][0] = c02/det ;
    B[2][1] = -c12/det ;
    B[2][2] = c22/det ;
}

void init_mat(double dt, double x, double y, double z, double dsx,  double dsy, double pixsiz, double flen, double elev) {
    F[0][3] = dt ;
    F[1][4] = dt ;
    F[2][5] = dt ;
    F[3][6] = dt ;
    F[4][7] = dt ;
    F[5][8] = dt ;    
    
    G[6][0] = sqrt(dt) ;
    G[7][1] = sqrt(dt) ;
    G[8][2] = sqrt(dt) ;    
    
    transpose_9x9(F,FT) ;
    transpose_3x9(H,HT) ;
    transpose_9x3(G,GT) ;

    double dist =sqrt(POW2(x) + POW2(y) + POW2(z)) ;

#if 0    
    R[0][0] = dist/200 * 0.01 ;
    R[2][2] = dist/200 * 0.01 ;
    R[1][1] = POW2(dist/200.0) * 0.75 ;
#else
    double theta = atan2( sqrt(dsx*dsx + dsy*dsy) * pixsiz, flen) ;
    double depth = dist * cos(theta) ;

    double delta = (pixsiz*depth)/flen + 0.1 ;  // obs error in the direction parallel to screen
    double sigma = pow(depth/200.0,2.0) * 0.5 + 0.1 ; // obs error in depth
    
    double phi = elev * M_PI/180.0 ;

    // fprintf(stderr,"theta=%lf  phi=%lf  depth= %lf  delta= %lf  sigma= %lf\n",theta*180.0/M_PI, elev, depth, delta, sigma) ;
    
    // X
    R[0][0] = POW2(delta) ;
    // Y
    R[1][1] = POW2(cos(phi)*sigma) + POW2(sin(phi)*delta) ;
    // Z
    R[2][2] = POW2(sin(phi)*sigma) + POW2(cos(phi)*delta) ;
    // Y-Z
    R[1][2] = R[2][1] = sin(phi)*cos(phi)*(POW2(sigma) - POW2(delta)) ;
#endif    
}

void update_state(double dt) {
    int i,j,k ;
    double W1[9][9]  ;
    double W2[9][9]  ;
    double W3[9][3]  ;
    double W4[9][9]  ;    
    
    px0 = px1 + vx1 * dt ;
    py0 = py1 + vy1 * dt ;
    pz0 = pz1 + vz1 * dt ;    
    vx0 = vx1 + ax1 * dt ;
    vy0 = vy1 + ay1 * dt ;
    vz0 = vz1 + az1 * dt ;
    ax0 = ax1 ;
    ay0 = ay1 ;
    az0 = az1 ;
    
    // FSF^t
    for (j=0; j<9; j++) {
        for (i=0; i<9; i++) {
            W1[i][j]=0.0 ;
            for (k=0; k<9; k++) {
                W1[i][j] += F[i][k] * S1[k][j] ;
            }
        }
    }
    for (j=0; j<9; j++) {
        for (i=0; i<9; i++) {
            W2[i][j] = 0.0 ;
            for (k=0; k<9; k++) {
                W2[i][j] += W1[i][k] * FT[k][j] ;
            }
        }
    }
    // GQG^t
    for (j=0 ; j<3 ; j++) {
        for (i=0; i<9 ; i++) {
            W3[i][j] = 0.0 ;
            for (k=0; k<3; k++) {
                W3[i][j] += G[i][k] * Q[k][j] ;
            }
        }
    }
    for (j=0 ; j<9 ; j++) {
        for (i=0; i<9 ; i++) {
            W4[i][j] = 0.0 ;
            for (k=0; k<3; k++) {
                W4[i][j] += W3[i][k] * GT[k][j] ;
            }
        }
    }

    for (j=0 ; j<9 ; j++) {
        for (i=0; i<9 ; i++) {
            S0[i][j] = W2[i][j] + W4[i][j] ;
        }
    }
}

void calc_kalman_gain() {
    int i,j,k ;
    double W1[9][3] ;
    double W2[3][9] ;
    double W3[3][3] ;
    double W4[3][3] ;

    // SH^t
    for (j=0 ; j<3 ; j++) {
        for (i=0 ; i<9 ; i++) {
            W1[i][j] = 0.0 ;
            for (k=0 ; k<9 ; k++) {
                W1[i][j] += S0[i][k] * HT[k][j] ;
            }
        }
    }
    // HSH^t
    for (j=0; j<9; j++) {
        for (i=0; i<3; i++) {
            W2[i][j] = 0.0 ;
            for (k=0; k<9; k++) {
                W2[i][j] += H[i][k] * S0[k][j] ;
            }
        }
    }
    for (j=0; j<3; j++) {
        for (i=0; i<3; i++) {
            W3[i][j] = 0.0 ;
            for (k=0; k<9; k++) {
                W3[i][j] += W2[i][k] * HT[k][j] ;
            }
        }
    }
    for (j=0; j<3; j++) {
        for (i=0; i<3; i++) {
            W3[i][j] += R[i][j] ;
        }
    }

    // W3^{-1} -> W4    
    inv_3x3(W3,W4) ;
    
    for (j=0; j<3; j++) {
        for (i=0; i<9; i++) {
            K[i][j]=0.0 ;
            for (k=0; k<3; k++) {
                K[i][j] += W1[i][k] * W4[k][j] ;

            }
            // fprintf(stderr,"K[%d,%d]=%lf\n",i,j,K[i][j]) ;            
        }
    }
}

void perform_adjustment(double obsx, double obsy, double obsz) {
    
    calc_kalman_gain() ;

    double dx = obsx - px0 ;
    double dy = obsy - py0 ;
    double dz = obsz - pz0 ;    
    px1 = px0 + K[0][0] * dx + 	K[0][1] * dy + K[0][2] * dz ;
    py1 = py0 + K[1][0] * dx + 	K[1][1] * dy + K[1][2] * dz ;
    pz1 = pz0 + K[2][0] * dx + 	K[2][1] * dy + K[2][2] * dz ;    
    vx1 = vx0 + K[3][0] * dx + 	K[3][1] * dy + K[3][2] * dz ;
    vy1 = vy0 + K[4][0] * dx + 	K[4][1] * dy + K[4][2] * dz ;
    vz1 = vz0 + K[5][0] * dx + 	K[5][1] * dy + K[5][2] * dz ;            
    ax1 = ax0 + K[6][0] * dx + 	K[6][1] * dy + K[6][2] * dz ;            
    ay1 = ay0 + K[7][0] * dx + 	K[7][1] * dy + K[7][2] * dz ;            
    az1 = az0 + K[8][0] * dx + 	K[8][1] * dy + K[8][2] * dz ;
    
    double W1[9][9] ;
    double W2[9][9] ;
    int i,j,k ;

    // I - KH
    for (j=0 ; j<9 ; j++) {
        for (i=0 ; i<9 ; i++) {
            W1[i][j] = E[i][j] ;
            for (k=0 ; k<3 ; k++) {
                W1[i][j] -= K[i][k] * H[k][j] ;
            }
        }
    }
    for (j=0 ; j<9 ; j++) {
        for (i=0 ; i<9 ; i++) {
            S1[i][j] = 0.0 ;
            for (k=0 ; k<9 ; k++) {
                S1[i][j] += W1[i][k] * S0[k][j] ;
            }
        }
    }
}

void kalman_filtering(
                      double obsdsx, double obsdsy,  // screen coordinate from center (pixel)
                      double pixsiz, // pixel size (m)
                      double flen, // focual length (m)
                      double elev, // elevation angle
                      double dt, // sampling time (sec)
                      double obsx, double obsy, double obsz,
                      double *px, double *py, double *pz,
                      double *vx, double *vy, double *vz,
                      double *ax, double *ay, double *az,                      
                      double S[][9],
                      double *epx, double *epy, double *epz,
                      double *evx, double *evy, double *evz,
                      double *eax, double *eay, double *eaz
                      )
{
    px0 = *px ;
    py0 = *py ;
    pz0 = *pz ;
    vx0 = *vx ;
    vy0 = *vy ;
    vz0 = *vz ;    
    ax0 = *ax ;
    ay0 = *ay ;
    az0 = *az ;    

    // fprintf(stderr,"\nOBS= %lf %lf   STATES= %lf %lf %lf %lf\n",obsx,obsy,*px,*py,*vx,*vy) ;
    // fprintf(stderr,"MAT= %lf %lf %lf %lf ...\n", *s00,*s01,*s02,*s03) ;

    int i,j ;
    for (i=0; i<9; i++) {
        for (j=0; j<9; j++) {
            S0[i][j] = S[i][j] ;
        }
    }

    init_mat(dt, obsx, obsy, obsz, obsdsx, obsdsy, pixsiz, flen, elev) ;
    
    perform_adjustment(obsx, obsy, obsz) ;
    update_state(dt) ;

    *px = px0 ;
    *py = py0 ;
    *pz = pz0 ;    
    *vx = vx0 ;
    *vy = vy0 ;
    *vz = vz0 ;
    *ax = ax0 ;
    *ay = ay0 ;
    *az = az0 ;    

    *epx = px1 ;
    *epy = py1 ;
    *epz = pz1 ;    
    *evx = vx1 ;
    *evy = vy1 ;
    *evz = vz1 ;
    *eax = ax1 ;
    *eay = ay1 ;
    *eaz = az1 ;    

    // fprintf(stderr,"ADJ= %lf %lf %lf %lf  STATES= %lf %lf %lf %lf\n",px1,py1, vx1,vy1, *px,*py,*vx,*vy) ;

    for (i=0; i<9; i++) {
        for (j=0; j<9; j++) {
            S[i][j] = S0[i][j] ;
        }
    }
}


void inv_mat9x9(double A[][9], double X[][9]) {
    int i,j,k,ell ;
    double aii,aji ;
    double W[9][9] ;
    for (i=0; i<9; i++) {
        for (j=0; j<9; j++) {
            if (i==j) X[i][j]=1.0 ;
            else X[i][j]=0.0 ;
            W[i][j] = A[i][j] ;
        }
    }

    for (i=0; i<9; i++) {
        double amax=0.0 ;
        k = i ;
        for (j=i; j<9; j++) {
            if (fabs(W[j][i])>amax) {
                amax = fabs(W[j][i]) ;
                k = j ;
            }
        }
        if (k!=i) {
            for (j=0; j<9; j++) {
                double w ;
                w = W[k][j] ;
                W[k][j] = W[i][j] ;
                W[i][j] = w ;
                w = X[k][j] ;
                X[k][j] = X[i][j] ;
                X[i][j] = w ;
            }
        }
        aii = W[i][i] ;
        for (j=0; j<9; j++) {
            W[i][j] = W[i][j]/aii ;
            X[i][j] = X[i][j]/aii ;
        }
        for (j=i+1 ; j<9 ; j++) {
            aji = W[j][i] ;
            for (k=0 ; k<9; k++) {
                W[j][k] = W[j][k] - W[i][k]*aji ;
                X[j][k] = X[j][k] - X[i][k]*aji ;                
            }
        }
    }
    
    for (i=9-1; i>0; i--) {
        for (j=0; j<i; j++) {
            aji = W[j][i] ;
            for (k=0; k<9; k++) {
                W[j][k] = W[j][k] - W[i][k]*aji ;
                X[j][k] = X[j][k] - X[i][k]*aji ;                    
            }
        }
    }
}


#define MAXFRAMES 2000

static double XLIST[MAXFRAMES][9] ;
static double VLIST[MAXFRAMES][9][9] ;
static double SLIST[MAXFRAMES][9][9] ;

void kalman_smoother(int n, double deltat[], double obs[][3],
                     double obsds[][2],
                     double pixsiz, double flen, double elev,
                     double epos[][3], double evel[][3], double eacc[][3]) {
    int i,j,k,ti ;

    if (n>MAXFRAMES) {
        fprintf(stderr,"kalman_smoother: EXCEEDS MAX DATA POINT\n") ;
        exit(0) ;
    }

    // phase 1: kalman filtering
    for (i=0; i<9; i++) {
        for (j=0; j<9; j++) {
            if (i==j) S0[i][j] = 1.0 ;
            else S0[i][j] = 0.0 ;
        }
    }

    px0 = obs[0][0] ; py0 = obs[0][1] ; pz0 = obs[0][2] ;
    vx0 = 0 ; vy0 = 0 ; vz0 = 0 ; ax0 = 0 ; ay0 = 0 ; az0 = 0 ;    
    for (ti=0 ; ti<n ; ti++) {
        double dt=deltat[ti] ;
        double obsx = obs[ti][0] ;
        double obsy = obs[ti][1] ;
        double obsz = obs[ti][2] ;
        double obsdsx = obsds[ti][0] ;
        double obsdsy = obsds[ti][1] ;
        init_mat(dt, obsx, obsy, obsz, obsdsx, obsdsy, pixsiz, flen, elev) ;
        
        perform_adjustment(obsx, obsy, obsz) ;
        XLIST[ti][0] = px1 ; XLIST[ti][1] = py1 ; XLIST[ti][2] = pz1 ;
        XLIST[ti][3] = vx1 ; XLIST[ti][4] = vy1 ; XLIST[ti][5] = vz1 ;
        XLIST[ti][6] = ax1 ; XLIST[ti][7] = ay1 ; XLIST[ti][8] = az1 ;                        
        for (i=0; i<9 ; i++) {
            for (j=0; j<9; j++) {
                VLIST[ti][i][j] = S1[i][j] ;
            }
        }
        update_state(dt) ;
        for (i=0; i<9 ; i++) {
            for (j=0; j<9; j++) {
                SLIST[ti][i][j] = S0[i][j] ;
            }
        }
    }

    // phase 2: RTS smoothing
    double V1[9][9],V[9][9],S[9][9] ;
    double SINV[9][9],C[9][9],CT[9][9],X[9],X1[9] ;
    double W1[9][9] ;

    epos[n-1][0] = px1 ; epos[n-1][1] = py1 ; epos[n-1][2] = pz1 ;
    evel[n-1][0] = vx1 ; evel[n-1][1] = vy1 ; evel[n-1][2] = vz1 ;
    eacc[n-1][0] = ax1 ; eacc[n-1][1] = ay1 ; eacc[n-1][2] = az1 ;

    for (i=0; i<9; i++) {
        X1[i] = XLIST[n-1][i] ;
        for (j=0; j<9; j++) {
            V1[i][j] = VLIST[n-1][i][j] ;
        }
    }

    for (ti=n-2 ; ti>=0; ti--) {
        for (i=0; i<9; i++) {
            X[i] = XLIST[ti][i] ;
            for (j=0; j<9; j++) {
                V[i][j] = VLIST[ti][i][j] ;
                S[i][j] = SLIST[ti][i][j] ;
            }
        }
        inv_mat9x9(S,SINV) ;

        double dt = deltat[ti] ;
        F[0][3] = dt ;
        F[1][4] = dt ;
        F[2][5] = dt ;
        F[3][6] = dt ;
        F[4][7] = dt ;
        F[5][8] = dt ;    
        transpose_9x9(F,FT) ;

        // C = V FT SINV
        for (i=0 ; i<9; i++) {
            for (j=0 ; j<9 ; j++) {
                W1[i][j] = 0.0 ;
                for (k=0; k<9; k++) {
                    W1[i][j] += V[i][k] * FT[k][j] ;
                }
            }
        }
        for (i=0 ; i<9; i++) {
            for (j=0 ; j<9 ; j++) {
                C[i][j] = 0.0 ;
                for (k=0; k<9; k++) {
                    C[i][j] += W1[i][k] * SINV[k][j] ;
                }
            }
        }
        transpose_9x9(C,CT) ;        

        // X1 = X + C.dot(X1 - F.dot(X))
        double w2[9],w ;
        for (i=0; i<9; i++) {
            w2[i]=X1[i] ;
            for (k=0; k<9; k++) {
                w2[i] -=  F[i][k] * X[k] ;
            }
        }
        for (i=0; i<9; i++) {
            w=0.0 ;
            for (k=0; k<9; k++) {
                w += C[i][k] * w2[k] ;
            }
            X1[i] = X[i] + w ;
        }
        epos[ti][0] = X1[0] ; epos[ti][1] = X1[1] ; epos[ti][2] = X1[2] ;
        evel[ti][0] = X1[3] ; evel[ti][1] = X1[4] ; evel[ti][2] = X1[5] ;
        eacc[ti][0] = X1[6] ; eacc[ti][1] = X1[7] ; eacc[ti][2] = X1[8] ;

        double W2[9][9] ;
        // V1 = V + C.dot((V1 - S).dot(C.T))
        for (i=0; i<9; i++) {
            for (j=0; j<9; j++) {
                W1[i][j] = 0 ;
                for (k=0; k<9; k++) {
                    W1[i][j] += C[i][k] * (V1[k][j] - S[k][j]) ;
                }
            }
        }
        for (i=0; i<9; i++) {
            for (j=0; j<9; j++) {
                W2[i][j] = 0 ;
                for (k=0; k<9; k++) {
                    W2[i][j] += W1[i][k] * CT[k][j] ;
                }
            }
        }
        for (i=0; i<9; i++) {
            for (j=0; j<9; j++) {
                V1[i][j] = V[i][j] + W2[i][j] ;
            }
        }
    }
}
