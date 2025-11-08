#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../c_libs/stb_image.h"
#include "../c_libs/stb_image_write.h"

static double** AllocMatrix(int m, int n){
    double **A = malloc(m * sizeof(double*));
    for(int i=0;i<m;i++) A[i] = calloc(n, sizeof(double));
    return A;
}
static void FreeMat(double **A,int m){
    for(int i=0;i<m;i++) free(A[i]);
    free(A);
}

static void normalize(double *v, int n){
    double norm = 0;
    for(int i=0;i<n;i++) norm += v[i]*v[i];
    norm = sqrt(norm);
    if(norm < 1e-12) return;
    for(int i=0;i<n;i++) v[i] /= norm;
}

static void MatVecMul1(double **A, double *v, double *out, int m, int n){
    for(int i=0;i<m;i++){
        double s = 0;
        for(int j=0;j<n;j++) s += A[i][j]*v[j];
        out[i] = s;
    }
}

static void MatVecMul2(double **A, double *u, double *out, int m, int n){
    for(int j=0;j<n;j++){
        double s = 0;
        for(int i=0;i<m;i++) s += A[i][j]*u[i];
        out[j] = s;
    }
}

static double PowIt(double **A, double *u, double *v, int m, int n, int iters){
    for(int i=0;i<n;i++) v[i] = ((double)rand()/RAND_MAX);
    normalize(v,n);
    double sigma = 0;
    for(int t=0;t<iters;t++){
        MatVecMul1(A, v, u, m, n);
        normalize(u, m);
        MatVecMul2(A, u, v, m, n);
        normalize(v, n);
    }
    double *temp = malloc(m*sizeof(double));
    MatVecMul1(A, v, temp, m, n);
    sigma = 0;
    for(int i=0;i<m;i++) sigma += u[i]*temp[i];
    free(temp);
    return sigma;
}

static void deflate(double **A, double *u, double *v, double sigma, int m, int n){
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++)
            A[i][j] -= sigma * u[i] * v[j];
}

static double fError(double **A,double **Ak,int m,int n){
    double sum=0;
    for(int i=0;i<m;i++)
        for(int j=0;j<n;j++){
            double d=A[i][j]-Ak[i][j];
            sum+=d*d;
        }
    return sqrt(sum);
}

int main(){
    srand(42);
    const char *input_dir = "../../figs/inputImages/";
    const char *output_dir = "../../figs/outputImages/";
    const char *images[] = {"image1.png", "image2.jpg", "image3.png"};
    int num_images = 3;
    int k_vals[] = {5,10,20,50,100};
    int nk = 5;

    for(int img_idx=0; img_idx<num_images; img_idx++){
        char inpath[256];
        sprintf(inpath, "%s%s", input_dir, images[img_idx]);
        int w,h,c;
        unsigned char *img = stbi_load(inpath,&w,&h,&c,1);
        if(!img){
            printf("Error: could not load %s\n", inpath);
            continue;
        }
        printf("\nProcessing %s (%dx%d)\n", images[img_idx], w, h);

        double **A = AllocMatrix(h,w);
        for(int i=0;i<h;i++)
            for(int j=0;j<w;j++)
                A[i][j] = img[i*w+j]/255.0;
        free(img);

        unsigned char *out = malloc(w*h);

        for(int idx=0; idx<nk; idx++){
            int k = k_vals[idx];
            double **Ak = AllocMatrix(h,w);
            double **Awork = AllocMatrix(h,w);
            for(int i=0;i<h;i++)
                for(int j=0;j<w;j++)
                    Awork[i][j] = A[i][j];

            for(int t=0; t<k; t++){
                double *u = calloc(h,sizeof(double));
                double *v = calloc(w,sizeof(double));
                double sig = PowIt(Awork,u,v,h,w,40);
                for(int i=0;i<h;i++)
                    for(int j=0;j<w;j++)
                        Ak[i][j] += sig * u[i] * v[j];
                deflate(Awork,u,v,sig,h,w);
                free(u); free(v);
            }

            double err = fError(A,Ak,h,w);
            double ratio = (double)(h*w)/(k*(h+w+1));
            printf("k=%d | Error=%.6f | Ratio=%.2f\n",k,err,ratio);

            for(int i=0;i<h;i++)
                for(int j=0;j<w;j++){
                    double v = Ak[i][j];
                    if(v<0) v=0; if(v>1) v=1;
                    out[i*w+j] = (unsigned char)(v*255);
                }

            char base[128];
                 strcpy(base, images[img_idx]);
                 char *dot = strrchr(base, '.');
                 if (dot) *dot = '\0';
                 char name[256];
                 sprintf(name, "%s%s_k%d.png", output_dir, base, k);
                 stbi_write_png(name, w, h, 1, out, w);
                 printf("Saved %s\n", name);

            FreeMat(Ak,h);
            FreeMat(Awork,h);
        }

        free(out);
        FreeMat(A,h);
        printf("Done processing %s.\n", images[img_idx]);
    }

    printf("\nAll images processed successfully.\n");
    return 0;
}
