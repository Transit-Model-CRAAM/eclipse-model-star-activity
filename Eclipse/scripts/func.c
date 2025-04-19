#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h> // Inclui memset para inicializar a matriz

int* criaEstrela(int lin, int col, int tamanhoMatriz, float raio, float intensidadeMaxima, float coeficienteHum, float coeficienteDois) {
	int i, j;
	int *estrela = (int*) malloc (lin * col * sizeof(int*));
	int index;

#pragma omp parallel {
    #pragma omp for collapse(2)
        for(i=0;i<lin;i++) {
            for(j=0;j<col;j++) {
                index = i*(lin) + j;
                estrela[index] = 0;	
            }
        }
        float distanciaCentro;
        float cosTheta;
        
    #pragma omp for collapse(2)
        for(j=0;j<col;j++){
            for(i=0;i<lin;i++){
                distanciaCentro = sqrt(pow(i-tamanhoMatriz/2,2) + pow(j-tamanhoMatriz/2,2));
                if(distanciaCentro <= raio){
                    cosTheta = sqrt(1-pow(distanciaCentro/raio,2));
                    index = i*(lin) + j;
                    estrela[index] = (int) (intensidadeMaxima * (1 - coeficienteHum * (1 - cosTheta) - coeficienteDois * (pow(1 - cosTheta,2))));
                }
            }
        }
    }
	return estrela;
}


// Função para criar uma estrela com o modelo de escurecimento de limbo de 4 parâmetros
int* criaEstrelaClaret(int lin, int col, int tamanhoMatriz, float raio, float intensidadeMaxima,
                       float coeficienteHum, float coeficienteDois, float coeficienteTres, float coeficienteQuatro) {
    // Aloca memória para a matriz da estrela
    int *estrela = (int*) malloc(lin * col * sizeof(int));
    if (!estrela) {
        fprintf(stderr, "Erro ao alocar memória para a matriz da estrela.\n");
        exit(EXIT_FAILURE);
    }

    // Inicializa a matriz com zeros
    memset(estrela, 0, lin * col * sizeof(int));

    int i, j, index;
    float distanciaCentro, cosTheta;

    // Processa a matriz da estrela em paralelo
    #pragma omp parallel for private(i, j, index, distanciaCentro, cosTheta) collapse(2)
    for (j = 0; j < col; j++) {
        for (i = 0; i < lin; i++) {
            // Calcula a distância do ponto (i, j) ao centro da estrela
            distanciaCentro = sqrt(pow(i - tamanhoMatriz / 2.0, 2) + pow(j - tamanhoMatriz / 2.0, 2));

            // Se o ponto estiver dentro do raio da estrela, calcula a intensidade
            if (distanciaCentro <= raio) {
                cosTheta = sqrt(1.0 - pow(distanciaCentro / raio, 2)); // Calcula cos(θ)
                index = i * lin + j; // Converte (i, j) para índice linear
                
                // Calcula a intensidade usando os 4 coeficientes de limbo
                float part1 = 1 - coeficienteHum * (1 - pow(cosTheta, 0.5)) - coeficienteDois * (1 - cosTheta);
                float part2 = - coeficienteTres * (1 - pow(cosTheta, 1.5)) - coeficienteQuatro * (1 - pow(cosTheta, 2.0));
                
                // Garante que a intensidade seja >= 0 (evita valores negativos)
                estrela[index] = (int) fmax(0.0, intensidadeMaxima * (part1 + part2));
            }
        }
    }

    return estrela; // Retorna a matriz gerada
}

// Função para criar uma estrela com o modelo padrão (escurecimento de limbo com 2 parâmetros)
double curvaLuz(double x0, double y0, int tamanhoMatriz, double raioPlanetaPixel, double *estrelaManchada, double *kk, double maxCurvaLuz) {
	double valor = 0;
	int i;
	
	// Processa a matriz da estrela em paralelo
	#pragma omp parallel for reduction(+:valor)
	for(i=0;i<tamanhoMatriz*tamanhoMatriz;i++) {
		if(pow((kk[i]/tamanhoMatriz-y0),2) + pow((kk[i]-tamanhoMatriz*floor(kk[i]/tamanhoMatriz)-x0),2) > pow(raioPlanetaPixel,2)) {
			valor += estrelaManchada[i];
		}
	}
	
	valor = valor/maxCurvaLuz;

	return valor;
}

double curvaLuzCME(double x0, double y0, int tamanhoMatriz, double raioPlanetaPixel, double *estrelaManchada, double *kk, double maxCurvaLuz, double *matrizCME, double opacidadeCME) {
    double valor = 0;
    int i;
    
	#pragma omp parallel for reduction(+:valor)
    for(i=0;i<tamanhoMatriz*tamanhoMatriz;i++){
        if(matrizCME[i] > 0){ // Caso a posição esteja passando em cima da CME
            if(pow((kk[i]/tamanhoMatriz-y0),2) + pow((kk[i]-tamanhoMatriz*floor(kk[i]/tamanhoMatriz)-x0),2) <= pow(raioPlanetaPixel,2)){ // Se o planeta estiver atrás
                valor += matrizCME[i] * opacidadeCME; // Matriz com estrela, CME e planeta (multiplica opacidade). Opacidade sempre de 0 a 1 (%)
            }
            else {
                valor += matrizCME[i]; // Matriz somente com estrela e CME
            }
        }
		// Procura pela posicao da matriz que não tem o planeta em frente
        else if (pow((kk[i]/tamanhoMatriz-y0),2) + pow((kk[i]-tamanhoMatriz*floor(kk[i]/tamanhoMatriz)-x0),2) > pow(raioPlanetaPixel,2)){
            valor += estrelaManchada[i];
        }
    }
    // Normalizacao
    valor = valor/maxCurvaLuz;    
    return valor;
}

double curvaLuzLua(double x0, double y0, double xm, double ym, double rMoon, int tamanhoMatriz, double raioPlanetaPixel, double *estrelaManchada, double *kk, double maxCurvaLuz) {
	double valor = 0;
	int i;
	
#pragma omp parallel for reduction(+:valor)
	for(i=0;i<tamanhoMatriz*tamanhoMatriz;i++) {
		// Procura pela posicao que não é planeta
		if((pow((kk[i]/tamanhoMatriz-y0),2) + pow((kk[i]-tamanhoMatriz*floor(kk[i]/tamanhoMatriz)-x0),2) > pow(raioPlanetaPixel,2)) && (pow((kk[i]/tamanhoMatriz-ym),2) + pow((kk[i]-tamanhoMatriz*floor(kk[i]/tamanhoMatriz)-xm),2) > pow(rMoon,2))) {
			valor += estrelaManchada[i];
		}
	}
	// Normalizacao
	valor = valor/maxCurvaLuz;
	return valor;
}