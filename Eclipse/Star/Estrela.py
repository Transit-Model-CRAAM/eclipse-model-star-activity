__author__ = "Adriana Valio, Beatriz Duque, Felipe Pereira Pinho"
__copyright__ = "..."
__credits__ = ["Universidade Presbiteriana Mackenzie, CRAAM"]
__license__ = ""
__version__ = ""
__maintainer__ = ""
__email__ = "biaduque7@hotmail.com"
__status__ = "Production"

'''
Este programa simula a plotagem de uma estrela com manchas, através de parâmetros como raio, intensidade, escurecimento 
de limbo, etc.
As bibliotecas importadas são: 
math
matplotlib
numpy
verify:função criada para validar entradas, por exemplo numeros nao float/int ou negativos
'''


from statistics import geometric_mean
from tokenize import String
from typing import List
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from Misc.Verify import Validar
from ctypes import *
from numpy.ctypeslib import ndpointer
import platform
import os
import cv2 as cv
import numpy as np
from scipy.io import readsav
from astropy.io import fits
from pathlib import Path

from PIL import Image
from matplotlib.animation import FuncAnimation
from scipy.ndimage import zoom
from enum import Enum, auto

class GeometriaCME(Enum):
        """
        Define as formas visuais possíveis para a Ejeção de Massa Coronal.
        """
        # 'CAPSULA'. Representa a visão "de cima" (top-down)
        # da base da erupção.
        PROJECAO_DISCO = auto()   
            
        # 'GOTA'. Representa a visão de "perfil" (side-on)
        # da erupção, como vista no limbo estelar.
        VISAO_PERFIL = auto()   

class Estrela:
    '''
    A classe estrela recebe como objeto o raio, intensidade maxima, coeficientes de escurecimento de limbo.
    A estrela é formata em uma matriz de tamanho defeault 856.
    São objetos pertencentes a classe os parâmetros passados à mancha, como: raio, intensidade, longitude e latitude 
    em relação à estrela. 
    ************ PARÂMETROS DA ESTRELA***************
    :parâmetro raio: O raio da estrela em pixel
    :parâmetro raioSun: O raio da estrela em unidades de Rsun
    :parâmetro intensidadeMaxima: Intensidade do centro da estrela
    :parâmetro coeficienteHum: Coeficiente de escurecimento de limbo 
    :parâmetro coeficienteDois: Coeficiete de escurecimento de limbo
    :parâmetro tamanhoMatriz: Tamanho da matriz em que será construída a estrela 
    :parâmetro estrela: Estrela construida com os coeficientes de escurecimento de limbo
    '''
   

    def __init__(self, raio, raioSun, intensidadeMaxima, coeficienteHum, coeficienteDois, tamanhoMatriz, coeficienteTres = None, coeficienteQuatro = None, useFits = False, fits_path=""):
        
        self.raio = raio # em pixel
        if useFits: 
            self.raioSun = 1 * 696340
        else : 
            self.raioSun = raioSun * 696340 # em relacao ao raio do Sol
            
        self.intensidadeMaxima = intensidadeMaxima
        self.coeficienteHum = coeficienteHum # to-do: mudar nome
        self.coeficienteDois = coeficienteDois # to-do: mudar nome
        self.tamanhoMatriz = tamanhoMatriz # to-do: mudar nome
        self.temperaturaEfetiva = 4875.0 # to-do: mudar nome, porque esta mockado?
        
        # Extra arguments
        self.coeficienteTres = coeficienteTres # to-do: mudar nome
        self.coeficienteQuatro = coeficienteQuatro # to-do: mudar nome
        
        self.useFits = useFits

        if useFits: 
            self.estrelaMatriz = []
            self.criaEstrelaByFits(fits_path)
        else: 
            self.estrelaMatriz = self.criaEstrela()

        self.Nx = self.tamanhoMatriz
        self.Ny = self.tamanhoMatriz
        self.color = "hot"

        self.manchas: List[Estrela.Mancha] = []
        self.faculas: List[Estrela.Facula] = []
        self.cme: Estrela.EjecaoMassa = None

        ### Prints para testes. Descomentar linhas abaixo se necessário ### 
        #print(self.estrelaMatriz)
        #self.color = random.choice(self.colors)
        #Plotar(self.tamanhoMatriz,self.estrelaMatriz)
        #end = time.time()
        #print(end - start)
    
    class Mancha: 
        def __init__(self, intensidade, raio, latitude, longitude):
            self.intensidade = intensidade # em relacao a intensidade da estrela (menor que 1)
            self.raio = raio # em relacao ao raio da estrela
            self.latitude = latitude 
            self.longitude = longitude

            self.area = 0.0

    class Facula:
        def __init__(self, intensidade, raio, latitude, longitude):
            self.intensidade = intensidade # em relacao a intensidade da estrela (maior que 1)
            self.raio = raio # em relacao ao raio da estrela
            self.latitude = latitude 
            self.longitude = longitude    

    class EjecaoMassa: 
        def __init__(self, raio, p0x, p0y, p1x, p1y, opacidade, temperatura, velocidade, taxa_esfriamento, altura_inicial, geometriaCME: GeometriaCME): 
            self.raio = raio
            self.temperatura = temperatura
            self.p0x = p0x
            self.p0y = p0y
            self.p1x = p1x
            self.p1y = p1y
            self.opacidade = opacidade
            self.velocidade = velocidade 
            self.taxa_esfriamento = taxa_esfriamento
            self.altura_inicial = altura_inicial
            self.geometriaCME = geometriaCME
        
    def criaEstrela(self): 
        # Obter o caminho absoluto do diretório atual
        dir_atual = os.path.dirname(os.path.abspath(__file__))

        # Voltar um diretório para chegar ao diretório pai
        dir_pai = os.path.dirname(dir_atual)

        # Verifica o SO e se o Python é 32 ou 64 bit
        if(platform.system() == "Windows"):
            if(platform.architecture()[0] == "32bit"):
                script_path = os.path.join(dir_pai, 'scripts', 'func32.dll')
                my_func = WinDLL("a", winmode = 0x8)
            elif(platform.architecture()[0] == "64bit"):
                script_path = os.path.join(dir_pai, 'scripts', 'func64.dll')
                my_func = WinDLL(script_path, winmode = 0x8)
        elif(platform.system() == "Darwin"):
            script_path = os.path.join(dir_pai, 'scripts', 'func64.dylib')
            my_func = cdll.LoadLibrary(script_path)
        else:
            script_path = os.path.join(dir_pai, 'scripts', 'func64.so')
            my_func = CDLL(script_path)

        
        
        linha = self.tamanhoMatriz
        coluna = self.tamanhoMatriz
        
        # Equacao com 4 coeficientes de limbo 
        # Não linear de quatro termos (Claret)
        
        # TODO: Ajustar o memory leak também pra função de Claret aqui
        if (self.coeficienteTres and self.coeficienteQuatro): 
            my_func.criaEstrelaClaret.restype = ndpointer(dtype=c_int, ndim=2, shape=(self.tamanhoMatriz,self.tamanhoMatriz))
            estrelaMatriz = my_func.criaEstrelaClaret(linha,
                                                    coluna,
                                                    self.tamanhoMatriz,
                                                    c_float(self.raio),
                                                    c_float(self.intensidadeMaxima),
                                                    c_float(self.coeficienteHum),
                                                    c_float(self.coeficienteDois),
                                                    c_float(self.coeficienteTres),
                                                    c_float(self.coeficienteQuatro))
            return estrelaMatriz
        # Equacao coeficiente de limbo quadrático

        # Cria a matriz estrela
        my_func.criaEstrela.restype = POINTER(c_int)
        estrela_ptr = my_func.criaEstrela(linha,
                                            coluna,
                                            self.tamanhoMatriz,
                                            c_float(self.raio),
                                            c_float(self.intensidadeMaxima),
                                            c_float(self.coeficienteHum),
                                            c_float(self.coeficienteDois))

        # Transforma a matriz estrela em um objeto numpy
        estrelaMatriz = np.ctypeslib.as_array(estrela_ptr, shape=(self.tamanhoMatriz, self.tamanhoMatriz)).copy()

        # Libera a memória alocada via malloc para a estrela
        my_func.liberaEstrela.argtypes = [POINTER(c_int)]
        my_func.liberaEstrela(estrela_ptr)

        del estrela_ptr

        return estrelaMatriz
    
    def criaEstrelaByFits(self, path: String):
        '''
        Cria estrela através de um arquivo .FITS da imagem da Estrela 
        '''
        self.tamanhoMatriz = 856

        # arquivo FITS dos dados
        # Obter o caminho absoluto do diretório atual
        dir_atual = os.path.dirname(os.path.abspath(__file__))

        # Voltar um diretório para chegar ao diretório pai
        dir_pai = os.path.dirname(dir_atual)
        dir_pai = os.path.dirname(dir_pai)
        
        path = os.path.join(dir_pai, "Sun", "sdo_aia_download", path)

        for nome_arquivo in os.listdir(path):
            if nome_arquivo.endswith('.fits'):
                hdul = fits.open(os.path.join(path, nome_arquivo))
                star_image = np.squeeze(hdul[1].data) # OS DADOS (IMAGENS) ESTAO AQUI
                normalized = star_image
                normalized[np.where(star_image<=0)]=1

                tamanhoMatriz = len(normalized[:, 0])
                
                resized = zoom(normalized, (self.tamanhoMatriz / tamanhoMatriz, self.tamanhoMatriz / tamanhoMatriz), order=0)

                radius_fits = hdul[1].header['RSUN_OBS']/hdul[1].header['CDELT1'] # radius in arcsec

                self.estrelaMatriz.append(resized[::-1].astype(np.float64))
                
        #self.estrelaMatriz.reverse()
        return self.estrelaMatriz

    '''
    Ruidos podem ser Manchas ou Fáculas
    '''
    def criaRuidos(self, ruidos): 
        for ruido in ruidos:
            raio_mancha_pixel = self.raio * ruido.raio # raio em funcao do raio da estrela em pixels

            #coordenadas de posicionamento da mancha em graus
            
            degreeToRadian = np.pi/180. #A read-only variable containing the floating-point value used to convert degrees to radians.
            latitudeMancha  = ruido.latitude * degreeToRadian 
            longitudeMancha =  ruido.longitude * degreeToRadian

            #posicao da mancha em pixels em relacao ao centro da estrela
            ys = self.raio*np.sin(latitudeMancha)  
            xs = self.raio*np.cos(latitudeMancha)*np.sin(longitudeMancha)
            anguloHelio = np.arccos(np.cos(latitudeMancha)*np.cos(longitudeMancha))

            # efeito de projecao pela mancha estar a um anguloHeliocentrico do centro da estrela - elipcidade
            yy = ys + self.Ny/2 # posicao em pixel com relacao à origem da matriz
            xx = xs + self.Nx/2 # posicao em pixel com relacao à origem da matriz

            kk = np.arange(self.Ny * self.Nx)
            vx = kk-self.Nx*np.int64(1.*kk/self.Nx) - xx
            vy = kk/self.Ny - yy

            # angulo de rotacao da mancha
            anguloRot=np.abs(np.arctan(ys/xs)) # em radianos
            if latitudeMancha*longitudeMancha > 0:
                anguloRot = -anguloRot
            elif latitudeMancha * longitudeMancha == 0:
                anguloRot = 0

            ii, = np.where((((vx*np.cos(anguloRot)-vy*np.sin(anguloRot))/np.cos(anguloHelio))**2+(vx*np.sin(anguloRot)+vy*np.cos(anguloRot))**2) < raio_mancha_pixel**2)
        
            spot = np.zeros(self.Ny * self.Nx) + 1
                
            spot[ii] = ruido.intensidade
            spot = spot.reshape([self.Ny, self.Nx])
    
            self.estrelaMatriz= self.estrelaMatriz * spot

        return self.estrelaMatriz 
    
    #######  Inserção de manchas
    def addMancha(self, mancha: Mancha): 
        '''
        Função onde são criadas as Manchas da estrela. Todos os parâmetros 
        são relacionados ao tamanho da estrela, podendo o usuário escolher valores 
        ou selecionar a opção default.
        '''
        if (mancha.intensidade > 1) : 
            print("O valor da intensidade da Mancha deve ser menor que 1.")
            return 

        self.manchas.append(mancha)

    def criaEstrelaManchada(self):
        self.criaRuidos(self.manchas)

    #######  Inserção de Fáculas
    def addFacula(self, facula: Facula): 
        '''
        Função onde são criadas as Fáculas da estrela. Todos os parâmetros 
        são relacionados ao tamanho da estrela, podendo o usuário escolher valores 
        ou selecionar a opção default.
        '''
        if (facula.intensidade < 1) : 
            print("O valor da intensidade da Fácula deve ser maior que 1.")
            return 

        self.faculas.append(facula)
    
    def criaEstrelaComFaculas(self): 
        self.criaRuidos(self.faculas)

    def addCme(self, cme: EjecaoMassa): 
        self.cme = cme

    ####### CME (Ejeção de Massa Estelar)
    def ejecaoDeMassa(self, temperatura, raio, opacidade_cme): 
        if self.cme.geometriaCME == GeometriaCME.VISAO_PERFIL: 
            return self.ejecaoDeMassaGota(temperatura, raio, opacidade_cme)

        coroa = self.createCoroa()
        
        p0 = (self.cme.p0x, self.cme.p0y)
        p1 = (self.cme.p1x, self.cme.p1y)

        intensidade = opacidade_cme * ((temperatura * 240) / self.temperaturaEfetiva) + (1 - opacidade_cme) * 240 # original
        print("intensidade >>>", intensidade)
        # to-do: testar essas duas equacoes
        #intensidade = opacidade_cme * ((temperatura / self.temperaturaEfetiva) * 240) + (1 - opacidade_cme) * 240
        # intensidade = opacidade * (fator_contraste * intensidade_max) + (1 - opacidade) * intensidade_max

        cv.line(coroa, p0, p1, intensidade, raio)

        return coroa

    def ejecaoDeMassaGota(self, temperatura, raio, opacidade_cme): 
        coroa = self.createCoroa()
        
        # --- Parâmetros para a gota ---
        # (p0x, p0y) será a ponta pontuda
        ponta_superior = (self.cme.p0x, self.cme.p0y) 
        
        # (p1x, p1y) será o centro da base redonda
        centro_base = (self.cme.p1x, self.cme.p1y)
        
        # 'raio' define o quão "gorda" será a base da gota
        raio_base = raio 
        
        intensidade = opacidade_cme * ((temperatura * 240) / self.temperaturaEfetiva) + (1 - opacidade_cme) * 240

        # --- Substituição da Lógica de Desenho ---
        # Trocamos a linha pela gota:
        # cv.line(coroa, ponta_superior, centro_base, intensidade, raio) # <- LINHA ANTIGA
        
        # Nova função de desenho:
        # (Assumindo que 'desenharGota' é um método da mesma classe, use self.desenharGota)
        self.desenharGota(coroa, ponta_superior, centro_base, raio_base, intensidade)

        return coroa

    def desenharGota(self, imagem, ponta_superior, centro_base, raio_base, cor):
        """
           Desenha uma 'gota' preenchida (círculo com um cone pontudo) na imagem.

           Argumentos:
           imagem (np.array): A imagem 'coroa' onde o desenho será feito.
           ponta_superior (tuple): Coordenada (x, y) da ponta superior da gota.
           centro_base (tuple): Coordenada (x, y) do centro da base redonda.
           raio_base (int): O raio da base redonda.
           cor (float): A intensidade (cor) escalar do desenho.
       """

        cor_bgr = (float(cor), float(cor), float(cor))

        # 1. Desenha a base redonda (círculo preenchido)
        # O -1 no final significa "preenchido"
        # Usamos a nova 'cor_bgr'
        cv.circle(imagem, centro_base, int(raio_base), cor_bgr, -1)

        # 2. Define os vértices do 'cone' (polígono) que liga a ponta ao círculo

        pt_esquerda = (
            int(centro_base[0] - raio_base), int(centro_base[1]))
        pt_direita = (int(centro_base[0] + raio_base), int(centro_base[1]))

        vertices = np.array(
            [pt_esquerda, pt_direita, ponta_superior], dtype=np.int32)

        # 3. Desenha o cone (polígono preenchido)
        # Usamos a nova 'cor_bgr' AQUI TAMBÉM
        cv.fillPoly(imagem, [vertices], cor_bgr)
    
        
    def createCoroa(self): 
        matriz_coroa = np.zeros((self.tamanhoMatriz, self.tamanhoMatriz))
        return matriz_coroa

    def get_area_cme_km2(self):
        """
        Calcula a área física da CME em quilômetros quadrados.
        """
        
        if not self.cme:
            return 0.0

        # --- 1. Calcular a escala de área (km²/pixel²) ---
        escala_km_por_pixel = self.raioSun / self.raio
        fator_de_conversao_area = escala_km_por_pixel ** 2

        # --- 2. Calcular a área da CME em pixels² ---
        area_cme_pixels = 0.0
        cme = self.cme
        r_pixel = cme.raio

        if cme.geometriaCME == GeometriaCME.PROJECAO_DISCO:
            # Área da "Cápsula"
            p0 = np.array([cme.p0x, cme.p0y])
            p1 = np.array([cme.p1x, cme.p1y])
            L = np.linalg.norm(p0 - p1)  # Distância entre p0 e p1 (comprimento do retângulo)
            area_circulo = np.pi * (r_pixel**2) # 2. Área do círculo (os dois semicírculos)
            area_retangulo = L * (2 * r_pixel)  # 3. Área do retângulo
            area_cme_pixels = area_circulo + area_retangulo
        
        elif cme.geometriaCME == GeometriaCME.VISAO_PERFIL:
            # Área da "Gota"
            area_semicirculo = 0.5 * np.pi * (r_pixel**2) # 1. Área do Semicírculo
            altura_triangulo = abs(cme.p0y - cme.p1y)  # 2. Altura do triângulo (distância vertical entre a ponta e a base)
            area_triangulo = r_pixel * altura_triangulo # 3. Área do triângulo (Base = 2*r)
            area_cme_pixels = area_semicirculo + area_triangulo

        # --- 3. Converter para Km² ---
        area_cme_km2 = area_cme_pixels * fator_de_conversao_area
        
        area_estrela_pixels = np.pi * (self.raio**2)
        area_relativa = area_cme_pixels / area_estrela_pixels

        print(f"Área da CME relativa à Estrela: {area_relativa * 100:.2f} %")
        print(f"Área da CME: {area_cme_km2:,.2f} km²")
        return area_cme_km2

   
    ####### WIP: Inserção de flares
    def flares(self): #recebe como parâmetro a estrela atualizada
        '''
        Função onde são criadas os flares da estrela. Todos os parâmetros 
        são relacionados ao tamanhdo da estrela, podendo o usuário escolher valores 
        ou selecionar a opção default.
        ---Parametros ainda nao definidos
        *********INICIO DOS PARÂMETROS FLARES*******
        :parâmetro 
        :parâmetro 
        :parâmetro 
        :parâmetro
        
        '''

        #vai sobrescrever a estrela que ele está criando, sendo ela a estrela ou a estrelaManchada.
        self.estrelaMatriz = self.estrelaMatriz
        return self.estrelaMatriz #retorna a decisão: se há flare ou não 

    #### Getters
    def getNx(self):
        '''
        Retorna parâmetro Nx, necessário para o Eclipse.
        '''
        return self.Nx
    def getNy(self):
        '''
        Retorna parâmetro Ny, necessário para o Eclipse.
        '''
        return self.Ny

    def getRaioStar(self):
        '''
        Retorna o raio da estrela em pixel, necessário para o programa Eclipse, visto que o raio do planeta se dá em 
        relação ao raio da estrela.
        '''
        return self.raio
    def getMatrizEstrela(self):
        '''
        Retorna a estrela, plotada sem as manchas, necessário caso o usuário escolha a plotagem sem manchas.
        '''
        if self.useFits:
            return self.estrelaMatriz[0]
        
        return self.estrelaMatriz

    def getu1(self):
        return self.coeficienteHum
    
    def getu2(self):
        return self.coeficienteDois

    def getTamanhoMatriz(self):
        return self.tamanhoMatriz
    
    def getRaioSun(self):
        return self.raioSun

    def getIntensidadeMaxima(self):
        return self.intensidadeMaxima

    def getError(self):
        '''
        Retorna valor de erro. Se não houverem erros, a variável assumirá 0. Se houverem erros, o programa manterá
        o valor origem da variável (que é -1).
        '''
        return self.error
    def setStarName(self,starName):
        self.starName = starName

    def getStarName(self):
        return self.starName
        
    def getCadence(self):
        return self.cadence

    # Setters
    def setCadence(self,cadence):
        self.cadence = cadence

    def Plotar(self,tamanhoMatriz,estrela):
        if self.useFits: 
            self.create_animation()
            return 
            
        Nx = tamanhoMatriz
        Ny = tamanhoMatriz
        plt.axis([0,Nx,0,Ny])
        plt.imshow(estrela,self.color)
        plt.gca().invert_yaxis()  # Corrige o eixo Y invertido
        plt.show()

    def create_animation(self, cmap="copper", interval=300):
        if not self.estrelaMatriz:
            raise ValueError("No images loaded. Call load_images() first.")

        fig, ax = plt.subplots()

        image = self.estrelaMatriz[0]
        image[np.where(self.estrelaMatriz[0]<=0)]=1
        im = ax.imshow(np.log10(image), cmap=cmap, animated=True)

        def update(frame):
            image = self.estrelaMatriz[frame]
            image[np.where(self.estrelaMatriz[frame]<=0)]=1
            im.set_array(np.log10(image))
            return [im]

        self.animation = FuncAnimation(
            fig,
            update,
            frames=len(self.estrelaMatriz),
            blit=True,
            interval=interval
        )

    def show_animation(self):
        if self.animation is None:
            raise ValueError("Animation not created. Call create_animation() first.")
        plt.show()