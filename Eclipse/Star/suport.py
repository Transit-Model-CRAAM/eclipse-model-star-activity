# cria Estrela
estrela_4 = Estrela(raio_estrela_pixel, raio_estrela, intensidade_maxima, coeficiente_um, coeficiente_dois, tamanho_matriz, useFits = True, fits_path="2017-04-24")
tamanho_matriz = estrela_4.getTamanhoMatriz()

Nx = estrela_4.getNx() 
Ny = estrela_4.getNy()
dtor = np.pi/180.  

# cria Planeta
planeta_ = Planeta(semi_eixo_UA, raio_plan_Jup, periodo, angulo_inclinacao, ecc, anomalia, estrela_4.getRaioSun(), mass_planeta)

estrela_4matriz = estrela_4.getMatrizEstrela()
estrela_4.Plotar(tamanho_matriz, estrela_4matriz)

# Eclipse com CME 

eclipse_ = Eclipse(Nx, Ny, raio_estrela_pixel, estrela_4, planeta_, 1)
estrela_4.Plotar(tamanho_matriz, estrela_4matriz)

tempoHoras = 1
eclipse_.geraTempoHoras(tempoHoras)
eclipse_.criarEclipse(anim=True)

print ("Tempo Total (Trânsito):", eclipse_.getTempoTransito()) 
tempoTransito = eclipse_.getTempoTransito()
curvaLuz_3 = eclipse_.getCurvaLuz()
tempoHoras = eclipse_.getTempoHoras()
#Plotagem da curva de luz 
pyplot.plot(tempoHoras, curvaLuz_3)
pyplot.axis([-tempoTransito/2, tempoTransito/2, min(curvaLuz_3)-0.001, 1.001])                       
pyplot.show()