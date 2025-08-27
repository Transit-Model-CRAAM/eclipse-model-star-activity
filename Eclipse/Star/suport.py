# cria Estrela
estrela_5 = Estrela(raio_estrela_pixel, raio_estrela, intensidade_maxima, coeficiente_um, coeficiente_dois, tamanho_matriz, useFits = True, fits_path="2022-10-01")
tamanho_matriz = estrela_5.getTamanhoMatriz()

Nx = estrela_5.getNx() 
Ny = estrela_5.getNy()
dtor = np.pi/180.  

# cria Planeta
angulo_inclinacao = 91.51  # em graus
planeta_ = Planeta(semi_eixo_UA, raio_plan_Jup, periodo, angulo_inclinacao, ecc, anomalia, estrela_5.getRaioSun(), mass_planeta)

estrela_5matriz = estrela_5.getMatrizEstrela()
estrela_5.Plotar(tamanho_matriz, estrela_5matriz)

# Eclipse com CME 

eclipse_ = Eclipse(Nx, Ny, raio_estrela_pixel, estrela_5, planeta_, 1)
estrela_5.Plotar(tamanho_matriz, estrela_5matriz)

tempoHoras = 1
eclipse_.geraTempoHoras(tempoHoras)
eclipse_.criarEclipse(anim=True)

print ("Tempo Total (Trânsito):", eclipse_.getTempoTransito()) 
tempoTransito = eclipse_.getTempoTransito()
curvaLuz_4 = eclipse_.getCurvaLuz()
tempoHoras = eclipse_.getTempoHoras()
#Plotagem da curva de luz 
pyplot.plot(tempoHoras, curvaLuz_4)
pyplot.axis([-tempoTransito/2, tempoTransito/2, min(curvaLuz_4)-0.001, 1.001])                       
pyplot.show()