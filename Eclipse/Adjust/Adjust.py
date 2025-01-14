from scipy import interpolate
import emcee
from Planet.Planeta import Planeta
from Star.Estrela import Estrela #estrela e eclipse:: extensões de programas auxiliares que realizam o cálculo da curva de luz.
from Planet.Eclipse import Eclipse
from Misc.Verify import converte
import numpy

class Ajuste:
    
    def __init__(self,tratamento, time, flux, nwalkers, niter, burnin, rsun = 1, periodo = 1):
        self.u1_p0 = 0.5
        self.u2_p0 = 0.1
        self.a_p0 = 0.05
        self.inc_p0 = 88.
        self.rp_p0 = 1
        self.rsun = rsun
        self.periodo = periodo

        self.time = time
        self.flux = flux

        self.flux_err = numpy.var(self.flux)
        self.data = (self.time, self.flux, self.flux_err)

        self.nwalkers = nwalkers
        self.niter = niter
        self.burnin = burnin

        self.initial = numpy.array([self.u1_p0, self.u2_p0, self.a_p0, self.inc_p0, self.rp_p0])
        self.ndim = len(self.initial)

        variations = numpy.array([0.001, 0.001, 0.001, 0.5, 0.01])

        self.p0 = [numpy.array(self.initial) + variations * numpy.random.randn(self.ndim) for i in range(self.nwalkers)]

        self.tratamento = tratamento

    #--------------------------------------------------#
    #----------------------MCMC------------------------#
    #--------------------------------------------------#
    def eclipse_mcmc(self, time, theta):
        u1, u2, semiEixoUA, anguloInclinacao, raioPlanJup = theta

        raioStar, raioPlanetaRstar, semiEixoRaioStar = converte(self.rsun,raioPlanJup,semiEixoUA)
        
        estrela_ = Estrela(373, self.rsun, 240., u1, u2, 856)
        Nx = estrela_.getNx()
        Ny = estrela_.getNy()
        raioEstrelaPixel = estrela_.getRaioStar()
        
        # semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anom, raioStar,mass): 
        planeta_ = Planeta(semiEixoUA, raioPlanJup, self.periodo, anguloInclinacao, 0, 0, estrela_.getRaioSun(), self.tratamento.mass)
        
        # Nx, Ny, raio_estrela_pixel, estrela_manchada: Estrela, planeta_: Planeta
        eclipse = Eclipse(Nx,Ny,raioEstrelaPixel,estrela_, planeta_)
        
        eclipse.setTempoHoras(1.)
        eclipse.criarEclipse(anim = False, plot= False)
        lc0 = numpy.array(eclipse.getCurvaLuz()) 
        ts0 = numpy.array(eclipse.getTempoHoras()) 
        return interpolate.interp1d(ts0,lc0,fill_value="extrapolate")(time)
        
    #--------------------------------------------------#
    def lnlike(self, theta, time, flux, flux_err):
        return -0.5 * numpy.sum(((flux - self.eclipse_mcmc(time, theta))/flux_err) ** 2)
    #--------------------------------------------------#
    def lnprior(self, theta):
        u1, u2, semiEixoUA, anguloInclinacao, rp = theta
        if 0.0 < u1 < 1.0 and 0.0 < u2 < 1.0 and 0.001 < semiEixoUA < 1 and 80. < anguloInclinacao < 90 and 0.01 < rp < 5:
            return 0.0
        return -numpy.inf
    #--------------------------------------------------#
    def lnprob(self, theta, time, flux, flux_err):
        lp = self.lnprior(theta)
        if not numpy.isfinite(lp):
            return -numpy.inf
        return lp + self.lnlike(theta, time, flux, flux_err)
    #--------------------------------------------------#
    def main(self):
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=self.data)

        print("Running burn-in...")
        self.p0, _, _ = self.sampler.run_mcmc(self.p0, self.burnin, progress=True)
        self.sampler.reset()

        print("Running production...")    
        self.pos, self.prob, self.state = self.sampler.run_mcmc(self.p0, self.niter, progress=True)

        return self.sampler, self.pos, self.prob, self.state
    #--------------------------------------------------#

class AjusteManchado: 
    def __init__(self,tratamento, time, flux, nwalkers, niter, burnin, ndim, eclipse: Eclipse, rsun = 1, periodo = 1):
        
        self.manchas: Estrela.Mancha = eclipse.estrela_.manchas

        self.u1 = eclipse.estrela_.coeficienteHum
        self.u2 = eclipse.estrela_.coeficienteDois
        self.semiEixoUA = eclipse.planeta_.semiEixoUA
        self.anguloInclinacao = eclipse.planeta_.anguloInclinacao
        self.raioPlanJup = eclipse.planeta_.raioPlanJup

        self.rsun = rsun
        self.periodo = periodo

        self.time = time
        self.flux = flux

        self.flux_err = numpy.var(self.flux)
        self.data = (self.time, self.flux, self.flux_err)

        self.nwalkers = nwalkers
        self.niter = niter
        self.burnin = burnin


        self.initial = numpy.array([])

        # limitacao do numero de manchas
        if(ndim > len(self.manchas)):
            ndim = len(self.manchas)
        elif(ndim < 1):
            ndim = 1
            
        for i in range(ndim):
            self.initial = numpy.append(self.initial, [self.manchas[i].longitude, self.manchas[i].latitude, self.manchas[i].raio, self.manchas[i].intensidade])

        variations = numpy.array([0.8, 0.8, 0.001, 0.01])

        ndim_variations = numpy.tile(variations, ndim)
        
        self.ndim = len(self.initial)
        self.p0 = [numpy.array(self.initial) + ndim_variations * numpy.random.randn(self.ndim) for i in range(self.nwalkers)]
        self.tratamento = tratamento

    #--------------------------------------------------#
    #----------------------MCMC------------------------#
    #--------------------------------------------------#
    def eclipse_mcmc(self, time, theta):
        raioStar, raioPlanetaRstar, semiEixoRaioStar = converte(self.rsun,self.raioPlanJup,self.semiEixoUA)
        
        estrela_ = Estrela(373, self.rsun, 240., self.u1, self.u2, 856)
        Nx = estrela_.getNx()
        Ny = estrela_.getNy()
        raioEstrelaPixel = estrela_.getRaioStar()
        
        
        for i in range(len(theta)//4):
            # TO-DO: Mudar essa função que esta sendo chamada aqui 
            # intensidade, raio, latitude, longitude
            raioRStar = theta[(i*4)+2]
            intensidade = theta[(i*4)+3]
            lat = theta[(i*4)+1]
            long = theta[i*4]

            estrela_.addMancha(Estrela.Mancha(intensidade, raioRStar, lat, long))
        
        estrela_.criaEstrelaManchada()
        # semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anom, raioStar,mass): 
        planeta_ = Planeta(self.semiEixoUA, self.raioPlanJup, self.periodo, self.anguloInclinacao, 0, 0, estrela_.getRaioSun(), self.tratamento.mass)
        
        # Nx, Ny, raio_estrela_pixel, estrela_manchada: Estrela, planeta_: Planeta
        eclipse = Eclipse(Nx,Ny,raioEstrelaPixel,estrela_, planeta_)

        eclipse.setTempoHoras(1.)
        eclipse.criarEclipse(anim = False, plot= False)
        
        lc0 = numpy.array(eclipse.getCurvaLuz())
        ts0 = numpy.array(eclipse.getTempoHoras()) 
        return interpolate.interp1d(ts0,lc0,fill_value="extrapolate")(time)
    #--------------------------------------------------#
    def lnlike(self, theta, time, flux, flux_err):
        return -0.5 * numpy.sum(((flux - self.eclipse_mcmc(time, theta))/flux_err) ** 2)
    #--------------------------------------------------#
    def lnprior(self, theta):
        for i in range(len(theta)//4):
            #if (0.0 < lat) and (0.0 < long) and (0.0 < raioRstar < 0.5) and (0.0 < intensidade <= 1):
            if (-70 <= theta[i*4] <= 70) and (-70 <= theta[(i*4)+1] <= 70) and (0.0 < theta[(i*4)+2] < 0.5) and (0.0 < theta[(i*4)+3] <= 1):
                continue
            return -numpy.inf
        return 0.0
    #--------------------------------------------------#
    def lnprob(self, theta, time, flux, flux_err):
        lp = self.lnprior(theta)
        if not numpy.isfinite(lp):
            return -numpy.inf
        return lp + self.lnlike(theta, time, flux, flux_err)
    #--------------------------------------------------#
    def main(self):
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=self.data)

        print("Running burn-in...")
        self.p0, _, _ = self.sampler.run_mcmc(self.p0, self.burnin, progress=True)
        self.sampler.reset()

        print("Running production...")    
        self.pos, self.prob, self.state = self.sampler.run_mcmc(self.p0, self.niter, progress=True)

        return self.sampler, self.pos, self.prob, self.state
    #--------------------------------------------------#

class AjusteCME: 
    def __init__(self,tratamento, time, flux, nwalkers, niter, burnin, ndim, eclipse: Eclipse, rsun = 1, periodo = 1):
        self.raio_cme = 50
        self.p0x = 400
        self.p0y = 220
        self.p1x = 410
        self.p1y = 250
        self.opacidade = 0.3
        # self.taxa_esfriamento = 1.5

        try:
            self.manchas: Estrela.Mancha = eclipse.estrela_.manchas
        except:
            self.manchas = []

        self.u1 = eclipse.estrela_.coeficienteHum
        self.u2 = eclipse.estrela_.coeficienteDois
        self.raio = eclipse.estrela_.raio
        self.semiEixoUA = eclipse.planeta_.semiEixoUA
        self.anguloInclinacao = eclipse.planeta_.anguloInclinacao
        self.raioPlanJup = eclipse.planeta_.raioPlanJup
        self.tamanhoMatriz = eclipse.estrela_.tamanhoMatriz

        self.rsun = rsun
        self.periodo = periodo

        self.time = time
        self.flux = flux

        self.flux_err = numpy.var(self.flux)
        self.data = (self.time, self.flux, self.flux_err)

        self.nwalkers = nwalkers
        self.niter = niter
        self.burnin = burnin

        self.initial = numpy.array([self.raio_cme, self.p0x, self.p0y, self.p1x, self.p1y, self.opacidade])
        self.ndim = len(self.initial)

        variations = numpy.array([1, 1, 1, 1, 1, 0.01])

        self.ndim = len(self.initial)
        self.p0 = [
            numpy.array(
                [
                    *(numpy.array(self.initial[:5]) + variations[:5] * numpy.random.randn(5)).astype(int), 
                    self.initial[5] + variations[5] * numpy.random.randn()
                ]
            )
            for _ in range(self.nwalkers)
        ]
        self.tratamento = tratamento

    # --------------------------------------------------#
    # ----------------------MCMC------------------------#
    # --------------------------------------------------#
    def eclipse_mcmc(self, time, theta):
        try:
            raioStar, raioPlanetaRstar, semiEixoRaioStar = converte(self.rsun,self.raioPlanJup,self.semiEixoUA)

            estrela_ = Estrela(self.raio, self.rsun, 240., self.u1, self.u2, 856)
            Nx = estrela_.getNx()
            Ny = estrela_.getNy()
            raioEstrelaPixel = estrela_.getRaioStar()

            # Adicionando manchas (Se existirem)
            for mancha in self.manchas:
                raioRStar = mancha.raio
                intensidade = mancha.intensidade
                lat = mancha.latitude
                long = mancha.longitude

                estrela_.addMancha(Estrela.Mancha(intensidade, raioRStar, lat, long))

            estrela_.criaEstrelaManchada()

            # Adicionando CME
            temperatura_cme = estrela_.temperaturaEfetiva
            raio_cme = int(theta[0])
            p0x = int(theta[1])
            p0y = int(theta[2])
            p1x = int(theta[3])
            p1y = int(theta[4])
            opacidade = theta[5]
            velocidade_cme = 0.1
            taxa_esfriamento = 1.5

            cme = Estrela.EjecaoMassa(raio_cme, p0x, p0y, p1x, p1y, opacidade, temperatura_cme, velocidade_cme, taxa_esfriamento)

            estrela_.addCme(cme)

            # semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anom, raioStar,mass):
            planeta_ = Planeta(self.semiEixoUA, self.raioPlanJup, self.periodo, self.anguloInclinacao, 0, 0, estrela_.getRaioSun(), self.tratamento.mass)

            # Nx, Ny, raio_estrela_pixel, estrela_manchada: Estrela, planeta_: Planeta
            eclipse = Eclipse(Nx,Ny,raioEstrelaPixel,estrela_, planeta_)

            eclipse.setTempoHoras(1.)
            eclipse.criarEclipse(anim = False, plot= False)

            lc0 = numpy.array(eclipse.getCurvaLuz())
            ts0 = numpy.array(eclipse.getTempoHoras()) 
            return interpolate.interp1d(ts0,lc0,fill_value="extrapolate")(time)
        except:
            raioStar, raioPlanetaRstar, semiEixoRaioStar = converte(self.rsun,self.raioPlanJup,self.semiEixoUA)

            estrela_ = Estrela(self.raio, self.rsun, 240., self.u1, self.u2, 856)
            Nx = estrela_.getNx()
            Ny = estrela_.getNy()
            raioEstrelaPixel = estrela_.getRaioStar()

            # Adicionando manchas (Se existirem)
            for mancha in self.manchas:
                raioRStar = mancha.raio
                intensidade = mancha.intensidade
                lat = mancha.latitude
                long = mancha.longitude

                estrela_.addMancha(Estrela.Mancha(intensidade, raioRStar, lat, long))

            estrela_.criaEstrelaManchada()

            
            temperatura_cme = estrela_.temperaturaEfetiva
            raio_cme = self.raio_cme
            p0x = self.p0x
            p0y = self.p0y
            p1x = self.p1x
            p1y = self.p1y
            opacidade = self.opacidade
            velocidade_cme = 0.1
            taxa_esfriamento = 1.5

            cme = Estrela.EjecaoMassa(raio_cme, p0x, p0y, p1x, p1y, opacidade, temperatura_cme, velocidade_cme, taxa_esfriamento)

            estrela_.addCme(cme)

            # semiEixoUA, raioPlanJup, periodo, anguloInclinacao, ecc, anom, raioStar,mass):
            planeta_ = Planeta(self.semiEixoUA, self.raioPlanJup, self.periodo, self.anguloInclinacao, 0, 0, estrela_.getRaioSun(), self.tratamento.mass)

            # Nx, Ny, raio_estrela_pixel, estrela_manchada: Estrela, planeta_: Planeta
            eclipse = Eclipse(Nx,Ny,raioEstrelaPixel,estrela_, planeta_)

            eclipse.setTempoHoras(1.)
            eclipse.criarEclipse(anim = False, plot= False)

            lc0 = numpy.array(eclipse.getCurvaLuz())
            ts0 = numpy.array(eclipse.getTempoHoras()) 
            return interpolate.interp1d(ts0,lc0,fill_value="extrapolate")(time)
    # --------------------------------------------------#
    def lnlike(self, theta, time, flux, flux_err):
        return -0.5 * numpy.sum(((flux - self.eclipse_mcmc(time, theta))/flux_err) ** 2)
    # --------------------------------------------------#
    def lnprior(self, theta):
        for i in range(len(theta)//4):
            # TODO: Descobrir valores mínimo e máximo de cada uma das variáveis
            if (
                (0 <= theta[0] <= self.raio)
                and (0 <= theta[1] <= self.tamanhoMatriz)
                and (0 <= theta[2] <= self.tamanhoMatriz)
                and (0 <= theta[3] <= self.tamanhoMatriz)
                and (0 <= theta[4] <= self.tamanhoMatriz)
                and (0 <= theta[5] <= 1)
            ):
                continue
            return -numpy.inf
        return 0.0
    # --------------------------------------------------#
    def lnprob(self, theta, time, flux, flux_err):
        lp = self.lnprior(theta)
        if not numpy.isfinite(lp):
            return -numpy.inf
        return lp + self.lnlike(theta, time, flux, flux_err)
    # --------------------------------------------------#
    def main(self):
        self.sampler = emcee.EnsembleSampler(self.nwalkers, self.ndim, self.lnprob, args=self.data)

        print("Running burn-in...")
        self.p0, _, _ = self.sampler.run_mcmc(self.p0, self.burnin, progress=True)
        self.sampler.reset()

        print("Running production...")    
        self.pos, self.prob, self.state = self.sampler.run_mcmc(self.p0, self.niter, progress=True)

        return self.sampler, self.pos, self.prob, self.state
    # --------------------------------------------------#
