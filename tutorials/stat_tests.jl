# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Julia Curso_datos_Julia 1.10.0
#     language: julia
#     name: julia-curso_datos_julia-1.10
# ---

# ## Pruebas de hipótesis
#
# ### $P(x^* \leq q(\alpha))$, Rechazas H0 donde $q()$ es el cuantil 
#
# ### Usando el criterio "p"
#
# ### $P(x^*)=cdf(x^*)=p$  
#
# ### Si $p \leq \alpha$ Rechazas H0
##################################################
# ESTE ES UN SCRIPT DE JULIA, 
#
#NO ES UN NOTEBOOK  DE PLUTO. 
#
#SE OBTUVO DE CONVERTIR EL PYTHON NOTEBOOK
# DEL MISMO NOMBRE. LA IDEA ES QUE VEAN
#CUALQUIERA DE ELLOS Y TOMEN LAS PARTES QUE LES
##SEAN ÚTILES PARA HACER LA TAREA. LES PONGO ALGUNOS
#COMENTARIOS ESPERANDO LES SIRVAN
#
#
##################################################
using Distributions, Random, Statistics, StatsBase,Plots; gr()

# +
#####################################################LIN58---LIN88 
#Es un ejemplo de tipo Montecarlo definiendo  "un estadístico" 
#calcular límites de confianza y determinar si aceptan o 
#rechazan hipótesis respecto al "origen" (distribución) de dicho estadístico. #Toman dos distribuciones Uniformes:
#una (Hip Nula) genera datos aleatorios entre [0,.1] (dist0). 
#La otra entre [0,mact] (dist1). 
#El Estadístico es la diferencia  entre el valor máximo y
#el valor mínimo de  muestras de 10 números (n) calculadas 
#con la función "ts". Se genera la estadística/distribucion de probabilidad
#de la hipótesis nula tomando 10^7 muestras de 10 números usando dist0. 
#En este caso calculan el valor qalpha (cuantil alpha) para alpha=0.05 (5%) que les dice: que la integral de -infinito a qalpha de la pdf de sus datos es =0.05.
#Prob( x <= qalpha )=0.05. Después toman una muestra del estadístico de la dist1.
#calculan la probabilidad de que ese estadístico provenga de dist0. Si el valor del estadístico es mayor que qalpha aceptan la hipótesis de que viene de
#dist0  si es menor a qalpha la rechazan con nivel de conf de 0.05 (95%): 
#hay 5% de probabilidad o menos de que la muestra venga de dist0. Pueden calcular
#el valor p de su muestra es decir la probabilidad de que dicho valor haya venido
#de dist0, Si esta probabilidad es menor de 0.05 rechazan la hipótesis.


Random.seed!(123)

n, N, alpha = 10, 10^7, 0.05
mact = 0.5
dist0, dist1 = Uniform(0,1), Uniform(0,mact) # define distribucuiones 

ts(muestra) = maximum(muestra) - minimum(muestra) #función que define el estadístico

empDistH0 = [ts(rand(dist0,n)) for _ in 1:N] # genera 10ˆ7 muestras del estadístico usando dist0
rejectVal = quantile(empDistH0,alpha) #define el cuantil alpha. valor tal que
 #Prob( x <= qalpha )=0.05

muestra = rand(dist1,n) #toma muestra de dist1
testStat = ts(muestra) #calcula estadístico
pValue = sum(empDistH0 .<= testStat)/N # Calcula valor p esta intruccion 
#primero checa cuántos de los 10^7 valores del estadístico estimados con dist0
#son menores al valor de la muestra tomada de dist1 pone un 1 donde si es menor y cero donde no es. La probabilidad (valor p) se calcula sumando todos los casos que si es menor y dividiende entre el total de casos (N=10^7)

if testStat > rejectVal
    print("No Rechazo : ", round(testStat,digits=4))
    print(" > ", round(rejectVal,digits=4))
else
    print("Rechaza: ", round(testStat,digits=4))
    print(" <= ", round(rejectVal,digits=4))
end
println("\np-value = $(round(pValue,digits=4))")

stephist(empDistH0, bins=100, c=:blue, normed=true, label="")  #Calcula hsitograma de dist0 empírico 
plot!([testStat, testStat], [0,4], c=:red, label="Prueba estadística Observada")
plot!([rejectVal, rejectVal], [0,4], c=:black, ls=:dash,
	label="Valor crítico (frontera) ", legend=:topleft, ylims=(0,4),
    	xlabel = "x", ylabel = "PDF")

##############################################################################
		# +
#import Pkg; Pkg.add("StatsBase")
#Pkg.add("LaTeXStrings")
#using StatsBase, LaTeXStrings

# +

###Lineas 113-137
# En este problema  nos dan una boya que pesa 17.5 kg 
#y se quiere determinar si esa boya es parte de un grupo de 
#boyas que tiene media=15kg,std=2 o de otro con media=18kg, std=2
#cdf es la función de dist de prob cumulativa ccdf = 1 - cdf (complemento)
#ccdf(dist0,tau)= probabilidad de tener valores mayores a tau=17.5 kg
#con dist0 y cdf(dist1,tau) es la prob de tener un valor menor a tau con la
#dist1. Esta probabilidad denominada beta te dice el error de aceptar H0
#y resulte que es falsa. Lo que se quiere es que beta sea pequeña y por tanto
#su complemento sea grande porque se quiere que la prob de estar en la zona
#marcada en rojo sea grande para la hipótesis 1 (dist1)

using StatsBase, LaTeXStrings;

mu0, mu1, sd, tau  = 15, 18, 2, 17.5
dist0, dist1 = Normal(mu0,sd), Normal(mu1,sd)
grid = 5:0.1:25
h0grid, h1grid = tau:0.1:25, 5:0.1:tau

println("Prob de error Tipo I (Rechazo H0 y es verdadera): ", ccdf(dist0,tau))
println("Prob de error Tipo II (Acepto H0 y es falsa): ", cdf(dist1,tau))

plot(grid, pdf.(dist0,grid),
	c=:blue, label="Boyas tipo 15kg")
plot!(h0grid, pdf.(dist0, h0grid), 
	c=:blue, fa=0.2, fillrange=[0 1], label="")
plot!(grid, pdf.(dist1,grid), 
	c=:green, label="Boyas tipo 18kg")
plot!(h1grid, pdf.(dist1, h1grid), 
	c=:green, fa=0.2, fillrange=[0 1], label="")
plot!([tau, 25],[0,0],
	c=:red, lw=3, label="Zona de Rechazo", 
	xlims=(5, 25), ylims=(0,0.25) , legend=:topleft,
    xlabel="x", ylabel="Densidad")
annotate!([(16, 0.02, text(L"\beta")),(18.5, 0.02, text(L"\alpha")),
            (15, 0.21, text(L"H_0")),(18, 0.21, text(L"H_1"))])
# -

# ## Intervalos de confianza para la varianza
#
# ### $P(\chi_{\frac{\alpha}{2},n-1}^2 \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi_{\frac{1-\alpha}{2},n-1}^2)= 1-\alpha$
#
#
# ### $\frac{(n-1)S^2}{\chi_{\frac{1-\alpha}{2},n-1}^2} \leq \sigma^2 \leq \frac{(n-1)S^2}{\chi_{\frac{\alpha}{2},n-1}^2}$
#
# ###  La varianza de la muestra tiene una distribución $\chi^2$
#
# ### $S^2 = \frac{1}{N-1}\sum_{1}^{N} (X_{i} - \bar{X})^2$ 
#
# ### $(n-1)*\frac{S^2}{\sigma^2} \approx \chi^2(n-1)$
#
# ### Si la distribución de $x$ es Normal!
#
# ### Comparar con Distribución Logística parecida a la Normal

# +
## DEMOSTRACIÓN  DE QUE LAS DISTRIBUCIONES DE LA VARIANZA
#VARÍAN DEPENDIENDO DE LA DISTRIBUCIÓN DE PROBABILIDAD DE LOS DATOS ORIGINALES
#USANDO MÉTODOS DE MONTECARLO 
mu, sig = 2, 3
eta = sqrt(3)*sig/pi
n, N = 15, 10^6
dNormal   = Normal(mu, sig)
dLogistic = Logistic(mu, eta)
xGrid = -8:0.1:12
varGrid = 0:.1:30

sNormal   = [var(rand(dNormal,n)) for _ in 1:N]
sLogistic = [var(rand(dLogistic,n)) for _ in 1:N]

p1 = plot(xGrid, pdf.(dNormal,xGrid), c=:blue, label="Normal")
p1 = plot!(xGrid, pdf.(dLogistic,xGrid), c=:red, label="Logistic", 
	xlabel="x",ylabel="Density", xlims= (-8,12), ylims=(0,0.16))

p2 = stephist(sNormal, bins=200, c=:blue, normed=true, label="Normal")
p2 = stephist!(sLogistic, bins=200, c=:red, normed=true, label="Logistic", 
	xlabel="Sample Variance", ylabel="Density", xlims=(0,30), ylims=(0,0.14))
p2 = plot!(varGrid,pdf.((sig^2/(n-1))*Chisq(n-1),varGrid),c=:black,label="Χ²(n-1)(σ²/h-1)")

plot(p1, p2, size=(800, 400))
# -

#
# ###  Para mayor demostración de que los siguientes intervalos aplican a distribución Normal
#
# ### $P(\chi_{\frac{\alpha}{2},n-1}^2 \leq \frac{(n-1)S^2}{\sigma^2} \leq \chi_{\frac{1-\alpha}{2},n-1}^2)= 1-\alpha$
#
#
# ### $\frac{(n-1)S^2}{\chi_{\frac{1-\alpha}{2},n-1}^2} \leq \sigma^2 \leq \frac{(n-1)S^2}{\chi_{\frac{\alpha}{2},n-1}^2}$
#
# ### Siguiente script toma muchas muestras de las distribuciones Normal y Logística calcula cuántas veces 
#
# ### el intervalo calculado contiene la varianza de la población.  Se estima $\alpha$ de la fórmula, pues 
# ### el número de veces que intervalo contiene a $\sigma^2$  de la población dividido entre el número total de pruebas tiene una probabilidad igial a $1- \alpha$. De ahí puedes calcular $\alpha$  y comparar contra el valor de $\alpha$  utilizado. Sólo en el caso de la Gaussiana se obtiene una línea recta

# +
mu, sig = 2, 3
eta = sqrt(3)*sig/pi
n, N = 15, 10^4
dNormal   = Normal(mu, sig)
dLogistic = Logistic(mu, eta)
alphaUsed = 0.001:0.001:0.1

function alphaSimulator(dist, n, alpha)
    popVar        = var(dist)
    coverageCount = 0
    for _ in 1:N
        sVar = var(rand(dist, n))
        L = (n - 1) * sVar / quantile(Chisq(n-1),1-alpha/2)
        U = (n - 1) * sVar / quantile(Chisq(n-1),alpha/2)
        coverageCount +=  L < popVar && popVar < U
    end
    1 - coverageCount/N
end
## SCRIPT PARA MOSTRAR QUE INTERVALOS DE CONFIANZA SIMULADOS Y TEÓRICOS
##SOLO COINCIDEN CUANDO LA DISTRIBUCION (PDF) ORIGINAL ES NORMAL

scatter(alphaUsed, alphaSimulator.(dNormal,n,alphaUsed), 
	c=:blue, msw=0, label="Normal")
scatter!(alphaUsed, alphaSimulator.(dLogistic, n, alphaUsed), 
	c=:red, msw=0, label="Logistic")
plot!([0,0.1],[0,0.1],c=:black, width=3, label="1:1 slope", 
	xlabel=L"\alpha"*" used", ylabel=L"\alpha"*" actual", 
	legend=:topleft, xlim=(0,0.1), ylims=(0,0.2))
# -

# ## Carga Librerías para leer matlab, probar hipótesis, etc

Pkg.add("MAT")
Pkg.add("Dates")
Pkg.add("Measures")
Pkg.add("HypothesisTests")
using MAT
using Dates
using Measures
using HypothesisTests

# ### Carga los datos de velocidad del anclaje

# +
vars=matread("/Users/julios/JULIA/curso_datos_julia/tutorials/LR5309_PM11_489m_h.mat")
v=vars["v"];u=vars["u"];time=vars["t"];prof=vars["pb"];temp=vars["tem"];
data1=v[:,20];data2=v[:,20]+randn(length(v[:,20])).*1.e-0;

#sampleData=temp[1:Int64(ceil(length(temp)/2))];
# -

# ### Para convertir fechas de matlab

# +

###LECTURA DE DATOS DE MATLAB Y FORMAS DE TRANSFORMAR FECHAS
#	const MATLAB_EPOCH = Dates.DateTime(-0001,12,31)
	const MATLAB_EPOCH = Dates.DateTime(-1,12,31)

"""
     datenum(d::Dates.DateTime)
Converts a Julia DateTime to a MATLAB style DateNumber.
MATLAB represents time as DateNumber, a double precision floating
point number being the the number of days since January 0, 0000
Example
    datenum(now())
"""
function datenum(d::Dates.DateTime)
    Dates.value(d - MATLAB_EPOCH) /(1000 * 60 * 60 * 24)
end
date2num(d::Dates.DateTime) = Dates.value(d-MATLAB_EPOCH)/(1000*60*60*24)
num2date(n::Number) =  MATLAB_EPOCH + Dates.Millisecond(round(Int64, n*1000*60*60*24))
# -

#LO QUE SIGUE SIRVE PARA SEPARAR DATOS EN COLUMNAS CON DATOS DIARIOS Y
#PARA LUEGO SACAR PROMEDIOS DIARIOS Y GENERAR UNA SERIE CON ELLOS
fecha=num2date.(time[21:end-13]);
temp2=temp[21:end-13];
length(fecha)/(24*7);
time2=time[21:end-13]
time2=reshape(time2,24,93*7);
temp2=reshape(temp2,24,93*7);
mtemp2=mean(temp2,dims=1);
mtime2=mean(time2,dims=1);
fecha2=num2date.(mtime2);
fecha2c=Dates.format.(fecha2,"yy-mm-dd")
include("julian_day.jl")
#date_example = Date(2024, 2, 29)
#datetime_example = DateTime(2024, 2, 29, 12, 30, 0)
#println("Julian day number for $date_example: ", julian_day_number(date_example))
#println("Julian date for $datetime_example: ", julian_date(datetime_example))
date_example = Date(2024, 2, 29)
datetime_example = DateTime(2024, 2, 29, 12, 30, 0)

##INTENTO PARA USAR DIAS JULIANOS EN JULIA CON SU CALENDARIO 
julian_date(date_example)
julian_date(datetime_example)
Dates.julian2datetime(julian_date(datetime_example))
Dates.julian2datetime(julian_date(fecha2[1]))
fecha2[1]
size(mtemp2)
Dates.DateTime(fecha2[1])


fecha2[1:10]

# +
#ESTO SIRVE PARA SACAR PROMEDIOS MENSUALES
months2=Dates.month.(fecha2);
sample = [months2 .==l for l=1:12];
daysm=[sum(sample[l]) for l in 1:12];
monthlym2=[sum(mtemp2.*sample[l])/daysm[l] for l in 1:12];

plot([1:12],monthlym2)
# -

sample[2]

months2

plot(vec(fecha2),vec(mtemp2))

fecha=num2date.(time)
plot(fecha,temp)

sampleData=temp[1:Int64(ceil(length(temp)/2))];
sampleData=mtemp2;
size(sampleData)

# +

#LO SIGUIENTE USA MÉTODO BOOTSTRAP PARA GENERAR DATOS CON MISMA ESTADÍSTICATION
#PARA ESTIMAR INTERVALOS DE CONFIANZA ES LO QUE NECESITAN PARA LA TAREA!!!
Random.seed!(0)

n, N = length(sampleData), 10^4
alpha = 0.05

bootstrapSampleMeans = [mean(rand(sampleData, n)) for i in 1:N]
Lmean = quantile(bootstrapSampleMeans, alpha/2)
Umean = quantile(bootstrapSampleMeans, 1-alpha/2)

bootstrapSampleMedians = [median(rand(sampleData, n)) for i in 1:N]
Lmed = quantile(bootstrapSampleMedians, alpha/2)
Umed = quantile(bootstrapSampleMedians, 1-alpha/2)

println("Bootstrap confidence interval for the mean: ", (Lmean, Umean) )
println("Bootstrap confidence interval for the median: ", (Lmed, Umed) )

stephist(bootstrapSampleMeans, bins=100, c=:blue,
    normed=true, label="Sample \nmeans")

plot!([Lmean, Lmean],[0,20], c=:black, ls=:dash, label="95% CI")
plot!([Umean, Umean],[0,20],c=:black, ls=:dash, label="",
  xlims=(8.7,9.0), xlabel="Sample Means", ylabel="Density")

#plot!([Lmean, Lmean],[0,2], c=:black, ls=:dash, label="90% CI")
#plot!([Umean, Umean],[0,2],c=:black, ls=:dash, label="",
#    xlims=(52,54), ylims=(0,2), xlabel="Sample Means", #ylabel="Density")

# -

#####LO SIGUIENTE SE RELACIONA CON LA PRUEBA DE KOLMOGOROV-SMIRNOV PARA  COMPARAR PDF DE DATOS GENERADOS CON DOS DISTRIBUCIONES DIFERENTES.
# SE CALCULA DIFERENCIA ENTRE DISTRIBUCIÓN EMPÍRICA Y TEÓRICA Y LUEGO SE DEMUESTRA QUE LA DISTRIBUCIÓN DE KOLMOGOROV LAS APROXIMA BASTANTE BIEN A LAS DOS 
# ## Kolmogorov-Smirnov

# +
#using Distributions, StatsBase, HypothesisTests, Plots, Random; pyplot()
Random.seed!(0)

n = 25
N = 10^4
xGrid = -10:0.001:10
kGrid = 0:0.01:5
dist1, dist2 = Exponential(1), Normal()

function ksStat(dist)
    data = rand(dist,n)
    Fhat = ecdf(data)
    sqrt(n)*maximum(abs.(Fhat.(xGrid) - cdf.(dist,xGrid)))
end

kStats1 = [ksStat(dist1) for _ in 1:N]
kStats2 = [ksStat(dist2) for _ in 1:N]

p1 = stephist(kStats1, bins=50, 
	c=:blue, label="KS stat (Exponential)", normed=true)
p1 = plot!(kGrid, pdf.(Kolmogorov(),kGrid), 
	c=:red, label="Kolmogorov PDF", xlabel="K", ylabel="Density")

p2 = stephist(kStats2, bins=50, 
	c=:blue, label="KS stat (Normal)", normed=true)
p2 = plot!(kGrid, pdf.(Kolmogorov(),kGrid), 
	c=:red, label="Kolmogorov PDF", xlabel="K", ylabel="Density")

plot(p1, p2, xlims=(0,2.5), ylims=(0,1.8), size=(800, 400))
# -

plot(vec(mtemp2.-mean(mtemp2)))

# +
#using Random,Distributions,StatsBase,Plots,HypothesisTests,Measures; pyplot()

## ESTA ES LA APLICACIÓN DE KOLMOGOROV-SMIRNOV A LOS DATOS DE ADCP
#DE TEMPERATURA (ESTANDARIZADOS (MEDIA=0/VARIANZA=1)) Y  PROBAR SI UNA PDF NORMAL CON STD=1 Y OTRA CON STD = 0.75 AJUSTA MEJOR LOS DATOS
Random.seed!(3)
data=vec((mtemp2.-mean(mtemp2))./std(mtemp2))
distH0 = Normal(0.0,0.8)
dist = Normal(mean(data),1.0)
n = Int64(ceil(length(data)))
#data = rand(dist,n)

Fhat = ecdf(data)
diffF(dist, x) = sqrt(n)*(Fhat(x) - cdf.(dist,x))
xGrid = -1.5:0.001:1.5
ksStat1 = maximum(abs.(diffF(distH0, xGrid)))

M = 10^5
KScdf(x) = sqrt(2pi)/x*sum([exp(-(2k-1)^2*pi^2 ./(8x.^2)) for k in 1:M])

println("p-value calculated via series: ",
	1-KScdf(ksStat1))
println("p-value calculated via Kolmogorov distribution: ",
	1-cdf(Kolmogorov(),ksStat1),"\n")

println(ApproximateOneSampleKSTest(data,distH0))

p1 = plot(xGrid, Fhat(xGrid), 
	c=:black, lw=1, label="ECDF from data")
p1 = plot!(xGrid, cdf.(dist,xGrid), 
	c=:blue, ls=:dot, label="CDF under \n alternative distribution")
p1 = plot!(xGrid, cdf.(distH0,xGrid), 
	c=:red, ls=:dot, label="CDF under \n postulated H0", 
	xlims=(-1.5,1.5), xlabel = "x", ylabel = "Probability")

p2= plot(cdf.(dist,xGrid), diffF(dist, xGrid),lw=0.5, 
	c=:blue,	label="KS Process under \n actual distribution")
p2 = plot!(cdf.(distH0,xGrid), diffF(distH0, xGrid), lw=0.5, 
	c=:red, xlims=(0.0,1.5), label="KS Process under \n postulated H0",
    xlabel = "t", ylabel = "K-S Process")

plot(p1, p2, legend=:bottomright, size=(800, 400), margin = 5mm)
# -
####COVARIANZAS y EJES PRiNCIPALES
m1=length(δv3[:,1]); #45
n1=length(δv3[1])
v2=v[21:end-13,:];
u2=u[21:end-13,:];
u3=[reshape(u2[:,i],24,93*7) for i=1:m1];
v3=[reshape(v2[:,i],24,93*7) for i=1:m1];
δu3=[mean(u3[i],dims=1) for i in 1:m1];
δv3=[mean(v3[i],dims=1) for i in 1:m1];
δv3=[δv3[i] .- mean(δv3[i]) for i in 1:m1];
δu3=[δu3[i] .- mean(δu3[i]) for i in 1:m1];
kk=[cov([δu3[i];δv3[i]]',[δu3[i];δv3[i]]') for i=1:m1];
using LinearAlgebra
eigval1,eigvec1=eigen(kk[40])
include("VectorPlot.jl")
scatter(vec(δu3[40]),vec(δv3[40]),aspect_ratio=:equal)
VectorPlot.plot_vec((0,0),Tuple(sqrt(eigval1[1]).*eigvec1[:,1]),lc=:black,lw=5)
VectorPlot.plot_vec((0,0),Tuple(sqrt(eigval1[2]).*eigvec1[:,2]),lc=:black,lw=5)
kkcov=[cov([δu3[i];δv3[i]]') for i in 1:m1]
include("get_ellipse_points.jl")
plot!(getellipsepoints([0 0], kkcov[40]; confidence=0.95),lw=5)

include("plot_ellipse_from_cov.jl")
plot_ellipse_from_covariance_matrix(kkcov[40],factor=[1,2,3],aspect_ratio=:equal,lw=5)
scatter!(vec(δu3[40]),vec(δv3[40]),aspect_ratio=:equal,markersize=1.5)
include("plot_ellipse_from_cov2.jl")
plot_ellipse_from_covariance_matrix2(vec(δu3[40]),vec(δv3[40]),factor=[1,2,3],aspect_ratio=:equal,lw=5)
size(kkcov)
m1
n1
include("plot_ellipse_from_cov.jl")
plot_ellipse_from_covariance_matrix(kkcov[40],factor=[1,2,3],lw=3)
scatter!(vec(δu3[40]),vec(δv3[40]),aspect_ratio=:equal,markersize=1.5)
# +
""""
Abajo se presenta la función de Julia para calcular intervalos de confianza para  la correlación entre dos series de datos usando el método de bootstrap:
"""

	using Statistics
	
	function bootstrap_correlation(data1::Vector, data2::Vector, n_bootstrap::Int, alpha::Float64)
	
	
	"""	Se calcula la correlación entre `data1` y `data2` se escribe la correlación original y el intervalo de confianza del 95%. Se pueden ajustar los parámetros `n_bootstrap` y `alpha` de acuerdo al análisis que se quiera hac
	"""
	

		# Calculate correlation coefficient from original data
		original_correlation = cor(data1, data2)
		
		# Bootstrap resampling
		bootstrap_correlations = Float64[]
		n = length(data1)
		for _ in 1:n_bootstrap
			# Resample with replacement
			indices = rand(1:n, n)
			resampled_data1 = data1[indices]
			resampled_data2 = data2[indices]
			
			# Calculate correlation coefficient from resampled data
			correlation = cor(resampled_data1, resampled_data2)
			push!(bootstrap_correlations, correlation)
		end
		
		# Sort bootstrap correlations
		sort!(bootstrap_correlations)
		
		# Calculate confidence intervals
		lower_index = round(Int, (alpha / 2) * n_bootstrap)
		upper_index = round(Int, (1 - alpha / 2) * n_bootstrap)
		lower_ci = bootstrap_correlations[lower_index]
		upper_ci = bootstrap_correlations[upper_index]
		
		return original_correlation, lower_ci, upper_ci
	end

	
#FIJARSE CÓMO SE GENERAN LAS MUESTRAS ALEATORIAS! 
#DATOS MUY SEPARADOS EN IEMPO NO TIENEN POR QUÉ ESTAR
#CORRELACIONADOS!
#Por ejemplo:
    include("Corr_Bstrap.jl")
	using LinearAlgebra
	function squeeze(A::AbstractArray)
		B=dropdims(A, dims=Tuple(findall(size(A) .== 1)))
		return B
	end
	data1 = vec(δv3[40])
	data2 =vec(δv3[30]) 
	prof1=prof[40]
	prof2=prof[30]
	n_bootstrap = 10000
	alpha = 0.05

	plot(squeeze(fecha2),data1,label="prof = $prof1 m",xlabel="date",ylabel="meridional velocity m/sec",size(800,300))
	plot!(squeeze(fecha2),data2,label="prof = $prof2 m")
	
	correlation, lower_ci, upper_ci = bootstrap_correlation(data1, data2, n_bootstrap, alpha)
	println("Coeficiente de correlación: ", correlation)
	println("Intervalo de confianza de 95% : ($lower_ci, $upper_ci)")
	include("Corr_Bstrap2.jl")
	bootstrap_correlation2(data1, data2, n_bootstrap, alpha)
	#Series serially correlated
	include("Corr_Bstrap3.jl")
	stationary_bootstrap_correlation(data1, data2,50, n_bootstrap, alpha)
##############################
# +
using NCDatasets
ds = NCDataset("aviso_LC_2018.nc")
ds["adt"]
adt=reshape(ds["adt"],28*52,365)
# Find indices of NaN values
adt_nan_indices = findall(isnan, adt[:,1])

# Find indices of non-NaN values
adt_non_nan_indices = findall(!isnan, adt[:,1])

adt2=adt[adt_non_nan_indices,:]
size(adt2)
adt2=adt2';
size(adt2)
madt2=mean(adt2,dims=1);
δssh=adt2 .- madt2
ma,na=size(δssh)
mean(δssh,dims=1)
U,S,V=svd(δssh/sqrt(ma-1),full=false)
size(V)
scatter(S)
varianza=(S.*S)./sum(S.*S)
vareof1=round(varianza[1]*100)
vareof2=round(varianza[2]*100)
eof=zeros((length(adt[:,1]),4))*NaN
size(eof)
[eof[adt_non_nan_indices,i]=V[:,i] for i in 1:4]
size(eof)
eof[adt_nan_indices]
lon=ds["longitude"]
lat=ds["latitude"]
ssheof1=reshape(eof[:,1],28,52)
ssheof2=reshape(eof[:,2],28,52)
p1 = contour(lon,lat,S[1].*ssheof1',fill=true,size=(400,600));
p2 = contour(lon,lat,S[1].*ssheof2',fill=true,size=(400,600));
p3 = plot(U[:,1]*sqrt(ma-1),grid=true,label="PC1 = $vareof1%");
p4 = plot(U[:,2]*sqrt(ma-1),grid=true,label="PC2 = $vareof2%");
plot(p1,p2,p3,p4, layout = (2,2) )

mean(U[:,1])