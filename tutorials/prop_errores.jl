### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 85b6d6ab-5c52-41f0-b2e9-7e6373cf43b2
using Distributions,Measurements,Plots, PlutoUI, Unitful, StatsPlots, DataFrames, LinearAlgebra, Latexify

# ╔═╡ 8be6ad87-d63b-472b-8666-58e6b14f019d
using Combinatorics, StatsBase

# ╔═╡ 7b7ddff0-087c-4205-bf82-5b9199b1aaac
begin

	using Random
Random.seed!(1)

function chabelo(cambia)
    regalo, escoge = rand(1:3), rand(1:3)
    if regalo == escoge
        revelap = rand(setdiff(1:3,escoge))
    else
        revelap = rand(setdiff(1:3,[regalo,escoge]))
    end

    if cambia
        escoge = setdiff(1:3,[revelap,escoge])[1]
    end
    return escoge == regalo
end

NP = 10^6
println("probabilidad de éxito si no cambias I (quedas): ", 
	sum([chabelo(false) for _ in 1:NP])/NP)
println("Probabilidad de éxito si cambias  II (cambias): ", 
	sum([chabelo(true) for _ in 1:NP])/NP)
end

# ╔═╡ ab3bd1bb-412a-4395-aa16-6c27fa825600
md"""
### CIFRAS SIGNIFICATIVAS, EXACTITUD,PRECISIÓN, INCERTIDUMBRES Y PROPAGACIÓN DE ERRORES
"""

# ╔═╡ 5e02dff8-b168-4a7c-810d-d9962c428907
md"## Load Packages"

# ╔═╡ a2cd2cbe-08a5-410c-940a-323af467e05f
gr()

# ╔═╡ 13cdba4e-4f96-4997-80cd-2f643640e055
md"""
* ### Cifras significativass
* ### Algunas reglas
* ### Siempre escribe la estimación (medición) y su error
* ### Incertidumbre o error relativo
"""

# ╔═╡ fb7af13a-94ce-49db-9035-8cc125373ca7
md"""Consideren:\
900 vs 900.0 \
0.00054 vs 1.00054 \
1245.3 ± 10 \
30 ± 4.5
"""

# ╔═╡ 33281294-80f7-4fda-9235-aec8bb7f11fc
md"
###### Algunas reglas para redondear cifras significativas:

1. **Todos los dígitos diferentes de cero son significativos:** Por ejemplo, en el número 123, los tres dígitos son significativos.

2. **Los ceros entre dígitos diferentes de cero son significativos:** En el número 1002, los cuatro dígitos son significativos.

3. **Los ceros a la izquierda del primer dígito diferente de cero no son significativos:** Por ejemplo, en 0.00456, solo los dígitos 456 son significativos.

4. **Si escribes los ceros a la derecha de un número decimal estos serán siempre significativos:** En 10.200, los cuatro dígitos son significativos. Indicas que tienes una precisión de \pm 0.001 si no escribes explícitamente la incertidumbre.

5. **Redondea al final de tus cálculos** en forma consistente con tus mediciones y errores

Recuerda  conservar el número apropiado de cifras significativas para mantener la precisión adecuada en tus resultados.
"

# ╔═╡ 861d6ad1-624d-47ae-9f54-dfc434688082
md"""
* ### Suponiendo errores pequeños
* ### Desarrollo Taylor
* ### Todavía no consideramos que el error es aleatorio
* ### Incertidumbre o error relativo
"""

# ╔═╡ 6ceebaf0-241f-44d5-9e4c-16b634548093
md"#### Estimación y  propagación elemental de incertidumbres"

# ╔═╡ fc63e3ff-a1f2-4c3d-829d-6399ec8c7bd5
md"""
```math
\begin{eqnarray}
	x&=&x_e \pm \delta x\\
	y&=&y_e \pm \delta y\\
    a&=& x+y= x_e + y_e \pm (\delta x + \delta y)\\
    b&=& x-y= x_e - y_e \pm (\delta x + \delta y)\\
    p&=&y*x\\
    q&=&y/x\\
    p&=&(y_e \pm \delta y) * (x_e \pm \delta x)\\
    p&=& {y_e}*{x_e} (1 \pm \frac{\delta y}{|{y_e}|})(1 \pm \frac{\delta x}{|{x_e}|})\\
    p & \approx & p_e( 1 \pm [\frac{\delta y}{|{y_e}|} + \frac{\delta x}{|{x_e}|}])\\
    p&=& p_e \pm \delta p \\
    \frac{\delta p}{p_e}&=& {[\frac{\delta y}{|{y_e}|} + \frac{\delta x}{|{x_e}|}]}\\
	q&=&\frac{y_e \pm \delta y}{x_e \pm \delta x}\\
     q&\approx & \frac{y_e}{x_e} (1 \pm \frac{\delta y}{|{y_e}|})(1 \mp \frac{\delta x}{|{x_e}|})\\
q&=& q_e \pm \delta q \\
    \frac{\delta q}{|q_e|}&=& {[\frac{\delta y}{|{y_e}|} + \frac{\delta x}{|{x_e}|}]}\\

\end{eqnarray}
```
"""


# ╔═╡ bac670a4-3cf4-455c-ab78-3c2c208af7c0
md"""
Si consideramos errores aleatorios sumamos en cuadratura

```math
\begin{eqnarray}
	p&=&x \pm y \\
	\delta q &=& \sqrt{(\delta x)^2 + (\delta y)^2}\\
    q&=&x*y \\
    o\\
    q&=&\frac{y}{x} \\
\frac{\delta q}{|q|} &=& \sqrt{(\frac{\delta x}{x})^2+  (\frac{\delta y}{y})^2}\\
 \end{eqnarray}
```
"""

# ╔═╡ 10c8be85-5e1c-470f-b8fd-1ee9fefe7925
md"#### Comparemos las estimaciones con ambos métodos"

# ╔═╡ 7a2d399f-de57-4ade-a1fd-3d41b8281a84
begin
t=1.6 ± 0.1
h=(46.2/3.33) ± 0.3
δt=Measurements.uncertainty(t);
δh=Measurements.uncertainty(h);
hₑ=Measurements.value(h);
tₑ=Measurements.value(t);
g=2h/t^2;
gₑ=2hₑ/tₑ^2;
δgᵣ=δh/hₑ+(2δt/tₑ);
δg=gₑ*δgᵣ
end

# ╔═╡ ee9a4c25-6d0b-4565-a54f-3e8221723008
typeof(δgᵣ)

# ╔═╡ 7173722b-3c66-484e-a8a1-722986172bdf
md"Estimación, incertidumbre, cuadratura"

# ╔═╡ 42db76e4-1bd0-4e94-b741-2fda5f9510e3
typeof(δg)

# ╔═╡ 05aedf3d-bf7f-4d64-a2b1-a039715d23ea
[round(g), round(δg), round(gₑ)]

# ╔═╡ db095cec-628b-4cc6-85c5-7d41fdfffe62
md"##### Algunas reglas para reportar mediciones y sus incertidumbres:

1. **Expresar la Medición:**
   - Reportar el valor medido con las unidades correspondientes.
   - Utilizar el número correcto de cifras significativas basado en la precisión de tu medición.

2. **Incluir la Incertidumbre:**
   - Proporciona la incertidumbre asociada con tu medición. Puede ser desde una estimación sencilla del error o en forma de desviación estándar, error estándar (si reportas el promedio de varias mediciones)  o un intervalo de confianza. Veremos cómo calcular y hacer eso.

3. **Formato para Medición e Incertidumbre Combinadas:**
   - Expresa el resultado en la forma: $( \text{Valor Medido} \pm \text{Incertidumbre})$.
   - Por ejemplo: $( 25.4 \, \text{cm} \pm 0.2 \, \text{cm})$.

4. **Usar Unidades Coherentes:**
   - Asegúrate de que las unidades de la medición y la incertidumbre sean coherentes. Por lo general mismas unidades.

5. **Expresar la Incertidumbre Apropiadamente:**
   - Medición e incertidumbre deben ser consistentes en cuanto a número de cifras significativas
   - Comunicar la incertidumbre a un nivel de confianza (por ejemplo, intervalo de confianza del 95%).
   - Especificar el método/suposición utilizado para calcular o estimar la incertidumbre.

6. **Incluye la Precisión del Instrumento:**
   - Si es aplicable, considera y reportar la precisión del instrumento de medición.

7. **Proporciona Contexto:**
   - Indica claramente las condiciones bajo las cuales se realizó la medición (por ejemplo, temperatura, presión), ya que pueden afectar el resultado.

8. **Sigue Pautas Estándar:**
   - Apega a cualquier pauta o estándar específico de informes relevante para tu área.

9. **Documenta el Proceso de Medición:**
   - Detalla los procedimientos y metodologías utilizados para obtener la medición.

10. **Repite las Mediciones:**
    - Si es posible, presenta resultados basados en múltiples mediciones para demostrar consistencia y confiabilidad.

Recuerda que reportar de manera clara y precisa las mediciones y sus incertidumbres es crucial para la interpretación y comparación adecuada de resultados científicos o experimentales.
"

# ╔═╡ 71f6c7c4-bc22-41f8-8f60-a5cf32012293
md"""
### Fórmulas elementales de propagación de errores

\

**Regla 1**  

Si $x$ e $y$ tienen errores
aleatorios independientes $\delta{x}$ y $\delta{y}$, entonces el
error en la suma\\resta $z=x \pm y$ es

$\delta{z}=\sqrt{\ \delta{x}^2 + \ \delta{y}^2},.$

\

**Regla 2**

Si $x \  \text{e} \ y$  tienen errores aleatorios independientes
$\delta{x}$ y $\delta{y}$, entonces el error relativo en la multiplicación
$(z=xy)$ es

$\frac{\delta{z}}{z}=\sqrt{\left(\frac{\delta{x}}{x}\right)^2 + \left(\frac{\delta{y}}{y}\right)^2},.$

\

**Regla 3** 

Si $(z=f(x))$, donde $f()$ es una función contínua, entonces

$\delta{z}=f'(x)\delta{x}$

"""


# ╔═╡ 6bd1ab2f-48e3-486e-80c5-055ea25404c5
md"""

**Formula general para propagación del error**

Sean $(x_1, x_2,....,x_N)$ variables con incertidumbres
$(\delta{x_1},\delta{x_2},....,\delta{x_3})$. 

Supongamos que queremos determinar $q$, una función de 
$(x_1,x_2,...,x_N)$ y su incertidumbre

$q \pm \delta q=q(x_1, x_2,...,x_N) \pm \delta q((x_1, x_2,....,x_N)$  


La incertudumbre asociada a $q$ es entonces

$\delta{q}=\sqrt{\left( \frac{\partial{q}}{\partial{x_1}}\delta{x_1}\right)^2 + ... +
                  \left( \frac{\partial{q}}{\partial{x_N}}\delta{x_N}\right)^2}$
\

Cuando no hay correlación entre las variables $(x_1,x_2,...,x_N)$

Por ejemplo, si $(q=x_1+x_2)$,

entonces obtenemos la regla 1:

$\frac{\partial{q}}{\partial{x_1}}=1$

$\frac{\partial{q}}{\partial{x_2}}=1$

$\delta{q}=\sqrt{\delta{x_1}^2 + \delta{x_2}^2}$


"""

# ╔═╡ 22f6fe10-ac4d-4063-84a6-67601f61da38
md"""

Si $q=x_1*x_2$ entonces obtenemos la regla 2:

$\frac{\partial{q}}{\partial{x_1}}=x_2$

$\frac{\partial{q}}{\partial{x_2}}=x_1$

$\delta{q}=\sqrt{x^2_2\delta{x_1}^2 + x^2_1\delta{x_2}^2}=\sqrt{q^2\left[ \left(\frac{\partial{x_1}}{x_1}\right)^2 + \left(\frac{\partial{x_2}}{x_2}\right)^2\right]}$

$\frac{\delta{q}}{q}=\sqrt{\left(\frac{\delta{x_1}^2}{x_1}\right) + \left(\frac{\delta{x_2}^2}{x_2}\right)}$


"""

# ╔═╡ 29dbf31b-82eb-456e-9957-5172ad24b6c3
md"# Estadística básica y Probabilidad"

# ╔═╡ 2845e243-684e-4703-a9f3-a8098e61e204
md"""
#### Estadística básica

Con la estadística tratamos de obtener estimaciones de las propiedades
de una población a partir de muestras parciales de la misma. Hablamos de
población y de muestra de una población. Propiedades de una población 
como la media,desviación estándar,etc son **parámetros o propiedades** 'establecidas'
de dicha población, aunque no se conozcan. En estadística utilizamos estas muestras
y definimos estimadores o estadísticos de dichos parámetros basados en ellas. 

Las poblaciones o espacios muestrales se caracterizan por tener una función de densidad de probabilidad (pdf) también conocida como función de distribución de probabilidad, con la que calculamos

$Prob(a< x < b) = ∫_{a}^{b} ρ(x) dx$ 


que nos dice la probabilidad de que x esté en el intervalo $(a,b)$ .

Se tiene tanbién la función de distribución cumulativa $F(x)$ :


$F(x) = ∫_{-∞}^{x} ρ(x') dx'$ 


que nos dice la probabilidad de tener valores menores a $x$ y además


$\frac{dF}{dx} = ρ(x)$


La probabilidad está normalizada (lo que significa que siempre obtendremos un valor
de la varable al tomar una muestra)


$∫_{-∞}^{∞} ρ(x) dx = 1$


\


Con la densidad de probabilidad podemos calcular sus momentos definidos por 


```math
\begin{equation}
M_{n}(x) = ∫_{-∞}^{∞} x^n ρ(x) dx
\end{equation}

```




El primer momento se conoce como el valor esperado de una variable o función estadística:

```math
\begin{equation}
 M_{1}(x) = E(x) = ∫_{-∞}^{∞} x ρ(x) dx = μ 
\end{equation}

```

y nos da el valor medio  (promedio) de la variable.



Utilizando este valor se calculan los momentos 'centrados' alrededor de $μ$:

\

```math
\begin{equation}
M_{n}(x - μ) = ∫_{-∞}^{∞} (x - μ)^n ρ(x) dx
\end{equation}

```



De los cuales uno de los más importantes es el segundo momento conocido como varianza $σ^2$.

```math
\begin{equation}
σ^2 = E((x-μ)^2) = ∫_{-∞}^{∞} (x - μ)^2 ρ(x) dx
\end{equation}
```

Para cualquier función $a(x)$ definimos su valor esperado como:


$E(a(x)) = ∫_{-∞}^{∞} a(x) ρ(x) dx$ 

Una propiedad común que se busca tengan los estimadores estadísticos es que sean no sesgados. Esto significa que el "valor esperado" del estimador debe ser igual al parámetro o 'valor verdadero'  correspondiente de la población. Para demostrar muchas propiedades de los estimadores estadísticos no necesitamos hacer explícitamente las integrales. Con utilizar las propiedades de la integral y suponer que la muestra se toma de una población con pdf o momentos definidos se puede demostrar en forma sencilla, por ejemplo que:

```
#\begin{center}
#%\includegraphics[width=0.5\textwidth]{estadistica_poblacion.pdf}
#\end{center}
```

"""

# ╔═╡ f6ee567a-45ea-43dc-ae86-ac1c7e530cfd
md"""

##### 1.  Estimador de la media o promedio

Un estimador de la media de la población a partir de una muestra de N valores ($x_i=x_1,x_2,...,x_N$) es

```math
\begin{equation}
\bar{x}=\frac{1}{N}\sum^N_{i=1}x_i
\end{equation}
```

Es no sesgado porque

$E(\bar{x})= E(\frac{1}{N}\sum^N_{i=1}x_i)= \frac{1}{N}\sum^N_{i=1}E(x_i)=\frac{1}{N}\sum^N_{i=1} μ=\frac{N}{N} μ = μ$


Este rresultado se obtiene considerando que las variables $x_i$  provienen
de una población o espacio muestral con media $μ$.

La media debe de diferenciarse de la mediana. La media es el momento de
orden uno. La mediana de una población es aquel valor numérico que
separa el 50\% de valores mas altos del 50\% de valores mas bajos. Se
puede calcular ordenando de menor a mayor el conjunto de valores y
escoger el valor central si el conjunto de datos es impar o el promedio
de los dos centrales si es par. \


##### 2.  La varianza:

La varianza de un una muestra de N valores $(x_i)$ es

```math
\begin{equation}
s^2=\frac{1}{N-1}\sum^N_{i=1}(x_i-\bar{x})^2 = \frac{1}{N-1}\sum^N_{i=1} (x_i')^2,
\end{equation}
```

donde las primas indican fluctuaciones alrededor de la media. La
varianza es una medida de cuán lejos estan los diferentes puntos de la
muestra de la media de la población o de la muestra. La varianza es el segundo momento
alrededor de la media. Al dividir por $N$ estamos subestimando la
verdadera varianza de la población. Al dividir por $(N-1)$ obtenemos un
estimador no sesgado. \

```math
\begin{equation}
E(s^2)=\frac{1}{N-1}\sum^N_{i=1}E((x_i-\bar{x})^2) \,,
\end{equation}
```


EJERCICIO: Demostrar porqué hay que dividir por $(N-1)$ en lugar de
$N$ para que el estimador clásico  de la varianza sea un estimador no sesgado. \



##### 3.  La desviación típica: 

Es la raíz cuadrada de la varianza. Se suele escribir como $(\sigma)$
para referirse a la población o como $s$ para su estimación estadística


```math
\begin{equation}
s=\sqrt{s^2}\,.
\end{equation}
```


##### 4. Momentos de orden superior: 


Podemos definir un momento alrededor de la media como:

```math
\begin{equation}
m_p=\frac{1}{N}\sum^N_{i=1}(x_i-\bar{x})^p=<x'^p>\,.
\end{equation}
```


De esta forma $m_2$ es la varianza, $m_3$ es la asimetría, y $m_4$
la curtosis. El momento $(m_3)$ indica la asimetría de la muestra
alrededor de la media $(m_3>0)$ implica distribución con cola larga en
la parte positiva y viceversa. $(m_4)$ indica el grado de esparcimiento
de las muestras alrededor de la media. Una mayor curtosis indica mayor
concentración de puntos alrededor de la media. Los momentos de orden
superior ($(>2)$) se suelen adimensionalizar dividiendo por la
desviación estandar:

```math
\begin{equation}
    m_3=\frac{1}{N}\sum^N_{i=1}\left[\frac{x_i-\bar{x}}{\sigma}\right]^3=<(x/\sigma)'^3>
\end{equation}
```

```math
\begin{equation} 
    m_4=\frac{1}{N}\sum^N_{i=1}\left[\frac{x_i-\bar{x}}{\sigma}\right]^4-3=<(x/\sigma)'^4>-3
\end{equation}
```
donde el factor $-3$ hace que la curtosis tome el valor cero para una
distribución Normal.


"""

# ╔═╡ 56a415b4-1811-4f8c-8f0a-55defeec8d10
md""" 
#### Covarianza y correlación


La covarianza entre dos variables $(x)$ e $(y)$ puede definirse como un
estadístico que relaciona  $(x)$ e $(y)$ de la siguiente forma

```math
[C_{xy}=<x'y'>=<(x-\bar{x})(y-\bar{y})>=\frac{1}{N-1}\sum\limits^N_{i=1} (x_i-\bar{x})(y_i-\bar{y})\,.]
```

La correlación es la covarianza normalizada

```math
[\rho_{x y}=\frac{C_{x y}}{s_x s_y}=\frac{<x' y'>}{\sqrt{<x'^2><y'^2>}}\,.]
```

Consideremos el modelo estadístico lineal de media cero (es una recta
que pasa por $(\overline{x},\overline{y})=(0,0))$

```math
[\hat{y}=\alpha x\,,]
```

donde $(\alpha)$ es una constante. El error cometido por este estimador
se define como el error cuadrático medio

```math
[\epsilon^2=<(\hat{y}-y)^2>=\alpha^2<x^2>+<y^2>-2\alpha<xy>]
```

y si queremos minimizar dicho error entonces tenemos que encontrar que
$(\alpha)$ es el que provoca que la derivada

$(\partial{\epsilon^2}/\partial{\alpha}\rightarrow{0})$. 

Es decir

$[\partial{\epsilon^2}/\partial{\alpha}=2\alpha<x^2>-2<xy>=0\,]$

y  $(\alpha)$ es

$[\alpha=\frac{<xy>}{<x^2>}\,.]$

El error cuadrado mínimo se encuentra substituyendo el valor de
$(\alpha)$ en la expresión del error $(\epsilon^2)$ de arriba

```math
[\epsilon^2=\frac{<xy>^2}{<x^2>} + <y^2> - 2\frac{<xy>^2}{<x^2>}=
           <y^2>\left(\frac{<xy>^2}{<x^2><y^2>}+1-2\frac{<xy>^2}{<x^2><y^2>}\right)=\]

[=<y^2>(1-\rho^2_{xy})\,.]
```

Si $(\rho^2_{xy}=1)$ entonces el error es cero que es error más pequeño de todos.
Por el contrario, si $(\rho^2_{xy}=0)$ entonces el error es igual a la
varianza, es decir, máximo error. Si $(\rho)$ toma valores intermedios,
i.e., $(\rho^2_{xy}=0.5)$ entonces el error es $(\epsilon^2=0.5<y^2>)$,
es decir, el error del modelo lineal es un (50\%) de la varianza. Por
lo tanto, la correlación al cuadrado puede definirse también ciomo la
eficiencia relativa del estimador $(\hat{y}^2)$ o la fracción de
varianza explicada por el modelo lineal

```math
[\rho^2_{xy}=\frac{<\hat{y}^2>}{<y^2>}=\frac{{varianza-explicada}}{{varianza-total}}\,.]
```
A este parámetro se le puede encontrar en la literatura inglesa denominado como
**skill** del modelo lineal.

"""

# ╔═╡ 90779043-df48-4374-ae1f-c7861a72f96c
md"""

Si queremos relacionar los errores cuadráticos $\delta x_i^2$ con estimadores estadísticos (varianza y desviación estándar) debemos tener un poco de cuidado en la forma de hacer los cálculos.

Para no generar confusión en vez de $x_1$ y $x_2$ llamemos a estas variables $x \, y$.
por lo que $q=q(x,y)$. Supongamos ahora que hacemos $N$ medicones de $x$ y $y$ con las que ontenemos $N$ valores de $q$.

Ahora hacemos una aproximación de primer orden de q alrededor de su valor medio:

$q(x,y) \approx q(\bar{x},\bar{y}) + (\frac{\partial q}{\partial x}) \biggr\rvert_{\bar{x},\bar{y}} \delta x +  (\frac{\partial q}{\partial y})\biggr\rvert_{\bar{x},\bar{y}} \delta y$ 

Es decir:

$\bar{q} = q(\bar{x},\bar{y})$ 

$\delta x = x-\bar{x}  ,\,\,\,  \delta y = y-\bar{y}$

$\frac{1}{N}\sum\limits^N_{i=1} x_i = \bar{x}$

$\frac{1}{N}\sum\limits^N_{i=1} y_i = \bar{y}$

Ahora usamos el subíndice $i$ para denotar cada una de nuestras $N$ mediciones:

$x_i ,\, y_i => q_i = q(x_i,y_i)$ 

De acuerdo a la fórmula para estimar la varianza o desviación estandar con base a N mediciones tenemos:

$s_q^2=\frac{1}{N-1}\sum\limits^N_{i=1} (q_i - \bar{q})^2$

$s_q^2 = \frac{1}{N-1}\sum\limits^N_{i=1} \biggl[(\frac{\partial q}{\partial x}) \biggr\rvert_{\bar{x},\bar{y}} (x_i-\bar{x}) + (\frac{\partial q}{\partial y}) \biggr\rvert_{\bar{x},\bar{y}} (y_i-\bar{y}) \biggr]^2$

$s_q^2 = \biggl(\frac{\partial q}{\partial x} \biggr\rvert_{\bar{x},\bar{y}} \biggr)^2 \frac{1}{N-1}\sum\limits^N_{i=1} (x_i-\bar{x})^2  \\
 + \biggl(\frac{\partial q}{\partial y} \biggr\rvert_{\bar{x},\bar{y}}\biggr)^2 \frac{1}{N-1}\sum\limits^N_{i=1} (y_i-\bar{y})^2 +$



$\ldots 2(\frac{\partial q}{\partial x})(\frac{\partial q}{\partial y}) 

\frac{1}{N-1}\sum\limits^N_{i=1}(x_i-\bar{x})(y_i-\bar{y})$


$s_q^2 =  (\frac{\partial q}{\partial x} \biggr\rvert_{\bar{x},\bar{y}})^2  s_x^2 + 

(\frac{\partial q}{\partial y} \biggr\rvert_{\bar{x},\bar{y}})^2  s_y^2 + 

2(\frac{\partial q}{\partial x})(\frac{\partial q} {\partial y}) \biggr\rvert_{\bar{x},\bar{y}} s_{xy}$ 

Donde 

$s_{xy} = \frac{1}{N-1}\sum\limits^N_{i=1}(x_i-\bar{x})(y_i-\bar{y})$ 

Es nuestro estimador de la covarianza entre $x$  y $y$. Si las variables son independientes la  covarianza es cero (lo contrario no siempre se cumple i.e. covarianza cero no necesariamente implica independencia) y recuperames las fórmulas anteriores de suma en cuadraturas donde ls incertidumbres se identifican con las desviaciones estándar

"""

# ╔═╡ c23fc7a8-85d6-4b3f-a5da-8e21caff006e
md"""

Si nuestra función depende de $M$ variables podemos hacer el mismo cálculo que con
dos variables. Si lo hacemos con una muestra, esto significa que tenemos que hacer las estimaciones con las $N$ mediciones de cada una de las $M$ variables.
Pero vamos a simplificar el cálculo usando el operador de expectación. Los cálculos
peden hacerse de dos maneras: considerando que conocemos la media y la varianza de las poblaciones de las M variables, en cuyo caso el resultado se expresa usando dicha información al expresar los valores esperados. Si no conocemos los valores de la 
población usamos las fórmulas para calcularlos a partir de la muestra de $N$ mediciones.

*Demstración:*

Queremos estimar la desviación estándar de la función

$[q=q(x_1, x_2,....,x_M)]$

que depende de $M$ variables independientes $x_1$, $x_2$,
$\ldots.,x_M$. 

El desarrollo de Taylor a primer orden de la función $q$ alrededor
de la media $(\bar{q})$ se puede escribir:

```math 
[q - \bar{q}] \approx  [(x_1 - \bar{x}_1)\frac{\partial{q}}{\partial{x_1}}
    + (x_2 - \bar{x}_2) \frac{\partial{q}}{\partial{x_2}} + \\ 
\ldots  + (x_M - \bar{x}_M) \frac{\partial{q}}{\partial{x_M}}\,.]

```
La varianza de la función $q$ es:

```math


\begin{eqnarray}

s^2_q = \sum\limits^M_{i=1}E( q_i-\bar{q})^2 \\


s^2_q = E(\biggl[ (x_1 - \bar{x}_1) \frac{\partial{q}}{\partial{x_1}} +

(x_2 - \bar{x}_2) \frac{\partial{q}}{\partial{x_2}} \\

+ \dots + \\

(x_M - \bar{x}_M) \frac{\partial{q}}{\partial{x_M}} \biggr]^2)

\end{eqnarray}
```

```math

\begin{eqnarray}

s^2_q = E(x_1 - \bar{x}_1)^2 (\frac{\partial{q}}{\partial{x_1}})^2 + 
E(x_2 - \bar{x}_2)^2 (\frac{\partial{q}}{\partial{x_2}})^2 + \\
2 E((x_1 - \bar{x}_1) (x_2 - \bar{x}_2) )
\frac{\partial{q}}{\partial{x_1}}\frac{\partial{q}}{\partial{x_2}} + \ldots ] \\



s^2_q = [s^2_{x_1}(\frac{\partial{q}}{\partial{x_1}})^2 + 
   
s^2_{x_2}(\frac{\partial{q}}{\partial{x_2}})^2 + 

2 s_{{x_1}{x_2}}\frac{\partial{q}}{\partial{x_1}}\frac{\partial{q}}{\partial{x_2}} 
+ \ldots ]

\end{eqnarray}

```

Para  dos variables recuperamos la expresión para la propagación del error obtenida enteriormente si nuestros estimadores de media y varianza corresponden a los de la muestra:

```math

\begin{eqnarray}

s_{q} = \sqrt{s^2_{x_1}(\frac{\partial{q}}{\partial{x_1}})^2 +
    s^2_{x_2}(\frac{\partial{q}}{\partial{x_2}})^2+ 
    2s_{{x_1}{x_2}}\frac{\partial{q}}{\partial{x_1}}\frac{\partial{q}}{\partial{x_2}}  + \ldots}

\end{eqnarray}

```
#### PERO OJO!

Con $M$ variables podemos tener covarianza entre todas ellas de manera que a diferencia de dos variables donde hay un solo valor de la covarianza, con M variables ahora tenemos lo que se denomina la "Matriz de Covarianza" en la que nos aparece la covarianza de cada variable con todas las demás. Esta es una matriz simétrica porque la covarianza entre la variable $x_i$ y $x_j$  es igual a la que hay entre $x_j$ y $x_i$

$\sigma_{ij} = Cov(x_i,x_j)$

Calculada a partir de una muestra de $N$  valores:

$s_{ij}= \frac{1}{N-1}\sum\limits^N_{n=1} (x^{(n)}_i -\bar{x_i})(x^{(n)}_j - \bar{x_j})$


$s_q^2 = \sum\limits^M_{i,j=1}\frac{\partial{q}}{\partial{x_i}}\frac{\partial{q}}{\partial{x_j}}s_{ij}$


 
"""

# ╔═╡ 586fac47-31c8-41f0-a046-15599aa5bbda
# ╠═╡ disabled = true
#=╠═╡
md"""
## Ejemplo:

La ecuación para el cálculo de la salinidad a partir de la
conductividad $(C)$ y temperatura $(T)$ es (Unesco EOS-80)

$[S=a_0+a_1 R_T^{1/2}+a_2 R_T+a_3 R_T^{3/2}+a_4 R_T^{2}+a_5 R_T^{5/2}+\Delta{S}\,]$

donde

$[R_T=\frac{R}{R_p r_t}\,\,\,,\,\,R=\frac{C(S,T,0)}{C(35,15,0)}]$

$(C(35,15,0))$ es la conductividad de un agua de salinidad práctica 35 a los $(15^\circ{C})$,

\

$[r_t=c_0 + c_1 T + c_2 T^2 + c_3T^3 + c_4T^4]$

$[R_p=1+\frac{P(e_1+e_2P+e_3P^2)}{(1+d_1T+d_2T^2+(d_3 + d_4T)R)}]$

y

$[\Delta{S}=\frac{T-15}{1+k(T-15)}(b_0+b_1 R_T^{1/2}+b_2 R_T+b_3 R_T^{3/2}+b_4 R_T^{2}+b_5 R_T^{5/2})]$

con los coeficientes $(a_i\,,b_i\,,c_i\,,d_i)\,$ y $e_i~$

\

$(a_0=0.0080\,\,\,\,\,\,\,\,\,\,\,\,\,\,b_0=0.0005\,\,\,\,\,\,\,\,\,\,\,\,\,\,c_0=0.6766097\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,d_1=3.426\,e^{-2}\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,e_1=2.070\,e^{-5})$

\

$(a_1=-0.1692\,\,\,\,\,\,\,\,\,\,b_1=-0.0056\,\,\,\,\,\,\,\,\,c_1=2.00564\,e^{-2}\,\,\,\,\,\,\,\,\,\,\,d_2=4.464\,e^{-4}\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,e_2=-6.370\,e^{-10})$


$(a_2=25.3851\,\,\,\,\,\,\,\,\,\,\,\,b=-0.0066\,\,\,\,\,\,\,\,\,\,\,c_2=1.104259\,e^{-4}\,\,\,\,\,\,\,\,\,d_3=4.215\,e^{-1}\,\,\,\,\,\,\,\,\,\,\,\,\,\,\,e_3=3.989\,e^{-15})$

\

$(a_3=14.0941\,\,\,\,\,\,\,\,\,\,\,\,b=-0.0375\,\,\,\,\,\,\,\,\,\,\,c_3=-6.9698,e^{-7}\,\,\,\,\,\,\,\,\,\,\,d_4=-3.107,e^{-3})$
~
$(a_4=-7.0261\,\,\,\,\,\,\,\,\,\,b_4=0.0636\,\,\,\,\,\,\,\,\,\,\,\,\,\,c_4=1.0031\,e^{-9})$
\

$(a_5=2.7081\,\,\,\,\,\,\,\,\,\,\,\,\,\,b_5=-0.0144\,\,\,\,\,\,\,\,\,\,)$

\

$(\sum a_i=35.0000\,\,\,\,\sum b_i=0.0000)$ $(k=0.0162)$

Si el error de precisión del termistor y de la celda de
conductividad del CTD es $(\delta{T}=0.001^\circ{C})$ y
$(\delta{C}=0.001,{S}{ m}^{-1})$, respectivamente, calcule la
incertidumbre asociada al cálculo de la salinidad $S$ a partir de la
formula general de propagación del error para
$(C=5,{\rm S}\,{\rm m}^{-1})$, $(T=28\,^\circ{C})$, y
$(P=50 \, {dbar})$. Suponga que $(\delta{P}=0)$, es decir, el
altímetro no tiene errores de precisión.\

**Resultado**:

El error asociado a la salinidad es

$[\delta{S}=\sqrt{\left( \frac{\partial{S}}{\partial{T}}\delta{T} \right)^2 + \left( \frac{\partial{S}}{\partial{C}}\delta{C}\right)^2}]$

donde

$[\frac{\delta{S}}{\delta{T}}=
      \left[(1+k(T-15))^{-1} - (k(T-15))(1+k(T-15))^{-2}\right]
[\left (b_0+b_1 R_T^{1/2}+b_2 R_T+b_3 R_T^{3/2}+b_4 R_T^{2}+b_5 R_T^{5/2}\right)]$

y

$[\frac{\delta{S}}{\delta{C}}=
                 \frac{1}{2} a_1 \frac{R_T^{-1/2}}{C(35,T,0)} +
         a_2 \frac{1}{C(35,T,0)} +
             \frac{3}{2}a_3 \frac{R_T^{1/2}}{C(35,T,0)}+ ]

[+2 a_4 \frac{R_T}{C(35,T,0)} +\frac{5}{2} a_5 \frac{R_T^{3/2}}{C(35,T,0)}+\frac{T-15}{(1+k(T-15))}]

[\left(b_1 \frac{R_T^{-1/2}}{C(35,T,0)} +
         b_2 \frac{1}{C(35,T,0)} +
             \frac{3}{2}b_3 \frac{R_T^{1/2}}{C(35,T,0)}+
               2 b_4 \frac{R_T}{C(35,T,0)} +
     \frac{5}{2} b_5 \frac{R_T^{3/2}}{C(35,T,0)}
     \right)\,]$

Por lo tanto, dada una medida de conductividad $(C)$ y
temperatura $(T)$ podemos calcular $({\delta{S}}/{\delta{T}})$ y
$({\delta{S}}/{\delta{C}})$ y obtener la desviación estándar asociado al
cálculo de la salinidad $(\delta{S})$.

"""
  ╠═╡ =#

# ╔═╡ 7c97760b-0b64-440e-bd89-d71a17a9243b
md"# Distribuciones de Probabilidad "

# ╔═╡ b659ae06-3180-4c22-bdda-419444458ae3
md""" 
Tomado del Tutorial de Doggo.jl 

 <https://github.com/julia4ta/tutorials/tree/master/Series%2009>
"""

# ╔═╡ fca5675a-0063-45aa-8af9-f13069c4cff0
md"## Distribución Normal"

# ╔═╡ 18567e0d-8f29-4ea4-92fc-6fabab0d5bac
md"Mean (μ): $(@bind mu Slider(-3.0:0.1:3.0, 0.0, true))"

# ╔═╡ b7c2233a-7957-4347-83d9-6f87d521eb3b
md"Standard Deviation (σ): $(@bind sigma Slider(0.5:0.01:3.0, 1.0, true))"

# ╔═╡ c25a9157-60e8-4309-b1e8-82f95453cd95
begin
	plot(Normal(mu, sigma),
		legend = false,
		xlims = (-5, 5),
		ylims = (0, 1),
		linewidth = 2,
		fill = true,
		alpha = 0.3,
		title = "Probability Density Function"
	)
	vline!([mu])
	vline!([mu+sigma, mu-sigma])
	vline!([mu+2sigma, mu-2sigma])
end

# ╔═╡ 690b2f82-69da-4420-ac87-b25ce10bbfa6
md"## Distribución Beta"

# ╔═╡ e599ffd3-2ca7-42ee-b08f-7cf980d5a9be
md"alpha (α): $(@bind alpha Slider(0.1:0.1:5.0, 1.0, true))"

# ╔═╡ 786bb65f-fc11-4e54-9577-58cbae0abe91
md"beta (β): $(@bind beta Slider(0.1:0.1:5.0, 1.0, true))"

# ╔═╡ e2192613-78a6-4266-9cd1-cce417aeb25a
	#print(quantile.(Normal(mu,sigma), [0.5, .75]))
#median(Normal(mu,sigma))
	median(Beta(alpha,beta))

# ╔═╡ e29357ed-2e57-48cb-b788-d3069401096f
plot(Beta(alpha, beta),
	legend = false,
	xlims = (0, 1),
	ylims = (0, 5),
	linewidth = 2,
	fill = true,
	alpha = 0.3,
	title = "Función de Densidad de Probabilidad"
)

# ╔═╡ 759f4a2b-b7a1-49e4-99a6-03eebf4d0573
md"## Distribución Binomial"

# ╔═╡ 4b0456cf-29ae-4551-a704-7f5fa906df89
md"Número de intentos (n): $(@bind n Slider(1:1:10, 1, true))"

# ╔═╡ 144a4161-0a84-42f1-b674-946f0200101f
md"Probabilidad de éxitos (p): $(@bind p Slider(0.00:0.01:1.0, 0.5, true))"

# ╔═╡ 0901abad-c184-436d-910f-ae429c15d122
begin
	sticks(Binomial(n, p),
		linewidth = 3,
		legend = false,
		xticks = collect(-1:1:10),
		xlims = (0, 10),
		ylims = (0, 1.0),
		widen =  true,
		title = "Probability Mass Function"
	)
	scatter!(Binomial(n, p),
		markersize = 8,
	)
end

# ╔═╡ 79cad4dd-107a-429a-a10f-2c8dfc6c9b33
md"nu (ν): $(@bind nu Slider(1:20, 2, true))"

# ╔═╡ 0f017343-6c40-4149-829a-1ef9f3f9e917
plot(Chisq(nu),
	legend = false,
	xlims = (0, 20),
	ylims = (0, 0.5),
	linewidth = 2,
	fill = true,
	alpha = 0.3,
	title = "Función de Densidad de Probabilidad"
)

# ╔═╡ fe1e3bd5-800a-481d-a3fe-ec7bf0c25b5a
begin
	N1, faces = 10^6, 1:6

numSol = sum([iseven(i+j) for i in faces, j in faces]) / length(faces)^2
mcEst  = sum([iseven(rand(faces) + rand(faces)) for i in 1:N1]) / N1
println("Numerical solution = $numSol \nMonte Carlo estimate = $mcEst")
end

# ╔═╡ 898b8ab4-f384-4898-943c-35f6a7e264b1
begin
matchE1(n) = 1 - prod([k/365 for k in 365:-1:365-n+1])
matchE2(n) = 1- factorial(365,365-big(n))/365^big(n)
m1=matchE1(20)
m2=matchE2(20)
#println("matchExists1 = $m1 \nmatchExists2 = $m2")
function cumpEv(n)
	#escoge n num de 1:365 en forma aleatoria
    cumples = rand(1:365,n)
	 
	#lo mismo que diasCs = counts(cumples,1:365)
    diasCs = counts(cumples,365)
    #compara y cuenta las veces que se repite mismo num mas de una vex
    return maximum(diasCs) > 1
end
probEst(n,N) = sum([cumpEv(n) for _ in 1:N])/N
npers=1:50
mcEst1 = [probEst(n,10^4) for n in npers]
sol1=[matchE1(n) for n in npers]
plot(npers,sol1)
scatter!(npers,mcEst1, shape=:xcross,c=:red, #ms=6, msw=0, shape=:xcross, 
	label="MC", xlims=(0,50), ylims=(0, 1), 
	xlabel="Num de personas", 
	ylabel="Probabilidad de mismo cumple", 
	legend=:topleft)
end

# ╔═╡ 157302a9-1e7c-4c35-865f-626a9d2afd33
begin
comb(n,r)=factorial(big(n))/(factorial(big(r))*factorial(big(n-r)))
comb(50,2)
end

# ╔═╡ 79ed327b-8f5e-469f-8149-b9026ed19044
begin
n1 =10 ; p1=0.5
binomialRV(n,p) = sum(rand(n) .< p)
n1n=rand(n1)
sn1n=sum(n1n .< p1)
println("Random Numbers = $n1n \nCuantos, $sn1n")
#println(sum(rand(n1) .< p1)
#binomialRV(10,0.5)
end

# ╔═╡ 51a304e9-3983-4f5c-a103-6e1d1aa77ee4
begin
r1=rand(1:6,3)
	sr1=(r1 .<= 4)
	s1=sum(r1[sr1])
#s1=sum(r1 .<= 4)
println("numbers $r1 \nsumr1 = $s1" )
end

# ╔═╡ bf179bd1-4ee9-4dea-ad72-8a502c826354
begin
pp, nn, N = 0.5, 10, 10^6

bDist = Binomial(nn,pp)
xGrid = 0:nn
bPmf = [pdf(bDist,i) for i in xGrid]
data = [binomialRV(nn,pp) for _ in 1:N]
pmfEst = counts(data,0:nn)/N

plot( xGrid, pmfEst, 
	line=:stem, marker=:circle, 
	c=:blue, ms=10, msw=0, lw=4, label="MC estimate")
plot!( xGrid, bPmf, 
	 line=:stem, marker=:xcross, c=:red, 
	 ms=6, msw=1, lw=2, label="PMF", xticks=(0:1:10),
	 ylims=(0,0.3), xlabel="x", ylabel="Probability")
end

# ╔═╡ 2aa0813c-0129-40c9-bcd9-790ea3915365
data

# ╔═╡ 70313dca-3a16-4fe0-8c83-aee2705b335d
md"### Poisson"

# ╔═╡ 23112c98-43f5-47be-b48f-308bc9be40db
begin
function prn(lambda)
    k, p = 0, 1
    while p > MathConstants.e^(-lambda)
        k += 1
        p *= rand()
    end
    return k-1
end

xGridL, lambda, NL = 0:16, 5.5, 10^6

pDistL = Poisson(lambda)
bPmfL = pdf.(pDistL,xGridL)
dataL = counts([prn(lambda) for _ in 1:NL],xGridL)/NL

plot( xGridL, dataL, 
	line=:stem, marker=:circle, 
	c=:blue, ms=10, msw=0, lw=4, label="MC estimate")
plot!( xGridL, bPmfL, line=:stem, 
	marker=:xcross, c=:red, ms=6, msw=0, lw=2, label="PMF",
	ylims=(0,0.2), xlabel="x", ylabel="Probability of x events")
end

# ╔═╡ 5481c532-c4b6-426d-8ccf-f3695b758b6d
begin
#using Random, StatsBase
#Random.seed!(1)

A = Set(['a','e','i','o','u'])
#B = Set(['x','y','z'])
B = Set(['a','b','c'])
omega = 'a':'z'
AunionB = union(A, B)
AintersectionB = intersect(A, B)
BdifferenceA = setdiff(B,A)
Bcomplement = setdiff(omega,B)
AsymDifferenceB = union(setdiff(A,B),setdiff(B,A))
AsymDifferenceB2 = symdiff(A,B)
println("A = $A, B = $B")
println("A union B = $AunionB")
println("A intersección B = $AintersectionB")
println("B diff A = $BdifferenceA")
println("B complemento = $Bcomplement")
println("A Simdif B = $AsymDifferenceB")
println("A Symdif B = $AsymDifferenceB2")

println("The element ’c’ is an element of A: $(in("c",A))")
Nb = 10^6

println("mcEst1 \t \tmcEst2")
for _ in 1:5
    mcEst1 = sum([in(sample(omega),A) || in(sample(omega),B) for _ in 1:Nb])/Nb
    mcEst2 = sum([in(sample(omega),union(A,B)) for _ in 1:Nb])/Nb
    println(mcEst1,"\t",mcEst2)
end
end

# ╔═╡ 71d4f8d6-1e9c-4ca2-8699-bfb744049b3a
md"""" 
## USO TEOREMA BAYES

Considera $E_k$ disjuntos tales que $S = \bigcup_{0}^{\infty}{E_k}$

$P(B) = \sum{P(B ∩ E_k) } = \sum {P(B | E_k) P(E_k)}$

$P(E_k|B)=\frac{P(E_k|B)*P(E_k)}{P(B)}$ 

$P(B)=\sum {P(B | E_k) P(E_k)}$

"""

# ╔═╡ 1ba39eee-eb40-4fcf-9584-232b346b142d
md"### CHABELO"

# ╔═╡ a845eafb-6a5e-4a6a-85ee-123e7129547b
md"#### Pruebas Médicas"

# ╔═╡ 9c8b5503-628d-4e05-898e-52494bc1dd38
md"prob de enfermedad en la población (Penf): $(@bind Penf Slider(0.001:0.001:0.01, 0.001, true))"

# ╔═╡ 8b1137f0-debc-4d4b-a396-2a48cacaed36
md"prob de enfermedad en la poblacion (Ptestnenf): $(@bind Ptestnenf Slider(0.01:0.01:0.05, 0.01, true))"

# ╔═╡ f10da811-6e52-412d-9ad3-957059e93495
begin
#Penf=0.001
Pnoenf=1 - Penf
Ptest_enf=1.0
Pntest_enf=0.0
#Ptest_nenf=0.05
Penf_test=Ptest_enf*Penf/(Ptest_enf*Penf + Ptestnenf*Pnoenf)
println("probabilidad de tener enfermedad si test posotivo $Penf_test")
end


# ╔═╡ a74f6819-3520-4d7a-a1fd-a5577380048f
md"### TEOREMA LÍMITE CENTRAL"

# ╔═╡ dfa53385-d100-4b53-b04c-759b2955b8c2
begin
nc, Nc = 30, 10^6

dist1 = Uniform(1-sqrt(3),1+sqrt(3))
dist2 = Exponential(1)
dist3 = Normal(1,1)

data1 = [mean(rand(dist1,nc)) for _ in 1:Nc]
data2 = [mean(rand(dist2,nc)) for _ in 1:Nc]
data3 = [mean(rand(dist3,nc)) for _ in 1:Nc]

stephist([data1 data2 data3], bins=100, 
    c=[:blue :red :green], xlabel = "x", ylabel = "Density",
    label=["Average of Uniforms" "Average of Exponentials" "Average of Normals"], 
    normed=true, xlims=(0,2), ylims=(0,2.5))
end

# ╔═╡ c18fc658-27be-41a9-8fb6-2bc5b777b483


# ╔═╡ c06d581e-7e73-4524-be32-fed0518ca5ed
md"### Funciones de distribución bidimensional"

# ╔═╡ f01553ce-4999-40fb-9316-ba9d96f90bda
begin
meanVect = [27.155405405405407, 26.163835263835264] 
covMat = [16.1253895583728 13.046961200891614; 13.046961200891614 12.367253570765161]

biNorm = MvNormal(meanVect,covMat)

NB = 10^3
points = rand(MvNormal(meanVect,covMat),NB)

support = 15:0.5:40
z = [ pdf(biNorm,[x,y]) for y in support, x in support ]

p1b = scatter(points[1,:], points[2,:], ms=0.5, c=:black, legend=:none)
p1b = contour!(support, support, z, 
		levels=[0.001, 0.005, 0.02], c=[:blue, :red, :green], 
		xlims=(15,40), ylims=(15,40), ratio=:equal, legend=:none, 
		xlabel="x", ylabel="y")
p2b = surface(support, support, z, lw=0.1, c=cgrad([:blue, :red]),
		 legend=:none, xlabel="x", ylabel="y",camera=(-35,20))

plot(p1b, p2b, size=(800, 400))
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Combinatorics = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[compat]
Combinatorics = "~1.0.2"
DataFrames = "~1.6.1"
Distributions = "~0.25.107"
Latexify = "~0.16.1"
Measurements = "~2.11.0"
Plots = "~1.39.0"
PlutoUI = "~0.7.54"
StatsBase = "~0.34.2"
StatsPlots = "~0.15.6"
Unitful = "~1.19.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "a6714c16958a7554d90c51107b7b314af4300b37"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "66771c8d21c8ff5e3a93379480a2307ac36863f7"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.0.1"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b859a208b2397a7a623a03449e4636bdb17bcf2"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.16.1+1"

[[deps.Calculus]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f641eb0a4f00c343bbc32346e1217b86f3ce9dad"
uuid = "49dc2e85-a5d0-5ad3-a950-438e2897f1b9"
version = "0.5.1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "c1deebd76f7a443d527fc0430d5758b8b2112ed8"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.19.1"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "407f38961ac11a6e14b2df7095a2577f7cb7cb1b"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "67c1f244b991cad9b0aa4b7540fb758c2488b129"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.24.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "75bd5b6fc5089df449b5d35fa501c846c9b6549b"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.12.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.5+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "04c738083f29f86e62c8afc341f0967d8717bdb8"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.6.1"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "ac67408d9ddf207de5cfa9a97e114352430f01ed"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.16"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "66c4c81f259586e8f002eacebc177e1fb06363b0"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.11"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[deps.Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "7c302d7a5fec5214eb8a5a4c466dcf7a51fcf169"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.107"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "dcb08a0d93ec0b1cdc4af184b26b591e9695423a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.10"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4558ab818dcceaab612d1bb8c19cee87eda2b83c"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.5.0+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "b57e3acbe22f8484b4b5ff66a7499717fe1a9cc8"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.1"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "ec22cbbcd01cba8f41eecd7d44aac1f23ee985e3"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.7.2"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c6033cc3892d0ef5bb9cd29b7f2f0331ea5184ea"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FillArrays]]
deps = ["LinearAlgebra", "Random"]
git-tree-sha1 = "5b93957f6dcd33fc343044af3d48c215be2562f1"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.9.3"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Pkg", "Zlib_jll"]
git-tree-sha1 = "21efd19106a55620a188615da6d3d06cd7f6ee03"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.93+0"

[[deps.Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll"]
git-tree-sha1 = "ff38ba61beff76b8f4acad8ab0c97ef73bb670cb"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.3.9+0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "UUIDs", "p7zip_jll"]
git-tree-sha1 = "27442171f28c952804dede8ff72828a96f2bfc1f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.72.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "025d171a2847f616becc0f84c8dc62fe18f0f6dd"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.72.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "e94c92c7bf4819685eb80186d51c43e71d4afa17"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.76.5+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "344bf40dcab1073aca04aa0df4fb092f920e4011"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "abbbb9ec3afd783a7cbd82ef01dcd088ea051398"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.1"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "8d511d5b81240fc8e6802386302675bdf47737b9"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.4"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "d75853a0bdbfb1ac815478bacd89cd27b550ace6"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.3"

[[deps.InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "9cc2baf75c6d09f9da536ddf58eb2f29dedaf461"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "5fdf2fe6724d8caabf43b557b84ce53f3b7e2f6b"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.0.2+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "721ec2cf720536ad005cb38f50dbba7b02419a15"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.14.7"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "a53ebe394b71470c7f97c2e7e170d51df21b17af"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.7"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "7e5d6779a1e09a36db2a7b6cff50942a0a7d0fca"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.5.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d986ce2d884d49126836ea94ed5bfb0f12679713"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "15.0.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e5b909bcf985c5e2605737d2ce278ed791b89be6"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.1+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "f428ae552340899a935973270b8d98e5a31c49fe"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.1"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.4.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll", "Pkg"]
git-tree-sha1 = "64613c82a59c120435c067c2b809fc61cf5166ae"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.8.7+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "c333716e46366857753e273ce6a69ee0945a6db9"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.42.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f9557a255370125b405568f9767d6d195822a175"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9c30530bf0effd46e15e0fdcf2b8636e78cbbd73"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.35.0+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "2da088d113af58221c52828a80378e16be7d037a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.5.1+1"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "7d6dd4e9212aebaeed356de34ccf262a3cd415aa"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.26"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "c1dd6d7978c12545b4179fb6153b9250c96b0075"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.0.3"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "72dc3cf284559eb8f53aa593fe62cb33f83ed0c0"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.0.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Measurements]]
deps = ["Calculus", "LinearAlgebra", "Printf", "Requires"]
git-tree-sha1 = "bdcde8ec04ca84aef5b124a17684bf3b302de00e"
uuid = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
version = "2.11.0"

    [deps.Measurements.extensions]
    MeasurementsBaseTypeExt = "BaseType"
    MeasurementsJunoExt = "Juno"
    MeasurementsRecipesBaseExt = "RecipesBase"
    MeasurementsSpecialFunctionsExt = "SpecialFunctions"
    MeasurementsUnitfulExt = "Unitful"

    [deps.Measurements.weakdeps]
    BaseType = "7fbed51b-1ef5-4d67-9085-a4a9b26f478c"
    Juno = "e5e0dc1b-0480-54bc-9374-aad01c23163d"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MultivariateStats]]
deps = ["Arpack", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "68bf5103e002c44adfd71fea6bd770b3f0586843"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.2"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ded64ff6d4fdd1cb68dfcbb818c69e144a5b2e4c"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.16"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "6a731f2b5c03157418a20c12195eb4b74c8f8621"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.13.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "51901a49222b09e3743c65b8847687ae5fc78eb2"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.1"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc6e1927ac521b659af340e0ca45828a3ffc748f"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.12+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51a08fb14ec28da2ec7a927c4337e4332c2a4720"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.2+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+0"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "1f03a2d339f42dca4a4da149c7e15e9b896ad899"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.1.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Preferences", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "ccee59c6e48e6f2edf8a5b64dc817b6729f99eb5"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.39.0"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "bd7c69c7f7173097e7b5e1be07cee2b8b7447f51"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.54"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "03b4c25b43cb84cee5c90aa9b5ea0a78fd848d2f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00805cd429dcb4870060ff49ef443486c262e38e"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.1"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "88b895d13d53b5577fd53379d913b9ab9ac82660"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.1"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "37b7bb7aabf9a085e0044307e1717436117f2b3b"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.5.3+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9b23c31e76e333e6fb4c1595ae6afa74966a729e"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.9.4"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "f65dcb5fa46aee0cf9ed6274ccbd597adc49aa7b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.1"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ed52fdd3382cf21947b15e8870ac0ddbff736da"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.4.0+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "0e7508ff27ba32f26cd459474ca2ede1bc10991f"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "f68dd04d131d9a8a8eb836173ee8f105c360b0c5"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.1"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "36b3d696ce6366023a0ea192b4cd442268995a0d"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.2"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "1d77abd07f617c4868c33d4f5b9e1dbb2643c9cf"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.2"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "f625d686d5a88bcd2b15cd81f18f98186fdc0c9a"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.0"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "9115a29e6c2cf66cf213ccc17ffd61e27e743b24"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.6"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "cb76cf677714c095e535e3501ac7954732aeea2d"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.11.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "3c793be6df9dd77a0cf49d80984ef9ff996948fa"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.19.0"

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

    [deps.Unitful.weakdeps]
    ConstructionBase = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "e2d817cc500e960fdbafcf988ac8436ba3208bfd"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.3"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "fcdae142c1cfc7d89de2d11e08721d0f2f86c98a"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.6"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "5f24e158cf4cee437052371455fe361f526da062"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "0.5.6"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "801cbe47eae69adc50f36c3caec4758d2650741b"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.12.2+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "Pkg", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "91844873c4085240b95e795f692c4cec4d805f8a"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.34+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522b8414d40c4cbbab8dee346ac3a09f9768f25d"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.4.5+0"

[[deps.Xorg_libICE_jll]]
deps = ["Libdl", "Pkg"]
git-tree-sha1 = "e5becd4411063bdcac16be8b66fc2f9f6f1e8fe5"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.0.10+1"

[[deps.Xorg_libSM_jll]]
deps = ["Libdl", "Pkg", "Xorg_libICE_jll"]
git-tree-sha1 = "4a9d9e4c180e1e8119b5ffc224a7b59d3a7f7e18"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.3+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6035850dcc70518ca32f012e46015b9beeda49d8"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "34d526d318358a859d7de23da945578e8e8727b7"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "b7c0aa8c376b31e4852b360222848637f481f8c3"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.4+4"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "19560f30fd49f4d4efbe7002a1037f8c43d43b96"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.10+4"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8fdda4c692503d44d04a0603d9ac0982054635f9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "b4bfde5d5b652e22b9c790ad00af08b6d042b97d"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.15.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "49ce682769cd5de6c72dcf1b94ed7790cd08974c"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.5+0"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a68c9655fbe6dfcab3d972808f1aafec151ce3f8"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.43.0+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3a2ea60308f0996d26f1e5354e10c24e9ef905d4"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.4.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.8.0+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.52.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╠═ab3bd1bb-412a-4395-aa16-6c27fa825600
# ╠═5e02dff8-b168-4a7c-810d-d9962c428907
# ╠═85b6d6ab-5c52-41f0-b2e9-7e6373cf43b2
# ╠═a2cd2cbe-08a5-410c-940a-323af467e05f
# ╠═13cdba4e-4f96-4997-80cd-2f643640e055
# ╟─fb7af13a-94ce-49db-9035-8cc125373ca7
# ╟─33281294-80f7-4fda-9235-aec8bb7f11fc
# ╠═861d6ad1-624d-47ae-9f54-dfc434688082
# ╠═6ceebaf0-241f-44d5-9e4c-16b634548093
# ╠═fc63e3ff-a1f2-4c3d-829d-6399ec8c7bd5
# ╟─bac670a4-3cf4-455c-ab78-3c2c208af7c0
# ╟─10c8be85-5e1c-470f-b8fd-1ee9fefe7925
# ╠═7a2d399f-de57-4ade-a1fd-3d41b8281a84
# ╠═ee9a4c25-6d0b-4565-a54f-3e8221723008
# ╠═7173722b-3c66-484e-a8a1-722986172bdf
# ╠═42db76e4-1bd0-4e94-b741-2fda5f9510e3
# ╠═05aedf3d-bf7f-4d64-a2b1-a039715d23ea
# ╠═db095cec-628b-4cc6-85c5-7d41fdfffe62
# ╠═71f6c7c4-bc22-41f8-8f60-a5cf32012293
# ╠═6bd1ab2f-48e3-486e-80c5-055ea25404c5
# ╠═22f6fe10-ac4d-4063-84a6-67601f61da38
# ╟─29dbf31b-82eb-456e-9957-5172ad24b6c3
# ╟─2845e243-684e-4703-a9f3-a8098e61e204
# ╟─f6ee567a-45ea-43dc-ae86-ac1c7e530cfd
# ╟─56a415b4-1811-4f8c-8f0a-55defeec8d10
# ╠═90779043-df48-4374-ae1f-c7861a72f96c
# ╠═c23fc7a8-85d6-4b3f-a5da-8e21caff006e
# ╠═586fac47-31c8-41f0-a046-15599aa5bbda
# ╟─7c97760b-0b64-440e-bd89-d71a17a9243b
# ╟─b659ae06-3180-4c22-bdda-419444458ae3
# ╟─fca5675a-0063-45aa-8af9-f13069c4cff0
# ╠═18567e0d-8f29-4ea4-92fc-6fabab0d5bac
# ╠═b7c2233a-7957-4347-83d9-6f87d521eb3b
# ╠═c25a9157-60e8-4309-b1e8-82f95453cd95
# ╠═e2192613-78a6-4266-9cd1-cce417aeb25a
# ╠═690b2f82-69da-4420-ac87-b25ce10bbfa6
# ╠═e599ffd3-2ca7-42ee-b08f-7cf980d5a9be
# ╠═786bb65f-fc11-4e54-9577-58cbae0abe91
# ╠═e29357ed-2e57-48cb-b788-d3069401096f
# ╟─759f4a2b-b7a1-49e4-99a6-03eebf4d0573
# ╠═4b0456cf-29ae-4551-a704-7f5fa906df89
# ╠═144a4161-0a84-42f1-b674-946f0200101f
# ╠═0901abad-c184-436d-910f-ae429c15d122
# ╠═79cad4dd-107a-429a-a10f-2c8dfc6c9b33
# ╠═0f017343-6c40-4149-829a-1ef9f3f9e917
# ╠═fe1e3bd5-800a-481d-a3fe-ec7bf0c25b5a
# ╠═8be6ad87-d63b-472b-8666-58e6b14f019d
# ╠═898b8ab4-f384-4898-943c-35f6a7e264b1
# ╠═157302a9-1e7c-4c35-865f-626a9d2afd33
# ╠═79ed327b-8f5e-469f-8149-b9026ed19044
# ╠═51a304e9-3983-4f5c-a103-6e1d1aa77ee4
# ╠═bf179bd1-4ee9-4dea-ad72-8a502c826354
# ╠═2aa0813c-0129-40c9-bcd9-790ea3915365
# ╠═70313dca-3a16-4fe0-8c83-aee2705b335d
# ╠═23112c98-43f5-47be-b48f-308bc9be40db
# ╠═5481c532-c4b6-426d-8ccf-f3695b758b6d
# ╠═71d4f8d6-1e9c-4ca2-8699-bfb744049b3a
# ╠═1ba39eee-eb40-4fcf-9584-232b346b142d
# ╠═7b7ddff0-087c-4205-bf82-5b9199b1aaac
# ╠═a845eafb-6a5e-4a6a-85ee-123e7129547b
# ╠═9c8b5503-628d-4e05-898e-52494bc1dd38
# ╠═8b1137f0-debc-4d4b-a396-2a48cacaed36
# ╠═f10da811-6e52-412d-9ad3-957059e93495
# ╠═a74f6819-3520-4d7a-a1fd-a5577380048f
# ╠═dfa53385-d100-4b53-b04c-759b2955b8c2
# ╠═c18fc658-27be-41a9-8fb6-2bc5b777b483
# ╠═c06d581e-7e73-4524-be32-fed0518ca5ed
# ╠═f01553ce-4999-40fb-9316-ba9d96f90bda
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
