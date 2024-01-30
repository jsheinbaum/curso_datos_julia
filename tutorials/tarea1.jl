### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ 10166b16-91dc-4ff1-94b5-37c759c4121b
md"# TAREA 1 CURSO ANÁLISIS DE DATOS"

# ╔═╡ 72a753d8-b6ea-11ee-3203-235468905873
md"""

### Ejercicios sobre Cifras Significativas:

1. **Suma y Resta:**
   Realiza la operación $(23.45 + 6.2 - 0.123)$ con el número adecuado de cifras significativas.

2. **Multiplicación y División:**
   Calcula $(4.56 \times 2.1 / 3.789)$ con el número correcto de cifras significativas.

3. **Operaciones Mixtas:**
   Resuelve la expresión $(8.21 + 3.45) \times 2.0 / 4.567 )$ asegurándote de mantener la precisión adecuada.

4. **Conversión de Unidades:**
   Convierte 375 metros a kilómetros con la cantidad apropiada de cifras significativas.

5. **Potencias:**
   Eleva $(5.67)$ a la potencia $(3.2)$ y expresa el resultado con el número correcto de cifras significativas.

### Ejercicios sobre Propagación de Errores:

6. **Multiplicación de Cantidades con Incertidumbre:**
   Si $( A = 2.5 \pm 0.1 )$ y $( B = 3.0 \pm 0.2 )$, calcula $( C = A \times B )$ con la incertidumbre asociada.

7. **Suma de Cantidades con Incertidumbre:**
   Suma $( X = 15.4 \pm 0.3 )$ y $( Y = 8.6 \pm 0.2 )$ y determina la incertidumbre de $( Z = X + Y )$.

8. **División de Cantidades con Incertidumbre:**
   Si $(P = 20.0 \pm 0.5 )$ y $( Q = 4.0 \pm 0.2 )$, calcula $((R=P)/Q)$ con su respectiva incertidumbre.

9. **Operaciones Combinadas:**
   Si $(D = (3.2 \pm 0.1) \times (5.0 \pm 0.2) + (2.0 \pm 0.3))$, encuentra la incertidumbre asociada a $( D )$.

10. **Comparación de Resultados:**
   Realiza dos mediciones $( M_1 = 14.2 \pm 0.1 )$ y $( M_2 = 14.5 \pm 0.2 )$ y determina si hay una diferencia significativa entre ellas.




"""

# ╔═╡ 87135584-d489-4646-ac82-53e4b471f019
md"""

### Mas Ejercicios sobre Cifras Significativas:

11. Evalúa $(3.45 + 1.2) \times 4.67)$ con el número correcto de cifras significativas.
12. Realiza la operación $(8.96 \div (2.3 - 0.123))$ considerando cifras significativas.
13. Expresa $(560.0 {m} + 12.34 {m} )$ con el número adecuado de cifras significativas.
14. Calcula $(5.67 \times 2.1)^{3.4})$ con la precisión apropiada.
15. Convierte $32.0$°C a Kelvin manteniendo el número correcto de cifras significativas.
16. Realiza la suma $(14.2 + 0.456 + 3.1)$ con el número correcto de cifras significativas.
17. Expresa $(6.789 \times 10^4)$ en notación científica con cifras significativas.
18. Determina cuántas cifras significativas tiene el número $0.00723$.
19. Convierte $2.345$ km a metros con la precisión adecuada.
10. Eleva $( (2.0 \pm 0.1))$ a la potencia $(3.0 \pm 0.2)) $  con el número correcto de cifras significativas.


### Ejercicios sobre Errores Relativos:

### Productos:

21. **Producto Simple:**
   Si $( A = 3.0 \pm 0.2 )$ y $( B = 2.5 \pm 0.1 )$, calcula el producto $( C = A \cdot B )$ y determina su error relativo.

22. **Producto Compuesto:**
   Con $( X = 5.0 \pm 0.3 )$ y 4( Y = 2.2 \pm 0.05 )$, encuentra $( Z = X \cdot Y^2 )$ y su error relativo.

23. **Producto con Potencia:**
   Si $( P = 4.8 \pm 0.4 )$ y  $Q = 3.0 \pm 0.2 )$, evalúa $( R = P^2 \cdot Q )$ y encuentra su error relativo.

24. **Producto de Varias Variables:**
   Dadas $( M = 1.5 \pm 0.1 )$, $(N = 2.0 \pm 0.3)$, y $( O = 3.2 \pm 0.2)$ , calcula $( P = M \cdot N \cdot O )$ y su error relativo.

25. **Producto con Constante:**
   Multiplica $( 7.0 \pm 0.5 )$ por $( 2.4 \pm 0.1 )$ y determina el error relativo del resultado.

### Cocientes:

26. **Cociente Simple:**
   Con $(A = 10.0 \pm 0.5 )$ y $(B = 2.5 \pm 0.1)$, calcula el cociente $( C = A / B)$ y determina su error relativo.

27. **Cociente Compuesto:**
   Evalúa $(X = 8.0 \pm 0.3 )$ dividido por $( Y = 2.2 \pm 0.05 )$ al cuadrado y encuentra su error relativo.

28. **Cociente con Potencia:**
   Si $( P = 6.0 \pm 0.3 )$ y $(Q = 2.5 \pm 0.2)$, determina $( R = \frac{P^2}{Q} )$  y su error relativo.

29. **Cociente de Varias Variables:**
   Dadas $( M = 3.0 \pm 0.2 )$, $( N = 1.5 \pm 0.1 )$, y $( O = 2.0 \pm 0.3 )$, calcula $( P = \frac{M \cdot N}{O} )$ y su error relativo.

30. **Cociente con Constante:**
    Divide $( 5.0 \pm 0.4 )$ por $( 1.2 \pm 0.1 )$ y determina el error relativo del resultado.

31. Si una medida es $( 15.6 \pm 0.4 )$, calcula su error relativo.
32. Determina el error relativo porcentual de una longitud medida como $( 18.2 \pm 0.5 )$ cm.
33. Encuentra el error relativo de una masa medida como $( 25.3 \pm 0.7 )$ g.
34. Calcula el error relativo porcentual de una velocidad medida como $( 30.0 \pm 1.5 )$ m/s.
35. Si el tiempo es $( 10.0 \pm 0.2 )$, determina su error relativo porcentual.

### Ejercicios sobre Propagación de Errores en Funciones de Dos Variables:

36. Dada $( f(x, y) = x^2 + 2y )$, encuentra el error relativo de $( f)$ si $( x = 3.0 \pm 0.1)$ y $( y = 1.5 \pm 0.2)$.
37. Para $( g(a, b) = \sqrt{a^2 + b^2} )$, calcula el error relativo de $( g )$ con $( a = 4.0 \pm 0.3 )$ y $(b = 2.5 \pm 0.1)$.
38. Encuentra el error relativo porcentual de $( h(u, v) = \frac{u}{v} )$ con $( u = 8.0 \pm 0.4 )$ y $( v = 2.0 \pm 0.1 )$.
39. Si $( k( m, n) = m^n )$, determina el error relativo de $k)$ con: 
            $m = 5.0 \pm 0.2$  y   $n = 2.0 \pm 0.1$.

40. Para $( p(x, y) = x \cdot e^y )$, calcula el error relativo de $( p )$ con $( x = 2.5 \pm 0.1 )$ y $( y = 0.8 \pm 0.05 )$.


"""

# ╔═╡ 16020e1f-88c0-40fc-8ad8-673f151e860b
md"""
Una serie de tiempo armónica se puede expresar en función de los coeficientes ${(X_1,X_2)}$  o de la amplitud $A$ y la fase $\theta$ como:

$T(t)= X_1 \sin(ωt) + X_2 \cos(ωt)$ 

$T(t) = A \cos(ωt -θ)$


Calcula el error en la amplitud y la fase en función de las desv estándar y correlación de ${(X_1,X_2)}$ dónde:

```math
\begin{eqnarray}
A=\sqrt{(X_1^2 +X_2^2)} \\
θ=tan^{-1}(\frac{X_2}{X_1})
\end{eqnarray}
```
"""

# ╔═╡ c815fedf-a5f6-4e10-905b-fa7e201be9b2
md"""
Les dan una resma de 500 hojas y una regla graduada un milímetros.
Medir el espesor de una hoja con una precisión del orden ~ $1.0 e{-3}$ milímetros.  
"""

# ╔═╡ Cell order:
# ╠═10166b16-91dc-4ff1-94b5-37c759c4121b
# ╟─72a753d8-b6ea-11ee-3203-235468905873
# ╟─87135584-d489-4646-ac82-53e4b471f019
# ╟─16020e1f-88c0-40fc-8ad8-673f151e860b
# ╠═c815fedf-a5f6-4e10-905b-fa7e201be9b2
