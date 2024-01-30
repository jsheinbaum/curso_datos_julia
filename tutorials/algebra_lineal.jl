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

# ╔═╡ 7ec18854-0620-11ed-2286-5d4f708b7418
using Symbolics

# ╔═╡ 3983aab6-8ec1-4fd6-a23c-8ad7b94cfc28
 using LinearAlgebra,Latexify, Printf,WGLMakie#,Plots

# ╔═╡ 6d38a682-f292-4f07-879d-2e4c11c74da6
using PlutoUI,InteractiveUtils,Markdown, LinearSolve, Krylov

# ╔═╡ 1c205fbf-c856-4a87-b1ac-76d8c7f6d128
md"""
# Elementos de Álgebra Lineal 
"""

# ╔═╡ 5f72b834-d650-4d87-8a11-3ca5c922f7cf
md"""
## LES RECOMIENDO checar:

### Libros/Cursos en la red:

ALGEBRA LINEAL CON JULIA Stephen Boyd & Lieven Vandenberghe \
[https://web.stanford.edu/~boyd/vmls/](https://web.stanford.edu/~boyd/vmls/) 

3BLUE1BROWN EXCELENTE! \
[https://www.youtube.com/c/3blue1brown](https://www.youtube.com/c/3blue1brown)

GILBERT STRANG MIT \
[https://math.mit.edu/~gs/](https://math.mit.edu/~gs/)

### Lenguajes de Programación: 

#### Julia
[https://julialang.org/](https://julialang.org/) 

[https://github.com/fonsp/Pluto.jl](https://github.com/fonsp/Pluto.jl)

#### Anaconda Python
[https://docs.anaconda.com/anaconda/install/(https://docs.anaconda.com/anaconda/install/)
"""

# ╔═╡ d3893588-6aa1-409d-8b1a-ee2c96fda393
md"""
### 1)Sistemas de Ecuaciones lineales
### 2)Formulación matricial, Espacios vectoriales,columna,renglón,bases
### 3)Sistemas determinados, sobre,sub
### 4)Cuadrados mínimos
### 5)Problema eigenvectores,eigenvalores
### 5)Descomposición de valor singular
### 7)matrices,rotaciones,números complejos
"""

# ╔═╡ a73af075-0eea-4c7d-ac6d-32e3679f8617
WGLMakie.activate!()

# ╔═╡ c1b1a793-92f4-4e74-b217-5ea366d5fd89
@variables x y z

# ╔═╡ 48c9f6d8-344c-434f-bb09-d3e118bfe9ca
round4(xm)=round(xm,digits=4)

# ╔═╡ 3a91da73-225c-48c3-89fa-fbf50540e14f
md"""
### Sistemas de ecuaciones lineales
"""

# ╔═╡ 0dc0f395-de9b-40b0-af83-f28f6d7da919
begin
A=[1 3 5;4 5 6;7 8 11];
B=[1 2;2 4];
f1=B*[x;y]
f=A*[x; y; z]
#A*[1 2 3]'
d2sys=(f1 ~ [0.0;0.0])
#([[f[1] ~ 2.0,f[2]~0.0,f[3]~0.0]])
#Symbolics.solve_for([f[1] ~ 2.0,f[2]~0.0,f[3]~0.0],[x y z])
end

# ╔═╡ 26c0956b-fb82-44ef-8a9c-10bd366fa886
latexify(A*[x;y;z] ~ [2;0;0])

# ╔═╡ a493f046-2453-4df5-a48b-71bc57f7944c
(f ~ [3;0;0])

# ╔═╡ a098cdd8-02a3-4944-9268-8e6bb5df1bf1
begin
fsol=Symbolics.solve_for([f[1] ~ 2.0,f[2]~0.0,f[3]~0.0],[x y z]);
rfsol=[round(fsol[l];digits=4) for l=1:length(fsol)];
#str1="the solution of this system is [x,y,z]= "
latexify(("[x,y,z]" ~ round4.(rfsol)))
#
#e1=([1; 3; 5])
#e2=([4 5 6]')
#e3=([7 8 11]')
#tt1=latexify(x*e1) 
#tt2=latexify(y*e2)
#latexify(x*e1+y*e2+z*e3 ~ [2;0;0])
end

# ╔═╡ 9e019683-8bdc-4305-82b7-d512cb28726d
md"#### El sistema de ecuaciones lo podemos escribir de esta forma"

# ╔═╡ 90b3b854-3353-473d-a012-72a0be61886d
md"""
```math
x * \vec{col_1A} + y * \vec{col_2A}+ z * \vec{col_3A}


``` 
"""

# ╔═╡ 92a9f1da-08b0-45f9-a5df-4cca91176470
begin

#using Latexify
struct Ket{T}
    x::T
end	
@latexrecipe function f(x::Ket)
return Expr(:latexifymerge, "", x.x, "")
end
latexify(:(x*$(Ket(A[:,1])) + y*$(Ket(A[:,2]))+z*$(Ket(A[:,3]))))
end

# ╔═╡ 7db18b00-81a6-4098-a7f0-b7598ca66070
md"### El sistema de ecuaciones de 2x2"

# ╔═╡ 3769d115-007e-4430-b67a-c213b69bc29f
begin
#Symbolics.solve_for([f1[1] ~ 1.0,f1[2] ~ 0.0],[x y])
Bb=copy(B)
Bb=Bb+([0.5 0;0 0])
#f3=B*[x;y];
f3=Bb*[x;y]
#f3=(B+ones(2,2))*[x;y];
(f3~[2; 2])
end

# ╔═╡ 73e97ad5-fb08-4946-8247-0598ac2c8dee
# ╠═╡ disabled = true
#=╠═╡
Symbolics.solve_for([f[1] ~ 2.0,f[2]~0.0,f[3]~0.0],[x y z])
  ╠═╡ =#

# ╔═╡ c41e3d38-a29c-4719-bb15-e2a58c1a7a86
md"""
#### La solución del sistem es: 
"""

# ╔═╡ f72f0686-f3f8-4226-8c90-33da55ce5e29
begin
fsol3=Symbolics.solve_for([f3[1] ~ 2.0,f3[2] ~ 2.0],[x y])
latexify(("[x,y]" ~ [round4(fsol3[1]);round4(fsol3[2])]))

#t1*string(fsol3)"
#println("The solution is [x,y,z]= " * string(fsol3))
end

# ╔═╡ 3da71a6a-5a20-4a27-acab-2a0f2c26b5e9
begin
@variables AA[1:3,1:3],b[1:3]
AA=ones(3,3)
	latexify("ones(3,3)" ~ AA)
end

# ╔═╡ 8169ad19-e3ad-4795-86de-eca3e3a4b5cd
md" #### Otros sistemas y su soluciones"

# ╔═╡ 8963b9fa-506b-43fa-9cf5-5a545a3ab3fd
@bind mfac1 Slider(-10.0:10.0,show_value=true,default=1)

# ╔═╡ 83d17a70-f8b0-430d-ab2d-014caeeb8fdf
begin
M1=copy(float(A))
M1[2,:]=2.0*M1[1,:]+[1.0,0.0,0.0]*mfac1
M1[3,:]=M1[1,:]-[0.0,0.0,1.0]*(mfac1/2.0)-3.0*[0.0,1.0,0.0]
latexify("M1" ~ M1)
end

# ╔═╡ 2a7364a7-7f92-46db-b176-77f7fb8ce7f9
latexify("rango" ~ rank(M1))

# ╔═╡ bed2c602-f4ad-428a-a8f8-e75105fbed1c
begin
M=A*[x;y;z]
#print(M)
M[1]=M[1] 
M[2]=2*M[1] +mfac1*x
M[3]=M[1] -mfac1/2*z-3y
#M[1]=M[1] +x+y+z
#M[2]=M[2] -x-3y-2z
#M[3]=M[3] +x+y	
latexify(M ~ [1.0;2.0; 1.0])
#Symbolics.solve_for([M[1] ~ 1.0,M[2] ~ 1.0,M[3] ~ 1.0],[x y z])
end


# ╔═╡ d3cb8670-5899-41b2-8e63-43550c74ac40
begin
#fsol4=Symbolics.solve_for([M[1] ~ 1.0,M[2] ~ 2.0,M[3] ~ 1.0],[x y z]);
	if(rank(M1) == 3)
	fsol4=Symbolics.solve_for([M[1] ~ 1.0,M[2] ~ 2.0,M[3] ~ 1.0],[x y z]);
	latexify(("[x,y,z]" ~ round4.(fsol4)))
	else
	latexify("Sistema-Singular-rango-incompleto")
	end
#latexify(("[x,y,z]" ~ ([round4(fsol4[1]);round4(fsol4[2]);round4(fsol4[3])])))
end

# ╔═╡ 11e6ecdc-5cbc-4bfe-a9cb-6cfc649f4a6d
md" #### Operaciones elementales, pivotes, descomposición LU"

# ╔═╡ 29a1ea27-9676-44dc-bb38-a29bcaf33985
latexify("A" ~ A)

# ╔═╡ 4c94c8db-2085-4714-99a7-bee128a9c1bd
begin
#A+[0 0 0;2 -1 0;1 0 -1]*A
#A+[0 0 0;0 0 0;1 0 -1]*A
P1=[1 0 0;4 -1 0;0 0 1]
P2=[1 0 0; 0 1 0;7 0 -1]
P3=[1 0 0; 0 1 0;0 13//7 -1]
PA=P1*A
PPA=P2*PA
PPPA=P3*PPA
latexify("P1" ~ [1 0 0;4 -1 0;0 0 1]),
latexify("P1A" ~ PA),
latexify("P2" ~ [1 0 0; 0 1 0;7 0 -1]),
latexify("P2P1A" ~ PPA),
latexify("P3" ~ [1 0 0; 0 1 0;0 13//7 -1]),
latexify("P1P2P3A" ~ PPPA)
end
#latexify(rank([1 3 5;2 6 10;1 2 0]))

# ╔═╡ 868c117d-fb9e-4b07-a5cf-e96a73373060
@bind indp Slider(1:3,show_value=true,default=1)

# ╔═╡ d81bacdf-ca72-480b-aab5-4be71020e3bf
begin
PS=("P"*string(indp));
PSE=eval(Meta.parse(PS))
PSEI=inv(PSE)
latexify(PS ~ PSE),
if indp < 3
latexify(PS*"^-1" ~ Int.(PSEI))
else
latexify(PS*"^-1" ~ PSEI)
end
#latexify(PS),
#latexify(inv(PS))
end

# ╔═╡ 188a7dda-1407-4365-bd51-b79728f71b45
latexify("A" ~ A),
latexify("SP" ~ [1 0 0;4 -1 -1;0 0 1]),
latexify("SPA" ~ [1 0 0;4 -1 -1;0 0 1]*A),
latexify("4row1A-row2A-row3A" ~ [4*A[1,:]-A[:2,:]-A[3,:]]')



# ╔═╡ f9e55d4b-1c1d-417f-b9ee-3f0d6a100a41
inv([1 0 0;4 -1 -1;0 0 1])

# ╔═╡ fb81b55f-0404-4d11-bbee-1bb08cf7a5a8
md" ### Descompoisición LU, A LU?"

# ╔═╡ 838b9237-1333-4afc-8f1f-8f06a9b882d7
begin
FLU=lu(A)
latexify(("L " ~round.(FLU.L,digits=4))),
latexify(("U " ~round.(FLU.U,digits=4))),
latexify(("LU " ~round.(FLU.L*FLU.U,digits=4)))
end

# ╔═╡ 15bcb9cf-b468-4f77-817c-3267d7226c8d
md" ##### Hay una permutación de renglones de por medio en el proceso!" 

# ╔═╡ 8e573ccb-86d9-4f3c-adbc-81018ecb7cf7
latexify("PA = LU")

# ╔═╡ f2027eec-5107-4ea2-9794-25bc159652e2
latexify("A" ~ A),
latexify(("P " ~ round4.(FLU.P))),
latexify("PA" ~ Int.(FLU.P*A))
#Latexify(("PA" ~ round4.(FLU.P)*A)))

# ╔═╡ b38d2850-c767-4b80-ac1a-e3a134bf44d8
latexify(("(P^-1)LU" ~ inv(FLU.P)*FLU.L*FLU.U))
#latexify(A[sort(FLU.p),:])

# ╔═╡ de9ce83c-0da6-4dd2-ace5-fdae8b4df025
md" #### permuta renglones 1-> 2, 2 -> 3, 3 -> 1"

# ╔═╡ ea91d037-6ad2-44f3-b04f-fa26353b4240
#latexify(inv(FLU.P)*FLU.p)
latexify(("P*[0;0;1]"~ FLU.P*[0;0;1]))

# ╔═╡ 22c9db03-cacd-4658-9994-68b4ff29a76d
latexify(("P*[1;0;0]"~ FLU.P*[1;0;0]))

# ╔═╡ 0d859ce4-fb41-41d4-a804-39aca7aaebb8
latexify(("P"~ FLU.P))

# ╔═╡ 41f00d4b-2235-4dfd-9b97-e5207255d3b9
latexify(("U=(L^-1)PA" ~ round.(inv(FLU.L)*FLU.P*A,digits=4)))

# ╔═╡ 76d92e16-4695-440c-b71a-610bf6f9c3ba
latexify("PA=LU, PAx=Pb, LUx=Pb,Ux=c, Lc=Pb")

# ╔═╡ 6f81ce75-1bca-4073-a76f-0aab6d20c8a7
md" ##### Resuelve para c (sistema Lc=Pb ) y luego resuelve Ux=c para obtener x"

# ╔═╡ f47e6d00-7eff-45b3-baf4-6529d866cf36
latexify(("c=(L^-1)P*[2;0;0]" ~ round4.(inv(FLU.L)*FLU.P*[2;0;0])))

# ╔═╡ 4573d0ce-aee5-4b4c-b1e2-29df6a1dfc00
begin
#FLU.U
#FLU.U*[x;y;z]#=
invlb=inv(FLU.L)*FLU.P*[2;0;0];
#FLU.p
end

# ╔═╡ a2726fff-8eff-4570-8a56-0a01b40c6a67
begin
Ux=FLU.U*[x;y;z]
fsol5=Symbolics.solve_for([Ux[1] ~ invlb[1],Ux[2] ~ invlb[2],Ux[3] ~ invlb[3]],[x y z])
latexify("U*[x,y,z] = c"), 
latexify(round4.(FLU.U)*[x;y;z] ~ round.(invlb,digits=3)),
latexify([x, y, z] ~ (round4.(fsol5)))
#latexify([round.(FLU.U,digits=3)(:($(Ket([x; y; z])))) ~ round.(invlb,digits=3) ,[x, y, z] ~ (round.(fsol5,digits=4))])
#latexify(round.(FLU.U,digits=4).*[x; y; z])
end

# ╔═╡ f9b646b1-d201-48ab-8dd8-bb4083ba8f67
md"# La solución 'back slash' [x;y;z]=A\b"

# ╔═╡ 3a44a0dd-4bb8-491a-bd74-c6f7e2c6c6f2
md"""

##### begin 

###### Definimos la función

function lf0(A,b) \
return A\b \
end

##### [x y, z]=lf0(A,[b1;b2;b3])

##### end
"""

# ╔═╡ c32cf782-607c-4a41-892a-1aae585e7010


# ╔═╡ 6371557e-1af0-4e2c-a7b3-37e989fff1ca
#Defined above remember!
#using Latexify
#struct Ket{T}
#    x::T
#end	
#@latexrecipe function f(x::Ket)
#return Expr(:latexifymerge, "", x.x, "")
#end

begin
lf0(A,b)=A\b
xsol0=lf0(A,[2.0;0.0;0.0])
#strAb=(A\b)
b3=[2.0;0.0;0.0]
#latexify(:(x*$(Ket(A[:,1])) + y*$(Ket(A[:,2]))+z*$(Ket(A[:,3])))),
latexify(:($(Ket(A[:,:]))*[x;y;z] = [b1;b2;b3])),#*[x;y;z]), #+ y*$(Ket(A[:,2]))+z*$(Ket(A[:,3])))),
latexify("b" ~ b3),
#latexify(strAb)
latexify(("[x,y,z]" ~ round.(xsol0,digits=4)))
end

# ╔═╡ 4f4033a6-b2a1-42fb-afdd-2f5a54b4c82f
md""" #### Cambia el lado derecho de la ecuación y resuelve x=A\b
"""

# ╔═╡ 34bf99c5-0a9f-4d60-8c26-2ba972f7e193
latexify("b" ~ [-2,1.0,2.0] )

# ╔═╡ b9a7f0bc-2fb4-4620-8364-c4419589e910
begin
lf(A,b)=A\b
xsol1=lf(A,[-2.0;1.0;2.0])
latexify(("[x,y,z]" ~ round.(xsol1,digits=3)))

end

# ╔═╡ ccc85829-c70b-4f32-bea4-3d9c568eadcc
md"### Cuadrados mínimos, caso típico de problema sobre-determinado"

# ╔═╡ 3a459c84-1a1c-4096-9c72-10ea11a7f2fc
@bind inoise Slider(0.0:0.2:5,show_value=true,default=1)

# ╔═╡ 9639d400-07a3-40c9-adbb-4269b9c017d3
begin
flsq = Figure(size = (500, 500))
axl=Axis(flsq[1, 1], backgroundcolor = "beige")
tt=LinRange(0,10,100);
#tt=0:.1:10.0;
noise=inoise*randn(length(tt));
Att=[tt[:] ones(length(tt),1)];
xv=[1.0,3.5];
xlsq=lf0(Att,Att*xv+noise);
yobs=Att*xv.+noise
yv=Att*xv
ylsq=Att*xlsq
pointsobs = Point2f.(tt, yobs)
pointsv=Point2f.(tt, yv)
lines!(axl,pointsv,linewidth=4.0,color=:black)
scatter!(axl,pointsobs)
lines!(axl,tt,ylsq,linewidth=4.0,color=:red)
#scatter(tt,Att*xv+noise)#plot(tt,Att*xlsq,color=:red)
#lines(tt,yv)
#xlsq=lf0(Att,Att*xv+noise);
#size(noise)
#size(Att*xv)
#limits!(axl,0.0,10.0,0.0,16.0)
flsq
end


# ╔═╡ b8bcf86b-8526-4c37-acd9-c266052acb84
begin
latexify("xlsq" ~ round4.(xlsq)),
latexify("xv" ~ xv)
end

# ╔═╡ 389ae6d5-ba80-4944-be4f-b61845a0fe2a
md"""
#### $A^t(A\vec{x} -b)= 0$
#### $A^tA\vec{x} = A^tb)$
#### $\vec{x} = (A^tA)^{-1}A^tb)$
##### Las diferencias (residuos) deben ser ortogonales al espacio columna de A
"""

# ╔═╡ cbc667b7-c224-4e4a-b17e-ba97c9ead32e
begin
slsq=L"$A^t(A\vec{x} -b)"
latexify(slsq ~ Att'*(Att*xlsq-yobs))
end

# ╔═╡ 48712159-24dd-4dba-90d0-84be4edea6ee
md"""
## Eigenvalores/Eigenvectores (Matrices cuadradas)

#### $A*\vec{v}_j=\lambda_{j} \vec{v}_{j}$

#### $$\Lambda = \begin{bmatrix} \lambda_{1} &  &  \\ & \ddots &  \\ & & \lambda_{n} \end{bmatrix}$$


#### $$AV = \begin{bmatrix} A\vec{v}_1 & A\vec{v}_2 &  \ldots & A\vec{v}_n\end{bmatrix}=  \begin{bmatrix} \lambda_{1} \vec{v}_1 & \lambda_{2} \vec{v}_2 &  \ldots & \lambda_{n} \vec{v}_n\end{bmatrix}$$

#### $A*V=V*\Lambda$
#### $V^{-1}*A*V=\Lambda$
#### $V*\Lambda*V^{-1}=A$

#### Descomposición en Valores Singulares (SVD)
#### $A^{m\times n} = U^{m \times m} S^{m \times n} V^{T{(n \times n })}$ 
#### $U^{T}*U= I^{m \times m}, V^{T}*V= I^{n \times n}$
#### U, V tienen columnas ortonormales
"""

# ╔═╡ 95e0cf44-a8a4-4357-90e9-4a0422d89880
begin
@variables a[1:2,1:2]
function g(a,c)
	return a=c
end
end

# ╔═╡ b477835b-ea36-4d84-a237-b37404b44d94
begin
	aa=g(a,[0 1;1 1]);
L1,V1=eigen(aa)
	latexify(["L = (V^-1)aa(V)" ~ round.(inv(V1)*aa*(V1),digits=4)])
#	V1*diagm(L1)*inv(V1)
end
	

# ╔═╡ ee6ebb3f-754f-4450-9842-f888ed09b682
latexify("aa" ~ aa)

# ╔═╡ 3c759c59-9495-40e3-8291-bc6c468ebb4c
latexify(round4.(L1))

# ╔═╡ 97505e60-c591-43d7-99c3-2801e710010c
latexify("eigvecs" ~ round4.(V1))

# ╔═╡ 15d1e543-5bce-41c9-b7fd-c7329b3d1e07
begin
ff = Figure(size = (500, 500))
ax=Axis(ff[1, 1], backgroundcolor = "beige")
xs = LinRange(-10, 10, 21)
ys = LinRange(-10, 10, 21)
us = [1.0 for x in xs, y in ys]
vs = [0.0 for x in xs, y in ys]
strength = vec(sqrt.(us .^ 2 .+ vs .^ 2))
#arrows!(xs, ys, us, vs, arrowsize = 10, lengthscale = 1.0, arrowcolor = strength, linecolor = :yellow)
#arrows!(xs, ys, vs, us, arrowsize = 10, lengthscale = 1.0, arrowcolor = strength, linecolor = :yellow)
strength2 = vec(sqrt.(V1[:,1] .^ 2 .+ V1[:,2].^ 2))
arrows!(ax,[0;0], [0;0], V1[1,:], V1[2,:],arrowsize =15.0 , lengthscale =1.,arrowcolor = strength2, linecolor =:red,linewidth=2.5)
#arrows!(ax,[0;0], [0;0], [0;1], [1;0],arrowsize =15.0 , lengthscale =1.,arrowcolor =[1;1], linecolor =:black)
arrows!(ax,[0;V1[1,1]],[0;0],[V1[1,1],0],[0,V1[2,1]],arrowsize =15.0 , lengthscale =1.,arrowcolor =[1;1], linecolor =:black)
arrows!(ax,[0;V1[1,2]],[0;0],[V1[1,2],0],[0,V1[2,2]],arrowsize =15.0 , lengthscale =1.,arrowcolor =[1;1], linecolor =:black)
θ=-atan(V1[2,2]/V1[1,2]);
θ=π/3
r90=[cos(θ) -sin(θ);sin(θ) cos(θ)];
V2=r90*V1;
arrows!(ax,[0;0], [0;0], V2[1,:], V2[2,:],arrowsize =15.0 , lengthscale =1.,arrowcolor = strength2, linecolor =:green,linewidth=2.5)
#arrows!(ax,[0;V1[1,1]],[0;0],[V1[1,1],0],[0,V1[2,1]],arrowsize =15.0 , lengthscale =1.,arrowcolor =[1;1], linecolor =:black)
#arrows!(ax,[0;0], [0;0], [0;1], [1;0],arrowsize =15.0 , lengthscale =1.,arrowcolor =[1;1], linecolor =:black)
limits!(ax, -1, 1, -1, 1)
ff
end


# ╔═╡ 99dc3b67-0bdb-426c-a925-4cd49d39b4c8
begin
LA,VA=eigen(A);
latexify("A" ~ A),
latexify("eivalsA" ~ round4.(LA)),
latexify("eigvecsA" ~ round4.(VA))
end

# ╔═╡ 8fbbe334-26e8-4d8a-9148-6559d4888f01
cc=[2 3;4 6]

# ╔═╡ acc3a162-6f44-436d-990b-afdf6addc877
kk=(cc+ ones(2,2))*[x y]'

# ╔═╡ 44862011-d0f8-4c63-91c0-2fc12d50161e
Symbolics.solve_for([kk[1] ~ 1,kk[2] ~ 1.5],[x y])

# ╔═╡ 79ce5c1e-d8ef-40ea-99e5-f605c9c2a24a
#dot(cc[:,1],[-2 1])
cc[:,1]'*[-2;1]

# ╔═╡ 5d18d532-80b1-42a3-bf37-4d7883e8de5b
Uc,S,Vc=svd(cc);

# ╔═╡ d990a16d-ee3e-436e-bf30-5c1ebcd0347e
#-.5*cc[:,1]
cc

# ╔═╡ 5110b4e7-a21d-4c3d-8bdf-933d8dd980e2
begin
function xx(yy)
return [-1.5yy,yy]
end
#xx[1.0]
end

# ╔═╡ b1aba160-1709-4758-a7cc-3bd543d474c4
begin
ff1 = Figure(size = (800, 800))
ax1=Axis(ff1[1, 1], backgroundcolor = "black")
#for l=-10:10
l=LinRange(-10, 10, 21)
xn=xx.(l)
#scatter!(ax1,xn[1],xn[2])#,xn[1]*cc[:,1]+xn[2]*cc[:,2])
#end

end

# ╔═╡ 01bfa680-833a-4669-b312-79b474d14708
begin
x1s=[xn[l1][1] for l1=1:length(l)];
y1s=[xn[l1][2] for l1=1:length(l)];
end

# ╔═╡ cc1194a1-8c1a-4fe8-aed7-14265f745cc1
md"Combinaciones lineales de las columnas de la matriz (lin indep) me dan cualquier vector"

# ╔═╡ 66837355-8d31-4b4d-81d7-411b2e8ca021
latexify("cc"~ cc)

# ╔═╡ e36c75ec-8a91-48be-afb0-5d6b524e533f
md"Vector $[50,10]$ lo puedo escribir como"

# ╔═╡ 03c0b05d-eff0-41cb-9144-4b221d286d24
latexify(10*cc[:,1]+10*cc[:,2] -100*cc[:,1] + 200/3*cc[:,2])

# ╔═╡ c50c63fa-018b-456c-9203-6a003efc9b15
factor=[-5:5];

# ╔═╡ 5beb7506-c03a-452b-930e-6f6fdb894532
begin
ff0 = Figure(size= (500, 500))
ax0=Axis(ff0[1, 1], backgroundcolor = "beige")
#scatter(xs,ys)
#scatter(2l,4l)
scatter!(vec([xs ys]),vec([-ys xs]))
limits!(-20,20,-20,20)
ff0
end

# ╔═╡ f9b16cbf-3475-42fd-95be-75d730a60a01


# ╔═╡ 6ec92519-f7cd-4796-862f-37d6843ec1e4
begin
BB=[1 0 -1;-1 1 -1;1 -2 3];
	latexify(BB)
end

# ╔═╡ 21a32643-939f-4faf-b7ad-ed832722d910
u,s,v=svd(BB);

# ╔═╡ b27264bb-d642-4cfb-9c72-183583635717
md""" ## descomposición QR, A=QR 
### Q, Orthogonal, R upper triangular
"""

# ╔═╡ a9209f0f-ae25-45fa-b48d-8a69f2a00634
begin
q,r=qr(BB);
latexify("R " ~ round4.(r)),
latexify("Q " ~ round4.(q))
end

# ╔═╡ ce27527d-be20-4851-9d1e-c994f6a4649c
begin
latexify("QR" ~ round4.(q*r)),
latexify("BB" ~ round4.(BB))
end

# ╔═╡ efb64145-cfcf-4250-a1fc-1019e886a251
begin
E1=[1 0 0;1 1 0;0 0 1]
E2=[1 0 0;0 1 0;-1 0 1]
E3=[1 0 0;1 0 0;0 2 1]
latexify("E1" ~ E1),
latexify("E2" ~ E2),
latexify("E3" ~ E3)
end

# ╔═╡ 5c0d1057-238a-42a6-977e-1585d17e52bf
begin
latexify("BB" ~ round4.(BB)),
latexify("E1*BB" ~ E1*BB),
latexify("E2*E1*BB" ~ E2*E1*BB),
latexify("E3*E2*E1*BB" ~ round4.(E3*E2*E1*BB))
end

# ╔═╡ 3855980b-50d5-4775-8706-9a12be3478e3
@bind d1 PlutoUI.Slider([[i  2i -i] for i in -2:2],show_value=true,default=1)

# ╔═╡ 8fa5884c-1227-40d4-9bda-c430ffc58b1b
@bind  d2 PlutoUI.Slider([[3j/2 4j j] for j in -2:2],show_value=true,default=1)#, for k in -2:2])

# ╔═╡ 63facbc6-5527-4231-a4b8-aee8d2202e7e
@bind  d3 PlutoUI.Slider([[3k -k/5 4k] for k in -2:2],show_value=true,default=1)

# ╔═╡ 2cb0eb87-e1a6-4fb0-92e2-86417ea3789e
begin
Md=[d1;d2;d3]
	latexify(Md)
end

# ╔═╡ 112ff420-49f4-4b6f-8735-8b4fb0304e00
begin
lf(Md,[2;1;3])
round.(lf(Md,[2;1;3]),digits=4)
end

# ╔═╡ 71f14de3-5131-46b4-901f-ab902bc1ade8
begin
EM=eigen(Md)
	EM.values
latexify("L" ~ diagm(round4.(EM.values)))
end

# ╔═╡ 3151f84b-e134-4e40-a1f3-f7753604874b
#EM.vectors
latexify("V" ~ (round4.(EM.vectors)))

# ╔═╡ 1b98a52e-d247-4d12-bcee-5d010731193c
round4.((rand(3,1)+im*rand(3,1))*10)

# ╔═╡ d00c1534-f5c6-4937-afbd-dd0ed51db090
begin
brhs=round.(rand(3,1)*20,digits=3)
Problema=LinearProblem(Md,brhs)
#sol = solve(Problema,IterativeSolversJL_GMRES())
#sol=solve(Problema)
#round.(sol.u,digits=3)
end

# ╔═╡ 0d1723d4-be40-4373-afe6-9282df45f0a8
begin
t=LinRange(0.0,100,200);
At=[ones(200,1) t];
bt=rand(200,1)*10.0;
xt=inv(transpose(At)*At)*transpose(At)*bt
xt2=At\bt
xt3=pinv(At)*bt
typeof(vec(bt))
(xt4,stats) = cgls(At,vec(bt))
latexify(["xt" ~ round4.(xt), "xt2" ~ round4.(xt3),"xt3" ~ round4.(xt3), "xt4" ~ round4.(xt4)])
end

# ╔═╡ 45ecb711-2a01-47a6-afb6-8f3e4cff7fd2
latexify(round4.(EM.vectors)),
latexify(round4.(EM.vectors')),
latexify(round4.(conj.(EM.vectors)))

# ╔═╡ 5df9929f-7581-44ab-a94d-7c5536e0e4c8
EMV=eigen((A+im*A)+(A+im*A)')

# ╔═╡ d4b00d94-1db9-4c52-a8fc-cbc74d1efd2f
begin
latexify(round4.((A+im*A)+(A+im*A)')),
latexify(round4.(EMV.vectors)),
latexify(round4.(EMV.values))
end

# ╔═╡ 26072ceb-052c-4836-8877-1e801cd27280
begin
EMVC=eigen(conj.((A+im*A)+(A+im*A)'))
latexify(round4.(conj.((A+im*A)+(A+im*A)'))),
latexify(round4.(EMVC.vectors)),
latexify(round4.(EMVC.values))
end

# ╔═╡ a767a0dd-a6c4-42b5-a8e8-41b45bbababb
begin
EMVD=eigen(conj.((A+im*A)-(A+im*A)'))
latexify(round4.(conj.((A+im*A)-(A+im*A)'))),
latexify(round4.(EMVD.vectors)),
latexify(round4.(EMVD.values))
end

# ╔═╡ 01d7fbce-efe1-4324-b243-409aee569632
begin
p1=[1+im; 2+3*im]
p2=[5-im;-3 + im]
latexify(p1'* p2),
latexify(conj(p2'*p1)),
latexify(p2'*p1)
end


# ╔═╡ 2f90671f-4527-4d1b-8309-79f13f670ff4
pascal(N) = [binomial(n, k) for n = 0:N, k=0:N]

# ╔═╡ cc377faa-486b-4e36-af69-1241eb83a6a1
Pas=pascal(5)

# ╔═╡ bcbfa135-d6ac-4105-a06f-f7ee5658e6fd
Pas-Pas'

# ╔═╡ ac50a0da-f924-4499-a7c7-b5134f33fbc2
begin
PasE=eigen(Pas-Pas')
@printf "%s " "eigenvalues pasc - pasc^T "
latexify(["eigenvalues-pasc-pasc'" ~ round4.(PasE.values)])
end

# ╔═╡ f3b16944-8c93-4c5f-81af-a92307ee9681
begin
	function rotating_matrices3d(α,β,θ)
		matz=[cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0];
		matx=[1.0 0.0 0.0;0.0 cos(α) -sin(α); 0.0 sin(α) cos(α)];
		maty=[cos(β) 0.0 sin(β); 0.0 1.0 0.0; -sin(β) 0.0 cos(β)];
		return matx,maty,matz
	end
end
	     

# ╔═╡ 612a3ea2-81cf-40fe-bd6f-01b2b860e9be
mat3d=rotating_matrices3d(0.0,π/2,π/2);

# ╔═╡ 1d93390f-306c-482a-b33e-0e2695859229
begin
latexify("I " ~ round4.(I(3))),
latexify("rotz" ~ round4.(mat3d[3]*I(3))),
latexify("rotx" ~ round4.(mat3d[1]*I(3))),
latexify("roty" ~ round4.(mat3d[2]*I(3)))
end

# ╔═╡ ba5e9605-71db-4818-bc16-0a2c8a45c0e9
@bind rotind PlutoUI.Slider([[rand(1:3),rand(1:3),rand(1:3)] for i =1:100],show_value=true,default=1)

# ╔═╡ 1ffcf623-542b-49d2-b69d-ff9f5caa3cea
begin
ff2 = Figure(size=(500,500))
ax10=Axis3(ff2[1, 1], backgroundcolor = "beige")
vec0= [Point3f(x, y, z) for x in [0.0] for y in [0.0] for z in [0.0]]
vec10= map(x -> Vec3f(1.0, 0.5, 0.5), vec0)
v10=map(x -> Vec3f(mat3d[rotind[1]]*mat3d[rotind[2]]*mat3d[rotind[3]]*vec10[1]),vec0)
#vec10=Vec3f([10,10,10])
#strength10 = norm(v10)
#arrows!(ax10,vec0,vec10, fxaa=true, arrowsize =0.05 , lengthscale =1.0,linecolor = :black, arrowcolor =:blue,linewidth = 0.025,align = :origin)

#arrows!(ax10,vec0,v10, fxaa=true, arrowsize =0.05 , lengthscale =1.0 ,linecolor = :green, arrowcolor =:red,linewidth = 0.025,align = :origin)

arrows!(ax10,vec0,vec10,align=:origin,linewidth = 0.05,arrowsize = Vec3f(0.2, 0.2, 0.2))
arrows!(ax10,vec0,v10,align=:origin,linecolor=:red,linewidth = 0.05,arrowsize = Vec3f(0.2, 0.2, 0.2))
xlims!(-2, 2)
ylims!(-2, 2)
zlims!(-2,2)
ff2
end

# ╔═╡ 2274bfd4-67b8-4f9a-91fd-b93b4484b9d5
begin
latexify("orig" ~ vec10),
latexify("rotated" ~ v10);
end

# ╔═╡ 99a97da1-96df-4111-b7bd-3d3a820ee8c9
begin
ff3=Figure(size=(500,500))
ax11=Axis3(ff3[1, 1], backgroundcolor = "beige")
ps = [Point3f(x1, y1, z1) for x1 in -10:2:10 for y1 in -10:2:10 for z1 in -10:2:10];
L=10.0
ns = map(p -> Vec3f(sin(p[1]*π/L)*cos(p[2]*π/L), -sin(p[2]*π/L)*cos(p[1]*π/L), p[3]*π/L), ps);
arrows!(
           ps, ns, fxaa=true, # turn on anti-aliasing
           linecolor = :red, arrowcolor = :black,
           linewidth = 0.1, arrowsize = Vec3f(0.3, 0.3, 0.4),
           align = :center)#,axis=(type=Axis3,))
ff3
end


# ╔═╡ 074bda6d-9e8e-4cfe-aba1-b99552581366
md"""
## MATRICES,NÚMEROS COMPLEJOS Y ROTACIONES
"""

# ╔═╡ d45d5016-e700-4659-a929-acd7a62a613e
begin
@variables β[1:2], γ[1:2]
function δ(β,γ)
#	(β[1] +im*β[2])*(γ[1]+ im*γ[2])
	complex(β[1],β[2])*complex(γ[1],γ[2])
end
δ1=δ(β,γ)
#latexify("δ" ~ (β[1] +im*β[2])*(γ[1]+ im*γ[2])),
latexify(" δ = δ(β,γ) "), # ~ real(δ1) + (im)*imag(δ1))#+ im*δ1[2]])
#latexify(" = "),
latexify(δ(β,γ))

end

# ╔═╡ 667e073d-57ba-41c6-90e9-3f7f1349dee0
latexify(δ([0;1.0],[0.0,1.0]))

# ╔═╡ 61623cb6-da6b-480b-b8bb-e1315d019765
begin
Β(β)=[β[1] -β[2]; β[2] β[1] ]
Γ(γ)=[γ[1] -γ[2]; γ[2] γ[1] ]
latexify("Γ" ~ Γ(γ)),
latexify("β" ~ Β(β)),
#latexify("βΓ" ~ Β([0.0;1.0])*Γ([1.0,0.0]))
latexify("βΓ" ~ Β(β)*Γ(γ))

end

# ╔═╡ c107aecc-62ed-4341-a432-9e879451eda2
begin
	@variables ϕ
	function expΩ(ϕ)
		cos(ϕ)*I(2)+ sin(ϕ)*[0.0 -1.0;1.0 0.0]
	end
(Β([0.0, 1.0]))*[1.0;0.0]
end

# ╔═╡ 3203640c-2810-41ad-95e6-cd6ce937eb94
latexify(round4.(expΩ(pi)))
#Β([0.0, 1.0])*ϕ

# ╔═╡ da35c661-ede7-4dc0-acaa-525f8772431d
begin
ϵ(ϕ)=(Β([0.0, 1.0]))*ϕ
function mexp(ϵ)
	cos(ϕ)*I(2)+ sin(ϕ)*[0.0 -1.0;1.0 0.0]
end
end

# ╔═╡ dd9fc92d-e4d1-4d78-9a48-86ba39e07b24
exp.(ϵ(ϕ))

# ╔═╡ bd6d32be-2f3c-49a3-92e3-9b08c44ddb1a
mexp(ϵ(ϕ))

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
Krylov = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LinearSolve = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
Markdown = "d6f4376e-aef5-505a-96c1-9c027394607a"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Printf = "de0858da-6303-5e67-8744-51eddeeeb8d7"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"
WGLMakie = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"

[compat]
Krylov = "~0.9.5"
Latexify = "~0.16.1"
LinearSolve = "~2.22.1"
PlutoUI = "~0.7.55"
Symbolics = "~5.14.1"
WGLMakie = "~0.9.4"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.4"
manifest_format = "2.0"
project_hash = "5022767ba6fb7daa5c4790fabbb3720ff978067b"

[[deps.ADTypes]]
git-tree-sha1 = "41c37aa88889c171f1300ceac1313c06e891d245"
uuid = "47edcb42-4c32-4615-8424-f2b9edc5f35b"
version = "0.2.6"

[[deps.AbstractAlgebra]]
deps = ["GroupsCore", "InteractiveUtils", "LinearAlgebra", "MacroTools", "Preferences", "Random", "RandomExtensions", "SparseArrays", "Test"]
git-tree-sha1 = "d7832de8cf7af26abac741f10372080ac6cb73df"
uuid = "c3fe647b-3220-5bb0-a1ea-a7954cac585d"
version = "0.34.7"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractLattices]]
git-tree-sha1 = "222ee9e50b98f51b5d78feb93dd928880df35f06"
uuid = "398f06c4-4d28-53ec-89ca-5b2656b7603d"
version = "0.3.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "c278dfab760520b8bb7e9511b968bf4ba38b7acc"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.2.3"

[[deps.AbstractTrees]]
git-tree-sha1 = "faa260e4cb5aba097a73fab382dd4b5819d8ec8c"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.4"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "cde29ddf7e5726c9fb511f340244ea3481267608"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "3.7.2"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.Animations]]
deps = ["Colors"]
git-tree-sha1 = "e81c509d2c8e49592413bfb0bb3b08150056c79d"
uuid = "27a7e980-b3e6-11e9-2bcd-0b925532e340"
version = "0.4.1"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra", "Requires", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "bbec08a37f8722786d87bedf84eae19c020c4efa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.7.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Automa]]
deps = ["PrecompileTools", "TranscodingStreams"]
git-tree-sha1 = "588e0d680ad1d7201d4c6a804dcb1cd9cba79fbb"
uuid = "67c07d97-cdcb-5c2c-af73-a7f9c32a568b"
version = "1.0.3"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.Bijections]]
git-tree-sha1 = "c9b163bd832e023571e86d0b90d9de92a9879088"
uuid = "e2ed5e7c-b2de-5872-ae92-c73ca462fb04"
version = "0.1.6"

[[deps.BitFlags]]
git-tree-sha1 = "2dc09997850d68179b69dafb58ae806167a32b1b"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.8"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "0c5f81f47bbbcf4aea7b2959135713459170798b"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.5"

[[deps.Bonito]]
deps = ["Base64", "CodecZlib", "Colors", "Dates", "Deno_jll", "HTTP", "Hyperscript", "LinearAlgebra", "Markdown", "MsgPack", "Observables", "RelocatableFolders", "SHA", "Sockets", "Tables", "ThreadPools", "URIs", "UUIDs", "WidgetsBase"]
git-tree-sha1 = "eb7ffbde27b8f321aaac74891dcddcd5676eeae9"
uuid = "824d6782-a2ef-11e9-3a09-e5662e0c26f8"
version = "3.0.4"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "9e2a6b69137e6969bab0152632dcb3bc108c8bdd"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+1"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Static"]
git-tree-sha1 = "601f7e7b3d36f18790e2caf83a882d88e9b71ff1"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.4"

[[deps.CRC32c]]
uuid = "8bf52ea8-c179-5cab-976a-9e18b702a9bc"

[[deps.CRlibm]]
deps = ["CRlibm_jll"]
git-tree-sha1 = "32abd86e3c2025db5172aa182b982debed519834"
uuid = "96374032-68de-5a5b-8d9e-752f78720389"
version = "1.0.1"

[[deps.CRlibm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e329286945d0cfc04456972ea732551869af1cfc"
uuid = "4e9b3aee-d8a1-5a3d-ad8b-7d824db253f0"
version = "1.0.1+0"

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

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "70232f82ffaab9dc52585e0dd043b5e0c6b714f1"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.12"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "cd67fc487743b2f0fd4380d4cbd3a24660d0eec8"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.3"

[[deps.ColorBrewer]]
deps = ["Colors", "JSON", "Test"]
git-tree-sha1 = "61c5334f33d91e570e1d0c3eb5465835242582c4"
uuid = "a2cac450-b92f-5266-8821-25eda20663c8"
version = "0.4.0"

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

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

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

[[deps.CompositeTypes]]
git-tree-sha1 = "02d2316b7ffceff992f3096ae48c7829a8aa0638"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.3"

[[deps.ConcreteStructs]]
git-tree-sha1 = "f749037478283d372048690eb3b5f92a79432b34"
uuid = "2569d6c7-a4a2-43d3-a901-331e8e4be471"
version = "0.2.3"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "8cfa272e8bdedfa88b6aefbbca7c19f1befac519"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.3.0"

[[deps.ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "c53fc348ca4d40d7b371e71fd52251839080cbc9"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.4"
weakdeps = ["IntervalSets", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "8da84edb865b0b5b0100c0666a9bc9a0b71c553c"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.15.0"

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

[[deps.DelaunayTriangulation]]
deps = ["DataStructures", "EnumX", "ExactPredicates", "Random", "SimpleGraphs"]
git-tree-sha1 = "26eb8e2331b55735c3d305d949aabd7363f07ba7"
uuid = "927a84f5-c5f4-47a5-9785-b46e178433df"
version = "0.8.11"

[[deps.Deno_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cd6756e833c377e0ce9cd63fb97689a255f12323"
uuid = "04572ae6-984a-583e-9378-9577a1c2574d"
version = "1.33.4+0"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

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

[[deps.DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "Random", "StaticArrays", "Statistics"]
git-tree-sha1 = "51b4b84d33ec5e0955b55ff4b748b99ce2c3faa9"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.6.7"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.DualNumbers]]
deps = ["Calculus", "NaNMath", "SpecialFunctions"]
git-tree-sha1 = "5837a837389fccf076445fce071c8ddaea35a566"
uuid = "fa6b7ba4-c1ee-5f82-b5fc-ecf0adba8f74"
version = "0.6.8"

[[deps.DynamicPolynomials]]
deps = ["Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "fea68c84ba262b121754539e6ea0546146515d4f"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.5.3"

[[deps.EarCut_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "e3290f2d49e661fbd94046d7e3726ffcb2d41053"
uuid = "5ae413db-bbd1-5e63-b57d-d24a61df00f5"
version = "2.2.4+0"

[[deps.EnumX]]
git-tree-sha1 = "bdb1942cd4c45e3c678fd11569d5cccd80976237"
uuid = "4e289a0a-7415-4d19-859d-a7e5c4648b56"
version = "1.0.4"

[[deps.ExactPredicates]]
deps = ["IntervalArithmetic", "Random", "StaticArrays"]
git-tree-sha1 = "e8b8c949551f417e040f16e5c431b6e83e306e54"
uuid = "429591f6-91af-11e9-00e2-59fbe8cec110"
version = "2.2.7"

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

[[deps.ExprTools]]
git-tree-sha1 = "27415f162e6028e81c72b82ef756bf321213b6ec"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.10"

[[deps.Extents]]
git-tree-sha1 = "2140cd04483da90b2da7f99b2add0750504fc39c"
uuid = "411431e0-e8b7-467b-b5e0-f676ba4f2910"
version = "0.1.2"

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

[[deps.FastLapackInterface]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "b12f05108e405dadcc2aff0008db7f831374e051"
uuid = "29a986be-02c6-4525-aec4-84b980013641"
version = "2.0.0"

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "c5c28c245101bd59154f649e19b038d15901b5dc"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.16.2"

[[deps.FilePaths]]
deps = ["FilePathsBase", "MacroTools", "Reexport", "Requires"]
git-tree-sha1 = "919d9412dbf53a2e6fe74af62a73ceed0bce0629"
uuid = "8fc22ac5-c921-52a6-82fd-178b2807b824"
version = "0.8.3"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Printf", "Test", "UUIDs"]
git-tree-sha1 = "9f00e42f8d99fdde64d40c8ea5d14269a2e2c1aa"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.21"

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

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "73d1214fec245096717847c62d389a5d2ac86504"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.22.0"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

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

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "cf0fe81336da9fb90944683b8c41984b08793dad"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.36"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType]]
deps = ["CEnum", "FreeType2_jll"]
git-tree-sha1 = "907369da0f8e80728ab49c1c7e09327bf0d6d999"
uuid = "b38be410-82b0-50bf-ab77-7b57e271db43"
version = "4.1.1"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "d8db6a5a2fe1381c1ea4ef2cab7c69c2de7f9ea0"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.1+0"

[[deps.FreeTypeAbstraction]]
deps = ["ColorVectorSpace", "Colors", "FreeType", "GeometryBasics"]
git-tree-sha1 = "055626e1a35f6771fe99060e835b72ca61a52621"
uuid = "663a7486-cb36-511b-a19d-713bb74d65c9"
version = "0.10.1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "aa31987c2ba8704e23c6c8ba8a4f769d5d7e4f91"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.10+0"

[[deps.FunctionWrappers]]
git-tree-sha1 = "d62485945ce5ae9c0c48f124a84998d755bae00e"
uuid = "069b7b12-0de2-55c6-9aab-29f3d0a68a2e"
version = "1.1.3"

[[deps.FunctionWrappersWrappers]]
deps = ["FunctionWrappers"]
git-tree-sha1 = "b104d487b34566608f8b4e1c39fb0b10aa279ff8"
uuid = "77dc65aa-8811-40c2-897b-53d922fa7daf"
version = "0.1.3"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "2d6ca471a6c7b536127afccfa7564b5b39227fe0"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.1.5"

[[deps.GeoInterface]]
deps = ["Extents"]
git-tree-sha1 = "d4f85701f569584f2cff7ba67a137d03f0cfb7d0"
uuid = "cf35fbd7-0cd7-5166-be24-54bfbe79505f"
version = "1.3.3"

[[deps.GeometryBasics]]
deps = ["EarCut_jll", "Extents", "GeoInterface", "IterTools", "LinearAlgebra", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "424a5a6ce7c5d97cca7bcc4eac551b97294c54af"
uuid = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
version = "0.4.9"

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

[[deps.GridLayoutBase]]
deps = ["GeometryBasics", "InteractiveUtils", "Observables"]
git-tree-sha1 = "af13a277efd8a6e716d79ef635d5342ccb75be61"
uuid = "3955a311-db13-416c-9275-1d80ed98e5e9"
version = "0.10.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.Groebner]]
deps = ["AbstractAlgebra", "Combinatorics", "ExprTools", "Logging", "MultivariatePolynomials", "PrecompileTools", "PrettyTables", "Primes", "Printf", "Random", "SIMD", "TimerOutputs"]
git-tree-sha1 = "6b505ef15e55bdc5bb3ddbcfebdff1c9e67081e8"
uuid = "0b43b601-686d-58a3-8a1c-6623616c7cd4"
version = "0.5.1"

[[deps.GroupsCore]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6df9cd6ee79fc59feab33f63a1b3c9e95e2461d5"
uuid = "d5909c97-4eac-4ecc-a3dc-fdd0858a4120"
version = "0.4.2"

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

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "eb8fed28f4994600e29beef49744639d985a04b2"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.16"

[[deps.HypergeometricFunctions]]
deps = ["DualNumbers", "LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "f218fe3736ddf977e0e772bc9a586b2383da2685"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.23"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "8b72179abc660bfab5e28472e019392b97d0985c"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.4"

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "2e4520d67b0cef90865b3ef727594d2a58e0e1f8"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.11"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageCore]]
deps = ["AbstractFFTs", "ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "fc5d1d3443a124fde6e92d0260cd9e064eba69f8"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.1"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs"]
git-tree-sha1 = "bca20b2f5d00c4fbc192c3212da8fa79f4688009"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.7"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "355e2b974f2e3212a75dfb60519de21361ad3cb7"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.9"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3d09a9f60edf77f8a4d99f9e015e8fbf9989605d"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.7+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "ea8031dea4aff6bd41f1df8f2fdfb25b33626381"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.4"

[[deps.IntegerMathUtils]]
git-tree-sha1 = "b8ffb903da9f7b8cf695a8bead8e01814aa24b30"
uuid = "18e54dd8-cb9d-406c-a71d-865a43cbb235"
version = "0.1.2"

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
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IntervalArithmetic]]
deps = ["CRlibm", "RoundingEmulator"]
git-tree-sha1 = "c274ec586ea58eb7b42afd0c5d67e50ff50229b5"
uuid = "d1acc4aa-44c8-5952-acd4-ba5d80a2a253"
version = "0.22.5"
weakdeps = ["DiffRules", "RecipesBase"]

    [deps.IntervalArithmetic.extensions]
    IntervalArithmeticDiffRulesExt = "DiffRules"
    IntervalArithmeticRecipesBaseExt = "RecipesBase"

[[deps.IntervalSets]]
deps = ["Dates", "Random"]
git-tree-sha1 = "3d8866c029dd6b16e69e0d4a939c4dfcb98fac47"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.8"
weakdeps = ["Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.Isoband]]
deps = ["isoband_jll"]
git-tree-sha1 = "f9b6d97355599074dc867318950adaa6f9946137"
uuid = "f1662d9f-8043-43de-a69a-05efc1cc6ff4"
version = "0.1.1"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "fa6d0bcff8583bac20f1ffa708c3913ca605c611"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.5"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "60b1194df0a3298f460063de985eae7b01bc011a"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.1+0"

[[deps.KLU]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse_jll"]
git-tree-sha1 = "884c2968c2e8e7e6bf5956af88cb46aa745c854b"
uuid = "ef3ab10e-7fda-4108-b977-705223b18434"
version = "0.4.1"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "fee018a29b60733876eb557804b5b109dd3dd8a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.8"

[[deps.Krylov]]
deps = ["LinearAlgebra", "Printf", "SparseArrays"]
git-tree-sha1 = "8a6837ec02fe5fb3def1abc907bb802ef11a0729"
uuid = "ba0b0d4f-ebba-5204-a429-3ac8c609bfb7"
version = "0.9.5"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "f6250b16881adf048549549fba48b1161acdac8c"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.1+0"

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

[[deps.LabelledArrays]]
deps = ["ArrayInterface", "ChainRulesCore", "ForwardDiff", "LinearAlgebra", "MacroTools", "PreallocationTools", "RecursiveArrayTools", "StaticArrays"]
git-tree-sha1 = "f12f2225c999886b69273f84713d1b9cb66faace"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.15.0"

[[deps.LambertW]]
git-tree-sha1 = "c5ffc834de5d61d00d2b0e18c96267cffc21f648"
uuid = "984bce1d-4616-540c-a9ee-88d1112d94c9"
version = "0.4.6"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "62edfee3211981241b57ff1cedf4d74d79519277"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.15"

[[deps.Lazy]]
deps = ["MacroTools"]
git-tree-sha1 = "1370f8202dac30758f3c345f9909b97f53d87d3f"
uuid = "50d2b5c4-7a5e-59d5-8109-a42b560f39c0"
version = "0.15.1"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "7f3efec06033682db852f8b3bc3c1d2b0a0ab066"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.36.0+0"

[[deps.LightXML]]
deps = ["Libdl", "XML2_jll"]
git-tree-sha1 = "3a994404d3f6709610701c7dabfc03fed87a81f8"
uuid = "9c8b4983-aa76-5018-a973-4c85ecc9e179"
version = "0.9.1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "7bbea35cec17305fc70a0e5b4641477dc0789d9d"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.2.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LinearAlgebraX]]
deps = ["LinearAlgebra", "Mods", "Permutations", "Primes", "SimplePolynomials"]
git-tree-sha1 = "1cb349a6a7656c2cbe2d288baabe863a208e46e3"
uuid = "9b3f67b0-2d00-526e-9884-9e4938f8fb88"
version = "0.2.6"

[[deps.LinearSolve]]
deps = ["ArrayInterface", "ConcreteStructs", "DocStringExtensions", "EnumX", "FastLapackInterface", "GPUArraysCore", "InteractiveUtils", "KLU", "Krylov", "Libdl", "LinearAlgebra", "MKL_jll", "PrecompileTools", "Preferences", "RecursiveFactorization", "Reexport", "SciMLBase", "SciMLOperators", "Setfield", "SparseArrays", "Sparspak", "StaticArraysCore", "UnPack"]
git-tree-sha1 = "6f8e084deabe3189416c4e505b1c53e1b590cae8"
uuid = "7ed4a6bd-45f5-4d41-b270-4a48e9bafcae"
version = "2.22.1"

    [deps.LinearSolve.extensions]
    LinearSolveBandedMatricesExt = "BandedMatrices"
    LinearSolveBlockDiagonalsExt = "BlockDiagonals"
    LinearSolveCUDAExt = "CUDA"
    LinearSolveEnzymeExt = ["Enzyme", "EnzymeCore"]
    LinearSolveFastAlmostBandedMatricesExt = ["FastAlmostBandedMatrices"]
    LinearSolveHYPREExt = "HYPRE"
    LinearSolveIterativeSolversExt = "IterativeSolvers"
    LinearSolveKernelAbstractionsExt = "KernelAbstractions"
    LinearSolveKrylovKitExt = "KrylovKit"
    LinearSolveMetalExt = "Metal"
    LinearSolvePardisoExt = "Pardiso"
    LinearSolveRecursiveArrayToolsExt = "RecursiveArrayTools"

    [deps.LinearSolve.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockDiagonals = "0a1fb500-61f7-11e9-3c65-f5ef3456f9f0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FastAlmostBandedMatrices = "9d29842c-ecb8-4973-b1e9-a27b1157504e"
    HYPRE = "b5ffcf37-a2bd-41ab-a3da-4bd9bc8ad771"
    IterativeSolvers = "42fd0dbc-a981-5370-80f2-aaf504508153"
    KernelAbstractions = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
    KrylovKit = "0b1a1467-8014-51b9-945f-bf0ae24f4b77"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2"
    RecursiveArrayTools = "731186ca-8d62-57ce-b412-fbd966d074cd"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "0f5648fbae0d015e3abe5867bca2b362f67a5894"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.166"
weakdeps = ["ChainRulesCore", "ForwardDiff", "SpecialFunctions"]

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    SpecialFunctionsExt = "SpecialFunctions"

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

[[deps.Makie]]
deps = ["Animations", "Base64", "CRC32c", "ColorBrewer", "ColorSchemes", "ColorTypes", "Colors", "Contour", "DelaunayTriangulation", "Distributions", "DocStringExtensions", "Downloads", "FFMPEG_jll", "FileIO", "FilePaths", "FixedPointNumbers", "Formatting", "FreeType", "FreeTypeAbstraction", "GeometryBasics", "GridLayoutBase", "ImageIO", "InteractiveUtils", "IntervalSets", "Isoband", "KernelDensity", "LaTeXStrings", "LinearAlgebra", "MacroTools", "MakieCore", "Markdown", "MathTeXEngine", "Observables", "OffsetArrays", "Packing", "PlotUtils", "PolygonOps", "PrecompileTools", "Printf", "REPL", "Random", "RelocatableFolders", "Scratch", "Setfield", "ShaderAbstractions", "Showoff", "SignedDistanceFields", "SparseArrays", "StableHashTraits", "Statistics", "StatsBase", "StatsFuns", "StructArrays", "TriplotBase", "UnicodeFun"]
git-tree-sha1 = "a37c6610dd20425b131caf65d52abdf859da5ab1"
uuid = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
version = "0.20.4"

[[deps.MakieCore]]
deps = ["Observables", "REPL"]
git-tree-sha1 = "ec5db7bb2dc9b85072658dcb2d3ad09569b09ac9"
uuid = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
version = "0.7.2"

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MathTeXEngine]]
deps = ["AbstractTrees", "Automa", "DataStructures", "FreeTypeAbstraction", "GeometryBasics", "LaTeXStrings", "REPL", "RelocatableFolders", "UnicodeFun"]
git-tree-sha1 = "96ca8a313eb6437db5ffe946c457a401bbb8ce1d"
uuid = "0a4f8689-d25c-4efe-a92b-7142dfc1aa53"
version = "0.5.7"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[deps.Mods]]
git-tree-sha1 = "924f962b524a71eef7a21dae1e6853817f9b658f"
uuid = "7475f97c-0381-53b1-977b-4c60186c8d62"
version = "2.2.4"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.MsgPack]]
deps = ["Serialization"]
git-tree-sha1 = "f5db02ae992c260e4826fe78c942954b48e1d9c2"
uuid = "99f44e22-a591-53d1-9472-aa23ef4bd671"
version = "1.2.1"

[[deps.Multisets]]
git-tree-sha1 = "8d852646862c96e226367ad10c8af56099b4047e"
uuid = "3b2b4ff1-bcff-5658-a3ee-dbcf1ce5ac09"
version = "0.4.4"

[[deps.MultivariatePolynomials]]
deps = ["ChainRulesCore", "DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "769c9175942d91ed9b83fa929eee4fe6a1d128ad"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.5.4"

[[deps.MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "806eea990fb41f9b36f1253e5697aa645bf6a9f8"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "1.4.0"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

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

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "327f53360fdb54df7ecd01e96ef1983536d1e633"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.2"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "a4ca623df1ae99d09bc9868b008262d0c0ac1e4f"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.1.4+0"

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

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "01f85d9269b13fedc61e63cc72ee2213565f7a72"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.7.8"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "67186a2bc9a90f9f85ff3cc8277868961fb57cbd"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.3"

[[deps.Packing]]
deps = ["GeometryBasics"]
git-tree-sha1 = "ec3edfe723df33528e085e632414499f26650501"
uuid = "19eb6ba3-879d-56ad-ad62-d5c202156566"
version = "0.5.0"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Permutations]]
deps = ["Combinatorics", "LinearAlgebra", "Random"]
git-tree-sha1 = "eb3f9df2457819bf0a9019bd93cc451697a0751e"
uuid = "2ae35dd2-176d-5d53-8349-f30d82d94d4f"
version = "0.4.20"

[[deps.PikaParser]]
deps = ["DocStringExtensions"]
git-tree-sha1 = "d6ff87de27ff3082131f31a714d25ab6d0a88abf"
uuid = "3bbf5609-3e7b-44cd-8549-7c69f321e792"
version = "0.6.1"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "64779bc4c9784fee475689a1752ef4d5747c5e87"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.42.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.2"

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "Statistics"]
git-tree-sha1 = "862942baf5663da528f66d24996eb6da85218e76"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.0"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "68723afdb616445c6caaef6255067a8339f91325"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.55"

[[deps.Polyester]]
deps = ["ArrayInterface", "BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "ManualMemory", "PolyesterWeave", "Requires", "Static", "StaticArrayInterface", "StrideArraysCore", "ThreadingUtilities"]
git-tree-sha1 = "fca25670784a1ae44546bcb17288218310af2778"
uuid = "f517fe37-dbe3-4b94-8317-1923a5111588"
version = "0.7.9"

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "240d7170f5ffdb285f9427b92333c3463bf65bf6"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.1"

[[deps.PolygonOps]]
git-tree-sha1 = "77b3d3605fc1cd0b42d95eba87dfcd2bf67d5ff6"
uuid = "647866c9-e3ac-4575-94e7-e3d426903924"
version = "0.1.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "RecipesBase", "Setfield", "SparseArrays"]
git-tree-sha1 = "a9c7a523d5ed375be3983db190f6a5874ae9286d"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.6"
weakdeps = ["ChainRulesCore", "FFTW", "MakieCore", "MutableArithmetics"]

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

[[deps.PreallocationTools]]
deps = ["Adapt", "ArrayInterface", "ForwardDiff", "Requires"]
git-tree-sha1 = "01ac95fca7daabe77a9cb705862bd87016af9ddb"
uuid = "d236fae5-4411-538c-8e31-a6e3d9e00b46"
version = "0.4.13"

    [deps.PreallocationTools.extensions]
    PreallocationToolsReverseDiffExt = "ReverseDiff"

    [deps.PreallocationTools.weakdeps]
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"

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

[[deps.Primes]]
deps = ["IntegerMathUtils"]
git-tree-sha1 = "1d05623b5952aed1307bf8b43bec8b8d1ef94b6e"
uuid = "27ebfcd6-29c5-5fa9-bf4b-fb8fc14df3ae"
version = "0.5.5"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "00099623ffee15972c16111bcf84c58a0051257c"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.9.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "18e8f4d1426e965c7b532ddd260599e1510d26ce"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.0"

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

[[deps.RandomExtensions]]
deps = ["Random", "SparseArrays"]
git-tree-sha1 = "b8a399e95663485820000f26b6a43c794e166a49"
uuid = "fb686558-2515-59ef-acaa-46db3789a887"
version = "0.4.4"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

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

[[deps.RecursiveArrayTools]]
deps = ["Adapt", "ArrayInterface", "DocStringExtensions", "GPUArraysCore", "IteratorInterfaceExtensions", "LinearAlgebra", "RecipesBase", "Requires", "SparseArrays", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables"]
git-tree-sha1 = "27ee1c03e732c488ecce1a25f0d7da9b5d936574"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "3.3.3"

    [deps.RecursiveArrayTools.extensions]
    RecursiveArrayToolsFastBroadcastExt = "FastBroadcast"
    RecursiveArrayToolsMeasurementsExt = "Measurements"
    RecursiveArrayToolsMonteCarloMeasurementsExt = "MonteCarloMeasurements"
    RecursiveArrayToolsTrackerExt = "Tracker"
    RecursiveArrayToolsZygoteExt = "Zygote"

    [deps.RecursiveArrayTools.weakdeps]
    FastBroadcast = "7034ab61-46d4-4ed7-9d0f-46aef9175898"
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    MonteCarloMeasurements = "0987c9cc-fe09-11e8-30f0-b96dd679fdca"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.RecursiveFactorization]]
deps = ["LinearAlgebra", "LoopVectorization", "Polyester", "PrecompileTools", "StrideArraysCore", "TriangularSolve"]
git-tree-sha1 = "8bc86c78c7d8e2a5fe559e3721c0f9c9e303b2ed"
uuid = "f2c3362d-daeb-58d1-803e-2bc74f2840b4"
version = "0.2.21"

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

[[deps.RingLists]]
deps = ["Random"]
git-tree-sha1 = "f39da63aa6d2d88e0c1bd20ed6a3ff9ea7171ada"
uuid = "286e9d63-9694-5540-9e3c-4e6708fa07b2"
version = "0.2.8"

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

[[deps.RoundingEmulator]]
git-tree-sha1 = "40b9edad2e5287e05bd413a38f61a8ff55b9557b"
uuid = "5eaf0fd0-dfba-4ccb-bf02-d820a40db705"
version = "0.2.1"

[[deps.RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "6aacc5eefe8415f47b3e34214c1d79d2674a0ba2"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.12"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "d8911cc125da009051fb35322415641d02d9e37f"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.4.6"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "3aac6d68c5e57449f5b9b865c9ba50ac2970c4cf"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.42"

[[deps.SciMLBase]]
deps = ["ADTypes", "ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "EnumX", "FillArrays", "FunctionWrappersWrappers", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "Markdown", "PrecompileTools", "Preferences", "Printf", "RecipesBase", "RecursiveArrayTools", "Reexport", "RuntimeGeneratedFunctions", "SciMLOperators", "StaticArraysCore", "Statistics", "SymbolicIndexingInterface", "Tables", "TruncatedStacktraces"]
git-tree-sha1 = "09324a0ae70c52a45b91b236c62065f78b099c37"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "2.15.2"

    [deps.SciMLBase.extensions]
    SciMLBaseChainRulesCoreExt = "ChainRulesCore"
    SciMLBasePartialFunctionsExt = "PartialFunctions"
    SciMLBasePyCallExt = "PyCall"
    SciMLBasePythonCallExt = "PythonCall"
    SciMLBaseRCallExt = "RCall"
    SciMLBaseZygoteExt = "Zygote"

    [deps.SciMLBase.weakdeps]
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    PartialFunctions = "570af359-4316-4cb7-8c74-252c00c2016b"
    PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
    PythonCall = "6099a3de-0909-46bc-b1f4-468b9a2dfc0d"
    RCall = "6f49c342-dc21-5d91-9882-a32aef131414"
    Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f"

[[deps.SciMLOperators]]
deps = ["ArrayInterface", "DocStringExtensions", "Lazy", "LinearAlgebra", "Setfield", "SparseArrays", "StaticArraysCore", "Tricks"]
git-tree-sha1 = "51ae235ff058a64815e0a2c34b1db7578a06813d"
uuid = "c0aeaf25-5076-4817-a8d5-81caf7dfa961"
version = "0.3.7"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.ShaderAbstractions]]
deps = ["ColorTypes", "FixedPointNumbers", "GeometryBasics", "LinearAlgebra", "Observables", "StaticArrays", "StructArrays", "Tables"]
git-tree-sha1 = "db0219befe4507878b1a90e07820fed3e62c289d"
uuid = "65257c39-d410-5151-9873-9b3e5be5013e"
version = "0.4.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SignedDistanceFields]]
deps = ["Random", "Statistics", "Test"]
git-tree-sha1 = "d263a08ec505853a5ff1c1ebde2070419e3f28e9"
uuid = "73760f76-fbc4-59ce-8f25-708e95d2df96"
version = "0.4.0"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "874e8867b33a00e784c8a7e4b60afe9e037b74e1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.1.0"

[[deps.SimpleGraphs]]
deps = ["AbstractLattices", "Combinatorics", "DataStructures", "IterTools", "LightXML", "LinearAlgebra", "LinearAlgebraX", "Optim", "Primes", "Random", "RingLists", "SimplePartitions", "SimplePolynomials", "SimpleRandom", "SparseArrays", "Statistics"]
git-tree-sha1 = "f65caa24a622f985cc341de81d3f9744435d0d0f"
uuid = "55797a34-41de-5266-9ec1-32ac4eb504d3"
version = "0.8.6"

[[deps.SimplePartitions]]
deps = ["AbstractLattices", "DataStructures", "Permutations"]
git-tree-sha1 = "e9330391d04241eafdc358713b48396619c83bcb"
uuid = "ec83eff0-a5b5-5643-ae32-5cbf6eedec9d"
version = "0.3.1"

[[deps.SimplePolynomials]]
deps = ["Mods", "Multisets", "Polynomials", "Primes"]
git-tree-sha1 = "7063828369cafa93f3187b3d0159f05582011405"
uuid = "cc47b68c-3164-5771-a705-2bc0097375a0"
version = "0.2.17"

[[deps.SimpleRandom]]
deps = ["Distributions", "LinearAlgebra", "Random"]
git-tree-sha1 = "3a6fb395e37afab81aeea85bae48a4db5cd7244a"
uuid = "a6525b86-64cd-54fa-8f65-62fc48bdc0e8"
version = "0.3.1"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "2da10356e31327c7096832eb9cd86307a50b1eb6"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.3"

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

[[deps.Sparspak]]
deps = ["Libdl", "LinearAlgebra", "Logging", "OffsetArrays", "Printf", "SparseArrays", "Test"]
git-tree-sha1 = "342cf4b449c299d8d1ceaf00b7a49f4fbc7940e7"
uuid = "e56a9233-b9d6-4f03-8d0f-1825330902ac"
version = "0.3.9"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "e2cfc4012a19088254b3950b85c3c1d8882d864d"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.3.1"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableHashTraits]]
deps = ["Compat", "PikaParser", "SHA", "Tables", "TupleTools"]
git-tree-sha1 = "008ca4718b5b55983dd0ffc63ce1f029c4a88f35"
uuid = "c5dd0088-6c3f-4803-b00e-f31a60c170fa"
version = "1.1.5"

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "46e589465204cd0c08b4bd97385e4fa79a0c770c"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.1"

[[deps.Static]]
deps = ["IfElse"]
git-tree-sha1 = "f295e0a1da4ca425659c57441bcb59abb035a4bc"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.8.8"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Requires", "SparseArrays", "Static", "SuiteSparse"]
git-tree-sha1 = "5d66818a39bb04bf328e92bc933ec5b4ee88e436"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.5.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

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

[[deps.StrideArraysCore]]
deps = ["ArrayInterface", "CloseOpenIntervals", "IfElse", "LayoutPointers", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface", "ThreadingUtilities"]
git-tree-sha1 = "d6415f66f3d89c615929af907fdc6a3e17af0d8c"
uuid = "7792a7ef-975c-4747-a70f-980b88e8d1da"
version = "0.5.2"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[deps.StructArrays]]
deps = ["Adapt", "ConstructionBase", "DataAPI", "GPUArraysCore", "StaticArraysCore", "Tables"]
git-tree-sha1 = "0a3db38e4cce3c54fe7a71f831cd7b6194a54213"
uuid = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
version = "0.6.16"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.SymbolicIndexingInterface]]
git-tree-sha1 = "be414bfd80c2c91197823890c66ef4b74f5bf5fe"
uuid = "2efcf032-c050-4f8e-a9bb-153293bab1f5"
version = "0.3.1"

[[deps.SymbolicUtils]]
deps = ["AbstractTrees", "Bijections", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DocStringExtensions", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "TimerOutputs", "Unityper"]
git-tree-sha1 = "849b1dfb1680a9e9f2c6023f79a49b694fb6d0da"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "1.5.0"

[[deps.Symbolics]]
deps = ["ArrayInterface", "Bijections", "ConstructionBase", "DataStructures", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "DynamicPolynomials", "Groebner", "IfElse", "LaTeXStrings", "LambertW", "Latexify", "Libdl", "LinearAlgebra", "LogExpFunctions", "MacroTools", "Markdown", "NaNMath", "PrecompileTools", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicIndexingInterface", "SymbolicUtils"]
git-tree-sha1 = "8d28ebc206dec9e250e21b9502a2662265897650"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "5.14.1"

    [deps.Symbolics.extensions]
    SymbolicsSymPyExt = "SymPy"

    [deps.Symbolics.weakdeps]
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

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

[[deps.ThreadPools]]
deps = ["Printf", "RecipesBase", "Statistics"]
git-tree-sha1 = "50cb5f85d5646bc1422aa0238aa5bfca99ca9ae7"
uuid = "b189fb0b-2eb5-4ed4-bc0c-d34c51242431"
version = "2.1.1"

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "eda08f7e9818eb53661b3deb74e3159460dfbc27"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.2"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "ProgressMeter", "UUIDs"]
git-tree-sha1 = "34cc045dd0aaa59b8bbe86c644679bc57f1d5bd0"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.6.8"

[[deps.TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "f548a9e9c490030e545f72074a41edfd0e5bcdd7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.23"

[[deps.TranscodingStreams]]
git-tree-sha1 = "1fbeaaca45801b4ba17c251dd8603ef24801dd84"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.10.2"
weakdeps = ["Random", "Test"]

    [deps.TranscodingStreams.extensions]
    TestExt = ["Test", "Random"]

[[deps.TriangularSolve]]
deps = ["CloseOpenIntervals", "IfElse", "LayoutPointers", "LinearAlgebra", "LoopVectorization", "Polyester", "Static", "VectorizationBase"]
git-tree-sha1 = "fadebab77bf3ae041f77346dd1c290173da5a443"
uuid = "d5829a12-d9aa-46ab-831f-fb7c9ab06edf"
version = "0.1.20"

[[deps.Tricks]]
git-tree-sha1 = "eae1bb484cd63b36999ee58be2de6c178105112f"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.8"

[[deps.TriplotBase]]
git-tree-sha1 = "4d4ed7f294cda19382ff7de4c137d24d16adc89b"
uuid = "981d1d27-644d-49a2-9326-4793e63143c3"
version = "0.1.0"

[[deps.TruncatedStacktraces]]
deps = ["InteractiveUtils", "MacroTools", "Preferences"]
git-tree-sha1 = "ea3e54c2bdde39062abf5a9758a23735558705e1"
uuid = "781d530d-4396-4725-bb49-402e4bee1e77"
version = "1.4.0"

[[deps.TupleTools]]
git-tree-sha1 = "155515ed4c4236db30049ac1495e2969cc06be9d"
uuid = "9d95972d-f1c8-5527-a6e0-b4b365fa01f6"
version = "1.4.3"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unityper]]
deps = ["ConstructionBase"]
git-tree-sha1 = "25008b734a03736c41e2a7dc314ecb95bd6bbdb0"
uuid = "a7c27f48-0311-42f6-a7f8-2c11e75eb415"
version = "0.1.6"

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "7209df901e6ed7489fe9b7aa3e46fb788e15db85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.65"

[[deps.WGLMakie]]
deps = ["Bonito", "Colors", "FileIO", "FreeTypeAbstraction", "GeometryBasics", "Hyperscript", "LinearAlgebra", "Makie", "Observables", "PNGFiles", "PrecompileTools", "RelocatableFolders", "ShaderAbstractions", "StaticArrays"]
git-tree-sha1 = "0d696472ce6d5e255ecc0db0e6e74e2963061ba2"
uuid = "276b4fcb-3e11-5398-bf8b-a0c2d153d008"
version = "0.9.4"

[[deps.WidgetsBase]]
deps = ["Observables"]
git-tree-sha1 = "30a1d631eb06e8c868c559599f915a62d55c2601"
uuid = "eead4739-05f7-45a1-878c-cee36b57321c"
version = "0.1.4"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

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

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e92a1a012a10506618f10b7047e478403a046c77"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.isoband_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "51b5eeb3f98367157a7a12a1fb0aa5328946c03c"
uuid = "9a68df92-36a6-505f-a73e-abb412b6bfb4"
version = "0.2.3+0"

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

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "daacc84a041563f965be61859a36e17c4e4fcd55"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.2+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "93284c28274d9e75218a416c65ec49d0e0fcdf3d"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.40+0"

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Pkg", "libpng_jll"]
git-tree-sha1 = "d4f63314c8aa1e48cd22aa0c17ed76cd1ae48c3c"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.3+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "b910cb81ef3fe6e78bf6acee440bda86fd6ae00c"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+1"

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
"""

# ╔═╡ Cell order:
# ╟─1c205fbf-c856-4a87-b1ac-76d8c7f6d128
# ╠═5f72b834-d650-4d87-8a11-3ca5c922f7cf
# ╟─d3893588-6aa1-409d-8b1a-ee2c96fda393
# ╠═7ec18854-0620-11ed-2286-5d4f708b7418
# ╠═3983aab6-8ec1-4fd6-a23c-8ad7b94cfc28
# ╠═a73af075-0eea-4c7d-ac6d-32e3679f8617
# ╟─6d38a682-f292-4f07-879d-2e4c11c74da6
# ╠═c1b1a793-92f4-4e74-b217-5ea366d5fd89
# ╟─48c9f6d8-344c-434f-bb09-d3e118bfe9ca
# ╟─3a91da73-225c-48c3-89fa-fbf50540e14f
# ╟─0dc0f395-de9b-40b0-af83-f28f6d7da919
# ╠═26c0956b-fb82-44ef-8a9c-10bd366fa886
# ╠═a493f046-2453-4df5-a48b-71bc57f7944c
# ╟─a098cdd8-02a3-4944-9268-8e6bb5df1bf1
# ╟─9e019683-8bdc-4305-82b7-d512cb28726d
# ╟─90b3b854-3353-473d-a012-72a0be61886d
# ╠═92a9f1da-08b0-45f9-a5df-4cca91176470
# ╟─7db18b00-81a6-4098-a7f0-b7598ca66070
# ╠═3769d115-007e-4430-b67a-c213b69bc29f
# ╠═73e97ad5-fb08-4946-8247-0598ac2c8dee
# ╟─c41e3d38-a29c-4719-bb15-e2a58c1a7a86
# ╠═f72f0686-f3f8-4226-8c90-33da55ce5e29
# ╟─3da71a6a-5a20-4a27-acab-2a0f2c26b5e9
# ╟─8169ad19-e3ad-4795-86de-eca3e3a4b5cd
# ╠═8963b9fa-506b-43fa-9cf5-5a545a3ab3fd
# ╟─83d17a70-f8b0-430d-ab2d-014caeeb8fdf
# ╟─2a7364a7-7f92-46db-b176-77f7fb8ce7f9
# ╟─bed2c602-f4ad-428a-a8f8-e75105fbed1c
# ╟─d3cb8670-5899-41b2-8e63-43550c74ac40
# ╟─11e6ecdc-5cbc-4bfe-a9cb-6cfc649f4a6d
# ╟─29a1ea27-9676-44dc-bb38-a29bcaf33985
# ╠═4c94c8db-2085-4714-99a7-bee128a9c1bd
# ╠═868c117d-fb9e-4b07-a5cf-e96a73373060
# ╠═d81bacdf-ca72-480b-aab5-4be71020e3bf
# ╠═188a7dda-1407-4365-bd51-b79728f71b45
# ╠═f9e55d4b-1c1d-417f-b9ee-3f0d6a100a41
# ╟─fb81b55f-0404-4d11-bbee-1bb08cf7a5a8
# ╟─838b9237-1333-4afc-8f1f-8f06a9b882d7
# ╟─15bcb9cf-b468-4f77-817c-3267d7226c8d
# ╟─8e573ccb-86d9-4f3c-adbc-81018ecb7cf7
# ╟─f2027eec-5107-4ea2-9794-25bc159652e2
# ╟─b38d2850-c767-4b80-ac1a-e3a134bf44d8
# ╟─de9ce83c-0da6-4dd2-ace5-fdae8b4df025
# ╟─ea91d037-6ad2-44f3-b04f-fa26353b4240
# ╟─22c9db03-cacd-4658-9994-68b4ff29a76d
# ╠═0d859ce4-fb41-41d4-a804-39aca7aaebb8
# ╟─41f00d4b-2235-4dfd-9b97-e5207255d3b9
# ╟─76d92e16-4695-440c-b71a-610bf6f9c3ba
# ╟─6f81ce75-1bca-4073-a76f-0aab6d20c8a7
# ╟─f47e6d00-7eff-45b3-baf4-6529d866cf36
# ╟─4573d0ce-aee5-4b4c-b1e2-29df6a1dfc00
# ╟─a2726fff-8eff-4570-8a56-0a01b40c6a67
# ╟─f9b646b1-d201-48ab-8dd8-bb4083ba8f67
# ╠═3a44a0dd-4bb8-491a-bd74-c6f7e2c6c6f2
# ╠═c32cf782-607c-4a41-892a-1aae585e7010
# ╠═6371557e-1af0-4e2c-a7b3-37e989fff1ca
# ╟─4f4033a6-b2a1-42fb-afdd-2f5a54b4c82f
# ╠═34bf99c5-0a9f-4d60-8c26-2ba972f7e193
# ╠═b9a7f0bc-2fb4-4620-8364-c4419589e910
# ╠═ccc85829-c70b-4f32-bea4-3d9c568eadcc
# ╠═3a459c84-1a1c-4096-9c72-10ea11a7f2fc
# ╠═9639d400-07a3-40c9-adbb-4269b9c017d3
# ╠═b8bcf86b-8526-4c37-acd9-c266052acb84
# ╠═389ae6d5-ba80-4944-be4f-b61845a0fe2a
# ╠═cbc667b7-c224-4e4a-b17e-ba97c9ead32e
# ╟─48712159-24dd-4dba-90d0-84be4edea6ee
# ╠═95e0cf44-a8a4-4357-90e9-4a0422d89880
# ╠═ee6ebb3f-754f-4450-9842-f888ed09b682
# ╠═3c759c59-9495-40e3-8291-bc6c468ebb4c
# ╠═b477835b-ea36-4d84-a237-b37404b44d94
# ╠═97505e60-c591-43d7-99c3-2801e710010c
# ╠═15d1e543-5bce-41c9-b7fd-c7329b3d1e07
# ╠═99dc3b67-0bdb-426c-a925-4cd49d39b4c8
# ╠═8fbbe334-26e8-4d8a-9148-6559d4888f01
# ╠═acc3a162-6f44-436d-990b-afdf6addc877
# ╠═44862011-d0f8-4c63-91c0-2fc12d50161e
# ╟─79ce5c1e-d8ef-40ea-99e5-f605c9c2a24a
# ╠═5d18d532-80b1-42a3-bf37-4d7883e8de5b
# ╠═d990a16d-ee3e-436e-bf30-5c1ebcd0347e
# ╠═5110b4e7-a21d-4c3d-8bdf-933d8dd980e2
# ╠═b1aba160-1709-4758-a7cc-3bd543d474c4
# ╠═01bfa680-833a-4669-b312-79b474d14708
# ╠═cc1194a1-8c1a-4fe8-aed7-14265f745cc1
# ╠═66837355-8d31-4b4d-81d7-411b2e8ca021
# ╠═e36c75ec-8a91-48be-afb0-5d6b524e533f
# ╠═03c0b05d-eff0-41cb-9144-4b221d286d24
# ╠═c50c63fa-018b-456c-9203-6a003efc9b15
# ╠═5beb7506-c03a-452b-930e-6f6fdb894532
# ╠═f9b16cbf-3475-42fd-95be-75d730a60a01
# ╟─6ec92519-f7cd-4796-862f-37d6843ec1e4
# ╟─21a32643-939f-4faf-b7ad-ed832722d910
# ╠═b27264bb-d642-4cfb-9c72-183583635717
# ╟─a9209f0f-ae25-45fa-b48d-8a69f2a00634
# ╟─ce27527d-be20-4851-9d1e-c994f6a4649c
# ╟─efb64145-cfcf-4250-a1fc-1019e886a251
# ╟─5c0d1057-238a-42a6-977e-1585d17e52bf
# ╠═3855980b-50d5-4775-8706-9a12be3478e3
# ╟─8fa5884c-1227-40d4-9bda-c430ffc58b1b
# ╠═63facbc6-5527-4231-a4b8-aee8d2202e7e
# ╠═2cb0eb87-e1a6-4fb0-92e2-86417ea3789e
# ╟─112ff420-49f4-4b6f-8735-8b4fb0304e00
# ╟─71f14de3-5131-46b4-901f-ab902bc1ade8
# ╟─3151f84b-e134-4e40-a1f3-f7753604874b
# ╠═1b98a52e-d247-4d12-bcee-5d010731193c
# ╟─d00c1534-f5c6-4937-afbd-dd0ed51db090
# ╠═0d1723d4-be40-4373-afe6-9282df45f0a8
# ╠═45ecb711-2a01-47a6-afb6-8f3e4cff7fd2
# ╠═5df9929f-7581-44ab-a94d-7c5536e0e4c8
# ╟─d4b00d94-1db9-4c52-a8fc-cbc74d1efd2f
# ╟─26072ceb-052c-4836-8877-1e801cd27280
# ╟─a767a0dd-a6c4-42b5-a8e8-41b45bbababb
# ╟─01d7fbce-efe1-4324-b243-409aee569632
# ╟─2f90671f-4527-4d1b-8309-79f13f670ff4
# ╠═cc377faa-486b-4e36-af69-1241eb83a6a1
# ╠═bcbfa135-d6ac-4105-a06f-f7ee5658e6fd
# ╠═ac50a0da-f924-4499-a7c7-b5134f33fbc2
# ╠═f3b16944-8c93-4c5f-81af-a92307ee9681
# ╠═612a3ea2-81cf-40fe-bd6f-01b2b860e9be
# ╟─1d93390f-306c-482a-b33e-0e2695859229
# ╠═ba5e9605-71db-4818-bc16-0a2c8a45c0e9
# ╠═1ffcf623-542b-49d2-b69d-ff9f5caa3cea
# ╠═2274bfd4-67b8-4f9a-91fd-b93b4484b9d5
# ╠═99a97da1-96df-4111-b7bd-3d3a820ee8c9
# ╠═074bda6d-9e8e-4cfe-aba1-b99552581366
# ╠═d45d5016-e700-4659-a929-acd7a62a613e
# ╠═667e073d-57ba-41c6-90e9-3f7f1349dee0
# ╠═61623cb6-da6b-480b-b8bb-e1315d019765
# ╠═c107aecc-62ed-4341-a432-9e879451eda2
# ╠═3203640c-2810-41ad-95e6-cd6ce937eb94
# ╠═da35c661-ede7-4dc0-acaa-525f8772431d
# ╠═dd9fc92d-e4d1-4d78-9a48-86ba39e07b24
# ╠═bd6d32be-2f3c-49a3-92e3-9b08c44ddb1a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
