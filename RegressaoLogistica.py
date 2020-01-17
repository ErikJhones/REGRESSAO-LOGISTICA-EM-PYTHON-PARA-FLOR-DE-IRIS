
# coding: utf-8

# In[27]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd

import math

import matplotlib

import numpy as np

import matplotlib.pyplot as plt

import random

from sklearn.preprocessing import MaxAbsScaler


# In[28]:


#plotar grafico scarterplot
def plotar(padroes, saidas, p1, p2):
    
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    
    n_p = [] #vetor de pontos de coloracao
    cord = 0.001 #variavel da dimensao do ponto
    #cira um vetor de posicoes bem pequenas para ser plotadas no grafico para colorir
    for j in range(100):
        nov = []
        cord2 = 0.001

        for i in range(100):
            nov = []
            nov.append(cord)
            nov.append(cord2)
        
            n_p.append(nov)
       
            cord2+=0.01
        
        cord+=0.01
    
    reta = [p1, p2]
    
    #colorir o grafico de acordo com  a reta
    for y in n_p:
        x = y[0]
        z = y[1]
        equacao_reta = ((reta[0][1] - reta[1][1]) * x) + ((reta[1][0] - reta[0][0]) * z) + ((reta[0][0] * reta[1][1]) - (reta[1][0] * reta[0][1]))

        if ( equacao_reta < 0):
            plt.scatter(y[0], y[1], s=40, c="green", alpha=0.2)
        elif (equacao_reta > 0): 
              
            plt.scatter(y[0], y[1], s=40, c="blue", alpha=0.2)
    
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="black", alpha=0.9)
        else:
            #print("lepra")  
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot(p1, p2) #reta
    plt.show()


    
#mplota o grafico media dos acertos vs epocas
def plot_media_acerto(epocas, valores):
    
    matplotlib.pyplot.plot(epocas, valores)
    matplotlib.pyplot.title('Taxa média de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Percentual Media')
    matplotlib.pyplot.ylim(50, 100)

    matplotlib.pyplot.show()

#plota o grafico variancia dos acertos vs epocas
def plot_vari_acerto(epocas, vari):
    
    matplotlib.pyplot.plot(epocas, vari)                       
    matplotlib.pyplot.title('Variancia de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Variancia')
    matplotlib.pyplot.ylim(0, 60)

    matplotlib.pyplot.show()
    
#plota o grafico do desvio padrao vs epoca
def plot_desv_acerto(epocas, desv):
    
    matplotlib.pyplot.plot(epocas, desv)                       
    matplotlib.pyplot.title('Desvio padrao de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Desvio padrao')
    matplotlib.pyplot.ylim(0, 10)

    matplotlib.pyplot.show()

    
#funcao de embaralhamento dos padroes
def embaralhar(padrao):
    
    padroes = padrao.copy()
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    saida_teste = []
    saida_treino = []

    random.shuffle(padroes) #embaralha o veto padroes
        
    l = len(padroes)

    lista = list(range(l)) #faz uma lista de 0 até tamanho do vetor padroes
    
    random.shuffle(lista) #embaralha o vetor lista
    
    tam_teste = int(0.2 * l) #tcalculo quanto é 20% da quantidae de padroes
        
    y = 0 #variavel auxiliar
        
    teste_tam = 0
    treino_tam = 0
    
    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
            teste_tam+=1
            teste.append(padroes[x].copy()) #passa um vator de padrao para o vetor de teste
                
        else:
            treino_tam+=1
            treino.append(padroes[x].copy())
        y+=1
    
    #povoa os vetores de saida
    for i in range(teste_tam):
        saida_teste.append(teste[i].pop(len(teste[i])-1))
    
    #povoa o vetor de treino
    for j in range(treino_tam):
        saida_treino.append(treino[j].pop(len(treino[j])-1))
    
    return treino, teste, saida_treino, saida_teste
    
#funcao que escolhe os valores pro treino e teste
def escolhe_valor_treino_teste(padroes, saidas): 
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    
    random.shuffle(padroes) #embaralha o veto padroes

    lista = list(range(len(padroes))) #faz uma lista de 0 até tamanho do vetor padroes

    random.shuffle(lista) #embaralha o vetor lista

    tam_teste = 0.2 * len(padroes) #tcalculo quanto é 20% da quantidae de padroes
    y = 0 #variavel auxiliar

    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
                teste.append(padroes[x])
        else:
                treino.append(padroes[x])
        y+=1

    return treino, teste


# In[29]:


#plotar grafico scarterplot
def plotar_sem_reta(padroes):
    
    area = 30  # tamanho das bolinhas
    
    cor1 = "red"
    cor2 = "blue"
    classes = ("ver", "zul")
    '''
    n_p = [] #vetor de pontos de coloracao
    cord = 0.001 #variavel da dimensao do ponto
    #cira um vetor de posicoes bem pequenas para ser plotadas no grafico para colorir
    for j in range(100):
        nov = []
        cord2 = 0.001

        for i in range(100):
            nov = []
            nov.append(cord)
            nov.append(cord2)
        
            n_p.append(nov)
       
            cord2+=0.01
        
        cord+=0.01
    
    reta = [p1, p2]
    
    #colorir o grafico de acordo com  a reta
    for y in n_p:
        x = y[0]
        z = y[1]
        equacao_reta = ((reta[0][1] - reta[1][1]) * x) + ((reta[1][0] - reta[0][0]) * z) + ((reta[0][0] * reta[1][1]) - (reta[1][0] * reta[0][1]))

        if ( equacao_reta < 0):
            plt.scatter(y[0], y[1], s=40, c="green", alpha=0.2)
        elif (equacao_reta > 0): 
              
            plt.scatter(y[0], y[1], s=40, c="blue", alpha=0.2)
    '''
    #print("\n",tam)
    for y in padroes:
        tam = len(y)-1       
        if (y[tam] == 0):
            plt.scatter(y[2], y[3], s=20, c="blue", alpha=0.9)
        elif (y[tam]==0.5):
            plt.scatter(y[2], y[3], s=20, c="red", alpha=0.9)
        else:
            plt.scatter(y[2], y[3], s=20, c="green", alpha=0.9)
    
    # titulo do grafico
    plt.title('Grafico scarter Plot')
 
    # insere legenda dos estados
    plt.legend(loc=1)
    #plt.plot(p1, p2) #reta
    plt.show()


# In[30]:


#funcao de embaralhamento dos padroes
def embaralhar(padrao):
    
    padroes = padrao.copy()
    
    teste = [] #vetor de teste 
    treino = [] #vetor de trino
    saida_teste = []
    saida_treino = []

    random.shuffle(padroes) #embaralha o veto padroes
        
    l = len(padroes)

    lista = list(range(l)) #faz uma lista de 0 até tamanho do vetor padroes
    
    random.shuffle(lista) #embaralha o vetor lista
    
    tam_teste = int(0.2 * l) #tcalculo quanto é 20% da quantidae de padroes
        
    y = 0 #variavel auxiliar
        
    teste_tam = 0
    treino_tam = 0
    
    for x in lista: #laço que povoa os vetores de treino e teste com os valores escolhidos
    
        if (y < tam_teste): #escolhe 20% de padores pra teste
            teste_tam+=1
            teste.append(padroes[x].copy()) #passa um vator de padrao para o vetor de teste
                
        else:
            treino_tam+=1
            treino.append(padroes[x].copy())
        y+=1
    
    #povoa os vetores de saida
    for i in range(teste_tam):
        saida_teste.append(teste[i].pop(len(teste[i])-1))
    
    #povoa o vetor de treino
    for j in range(treino_tam):
        saida_treino.append(treino[j].pop(len(treino[j])-1))
    
    return treino, teste, saida_treino, saida_teste


# In[31]:


def insert_ones(X):
    ones = np.ones([X.shape[0],1])
    return np.concatenate((ones,X),axis=1)


# In[45]:





# In[33]:



def sigmoid(z):
    return 1 / ( 1 + np.exp(-z))


# In[60]:


#vendo a sigmoid
nums = np.arange(-10, 10, step=1) ## criando uma distribuição entre -10 e 10
print(nums)
print(sigmoid(nums))
## plotando 
fig, ax = plt.subplots(figsize=(6,4))  
ax.plot(nums, sigmoid(nums), 'r')


# In[35]:


#calculo da funccao custo
def binary_cross_entropy(w, X, y):    
    
    m = len(X)
    
    parte1 = np.multiply(-y, np.log(sigmoid(X @ w.T)))
    parte2 = np.multiply((1 - y), np.log(1 - sigmoid(X @ w.T)))
    
    somatorio = np.sum(parte1 - parte2)
    
    return  somatorio/m


# In[36]:


#funcao que calcula o gradiente descendente
def gradient_descent(w,X,y,alpha,epoch):
    cost = np.zeros(epoch)
    for i in range(epoch):
        w = w - (alpha/len(X)) * np.sum((sigmoid(X @ w.T) - y)*X, axis=0)
        cost[i] = binary_cross_entropy(w, X, y)
    
    return w,cost


# In[149]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
#ler arquivo de dado e retorna uma lista com os padroes(rotulos)
def ler_arquivo(nome_arquivo, classe):
    
    arq = open(nome_arquivo, 'r')
    texto = arq.readlines()
    x = []
    padroes = []
    cont = 0

    for linha in texto :
    
        linha = linha.replace("\n","")
        linha = linha.replace(",", " ")
        c = []

        x = linha.split()
    
        if (len(x) == 0):
            break;
        #verifica quantas classes quero separar os padroes    
        if (classe == 2):    
            if(cont <50):
                x[len(x)-1] = 1
            else:
                x[len(x)-1] = 0
        elif (classe == 3):
            if(cont < 50):
                x[len(x)-1] = 0
            elif (cont >= 50 and cont < 100):
                x[len(x)-1] = 0.5
            else:
                x[len(x)-1] = 1
        
        for i in x:
            c.append(float(i))
            
        padroes.append(c)   
        cont+=1    
        
    
    arq.close()    
    return padroes

c = ler_arquivo("iris.data", 2)
#print("padroes\n", c)
#plotar(c,c,p1,p2)

dados = np.array(c)

# Instancia o MaxAbsScaler
p=MaxAbsScaler()

# Analisa os dados e prepara o padronizador
p.fit(dados)

dados = p.transform(dados)

f = dados.tolist() #vetor de padroes da iris 

nf = insert_ones(np.array(f))


alpha=0.01 # taxa de aprendizado

epoch = 10000


print("em verde estão os padrões para iris setosa e em azul os outros padrões")

plotar_sem_reta(f)


# In[137]:





# In[131]:


def graf_erro_epoca(epoch, cost):
    fig, ax = plt.subplots()  
    ax.plot(np.arange(epoch), cost, 'r')  
    ax.set_xlabel('Iterações')  
    ax.set_ylabel('Custo')  
    ax.set_title('Erro vs. Epoch')


# In[138]:


w


# In[139]:


#funcao de teste
def func_teste(w, X, threshold=0.5):
    p = sigmoid(X @ w.T) >= threshold
    return (p.astype('int'))


# In[142]:



    


# In[ ]:


array([[ 1.72567485, -0.16488934,  1.70765133, -3.53322681, -4.74774267]])  #w otimo


# In[156]:


#mplota o grafico media dos acertos vs epocas
def plot_media_acerto(epocas, valores):
    
    matplotlib.pyplot.plot(epocas, valores)
    matplotlib.pyplot.title('Taxa média de acertos por épocas')
    matplotlib.pyplot.xlabel('Epocas')
    matplotlib.pyplot.ylabel('Percentual Media')
    matplotlib.pyplot.ylim(90, 110)

    matplotlib.pyplot.show()


# In[159]:


#teste para algumas amostras

valor = 2 #numero de execucoes
limite = 0

acertos = []
epocas = []

while (limite < valor):
    epocas.append(limite)
    treino, teste, saida_treino, saida_teste = embaralhar(nf.tolist())
    w = np.random.rand(1,5) ## valores entre 0 e 1
    
    #faz o vetor de saida do treino ficar nas mesmas dimensoes do vetor de sigmoid
    saida = np.zeros(((120,1)))  
    for i in range(len(saida_treino)):
        saida[i] = saida_treino[i]

    w, cost = gradient_descent(w, treino, saida, alpha, epoch) #calcula o vetor de pesos
    
    #calcula os acertos para cada vetor de teste
    acerto = 0
    for i in range(len(teste)):
        if(func_teste(w,teste[i]) == saida_teste[i]):
            acerto += 1
        #print(func_teste(w,teste[i]), saida_teste[i])
        
    taxa_acerto = acerto*100 / len(teste)
    acertos.append(taxa_acerto)
    
    #print("acurácia = ",  taxa_acerto)
    print("peso final da epoca ", limite, " é igual a = ", w)
    limite+=1
    
plot_media_acerto(epocas, acertos)

