import random
import math
import csv

def importer_donnees_csv(nom_fichier):
    donnees = []
    with open(nom_fichier, 'r', newline='') as fichier_csv:
        lecteur_csv = csv.reader(fichier_csv)
        for ligne in lecteur_csv:
            donnees.append(ligne)
    return donnees

def convertir_en_nombre(liste_caracteres):
    liste_nombres = []
    for caractere in liste_caracteres:
        try:
            nombre = float(caractere)
            liste_nombres.append(nombre)
        except ValueError:
            # Gestion de l'erreur si le caractère ne peut pas être converti en nombre
            print(f"Le caractère '{caractere}' ne peut pas être converti en nombre.")
    return liste_nombres



def initialisation(n0, n1, n2):
    W1 = [[random.random() for _ in range(n0)] for _ in range(n1)]
    b1 = [random.random() for _ in range(n1)]
    W2 = [[random.random() for _ in range(n1)] for _ in range(n2)]
    b2 = [random.random() for _ in range(n2)]
    
    parametres = {
        'W1' : W1,
        'b1' : b1,
        'W2' : W2,
        'b2' : b2
    }
    return parametres

def mat_mul(matrix1, matrix2):
    rows1 = len(matrix1)
    cols1 = len(matrix1[0])
    rows2 = len(matrix2)
    cols2 = len(matrix2[0])

    if cols1 != rows2:
        raise ValueError("Le nombre de colonnes de la première matrice doit correspondre au nombre de lignes de la deuxième matrice.")

    result = [[0] * cols2 for _ in range(rows1)]

    for i in range(rows1):
        for j in range(cols2):
            for k in range(cols1):
                result[i][j] += matrix1[i][k] * matrix2[k][j]

    return result

def mat_sum(matrice, vector):
    
    if len(matrice) != len(vector):
        raise ValueError("le nombre de ligne des deux doivent etre egaux")

    result = [[0 for _ in range(len(matrice[0]))] for _ in range(len(matrice))]
    for  i in range(len(matrice)):
        for j in range(len(matrice[0])):
            result[i][j] = vector[i] + matrice[i][j]

    return result

def activation_sigmoid(Z):
    if isinstance(Z, list):
        if isinstance(Z[0], list):
            result = [[0 for _ in range(len(Z[0]))] for _ in range(len(Z))]
            for i in range(len(Z)):
                for j in range(len(Z[0])):
                    result[i][j] = 1/(1 + math.exp(-Z[i][j]))
        else:
            result = [0 for _ in range(len(Z))]

            for k in range(len(Z)):
                result[k] = 1/(1 + math.exp(-Z[k]))
    else:
        raise ValueError("entrer une list ou une matrice")
    return result

def forward_propagation(X, parametres):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    X_T = transpose(X)
    Z1 = mat_sum(mat_mul(W1, X_T), b1)
    A1 = activation_sigmoid(Z1)

    Z2 = mat_sum(mat_mul(W2, A1), b2)
    A2 = activation_sigmoid(Z2)

    activations = {
        'A1' : A1,
        'A2' : A2
    }
    return activations

def vec_sub(A2, Y):
    if isinstance(Y, list) and isinstance(Y[0], list):
        result = [[0 for _ in range(len(A2[0]))]]
        for i in range(len(A2[0])):
            result[0][i] = A2[0][i] - Y[0][i]
    else:
        raise ValueError('entrer un vecteur colonne')
    return result

def transpose(M):
    if isinstance(M, list) and isinstance(M[0], list):
        result = [[0] * len(M) for _ in range(len(M[0]))]
        for i in range(len(M[0])):
            for j in range(len(M)):
                result[i][j] = M[j][i]
    else:
        raise ValueError("entrer une matrice carre")
    return result

def sca_mul(M,a):
    if isinstance(M, list):
        if isinstance(M[0], list):
            result = [[0 for _ in range(len(M[0]))] for _ in range(len(M))]
            for i in range(len(M)):
                for j in range(len(M[0])):
                    result[i][j] = a * M[i][j]
        else:
            result = [0] * len(M)
            for i in range(len(M)):
                result[i] = a * M[i]
    else:
        raise ValueError("entrer une matrice ou un vecteur")
    return result

def sum_col(M):
    if isinstance(M, list):
        if isinstance(M[0], list):
            result = [0] * len(M)
            for i in range(len(M)):
                for j in range(len(M[0])):
                    result[i] += M[i][j]
            return result
        return M
    else:
        raise ValueError("entrer une matrice")

def multiplie(mat1, mat2):
    if isinstance(mat1, list) and isinstance(mat2, list) and len(mat1) == len(mat2):
        if isinstance(mat1[0], list) and isinstance(mat2[0], list) and len(mat1[0]) == len(mat2[0]):
            result = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
            for i in range(len(mat1)):
                for j in range(len(mat1[0])):
                    result[i][j] = mat1[i][j] * mat2[i][j] 
            return result
        elif isinstance(mat1[0], list) and isinstance(mat2[0], list) and len(mat1[0]) != len(mat2[0]):
            raise ValueError('entrer deux matrices de meme dimension')
            
        else:
            result = [0] * len(mat1)
            for i in range(len(mat1)):
                result[i] = mat1[i] * mat2[i]
            return result
    else:
        raise ValueError("entrer deux matrices ou deux vecteurs de meme taille")

def mat_sub(mat1, mat2):
    if isinstance(mat1, list) and isinstance(mat2, list) and len(mat1) == len(mat2):
        if isinstance(mat1[0], list) and isinstance(mat2[0], list) and len(mat1[0]) == len(mat2[0]):
            result = [[0 for _ in range(len(mat1[0]))] for _ in range(len(mat1))]
            for i in range(len(mat1)):
                for j in range(len(mat1[0])):
                    result[i][j] = mat1[i][j] - mat2[i][j]
            return result
        result = [0] * len(mat1)
        for i in range(len(mat1)):
            result[i] = mat1[i] - mat2[i]
        return result
    else:
        raise ValueError('entrer deux matrices ou vecteurs de meme taille')


def substrate(a, A):
    if isinstance(A, list):
        if isinstance(A[0], list):
            result = [[0] * len(A[0]) for _ in range(len(A))]
            for i in range(len(A)):
                for j in range(len(A[0])):
                    result[i][j] = a - A[i][j]
            return result
        result = [0] * len(A)
        for i in range(len(A)):
            result[i] = a - A[i]
        return result
    else:
        raise ValueError('veillez entrer un nombre en premier parametre et une liste en second')


def back_propagation(X, Y, parametres, activations):
    A1 = activations['A1']
    A2 = activations['A2']
    W2 = parametres['W2']

    m = len(Y[0])

    dZ2 = vec_sub(A2, Y)
    A1_T = transpose(A1)
    dW2 = sca_mul(mat_mul(dZ2, A1_T), 1/m)
    db2 = sca_mul(sum_col(dZ2), 1/m)

    W2_T = transpose(W2)
    dz1 = multiplie(mat_mul(W2_T, dZ2), A1)
    dZ1 = multiplie(dz1, substrate(1, A1))
    X_T = transpose(X)
    dW1 = sca_mul(mat_mul(dZ1, X), 1/m)
    db1 = sca_mul(sum_col(dZ1), 1/m)

    print(len(dZ1),len(dZ1[0]),len(X),len(X[0]))
    print(len(dZ2),len(dZ2[0]),len(A1_T),len(A1_T[0]))

    gradients = {
        'dW1' : dW1,
        'db1' : db1,
        'dW2' : dW2,
        'db2' : db2
    }
    return gradients

def update(gradients, parametres, learning_rate):

    W1 = parametres['W1']
    b1 = parametres['b1']
    W2 = parametres['W2']
    b2 = parametres['b2']

    dW1 = gradients['dW1']
    db1 = gradients['db1']
    dW2 = gradients['dW2']
    db2 = gradients['db2']

    W1 = mat_sub(W1, sca_mul(dW1, learning_rate))
    b1 = mat_sub(b1, sca_mul(db1, learning_rate))
    W2 = mat_sub(W2, sca_mul(dW2, learning_rate))
    b2 = mat_sub(b2, sca_mul(db2, learning_rate))

    parametres = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    return parametres


def predict(X, parametres):
    activations = forward_propagation(X, parametres)
    A2 = activations['A2']
    return A2 >= 0.5


def neural_network(X, Y, n1 = 30, learning_rate = 0.1, n_iter = 10):

    #initialisation des parametres
    n0 = len(X[0])
    n2 = len(Y)
    random.seed(0)
    parametres = initialisation(n0, n1, n2)

    #descente de gradiant
    for i in range(n_iter):
        activations = forward_propagation(X, parametres)
        print('.', end="")
        if (n_iter % 50) == 0:
            print()
        # mise a jour
        gradients = back_propagation(X, Y, parametres, activations)
        parametres = update(gradients, parametres, learning_rate)

    return parametres


nom_fichier = 'data_set.csv'  
donnees_importees = importer_donnees_csv(nom_fichier)

data_set = []
for ligne in donnees_importees[1:]:
    liste_caracteres = ligne
    liste_nombres = convertir_en_nombre(liste_caracteres)
    data_set.append(liste_nombres)

data_set_train = data_set[:953]
data_set_test = data_set[953:]

X_train = [liste[:-1] for liste in data_set_train]
Y_train = [liste[-1] for liste in data_set_train]

X_test = [liste[:-1] for liste in data_set_test]
Y_test = [liste[-1] for liste in data_set_test]


y_l = []
y_l.append(Y_train)
p = neural_network(X_train, y_l, n1=20)


X_t = []
X_t.append(X_test[87])
W1 = p['W1']
activation = forward_propagation(X_t, p)
rep = activation['A2']
print(rep)
print(Y_test[87])