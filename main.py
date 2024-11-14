# from PIL import Image
# import matplotlib.pyplot as plt
# import numpy as np
# import math
#
# import tkinter as tk
# from tkinter import *
#
# def addImagePath(window):
#     lbl = Label(window, text="Image path")
#     lbl.grid(column=0, row=7)
#     line = Entry(window)
#     line.grid(column=1, row=7)
#
# def prepareMatrix(image):
#     image = image.convert("L")
#     data = image.getdata()
#     data = np.matrix(data)
#
#     data = data / 255
#     data = 1 - data
#
#     data = np.reshape(data,(7,5))
#
#     return data
#
# # ветктор по столбцам
# def prepareVector(image):
#     image = image.convert("L")
#     data = image.getdata()
#     data = np.matrix(data)
#
#     data = data / 255
#     data = 1 - data
#
#     data = np.reshape(data, (7, 5))
#     data = np.reshape(data, data.size, order='F')
#
#     return data
#
# #развернуть вектор
# def revertVector(vector):
#     return np.reshape(vector,(35,1))
#
# def revertVector2(vector):
#     return np.reshape(vector,(1,35))
#
# # Косинус фи
# def getCosFi(vectorFirst, vectorSecond):
#
#     vectorFirstRevert = revertVector(vectorFirst)
#     vectorSecondRevert = revertVector(vectorSecond)
#
#     dot = np.dot(vectorFirst, vectorFirstRevert)
#     VectorFirstPrepare = np.sqrt(np.dot(vectorFirst, vectorFirstRevert))
#     VectorSecondPrepare = np.sqrt(np.dot(vectorSecond, vectorSecondRevert))
#
#     result = dot / (VectorFirstPrepare * VectorSecondPrepare)
#     print(np.around(dot[0,0], 3) )
#     print(np.around(VectorFirstPrepare[0,0], 3))
#     print(np.around(VectorSecondPrepare[0,0], 3))
#     print(np.around(result[0,0], 3))
#     print("Радианы = ", np.around(math.acos(result[0,0]), 3))
#     print("Градусы = ", np.around(math.acos(result[0,0]) * 57.296 ,3))
#
#     return result[0,0]
#
#
# def run():
#     from PIL import Image
#
#     MatrixVectorA1Result = []
#     MatrixVectorA2Result = []
#     MatrixVectorB1Result = []
#     MatrixVectorB2Result = []
#
#     Ampl1 = 0
#     Ampl2 = 0
#     Ampl3 = 0
#     Ampl4 = 0
#
#     ver1 = 0
#     ver2 = 0
#     ver3 = 0
#     ver4 = 0
#
#     imageFirstPath = firstVarTxt.get()
#     imageSecondPath = secondVarTxt.get()
#     imageThirdPath = ThirdVarTxt.get()
#     coeficent = int(CoefTxt.get())
#
#     for x in range(100):
#         np.set_printoptions(precision=0)
#
#         imageFirstPath = r"C:\Users\User\Image\1.jpg"
#         #imageSecondPath = "C:\Users\User\Image\2.jpg"
#         #imageThirdPath = "C:\Users\User\Image\3.jpg"
#
#         imageFirst = Image.open(imageFirstPath)
#         imageSecond = Image.open(imageSecondPath)
#         imageThird = Image.open(imageThirdPath)
#
#         matrixFirst = prepareMatrix(imageFirst)
#         matrixSecond = prepareMatrix(imageSecond)
#         matrixThird = prepareMatrix(imageThird)
#
#         print("Матрица первого изображения\n", matrixFirst, "\n")
#         print("Матрица второго изображения\n", matrixSecond, "\n")
#         print("Матрица третьего изображения\n", matrixThird, "\n")
#
#         vectorFirst = prepareVector(imageFirst)
#         vectorSecond = prepareVector(imageSecond)
#         vectorThird = prepareVector(imageThird)
#
#         print("Вектор первого изображения\n", vectorFirst, "\n")
#         print("Вектор второго изображения\n", vectorSecond, "\n")
#         print("Вектор третьего изображения\n", vectorThird, "\n")
#
#         # развернуть вектор
#
#         vectorFirstRevert = revertVector(vectorFirst)
#         vectorSecondRevert = revertVector(vectorSecond)
#         vectorThirdRevert = revertVector(vectorThird)
#
#         print("Развернутый вектор первого изображения\n", vectorFirstRevert, "\n")
#         print("Развернутый вектор второго изображения\n", vectorSecondRevert, "\n")
#         print("Развернутый вектор третьего изображения\n", vectorThirdRevert, "\n")
#
#         # получить косинусы фи
#         print("Фи для вектора 1 и 2")
#         getCosFi(vectorFirst, vectorSecond)
#         print("Фи для вектора 2 и 3")
#         getCosFi(vectorSecond, vectorThird)
#
#         # сложить вектор
#
#         matrixA = np.hstack((vectorFirstRevert, vectorSecondRevert))
#         matrixB = np.hstack((vectorSecondRevert, vectorThirdRevert))
#
#         print("Матрица А\n", matrixA, "\n")
#         print("Матрица В\n", matrixB, "\n")
#
#         matrixATransponse = np.transpose(matrixA)
#         matrixBTransponse = np.transpose(matrixB)
#
#         print("Матрица А транспанированная \n", matrixATransponse, "\n")
#         print("Матрица B транспанированная \n", matrixBTransponse, "\n")
#
#         dotA = np.dot(matrixATransponse, matrixA)
#         dotB = np.dot(matrixBTransponse, matrixB)
#
#         print("matrixATransponse * matrixA \n", dotA, "\n")
#         print("matrixBTransponse * matrixA \n", dotB, "\n")
#
#         inversedotA = np.linalg.inv(dotA)
#         inversedotB = np.linalg.inv(dotB)
#
#         print("inversedotA \n", inversedotA, "\n")
#         print("inversedotB \n", inversedotB, "\n")
#
#         dotInverseA = np.dot(inversedotA, matrixATransponse)
#         dotInverseB = np.dot(inversedotB, matrixBTransponse)
#
#         print("dotInverse \n", inversedotA, "\n")
#         print("dotInverse \n", inversedotB, "\n")
#
#         # Проверка обратных матриц для matrixA и matrixB
#         print("Проверка обратной матрицы для matrixA:")
#         identity_matrix_A = np.dot(dotA, inversedotA)
#         print("Произведение matrixA на её обратную матрицу:")
#         print(np.around(identity_matrix_A, 2))
#
#         print("\nПроверка обратной матрицы для matrixB:")
#         identity_matrix_B = np.dot(dotB, inversedotB)
#         print("Произведение matrixB на её обратную матрицу:")
#         print(np.around(identity_matrix_B, 2))
#
#
# # x1-x2
#         # для матрицы 1
#         resultAFirst = np.dot(dotInverseA, vectorFirstRevert)
#         resultAFirst = np.around(resultAFirst)
#
#         resultASecond = np.dot(dotInverseA, vectorSecondRevert)
#         resultASecond = np.around(resultASecond)
#
#         print("\nМатрица А")
#         print("dotInverse \n", resultAFirst)
#         print("dotInverse \n", resultASecond)
#
#         xOne = resultAFirst[0] - resultAFirst[1]
#         xTwo = resultASecond[0] - resultASecond[1]
#
#         print("\nДля первой матрицы")
#         print("xOne = x1 - x2 \n", xOne)
#         print("xTwo = x1 - x2 \n", xTwo)
#
#         # для матрицы 2
#
#         resultBFirst = np.dot(dotInverseB, vectorSecondRevert)
#         resultBFirst = np.around(resultBFirst)
#
#         resultBSecond = np.dot(dotInverseB, vectorThirdRevert)
#         resultBSecond = np.around(resultBSecond)
#
#         print("\nМатрица B")
#         print("dotInverse \n", resultBFirst)
#         print("dotInverse \n", resultBSecond)
#
#         xOne = resultBFirst[0] - resultBFirst[1]
#         xTwo = resultBSecond[0] - resultBSecond[1]
#
#         print("\nДля второй матрицы")
#         print("xOne = x1 - x2 \n", xOne)
#         print("xTwo = x1 - x2 \n", xTwo)
#
#         det_A = determinant(matrixA)
#         det_B = determinant(matrixB)
#         print("Определитель матрицы A:", det_A)
#         print("Определитель матрицы B:", det_B)
#
# # графический интерфейс
# window = Tk()
#
# # настройки окна
# window.title("M021")
# window.geometry('350x200')
#
# # метки
# lbl1 = Label(window, text="First image path")
# lbl2 = Label(window, text="Second image path")
# lbl3 = Label(window, text="Third image path")
# lbl4 = Label(window, text="Coefficent k")
#
# lbl1.grid(column=0, row=0)
# lbl2.grid(column=0, row=1)
# lbl3.grid(column=0, row=2)
# lbl4.grid(column=0, row=3)
#
# # поля для ввода текста
# firstVarTxt = Entry(window)
# secondVarTxt = Entry(window)
# ThirdVarTxt = Entry(window)
# CoefTxt = Entry(window)
#
# firstVarTxt.grid(column=1, row=0)
# secondVarTxt.grid(column=1, row=1)
# ThirdVarTxt.grid(column=1, row=2)
# CoefTxt.grid(column=1, row=3)
#
# # кнопки
# btnStart = Button(window, text="Start calc", command=run)
# btnAddImage = Button(window, text="Add image line", command=addImagePath)
#
# btnStart.grid(column=2, row=0)
#
# btnAddImage.grid(column=2, row=2)
#
#
# window.mainloop()


import PIL
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

import tkinter as tk
from tkinter import *

def addImagePath(window):
    lbl = Label(window, text="Image path")
    lbl.grid(column=0, row=7)
    line = Entry(window)
    line.grid(column=1, row=7)

def prepare_matrix(image) -> np.matrix:
    image = image.convert('L')
    data = np.matrix(image.getdata())
    data = (data < 255 * .5).astype('int').reshape(image.height, image.width)
    return data

def get_cos_phi(vectorFirst: np.matrix, vectorSecond: np.matrix):
    """Вычисление косинуса фи"""

    vectorFirstRevert = vectorFirst.transpose()
    vectorSecondRevert = vectorSecond.transpose()

    dot = np.dot(vectorFirst, vectorFirstRevert)
    VectorFirstPrepare = np.sqrt(np.dot(vectorFirst, vectorFirstRevert))
    VectorSecondPrepare = np.sqrt(np.dot(vectorSecond, vectorSecondRevert))

    result = dot / (VectorFirstPrepare * VectorSecondPrepare)
    print(np.around(dot[0, 0], 3))
    print(np.around(VectorFirstPrepare[0, 0], 3))
    print(np.around(VectorSecondPrepare[0, 0], 3))
    print(np.around(result[0, 0], 3))
    print("Радианы = ", np.around(math.acos(result[0, 0]), 3))
    print("Градусы = ", np.around(math.acos(result[0, 0]) * 57.296, 3))

    return result[0, 0]

def determinant(matrix):
    return np.linalg.det(matrix)

def run():
    from PIL import Image
    errors_A_first_vector = []
    errors_A_second_vector = []
    errors_B_second_vector = []
    errors_B_third_vector = []

    # Загрузка и подготовка матриц и векторов из изображений
    image_first_path = firstVarTxt.get()
    image_second_path = secondVarTxt.get()
    image_third_path = ThirdVarTxt.get()

    if any([not image_first_path, not image_second_path, not image_third_path]):
        print('Введите путь до изображений!')
        return

    paths = [
        image_first_path,
        image_second_path,
        image_third_path
    ]

    images = [Image.open(path) for path in paths]

    matrices = [prepare_matrix(image) for image in images]

    print(f'Матрица первого изображения: \n {matrices[0]}', end='\n\n')
    print(f'Матрица второго изображения: \n {matrices[1]}', end='\n\n')
    print(f'Матрица третьего изображения: \n {matrices[2]}', end='\n\n')

    vectors = [matrix.flatten() for matrix in matrices]

    print(f'Вектор первого изображения: \n {vectors[0]}', end='\n\n')
    print(f'Вектор второго изображения: \n {vectors[1]}', end='\n\n')
    print(f'Вектор третьего изображения: \n {vectors[2]}', end='\n\n')

    reverted_vectors = [vector.transpose() for vector in vectors]

    print(f'Развернутый вектор первого изображения: \n {reverted_vectors[0]}', end='\n\n')
    print(f'Развернутый вектор второго изображения: \n {reverted_vectors[1]}', end='\n\n')
    print(f'Развернутый вектор третьего изображения: \n {reverted_vectors[2]}', end='\n\n')

    print('Фи для вектора 1 и 2: ')
    get_cos_phi(vectors[0], vectors[1])

    print('Фи для вектора 2 и 3: ')
    get_cos_phi(vectors[1], vectors[2])

    A = np.hstack((vectors[0].transpose(), vectors[1].transpose()))
    B = np.hstack((vectors[1].transpose(), vectors[2].transpose()))

    print(f'Матрица A: \n {A}', end='\n\n')
    print(f'Матрица B: \n {B}', end='\n\n')

    print(f'Матрица A транспонированная: \n {A.transpose()}', end='\n\n')
    print(f'Матрица B транспонированная: \n {B.transpose()}', end='\n\n')

    A_dot = A.transpose() * A
    B_dot = B.transpose() * B

    print(f'(Матрица A транспонированная) * (Матрица A): \n {A_dot}', end='\n\n')
    print(f'(Матрица B транспонированная) * (Матрица B): \n {B_dot}', end='\n\n')

    A_dot_inv = np.linalg.inv(A_dot)
    B_dot_inv = np.linalg.inv(B_dot)

    print(f'Инвертированная матрица = (Матрица A транспонированная) * (Матрица A): \n {A_dot_inv}', end='\n\n')
    print(f'Инвертированная матрица = (Матрица B транспонированная) * (Матрица B): \n {B_dot_inv}', end='\n\n')

    # =========== Прямой метод ===========

    A_dot_inv_dot = A_dot_inv * A.transpose()
    B_dot_inv_dot = B_dot_inv * B.transpose()

    print(f'A_dot_inv_dot: \n {A_dot_inv_dot}', end='\n\n')
    print(f'B_dot_inv_dot: \n {B_dot_inv_dot}', end='\n\n')

    print('Результат прямого метода', end='\n\n')

    # ------ Матрица A
    result_A_first_vector = np.round(A_dot_inv_dot * vectors[0].transpose())
    result_A_second_vector = np.round(A_dot_inv_dot * vectors[1].transpose())

    print(f'Для матрицы A и первого вектора: \n {result_A_first_vector}', end='\n\n')
    print(f'Для матрицы A и второго вектора: \n {result_A_second_vector}', end='\n\n')

    A_x_one = result_A_first_vector[0] - result_A_first_vector[1]
    A_x_two = result_A_second_vector[0] - result_A_second_vector[1]

    print(f'x_one = x1 - x2: \n {A_x_one}')
    print(f'x_two = x1 - x2: \n {A_x_two}', end='\n\n')

    # ------ Матрица B

    result_B_second_vector = np.round(B_dot_inv_dot * vectors[1].transpose())
    result_B_third_vector = np.round(B_dot_inv_dot * vectors[2].transpose())

    print(f'Для матрицы B и второго вектора: \n {result_B_second_vector}', end='\n\n')
    print(f'Для матрицы B и третьего вектора: \n {result_B_third_vector}', end='\n\n')

    B_x_one = result_B_second_vector[0] - result_B_second_vector[1]
    B_x_two = result_B_third_vector[0] - result_B_third_vector[1]

    print(f'x_one = x1 - x2: \n {B_x_one}')
    print(f'x_two = x1 - x2: \n {B_x_two}')

    # =========== Итерационный метод ===========

    try:
        n_iters = int(NIterTxt.get())
        coefficient = float(CoefTxt.get())
    except ValueError:
        print('Для итерационного метода введите коеффициент и количество итераций !')
        return

    iter_results_A_first_vector = [
        A.transpose() * vectors[0].transpose() * coefficient
    ]

    iter_results_A_second_vector = [
        A.transpose() * vectors[1].transpose() * coefficient
    ]

    iter_results_B_second_vector = [
        B.transpose() * vectors[1].transpose() * coefficient
    ]

    iter_results_B_third_vector = [
        B.transpose() * vectors[2].transpose() * coefficient
    ]

    for i in range(1, n_iters + 1):
        print(f'Результаты итерации №{i}', end='\n\n')

        iter_results_A_first_vector.append(
            (np.identity(A_dot.shape[0]) - A_dot * coefficient) * iter_results_A_first_vector[i - 1] + A.transpose() *
            vectors[0].transpose() * coefficient
        )

        iter_results_A_second_vector.append(
            (np.identity(A_dot.shape[0]) - A_dot * coefficient) * iter_results_A_second_vector[i - 1] + A.transpose() *
            vectors[1].transpose() * coefficient
        )

        iter_results_B_second_vector.append(
            (np.identity(B_dot.shape[0]) - B_dot * coefficient) * iter_results_B_second_vector[i - 1] + B.transpose() *
            vectors[1].transpose() * coefficient
        )

        iter_results_B_third_vector.append(
            (np.identity(B_dot.shape[0]) - B_dot * coefficient) * iter_results_B_third_vector[i - 1] + B.transpose() *
            vectors[2].transpose() * coefficient
        )

        print(f'Для матрицы A и первого вектора: \n {iter_results_A_first_vector[-1]}', end='\n\n')
        print(f'Для матрицы A и второго вектора: \n {iter_results_A_second_vector[-1]}', end='\n\n')

        iter_A_x_one = iter_results_A_first_vector[-1][0] - iter_results_A_first_vector[-1][1]
        iter_A_x_two = iter_results_A_second_vector[-1][0] - iter_results_A_second_vector[-1][1]

        print(f'x_one = x1 - x2: \n {iter_A_x_one}')
        print(f'x_two = x1 - x2: \n {iter_A_x_two}', end='\n\n')

        print(f'Для матрицы B и второго вектора: \n {iter_results_B_second_vector[-1]}', end='\n\n')
        print(f'Для матрицы B и третьего вектора: \n {iter_results_B_third_vector[-1]}', end='\n\n')

        iter_B_x_one = iter_results_B_second_vector[-1][0] - iter_results_B_second_vector[-1][1]
        iter_B_x_two = iter_results_B_third_vector[-1][0] - iter_results_B_third_vector[-1][1]

        print(f'x_one = x1 - x2: \n {iter_B_x_one}')
        print(f'x_two = x1 - x2: \n {iter_B_x_two}')

        if i > 1:  # начиная со второй итерации, так как для первой нет "предыдущего" результата
            error_A_first_vector = np.linalg.norm(iter_results_A_first_vector[i] - iter_results_A_first_vector[i - 1])
            error_A_second_vector = np.linalg.norm(
                iter_results_A_second_vector[i] - iter_results_A_second_vector[i - 1])
            error_B_second_vector = np.linalg.norm(
                iter_results_B_second_vector[i] - iter_results_B_second_vector[i - 1])
            error_B_third_vector = np.linalg.norm(iter_results_B_third_vector[i] - iter_results_B_third_vector[i - 1])

            errors_A_first_vector.append(error_A_first_vector)
            errors_A_second_vector.append(error_A_second_vector)
            errors_B_second_vector.append(error_B_second_vector)
            errors_B_third_vector.append(error_B_third_vector)

    print('\nСравнение результатов прямого и итеративного методов', end='\n\n')

    print('Для матрицы A и первого вектора', end='\n\n')

    print(f'Прямой метод: \n{np.around(result_A_first_vector, 4)}', end='\n\n')
    print(f'Итеративный метод: \n{np.around(iter_results_A_first_vector[-1], 4)}', end='\n\n')

    print('\nДля матрицы A и второго вектора', end='\n\n')

    print(f'Прямой метод: \n{np.around(result_A_second_vector, 4)}', end='\n\n')
    print(f'Итеративный метод: \n{np.around(iter_results_A_second_vector[-1], 4)}', end='\n\n')

    print('\nДля матрицы B и второго вектора', end='\n\n')

    print(f'Прямой метод: \n{np.around(result_B_second_vector, 4)}', end='\n\n')
    print(f'Итеративный метод: \n{np.around(iter_results_B_second_vector[-1], 4)}', end='\n\n')

    print('\nДля матрицы B и третьего вектора', end='\n\n')

    print(f'Прямой метод: \n{np.around(result_B_third_vector, 4)}', end='\n\n')
    print(f'Итеративный метод: \n{np.around(iter_results_B_third_vector[-1], 4)}', end='\n\n')

    plt.figure(figsize=(12, 10))
    plt.plot(errors_A_first_vector, label='Ошибка первого вектора для матрицы А')
    plt.plot(errors_A_second_vector, label='Ошибка второго вектора для матрицы А')
    plt.plot(errors_B_second_vector, label='Ошибка второго вектора для матрицы В')
    plt.plot(errors_B_third_vector, label='Ошибка третьего вектора для матрицы В')
    plt.xlabel('i')
    plt.ylabel('Остаточная ошибка')
    plt.title('График зависимости остаточной ошибки распознавания изображений от номера итерации')
    plt.legend()
    plt.show()

# графический интерфейс
window = Tk()

# настройки окна
window.title("M021")
window.geometry('400x200')

# метки
lbl1 = Label(window, text="Первый путь к изображению")
lbl2 = Label(window, text="Второй путь к изображению")
lbl3 = Label(window, text="Третий путь к изображению")
lbl4 = Label(window, text="Коэффициент s")
lbl5 = Label(window, text="Количество итераций")

lbl1.grid(column=0, row=0)
lbl2.grid(column=0, row=1)
lbl3.grid(column=0, row=2)
lbl4.grid(column=0, row=3)
lbl5.grid(column=0, row=4)

# поля для ввода текста
firstVarTxt = Entry(window)
secondVarTxt = Entry(window)
ThirdVarTxt = Entry(window)
CoefTxt = Entry(window)
CoefTxt.insert(-1, '0.02')
NIterTxt = Entry(window)
NIterTxt.insert(-1, '100')

firstVarTxt.grid(column=1, row=0)
secondVarTxt.grid(column=1, row=1)
ThirdVarTxt.grid(column=1, row=2)
CoefTxt.grid(column=1, row=3)
NIterTxt.grid(column=1, row=4)

# кнопки
btnStart = Button(window, text="Start calc", command=run)
btnAddImage = Button(window, text="Add image line", command=lambda: addImagePath(window))

btnStart.grid(column=2, row=0)

btnAddImage.grid(column=2, row=2)

window.mainloop()
