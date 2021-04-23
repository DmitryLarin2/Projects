# import numpy as np
# import random
# number = np.random.randint(1,101)


# def game_core_v2_1(number):
#     ### Первый алгорит подобен образцу, просто применен более крупный шаг, 
#     ### что позволяет раньше прийти к искомому числу
    
#     count = 1
#     predict = np.random.randint(1,101)
    
#     while number!=predict:
#         count+=1
#         #predict
#         if number > predict:
#             predict = predict + 7  
#         elif number < predict:
#             predict = predict - 4  
#     return(count)

# def game_core_v2_2(number):
#     ### Второй алгоритм основан на сравнении загаданного числа с нашей попыткой угадать.
#     ### В дальнейшем идет переопределение границ для угадывания
    
#     x = 1
#     y = 100
#     predict = random.randint (1,101) 
#     count = 1
    
#     while number != predict: 
#             if number > predict:
#                 x = predict
#                 predict = random.randint(x, y)
#                 count += 1
#             elif number < predict:
#                 y = predict
#                 predict = random.randint(x, y)
#                 count += 1
#     return(count)           
    

# def score_game(game_core):
#     '''Запускаем игру 1000 раз, чтобы узнать, как быстро игра угадывает число'''
#     count_ls = []
#     np.random.seed(1)  # фиксируем RANDOM SEED, чтобы ваш эксперимент был воспроизводим!
#     random_array = np.random.randint(1,101, size=(1000))
#     for number in random_array:
#         count_ls.append(game_core(number))
#     score = int(np.mean(count_ls))
#     print(f"Ваш алгоритм угадывает число в среднем за {score} попыток")
#     return(score)

# score_game(game_core_v2_1) #запускаем игру по первому алгоритму
# score_game(game_core_v2_2) #запускаем игру по второму алгоритму

#print("hello_world")
list1=['a','d','f']
len(list1)
