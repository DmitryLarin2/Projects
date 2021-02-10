#!/usr/bin/env python
# coding: utf-8

# ![](https://www.pata.org/wp-content/uploads/2014/09/TripAdvisor_Logo-300x119.png)
# # Predict TripAdvisor Rating
# ## В этом соревновании нам предстоит предсказать рейтинг ресторана в TripAdvisor
# **По ходу задачи:**
# * Прокачаем работу с pandas
# * Научимся работать с Kaggle Notebooks
# * Поймем как делать предобработку различных данных
# * Научимся работать с пропущенными данными (Nan)
# * Познакомимся с различными видами кодирования признаков
# * Немного попробуем [Feature Engineering](https://ru.wikipedia.org/wiki/Конструирование_признаков) (генерировать новые признаки)
# * И совсем немного затронем ML
# * И многое другое...   
# 
# 
# 
# ### И самое важное, все это вы сможете сделать самостоятельно!
# 
# *Этот Ноутбук являетсся Примером/Шаблоном к этому соревнованию (Baseline) и не служит готовым решением!*   
# Вы можете использовать его как основу для построения своего решения.
# 
# > что такое baseline решение, зачем оно нужно и почему предоставлять baseline к соревнованию стало важным стандартом на kaggle и других площадках.   
# **baseline** создается больше как шаблон, где можно посмотреть как происходит обращение с входящими данными и что нужно получить на выходе. При этом МЛ начинка может быть достаточно простой, просто для примера. Это помогает быстрее приступить к самому МЛ, а не тратить ценное время на чисто инженерные задачи. 
# Также baseline являеться хорошей опорной точкой по метрике. Если твое решение хуже baseline - ты явно делаешь что-то не то и стоит попробовать другой путь) 
# 
# В контексте нашего соревнования baseline идет с небольшими примерами того, что можно делать с данными, и с инструкцией, что делать дальше, чтобы улучшить результат.  Вообще готовым решением это сложно назвать, так как используются всего 2 самых простых признака (а остальные исключаются).

# # import

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')

from collections import Counter
import re
from datetime import datetime, timedelta





# Загружаем специальный удобный инструмент для разделения датасета:
from sklearn.model_selection import train_test_split

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[2]:


# всегда фиксируйте RANDOM_SEED, чтобы ваши эксперименты были воспроизводимы!
RANDOM_SEED = 42


# In[3]:


# зафиксируем версию пакетов, чтобы эксперименты были воспроизводимы:
get_ipython().system('pip freeze > requirements.txt')


# In[4]:


def find_IQR(column):
    perc_25 = data[column].quantile(0.25, interpolation="midpoint")
    perc_75 = data[column].quantile(0.75, interpolation="midpoint")
    IQR = perc_75 - perc_25
    print('Q1: {}'.format(perc_25), 'Q3: {}'.format(perc_75), 'IQR: {}'.format(IQR),
          'Граница выбросов: [{a},{b}]'.format(a=perc_25 - 1.5*IQR, b=perc_75 + 1.5*IQR), sep='\n')


# # DATA

# In[5]:


DATA_DIR = '/kaggle/input/sf-dst-restaurant-rating/'
df_train = pd.read_csv(DATA_DIR+'/main_task.csv')
df_test = pd.read_csv(DATA_DIR+'kaggle_task.csv')
sample_submission = pd.read_csv(DATA_DIR+'/sample_submission.csv')
df_cit = pd.read_csv('/kaggle/input/world-cities/worldcities.csv')
df_cost=pd.read_csv('/kaggle/input/2020-cost-of-living/cost of living 2020.csv')


# ### Посмотрим на все файлы, чтобы представлять, с чем мы будем оперировать

# In[6]:


df_train.head()


# In[7]:


df_train.info()


# In[8]:


df_train.isna().sum()


# In[9]:


df_train.columns


# In[10]:


df_cit.head()


# In[11]:


df_cit.info()


# In[12]:


df_cit.isna().sum()


# In[13]:


df_cost.head()


# In[14]:


df_cost.isna().sum()


# In[15]:


# ВАЖНО! дря корректной обработки признаков объединяем трейн и тест в один датасет
df_train['sample'] = 1 # помечаем где у нас трейн
df_test['sample'] = 0 # помечаем где у нас тест
df_test['Rating'] = 0 # в тесте у нас нет значения Rating, мы его должны предсказать, по этому пока просто заполняем нулями

data = df_test.append(df_train, sort=False).reset_index(drop=True) # объединяем


# In[16]:


data.info()


# ## Restaurant_id

# In[17]:


data['Restaurant_id'].value_counts()


# Тут все хорошо

# ## City

# In[18]:


data.City.unique()


# In[19]:


data.City.nunique()


# In[20]:


cityes=set()
data['City'].apply(lambda x: cityes.add(x))
cityes
len(cityes)


# In[21]:


df_cit.columns


# In[22]:


# Уберем лишние колонки, а так же заметим, что европейские города как-то попали в столбец country в United States, 
#это надо будет исправить
df_cit = df_cit.drop(['city_ascii', 'lat', 'lng', 'iso2',
                      'iso3', 'admin_name', 'capital', 'id'], axis='columns')
df_cit = df_cit.loc[df_cit.city.isin(cityes)]
df_cit


# In[23]:


df_cit=df_cit.loc[df_cit['country']!='United States']
new_city=set()
df_cit.city.apply(lambda x: new_city.add(x))

#Все ли города совпадают? Проверим
cityes-new_city 


# In[24]:


#Указанных выше городов нет, добавим их вручную
df_cit.loc[2586]=['Krakow', 'Poland',779115]
df_cit.loc[2587]=['Oporto', 'Portugal',240000]
df_cit.loc[2588]=['Zurich', 'Germany',1300000]
df_cit.tail(5)


# In[25]:


df_cost.head()


# In[26]:


#Уберем часть лишних колонок, оставим индекс цен в ресторанах, индекс бигмака, Индекс местной покупательной способности
df_cost = df_cost.drop(['Rank 2020', 'Cost of Living Index', 'Cost of Living Plus Rent Index', 'Groceries Index',
                        'Unnamed: 9', 'Rent Index'], axis='columns')
df_cost = df_cost.loc[df_cost.Country.isin(df_cit['country'])]
df_cost.sample(5)


# In[27]:


#Объединим два небольших датафрейма между собой
df_cit=df_cit.merge(df_cost, left_on='country',right_on= 'Country', how='inner')
df_cit=df_cit.drop(['country','Country'], axis='columns')
df_cit.head(5)


# In[28]:


data.info()


# In[29]:


df_cit.info()


# In[30]:


df_cit.head()


# In[31]:


# Объединим теперь с большим фреймом
#data=data.merge(df_cit, left_on='City',right_on= 'city', how='inner')
#data.head(5)


# In[32]:


sum(data.City.value_counts())


# In[33]:


data.info()


# ## Cuisine Style

# In[34]:


data['Cuisine Style']=data['Cuisine Style'].fillna("['no_info']") #заменяем пропуск 
def make_a_list (x):
    x=x[1:-1]
    x=x.replace("'", "")
    x=x.split(', ')
    return (x)
cousine_list=[]
data['Cuisine Style']=data['Cuisine Style'].apply(make_a_list)
data['Cuisine Style'].apply(lambda x: cousine_list.extend(x))
Counter(cousine_list).most_common(3)


# In[35]:


def change_cuisine(x):
    if x == ['no_info']:
        return ['European', 'Vegetarian Friendly']
    else:
        return x
data['Cuisine Style']=data['Cuisine Style'].apply(change_cuisine)
cousine_list=[]
data['Cuisine Style'].apply(lambda x: cousine_list.extend(x))
Counter(cousine_list).most_common(5)


# In[36]:


data['Cuisine Style']=data['Cuisine Style'].apply(lambda x: len(x))
data.head()


# ## Price Range

# In[37]:


data['Price Range'].value_counts()


# In[38]:


#ata['Price Range']=data['Price Range'].fillna('$$ - $$$')


# In[39]:


data['Price Range'].isna().sum()


# In[40]:


price = {'$' : 1, '$$ - $$$' : 2, '$$$$' : 3}
data['Price Range'] = data['Price Range'].map(price)


# In[41]:


data['price_range_is_NAN'] = pd.isna(data['Price Range']).astype('uint8')


# In[42]:


data['Price Range'] = data['Price Range'].fillna(0)


# In[43]:


data['Price Range'].value_counts()


# In[44]:


data.head()


# ## Number of Reviews

# In[45]:


data['Number of Reviews'].value_counts()


# In[46]:


data['Number of Reviews'].hist(bins=100)
data['Number of Reviews'].describe()


# In[47]:


mode=data['Number of Reviews'].mode()
mean=data['Number of Reviews'].mean()
median=data['Number of Reviews'].median()


# In[48]:


print("Mode: {}, \n\nMean: {},\nMedian: {}".format(mode,mean,median))


# In[49]:


median = data['Number of Reviews'].median()
IQR = data['Number of Reviews'].quantile(0.75) - data['Number of Reviews'].quantile(0.25)
perc25 = data['Number of Reviews'].quantile(0.25)
perc75 = data['Number of Reviews'].quantile(0.75)
print('25-й перцентиль: {},'.format(perc25), '75-й перцентиль: {},'.format(perc75)
      , "IQR: {}, ".format(IQR),"Границы выбросов: [{f}, {l}].".format(f=perc25 - 1.5*IQR, l=perc75 + 1.5*IQR))
data['Number of Reviews'].loc[data['Number of Reviews'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)].hist(bins = 50, range = (0, 40), 
                                                                                             label = 'IQR')
plt.legend()


# In[50]:


data['Number of Reviews'] = data['Number of Reviews'][data['Number of Reviews'].between(perc25 - 1.5*IQR, perc75 + 1.5*IQR)]


# In[51]:


mean_val = round(data['Number of Reviews'].mean(), 0)
data['Number of Reviews'] = data['Number of Reviews'].fillna(mean_val)


# In[52]:


data['Number of Reviews'].hist()
data['Number of Reviews'].describe()


# ## Reviews

# In[53]:


for i, cell in enumerate(data['Reviews']):
    print(cell)
    if i == 20:
        break


# In[54]:


data.Reviews.isnull().sum()


# In[55]:


data['review_is_NAN'] = (data.Reviews == '[[], []]').astype('uint8')


# In[56]:


data['review_is_NAN'] = pd.isna(data.Reviews).astype('uint8')


# In[57]:


data['date_of_review'] = data['Reviews'].apply(
    lambda x: [0] if pd.isna(x) else x[2:-2].split('], [')[1][1:-1].split("', '"))


# In[58]:


data['days_between_reviews'] = data.date_of_review.apply(lambda x: None if x == [] else pd.to_datetime(x).max() - pd.to_datetime(x).min())


# In[59]:


data['days_between_reviews'] = data['days_between_reviews'].apply(lambda x: x.days)


# In[60]:


find_IQR('days_between_reviews')


# In[61]:


data.days_between_reviews.hist(bins=100)


# In[62]:


len(data.query('days_between_reviews > 1200'))


# In[63]:


data.days_between_reviews = data.days_between_reviews.apply(lambda x: 1200 if x > 1200 else x)


# In[64]:


data.days_between_reviews.isnull().sum()


# In[65]:


data.days_between_reviews.describe()


# In[66]:


data.days_between_reviews = data.days_between_reviews.fillna(data.days_between_reviews.median())


# In[67]:


data.days_between_reviews.isnull().sum()


# In[68]:


data['days_to_now'] = data.date_of_review.apply(
    lambda x: None if x == [] else datetime.now() - pd.to_datetime(x).max())


# In[69]:


data['days_to_now'] = data['days_to_now'].apply(lambda x: x.days)


# In[70]:


data.days_to_now.isnull().sum()


# In[71]:


data.days_to_now.describe()


# In[72]:


data.days_to_now = data.days_to_now.fillna(data.days_to_now.mean())


# In[73]:


data.days_to_now.isnull().sum()


# In[74]:


data.head(3)


# ## Добавим новых признаков

# In[75]:


Jan_temp={'Paris':4.8, 'Stockholm':-1.4, 'London':5.0, 'Berlin':1.3, 'Munich':0, 'Oporto':11.4,
       'Milan':4.8, 'Bratislava':0.4, 'Vienna':0.7, 'Rome':8.8, 'Barcelona':10.7, 'Madrid':7.6,
       'Dublin':6, 'Brussels':4.1, 'Zurich':0.9, 'Warsaw':-1.1, 'Budapest':0.6, 'Copenhagen':2.4,
       'Amsterdam':5, 'Lyon':3.4, 'Hamburg':2, 'Lisbon':12.7, 'Prague':0, 'Oslo':-4.5,
       'Helsinki':-2.8, 'Edinburgh':4.1, 'Geneva':0.1, 'Ljubljana':0.6, 'Athens':10.1,
       'Luxembourg':1.7, 'Krakow':-1.2}
Apr_temp={'Paris':11.7, 'Stockholm':5.9, 'London':10.8, 'Berlin':10.8, 'Munich':9.6, 'Oporto':14.9,
       'Milan':13.7, 'Bratislava':11.9, 'Vienna':11.9, 'Rome':15.1, 'Barcelona':15.3, 'Madrid':14,
       'Dublin':8.5, 'Brussels':11.1, 'Zurich':10, 'Warsaw':10.3, 'Budapest':13.5, 'Copenhagen':7.8,
       'Amsterdam':10.1, 'Lyon':11.5, 'Hamburg':9.2, 'Lisbon':16.3, 'Prague':10.2, 'Oslo':5,
       'Helsinki':3.9, 'Edinburgh':7.8, 'Geneva':11.7, 'Ljubljana':10.8, 'Athens':16.4,
       'Luxembourg':9.8, 'Krakow':10.1}
Jul_temp={'Paris':20.1, 'Stockholm':17.7, 'London':19.1, 'Berlin':20.5, 'Munich':18.4, 'Oporto':21.6,
       'Milan':24.5, 'Bratislava':22.1, 'Vienna':21.8, 'Rome':26.6, 'Barcelona':26, 'Madrid':28.1,
       'Dublin':15.5, 'Brussels':19.1, 'Zurich':19, 'Warsaw':20.9, 'Budapest':23.6, 'Copenhagen':18.1,
       'Amsterdam':18.3, 'Lyon':20.7, 'Hamburg':18, 'Lisbon':22.4, 'Prague':19.7, 'Oslo':16.3,
       'Helsinki':17.7, 'Edinburgh':14.7, 'Geneva':18, 'Ljubljana':20.5, 'Athens':28.2,
       'Luxembourg':17.9, 'Krakow':19.6}
Oct_temp={'Paris':13.5, 'Stockholm':8.1, 'London':13.1, 'Berlin':11.6, 'Munich':9.8, 'Oporto':18.8,
       'Milan':16.1, 'Bratislava':12, 'Vienna':12, 'Rome':19, 'Barcelona':19.7, 'Madrid':18.3,
       'Dublin':11.5, 'Brussels':12.3, 'Zurich':10.5, 'Warsaw':10.8, 'Budapest':13.6, 'Copenhagen':11.3,
       'Amsterdam':12.7, 'Lyon':13, 'Hamburg':10.9, 'Lisbon':20.3, 'Prague':10.7, 'Oslo':6.1,
       'Helsinki':7.5, 'Edinburgh':10.1, 'Geneva':10.3, 'Ljubljana':10.9, 'Athens':20.7,
       'Luxembourg':10.5, 'Krakow':10.5}
Mean_sal={'Paris':2900, 'Stockholm':4329, 'London':2507, 'Berlin':2596, 'Munich':2500, 'Oporto':1200,
       'Milan':1916, 'Bratislava':1176, 'Vienna':3406, 'Rome':1847, 'Barcelona':2000, 'Madrid':2000,
       'Dublin':784, 'Brussels':3200, 'Zurich':7839, 'Warsaw':887, 'Budapest':682, 'Copenhagen':3100,
       'Amsterdam':2152, 'Lyon':1691, 'Hamburg':2500, 'Lisbon':852, 'Prague':1275, 'Oslo':4400,
       'Helsinki':2600, 'Edinburgh':3698, 'Geneva':7600, 'Ljubljana': 1172, 'Athens':2700,
       'Luxembourg':3300, 'Krakow':757}
Min_sal={'Paris':1219, 'Stockholm':1101, 'London':1237, 'Berlin':646, 'Munich':646, 'Oporto':649,
       'Milan':800, 'Bratislava':335, 'Vienna':1500, 'Rome':800, 'Barcelona':600, 'Madrid':580,
       'Dublin':480, 'Brussels':1594, 'Zurich':1762, 'Warsaw':523, 'Budapest':582, 'Copenhagen':2000,
       'Amsterdam':1653, 'Lyon':1200, 'Hamburg':1496, 'Lisbon':600, 'Prague':562, 'Oslo':2200,
       'Helsinki':1900, 'Edinburgh':432, 'Geneva':1660, 'Ljubljana':0.6, 'Athens':758,
       'Luxembourg':2300, 'Krakow':509}
tourists={'Paris':19.0, 'Stockholm':2.7, 'London':19.5, 'Berlin':6.2, 'Munich':4.2, 'Oporto':2.8,
       'Milan':6.6, 'Bratislava':1, 'Vienna':6.6, 'Rome':10.3, 'Barcelona':7.0, 'Madrid':5.6,
       'Dublin':5.4, 'Brussels':4.2, 'Zurich':1.5, 'Warsaw':2.8, 'Budapest':4.0, 'Copenhagen':3.2,
       'Amsterdam':8.8, 'Lyon':3.5, 'Hamburg':6.8, 'Lisbon':3.6, 'Prague': 9.1, 'Oslo':0.7,
       'Helsinki':0.4, 'Edinburgh':4.4, 'Geneva':1.3, 'Ljubljana':0.4, 'Athens':0.24,
       'Luxembourg':0.9, 'Krakow':8.1}
rains={'Paris':6.37, 'Stockholm':5.27, 'London':6.21, 'Berlin':5.7, 'Munich':6.22, 'Oporto':11.78,
       'Milan':10.13, 'Bratislava':6.94, 'Vienna':10.31, 'Rome':9.34, 'Barcelona':6.12, 'Madrid':4.5,
       'Dublin':7.67, 'Brussels':7.82, 'Zurich':10.85, 'Warsaw':10.02, 'Budapest':5.64, 'Copenhagen':11.64,
       'Amsterdam':8.05, 'Lyon':7.63, 'Hamburg':7.38, 'Lisbon':6.91, 'Prague': 4.86, 'Oslo':7.40,
       'Helsinki':6.5, 'Edinburgh':7.06, 'Geneva':9.34, 'Ljubljana':12.90, 'Athens':3.97,
       'Luxembourg':8.31, 'Krakow':6.78}

population={'Paris':9904000, 'Stockholm':1264000, 'London':8567000, 'Berlin':3406000, 'Munich':1275000, 'Oporto':240000,
       'Milan':2945000, 'Bratislava':423737, 'Vienna':2400000, 'Rome':3339000, 'Barcelona':4920000, 'Madrid':5567000,
       'Dublin':1059000, 'Brussels':1743000, 'Zurich':1300000, 'Warsaw':1707000, 'Budapest':1679000, 'Copenhagen':1085000,
       'Amsterdam':1031000, 'Lyon':1423000, 'Hamburg':1757000, 'Lisbon':2812000, 'Prague': 1309000 , 'Oslo':835000,
       'Helsinki':1115000, 'Edinburgh':504966, 'Geneva':1240000, 'Ljubljana':314807, 'Athens':3242000,
       'Luxembourg':107260, 'Krakow':779115}
Grocrs_index={'Paris':72.34, 'Stockholm':63.62, 'London':51.15, 'Berlin':50.29, 'Munich':50.29, 'Oporto':38.97,
       'Milan':58.29, 'Bratislava':40.70, 'Vienna':63.63, 'Rome':58.29, 'Barcelona':44.89, 'Madrid':44.89,
       'Dublin':61.63, 'Brussels':61.73, 'Zurich':50.29, 'Warsaw':30.72, 'Budapest':31.60, 'Copenhagen':65.85,
       'Amsterdam':57.81, 'Lyon':72.34, 'Hamburg':50.29, 'Lisbon':38.97, 'Prague': 54.72, 'Oslo':89.55,
       'Helsinki':59.61, 'Edinburgh':51.15, 'Geneva':124.93, 'Ljubljana':46.42, 'Athens':43.67,
       'Luxembourg':74.03, 'Krakow':30.72}
Rest_Price_index={'Paris':69.07, 'Stockholm':71.03, 'London':68.45, 'Berlin':60.53, 'Munich':60.53, 'Oporto':39.26,
       'Milan':68.80, 'Bratislava':34.52, 'Vienna':66.24, 'Rome':68.80, 'Barcelona':51.68, 'Madrid':51.68,
       'Dublin':76.61, 'Brussels':75.62, 'Zurich':60.53, 'Warsaw':32.44, 'Budapest':31.09, 'Copenhagen':95.71,
       'Amsterdam':76.09, 'Lyon':69.07, 'Hamburg':60.53, 'Lisbon':39.26, 'Prague': 46.78, 'Oslo':96.81,
       'Helsinki':77.75, 'Edinburgh':68.45, 'Geneva':118.55, 'Ljubljana':44.15, 'Athens':52.42,
       'Luxembourg':86.29, 'Krakow':32.44}
Loc_purch_power_ind={'Paris':76.21, 'Stockholm':96.04, 'London':86.98, 'Berlin':97.41, 'Munich':97.41, 'Oporto':47.10,
       'Milan':61.80, 'Bratislava':53.39, 'Vienna':79.38, 'Rome':61.80, 'Barcelona':67.73, 'Madrid':67.73,
       'Dublin':74.58, 'Brussels':80.12, 'Zurich':97.41, 'Warsaw':52.30, 'Budapest':46.30, 'Copenhagen':101.27,
       'Amsterdam':86.76, 'Lyon':76.21, 'Hamburg':97.41, 'Lisbon':47.10, 'Prague': 59.61, 'Oslo':83.40,
       'Helsinki':93.94, 'Edinburgh':86.98, 'Geneva':114.83, 'Ljubljana':60.61, 'Athens':40.75,
       'Luxembourg':99.86, 'Krakow':52.30}
McMeal_index={'Paris':10.11, 'Stockholm':9.36, 'London':8.01, 'Berlin':9.52, 'Munich':9.52, 'Oporto':7.14,
       'Milan':9.52, 'Bratislava':7.14, 'Vienna':9.52, 'Rome':9.52, 'Barcelona':8.92, 'Madrid':8.92,
       'Dublin':9.52, 'Brussels':9.70, 'Zurich':9.52, 'Warsaw':5.33, 'Budapest':5.28, 'Copenhagen':12.63,
       'Amsterdam':9.52, 'Lyon':10.11, 'Hamburg':9.52, 'Lisbon':7.14, 'Prague': 7.68, 'Oslo':12.36,
       'Helsinki':9.52, 'Edinburgh':8.01, 'Geneva':15.36, 'Ljubljana':7.02, 'Athens':7.55,
       'Luxembourg':0, 'Krakow':5.33}


# In[76]:


def january_temp_column(C):
    for  city in Jan_temp:
        if city==C:
            return(Jan_temp[city])
data['January_temp']=data['City'].apply(january_temp_column)

def april_temp_column(C):
    for  city in Apr_temp:
        if city==C:
            return(Apr_temp[city])
data['April_temp']=data['City'].apply(april_temp_column)

def july_temp_column(C):
    for  city in Jul_temp:
        if city==C:
            return(Jul_temp[city])
data['July_temp']=data['City'].apply(july_temp_column)

def october_temp_column(C):
    for  city in Oct_temp:
        if city==C:
            return(Oct_temp[city])
data['October_temp']=data['City'].apply(october_temp_column)

def mean_salary_column(C):
    for  city in Mean_sal:
        if city==C:
            return(Mean_sal[city])
data['Mean_salary']=data['City'].apply(mean_salary_column)

def min_sal_column(C):
    for  city in Min_sal:
        if city==C:
            return(Min_sal[city])
data['Min_salary']=data['City'].apply(min_sal_column)

def tourist_flow_column(C):
    for  city in tourists:
        if city==C:
            return(tourists[city])
data['tourists_flow']=data['City'].apply(tourist_flow_column)

def rain_column(C):
    for  city in tourists:
        if city==C:
            return(rains[city])
data['rains']=data['City'].apply(rain_column)

def population_column(C):
    for  city in population:
        if city==C:
            return(population[city])
data['Population']=data['City'].apply(population_column)

def Grocrs_index_column(C):
    for  city in Grocrs_index:
        if city==C:
            return(Grocrs_index[city])
data['Grocrs_index']=data['City'].apply(Grocrs_index_column)

def Rest_Price_index_column(C):
    for  city in Rest_Price_index:
        if city==C:
            return(Rest_Price_index[city])
data['Rest_Price_index']=data['City'].apply(Rest_Price_index_column)

def Loc_purch_power_ind_column(C):
    for  city in Loc_purch_power_ind:
        if city==C:
            return(Loc_purch_power_ind[city])
data['Loc_purch_power_ind']=data['City'].apply(Loc_purch_power_ind_column)

def McMeal_index_column(C):
    for  city in McMeal_index:
        if city==C:
            return(McMeal_index[city])
data['McMeal_index']=data['City'].apply(McMeal_index_column)


# In[77]:


data.head()


# In[78]:


data = pd.get_dummies(data, columns=['City'], dummy_na=True)


# In[79]:


data.info()


# In[80]:


pd.set_option('display.max_rows', 50)  # показывать больше строк
pd.set_option('display.max_columns', 100)  # показывать больше колонок

data.sample(5)


# In[81]:


df_preproc = data.drop(
    ['Restaurant_id', 'Reviews', 'URL_TA','ID_TA','date_of_review'], axis=1)


# In[82]:


df_preproc.info()


# In[83]:


df_preproc1=df_preproc
df_preproc2=df_preproc


# In[84]:


# Теперь выделим тестовую часть
train_data = df_preproc.query('sample == 1').drop(['sample'], axis=1)
test_data = df_preproc.query('sample == 0').drop(['sample'], axis=1)

y = train_data.Rating.values            # наш таргет
X = train_data.drop(['Rating'], axis=1)


# In[85]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEED)


# In[86]:


test_data.shape, train_data.shape, X.shape, X_train.shape, X_test.shape


# ## Model

# In[87]:


from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics
from sklearn.model_selection import train_test_split


# In[88]:


model = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)


# In[89]:


# Обучаем модель на тестовом наборе данных
model.fit(X_train, y_train)

# Используем обученную модель для предсказания рейтинга ресторанов в тестовой выборке.
# Предсказанные значения записываем в переменную y_pred
y_pred = model.predict(X_test)


# In[90]:


y_pred = np.round(y_pred * 2) / 2


# In[91]:


# Сравниваем предсказанные значения (y_pred) с реальными (y_test), и смотрим насколько они в среднем отличаются
# Метрика называется Mean Absolute Error (MAE) и показывает среднее отклонение предсказанных значений от фактических.

print('MAE:', metrics.mean_absolute_error(y_test, y_pred))


# In[92]:


# Посмотрим самые важные признаки для модели

plt.rcParams['figure.figsize'] = (10,10)
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(40).plot(kind='barh')


# In[ ]:





# In[93]:


from sklearn.preprocessing import MinMaxScaler
names1 = df_preproc1.columns.values
scaler1=MinMaxScaler()
df_preproc1 = pd.DataFrame(scaler1.fit_transform(df_preproc1))
df_preproc1.columns=names1

train_data1 = df_preproc1.query('sample == 1').drop(['sample'], axis=1)
test_data1 = df_preproc1.query('sample == 0').drop(['sample'], axis=1)

y1 = train_data1.Rating.values            # наш таргет
X1 = train_data1.drop(['Rating'], axis=1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=RANDOM_SEED)

model1 = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

model1.fit(X1_train, y1_train)
y1_pred = model1.predict(X1_test)

#y1_pred = np.round(y1_pred * 2) / 2

print('MAE:', metrics.mean_absolute_error(y1_test, y1_pred))


# In[94]:


from sklearn.preprocessing import StandardScaler
names2 = df_preproc2.columns.values
scaler2=StandardScaler()
df_preproc2 = pd.DataFrame(scaler2.fit_transform(df_preproc2))
df_preproc2.columns=names2
df_preproc2['sample']=np.round(df_preproc2['sample'],0)

train_data2 = df_preproc2.query('sample == 0.0').drop(['sample'], axis=1)
test_data2 = df_preproc2.query('sample == -2.0 ').drop(['sample'], axis=1)
y2 = train_data2.Rating.values            # наш таргет
X2 = train_data2.drop(['Rating'], axis=1)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.25, random_state=RANDOM_SEED)
model2 = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
model2.fit(X2_train, y2_train)
y2_pred = model2.predict(X2_test)
#y2_pred = np.round(y2_pred * 2) / 2
print('MAE:', metrics.mean_absolute_error(y2_test, y2_pred))


# In[95]:


from sklearn.ensemble import RandomForestRegressor # инструмент для создания и обучения модели
from sklearn import metrics
from sklearn.model_selection import train_test_split
df = pd.read_csv(DATA_DIR+'main_task.csv')
df=df.dropna(axis=1)
df=df.fillna(0)
df.drop(columns=['City','Reviews', 'URL_TA','ID_TA'],  inplace=True)
X0 = df.drop(['Restaurant_id', 'Rating'], axis = 1)
y0 = df['Rating']
X0_train, X0_test, y0_train, y0_test = train_test_split(X0, y0, test_size=0.25)
regr = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)
regr.fit(X0_train, y0_train)
y0_pred = regr.predict(X0_test)
print('MAE:', metrics.mean_absolute_error(y0_test, y0_pred))


# In[96]:


print("МАЕ для сырых данных: {}, \nМАЕ для обработанных данных: {}, \nМАЕ для MinMaxScaler: {}, \nМАЕ для StandardScaler: {}.".format(metrics.mean_absolute_error(y0_test, y0_pred), metrics.mean_absolute_error(
    y_test, y_pred), metrics.mean_absolute_error(y1_test, y1_pred), metrics.mean_absolute_error(y2_test, y2_pred)))


# In[97]:


test_data.sample(5)
test_data3=test_data
test_data3.sample(5)


# In[98]:


df_preproc3=df_preproc


# In[99]:


df_preproc3 = df_preproc3.drop(['City_Amsterdam','City_Athens','City_Barcelona','City_Berlin','City_Bratislava','City_Brussels','City_Budapest','City_Copenhagen','City_Dublin','City_Edinburgh','City_Geneva','City_Hamburg','City_Helsinki','City_Krakow','City_Lisbon','City_Ljubljana','City_London','City_Luxembourg','City_Lyon','City_Madrid','City_Milan','City_Munich','City_Oporto','City_Oslo','City_Paris','City_Prague','City_Rome','City_Stockholm','City_Vienna','City_Warsaw','City_Zurich','City_nan'], axis=1)
train_data3 = df_preproc3.query('sample == 1').drop(['sample'], axis=1)
test_data3 = df_preproc3.query('sample == 0').drop(['sample'], axis=1)

y3 = train_data3.Rating.values            # наш таргет
X3 = train_data3.drop(['Rating'], axis=1)
                                
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.2, random_state=RANDOM_SEED)                                
                                
model3 = RandomForestRegressor(n_estimators=100, verbose=1, n_jobs=-1, random_state=RANDOM_SEED)

model3.fit(X3_train, y3_train)

y3_pred = model3.predict(X3_test)     
                                
print('MAE:', metrics.mean_absolute_error(y3_test, y3_pred))                                


# Что же, играясь с верхним кодом, не удалось улучшить значение МАЕ

# Как итог, обработанные данные нам и будут интересны (МАЕ=0.18...). Построим тепловую карту для наглядности

# fig, ax = plt.subplots(figsize=(19, 19))
# correlation = df_preproc3.corr()
# sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidth=0.7)

# Сделаем финальные действия

# In[101]:


test_data = test_data.drop(['Rating'], axis=1)
sample_submission


# In[102]:


predict_submission = model.predict(test_data)
predict_submission


# In[103]:


sample_submission['Rating'] = predict_submission
sample_submission.to_csv('submission.csv', index=False)
sample_submission.head(10)


#     Какой хочется подвести итог проделанной работы:
# 1) Интересная, творческая, потребовала довольно много времени на поиск как дополнительной информации, так и просто полезных методов для решения той или иной части задания.
# 
# 2) Были проблемы с kaggle, поскольку интерфейс в видео на платформе Skillfactory и на kaggle уже различаются, не все вещи можно повторить 1 в 1.
# 
# 3) Лично у меня по ходу кода были проблемы с merge с другими дата-фреймами. Признаюсь, это подсмотрел у другого человека. Нашлись косяки в дополнительно датафреме с kaggle, однако в одном случае при мерже "рождались" дополнительные 5000+ строк, а при устранении нескольких косяков (например, город London оказался еще и country='Canada') наоборот "умирали" 2000+ строк. Наверное, при самом основательном разборе этих "скользких" мест и удалось бы сделать все аккуратно и правильно, однако выяснение этой проблемы на фоне других уже отняло много времени, поэтому было принято решение данные со сторонни датафреймов перенести вручную (благо, всего 31 город).
# 
# 4) В принципе я доволне проделанной работой, но осталась часть вопросов:
#     - Некоторые строки обрабатываются крайне долго (сет данных большой) - есть ли какие-то способы/методы ускорить?
#     - Если у человека хороший компьютер, как его подсоединить к kaggle, чтобы добавить мощностей? В интеренете есть советы, но не все работает, компы разные, ОС разные. Может быть есть какое-то хитрое решение и не самое сложное по исполнению...
# 5) Я применил к данным MinMaxScaler и StandardScaler - при их использовании МАЕ значительно улучшается, однако и колонка Rating перестает выдавать данные в нужном масштабе. При этих скейлерах Rating от 0 до 1 для MinMaxScaler и часть отрицательных значений для StandardScaler. Наверное, можно сделать обратный скейлинг с сохранением улучшенного МАЕ. Но можно ли это сделать и не будет ли это считаться обходом системы? В учебных материалах пробовал сделать скейлинг скейлинга (MinMax на Standard и наоборот). Вот такие финты ушами могут ли еще улучшить данные? Есьть ли смысл вообще в таких сложных преобразованиях, не поломает ли это в данном соревновании процесс предсказания? Ну и как в таком случае сделать обратный рескейлинг, чтобы получить оценки в нужных диапазонах? К сожалению в учебных материалах это все опущено.
# 

# In[ ]:




