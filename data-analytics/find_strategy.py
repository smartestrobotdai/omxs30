#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from importlib import reload
import tradestrategy
StrategyModel = tradestrategy.StrategyModel

import investmentmodel
reload(investmentmodel)
InvestmentModel = investmentmodel.InvestmentModel

import datamanipulator
reload(datamanipulator)

import statefullstmmodel
reload(statefullstmmodel)



# In[14]:


data = np.load("./npy_files/ema20_beta99_5.npy", allow_pickle=True)


# In[15]:


input_data = data[:60,6:-5,[-2,-3,-1]]


# In[ ]:


strategy_model_list = []
for i in range(5):
    strategy_model = StrategyModel(n_max_trades_per_day=4, slippage=0.00015, courtage=0, max_iter=300)
    strategy_model.optimize(input_data)
    strategy_model_list.append(strategy_model)


# In[ ]:


investment_model = InvestmentModel('Nordel', 5)
investment_model.optimize(strategy_model_list, is_test=False)


# In[ ]:




