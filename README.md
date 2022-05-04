# credit-card-clustering
Credit Card Dataset for Clustering from [Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)

## About Dataset

This case requires to develop a customer segmentation to define marketing strategy. The sample Dataset summarizes the usage behavior of about 9000 active credit card holders during the last 6 months. The file is at a customer level with 18 behavioral variables.  
  
Following is the Data Dictionary for Credit Card dataset :-  

CUSTID : Identification of Credit Card holder (Categorical)  
BALANCE : Balance amount left in their account to make purchases  
BALANCEFREQUENCY : How frequently the Balance is updated, score between 0 and 1 (1 = frequently updated, 0 = not frequently updated)  
PURCHASES : Amount of purchases made from account  
ONEOFFPURCHASES : Maximum purchase amount done in one-go  
INSTALLMENTSPURCHASES : Amount of purchase done in installment  
CASHADVANCE : Cash in advance given by the user  
PURCHASESFREQUENCY : How frequently the Purchases are being made, score between 0 and 1 (1 = frequently purchased, 0 = not frequently purchased)  
ONEOFFPURCHASESFREQUENCY : How frequently Purchases are happening in one-go (1 = frequently purchased, 0 = not frequently purchased)  
PURCHASESINSTALLMENTSFREQUENCY : How frequently purchases in installments are being done (1 = frequently done, 0 = not frequently done)  
CASHADVANCEFREQUENCY : How frequently the cash in advance being paid  
CASHADVANCETRX : Number of Transactions made with "Cash in Advanced"  
PURCHASESTRX : Numbe of purchase transactions made  
CREDITLIMIT : Limit of Credit Card for user  
PAYMENTS : Amount of Payment done by user  
MINIMUM_PAYMENTS : Minimum amount of payments made by user  
PRCFULLPAYMENT : Percent of full payment paid by user  
TENURE : Tenure of credit card service for user  

## Repository Structure and Files

- [data](data): contains all the data files
- [notebooks](notebooks): contains jupyter notebooks used for exploration, explanation and visualisation
- [source files](src): source scripts for modules
- [main pipeline](main.py): run this script to execute the entire clustering pipeline. See below on usage.
- [dependencies](requirements.txt): install the dependencies into your environment to replicate it. See below on installation.


## Installation and Usage

Set up your own environment  
```
python3 -m venv .venv

source .venv/bin/activate # activate environment

deactivate # deactivate environment
```

Install all the necessary dependencies  
```
pip install -r requirements.txt
```

Execute the files
```
# execute entire pipeline with default model
python3 main.py

# execute entire pipeline with using DBSCAN
python3 main.py --model=dbscan
```

If you need to execute specific scripts in the `/src` directory, the commands will be somewhat verbose but this is due to the usage of relative imports in each of the source scripts. You will have to run it using `-m`, refer below for an example. Read [thread](https://stackoverflow.com/questions/16981921/relative-imports-in-python-3) for more details. 
```
python3 -m src.preprocessing
```

For more help options, simply run `--help` option
```
python3 main.py --help
Usage: main.py [OPTIONS]

Options:
  --model TEXT  Clustering Model to be used. Options: <kmeans> or <dbscan>.
  --help        Show this message and exit.
```
