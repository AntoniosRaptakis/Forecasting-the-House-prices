# Predicting the price of houses

![house_dream](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/eb00c8c1-209a-45df-bfc5-af6e8a4fe157)



Buying a house is one of the biggest dreams of any family, or even individuals. It is a "long-time" project and and life-time investment, because one pays the loan to the financial institution for several years. People consider that they are paying the rent for the house, but after few years this house belongs to them. 

Let's see some of the reasons why one considers buying his own home. 

- **Pride of ownership** is probably the number one reason that people enjoy owning their own homes. It means that one can, for example, paint the walls any color he desires, turn the music up, attach permanent fixtures, and decorate the home according to his taste.

- The **mortgage interest deduction** can overshadow the desire for the pride of ownership as well. As long as the mortgage balance is smaller than the price of the home, mortgage interest is fully deductible on the tax return. For a large portion of the time one pays down your mortgage, interest is the largest component of your mortgage payment.

- In general, one can **deduct state and local real estate taxes**. Most homeowners pay their property taxes as part of their monthly mortgage payments.

- **Mortgage Reduction Builds Equity**: Each month, part of the monthly payment is applied to the loan's principal balance, which reduces the obligation. The longer one lives in the home, the more equity he is are building with each payment.

Source: https://www.thebalancemoney.com/eight-reasons-to-buy-a-home-1798233

This project aims to predict the price of a house. The dataset has been taken by kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The region that the data have been collected is Ames at Iowa in the United States of America.

**Goal of the project**: Create a predictive model for the house price.

The libraries that I used:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- XGBoost
- Plotly

<ins>**The dataset**<ins>:
	
I start with reading the dataset using pandas. There are 79 columns and 1460 rows. Let's see their meaning:
	
  - **MSSubClass**: Identifies the type of dwelling involved in the sale.	
  - **MSZoning**: Identifies the general zoning classification of the sale.
  - **LotFrontage**: Linear feet of street connected to property
  - **LotArea**: Lot size in square feet
  - **Street**: Type of road access to property
  - **Alley** : Type of alley access to property
  - **LotShape**: General shape of property
  - **LandContour**: Flatness of the property
  - **Utilities**: Type of utilities available
  - **LotConfig**: Lot configuration
  - **LandSlope**: Slope of property
  - **Neighborhood**: Physical locations within Ames city limits
  - **Condition1**: Proximity to various conditions
  - **Condition2**: Proximity to various conditions (if more than one is present)
  - **BldgType**: Type of dwelling
  - **HouseStyle**: Style of dwelling
  - **OverallQual**: Rates the overall material and finish of the house
  - **OverallCond**: Rates the overall condition of the house
  - **YearBuilt**: Original construction date
  - **YearRemodAdd**: Remodel date (same as construction date if no remodeling or additions)
  - **RoofStyle**: Type of roof
  - **RoofMatl**: Roof material
  - **Exterior1st**: Exterior covering on house
  - **Exterior2nd**: Exterior covering on house (if more than one material)
  - **MasVnrType**: Masonry veneer type
  - **MasVnrArea**: Masonry veneer area in square feet
  - **ExterQual**: Evaluates the quality of the material on the exterior 
  - **ExterCond**: Evaluates the present condition of the material on the exterior
  - **Foundation**: Type of foundation
  - **BsmtQual**: Evaluates the height of the basement
  - **BsmtCond**: Evaluates the general condition of the basement
  - **BsmtExposure**: Refers to walkout or garden level walls
  - **BsmtFinType1**: Rating of basement finished area
  - **BsmtFinSF1**: Type 1 finished square feet
  - **BsmtFinType2**: Rating of basement finished area (if multiple types)
  - **BsmtFinSF2**: Type 2 finished square feet
  - **BsmtUnfSF**: Unfinished square feet of basement area
  - **TotalBsmtSF**: Total square feet of basement area
  - **Heating**: Type of heating
  - **HeatingQC**: Heating quality and condition
  - **CentralAir**: Central air conditioning
  - **Electrical**: Electrical system
  - **1stFlrSF**: First Floor square feet
  - **2ndFlrSF**: Second floor square feet
  - **LowQualFinSF**: Low quality finished square feet (all floors)
  - **GrLivArea**: Above grade (ground) living area square feet
  - **BsmtFullBath**: Basement full bathrooms
  - **BsmtHalfBath**: Basement half bathrooms
  - **FullBath**: Full bathrooms above grade
  - **HalfBath**: Half baths above grade
  - **Bedroom**: Bedrooms above grade (does NOT include basement bedrooms)
  - **Kitchen**: Kitchens above grade
  - **KitchenQual**: Kitchen quality
  - **TotRmsAbvGrd**: Total rooms above grade (does not include bathrooms)
  - **Functional**: Home functionality (Assume typical unless deductions are warranted)
  - **Fireplaces**: Number of fireplaces
  - **FireplaceQu**: Fireplace quality
  - **GarageType**: Garage location
  - **GarageYrBlt**: Year garage was built
  - **GarageFinish**: Interior finish of the garage
  - **GarageCars**: Size of garage in car capacity
  - **GarageArea**: Size of garage in square feet
  - **GarageQual**: Garage quality
  - **GarageCond**: Garage condition
  - **PavedDrive**: Paved driveway
  - **WoodDeckSF**: Wood deck area in square feet
  - **OpenPorchSF**: Open porch area in square feet
  - **EnclosedPorch**: Enclosed porch area in square feet
  - **3SsnPorch**: Three season porch area in square feet
  - **ScreenPorch**: Screen porch area in square feet
  - **PoolArea**: Pool area in square feet
  - **PoolQC**: Pool quality
  - **Fence**: Fence quality
  - **MiscFeature**: Miscellaneous feature not covered in other categories
  - **MiscVal**: Value of miscellaneous feature
  - **MoSold**: Month Sold (MM)
  - **YrSold**: Year Sold (YYYY)
  - **SaleType**: Type of sale
  - **SaleCondition**: Condition of sale
  - **SalePrice**: Price of sale

## <ins>**Data Cleaning**<ins>:
	
Part of the data cleaning is to find out missing values. If we assume the dataframe as a matrix, where the columns are on the horizontal direction and the rows on the vertical, the missing values can be clearly visualized with the yellow lines, as shown in the figure below.

<img src="https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/01d2ac55-61c1-4006-a120-74b23b855e6c" width="4000" height="400">
	
	
How can I tackle this problem? 

Firstly, I am searching which columns have data missing and if the percentage of missing values is equal to or greater than a threshold of 40%, then this column has to be deleted. After encoding the binary and categorical variables, I use the follow:
	
	missing_values_threshold = data.shape[0]*0.4
	columns_for_deletion, variables_with_missing_values = [], []
	for x in data.columns:
	    if data[x].isna().sum() >= missing_values_threshold:
	        columns_for_deletion.append(x)
	    else:
	        variables_with_missing_values.append(x)
	data = data.drop(columns_for_deletion, axis=1)

Which columns will be deleted?
- Alley
- FireplaceQu
- PoolQC
- Fence
- MiscFeature



The rest, which have simply some missing values, have been stored in a list. What will I do with these cases? I will follow two methods:
	
**Method 1**: I will drop all rows with nan values 
	
	# copy the data
	data_dropped_na = data.copy()

	# drop the missing values
	data_dropped_na = data_dropped_na.dropna()

	# reset index by dropping the index column instead of inserting it back into the DataFrame
	data_dropped_na = data_dropped_na.reset_index(drop=True)
	
	
**Method 2**: I use the KNNImputer from the impute class of the sklearn library in order to fill in the missing values. Apparently, it uses the KNN algorithm in order to fill in the missing values. 
	
	from sklearn.impute import KNNImputer
	
	imputer = KNNImputer(n_neighbors=8)
	data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
	

Other parts of data cleaning include the check of the data types, duplicated rows and outliers. The data types are as expected and there are no duplicated rows. For the outliers, it will be discussed later.


## <ins>**Correlations**<ins>:
	
Before I start building up the predictive model, I would like to see the correlations between the given variables and the target one. There are different methods to check the correlations between a continuous variable (target) and binaries or categorical or continuous variable. One method to check the correlation between a binary or categorical variable with a continuous one is the Point-Biserial The data must meet some requirements: 

- they must follow the normal distribution,
- there must be no outliers, and 
- the variances between the groups must be equal.

Point-Biserial is a special case of the pearson correlation. The pearson correlation is supported by pandas and it is the default method for the correlation. The command is data.corr(method="...") or data.corrwith(target column, method="...").
	
Another method is the ANOVA (ANalysis Of VAriance). If we have one factor (one categorical variable), then we use the one-way ANOVA. If we use two or more factors, then we use multifactorial ANOVA. The requirements of ANOVA for the continuous variable are the same as for the Point-Biserial. ANOVA is a statistical test and we see if the groups are statistically significant based on the threshold that we put (alpha value).

We must not ignore the corellation between continuous variables, which can be checked using the linear correlation.
	
Here, I will use the pearson method to check the correlation between the independent variables of the predictive model that I will build and the target (or dependent) variable. In the jupyter-notebook I have applied other tests as well.

Before I check the correlation as described above, I will drop some more columns that I do not need. 
	
	columns_to_be_removed = ['YearRemodAdd','MoSold','SaleType','GarageYrBlt','GarageFinish','GarageCars',
                    	         'PavedDrive','Foundation']

	data_dropped_na = data_dropped_na.drop(columns_to_be_removed, axis=1)
	data_imputer = data_imputer.drop(columns_to_be_removed, axis=1)
	

![correlation_with_SalePrices_pearsonr](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/ec02147f-9fe8-4edf-99c7-22a32d780e2a)

The figure above shows the correlation between each variable with the target variable using the pearson method for the two datasets that I have. There are some variables that they have a very low correlation. Thus, I will apply on the datasets (data_dropped_na and data_imputer) a threshold of 10% (corr_coef>=0.1) and reduce the independent variables.
	
![methods_for_data_modelling](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/b7ddcb9e-b8d1-46d2-8b8c-87e93d392f8b)

For the predictive model, I will check the four of those cases (two different methods to deal with the missing values and for each of them I apply the threshold as result of the pearson correlation.)

## <ins>**Predictive model**<ins>:
	
One part of the model is to find out the independent variables. Another part is the machine learning algorithm that fits the best on the data. Thus, I will compare the result on the datasets using different algorithms. Specifically:

- from the linear_model: Ridge, Lasso & ElasticNet
- KNN
- ensemble methods: RandomForestRegressor, GradientBoostingRegressor & XGBRegressor

From XGBRegressor I will use the different booster methods, three and gblinear.

I will tune the hyperparameters in the training dataset (70% of the whole dataset) and then cross-validate the result using cross_val_score using as scoring method the same metrics as GridSearchCV, the $R^2$.

<img width="795" alt="results_first_model" src="https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/8d6fc030-5084-4f22-b417-77af512bb3bd">

## <ins>**Pipeline of the predictive model**<ins>:

![pipeline_with_regressors](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/92ff7681-6256-44d0-a081-eb3d3520d308)
	
	
![both_regressors_actual_predicted_values](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/fb7ebfc8-f91b-4fe4-b85b-427291a893c9)
	
![predicted_actual_values](https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/71ab000b-ccec-4671-aaf8-573b32ae4018)

