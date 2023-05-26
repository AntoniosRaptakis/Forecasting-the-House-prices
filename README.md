# Predicting the price of houses


Buying a house is one of the biggest dreams of any family, or even individuals. It is a "long-time" project and investment, because one pays the loan to the financial institution for several years. People consider that they are paying the rent for the house, but after few years this house belongs to them.
This project aims to predict the price of a house. The dataset has been taken by kaggle: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

The region that the data have been collected is Ames at Iowa in the United States of America.

**Goal of the project**: Create a predictive model for the house price.

The libraries that I used:
- Numpy
- Pandas
- Seaborn
- Matplotlib
- Scikit-Learn
- Plotly

<ins>**The dataset**<ins>:
	
I start with reading the dataset using pandas. There 79 columns. Let's see their meaning:
	
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


I realized that there are missing values. If we assume the dataframe as a matrix, where the columns are on the horizontal direction and the rows on the vertical, the missing values can be clearly visualized with the yellow lines, as shown in the figure below.

<img src="https://github.com/AntoniosRaptakis/Forecasting-the-House-prices/assets/86191637/de1d2ff3-32e1-4735-857d-c68d960e6d83" width="3500" height="400">
