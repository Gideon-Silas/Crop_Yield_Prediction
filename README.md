# Crop Yield Prediction
Predicting crop yield from climate and agricultural inputs with Machine Learning 

## About
Most small-scale farmers are yet to embrace modern technology that could help them predict their yield before harvesting. This study employs historical climate and agricultural data to predict crop yield in hg/ha using a machine learning model and determine the highest predictive variables for yield in different world regions.
A random forest regressor trained on the dataset achieved a performance score of R² = 0.986 in predicting the yield.

## Dataset
Crop Yield Prediction Dataset from Kaggle.
The data spans 28,242 records from 1990 to 2013 for 10 different kinds of crops and 101 countries.

## Key Findings
- Crop type is the most important determinant of yield in the dataset (61%), with tubers and roots producing 5-10x more produce per hectare than cereals, legumes, and other crops.
- The impact of climate on yield appears to be masked when considering all crops together, but becomes apparent when disaggregated. For instance, temperature demonstrates a strong negative association with most crop groups (Maize: -0.55, Wheat: -0.37).
- Finally, while rainfall and pesticide application have some impact (35%), they are not nearly as important as crop type or temperature.

## Project structure
crop-yield-prediction/
│
├── data/
│   └── yield_df.csv
│
├── notebooks/
│   └── 3MTT_Capstone_Project.ipynb
│
├── report/
│   └── Crop_Yield_Capstone_Report.md
│
├── images/
│   └── (chart images used in the report)
│
├── requirements.txt
└── README.md

## How to run
git clone https://github.com/Gideon-Silas/crop-yield-prediction.git
cd Crop-Yield-Prediction
pip install -r requirements.txt
jupyter notebook notebooks/3MTT_Capstone_Project.ipynb

## Tech stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LabelEncoder, StandardScaler, RandomForestRegressor)

## Full write-up
The complete analysis, with charts and explanations, is in report/Crop_Yield_Capstone_Report.md.

## Author
Connect with me on [LinkedIn](https://linkedin.com/in/gideonsilas), [Twitter](https://twitter.com/Mr_Goodboi1), and [Medium](https://medium.com/@Just_Gideons).
