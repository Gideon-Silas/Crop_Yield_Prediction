# Crop Yield Prediction
As we know, most small-scale farmers are yet to embrace modern technology that could help them predict their yield before harvesting. In this project, I employ historical climate and agricultural data to predict crop yield in hg/ha using a machine learning model and determine the highest predictive variables for yield in different world regions.
A random forest regressor trained on the dataset achieved a performance score of R² = 0.986 in predicting the yield.

## Dataset
Crop Yield Prediction Dataset from Kaggle.
The data spans 28,242 records from 1990 to 2013 for 10 different kinds of crops and 101 countries.

## Key Findings
- Crop type is the most important determinant of yield in the dataset (61%), with tubers and roots producing 5-10x more produce per hectare than cereals, legumes, and other crops.
- The impact of climate on yield appears to be masked when considering all crops together, but becomes apparent when disaggregated. For instance, temperature demonstrates a strong negative association with most crop groups (Maize: -0.55, Wheat: -0.37).
- Finally, while rainfall and pesticide application have some impact (35%), they are not nearly as important as crop type or temperature.

## Project structure
```
Crop_Yield_Prediction/
├── notebooks/
│   └── 3MTT_Capstone_Project.ipynb
├── data/
│   └──yield_df.csv)
├── README.md
└── requirements.txt
```

## How to run
1. Clone the repository
2. Navigate into the project folder
3. Install the required dependencies
4. Download the dataset in the "data" folder.
5. Open the notebook
6. Run the cells in order to reproduce the analysis and model training.

## Tech stack
- Python
- pandas, numpy
- matplotlib, seaborn
- scikit-learn (LabelEncoder, StandardScaler, RandomForestRegressor)

## Full write-up
The complete analysis, with charts and explanations, is in report/Crop_Yield_Capstone_Report.md.

## Author
Connect with me on [LinkedIn](https://linkedin.com/in/gideonsilas), [Twitter](https://twitter.com/Mr_Goodboi1), and [Medium](https://medium.com/@Just_Gideons).
