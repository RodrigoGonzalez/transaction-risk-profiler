## v0.4.0 (2023-09-27)

### Feat

- update features and data transformer files (#27)

### Fix

- **aggregations**: update functions to fix column errors, and add docstrings (#25)

## v0.3.0 (2023-08-31)

### Feat

- **aggregations**: added new functions to aggregate list variables (#22)
- **transformer**: add transformers module to support custom transformations that can be used in scikit learn (#20)
- **notebooks**: add histogram charts to eda notebook (#19)

## v0.2.0 (2023-08-28)

### Feat

- **data-transporter**: update data transporter to incorporate datetime features, splitting dataset, and dropping predefined columns (#9)
- **io**: add dataset loader and corresponding helper functions and enums (#8)
- **feature-engineering**: add functions for categorical and simple transforms (#7)
- **preprocessing**: added text and knn files and refactored code to increase readability (#6)
- **main-application**: added preprocessing, configs, eda, feature engineering and utils directories (#5)
- **main-app**: add folder structure (#4)
- **utils**: add utility directory for utility files (#3)
- **project**: update project completely add poetry, linters, move python files to package, add versioning (#1)

### Refactor

- **preprocessing**: update text transformations and processing, adding docstrings and making the code more efficient (#17)
- **preprocessing**: refactor tfidf file to contain better docstrings and improve code (#16)
- **simple-transforms**: update to include addition functions that apply simple transformations (#15)
- **eda**: update charting functions to accept labels as parameters (#14)
- **main-application**: refactored charts to accept any column value names, add EDA notebook, update dataset transfer class (#13)
- **modeling**: update functions with type hints, better docstrings, and improve code overall (#12)
- **models**: refactor baseline file adding type hints and documentation and improving code (#11)
- **models**: update model files with efficiency improvements, type hints, and docstrings (#10)
