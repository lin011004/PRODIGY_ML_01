# House Prices Prediction

This project involves predicting house prices using a linear regression model based on key features such as square footage, number of bedrooms, and number of bathrooms. The dataset used for this project is from the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle.

## Project Structure

The project consists of the following files:

- `train.csv`: The training dataset containing house features and their corresponding prices.
- `test.csv`: The testing dataset containing house features for which we need to predict the prices.
- `submission.csv`: The file where the predicted house prices are saved.
- `Task1.py`: The Python script containing the code for training the model and making predictions.
- `README.md`: This file.

## Requirements

- Python 3.6+
- pandas
- scikit-learn
- matplotlib

You can install the required libraries using pip:

```sh
pip install pandas scikit-learn matplotlib
```

## Code Overview

The script `house_prices_prediction.py` performs the following steps:

1. **Load the Training Data**: The training data is loaded from `train.csv`.

    ```python
    train_data = pd.read_csv("C:\\Users\\Lingesh\\OneDrive\\Desktop\\ML Intern\\house-prices-advanced-regression-techniques\\train.csv")
    ```

2. **Feature Selection**: Select features (`GrLivArea`, `BedroomAbvGr`, `FullBath`, `HalfBath`) and target (`SalePrice`) from the training dataset.

    ```python
    X_train = train_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
    y_train = train_data['SalePrice']
    ```

3. **Load the Testing Data**: The testing data is loaded from `test.csv`.

    ```python
    test_data = pd.read_csv("C:\\Users\\Lingesh\\OneDrive\\Desktop\\ML Intern\\house-prices-advanced-regression-techniques\\test.csv")
    ```

4. **Select Features for Testing**: Select the same features for the testing dataset.

    ```python
    X_test = test_data[['GrLivArea', 'BedroomAbvGr', 'FullBath', 'HalfBath']]
    ```

5. **Initialize and Train the Model**: A linear regression model is initialized and trained on the training data.

    ```python
    model = LinearRegression()
    model.fit(X_train, y_train)
    ```

6. **Make Predictions**: The trained model is used to make predictions on the testing dataset.

    ```python
    predicted_prices = model.predict(X_test)
    ```

7. **Create a Submission DataFrame**: A DataFrame is created to store the predicted prices along with the corresponding IDs.

    ```python
    submission_df = pd.DataFrame({
        'ID': test_data['Id'],
        'SalePrice': predicted_prices
    })
    ```

8. **Save Predictions to CSV**: The DataFrame is saved to `submission.csv`.

    ```python
    submission_df.to_csv('submission.csv', index=False)
    ```

9. **Plot Predicted Prices**: A line graph of the predicted prices is plotted.

    ```python
    plt.plot(submission_df['SalePrice'])
    plt.xlabel('House Index')
    plt.ylabel('Predicted Price')
    plt.title('Predicted Prices of Houses')
    plt.show()
    ```

## Running the Code

1. Ensure that `train.csv` and `test.csv` are placed in the appropriate directory as specified in the script.
2. Run the script `house_prices_prediction.py`.

    ```sh
    python house_prices_prediction.py
    ```

3. The predicted house prices will be saved in `submission.csv`, and a line graph of the predicted prices will be displayed.

## Acknowledgements

This project uses the dataset from the [House Prices: Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on Kaggle.

