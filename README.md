# Diabetes Prediction using Machine Learning

This repository contains the code and resources for predicting diabetes using machine learning algorithms. It includes exploratory data analysis (EDA) notebooks, trained models, and Flask code for integrating the model into a web application.

## Repository Structure

The repository consists of the following files:

1. `symptoms_data_exploration.ipynb`: This Jupyter notebook performs exploratory data analysis on the symptoms dataset to analyze important features for training the machine learning model. It also includes code for training models using different algorithms and performing cross-validation to select the best model.

2. `user_demographics_exploration.ipynb`: This notebook focuses on the exploratory data analysis of the user demographics dataset. It explores the dataset, identifies significant features for model training, and compares performance across various machine learning algorithms using cross-validation techniques.

3. `final_diabetes_prediction.ipynb`: This notebook contains the final machine learning code for diabetes prediction. It combines the relevant features from the symptoms and user demographics datasets, performs feature engineering if required, and trains the final machine learning model.

4. Pickle Files:
   - `model_symptom.pkl`: Pickle file containing the trained machine learning model based on the symptoms dataset.
   - `model_demo.pkl`: Pickle file containing the trained machine learning model based on the user demographics dataset.
   - `scalar_symptom.pkl`: Pickle file containing the scaled symptoms dataset used for prediction in the Flask application.
   - `scalar_demo.pkl`: Pickle file containing the scaled user demographics dataset used for prediction in the Flask application.

5. `Diabetes_Flask.py`: Flask application code that allows users to input their symptoms and demographics through a web interface. It uses the pickled models and scaled datasets to make predictions and displays the results on the website.

## Usage

To use the diabetes prediction model and web application:

1. Clone the repository to your local machine using `git clone [<repository-url>](https://github.com/Aishujain/Diabetes_prediction/tree/main)`.

2. Open the notebooks `symptoms_data_exploration.ipynb` and `user_demographics_exploration.ipynb` to explore the data and model development process.

3. Execute the `final_diabetes_prediction.ipynb` notebook to train the final machine learning model and evaluate its performance.

4. Run the Flask application by executing `Diabetes_Flask.py` using Python. Ensure that the necessary dependencies are installed by running `pip install -r requirements.txt`.

5. Access the web interface in your browser and provide the required information for diabetes prediction.

## Dependencies

To run the code in this repository requires minimum version of the following libraries and frameworks:

- Python (version 3.5)
- NumPy (version 1.17.3)
- Pandas (version 1.0.5)
- Scikit-learn (version 0.22)
- Flask (version 3.8.2)

Make sure to install these dependencies using `pip` or any preferred package manager.

## Contributing

Contributions to this project are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments
Data set obtained from
- https://data.world/turba/early-stage-diabetes-risk-prediction-using-machine-learning
- https://data.world/turba/early-stage-diabetes-risk-prediction-using-machine-learning

This code has been used for Monash university Final year personal project for creation of website Type2Thrive.link to help type2-diabetic patients
