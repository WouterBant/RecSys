## Notebooks

- [beyond_accuracy_plots.ipynb](beyond_accuracy_plots.ipynb): visualizing the beyond accuracy results in histograms.

- [heuristic_methods.ipynb](heuristic_methods.ipynb): computes the metrics for the baselines presented in the paper. This includes the random method, predicting the most frequent category in the click history of the user, and predicting the closest publish time compared to the last clicked article of the user.

- [model_inspection.ipynb](model_inspection.ipynb): visualizes the attention at various layers of the model for an example input.

- [n_gram.ipynb](n_gram.ipynb): investigation of how far you can get with learning n-grams or using specific features. Conclusion was not so far in general, but predicting always the article with the closest publish time to the latest clicked article leads to 65% accuracy when presented with a negative and positive example.

- [num_params.ipynb](num_params.ipynb): shows the number of parameters for each model.

- [pearson_correlation.ipynb](pearson_correlation.ipynb): calculates the Pearson correlation between previously clicked articles of the user and the model predictions.

- [preprocess_data.ipynb](preprocess_data.ipynb): shows how convert the provided parquet files to a huggingface dataset.

- [understand_model_predictions.ipynb](understand_model_predictions.ipynb): is used to beter understand the predictions and visualizes the outputs in different scenarios.