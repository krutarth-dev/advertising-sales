# advertising-sales

During the course of the project, we encountered several challenges and successfully addressed them. One of the issues we faced was the deprecation warning regarding the xgboost.rabit submodule. We resolved this by updating our code to use xgboost.collective instead, ensuring compatibility with future versions.

Another problem we encountered was the TypeError caused by the mean_squared_error function, which expected sequence or array-like inputs but received a Dask array instead. To resolve this, we converted the Dask array to a NumPy array using the compute() method before calculating the mean squared error.

After overcoming these challenges, we were able to evaluate the performance of our model. The mean squared error (MSE) obtained was 1.707, indicating that the model's predictions, on average, deviated from the actual values by a squared distance of approximately 1.707. Furthermore, the R2 score achieved was 0.930, indicating that the model explains approximately 93% of the variance in the target variable.

Based on these results, we can conclude that our current model is performing well, with a relatively low MSE and a high R2 score. However, depending on the project requirements and goals, there is a possibility of conducting further experiments to explore alternative models or techniques that might potentially improve the performance even further.
