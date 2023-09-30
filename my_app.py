import h2o
import streamlit as st
from h2o.estimators.glm import H2OGeneralizedLinearEstimator

st.set_page_config(
    page_title="dk - h2o AutoMl",
    layout="wide",
    initial_sidebar_state="expanded",
)

h2o.init()

st.markdown('## `h2o` Classification')
with st.spinner('Preparing...'):
    prostate = h2o.import_file("https://h2o-public-test-data.s3.amazonaws.com/smalldata/prostate/prostate.csv")

    # convert columns to factors
    prostate['CAPSULE'] = prostate['CAPSULE'].asfactor()
    prostate['RACE'] = prostate['RACE'].asfactor()
    prostate['DCAPS'] = prostate['DCAPS'].asfactor()
    prostate['DPROS'] = prostate['DPROS'].asfactor()

    # set the predictor and response columns
    predictors = ["AGE", "RACE", "VOL", "GLEASON"]
    response_col = "CAPSULE"

    # split into train and testing sets
    train_cls, test_cls = prostate.split_frame(ratios=[0.8], seed=21)
    st.markdown('#### `train.head()`')
    st.write(train_cls.head())
    st.markdown('#### `test.head()`')
    st.write(test_cls.head())

    # set GLM modeling parameters
    # and initialize model training
    glm_model = H2OGeneralizedLinearEstimator(family="binomial",
                                              lambda_=0,
                                              compute_p_values=True)
    glm_model.train(predictors, response_col, training_frame=train_cls)

    # predict using the model and the testing dataset
    predict_cls = glm_model.predict(test_cls)

    # View a summary of the prediction
    st.markdown('#### Predictions')
    st.write(predict_cls)

st.markdown('## `h2o` Regression')
with st.spinner('Preparing...'):
    boston = h2o.import_file("https://s3.amazonaws.com/h2o-public-test-data/smalldata/gbm_test/BostonHousing.csv")

    # set the predictor columns
    predictors = boston.columns[:-1]

    # this example will predict the medv column
    # you can run the following to see that medv is indeed a numeric value
    boston["medv"].isnumeric()
    # set the response column to "medv", which is the median value of owner-occupied homes in $1000's
    response = "medv"

    # convert the `chas` column to a factor
    # `chas` = Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    boston['chas'] = boston['chas'].asfactor()

    # split into train and testing sets
    train_reg, test_reg = boston.split_frame(ratios=[0.8], seed=21)
    st.markdown('#### `train.head()`')
    st.write(train_reg.head())
    st.markdown('#### `test.head()`')
    st.write(test_reg.head())

    # set the `alpha` parameter to 0.25
    # then initialize the estimator then train the model
    boston_glm = H2OGeneralizedLinearEstimator(alpha=0.25)
    boston_glm.train(x=predictors,
                     y=response,
                     training_frame=train_reg)

    # predict using the model and the testing dataset
    predict_reg = boston_glm.predict(test_reg)

    # View a summary of the prediction
    st.markdown('#### Predictions')
    st.write(predict_reg)
