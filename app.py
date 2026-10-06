#Daraz style daily order predeiction
# 1. load dataset
# 2. choose x and y
# 3. split into train test split
# 4.train linear regression
# 5.evaluate it with MAE and R2
# 6. predict orders for a new business scenario
# 7.build an interactive streamlit interface


##import libraries

import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

 #Page settting
st.set_page_config(
    page_title="Daraz Style Daily Order Prediction",
        page_icon="📦",
        layout="wide"
 )

df = pd.read_csv("daraz_daily_orders_synthetic(1).csv")

#4.choose features x andtarget y
feature_columns = [
"Website_Visitors",
"Ad_Spend_PKR",
"Discount_Percent",
"Weekend"
]

x = df[feature_columns]
y = df["Daily_Orders"]

#5 Train test split

X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=42) #train size is 80% then


#6. Train linear regression model
model= LinearRegression()

model.fit(X_train,y_train)


# 7.Test the model
test_predictions = model.predict(X_test)
mae = mean_absolute_error(y_test,test_predictions)
r2 = r2_score(y_test,test_predictions)


# 8. Predict tables used by app
test_result = X_test.copy()
test_result["Actual_Orders"] = y_test
test_result["Predicted_Orders"] = test_predictions.round(1)

test_result["Residual"] = (test_result["Actual_Orders"] - test_result["Predicted_Orders"]
).round(1)

coefficient = pd.DataFrame(
    {
        "Feature" : feature_columns,
        "Coefficient" : model.coef_.round(4)
    }
)

#Build the streamlit interface
st.title("📦Daraz Style Daily Order Prediction")
st.write("""
This app predicts the daily orders for a Daraz style business based on various features such as website visitors, ad spend, discount percentage, and whether it's a weekend or not. The model is trained)
""")

metric_1,metric_2,metric_3,metric_4 = st.columns(4)

metric_1.metric("Data Rows",len(df))
metric_2.metric("Training Rows",len(X_train))
metric_3.metric("Test MAE",f"{mae:.2f} orders")
metric_4.metric("Test R2",f"{r2:.2f}")

st.caption("MAE is the average difference between the actual and predicted orders. R2 indicates how well the model explains the variance in the data.")

# create the main tabs

predict_tab,insights_tab,data_tab = st.tabs(
    [
        "📦 Predict Orders",
        "📊 Insights",
        "📁 Explore Data"
    ]
)

with predict_tab:
    st.subheader("Predict Daily Orders for a New Business Scenario")
    st.write(
        """
        Use the sliders below to input the features for a new business scenario and predict the daily orders.
        """
    )

    # Create sliders for user input
    with st.form("Prediction Form"):
        left_column,right_column = st.columns(2)
        with left_column:
            website_visitors = st.slider(
                "Website Visitors",
                min_value = int(df["Website_Visitors"].min()),
                max_value = int(df["Website_Visitors"].max()),
                value = 3500,
                step = 50,
                help="How many ppl visited today?"
            )

            ad_spend = st.slider(
                "Ad Spend (PKR)",
                min_value = int(df["Ad_Spend_PKR"].min()),
                max_value = int(df["Ad_Spend_PKR"].max()),
                value = 15000,
                step = 500,
                help="How much did you spend on ads today?"
            )


        with right_column:
            discount_percent = st.slider(
            "Discount (%)",
            min_value = int(df["Discount_Percent"].min()),
            max_value = int(df["Discount_Percent"].max()),
            value = 15,
            step = 1,
            help = "What discount percentage is being offered? "
            )

            day_type = st.radio(
                "Day Type",
                options = ["Weekday", "Weekend"],
                horizontal=True
            )

        predict_button = st.form_submit_button(
            "Predict Daily Orders",
            type = "primary",
            use_container_width=True
        )

        if predict_button:
            weekend = 1 if day_type == "weekend" else 0

            new_day = pd.DataFrame(
                {
                    "Website_Visitors": [website_visitors],
                    "Ad_Spend_PKR":[ad_spend],
                    "Discount_Percent":[discount_percent],
                    "Weekend":[weekend]

                }
            )


            #predict() an array model can predict rows
            predicted_orders = model.predict(new_day)[0]
            st.success("Prediction completed successfully!")
            st.metric(
                "Estimated Daily Orders",
                f"{round(predicted_orders)} : orders"
            )

            st.caption(
                f"Raw model estimate: {predicted_orders: .2f} orders"
            )

            with st.expander("See the exact inputs sent to the model by you"):
             st.dataframe(
                new_day,
                use_container_width=True,
                hide_index = True
            )


            st.warning(
                "This is a model estimation based on classroom data and does not guarantee future sales number. Change the data with the real world data to get the real results"

            )
with insights_tab:

    st.subheader("How well did the model perform")
    score_1,score_2 = st.columns(2)
    score_1.metric("Mean Absolute Error", f"{mae:.2f} orders")
    score_2.metric("R2 Score", f"{r2:.3f}")

    st.write(
        "MAE Prediction are about "
        f"{mae: .2f} order are awaya from the actual orders on an avergae"

    )


    st.write(
        "R2 is about "f"{r2*200:.1f}% of the variation in test orders is explained"
    )

    st.divider()

#Intercept and coefficient
    st.subheader("What did linear regression learn?")
    st.metric("Intercept", f"{model.intercept_:.2f}")
    st.dataframe(
        coefficient,
        use_container_width=True,
        hide_index=True
    )

    st.bar_chart(
        coefficient.set_index("Feature")
    )

    st.caption(
    "Coefficients show how each input affects the model's prediction. "
    "For example, a positive ad spend coefficient means the model predicts "
    "more sales when ad spend increases."
    "features are held constant and do not automatically prove causation"

    )


    st.divider()

    st.subheader("Actual vs Predicted Orders")

    st.scatter_chart(
        test_result,x="Actual_Orders",y="Predicted_Orders"
    )

    with st.expander("Open to see the test-set prediction table"):
        st.dataframe(
            test_result,
            use_container_width=True,

        )
with data_tab:
    st.subheader("Explore the Dataset")
    st.write(
        "Each row represents one day of data, and Daily_Orders is the target—the value our model learns to predict"
    )

    selected_feature = st.selectbox(
        "Choose a feature for comparison with daily orders",
        options = feature_columns
    )

    st.scatter_chart(
        df,
        x= selected_feature,
        y="Daily_Orders"
    )

    show_full_data = st.checkbox("Show complete dataset")

    if show_full_data:
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )


    with st.expander("View Summary Statistics"):
        st.dataframe(
            df.describe(),
            use_container_width=True
        )

    
