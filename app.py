from flask import Flask ,request,jsonify
import pandas as pd 
import joblib
from xgboost import XGBClassifier

app=Flask(__name__)

# models
logistic_model=joblib.load('logistic_model.pkl')
rf_model=joblib.load('rf_model.pkl')
xgb_model = XGBClassifier()
xgb_model.load_model("xgb_model.json")

# preprocessing tools
scaler=joblib.load('scaler.pkl')
encoders=joblib.load('encoders.pkl')

feature_order = ['Age', 'Gender', 'Hormonal Changes', 'Family History', 'Race/Ethnicity',
                 'Body Weight', 'Calcium Intake', 'Vitamin D Intake', 'Physical Activity',
                 'Smoking', 'Alcohol Consumption', 'Medical Conditions', 'Medications',
                 'Prior Fractures']

@app.route('/predict',methods=['POST'])
def predict():
    data=request.json 

    input_df=pd.DataFrame([data],columns=feature_order)

    for c in input_df.columns:
        if c in encoders:
            le=encoders[c]
            input_df[c]=le.transform([input_df[c]])

    input_scaled=scaler.transform(input_df)

    re_pred=rf_model.predict(input_df)[0]
    xgb_pred=xgb_model.predict(input_df)[0]
    lg_pred=logistic_model.predict(input_scaled)[0]

    final_pred=int((re_pred+xgb_pred+lg_pred)>=2)

    user_influence=input_scaled[0]*logistic_model.coef_[0]
    influence_series=pd.Series(user_influence,index=feature_order)

    top_risk=influence_series.sort_values(ascending=False).head(3)
    top_protective=influence_series.sort_values().head(3)

    return jsonify({
        'prediction':int(final_pred),
        'top_risk_factors':top_risk.to_dict(),
        'top_protective_factors':top_protective.to_dict()
    })

if __name__=='__main__':
    app.run(debug=True)