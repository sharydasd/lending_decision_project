import pandas as pd

def make_lending_decision(df_risk, total_credit):
    df = df_risk.copy()
    df['lend_flag'] = df['combined_risk'] < 0.4
    df['lend_weight'] = 1 - df['combined_risk']
    df['lend_amount'] = df['lend_weight'] / df['lend_weight'].sum() * total_credit
    rate_map = {'A':0.05,'B':0.08,'C':0.12}
    df['loan_rate'] = df['rating'].map(rate_map)
    return df[['企业代号','企业名称','industry','rating','pred_default',
              'default_prob','combined_risk','lend_flag','lend_amount','loan_rate']]