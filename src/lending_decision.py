import pandas as pd

def make_lending_decision(df_risk,
                          total_credit):
    """
    根据combined_risk决定:
    - lend_flag: 是否放贷 (风险<0.4)
    - lend_amount: 按 (1-risk) 分配年度信贷总额
    - loan_rate: A/B/C 对应利率
    """
    df = df_risk.copy()
    df['lend_flag'] = df['combined_risk'] < 0.4
    df['lend_weight'] = 1 - df['combined_risk']
    df['lend_amount'] = (
        df['lend_weight'] / df['lend_weight'].sum() *
        total_credit
    )
    rate_map = {'A':0.05,'B':0.08,'C':0.12}
    df['loan_rate'] = df['rating'].map(rate_map)
    return df[['企业代号','rating','pred_default',
              'default_prob','combined_risk',
              'lend_flag','lend_amount','loan_rate']]