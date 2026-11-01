"""
Dataset Generation Script for Telecom Customer Churn Prediction.

Generates a realistic synthetic telecom customer churn dataset matching
the schema, distributions, and characteristics of the Kaggle Telco Churn benchmark (7,043 rows).
"""

import os
import random
import numpy as np
import pandas as pd


def generate_telecom_churn_data(
    num_records: int = 7043,
    random_seed: int = 42,
    output_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic telecom churn dataset with realistic relationships.

    Parameters
    ----------
    num_records : int, default=7043
        Number of customer records to generate.
    random_seed : int, default=42
        Seed for reproducibility.
    output_path : str, optional
        Path to save CSV file.

    Returns
    -------
    pd.DataFrame
        Synthetic telecom churn dataset.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    # 1. Unique Customer IDs (format: '1234-ABCD')
    customer_ids = []
    seen_ids = set()
    while len(customer_ids) < num_records:
        digits = f"{np.random.randint(1000, 9999)}"
        letters = "".join(random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=5))
        cid = f"{digits}-{letters}"
        if cid not in seen_ids:
            seen_ids.add(cid)
            customer_ids.append(cid)

    # 2. Demographics
    gender = np.random.choice(["Male", "Female"], size=num_records, p=[0.505, 0.495])
    senior_citizen = np.random.choice([0, 1], size=num_records, p=[0.838, 0.162])
    partner = np.random.choice(["Yes", "No"], size=num_records, p=[0.483, 0.517])

    # Dependents depend somewhat on partner
    dependents = []
    for p in partner:
        if p == "Yes":
            dependents.append(np.random.choice(["Yes", "No"], p=[0.51, 0.49]))
        else:
            dependents.append(np.random.choice(["Yes", "No"], p=[0.12, 0.88]))
    dependents = np.array(dependents)

    # 3. Tenure (1 to 72 months, with exactly 11 new customers having tenure = 0)
    tenure_mix = np.random.rand(num_records)
    tenure = np.ones(num_records, dtype=int)
    for i, r in enumerate(tenure_mix):
        if r < 0.30:
            tenure[i] = int(np.clip(1 + np.random.exponential(scale=6), 1, 12))
        elif r < 0.65:
            tenure[i] = int(np.random.randint(13, 60))
        else:
            tenure[i] = int(np.clip(np.random.normal(loc=65, scale=5), 60, 72))

    # Introduce exactly 11 records with tenure = 0 (benchmark Kaggle Telco Churn)
    zero_indices = np.random.choice(num_records, size=11, replace=False)
    tenure[zero_indices] = 0

    # 4. Phone Service & Multiple Lines
    phone_service = np.random.choice(["Yes", "No"], size=num_records, p=[0.903, 0.097])
    multiple_lines = []
    for ps in phone_service:
        if ps == "No":
            multiple_lines.append("No phone service")
        else:
            multiple_lines.append(np.random.choice(["No", "Yes"], p=[0.53, 0.47]))
    multiple_lines = np.array(multiple_lines)

    # 5. Internet Service
    internet_service = np.random.choice(
        ["DSL", "Fiber optic", "No"],
        size=num_records,
        p=[0.344, 0.440, 0.216]
    )

    # 6. Internet Add-on Services
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []

    for iserv in internet_service:
        if iserv == "No":
            online_security.append("No internet service")
            online_backup.append("No internet service")
            device_protection.append("No internet service")
            tech_support.append("No internet service")
            streaming_tv.append("No internet service")
            streaming_movies.append("No internet service")
        else:
            # DSL vs Fiber optic adoption variations
            sec_p = [0.65, 0.35] if iserv == "DSL" else [0.70, 0.30]
            online_security.append(np.random.choice(["No", "Yes"], p=sec_p))

            bk_p = [0.60, 0.40] if iserv == "DSL" else [0.55, 0.45]
            online_backup.append(np.random.choice(["No", "Yes"], p=bk_p))

            dp_p = [0.62, 0.38] if iserv == "DSL" else [0.55, 0.45]
            device_protection.append(np.random.choice(["No", "Yes"], p=dp_p))

            ts_p = [0.64, 0.36] if iserv == "DSL" else [0.71, 0.29]
            tech_support.append(np.random.choice(["No", "Yes"], p=ts_p))

            tv_p = [0.65, 0.35] if iserv == "DSL" else [0.45, 0.55]
            streaming_tv.append(np.random.choice(["No", "Yes"], p=tv_p))

            mov_p = [0.64, 0.36] if iserv == "DSL" else [0.44, 0.56]
            streaming_movies.append(np.random.choice(["No", "Yes"], p=mov_p))

    online_security = np.array(online_security)
    online_backup = np.array(online_backup)
    device_protection = np.array(device_protection)
    tech_support = np.array(tech_support)
    streaming_tv = np.array(streaming_tv)
    streaming_movies = np.array(streaming_movies)

    # 7. Contract (Tenure influences contract)
    contract = []
    for t in tenure:
        if t <= 12:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.85, 0.11, 0.04]))
        elif t <= 36:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.55, 0.30, 0.15]))
        else:
            contract.append(np.random.choice(["Month-to-month", "One year", "Two year"], p=[0.20, 0.35, 0.45]))
    contract = np.array(contract)

    # 8. Paperless Billing & Payment Method
    paperless_billing = np.random.choice(["Yes", "No"], size=num_records, p=[0.592, 0.408])

    payment_method = []
    for sc, iserv in zip(senior_citizen, internet_service):
        if iserv == "Fiber optic":
            pm = np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                p=[0.48, 0.10, 0.21, 0.21]
            )
        else:
            pm = np.random.choice(
                ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
                p=[0.25, 0.28, 0.24, 0.23]
            )
        payment_method.append(pm)
    payment_method = np.array(payment_method)

    # 9. Monthly Charges (Realistic calculation based on services subscribed)
    monthly_charges = np.zeros(num_records)
    for i in range(num_records):
        base = 0.0
        if phone_service[i] == "Yes":
            base += 20.0
            if multiple_lines[i] == "Yes":
                base += 5.0
        if internet_service[i] == "DSL":
            base += 25.0
        elif internet_service[i] == "Fiber optic":
            base += 45.0

        if online_security[i] == "Yes":
            base += 5.0
        if online_backup[i] == "Yes":
            base += 5.0
        if device_protection[i] == "Yes":
            base += 5.0
        if tech_support[i] == "Yes":
            base += 5.0
        if streaming_tv[i] == "Yes":
            base += 10.0
        if streaming_movies[i] == "Yes":
            base += 10.0

        # Add modest random variation for realistic decimal billing
        noise = np.random.uniform(-1.75, 2.25)
        monthly_charges[i] = round(max(18.25, min(118.75, base + noise)), 2)

    # 10. Total Charges
    # For tenure == 0, TotalCharges is blank string " "
    total_charges = []
    for i in range(num_records):
        if tenure[i] == 0:
            total_charges.append(" ")
        else:
            # MonthlyCharges * tenure + small fluctuation for usage/taxes
            tot = monthly_charges[i] * tenure[i] + np.random.uniform(-5.0, 5.0)
            tot = max(monthly_charges[i], round(tot, 2))
            total_charges.append(str(tot))
    total_charges = np.array(total_charges)

    # 11. Churn Generation (Realistic logistic probability modeling with market noise)
    # Target: ~26.5% overall churn rate with realistic ~0.85 ROC-AUC benchmark
    log_odds = np.zeros(num_records)

    # Base intercept
    log_odds -= 1.05

    # Contract impact (strongest retention anchor)
    log_odds += np.where(contract == "Month-to-month", 1.35, 0.0)
    log_odds -= np.where(contract == "One year", 0.65, 0.0)
    log_odds -= np.where(contract == "Two year", 1.60, 0.0)

    # Tenure impact (exponential decay of churn probability over customer lifetime)
    log_odds -= 0.038 * tenure

    # Monthly Charges impact (price sensitivity)
    log_odds += 0.018 * (monthly_charges - 64.76)

    # Internet service impact (Fiber has higher churn, No internet has lower)
    log_odds += np.where(internet_service == "Fiber optic", 0.55, 0.0)
    log_odds -= np.where(internet_service == "No", 0.65, 0.0)

    # Tech Support / Online Security retention effect
    log_odds -= np.where(tech_support == "Yes", 0.38, 0.0)
    log_odds -= np.where(online_security == "Yes", 0.32, 0.0)

    # Payment Method impact (Electronic check has significantly higher churn)
    log_odds += np.where(payment_method == "Electronic check", 0.48, 0.0)

    # Senior Citizen slight risk increase
    log_odds += np.where(senior_citizen == 1, 0.22, 0.0)

    # Paperless billing slight risk increase
    log_odds += np.where(paperless_billing == "Yes", 0.18, 0.0)

    # Partner / Dependents loyalty effect
    log_odds -= np.where(partner == "Yes", 0.14, 0.0)
    log_odds -= np.where(dependents == "Yes", 0.18, 0.0)

    # Add realistic behavioral and unobserved market noise (logistic distribution)
    # This prevents artificial 100% linear separability and matches real-world ~0.85 AUC
    market_noise = np.random.logistic(loc=0.0, scale=0.62, size=num_records)
    log_odds += market_noise

    # Compute probability via sigmoid
    probs = 1.0 / (1.0 + np.exp(-log_odds))

    # Add controlled threshold calibration to get exactly ~26.5% churn
    target_count = int(round(num_records * 0.2654))  # 1869 out of 7043
    threshold = np.sort(probs)[num_records - target_count]
    churn = np.where(probs >= threshold, "Yes", "No")

    # Assemble DataFrame
    df = pd.DataFrame({
        "customerID": customer_ids,
        "gender": gender,
        "SeniorCitizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
        "Churn": churn
    })

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f" [Dataset Generator] Successfully generated {len(df)} records at: {output_path}")
        print(f" [Dataset Generator] Churn Rate: {(df['Churn'] == 'Yes').mean():.2%}")

    return df


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "data", "telecom_churn.csv")
    generate_telecom_churn_data(output_path=out)
